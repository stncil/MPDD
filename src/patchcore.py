"""
patchcore_baseline.py
======================
A minimal, dependency‑light implementation of PatchCore for MPDD.

This script builds a memory bank of normal patch features ("training"), then
runs inference to produce image‑ and pixel‑level anomaly scores and masks.

Usage
-----
python src/patchcore.py \
    --data_root /mnt/c/Users/akhil/All_my_codes/Portfolio/MPDD/anomaly_dataset_og \
    --classes metal_plate bracket_black \
    --tile_size 512 --stride 256 \
    --backbone resnet50 \
    --layers layer2 layer3 \
    --coreset_size 0.05 \
    --save_dir runs/patchcore_r50

Dependencies: torch, torchvision, numpy, faiss-cpu (optional but faster),
opencv-python, albumentations.  Everything else is in the standard library.

This is intentionally self‑contained: no Lightning, no external patchcore repo.

"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple, Dict

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms as tvT

try:
    import faiss  # type: ignore
    HAS_FAISS = True
except Exception:
    HAS_FAISS = False

# -----------------------------------------------------------------------------
# Helpers for tiling & merging
# -----------------------------------------------------------------------------

def tile_image(img: np.ndarray, tile: int, stride: int) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
    """Return list of tiles and their (y,x) top-left coords.
    img: HxWxC uint8
    """
    H, W = img.shape[:2]
    tiles, coords = [], []
    for y in range(0, max(H - tile, 0) + 1, stride):
        for x in range(0, max(W - tile, 0) + 1, stride):
            y1, x1 = y + tile, x + tile
            patch = img[y:y1, x:x1]
            if patch.shape[0] != tile or patch.shape[1] != tile:
                # pad to tile size
                pad_h = tile - patch.shape[0]
                pad_w = tile - patch.shape[1]
                patch = cv2.copyMakeBorder(patch, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
            tiles.append(patch)
            coords.append((y, x))
    return tiles, coords

def merge_score_maps(score_maps: List[np.ndarray], coords: List[Tuple[int, int]], full_h: int, full_w: int) -> np.ndarray:
    """Merge overlapping tile score maps by averaging overlaps.
    score_maps: list of Ht x Wt float
    coords:     list of (y,x)
    """
    full = np.zeros((full_h, full_w), dtype=np.float32)
    weight = np.zeros((full_h, full_w), dtype=np.float32)
    tile_h, tile_w = score_maps[0].shape
    for s, (y, x) in zip(score_maps, coords):
        h = min(tile_h, full_h - y)
        w = min(tile_w, full_w - x)
        full[y:y+h, x:x+w] += s[:h, :w]
        weight[y:y+h, x:x+w] += 1.0
    full /= np.maximum(weight, 1e-6)
    return full

# -----------------------------------------------------------------------------
# Feature extractor wrapper
# -----------------------------------------------------------------------------
class BackboneWrapper(nn.Module):
    def __init__(self, name: str = "resnet50", layers: List[str] = ["layer2", "layer3"], pretrained: bool = True):
        super().__init__()
        if name == "resnet50":
            net = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
            self.out_dims = {"layer2": 512, "layer3": 1024}
        elif name == "resnet18":
            net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
            self.out_dims = {"layer2": 128, "layer3": 256}
        else:
            raise ValueError(f"Unsupported backbone: {name}")
        self.layers = layers
        # register forward hooks
        self._features: Dict[str, torch.Tensor] = {}
        def get_hook(name):
            def hook(module, inp, out):
                self._features[name] = out
            return hook
        for l in layers:
            getattr(net, l).register_forward_hook(get_hook(l))
        # Remove FC head
        self.body = nn.Sequential(*list(net.children())[:-2])
        # Normalisation transform
        self.normalize = tvT.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        self._features.clear()
        x = self.normalize(x)
        _ = self.body(x)
        feats = [self._features[l] for l in self.layers]
        return feats

# -----------------------------------------------------------------------------
# Core-set selection (greedy farthest point / k-center)
# -----------------------------------------------------------------------------

def kcenter_greedy(X: np.ndarray, budget: int) -> np.ndarray:
    """Select core-set indices from X (Nxd) using greedy k-center.
    Returns indices of selected points. X assumed L2-normalised or not—distance is Euclidean.
    """
    n = X.shape[0]
    first = np.random.randint(0, n)
    selected = [first]
    # Squared distances to selected set
    dist = np.full(n, np.inf, dtype=np.float32)
    dist = np.minimum(dist, np.sum((X - X[first]) ** 2, axis=1))
    for _ in range(1, budget):
        idx = int(np.argmax(dist))
        selected.append(idx)
        dist = np.minimum(dist, np.sum((X - X[idx]) ** 2, axis=1))
    return np.array(selected)

# -----------------------------------------------------------------------------
# Distance search (kNN via FAISS or Torch)
# -----------------------------------------------------------------------------

def knn_search(query: np.ndarray, base: np.ndarray, k: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    if HAS_FAISS:
        d = base.shape[1]
        index = faiss.IndexFlatL2(d)
        index.add(base.astype(np.float32))
        D, I = index.search(query.astype(np.float32), k)
        return D, I
    # Torch fallback
    q = torch.from_numpy(query)
    b = torch.from_numpy(base)
    # (Q,B) distances
    # using (a-b)^2 = a^2 + b^2 -2ab
    q2 = (q**2).sum(1, keepdim=True)
    b2 = (b**2).sum(1)
    D = q2 + b2 - 2.0 * (q @ b.t())
    D, I = torch.topk(D, k, dim=1, largest=False)
    return D.numpy(), I.numpy()

# -----------------------------------------------------------------------------
# Dataset utilities
# -----------------------------------------------------------------------------

def list_images(root: Path, split: str, classes: List[str]) -> List[Tuple[str, str]]:
    """Return list of (img_path, mask_path_or_none). Assumes MPDD-like layout.
    split: 'train' or 'test'
    For train: only 'good' images, masks are None.
    For test: 'good' and defect types; masks exist only for defects.
    """
    items = []
    for cls in classes:
        cls_root = root / cls / split
        if split == 'train':
            good_dir = cls_root / 'good'
            for img_name in sorted(os.listdir(good_dir)):
                items.append((str(good_dir / img_name), None))
        else:
            for defect_type in os.listdir(cls_root):
                img_dir = cls_root / defect_type
                for img_name in sorted(os.listdir(img_dir)):
                    img_path = img_dir / img_name
                    mask_path = None
                    if defect_type != 'good':
                        mask_dir = root / 'ground_truth' / cls / defect_type
                        mask_path = str(mask_dir / img_name.replace('.png', '_mask.png'))
                    items.append((str(img_path), mask_path))
    return items

# -----------------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------------

def extract_features(backbone: BackboneWrapper, imgs: List[np.ndarray], device: torch.device) -> List[torch.Tensor]:
    """Forward a list of HxWxC uint8 images; returns list of feature maps per layer."""
    backbone.eval()
    feats_layers: List[List[torch.Tensor]] = [[] for _ in backbone.layers]
    to_tensor = tvT.ToTensor()
    with torch.no_grad():
        for img in imgs:
            t = to_tensor(img)  # CxHxW float0-1
            t = t.unsqueeze(0).to(device)
            feats = backbone(t)
            for i, f in enumerate(feats):
                feats_layers[i].append(f.cpu())
    # Stack along batch dimension for each layer
    feats_layers = [torch.cat(flist, dim=0) for flist in feats_layers]
    return feats_layers

def features_to_patches(feat: torch.Tensor) -> torch.Tensor:
    """Convert feature map BxCxHxW to (B*H*W) x C patches."""
    B, C, H, W = feat.shape
    feat = feat.permute(0, 2, 3, 1).reshape(-1, C)
    return feat


def train_memory_bank(args, device):
    # 1. list train images
    items = list_images(Path(args.data_root), 'train', args.classes)
    # 2. load + tile images
    tiles_all = []
    for img_path, _ in items:
        img = np.ascontiguousarray(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB))
        tiles, _ = tile_image(img, args.tile_size, args.stride)
        tiles_all.extend(tiles)
    print(f"Train: {len(tiles_all)} tiles")

    # 3. feature extraction
    backbone = BackboneWrapper(args.backbone, args.layers).to(device)
    feats_layers = extract_features(backbone, tiles_all, device)

    # 4. concat layers (channel-wise)
    patches_list = []
    for feat in feats_layers:
        patches_list.append(features_to_patches(feat))
    patches = torch.cat(patches_list, dim=1).cpu().numpy()
    print("Raw patches:", patches.shape)

    # 5. random projection / whitening optional (skip for simplicity)

    # 6. coreset selection
    budget = max(1, int(patches.shape[0] * args.coreset_size))
    idxs = kcenter_greedy(patches, budget)
    memory_bank = patches[idxs]
    print("Memory bank:", memory_bank.shape)

    # 7. save
    os.makedirs(args.save_dir, exist_ok=True)
    np.save(os.path.join(args.save_dir, 'memory_bank.npy'), memory_bank)
    conf = {
        'backbone': args.backbone,
        'layers': args.layers,
        'tile_size': args.tile_size,
        'stride': args.stride,
        'coreset_size': args.coreset_size,
        'classes': args.classes,
    }
    with open(os.path.join(args.save_dir, 'config.json'), 'w') as f:
        json.dump(conf, f, indent=2)
    torch.save(backbone.state_dict(), os.path.join(args.save_dir, 'backbone.pt'))
    print("Saved memory_bank and config to", args.save_dir)


def inference(args, device):
    # load config & memory bank
    with open(os.path.join(args.save_dir, 'config.json'), 'r') as f:
        conf = json.load(f)
    memory_bank = np.load(os.path.join(args.save_dir, 'memory_bank.npy'))
    backbone = BackboneWrapper(conf['backbone'], conf['layers']).to(device)
    backbone.load_state_dict(torch.load(os.path.join(args.save_dir, 'backbone.pt'), map_location=device))
    backbone.eval()

    # list test images
    items = list_images(Path(args.data_root), 'test', conf['classes'])

    image_scores = []
    pixel_maps = []
    img_paths = []

    to_tensor = tvT.ToTensor()
    with torch.no_grad():
        for img_path, _ in items:
            img = np.ascontiguousarray(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB))
            H, W = img.shape[:2]
            tiles, coords = tile_image(img, conf['tile_size'], conf['stride'])
            tile_scores_maps = []
            max_tile_score = 0.0
            for tile in tiles:
                tile = np.ascontiguousarray(tile)
                t = to_tensor(tile).unsqueeze(0).to(device)
                feats = backbone(t)
                patches_list = [features_to_patches(f).cpu().numpy() for f in feats]
                patches_tile = np.concatenate(patches_list, axis=1)
                # kNN distance to memory bank
                D, _ = knn_search(patches_tile, memory_bank, k=1)
                # reshape back to HxW
                # infer feat map size from first layer
                f0 = feats[0]
                fh, fw = f0.shape[2], f0.shape[3]
                score_map = D.reshape(fh, fw)
                # upsample to tile size
                score_map = cv2.resize(score_map, (conf['tile_size'], conf['tile_size']), interpolation=cv2.INTER_LINEAR)
                tile_scores_maps.append(score_map)
                max_tile_score = max(max_tile_score, float(score_map.max()))
            # merge
            full_score = merge_score_maps(tile_scores_maps, coords, H, W)
            # optional Gaussian blur to smooth seams
            full_score = cv2.GaussianBlur(full_score, (0, 0), sigmaX=4, sigmaY=4)
            image_scores.append(max_tile_score)
            pixel_maps.append(full_score)
            img_paths.append(img_path)

    # normalise pixel maps (min-max over normal validation would be better; here use per‑image)
    maps_norm = []
    for m in pixel_maps:
        mn, mx = np.min(m), np.max(m)
        maps_norm.append((m - mn) / (mx - mn + 1e-6))

    # save outputs
    out_dir = Path(args.save_dir) / 'inference'
    out_dir.mkdir(parents=True, exist_ok=True)
    for path, score, m in zip(img_paths, image_scores, maps_norm):
        base = Path(path).stem
        np.save(out_dir / f"{base}_score.npy", m.astype(np.float32))
        # pseudo mask using 97.5 percentile
        thr = np.percentile(m, 97.5)
        mask = (m >= thr).astype(np.uint8) * 255
        cv2.imwrite(str(out_dir / f"{base}_mask.png"), mask)
    # save json with image-level scores
    with open(out_dir / 'image_scores.json', 'w') as f:
        json.dump({p: float(s) for p, s in zip(img_paths, image_scores)}, f, indent=2)
    print("Inference results saved to", out_dir)

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_root', type=str, required=True)
    p.add_argument('--classes', nargs='+', default=['metal_plate'])
    p.add_argument('--tile_size', type=int, default=512)
    p.add_argument('--stride', type=int, default=256)
    p.add_argument('--backbone', type=str, default='resnet50')
    p.add_argument('--layers', nargs='+', default=['layer2', 'layer3'])
    p.add_argument('--coreset_size', type=float, default=0.05)
    p.add_argument('--save_dir', type=str, default='runs/patchcore')
    p.add_argument('--mode', choices=['train', 'test', 'both'], default='both')
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.mode in ['train', 'both']:
        train_memory_bank(args, device)
    if args.mode in ['test', 'both']:
        inference(args, device)


if __name__ == '__main__':
    main()
