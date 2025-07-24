"""Reusable preprocessing and augmentation transforms for MPDD-style metal-defect
inspection.  Designed to work with both images and segmentation masks.

Key features
------------
* Longest-side resize to 1024 px (keeps aspect ratio).
* CLAHE on the Lab-L channel to boost local contrast without over-saturation.
* Optional background crop for the bracket classes (edge-based).
* Geometric + photometric augmentations (rot/flip, elastic, HSV jitter).
* CutBlur augmentation (self-supervised AD friendly).
* Sliding-window tiling utility (512² tiles, 50 % overlap).

All transforms are built with **Albumentations** so they operate jointly on
`image` and `mask`.  Use `get_train_transforms()` / `get_test_transforms()` to
obtain ready-made pipelines.
"""
from __future__ import annotations

import cv2
import math
import numpy as np
import torch
from typing import List, Tuple, Iterator

import albumentations as A
from albumentations.pytorch import ToTensorV2


# -----------------------------------------------------------------------------
# --- Custom Albumentations transforms ----------------------------------------
# -----------------------------------------------------------------------------

class CLAHELab(A.ImageOnlyTransform):
    """Apply CLAHE to the L channel of Lab colour‑space.

    Parameters
    ----------
    clip_limit : float
        Threshold for contrast limiting.
    tile_grid_size : Tuple[int, int]
        Size of grid for histogram equalisation.
    always_apply : bool
        Forwarded to base class.
    p : float
        Probability of applying the transform.
    """

    def __init__(self, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8), *, always_apply: bool = False, p: float = 1.0):
        super().__init__(always_apply, p)
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self._clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    def apply(self, img: np.ndarray, **params):  # type: ignore[override]
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        l_eq = self._clahe.apply(l)
        lab_eq = cv2.merge((l_eq, a, b))
        return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)


class CutBlur(A.ImageOnlyTransform):
    """CutBlur augmentation from *CutMix Data Augmentation for Image Classification*.

    For industrial AD we paste *blurred* patches back onto the clear image (or
    vice‑versa) so that the model learns high‑frequency consistency.
    """

    def __init__(self, alpha: Tuple[int, int] = (32, 128), *, down_ratio: int = 4, p: float = 0.5):
        super().__init__(p=p)
        self.alpha = alpha
        self.down_ratio = down_ratio

    def apply(self, img: np.ndarray, **params):  # type: ignore[override]
        h, w, _ = img.shape
        cut = np.random.randint(self.alpha[0], self.alpha[1])
        cx = np.random.randint(0, w)
        cy = np.random.randint(0, h)
        x1 = np.clip(cx - cut // 2, 0, w)
        y1 = np.clip(cy - cut // 2, 0, h)
        x2 = np.clip(cx + cut // 2, 0, w)
        y2 = np.clip(cy + cut // 2, 0, h)

        # Prepare blurred version
        k = max(3, (cut // self.down_ratio) | 1)  # kernel must be odd
        blurred = cv2.GaussianBlur(img, (k, k), 0)

        if np.random.rand() < 0.5:
            img[y1:y2, x1:x2] = blurred[y1:y2, x1:x2]
        else:
            temp = img.copy()
            temp[y1:y2, x1:x2] = blurred[y1:y2, x1:x2]
            img = temp
        return img


# -----------------------------------------------------------------------------
# --- Convenience builders -----------------------------------------------------
# -----------------------------------------------------------------------------

MAX_SIDE = 1024  # pixels
TRAIN_TILE_SIZE = 512
TRAIN_STRIDE = 256


def _resize_longest(max_side: int = MAX_SIDE):
    """Return Albumentations resize keeping aspect ratio with longest side = max_side."""
    return A.LongestMaxSize(max_size=max_side, interpolation=cv2.INTER_AREA)


def get_train_transforms() -> A.Compose:
    """Compose of train‑time transforms (image + mask)."""
    return A.Compose(
        [
            _resize_longest(),
            CLAHELab(clip_limit=2.0, p=1.0),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.7),
            A.Rotate(limit=15, border_mode=cv2.BORDER_REFLECT101, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.ElasticTransform(alpha=40, sigma=6, alpha_affine=0, border_mode=cv2.BORDER_REFLECT101, p=0.3),
            CutBlur(alpha=(32, 128), p=0.5),
            A.Normalize(),
            ToTensorV2(),
        ],
        additional_targets={"mask": "mask"},
    )


def get_test_transforms() -> A.Compose:
    """Test‑time transforms – only deterministic resize + normalisation."""
    return A.Compose(
        [
            _resize_longest(),
            A.Normalize(),
            ToTensorV2(),
        ],
        additional_targets={"mask": "mask"},
    )


# -----------------------------------------------------------------------------
# --- Sliding‑window tiling utilities -----------------------------------------
# -----------------------------------------------------------------------------

def tile_image(
    img: np.ndarray | torch.Tensor,
    tile_size: int = TRAIN_TILE_SIZE,
    stride: int = TRAIN_STRIDE,
) -> List[np.ndarray]:
    """Tile an image into overlapping square patches.

    Parameters
    ----------
    img : np.ndarray | torch.Tensor
        H×W×C RGB image (NumPy) or C×H×W tensor.
    tile_size : int
        Size of the square crop.
    stride : int
        Step between successive tiles (controls overlap).

    Returns
    -------
    List[np.ndarray]
        List of patches as NumPy arrays.
    """
    if isinstance(img, torch.Tensor):
        img_np = img.permute(1, 2, 0).cpu().numpy()
    else:
        img_np = img
    h, w = img_np.shape[:2]
    patches = []
    for y in range(0, max(h - tile_size, 0) + 1, stride):
        for x in range(0, max(w - tile_size, 0) + 1, stride):
            patch = img_np[y : y + tile_size, x : x + tile_size].copy()
            # Pad if we hit image border
            ph, pw = patch.shape[:2]
            if ph < tile_size or pw < tile_size:
                pad_h = tile_size - ph
                pad_w = tile_size - pw
                patch = cv2.copyMakeBorder(
                    patch,
                    0,
                    pad_h,
                    0,
                    pad_w,
                    cv2.BORDER_REFLECT101,
                )
            patches.append(patch)
    return patches


def merge_heatmap(
    heatmap_tiles: List[np.ndarray],
    img_shape: Tuple[int, int],
    tile_size: int = TRAIN_TILE_SIZE,
    stride: int = TRAIN_STRIDE,
) -> np.ndarray:
    """Reconstruct a full‑resolution heat‑map from tiled predictions.

    Uses simple averaging in overlapping regions.
    """
    h, w = img_shape[:2]
    heat = np.zeros((h, w), dtype=np.float32)
    count = np.zeros((h, w), dtype=np.float32)

    idx = 0
    for y in range(0, max(h - tile_size, 0) + 1, stride):
        for x in range(0, max(w - tile_size, 0) + 1, stride):
            heat[y : y + tile_size, x : x + tile_size] += heatmap_tiles[idx][: tile_size, : tile_size]
            count[y : y + tile_size, x : x + tile_size] += 1.0
            idx += 1

    count[count == 0] = 1.0  # avoid div‑by‑zero at borders
    return heat / count


# -----------------------------------------------------------------------------
# --- Quick test ---------------------------------------------------------------
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import pathlib
    import matplotlib.pyplot as plt
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--img", type=pathlib.Path, required=False, help="Path to an RGB image")
    parser.add_argument("--save", action="store_true", help="Save plots")
    args = parser.parse_args()

    # Load or create test image
    if args.img and args.img.exists():
        img = cv2.cvtColor(cv2.imread(str(args.img)), cv2.COLOR_BGR2RGB)
        print(f"Loaded: {args.img} {img.shape}")
    else:
        img = np.zeros((600, 800, 3), dtype=np.uint8)
        cv2.rectangle(img, (100, 100), (500, 400), (120, 120, 120), -1)
        cv2.circle(img, (300, 250), 60, (180, 180, 180), -1)
        print(f"Created test image {img.shape}")

    # Get transforms and apply
    train_transforms = get_train_transforms()
    test_transforms = get_test_transforms()
    
    train_out = train_transforms(image=img, mask=np.zeros(img.shape[:2], dtype=np.uint8))
    test_out = test_transforms(image=img, mask=np.zeros(img.shape[:2], dtype=np.uint8))
    
    # Denormalize for display
    mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
    train_img = (train_out["image"].permute(1, 2, 0).numpy() * std + mean).clip(0, 1)
    test_img = (test_out["image"].permute(1, 2, 0).numpy() * std + mean).clip(0, 1)
    
    # Get tiles
    tiles = tile_image(test_out["image"])
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle("Preprocessing Pipeline Demo", fontweight='bold')
    
    # Original
    axes[0, 0].imshow(img)
    axes[0, 0].set_title("Original")
    axes[0, 0].axis('off')
    
    # Train transform (augmented)
    axes[0, 1].imshow(train_img)
    axes[0, 1].set_title("Train Transform")
    axes[0, 1].axis('off')
    
    # Test transform (clean)
    axes[1, 0].imshow(test_img)
    axes[1, 0].set_title("Test Transform")
    axes[1, 0].axis('off')
    
    # First tile
    tile_display = (tiles[0] * std + mean).clip(0, 1)
    axes[1, 1].imshow(tile_display)
    axes[1, 1].set_title(f"Tile (512x512)")
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    if args.save:
        plt.savefig("preprocessing_demo.png", dpi=150, bbox_inches='tight')
        print("Saved: preprocessing_demo.png")
    plt.show()
    
    print(f"✅ Extracted {len(tiles)} tiles from {img.shape} image")
