"""
Evaluation utilities for MPDD-style anomaly detection.

This script computes:
    * Image-level AUROC (good vs bad) from image_scores.json
    * Pixel-level AUROC from continuous heat-maps vs. GT masks
    * PRO / sPRO (Per-Region Overlap) curve and AUC, following MVTec AD
      and MPDD evaluation practice (area under PRO for FPR <= 0.3 by default).

Assumptions / Conventions
-------------------------
Predictions directory (--pred_dir) contains files produced by patchcore_baseline.py:
    inference/
        image_scores.json            # {"/abs/path/to/img.png": score, ...}
        <stem>_score.npy             # continuous anomaly map in [0,1]
        <stem>_mask.png              # optional binarised mask (not used here)

Ground-truth directory (--gt_dir) should mirror stems of test images and contain
binary masks per anomalous image (255 for defect, 0 for background). Normal
images may have a 0-mask or be absent; missing masks are treated as all-zero.

Usage
-----
python eval_metrics.py \
    --pred_dir runs/patchcore_r50/inference \
    --gt_dir   data/test_masks \
    --image_root data/test \
    --fpr_limit 0.3 \
    --out_csv  runs/patchcore_r50/metrics.csv \
    --per_class

Dependencies: numpy, scikit-image, scikit-learn, opencv-python, pandas (optional)

Copyright 2025. MIT License.
"""

from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import cv2
from sklearn.metrics import roc_auc_score
try:
    import pandas as pd
except Exception:
    pd = None
from skimage.measure import label as cc_label

# -----------------------------------------------------------------------------
# Utility loaders
# -----------------------------------------------------------------------------

def load_image_scores(json_path: Path) -> Dict[str, float]:
    with open(json_path, "r") as f:
        data = json.load(f)
    return data


def collect_pixel_data(pred_dir: Path, gt_dir: Path, image_root: Path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Collect flattened pixel scores and GT labels for AUROC / PRO.

    Returns
    -------
    y_true : np.ndarray  (N,)
        0/1 labels per pixel (1 = defect)
    y_score : np.ndarray (N,)
        Continuous anomaly score per pixel
    stems : list[str]
        Image stems in the same order as maps were processed
    """
    score_files = sorted(pred_dir.glob("*_score.npy"))
    y_true_list, y_score_list, stems = [], [], []

    for sf in score_files:
        stem = sf.stem.replace("_score", "")
        stems.append(stem)
        score_map = np.load(sf).astype(np.float32)
        # Try to find the GT mask
        # Priority: exact filename match in gt_dir; fallback to zero mask
        gt_path_candidates = list(gt_dir.glob(f"{stem}*"))
        if len(gt_path_candidates) == 0:
            # normal image: all-zero mask
            gt_mask = np.zeros(score_map.shape, dtype=np.uint8)
        else:
            # pick the first candidate that is an image
            gt_path = gt_path_candidates[0]
            gt_mask = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
            if gt_mask is None:
                raise FileNotFoundError(f"Failed to read GT mask {gt_path}")
            # Resize GT mask if needed
            if gt_mask.shape != score_map.shape:
                gt_mask = cv2.resize(gt_mask, (score_map.shape[1], score_map.shape[0]), interpolation=cv2.INTER_NEAREST)
            gt_mask = (gt_mask > 0).astype(np.uint8)

        y_true_list.append(gt_mask.flatten())
        y_score_list.append(score_map.flatten())

    y_true = np.concatenate(y_true_list, axis=0)
    y_score = np.concatenate(y_score_list, axis=0)
    return y_true, y_score, stems

# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------

def image_level_auroc(image_scores: Dict[str, float], image_root: Path) -> float:
    """Compute AUROC over images using score dict and label from path (good/bad).

    Assumes path contains '/good/' for normal and '/bad/' for anomalous.
    """
    y_true, y_score = [], []
    for path, score in image_scores.items():
        # normal = 0, bad = 1
        lbl = 1 if "/bad/" in path.replace("\\", "/") else 0
        y_true.append(lbl)
        y_score.append(score)
    return roc_auc_score(y_true, y_score)


def pixel_level_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    return roc_auc_score(y_true, y_score)


def pro_curve(y_true: np.ndarray, y_score: np.ndarray, num_thresh: int = 200) -> Tuple[np.ndarray, np.ndarray]:
    """Compute PRO curve (Per-Region Overlap) vs FPR.

    Steps (following MVTec / MPDD conventions):
    1. Threshold score map at many levels.
    2. For each threshold, compute:
        - FPR = FP / #neg_pixels
        - PRO = mean over GT connected components of |pred ∩ gt_cc| / |gt_cc|
    3. Return arrays (fpr_list, pro_list) sorted by FPR.

    Note: This implementation expects y_true/y_score from a *single* dataset aggregation.
          It computes region overlaps per image implicitly via connected components across
          the flattened vector. For strict correctness per-image CCs should be used.
          We approximate by treating the flattened array as 1-D; instead, we operate
          globally. If you need exact per-image handling, pass maps one-by-one.
    """
    # We need GT connected components on a per-image basis for exact PRO. Since we
    # flattened the data, let's approximate by operating on a per-image basis
    # outside this function. Here, we implement a global variant for simplicity.
    # For better correctness, see `pro_curve_per_image` below.
    raise NotImplementedError("Use pro_curve_per_image for accurate computation.")


def pro_curve_per_image(score_files: List[Path], gt_dir: Path, fpr_limit: float = 0.3, num_thresh: int = 200) -> Tuple[float, np.ndarray, np.ndarray]:
    """Compute sPRO AUC (area under PRO curve for FPR ≤ fpr_limit), plus the curve.

    We iterate over images, accumulate TP/FP counts region-wise to compute global FPR,
    and average PRO over GT regions.
    """
    # Collect negatives/global counts for FPR
    # We'll build the curve by sweeping thresholds over the global score distribution
    all_scores = []
    for sf in score_files:
        m = np.load(sf)
        all_scores.append(m.flatten())
    all_scores = np.concatenate(all_scores)
    thrs = np.linspace(all_scores.min(), all_scores.max(), num_thresh)

    # Globals
    pro_vals = []  # PRO at each threshold
    fpr_vals = []  # FPR at each threshold

    # Pre-compute negative pixel count (total negatives across all images)
    total_neg = 0
    # We'll compute this inside the loop; combine later

    # For each threshold, accumulate TP, FP and compute PRO across images
    for thr in thrs:
        tp_total = 0
        fp_total = 0
        pro_regions = []
        neg_count_total = 0

        for sf in score_files:
            stem = sf.stem.replace("_score", "")
            score_map = np.load(sf)
            pred_mask = (score_map >= thr).astype(np.uint8)

            # GT mask
            gt_path_candidates = list(gt_dir.glob(f"{stem}*"))
            if len(gt_path_candidates) == 0:
                gt_mask = np.zeros(score_map.shape, dtype=np.uint8)
            else:
                gt_mask = cv2.imread(str(gt_path_candidates[0]), cv2.IMREAD_GRAYSCALE)
                gt_mask = (gt_mask > 0).astype(np.uint8)
                if gt_mask.shape != pred_mask.shape:
                    gt_mask = cv2.resize(gt_mask, (pred_mask.shape[1], pred_mask.shape[0]), interpolation=cv2.INTER_NEAREST)

            # FPR components
            neg = (gt_mask == 0)
            fp = (pred_mask == 1) & neg
            fp_total += fp.sum()
            neg_count_total += neg.sum()

            # PRO per connected component of GT
            if gt_mask.any():
                lab = cc_label(gt_mask, connectivity=2)
                for rid in range(1, lab.max() + 1):
                    region = (lab == rid)
                    inter = (pred_mask & region).sum()
                    pro_regions.append(inter / float(region.sum()))
            # If no defects in image, nothing contributes to PRO
        # End for each image

        # Aggregate for this threshold
        pro = np.mean(pro_regions) if len(pro_regions) > 0 else 0.0
        fpr = fp_total / float(neg_count_total + 1e-8)
        pro_vals.append(pro)
        fpr_vals.append(fpr)

    # Convert to arrays, sort by FPR
    fpr_vals = np.array(fpr_vals)
    pro_vals = np.array(pro_vals)
    order = np.argsort(fpr_vals)
    fpr_vals = fpr_vals[order]
    pro_vals = pro_vals[order]

    # Clip to fpr_limit and integrate
    mask = fpr_vals <= fpr_limit
    if mask.sum() < 2:
        s_pro_auc = 0.0
    else:
        from numpy import trapz
        s_pro_auc = trapz(pro_vals[mask], fpr_vals[mask]) / fpr_limit
    return s_pro_auc, fpr_vals, pro_vals

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Evaluate anomaly detection outputs.")
    ap.add_argument("--pred_dir", required=True, type=str, help="Path to predictions (inference) folder")
    ap.add_argument("--gt_dir", required=True, type=str, help="Path to GT masks folder")
    ap.add_argument("--image_root", required=True, type=str, help="Root of test images (to infer labels)")
    ap.add_argument("--fpr_limit", type=float, default=0.3, help="FPR cap for sPRO AUC")
    ap.add_argument("--num_thresh", type=int, default=200, help="Threshold samples for PRO curve")
    ap.add_argument("--out_csv", type=str, default="metrics.csv")
    ap.add_argument("--per_class", action="store_true", help="Report metrics per class as well")
    args = ap.parse_args()

    pred_dir = Path(args.pred_dir)
    gt_dir = Path(args.gt_dir)
    image_root = Path(args.image_root)

    # 1. Image-level AUROC
    image_scores = load_image_scores(pred_dir / "image_scores.json")
    img_auroc = image_level_auroc(image_scores, image_root)

    # 2. Pixel-level AUROC
    y_true, y_score, stems = collect_pixel_data(pred_dir, gt_dir, image_root)
    pix_auroc = pixel_level_auroc(y_true, y_score)

    # 3. sPRO calculation (global)
    score_files = sorted(pred_dir.glob("*_score.npy"))
    s_pro_auc, fpr_curve, pro_curve_vals = pro_curve_per_image(score_files, gt_dir, args.fpr_limit, args.num_thresh)

    results = {
        "image_auroc": img_auroc,
        "pixel_auroc": pix_auroc,
        "sPRO_AUC@{:.2f}".format(args.fpr_limit): s_pro_auc,
    }

    # Optional: per-classmetrics by parsing image path
    if args.per_class:
        class_to_scores = {}
        class_to_labels_img = {}
        # image-level
        for path, score in image_scores.items():
            cls = Path(path).parts[-3]  # data/test/<class>/<good|bad>/xxx
            lbl = 1 if "/bad/" in path.replace("\\", "/") else 0
            class_to_scores.setdefault(cls, []).append(score)
            class_to_labels_img.setdefault(cls, []).append(lbl)
        for cls in class_to_scores:
            auroc_c = roc_auc_score(class_to_labels_img[cls], class_to_scores[cls])
            results[f"image_auroc_{cls}"] = auroc_c

        # pixel-level per class
        # We'll re-load per image to avoid huge memory costs
        for cls in class_to_scores:
            y_t_c, y_s_c = [], []
            for sf in score_files:
                stem = sf.stem.replace("_score", "")
                # find image path to infer class
                # assume image root has the same structure
                # try data/test/<cls>/*/<stem>.png
                candidates = list(image_root.glob(f"{cls}/**/{stem}.*"))
                if not candidates:
                    continue
                score_map = np.load(sf)
                gt_candidates = list(gt_dir.glob(f"{stem}*"))
                if len(gt_candidates) == 0:
                    gt_mask = np.zeros(score_map.shape, dtype=np.uint8)
                else:
                    gt_mask = cv2.imread(str(gt_candidates[0]), cv2.IMREAD_GRAYSCALE)
                    gt_mask = (gt_mask > 0).astype(np.uint8)
                    if gt_mask.shape != score_map.shape:
                        gt_mask = cv2.resize(gt_mask, (score_map.shape[1], score_map.shape[0]), interpolation=cv2.INTER_NEAREST)
                y_t_c.append(gt_mask.flatten())
                y_s_c.append(score_map.flatten())
            if y_t_c:
                y_t_c = np.concatenate(y_t_c)
                y_s_c = np.concatenate(y_s_c)
                results[f"pixel_auroc_{cls}"] = roc_auc_score(y_t_c, y_s_c)

    # Save CSV / print
    if pd is not None:
        df = pd.DataFrame([results])
        df.to_csv(args.out_csv, index=False)
        print(df.to_markdown(index=False))
    else:
        print("Results:")
        for k, v in results.items():
            print(f"{k}: {v:.4f}")
        # also dump json alongside
        with open(Path(args.out_csv).with_suffix('.json'), 'w') as f:
            json.dump(results, f, indent=2)

    # Optional: save PRO curve
    np.save(Path(args.out_csv).with_suffix('.fpr.npy'), fpr_curve)
    np.save(Path(args.out_csv).with_suffix('.pro.npy'), pro_curve_vals)


if __name__ == "__main__":
    main()
