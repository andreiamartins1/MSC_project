#!/usr/bin/env python3

import os
import re
import csv
import random
import pickle

import cv2
import numpy as np
import matplotlib.pyplot as plt
import shap

from scipy.stats import skew
from scipy.ndimage import center_of_mass, gaussian_filter
from skimage.filters import threshold_otsu
from tensorflow.keras.models import load_model

# Use tight layout for plots
plt.rcParams['figure.constrained_layout.use'] = True

# === Path configuration (adjust as needed) ===
explainer_path   = "my_explainer.pkl"
model_path       = "/DATASERVER/.../best_model.h5"
file_list_path   = "/DATASERVER/.../test_update.txt"
output_root      = "/DATASERVER/.../results/result_sequence"

INPUT_SIZE = (256, 256)
BACKGROUND_SAMPLES = 50

# === Load model (inference mode) ===
model = load_model(model_path, compile=False)
model.summary()


def preprocess_image(img_path, ann_path, input_size=INPUT_SIZE, padding=0.2):
    """
    1) Load image and annotation mask.
    2) Resize mask to image size and threshold to binary.
    3) Compute padded bounding box around mask.
    4) Crop & resize to model input, return image tensor and mask tensor.
    """
    full_bgr = cv2.imread(img_path)
    if full_bgr is None:
        raise FileNotFoundError(f"Image not found: {img_path}")
    full_img = cv2.cvtColor(full_bgr, cv2.COLOR_BGR2RGB)
    h, w = full_img.shape[:2]

    ann = cv2.imread(ann_path, cv2.IMREAD_GRAYSCALE)
    if ann is None:
        raise FileNotFoundError(f"Annotation not found: {ann_path}")

    # Binary mask
    mask_full = cv2.resize(ann, (w, h), interpolation=cv2.INTER_NEAREST) > 10
    ys, xs = np.where(mask_full)
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    box_w, box_h = x_max - x_min + 1, y_max - y_min + 1

    pad_x = int(padding * box_w)
    pad_y = int(padding * box_h)
    x0 = max(0, x_min - pad_x)
    x1 = min(w, x_max + pad_x)
    y0 = max(0, y_min - pad_y)
    y1 = min(h, y_max + pad_y)

    # Crop image and mask
    crop_img = full_img[y0:y1, x0:x1]
    crop_mask = mask_full[y0:y1, x0:x1]

    # Resize both to model input size
    crop_resized = cv2.resize(crop_img, input_size, interpolation=cv2.INTER_LINEAR).astype('float32')
    mask_resized = cv2.resize(crop_mask.astype(np.uint8), input_size,
                              interpolation=cv2.INTER_NEAREST).astype(bool)

    # Apply mask to background & input: zero out non-mask pixels
    crop_resized[~mask_resized] = 0

    # Batch dimension
    return np.expand_dims(crop_resized, axis=0), mask_resized, (x0, y0, x1, y1), full_img


def build_explainer(background, model):
    """
    Load or create a SHAP GradientExplainer with masked background.
    """
    if os.path.exists(explainer_path):
        with open(explainer_path, 'rb') as f:
            return pickle.load(f)

    explainer = shap.GradientExplainer(model, background)
    with open(explainer_path, 'wb') as f:
        pickle.dump(explainer, f)
    return explainer


def compute_shap(crop_tensor, mask, model, explainer):
    """
    Run model + explainer on one masked crop.
    Zero out SHAP values outside mask.
    Returns: raw heatmap (H,W), predicted class, predicted prob.
    """
    shap_vals = explainer.shap_values(crop_tensor)
    preds = model.predict(crop_tensor)
    pred_prob  = float(np.max(preds, axis=-1)[0])
    pred_class = int(np.argmax(preds, axis=-1)[0])

    # CHW -> HWC then average channels
    sv = shap_vals[pred_class][0]
    heatmap = sv.mean(axis=-1)

    # Zero out outside annotation
    heatmap[~mask] = 0
    return heatmap, pred_class, pred_prob


def run_shap(img_path, ann_path, explainer, model):
    crop_tensor, mask_resized, bbox, full_img = preprocess_image(img_path, ann_path)
    raw_map, cls, prob = compute_shap(crop_tensor, mask_resized, model, explainer)
    x0, y0, x1, y1 = bbox
    h_box, w_box = y1 - y0, x1 - x0

    # Upscale raw_map to crop size, then apply mask there too
    heatmap_up = cv2.resize(raw_map, (w_box, h_box), interpolation=cv2.INTER_NEAREST)
    # Create full-size mask and apply
    full_mask = cv2.resize(mask_resized.astype(np.uint8), (w_box, h_box),
                           interpolation=cv2.INTER_NEAREST).astype(bool)
    heatmap_up[~full_mask] = 0
    return full_img, heatmap_up, cls, prob, bbox


def build_background_batch(file_list, n_samples=BACKGROUND_SAMPLES):
    """
    Sample random masked crops as background for SHAP.
    """
    with open(file_list, 'r') as f:
        lines = f.readlines()
    random.shuffle(lines)

    samples = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        img_p, ann_p = parts[0], parts[1]
        if not (os.path.exists(img_p) and os.path.exists(ann_p)):
            continue
        try:
            tensor, mask, _, _ = preprocess_image(img_p, ann_p)
            samples.append(tensor)
            if len(samples) >= n_samples:
                break
        except Exception:
            continue

    if not samples:
        raise RuntimeError("No valid background samples for SHAP!")
    return np.concatenate(samples, axis=0)

# === Prepare SHAP explainer with masked background ===
background = build_background_batch(file_list_path)
explainer = build_explainer(background, model)
# === Read and group file list by sequence ===
with open(file_list_path, 'r') as f:
    lines = f.readlines()

grouped = {}
for L in lines:
    parts = L.strip().split()
    if len(parts) < 2:
        continue
    seq = re.search(r'(p\d+)', parts[0], re.IGNORECASE)
    key = seq.group(1) if seq else 'unknown'
    grouped.setdefault(key, []).append(parts)

# === Pick random sequences/images to process ===
selected = random.sample(list(grouped.keys()), min(30, len(grouped)))
to_process = []
for seq in selected:
    chosen = random.sample(grouped[seq], min(20, len(grouped[seq])))
    for img_p, ann_p, *rest in chosen:
        gt = rest[0] if rest else 'Unknown'
        to_process.append((seq, img_p, ann_p, gt))

# === Main loop: compute SHAP, metrics, and save ===
results = []
for seq, img_p, ann_p, gt_label in to_process:
    if not (os.path.exists(img_p) and os.path.exists(ann_p)):
        print(f"Warning: missing {img_p} or {ann_p}, skipping")
        continue

    try:
        full_img, heatmap_up, pred_class, pred_prob, bbox = run_shap(img_p, ann_p, explainer, model)
    except Exception as e:
        print(f"Error on {img_p}: {e}")
        continue

    x0, y0, x1, y1 = bbox

    # Ground-truth crop
    ann_full = cv2.imread(ann_p, cv2.IMREAD_GRAYSCALE)
    mask_full = cv2.resize(ann_full, (full_img.shape[1], full_img.shape[0]),
                           interpolation=cv2.INTER_NEAREST) > 10
    gt_crop = mask_full[y0:y1, x0:x1]

    # Smooth and threshold
    smooth_map = gaussian_filter(heatmap_up, sigma=1.0)
    th_pos = threshold_otsu(smooth_map)
    pos_mask = smooth_map >= th_pos

    neg_vals = -smooth_map[smooth_map < 0]
    th_neg = threshold_otsu(neg_vals) if neg_vals.size else 0
    neg_mask = smooth_map <= -th_neg

    # IoU metrics
    overlap_p = np.logical_and(gt_crop, pos_mask).sum()
    union_p   = np.logical_or(gt_crop, pos_mask).sum() + 1e-6
    iou_pos   = overlap_p / union_p * 100

    overlap_n = np.logical_and(gt_crop, neg_mask).sum()
    union_n   = np.logical_or(gt_crop, neg_mask).sum() + 1e-6
    iou_neg   = overlap_n / union_n * 100

    # Center of mass
    com_p = center_of_mass(pos_mask.astype(float))
    com_n = center_of_mass(neg_mask.astype(float))
    com_p_xy = (com_p[1], com_p[0]) if not np.isnan(com_p[0]) else (np.nan, np.nan)
    com_n_xy = (com_n[1], com_n[0]) if not np.isnan(com_n[0]) else (np.nan, np.nan)

    # Overlay colours
    overlay = full_img.copy()
    patch = overlay[y0:y1, x0:x1]
    alpha = 0.5
    patch[pos_mask] = ((1-alpha)*patch[pos_mask] + alpha*np.array([255,0,0])).astype(np.uint8)
    patch[neg_mask] = ((1-alpha)*patch[neg_mask] + alpha*np.array([0,0,255])).astype(np.uint8)
    overlay[y0:y1, x0:x1] = patch

    # Image‐quality metrics
    gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    hsv = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([hsv], [2], None, [256], [0,256]).flatten()
    skewness = skew(hist)

    # Plot and save
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    axs[0].imshow(full_img);     axs[0].axis('off'); axs[0].set_title('Original Image')
    axs[1].imshow(gt_crop, cmap='gray'); axs[1].axis('off'); axs[1].set_title('GT Mask')
    axs[2].imshow(overlay);      axs[2].axis('off')
    axs[2].set_title(f'SHAP Overlay\nIoU+:{iou_pos:.1f}%, IoU-:{iou_neg:.1f}%')
    axs[3].imshow(smooth_map, cmap='seismic'); axs[3].axis('off'); axs[3].set_title('SHAP Heatmap')

    seq_dir = os.path.join(output_root, seq)
    os.makedirs(seq_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(img_p))[0]
    fig.savefig(os.path.join(seq_dir, f"result_{base}.png"), dpi=200, bbox_inches='tight')
    plt.close(fig)

    # Record metrics
    results.append({
        'sequence':        seq,
        'image':           base,
        'predicted_class': pred_class,
        'probability':     pred_prob,
        'ground_truth':    gt_label,
        'iou_positive':    iou_pos,
        'iou_negative':    iou_neg,
        'laplacian_var':   lap_var,
        'skewness':        skewness,
        'com_pos_x':       com_p_xy[0],
        'com_pos_y':       com_p_xy[1],
        'com_neg_x':       com_n_xy[0],
        'com_neg_y':       com_n_xy[1],
    })

# Save all results to CSV
os.makedirs(output_root, exist_ok=True)
csv_path = os.path.join(output_root, 'results.csv')
if results:
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"Results CSV saved to {csv_path}")
else:
    print("No results to save.")
