from __future__ import annotations

import hashlib
import json
import os
import random
from functools import lru_cache
from pathlib import Path

import numpy as np
import pydicom
import cv2
import torch

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
from IPython.display import display
from tqdm.auto import tqdm

from scipy.ndimage import binary_fill_holes, label
from skimage.morphology import dilation, disk

import config


def set_deterministic(seed=None):
    """Seed python, numpy and torch through Keras, then pin the cuDNN algorithm."""
    seed = config.RANDOM_STATE if seed is None else seed

    # Keras reads the backend before it is imported; the torch backend seeds torch for us
    os.environ.setdefault("KERAS_BACKEND", "torch")

    import keras

    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    keras.utils.set_random_seed(seed)

    # Keras only seeds the CPU generator, and torchvision draws its sampling from CUDA
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@lru_cache(maxsize=512)
def read_dcm(path: str) -> np.ndarray:
    """Read a DICOM image and normalize it to [0, 1] as float32."""
    ds = pydicom.dcmread(path)

    image = ds.pixel_array.astype(np.float32)

    image -= image.min()

    if image.max() > 0:
        image /= image.max()

    return image


@lru_cache(maxsize=64)
def read_image(path: str) -> np.ndarray:
    """Read a preprocessed .npy image as float32."""
    return np.load(path).astype(np.float32)


def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Bounding box of a binary mask as (xmin, ymin, xmax, ymax)."""
    ys, xs = np.where(mask > 0)

    if len(xs) == 0:
        raise ValueError("Mask is empty.")

    return (xs.min(), ys.min(), xs.max(), ys.max())


def align_mask_to_image(mask: np.ndarray, image_shape: tuple[int, int], tolerance: float) -> np.ndarray:
    """Resize a mask onto the image grid, raising if the shape differs by more than `tolerance`."""
    if mask.shape == image_shape:
        return mask

    height_ratio = abs(mask.shape[0] - image_shape[0]) / image_shape[0]
    width_ratio = abs(mask.shape[1] - image_shape[1]) / image_shape[1]

    if height_ratio > tolerance or width_ratio > tolerance:
        raise ValueError(
            f"Mask shape {mask.shape} differs from image shape {image_shape} "
            f"by more than {tolerance:.0%} (height diff {height_ratio:.1%}, "
            f"width diff {width_ratio:.1%}) "
        )

    return cv2.resize(mask, (image_shape[1], image_shape[0]), interpolation=cv2.INTER_NEAREST)


def resize_image(image, max_side, is_mask=False):
    """Scale the long side to `max_side`, downscale only; masks use nearest-neighbor."""
    interpolation = cv2.INTER_NEAREST if is_mask else cv2.INTER_AREA

    height, width = image.shape
    scale = max_side / max(height, width)

    if scale >= 1.0:
        return image

    new_width = max(1, round(width * scale))
    new_height = max(1, round(height * scale))

    return cv2.resize(image, (new_width, new_height), interpolation=interpolation)


def visualize_sample(row):
    """Display a mammogram with its breast mask, lesion mask, overlay and crops."""

    full = read_image(row["full_image"])
    crop = read_image(row["crop_image"])
    lesion_mask = read_image(row["mask_image"])
    breast_mask = read_image(row["breast_mask"]).astype(bool)

    breast = full.copy()
    breast[~breast_mask] = 0

    xmin, ymin, xmax, ymax = mask_bbox(lesion_mask)
    lesion_crop = full[ymin : ymax + 1, xmin : xmax + 1]

    fig, ax = plt.subplots(1, 7, figsize=(35, 5))

    # Full image
    ax[0].imshow(full, cmap="gray")
    ax[0].set_title("Full")
    ax[0].axis("off")

    # Breast mask
    ax[1].imshow(breast_mask, cmap="gray")
    ax[1].set_title("Breast Mask")
    ax[1].axis("off")

    # Breast image
    ax[2].imshow(breast, cmap="gray")
    ax[2].set_title("Breast")
    ax[2].axis("off")

    # Lesion mask
    ax[3].imshow(lesion_mask, cmap="gray")
    ax[3].set_title("Lesion Mask")
    ax[3].axis("off")

    # Overlay
    ax[4].imshow(full, cmap="gray")
    ax[4].imshow(lesion_mask, cmap="Reds", alpha=0.35)
    ax[4].set_title("Overlay")
    ax[4].axis("off")

    # Lesion bbox crop
    ax[5].imshow(lesion_crop, cmap="gray")
    ax[5].set_title("BBox Crop")
    ax[5].axis("off")

    # Original crop
    ax[6].imshow(crop, cmap="gray")
    ax[6].set_title("Original Crop")
    ax[6].axis("off")

    plt.suptitle(
        f'{row["patient_id"]} | '
        f'{row["left or right breast"]}_{row["image view"]} | '
        f'{row["abnormality type"]} #{row["abnormality id"]}'
    )

    display(fig)
    plt.close(fig)


# Per-class box colour: benign green, malignant red, anything else grey
CLASS_COLOURS = {"benign": "lime", "malignant": "red"}


def class_colour(label, class_names=None) -> str:
    """Colour of one class index; benign green, malignant red, anything else grey."""
    class_names = config.CLASS_NAMES if class_names is None else class_names

    name = str(class_names[int(label)]).strip().lower() if 0 <= int(label) < len(class_names) else ""

    return CLASS_COLOURS.get(name, "grey")


def visualize_detections(image, gt_boxes, gt_labels, boxes, scores, labels, class_names=None):
    """Original, ground truth and predictions side by side, every box coloured by class."""
    class_names = config.CLASS_NAMES if class_names is None else class_names

    figure, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(image, cmap="gray")
    for (x1, y1, x2, y2), label in zip(gt_boxes, gt_labels):
        colour = class_colour(label, class_names)
        axes[1].add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=colour, linewidth=2))
        axes[1].text(x1, max(0, y1 - 6), class_names[int(label)], color=colour, fontsize=9)
    axes[1].set_title("Ground truth")
    axes[1].axis("off")

    axes[2].imshow(image, cmap="gray")
    for (x1, y1, x2, y2), score, label in zip(boxes, scores, labels):
        colour = class_colour(label, class_names)
        axes[2].add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=colour, linewidth=2))
        axes[2].text(x1, max(0, y1 - 6), f"{class_names[int(label)]} {score:.2f}", color=colour, fontsize=9)
    axes[2].set_title(f"Prediction ({len(boxes)} boxes)")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()

    return figure


def plot_heatmaps(image, panels, suptitle="", mask=None):
    """An image with one low-resolution heatmap per panel, drawn on the image's own pixel frame.

    Each panel is `(values, title)`, where `values` may be coarser than the image; it is
    stretched over the image extent. `mask` gates the values, so a model's map can be shown
    only where the breast is. Every panel shares one colour scale, so they stay comparable.
    """
    panels = [(np.asarray(values, np.float32), title) for values, title in panels]

    low = min(0.0, min(float(values.min()) for values, _ in panels))
    high = max(float(values.max()) for values, _ in panels)
    high = high if high > low else low + 1.0

    extent = (0, image.shape[1], image.shape[0], 0)

    figure, axes = plt.subplots(1, len(panels), figsize=(4.8 * len(panels), 4.9))
    axes = np.atleast_1d(axes)

    drawn = None

    for axis, (values, title) in zip(axes, panels):
        axis.imshow(image, cmap="gray")

        if mask is not None:
            values = np.where(np.asarray(mask, bool), values, np.nan)

        drawn = axis.imshow(values, cmap="magma", alpha=0.55, extent=extent, vmin=low, vmax=high)
        axis.set_title(title)
        axis.axis("off")

    figure.colorbar(drawn, ax=list(axes), fraction=0.03, pad=0.02)

    if suptitle:
        figure.suptitle(suptitle)

    plt.show()

    return figure


def plot_boxes_by_score(image, boxes, scores=None, gt_boxes=None, title=""):
    """An image with every box drawn on it, coloured by score, ground truth in lime.

    Without `scores` every box is drawn in one colour, which is what a proposal stage that
    returns boxes only needs.
    """
    boxes = np.asarray(boxes, np.float32).reshape(-1, 4)

    figure, axis = plt.subplots(figsize=(6.4, 6.4))
    axis.imshow(image, cmap="gray")

    if scores is None:
        colours = ["deepskyblue"] * len(boxes)
        bar = None
    else:
        scores = np.asarray(scores, np.float32).reshape(-1)
        limit = max(float(scores.max()), 1e-6)
        colours = plt.cm.magma(scores / limit)
        bar = plt.cm.ScalarMappable(cmap="magma", norm=plt.Normalize(0.0, limit))

    for (x1, y1, x2, y2), colour in zip(boxes, colours):
        axis.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=colour, linewidth=1.2))

    if gt_boxes is not None:
        for x1, y1, x2, y2 in np.asarray(gt_boxes, np.float32).reshape(-1, 4):
            axis.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="lime", linewidth=2.0))

    if bar is not None:
        figure.colorbar(bar, ax=axis, fraction=0.046, pad=0.04)

    axis.set_title(title)
    axis.axis("off")

    plt.show()

    return figure


def box_density_map(shape, boxes, scores=None, cells=48):
    """Score-weighted box count per cell, so a field of boxes can be read as a heatmap."""
    if not len(boxes):
        return np.zeros((cells, cells), np.float32)

    boxes = np.asarray(boxes, np.float32).reshape(-1, 4)
    weights = np.ones(len(boxes), np.float32) if scores is None else np.asarray(scores, np.float32).reshape(-1)

    height, width = shape
    xs = np.clip(((0.5 * (boxes[:, 0] + boxes[:, 2])) / width * cells).astype(int), 0, cells - 1)
    ys = np.clip(((0.5 * (boxes[:, 1] + boxes[:, 3])) / height * cells).astype(int), 0, cells - 1)

    density = np.zeros((cells, cells), np.float32)
    np.add.at(density, (ys, xs), weights)

    return density


def cap_detections(raw, cap):
    """Keep each mammogram's top `cap` detections by score; 0 or None keeps everything."""
    if not cap:
        return raw

    capped = []

    for entry in raw:
        order = np.argsort(-entry["scores"])[: int(cap)]

        capped.append(
            {
                **entry,
                "boxes": entry["boxes"][order],
                "scores": entry["scores"][order],
                "labels": entry["labels"][order],
                "class_scores": entry["class_scores"][order],
            }
        )

    return capped


def visualize_comparison(image, gt_boxes, gt_labels, panels, class_names=None):
    """Original, ground truth and one axis per prediction set, every box coloured by class."""
    class_names = config.CLASS_NAMES if class_names is None else class_names

    figure, axes = plt.subplots(1, 2 + len(panels), figsize=(5.5 * (2 + len(panels)), 6))

    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(image, cmap="gray")
    for (x1, y1, x2, y2), label in zip(gt_boxes, gt_labels):
        colour = class_colour(label, class_names)
        axes[1].add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=colour, linewidth=2))
        axes[1].text(x1, max(0, y1 - 6), class_names[int(label)], color=colour, fontsize=9)
    axes[1].set_title(f"Ground truth ({len(gt_boxes)} boxes)")
    axes[1].axis("off")

    for axis, panel in zip(axes[2:], panels):
        title, boxes, scores, labels = panel[:4]
        captions = panel[4] if len(panel) > 4 else None

        axis.imshow(image, cmap="gray")
        for index, ((x1, y1, x2, y2), score, label) in enumerate(zip(boxes, scores, labels)):
            colour = class_colour(label, class_names)
            axis.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=colour, linewidth=2))
            axis.text(
                x1,
                max(0, y1 - 6),
                captions[index] if captions is not None else f"{class_names[int(label)]} {score:.2f}",
                color=colour,
                fontsize=8,
            )
        axis.set_title(f"{title} ({len(boxes)} boxes)")
        axis.axis("off")

    plt.tight_layout()
    plt.show()

    return figure


def plot_confusion(matrix, title="Class confusion"):
    """Confusion matrix as a heatmap with the count printed in every cell."""
    values = matrix.to_numpy()

    figure, axes = plt.subplots(figsize=(1.7 * len(matrix.columns) + 2.5, 1.1 * len(matrix.index) + 2))

    axes.imshow(values, cmap="Blues")

    axes.set_xticks(range(len(matrix.columns)), labels=matrix.columns)
    axes.set_yticks(range(len(matrix.index)), labels=matrix.index)

    axes.set_xlabel("Predicted")
    axes.set_ylabel("Ground truth")
    axes.set_title(title)

    limit = values.max() if values.size and values.max() > 0 else 1

    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            axes.text(
                column,
                row,
                f"{values[row, column]:,}",
                ha="center",
                va="center",
                color="white" if values[row, column] > 0.6 * limit else "black",
            )

    plt.tight_layout()
    plt.show()

    return figure


def extract_breast(image, dilation_radius=5):
    """Extract the breast mask, including the low-contrast outer boundary."""
    # Ignore true zero background when computing the low threshold
    non_zero_pixels = image[image > 0]
    if len(non_zero_pixels) == 0:
        raise RuntimeError("No foreground found.")

    # Low percentile threshold keeps the low-contrast skin line
    thresh = np.percentile(non_zero_pixels, 5)

    binary = image > thresh

    # Keep the largest connected component as the breast body
    labels, num = label(binary)
    if num == 0:
        raise RuntimeError("No foreground found.")

    counts = np.bincount(labels.ravel())
    counts[0] = 0

    largest_label = counts.argmax()
    breast_mask = labels == largest_label

    breast_mask = binary_fill_holes(breast_mask)

    # Dilate to recover the soft outer boundary
    if dilation_radius > 0:
        breast_mask = dilation(breast_mask, footprint=disk(dilation_radius))

    return breast_mask.astype(np.uint8), mask_bbox(breast_mask)


def masks_to_boxes(mask_paths):
    """Bounding boxes of lesion masks, shape (N, 4) in xyxy format."""
    boxes = []
    for path in mask_paths:
        xmin, ymin, xmax, ymax = mask_bbox(read_image(path))
        boxes.append([xmin, ymin, xmax, ymax])

    return np.asarray(boxes, dtype=np.float32)


def match_detections_to_gt(pred_boxes, pred_scores, gt_boxes, iou_threshold=0.5):
    """Greedy highest-score match of detections to GT boxes, as a GT index per detection."""
    matches = np.full(len(pred_boxes), -1, dtype=np.int64)

    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return matches

    taken = np.zeros(len(gt_boxes), dtype=bool)

    for index in np.argsort(-np.asarray(pred_scores, dtype=np.float32)):
        overlaps = iou_one_vs_many(pred_boxes[index], gt_boxes)
        overlaps[taken] = -1.0

        best = int(overlaps.argmax())

        if overlaps[best] >= iou_threshold:
            taken[best] = True
            matches[index] = best

    return matches


def match_detections(pred_boxes, pred_scores, gt_boxes, iou_threshold=0.5):
    """Greedy match of detections to ground truth, as (score, is_true_positive) pairs."""
    matches = match_detections_to_gt(pred_boxes, pred_scores, gt_boxes, iou_threshold)
    order = np.argsort(-np.asarray(pred_scores, dtype=np.float32)) if len(pred_boxes) else []

    return [(float(pred_scores[index]), bool(matches[index] >= 0)) for index in order]


def precision_recall_curve(matches, total_ground_truth):
    """Cumulative precision and recall from (score, is_tp) pairs, highest score first."""
    matches = sorted(matches, reverse=True)

    tp = 0
    fp = 0

    precision = []
    recall = []

    for _, is_tp in matches:
        if is_tp:
            tp += 1
        else:
            fp += 1

        precision.append(tp / (tp + fp))
        recall.append(tp / total_ground_truth)
    return np.asarray(precision), np.asarray(recall)


def average_precision(precision, recall):
    """Average precision with all-point interpolation."""

    precision = np.concatenate(([0.0], precision, [0.0]))
    recall = np.concatenate(([0.0], recall, [1.0]))

    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])

    indices = np.where(recall[1:] != recall[:-1])[0]
    ap = np.sum((recall[indices + 1] - recall[indices]) * precision[indices + 1])

    return float(ap)


def evaluate_detector(predictions, targets, iou_threshold=0.5, operating_threshold=None):
    """Class-agnostic AP50 plus precision, recall and F1."""
    matches = []
    total_gt = 0

    for prediction, target in zip(predictions, targets):
        gt_boxes = target["boxes"]

        total_gt += len(gt_boxes)

        matches.extend(match_detections(prediction["boxes"], prediction["scores"], gt_boxes, iou_threshold))

    precision, recall = precision_recall_curve(matches, total_gt)

    ap = average_precision(precision, recall)

    if operating_threshold is None:
        operating = matches
    else:
        operating = [match for match in matches if match[0] >= operating_threshold]

    true_positives = sum(1 for _, is_tp in operating if is_tp)

    final_precision = true_positives / len(operating) if operating else 0.0
    final_recall = true_positives / total_gt if total_gt else 0.0

    f1 = 2 * final_precision * final_recall / max(final_precision + final_recall, 1e-12)

    return {"ap50": ap, "precision": final_precision, "recall": final_recall, "f1": f1}


def evaluate_detector_by_class(
    predictions, targets, iou_threshold=0.5, operating_threshold=None, n_classes=None
) -> dict:
    """Class-agnostic AP50 plus per-class AP50 and the macro average."""
    metrics: dict = evaluate_detector(
        predictions, targets, iou_threshold=iou_threshold, operating_threshold=operating_threshold
    )

    n_classes = len(config.CLASS_NAMES) if n_classes is None else n_classes

    per_class = []
    for class_label in range(n_classes):
        class_predictions = [
            {
                "boxes": prediction["boxes"][prediction["labels"] == class_label],
                "scores": prediction["scores"][prediction["labels"] == class_label],
                "labels": prediction["labels"][prediction["labels"] == class_label],
            }
            for prediction in predictions
        ]
        class_targets = [
            {
                "boxes": target["boxes"][target["labels"] == class_label],
                "labels": target["labels"][target["labels"] == class_label],
            }
            for target in targets
        ]

        if sum(len(target["boxes"]) for target in class_targets) == 0:
            per_class.append(float("nan"))
            continue

        per_class.append(evaluate_detector(class_predictions, class_targets, iou_threshold=iou_threshold)["ap50"])

    metrics["ap50_per_class"] = tuple(per_class)
    metrics["ap50_macro"] = float(np.nanmean(per_class))

    return metrics


def iou_many_vs_many(boxes, others):
    """IoU between every box of an (N, 4) array and every box of an (M, 4) array."""
    x1 = np.maximum(boxes[:, 0][:, None], others[:, 0][None, :])
    y1 = np.maximum(boxes[:, 1][:, None], others[:, 1][None, :])
    x2 = np.minimum(boxes[:, 2][:, None], others[:, 2][None, :])
    y2 = np.minimum(boxes[:, 3][:, None], others[:, 3][None, :])

    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)

    box_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    other_area = (others[:, 2] - others[:, 0]) * (others[:, 3] - others[:, 1])
    union = box_area[:, None] + other_area[None, :] - inter

    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def iou_one_vs_many(box, boxes):
    """IoU between one xyxy box and an (N, 4) box array, as an (N,) array."""
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = box_area + boxes_area - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def containment_one_vs_many(box, boxes):
    """Intersection over the smaller box area, one xyxy box against an (N, 4) array."""
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    smaller = np.minimum(box_area, boxes_area)
    return np.where(smaller > 0, inter / np.maximum(smaller, 1e-12), 0.0)


def resize_with_boxes(image, boxes, max_size=1024):
    """Scale the long side to `max_size` and scale the xyxy boxes with it; downscale only."""

    height, width = image.shape
    scale = max_size / max(height, width)

    if scale >= 1.0:
        return image, boxes

    new_width = int(round(width * scale))
    new_height = int(round(height * scale))

    image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    boxes = boxes.astype(np.float32).copy()
    boxes *= scale

    return image, boxes


def pathology_labels(pathologies):
    """Map pathology strings onto class indexes in config.CLASS_LABELS."""
    return np.asarray([config.CLASS_LABELS[str(pathology).strip()] for pathology in pathologies], dtype=np.int64)


DETECTION_RECORD_COLUMNS = (
    "image_id",
    "x1",
    "y1",
    "x2",
    "y2",
    "score",
    "class_label",
    "class_score",
    "frame_height",
    "frame_width",
)


def detection_records(records):
    """One row per detection on the schema shared by the three detectors."""
    rows = []

    for record in records:
        boxes = np.asarray(record["boxes"], dtype=np.float32).reshape(-1, 4)
        scores = np.asarray(record["scores"], dtype=np.float32).reshape(-1)
        labels = np.asarray(record.get("labels", np.zeros(len(boxes), dtype=np.int64)), dtype=np.int64).reshape(-1)
        class_scores = np.asarray(record.get("class_scores", scores), dtype=np.float32).reshape(-1)

        height, width = record["image_shape"]

        for (x1, y1, x2, y2), score, class_label, class_score in zip(boxes, scores, labels, class_scores):
            rows.append(
                {
                    "image_id": record["image_id"],
                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2),
                    "score": float(score),
                    "class_label": int(class_label),
                    "class_score": float(class_score),
                    "frame_height": int(height),
                    "frame_width": int(width),
                }
            )

    return pd.DataFrame(rows, columns=DETECTION_RECORD_COLUMNS)


def detection_confusion(records, iou_threshold=0.5, class_names=None):
    """Ground-truth class against predicted class over matched boxes."""
    class_names = list(config.CLASS_NAMES if class_names is None else class_names)

    n_classes = len(class_names)
    labels = class_names + ["background"]

    counts = np.zeros((n_classes + 1, n_classes + 1), dtype=np.int64)

    for record in records:
        boxes = np.asarray(record["boxes"], dtype=np.float32).reshape(-1, 4)
        scores = np.asarray(record["scores"], dtype=np.float32).reshape(-1)
        predicted = np.asarray(record["labels"], dtype=np.int64).reshape(-1)

        gt_boxes = np.asarray(record["gt_boxes"], dtype=np.float32).reshape(-1, 4)
        gt_labels = np.asarray(record.get("gt_labels", np.zeros(len(gt_boxes))), dtype=np.int64).reshape(-1)

        matched = match_detections_to_gt(boxes, scores, gt_boxes, iou_threshold)

        claimed = set()

        for index, gt_index in enumerate(matched):
            if gt_index < 0:
                counts[n_classes, predicted[index]] += 1
            else:
                counts[gt_labels[gt_index], predicted[index]] += 1
                claimed.add(int(gt_index))

        # Ground-truth boxes no detection claimed were missed
        for gt_index in range(len(gt_boxes)):
            if gt_index not in claimed:
                counts[gt_labels[gt_index], n_classes] += 1

    return pd.DataFrame(counts, index=labels, columns=labels)


# Out-of-fold run bookkeeping, shared by the three detectors and parameterised by explicit paths.
FOLD_ARTIFACT_KEYS = ("history", "records", "config", "model")


def write_atomically(write, path) -> Path:
    """Run `write` on a temporary path, then rename it into place."""
    path = Path(path)
    temporary = path.with_name(f"{path.stem}.tmp{path.suffix}")
    write(temporary)
    temporary.replace(path)
    return path


def fold_artifacts(paths) -> tuple[Path, ...]:
    """The per-fold artifacts out of a detector's path set, in FOLD_ARTIFACT_KEYS order."""
    return tuple(Path(paths[key]) for key in FOLD_ARTIFACT_KEYS)


def oof_split(dataframe: pd.DataFrame, fold: int):
    """Split trainval rows on one fold; the test split never enters either side."""
    train = dataframe[dataframe["fold"] != fold].reset_index(drop=True)
    val = dataframe[dataframe["fold"] == fold].reset_index(drop=True)
    return train, val


def read_oof_manifest(path) -> dict:
    """Fold completion record; a missing manifest means no fold is complete yet."""
    path = Path(path)
    if not path.exists():
        return {"folds": {}}
    return json.loads(path.read_text())


def write_oof_manifest(path, manifest: dict) -> Path:
    """Atomic manifest write, so an interrupted run cannot leave it half written."""
    return write_atomically(lambda temporary: temporary.write_text(json.dumps(manifest, indent=2)), path)


def oof_records(records_paths) -> dict:
    """Detection records of every fold whose CSV is already on disk."""
    return {fold: pd.read_csv(path) for fold, path in dict(records_paths).items() if Path(path).exists()}


def read_fold_state(path, fingerprint: str):
    """Training state of an interrupted fold, or None when there is none to resume."""
    path = Path(path)
    if not path.exists():
        return None

    state = json.loads(path.read_text())

    if state.get("fingerprint") != fingerprint:
        return None

    return state


def write_fold_state(path, fingerprint: str, epoch: int, best_epoch: int, best_ap50: float, history, **extra) -> Path:
    """Atomic per-epoch training state, so a resumed fold keeps its history and best score."""
    payload = {
        "fingerprint": fingerprint,
        "epoch": int(epoch),
        "best_epoch": int(best_epoch),
        "best_ap50": float(best_ap50),
        "history": [{key: float(value) for key, value in row.items()} for row in history],
        **extra,
    }
    return write_atomically(lambda temporary: temporary.write_text(json.dumps(payload, indent=2)), path)


EPOCH_TABLE_CELL_WIDTH = 6  # width of a 0.0000 metric cell


class EpochTable:
    """Fixed-width per-epoch rows, so a run streams out aligned."""

    def __init__(self, columns, keys, formats, n_epochs):
        self.columns = tuple(columns)
        self.keys = tuple(keys)
        self.formats = tuple(formats)
        self.n_epochs = int(n_epochs)
        self.widths = [max(len(self.columns[0]), len(str(self.n_epochs)))]
        self.widths.extend(max(len(name), EPOCH_TABLE_CELL_WIDTH) for name in self.columns[1:])

    def render(self, cells):
        return " | ".join(cell.ljust(width) for cell, width in zip(cells, self.widths)).rstrip()

    def header(self):
        return self.render(self.columns)

    def row(self, entry):
        cells = [spec.format(entry.get(key, float("nan"))) for key, spec in zip(self.keys, self.formats)]
        return self.render(cells)


def write_history_table(history, columns, keys, formats, n_epochs, best_epoch=None):
    """Print a stored fold history the way the run that produced it streamed it."""
    table = EpochTable(columns, keys, formats, n_epochs)

    tqdm.write(table.header())

    for entry in history.to_dict(orient="records"):
        best = best_epoch is not None and int(entry.get("epoch", 0)) == int(best_epoch)
        tqdm.write(table.row(entry) + (" | Saved best model" if best else ""))


class FitHistory:
    """Per-epoch rows plus the best epoch so far, shared by the detectors' fit loops."""

    def __init__(self, history=None, best_epoch=0, best_ap50=0.0, best_weights=None):
        self.rows = [dict(row) for row in history] if history else []
        self.best_epoch = int(best_epoch)
        self.best_ap50 = float(best_ap50)
        self.best_weights = best_weights

    @property
    def history(self):
        return self.rows

    def add(self, entry, ap50, weights=None):
        """Append an epoch and adopt it as the best one when AP50 improves."""
        self.rows.append(entry)

        if ap50 > self.best_ap50:
            self.best_ap50 = float(ap50)
            self.best_epoch = int(entry["epoch"])
            self.best_weights = weights
            return True

        return False


def layer_signature(layer):
    """Layer config with the auto-generated name dropped; Keras numbers auto-named layers per build."""
    settings = dict(layer.get_config())
    settings.pop("name", None)

    return {"class": layer.__class__.__name__, "settings": settings}


def keras_model_signature(model):
    """Input shapes, output names and layer configs of a Keras model, stable across builds."""
    return {
        "inputs": [str(tuple(tensor.shape)) for tensor in model.inputs],
        "outputs": sorted(model.output_names),
        "layers": [layer_signature(layer) for layer in model.layers],
    }


def torch_model_signature(model):
    """Parameter and buffer names with their shapes, which is this model's whole graph."""
    return {"state_dict": sorted((name, tuple(tensor.shape)) for name, tensor in model.state_dict().items())}


def fingerprint(graph, settings, seed: int) -> str:
    """Short hash of a model graph and the settings its artifacts depend on."""
    settings = dict(settings)
    settings.pop("best_epoch", None)
    settings.pop("best_val_ap50", None)
    settings["seed"] = seed

    payload = json.dumps({"graph": graph, "settings": settings}, sort_keys=True, default=str)

    return hashlib.sha1(payload.encode()).hexdigest()[:12]


def group_mammograms(metadata: pd.DataFrame) -> pd.DataFrame:
    """Collapse per-lesion metadata rows into one row per mammogram."""
    groups = []
    for full_image, group in metadata.groupby("full_image", sort=False):
        groups.append(
            {
                "full_image": full_image,
                "breast_mask": group.iloc[0]["breast_mask"],
                "split": group.iloc[0]["split"],
                "fold": group.iloc[0]["fold"],
                "mask_paths": group["mask_image"].tolist(),
                "pathologies": group["pathology"].tolist(),
            }
        )
    return pd.DataFrame(groups)


def mammogram_id(full_image) -> str:
    """Stable per-mammogram key, '<patient>_<view>', matching the YOLO export name."""
    path = Path(full_image)
    return f"{path.parent.parent.name}_{path.parent.name}"
