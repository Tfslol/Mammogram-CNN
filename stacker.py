"""Detection stacker: fuses the three detectors' out-of-fold candidates on the anchor grid."""

from __future__ import annotations

import os

# Keras reads the backend before it is imported
os.environ.setdefault("KERAS_BACKEND", "torch")

import cv2
import numpy as np
import pandas as pd
import keras
from keras import layers, ops
from tqdm.auto import tqdm

import config
import utility
import Detector
import cnn_detector

# This module's row in config.FOLD_LAYOUT
MODEL_NAME = "stacker"

BASE_MODELS = config.STACKER_BASE_MODELS
BASE_LABELS = {"cnn": "CNN", "faster_rcnn": "Faster R-CNN", "yolo": "YOLO"}
CHANNELS_PER_MODEL = 1 + 4 + config.CLASS_COUNT
INPUT_CHANNELS = CHANNELS_PER_MODEL * len(BASE_MODELS)

GRID_SIZE = cnn_detector.CNN_GRID_SIZE
ANCHOR_COUNT = cnn_detector.CNN_ANCHOR_COUNT
TOTAL_ANCHORS = GRID_SIZE * GRID_SIZE * ANCHOR_COUNT


def assign_anchors(boxes):
    """Highest-IoU anchor of every box, plus that IoU."""
    if len(boxes) == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.float32)

    overlaps = utility.iou_many_vs_many(boxes, cnn_detector.CNN_ANCHOR_CORNERS)
    index = overlaps.argmax(axis=1)

    return index, overlaps[np.arange(len(index)), index]


def anchor_deltas(boxes, index):
    """Anchor deltas taking each assigned anchor to its box, in the CNN's parameterisation."""
    anchors = cnn_detector.CNN_ANCHOR_CORNERS[index]

    width = anchors[:, 2] - anchors[:, 0]
    height = anchors[:, 3] - anchors[:, 1]
    center_x = anchors[:, 0] + 0.5 * width
    center_y = anchors[:, 1] + 0.5 * height

    return np.stack(
        [
            (0.5 * (boxes[:, 0] + boxes[:, 2]) - center_x) / width,
            (0.5 * (boxes[:, 1] + boxes[:, 3]) - center_y) / height,
            np.log(np.maximum(boxes[:, 2] - boxes[:, 0], 1e-6) / width),
            np.log(np.maximum(boxes[:, 3] - boxes[:, 1], 1e-6) / height),
        ],
        axis=-1,
    ).astype(np.float32)


def rasterize(records):
    """One detector's records for one mammogram as (anchor index, 7 channels) arrays."""
    boxes = records[["x1", "y1", "x2", "y2"]].to_numpy(np.float32)
    scores = records["score"].to_numpy(np.float32)
    labels = records["class_label"].to_numpy(np.int64)
    class_scores = records["class_score"].to_numpy(np.float32)

    index, _ = assign_anchors(boxes)

    values = np.zeros((len(boxes), CHANNELS_PER_MODEL), np.float32)
    values[:, 0] = scores
    values[:, 1:5] = anchor_deltas(boxes, index)
    values[labels == 0, 5] = class_scores[labels == 0]
    values[labels == 1, 6] = class_scores[labels == 1]

    # Descending score, so the first occurrence of an anchor is its strongest box
    order = np.argsort(-scores)
    index, values = index[order], values[order]

    _, first = np.unique(index, return_index=True)

    return index[first], values[first]


def features_for(raster):
    """The 21-channel stacker input of one mammogram."""
    flat = np.zeros((TOTAL_ANCHORS, INPUT_CHANNELS), np.float32)

    for position, model in enumerate(BASE_MODELS):
        index, values = raster[model]
        flat[index, position * CHANNELS_PER_MODEL : (position + 1) * CHANNELS_PER_MODEL] = values

    return flat.reshape(GRID_SIZE, GRID_SIZE, ANCHOR_COUNT, INPUT_CHANNELS)


def ground_truth(row):
    """Ground-truth boxes, class labels and breast grid of a row, in the CNN's canvas frame."""
    mask = utility.read_image(row.breast_mask)
    boxes = utility.masks_to_boxes(row.mask_paths)
    labels = utility.pathology_labels(row.pathologies)

    scale = config.CNN_IMAGE_SIZE / max(mask.shape)

    if scale < 1.0:
        mask = cv2.resize(
            mask,
            (int(round(mask.shape[1] * scale)), int(round(mask.shape[0] * scale))),
            interpolation=cv2.INTER_NEAREST,
        )
        boxes = boxes.astype(np.float32) * scale

    canvas = np.zeros((config.CNN_IMAGE_SIZE, config.CNN_IMAGE_SIZE), np.float32)
    canvas[: mask.shape[0], : mask.shape[1]] = mask

    return boxes, labels, cnn_detector.cnn_breast_grid(canvas)


def records_by_image(lesion_type, folds):
    """Every fold's exported records joined into {image_id: {model: frame}}."""
    lookup = {}

    for fold in folds:
        for model in BASE_MODELS:
            path = config.fold_paths(model, lesion_type, fold)["records"]

            if not path.exists():
                raise FileNotFoundError(f"{model} fold {fold} records are missing: {path}")

            for image_id, frame in pd.read_csv(path).groupby("image_id", sort=False):
                lookup.setdefault(image_id, {})[model] = frame

    return lookup


def build_grid(dataframe, lesion_type, folds, lookup=None):
    """Per-mammogram stacker input and ground truth, keyed by image id."""
    explicit = lookup is not None
    lookup = records_by_image(lesion_type, folds) if lookup is None else lookup
    empty = pd.DataFrame(columns=utility.DETECTION_RECORD_COLUMNS)

    grid = {}

    for row in dataframe.itertuples():
        image_id = utility.mammogram_id(row.full_image)

        if image_id not in lookup and not explicit:
            raise KeyError(f"{image_id} has no out-of-fold predictions")

        found = lookup.get(image_id, {})
        boxes, labels, breast = ground_truth(row)

        grid[image_id] = {
            "raster": {model: rasterize(found[model]) if model in found else rasterize(empty) for model in BASE_MODELS},
            "gt_boxes": boxes,
            "gt_labels": labels,
            "breast": breast,
        }

    return grid


def sample_targets(targets, forced=None):
    """Every positive plus a random sample of negatives; the rest is ignored."""
    flat = targets.reshape(-1).copy()

    positive = flat == 1
    negative = np.nonzero(flat == 0)[0]

    count = int(positive.sum())
    wanted = (
        min(config.STACKER_MAX_NEGATIVES, count * config.STACKER_NEGATIVE_RATIO)
        if count
        else min(config.STACKER_MAX_NEGATIVES, len(negative))
    )

    if wanted < len(negative):
        negative = np.random.choice(negative, size=wanted, replace=False)

    keep = np.zeros(len(flat), bool)
    keep[positive] = True
    keep[negative] = True

    if forced is not None:
        keep |= forced.reshape(-1)

    flat[~keep] = -1.0

    return flat.reshape(targets.shape)


class StackerDataset(keras.utils.PyDataset):
    """Stacker inputs with the CNN's dense anchor targets. Nothing here is augmented."""

    def __init__(self, dataframe, grid, batch_size, candidate_supervision=False, **kwargs):
        super().__init__(**kwargs)
        self.dataframe = dataframe.reset_index(drop=True)
        self.grid = grid
        self.batch_size = batch_size
        self.candidate_supervision = candidate_supervision

    def __len__(self):
        return int(np.ceil(len(self.dataframe) / self.batch_size))

    def __getitem__(self, index):
        rows = self.dataframe.iloc[index * self.batch_size : (index + 1) * self.batch_size]

        features = np.empty((len(rows), GRID_SIZE, GRID_SIZE, ANCHOR_COUNT, INPUT_CHANNELS), np.float32)
        objectness = np.empty((len(rows), GRID_SIZE, GRID_SIZE, ANCHOR_COUNT), np.float32)
        boxes = np.empty((len(rows), GRID_SIZE, GRID_SIZE, ANCHOR_COUNT, 5), np.float32)
        classes = np.empty((len(rows), GRID_SIZE, GRID_SIZE, ANCHOR_COUNT, 2), np.float32)

        for i, row in enumerate(rows.itertuples()):
            entry = self.grid[utility.mammogram_id(row.full_image)]

            labels, deltas, class_targets = cnn_detector.cnn_dense_targets(
                entry["gt_boxes"], entry["gt_labels"], entry["breast"]
            )

            forced = occupied_map(entry["raster"], dilate=False) if self.candidate_supervision else None

            if forced is not None:
                # An occupied anchor is a real candidate, so it ranks as a negative
                labels[(labels == -1.0) & forced] = 0.0

            labels = sample_targets(labels, forced)

            # Class divisor, kept on the same population the objectness loss normalises over
            divisor = labels != -1.0

            features[i] = features_for(entry["raster"])
            objectness[i] = labels
            boxes[i] = np.concatenate([labels[..., None], deltas], axis=-1)
            classes[i, ..., 0] = class_targets
            classes[i, ..., 1] = divisor

        return features, {"objectness": objectness, "box": boxes, "classes": classes}


def stacker_normalization(norm=None):
    """Per-position LayerNorm, because a batch or group norm would only see the empty anchors."""
    norm = config.STACKER_NORM if norm is None else norm

    if norm == "none":
        return layers.Identity()
    if norm == "batch":
        return layers.BatchNormalization()
    if norm == "group":
        return layers.GroupNormalization(groups=config.CNN_GROUP_NORM_GROUPS)

    return layers.LayerNormalization()


def stacker_conv(x, filters, kernel=3, double=None, norm=None):
    """Conv + norm + ReLU, twice when `double` mirrors the detector's `conv_block`."""
    double = config.STACKER_DOUBLE_CONV if double is None else double

    for _ in range(2 if double else 1):
        x = layers.Conv2D(
            filters, kernel, padding="same", use_bias=False, kernel_regularizer=cnn_detector.l2_regularizer()
        )(x)
        x = stacker_normalization(norm)(x)
        x = layers.ReLU()(x)

    return x


@keras.saving.register_keras_serializable(package="stacker")
class ObjectnessPrior(keras.layers.Layer):
    """Adds the detectors' own best score, as a logit, to the stacker's objectness logits."""

    def call(self, inputs):
        predictions, logits = inputs

        channels = [index * CHANNELS_PER_MODEL for index in range(len(BASE_MODELS))]
        prior = ops.max(ops.take(predictions, np.asarray(channels), axis=-1), axis=-1)
        prior = ops.clip(prior, 1e-4, 1.0 - 1e-4)

        return logits + ops.log(prior) - ops.log(1.0 - prior)


def build_stacker(
    filters=None,
    stages=None,
    head_filters=None,
    kernel=None,
    double=None,
    projection=None,
    anchor_width=None,
    class_stop_gradient=None,
    norm=None,
    objectness_prior=None,
    candidate_supervision=None,
):
    """The stacked dense refiner: concatenated predictions in, refined detections out."""
    filters = config.STACKER_FILTERS if filters is None else filters
    stages = config.STACKER_STAGES if stages is None else stages
    head_filters = config.STACKER_HEAD_FILTERS if head_filters is None else head_filters
    kernel = config.STACKER_KERNEL_SIZE if kernel is None else kernel
    projection = config.STACKER_PROJECTION if projection is None else projection
    anchor_width = config.STACKER_ANCHOR_WIDTH if anchor_width is None else anchor_width
    class_stop_gradient = config.STACKER_CLASS_STOP_GRADIENT if class_stop_gradient is None else class_stop_gradient
    objectness_prior = config.STACKER_OBJECTNESS_PRIOR if objectness_prior is None else objectness_prior

    width = ANCHOR_COUNT if anchor_width else 1

    inputs = keras.Input((GRID_SIZE, GRID_SIZE, ANCHOR_COUNT, INPUT_CHANNELS), name="predictions")
    x = layers.Reshape((GRID_SIZE, GRID_SIZE, ANCHOR_COUNT * INPUT_CHANNELS))(inputs)

    if projection:
        x = stacker_conv(x, width * filters, kernel=1, norm=norm)

    for _ in range(stages):
        x = stacker_conv(x, width * filters, kernel=kernel, norm=norm)

    # The class head reads a gated copy when the class loss must not reach the trunk
    gated = cnn_detector.StopGradient()(x) if class_stop_gradient else x

    outputs = {}

    for name, count, source in (("objectness", 1, x), ("box", 4, x), ("classes", config.CLASS_COUNT, gated)):
        head = stacker_conv(source, width * head_filters, kernel=kernel, norm=norm)
        head = layers.Conv2D(ANCHOR_COUNT * count, 1, name=f"{name}_logits")(head)

        # Objectness carries no trailing axis, matching the detector's own output
        shape = (GRID_SIZE, GRID_SIZE, ANCHOR_COUNT) if count == 1 else (GRID_SIZE, GRID_SIZE, ANCHOR_COUNT, count)

        if name == "objectness" and objectness_prior:
            outputs[name] = ObjectnessPrior(name="objectness")([inputs, layers.Reshape(shape)(head)])
        else:
            outputs[name] = layers.Reshape(shape, name=name)(head)

    return keras.Model(inputs, outputs, name="stacker")


def occupied_map(raster, dilate=True):
    """Anchors that hold a candidate, optionally dilated by one cell."""
    grid = np.zeros((GRID_SIZE, GRID_SIZE, ANCHOR_COUNT), bool)

    flat = grid.reshape(-1)

    for model in BASE_MODELS:
        index, values = raster[model]
        if len(index):
            flat[index] = True

    if dilate:
        grid = cv2.dilate(grid.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)

    return grid


def stacker_predict_raw(
    model, dataframe, grid, score_threshold=None, batch_size=None, desc="predict", progress=None, gate_empty=None
):
    """Decoded candidates per mammogram, after NMS, with their ground truth."""
    score_threshold = config.STACKER_SCORE_THRESHOLD if score_threshold is None else score_threshold
    batch_size = config.STACKER_BATCH_SIZE if batch_size is None else batch_size
    gate_empty = config.STACKER_GATE_EMPTY if gate_empty is None else gate_empty

    dataframe = dataframe.reset_index(drop=True)
    batches = list(range(0, len(dataframe), batch_size))

    local_bar = progress is None
    if local_bar:
        progress = tqdm(total=len(batches), desc=desc, leave=False)
    else:
        progress.reset(total=len(batches))
        progress.set_description(desc)

    raw = []

    try:
        for start in batches:
            rows = dataframe.iloc[start : start + batch_size]
            entries = [grid[utility.mammogram_id(path)] for path in rows["full_image"]]
            features = np.stack([features_for(entry["raster"]) for entry in entries])
            outputs = model.predict_on_batch(features)

            for entry, score_map, box_map, class_map in zip(
                entries, outputs["objectness"], outputs["box"], outputs["classes"]
            ):
                if gate_empty:
                    score_map = np.where(occupied_map(entry["raster"]), score_map, -30.0)

                boxes, scores, labels, class_scores = cnn_detector.cnn_decode(
                    score_map, box_map, class_map, entry["breast"], score_threshold
                )
                keep = cnn_detector.cnn_nms(
                    boxes,
                    scores,
                    iou_threshold=config.STACKER_NMS_THRESHOLD,
                    containment_threshold=config.STACKER_NMS_CONTAINMENT,
                )

                raw.append(
                    {
                        # The CNN's square canvas, so the frame includes the zero padding
                        "image_shape": (config.CNN_IMAGE_SIZE, config.CNN_IMAGE_SIZE),
                        "boxes": boxes[keep],
                        "scores": scores[keep],
                        "labels": labels[keep],
                        "class_scores": class_scores[keep],
                        "gt_boxes": entry["gt_boxes"],
                        "gt_labels": entry["gt_labels"],
                    }
                )

            progress.update(1)
    finally:
        if local_bar:
            progress.close()

    return raw


def stacker_validate(
    model, dataframe, grid, score_threshold=config.STACKER_SCORE_THRESHOLD, desc="val", progress=None, gate_empty=None
):
    """AP50 plus precision, recall, F1 and detections per image at the operating threshold."""
    raw = stacker_predict_raw(
        model, dataframe, grid, score_threshold, desc=desc, progress=progress, gate_empty=gate_empty
    )

    return Detector.detection_metrics(raw, config.STACKER_IOU_THRESHOLD, config.STACKER_INFERENCE_SCORE_THRESHOLD)


def stacker_predict_image(model, row, entry, score_threshold=None):
    """Full stacker pipeline for one mammogram: canvas and ground truth, then final boxes."""
    score_threshold = config.STACKER_SCORE_THRESHOLD if score_threshold is None else score_threshold

    image, _, gt_boxes, gt_labels = cnn_detector.cnn_prepare_image(row)

    outputs = model.predict_on_batch(features_for(entry["raster"])[None])
    boxes, scores, labels, _ = cnn_detector.cnn_decode(
        outputs["objectness"][0], outputs["box"][0], outputs["classes"][0], entry["breast"], score_threshold
    )

    keep = scores >= config.STACKER_INFERENCE_SCORE_THRESHOLD
    boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

    if len(boxes):
        keep = cnn_detector.cnn_nms(
            boxes,
            scores,
            iou_threshold=config.STACKER_NMS_THRESHOLD,
            containment_threshold=config.STACKER_NMS_CONTAINMENT,
        )
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

    return image, gt_boxes, gt_labels, boxes, scores, labels


class StackerMetrics(keras.callbacks.Callback):
    """Epoch rows, the best raw weights and whichever progress bars the caller passed in."""

    def __init__(
        self,
        dataframe,
        grid,
        outer,
        inner,
        train_steps,
        n_epochs,
        tag="",
        history=None,
        best_ap50=0.0,
        best_epoch=0,
        best_weights=None,
    ):
        super().__init__()
        self.dataframe = dataframe
        self.grid = grid
        self.outer = outer
        self.inner = inner
        self.train_steps = train_steps
        self.n_epochs = n_epochs
        self.tag = tag
        self.table = utility.EpochTable(
            cnn_detector.EPOCH_TABLE_COLUMNS, cnn_detector.EPOCH_TABLE_KEYS, cnn_detector.EPOCH_TABLE_FORMATS, n_epochs
        )
        self.fit = utility.FitHistory(history, best_epoch, best_ap50, best_weights)
        self.print_header = True

    @property
    def history(self):
        return self.fit.rows

    @property
    def best_epoch(self):
        return self.fit.best_epoch

    @property
    def best_ap50(self):
        return self.fit.best_ap50

    @property
    def best_weights(self):
        return self.fit.best_weights

    def on_epoch_begin(self, epoch, logs=None):
        if self.print_header:
            self.print_header = False
            tqdm.write(self.table.header())

            # Re-print finished epochs so a resumed fold still shows a complete table
            for entry in self.history:
                tqdm.write(self.table.row(entry))

        self.outer.set_description(f"Epochs {epoch + 1}/{self.n_epochs}")
        self.inner.reset(total=self.train_steps)
        self.inner.set_description(f"{self.tag}Epoch {epoch + 1}/{self.n_epochs} [Train]")

    def on_train_batch_end(self, batch, logs=None):
        logs = logs or {}
        self.inner.update(1)
        self.inner.set_postfix_str(f"Loss: {logs.get('loss', float('nan')):.4f}")

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        self.inner.set_postfix_str("")

        metrics = stacker_validate(
            self.model,
            self.dataframe,
            self.grid,
            desc=f"{self.tag}Epoch {epoch + 1}/{self.n_epochs} [Val]",
            progress=self.inner,
        )

        loss_terms = {f"{head}_loss": logs.get(f"{head}_loss", np.nan) for head in ("objectness", "box", "classes")}

        entry = {
            "epoch": epoch + 1,
            "train_loss": logs.get("loss", np.nan),
            **loss_terms,
            **{key: value for key, value in metrics.items() if key != "ap50_per_class"},
            **{f"ap50_{name}": value for name, value in zip(config.CLASS_NAMES, metrics["ap50_per_class"])},
        }

        improved = self.fit.add(entry, metrics["ap50"], [weight.copy() for weight in self.model.get_weights()])

        tqdm.write(self.table.row(entry) + (" | Saved best model" if improved else ""))

        self.outer.update(1)


def base_records(lesion_type, model, folds):
    """Every fold's exported records of one detector, concatenated into one frame."""
    frames = []

    for fold in folds:
        path = config.fold_paths(model, lesion_type, fold)["records"]

        if path.exists():
            frames.append(pd.read_csv(path))

    if not frames:
        return pd.DataFrame(columns=utility.DETECTION_RECORD_COLUMNS)

    return pd.concat(frames, ignore_index=True)


def base_raw(records, grid, image_ids):
    """Per-image predictions of one recorded detector against the shared ground truth."""
    grouped = {image_id: frame for image_id, frame in records.groupby("image_id", sort=False)}
    empty = pd.DataFrame(columns=utility.DETECTION_RECORD_COLUMNS)

    raw = []

    for image_id in image_ids:
        frame = grouped.get(image_id, empty)
        entry = grid[image_id]

        raw.append(
            {
                "image_shape": (config.CNN_IMAGE_SIZE, config.CNN_IMAGE_SIZE),
                "boxes": frame[["x1", "y1", "x2", "y2"]].to_numpy(np.float32),
                "scores": frame["score"].to_numpy(np.float32),
                "labels": frame["class_label"].to_numpy(np.int64),
                "class_scores": frame["class_score"].to_numpy(np.float32),
                "gt_boxes": entry["gt_boxes"],
                "gt_labels": entry["gt_labels"],
            }
        )

    return raw


def oracle_recall(records, grid, image_ids, iou_threshold=0.5):
    """Share of ground-truth boxes a candidate set reaches, ignoring how they are ranked."""
    grouped = {image_id: frame for image_id, frame in records.groupby("image_id", sort=False)}

    found = 0
    total = 0

    for image_id in image_ids:
        gt_boxes = grid[image_id]["gt_boxes"]
        total += len(gt_boxes)

        frame = grouped.get(image_id)
        if len(gt_boxes) == 0 or frame is None:
            continue

        boxes = frame[["x1", "y1", "x2", "y2"]].to_numpy(np.float32)
        found += int((utility.iou_many_vs_many(boxes, gt_boxes).max(axis=0) >= iou_threshold).sum())

    return found / max(total, 1)


def architecture_defaults():
    """The baseline architecture of stacker_architecture.md, as build_stacker's arguments."""
    return {
        "filters": config.STACKER_FILTERS,
        "stages": config.STACKER_STAGES,
        "head_filters": config.STACKER_HEAD_FILTERS,
        "kernel": config.STACKER_KERNEL_SIZE,
        "double": config.STACKER_DOUBLE_CONV,
        "projection": config.STACKER_PROJECTION,
        "anchor_width": config.STACKER_ANCHOR_WIDTH,
        "class_stop_gradient": config.STACKER_CLASS_STOP_GRADIENT,
        "norm": config.STACKER_NORM,
        "objectness_prior": config.STACKER_OBJECTNESS_PRIOR,
        "candidate_supervision": config.STACKER_CANDIDATE_SUPERVISION,
    }


def base_fingerprints(lesion_type):
    """Recorded fold fingerprints of the three detectors the stacker reads."""
    recorded = {}

    for model in BASE_MODELS:
        manifest = utility.read_oof_manifest(config.oof_manifest_path(model, lesion_type))
        recorded[model] = {fold: entry.get("fingerprint") for fold, entry in sorted(manifest["folds"].items())}

    return recorded


def training_config(lesion_type, architecture, base_models, best_epoch=None, best_val_ap50=None):
    """Pipeline settings a checkpoint depends on, so any consumer can reproduce it."""
    return {
        "lesion_type": lesion_type,
        "base_models": list(base_models),
        "base_fingerprints": base_fingerprints(lesion_type),
        "image_size": config.CNN_IMAGE_SIZE,
        "feature_stride": config.CNN_FEATURE_STRIDE,
        "anchor_sizes": list(config.CNN_ANCHOR_SIZES),
        "anchor_aspect_ratios": list(config.CNN_ANCHOR_ASPECT_RATIOS),
        "anchor_iou_positive": config.CNN_ANCHOR_IOU_POSITIVE,
        "anchor_iou_negative": config.CNN_ANCHOR_IOU_NEGATIVE,
        "mask_min_fraction": config.CNN_MASK_MIN_FRACTION,
        "class_names": list(config.CLASS_NAMES),
        "architecture": dict(architecture),
        "kernel_l2": config.CNN_KERNEL_L2,
        "norm": config.STACKER_NORM,
        "group_norm_groups": config.CNN_GROUP_NORM_GROUPS,
        "negative_ratio": config.STACKER_NEGATIVE_RATIO,
        "max_negatives": config.STACKER_MAX_NEGATIVES,
        "box_loss_weight": config.STACKER_BOX_LOSS_WEIGHT,
        "class_loss_weight": config.STACKER_CLASS_LOSS_WEIGHT,
        "class_loss_weight_start": config.STACKER_CLASS_LOSS_WEIGHT_START,
        "class_loss_ramp_epochs": config.STACKER_CLASS_LOSS_RAMP_EPOCHS,
        "batch_size": config.STACKER_BATCH_SIZE,
        "epochs": config.STACKER_NUM_EPOCHS,
        "score_threshold": config.STACKER_SCORE_THRESHOLD,
        "inference_score_threshold": config.STACKER_INFERENCE_SCORE_THRESHOLD,
        "iou_threshold": config.STACKER_IOU_THRESHOLD,
        "nms_threshold": config.STACKER_NMS_THRESHOLD,
        "nms_containment": config.STACKER_NMS_CONTAINMENT,
        "best_epoch": best_epoch,
        "best_val_ap50": best_val_ap50,
    }


class Stacker(Detector.Detector):
    """Fuses the three detectors' out-of-fold candidates into refined detections."""

    model_name = MODEL_NAME
    default_epochs = config.STACKER_NUM_EPOCHS
    score_threshold = config.STACKER_SCORE_THRESHOLD
    inference_score_threshold = config.STACKER_INFERENCE_SCORE_THRESHOLD
    iou_threshold = config.STACKER_IOU_THRESHOLD
    stacker_score_threshold = config.STACKER_SCORE_THRESHOLD
    stacker_top_k = config.STACKER_TOP_K
    resume_artifact_keys = ("progress", "state", "best")
    epoch_table_columns = cnn_detector.EPOCH_TABLE_COLUMNS
    epoch_table_keys = cnn_detector.EPOCH_TABLE_KEYS
    epoch_table_formats = cnn_detector.EPOCH_TABLE_FORMATS

    def __init__(self, architecture=None, lesion_type="mass", tag=None):
        super().__init__()
        self.architecture = architecture_defaults() if architecture is None else dict(architecture)
        self.lesion_type = lesion_type
        self.tag = tag
        # A variant is a separate row in the fold layout, so it gets its own artifacts
        self.model_name = MODEL_NAME if tag is None else f"{MODEL_NAME}_{tag}"
        config.FOLD_LAYOUT.setdefault(self.model_name, (config.STACKER_WORKSPACE, ".keras"))
        self.grid = None

    # model

    def build(self):
        self.model = build_stacker(**self.architecture)
        return self.model

    def prepare_model(self):
        """Build the fresh model whose graph the fold's fingerprint is taken over."""
        self.build()

    def model_signature(self, lesion_type):
        return utility.keras_model_signature(self.model)

    def training_config(self, lesion_type, best_epoch=None, best_val_ap50=None):
        return training_config(lesion_type, self.architecture, BASE_MODELS, best_epoch, best_val_ap50)

    def compile(self, steps_per_epoch, epochs=None):
        """Compile with the warmup + cosine schedule sized for the run's own epoch budget."""
        epochs = config.STACKER_NUM_EPOCHS if epochs is None else int(epochs)

        schedule = cnn_detector.WarmupCosineDecay(
            peak_learning_rate=config.STACKER_LR,
            warmup_steps=config.STACKER_WARMUP_EPOCHS * steps_per_epoch,
            total_steps=epochs * steps_per_epoch,
            min_learning_rate=config.STACKER_MIN_LR,
        )

        optimizer = cnn_detector.exclude_from_weight_decay(
            keras.optimizers.AdamW(learning_rate=schedule, weight_decay=config.STACKER_WEIGHT_DECAY)
        )

        self.model.compile(
            optimizer=optimizer,
            loss={
                "objectness": cnn_detector.MaskedBinaryCrossentropy(),
                "box": cnn_detector.MaskedHuber(),
                "classes": cnn_detector.MaskedClassCrossentropy(),
            },
            loss_weights={
                "objectness": 1.0,
                "box": config.STACKER_BOX_LOSS_WEIGHT,
                "classes": config.STACKER_CLASS_LOSS_WEIGHT,
            },
        )

        return schedule

    def dataset(self, dataframe, augment=True):
        return StackerDataset(
            dataframe,
            self.grid,
            config.STACKER_BATCH_SIZE,
            candidate_supervision=self.architecture.get("candidate_supervision", False),
        )

    def prepare(self, row):
        """Stacker input, ground-truth boxes and class labels of one row."""
        entry = self.grid[utility.mammogram_id(row.full_image)]

        return features_for(entry["raster"]), entry["gt_boxes"], entry["gt_labels"]

    def prepare_fold_data(self, trainval_df, fold, lesion_type):
        """Build the stacker input and the ground truth of every trainval mammogram."""
        self.lesion_type = lesion_type
        self.grid = build_grid(trainval_df, lesion_type, range(config.N_FOLDS))

    # training

    def train(
        self,
        train_df,
        val_df,
        epochs=None,
        tag="",
        outer=None,
        inner=None,
        resume=None,
        checkpoint_paths=None,
        fingerprint=None,
    ):
        """Fit the compiled stacker, validating the best raw weights every epoch."""
        epochs = config.STACKER_NUM_EPOCHS if epochs is None else epochs

        dataset = self.dataset(train_df)

        if resume is None:
            # A fresh fold compiles its own warmup and cosine schedule over the fold's step count
            self.compile(steps_per_epoch=len(dataset), epochs=epochs)

        resume = {} if resume is None else resume
        initial_epoch = int(resume.get("initial_epoch", 0))

        own_bars = outer is None
        if own_bars:
            outer = tqdm(total=max(epochs - initial_epoch, 0), position=0, desc="Epochs")
            inner = tqdm(total=0, position=1, desc="Initializing...")

        metrics = StackerMetrics(
            val_df,
            self.grid,
            outer,
            inner,
            train_steps=len(dataset),
            n_epochs=epochs,
            tag=tag,
            history=resume.get("history"),
            best_ap50=resume.get("best_ap50", 0.0),
            best_epoch=resume.get("best_epoch", 0),
            best_weights=resume.get("best_weights"),
        )

        callbacks = [
            metrics,
            cnn_detector.ClassLossRamp(
                start=config.STACKER_CLASS_LOSS_WEIGHT_START,
                end=config.STACKER_CLASS_LOSS_WEIGHT,
                ramp_epochs=config.STACKER_CLASS_LOSS_RAMP_EPOCHS,
                total_epochs=epochs,
            ),
        ]
        if checkpoint_paths is not None:
            callbacks.append(cnn_detector.FoldCheckpoint(metrics, checkpoint_paths, fingerprint))

        try:
            if initial_epoch < epochs:
                self.model.fit(dataset, epochs=epochs, initial_epoch=initial_epoch, verbose=0, callbacks=callbacks)
        finally:
            if own_bars:
                inner.close()
                outer.close()

        if metrics.best_weights is not None:
            self.model.set_weights(metrics.best_weights)

        return Detector.TrainResult(
            history=metrics.history,
            best_epoch=metrics.best_epoch,
            best_ap50=metrics.best_ap50,
            best_weights=metrics.best_weights,
        )

    def resume_state(self, paths, fold, epochs, fingerprint):
        """Training state of an interrupted fold, or None to start the fold fresh."""
        state = utility.read_fold_state(paths["state"], fingerprint)

        if state is None or not paths["progress"].exists():
            # The weights to continue from are gone, so the state is unusable
            return None

        resume = {
            "initial_epoch": int(state["epoch"]),
            "history": list(state["history"]),
            "best_ap50": float(state["best_ap50"]),
            "best_epoch": int(state["best_epoch"]),
            "best_weights": (
                keras.models.load_model(paths["best"], compile=False).get_weights() if state["best_epoch"] else None
            ),
        }

        # Loading the compiled progress model restores the optimizer, so the schedule continues
        self.model = keras.models.load_model(paths["progress"])
        cnn_detector.exclude_from_weight_decay(self.model.optimizer)

        tqdm.write(
            f"Fold {fold}: resuming at epoch {resume['initial_epoch'] + 1}/{epochs} "
            f"(best val AP50 {resume['best_ap50']:.4f} at epoch {resume['best_epoch']})"
        )

        return resume

    def load(self, path=None):
        """Restore the model from a checkpoint, defaulting to the one this detector was built with."""
        path = self.weights_path if path is None else path
        if path is None:
            raise ValueError("No weights path given.")

        self.weights_path = path
        self.model = keras.models.load_model(path, compile=False)

        return self.model

    def save(self, path, fingerprint=None, epoch=0, ap50=0.0, history=None, lesion_type="mass"):
        """Write the weights; the fold's settings live in the config JSON beside them."""
        self.model.save(path)
        return path

    # inference

    def predict_raw(
        self, dataframe, score_threshold=None, desc="predict", progress=None, batch_size=None, gate_empty=None
    ):
        """Every candidate above the score threshold, after NMS, with its ground truth."""
        if self.model is None:
            raise ValueError("Call build() or load() before predicting.")

        return stacker_predict_raw(
            self.model,
            dataframe,
            self.grid,
            score_threshold,
            batch_size,
            desc=desc,
            progress=progress,
            gate_empty=gate_empty,
        )

    def predict_image(self, row, score_threshold=None):
        entry = self.grid[utility.mammogram_id(row.full_image)]

        return stacker_predict_image(self.model, row, entry, score_threshold=score_threshold)


def run_variant(trainval_df, lesion_type, architecture, tag=None, fold=0, epochs=None):
    """Train one stacker architecture on one fold, reusing its artifacts when they exist."""
    return Stacker(architecture, lesion_type, tag=tag).oof_fold(trainval_df, fold, lesion_type, epochs=epochs)
