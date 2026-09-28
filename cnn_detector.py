"""Dense objectness CNN detector: input, preprocess, model and output, shared by the notebook and the stacker."""

from __future__ import annotations
import os

# Keras reads the backend before it is imported
os.environ["KERAS_BACKEND"] = "torch"


import re
from pathlib import Path

import cv2
import numpy as np
import keras
from keras import layers, ops
from scipy.special import expit, softmax
from tqdm.auto import tqdm

import config
import utility
import Detector

# This module's row in config.FOLD_LAYOUT
MODEL_NAME = "cnn"


def set_class_loss_weight(model, weight):
    """Update the class term's weight in both the compile config and the live loss entries."""
    # Keras snapshots the loss weights on the first batch, so both copies must change
    weight = float(weight)

    compile_loss = getattr(model, "_compile_loss", None)
    if compile_loss is None:
        raise ValueError("Set the class loss weight on a compiled model.")

    weights = getattr(getattr(model, "_compile_config", None), "config", {}).get("loss_weights")
    if weights is not None:
        weights["classes"] = weight

    entries = compile_loss._flat_losses
    for index, (path, loss_fn, _, name) in enumerate(entries or []):
        if (path and path[-1] == "classes") or name == "classes":
            entries[index] = (path, loss_fn, weight, name)


def exclude_from_weight_decay(optimizer):
    """Keep AdamW decay off the norm scale/shift and the biases."""
    names = list(config.CNN_WEIGHT_DECAY_EXCLUDE)

    try:
        optimizer.exclude_from_weight_decay(var_names=names)
    except ValueError:
        optimizer._exclude_from_weight_decay_pattern = re.compile("|".join(names))
        optimizer._exclude_from_weight_decay_cache = {}

    return optimizer


def cnn_anchor_grid(sizes=None, ratios=None, image_size=None, feature_stride=None):
    """Anchor boxes of every size and aspect ratio centred on every grid location."""
    sizes = config.CNN_ANCHOR_SIZES if sizes is None else sizes
    ratios = config.CNN_ANCHOR_ASPECT_RATIOS if ratios is None else ratios
    image_size = config.CNN_IMAGE_SIZE if image_size is None else image_size
    feature_stride = config.CNN_FEATURE_STRIDE if feature_stride is None else feature_stride

    centers = np.arange(feature_stride / 2, image_size, feature_stride, dtype=np.float32)
    yy, xx = np.meshgrid(centers, centers, indexing="ij")

    boxes = []
    for size in sizes:
        for ratio in ratios:
            width = size * np.sqrt(ratio)
            height = size / np.sqrt(ratio)
            boxes.append(np.stack([xx - width / 2, yy - height / 2, xx + width / 2, yy + height / 2], axis=-1))

    return np.stack(boxes, axis=2)


CNN_GRID_SIZE = config.CNN_IMAGE_SIZE // config.CNN_FEATURE_STRIDE
CNN_ANCHOR_COUNT = len(config.CNN_ANCHOR_SIZES) * len(config.CNN_ANCHOR_ASPECT_RATIOS)
CNN_ANCHORS = cnn_anchor_grid()
CNN_ANCHOR_CORNERS = CNN_ANCHORS.reshape(-1, 4)
CNN_CLASS_COUNT = config.CLASS_COUNT


def cnn_augment(
    image,
    mask,
    boxes,
    class_labels,
    flip_probability=None,
    rotation_degrees=None,
    scale_range=None,
    translate_fraction=None,
    gamma_range=None,
):
    """Random flip, affine transform and intensity jitter; the mask, boxes and classes follow."""
    image, boxes, class_labels, mask = Detector.augment_sample(
        image,
        boxes,
        class_labels,
        mask=mask,
        flip_probability=config.CNN_AUGMENT_FLIP_PROBABILITY if flip_probability is None else flip_probability,
        rotation_degrees=config.CNN_AUGMENT_ROTATION_DEGREES if rotation_degrees is None else rotation_degrees,
        scale_range=config.CNN_AUGMENT_SCALE_RANGE if scale_range is None else scale_range,
        translate_fraction=(
            config.CNN_AUGMENT_TRANSLATE_FRACTION if translate_fraction is None else translate_fraction
        ),
        gamma_range=config.CNN_AUGMENT_GAMMA_RANGE if gamma_range is None else gamma_range,
    )

    return image, mask, boxes, class_labels


def cnn_prepare_image(row, augment=False):
    """Breast-masked canvas, breast mask, GT boxes and one class label per box; augment is training only."""
    image = utility.read_image(row.full_image)
    mask = utility.read_image(row.breast_mask)
    image = image * (mask > 0)
    boxes = utility.masks_to_boxes(row.mask_paths)
    class_labels = utility.pathology_labels(row.pathologies)

    scale = config.CNN_IMAGE_SIZE / max(image.shape)

    if scale < 1.0:
        width = int(round(image.shape[1] * scale))
        height = int(round(image.shape[0] * scale))
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        boxes = boxes.astype(np.float32) * scale

    if augment:
        image, mask, boxes, class_labels = cnn_augment(image, mask, boxes, class_labels)

    height, width = image.shape

    image_canvas = np.zeros((config.CNN_IMAGE_SIZE, config.CNN_IMAGE_SIZE), dtype=np.float32)
    image_canvas[:height, :width] = image

    mask_canvas = np.zeros((config.CNN_IMAGE_SIZE, config.CNN_IMAGE_SIZE), dtype=np.float32)
    mask_canvas[:height, :width] = mask

    return image_canvas, mask_canvas, boxes, class_labels


def cnn_breast_grid(mask_canvas):
    """Fraction of breast pixels per grid cell, used to gate anchors."""
    breast = (mask_canvas > 0).astype(np.float32)
    return cv2.resize(breast, (CNN_GRID_SIZE, CNN_GRID_SIZE), interpolation=cv2.INTER_AREA)


def cnn_anchor_deltas(matched):
    """Box regression targets taking each anchor to its matched GT box."""
    width = CNN_ANCHOR_CORNERS[:, 2] - CNN_ANCHOR_CORNERS[:, 0]
    height = CNN_ANCHOR_CORNERS[:, 3] - CNN_ANCHOR_CORNERS[:, 1]
    center_x = CNN_ANCHOR_CORNERS[:, 0] + 0.5 * width
    center_y = CNN_ANCHOR_CORNERS[:, 1] + 0.5 * height

    target_width = matched[:, 2] - matched[:, 0]
    target_height = matched[:, 3] - matched[:, 1]
    target_x = 0.5 * (matched[:, 0] + matched[:, 2])
    target_y = 0.5 * (matched[:, 1] + matched[:, 3])

    deltas = np.stack(
        [
            (target_x - center_x) / width,
            (target_y - center_y) / height,
            np.log(np.maximum(target_width, 1e-6) / width),
            np.log(np.maximum(target_height, 1e-6) / height),
        ],
        axis=-1,
    )

    return deltas.reshape(CNN_ANCHORS.shape).astype(np.float32)


def cnn_dense_targets(gt_boxes, class_labels, breast):
    """Objectness labels (1/0/-1), box deltas and class targets per anchor."""
    overlaps = None

    class_labels = np.asarray(class_labels, dtype=np.int64).reshape(-1)

    if len(gt_boxes) == 0:
        iou = np.zeros(len(CNN_ANCHOR_CORNERS), dtype=np.float32)
        matched = np.zeros((len(CNN_ANCHOR_CORNERS), 4), dtype=np.float32)
        matched_class = np.zeros(len(CNN_ANCHOR_CORNERS), dtype=np.int64)
    else:
        overlaps = utility.iou_many_vs_many(CNN_ANCHOR_CORNERS, gt_boxes)
        best = overlaps.argmax(axis=1)
        iou = overlaps.max(axis=1)
        matched = gt_boxes[best]
        matched_class = class_labels[best]

    labels = np.zeros(len(iou), dtype=np.float32)
    labels[(iou > config.CNN_ANCHOR_IOU_NEGATIVE) & (iou < config.CNN_ANCHOR_IOU_POSITIVE)] = -1.0

    labels = labels.reshape(CNN_ANCHORS.shape[:3])
    labels[breast < config.CNN_MASK_MIN_FRACTION] = -1.0

    # Positives last so they override the ignore band and the breast gate
    labels[(iou >= config.CNN_ANCHOR_IOU_POSITIVE).reshape(CNN_ANCHORS.shape[:3])] = 1.0

    if overlaps is not None:
        labels.reshape(-1)[overlaps.argmax(axis=0)] = 1.0

    positive = labels.reshape(-1) == 1.0
    classes = np.full(len(iou), -1.0, dtype=np.float32)
    classes[positive] = matched_class[positive]

    return labels, cnn_anchor_deltas(matched), classes.reshape(CNN_ANCHORS.shape[:3])


def cnn_sample_targets(targets):
    """Keep every positive and a random sample of negatives; the rest is ignored."""
    flat = targets.reshape(-1).copy()

    positive = flat == 1
    negative = np.nonzero(flat == 0)[0]

    n_pos = int(positive.sum())
    n_neg = (
        min(config.CNN_MAX_NEGATIVES, n_pos * config.CNN_NEGATIVE_RATIO)
        if n_pos
        else min(config.CNN_MAX_NEGATIVES, len(negative))
    )

    if n_neg < len(negative):
        negative = np.random.choice(negative, size=n_neg, replace=False)

    keep = np.zeros(len(flat), dtype=bool)
    keep[positive] = True
    keep[negative] = True
    flat[~keep] = -1.0

    return flat.reshape(targets.shape)


def cnn_apply_deltas(deltas):
    """Anchor grid offset by the predicted box deltas, clipped to the canvas."""
    width = CNN_ANCHORS[..., 2] - CNN_ANCHORS[..., 0]
    height = CNN_ANCHORS[..., 3] - CNN_ANCHORS[..., 1]
    center_x = CNN_ANCHORS[..., 0] + 0.5 * width
    center_y = CNN_ANCHORS[..., 1] + 0.5 * height

    center_x = center_x + deltas[..., 0] * width
    center_y = center_y + deltas[..., 1] * height
    width = np.exp(np.clip(deltas[..., 2], -4.0, 4.0)) * width
    height = np.exp(np.clip(deltas[..., 3], -4.0, 4.0)) * height

    boxes = np.stack(
        [
            center_x - 0.5 * width,
            center_y - 0.5 * height,
            center_x + 0.5 * width,
            center_y + 0.5 * height,
        ],
        axis=-1,
    )

    return np.clip(boxes, 0.0, config.CNN_IMAGE_SIZE)


def cnn_decode(score_map, box_map, class_map, breast, score_threshold=None):
    """Candidates from the score, delta and class maps; anchors outside the breast are dropped."""
    score_threshold = config.CNN_SCORE_THRESHOLD if score_threshold is None else score_threshold

    probs = expit(np.asarray(score_map, dtype=np.float32))
    probs = np.where((breast >= config.CNN_MASK_MIN_FRACTION)[..., None], probs, 0.0)

    class_probs = softmax(np.asarray(class_map, dtype=np.float32), axis=-1).reshape(-1, CNN_CLASS_COUNT)

    boxes = cnn_apply_deltas(box_map.astype(np.float32)).reshape(-1, 4)
    scores = probs.reshape(-1)

    keep = scores >= score_threshold

    labels = class_probs[keep].argmax(axis=1).astype(np.int64)
    class_scores = class_probs[keep].max(axis=1).astype(np.float32)

    return boxes[keep], scores[keep], labels, class_scores


def cnn_nms(boxes, scores, iou_threshold=None, containment_threshold=None):
    """Greedy NMS. `containment_threshold` adds nested-box suppression."""
    iou_threshold = config.CNN_NMS_THRESHOLD if iou_threshold is None else iou_threshold
    containment_threshold = config.CNN_NMS_CONTAINMENT if containment_threshold is None else containment_threshold

    order = np.argsort(-scores)
    keep = []

    while len(order):
        best = order[0]
        keep.append(best)

        if len(order) == 1:
            break

        remaining = order[1:]

        suppress = utility.iou_one_vs_many(boxes[best], boxes[remaining]) >= iou_threshold

        if containment_threshold is not None:
            containment = utility.containment_one_vs_many(boxes[best], boxes[remaining])
            suppress |= containment >= containment_threshold

        order = remaining[~suppress]

    return np.asarray(keep, dtype=np.int64)


def cnn_predict_raw(model, dataframe, score_threshold=None, batch_size=None, desc="predict", progress=None):
    """Candidates and GT boxes per image across a dataframe, reusing a passed progress bar."""
    score_threshold = config.CNN_SCORE_THRESHOLD if score_threshold is None else score_threshold
    batch_size = config.CNN_BATCH_SIZE if batch_size is None else batch_size

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
            prepared = [cnn_prepare_image(row) for _, row in dataframe.iloc[start : start + batch_size].iterrows()]

            images = np.stack([image for image, _, _, _ in prepared])[..., None].astype(np.float32)
            outputs = model.predict_on_batch(images)

            for (_, mask, gt_boxes, gt_labels), score_map, box_map, class_map in zip(
                prepared, outputs["objectness"], outputs["box"], outputs["classes"]
            ):
                boxes, scores, labels, class_scores = cnn_decode(
                    score_map, box_map, class_map, cnn_breast_grid(mask), score_threshold
                )
                raw.append(
                    {
                        # Square canvas, so the frame includes the zero padding
                        "image_shape": (config.CNN_IMAGE_SIZE, config.CNN_IMAGE_SIZE),
                        "boxes": boxes,
                        "scores": scores,
                        "labels": labels,
                        "class_scores": class_scores,
                        "gt_boxes": gt_boxes,
                        "gt_labels": gt_labels,
                    }
                )

            progress.update(1)
    finally:
        if local_bar:
            progress.close()

    return raw


def cnn_predict_image(model, row, score_threshold=None):
    """Full CNN pipeline for one mammogram: image in, final boxes, scores and classes out."""
    score_threshold = config.CNN_SCORE_THRESHOLD if score_threshold is None else score_threshold

    image, mask, gt_boxes, class_labels = cnn_prepare_image(row)

    outputs = model.predict_on_batch(image[None, :, :, None].astype(np.float32))
    boxes, scores, labels, class_scores = cnn_decode(
        outputs["objectness"][0], outputs["box"][0], outputs["classes"][0], cnn_breast_grid(mask), score_threshold
    )

    keep = scores >= config.CNN_INFERENCE_SCORE_THRESHOLD
    boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

    if len(boxes):
        keep = cnn_nms(boxes, scores)
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

    return image, gt_boxes, class_labels, boxes, scores, labels


def cnn_attention_map(model, row):
    """Objectness probability and class distribution per anchor, plus the breast gate.

    This is what the dense head actually produces, before any decode or NMS: one probability
    per anchor on the 96 x 96 x 4 grid, so the model's own view of a mammogram can be drawn.
    """
    image, mask, gt_boxes, class_labels = cnn_prepare_image(row)

    outputs = model.predict_on_batch(image[None, :, :, None].astype(np.float32))

    objectness = expit(np.asarray(outputs["objectness"][0], np.float32))
    classes = softmax(np.asarray(outputs["classes"][0], np.float32), axis=-1)

    return image, gt_boxes, class_labels, objectness, classes, cnn_breast_grid(mask)


def l2_regularizer():
    """Kernel penalty for the trunk and fork convs, or None when it is switched off."""
    return keras.regularizers.L2(config.CNN_KERNEL_L2) if config.CNN_KERNEL_L2 > 0 else None


def normalization(name=None):
    """BatchNorm, or GroupNorm when CNN_NORM selects it."""
    if config.CNN_NORM == "group":
        return layers.GroupNormalization(groups=config.CNN_GROUP_NORM_GROUPS, name=name)
    return layers.BatchNormalization(name=name)


def conv_block(x, filters, stride=1, dilation=1):
    x = layers.Conv2D(
        filters,
        3,
        strides=stride,
        padding="same",
        use_bias=False,
        dilation_rate=dilation,
        kernel_regularizer=l2_regularizer(),
    )(x)
    x = normalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(
        filters, 3, padding="same", use_bias=False, dilation_rate=dilation, kernel_regularizer=l2_regularizer()
    )(x)
    x = normalization()(x)
    return layers.ReLU()(x)


@keras.saving.register_keras_serializable(package="cnn")
class StopGradient(keras.layers.Layer):
    """Identity forward, zero gradient backward; keeps the class loss off the trunk."""

    def call(self, inputs):
        return ops.stop_gradient(inputs)


def build_model():
    """Objectness, box delta and class heads per anchor, fed by dilated stride-8 features."""
    inputs = keras.Input((config.CNN_IMAGE_SIZE, config.CNN_IMAGE_SIZE, 1), name="image")

    x = conv_block(inputs, 32)  # stride 1
    x = conv_block(x, 32, stride=2)  # stride 2
    lateral = conv_block(x, 64, stride=2)  # stride 4, the class head's lateral input
    x = conv_block(lateral, 128, stride=2)  # stride 8
    x = conv_block(x, 128)

    features = layers.Conv2D(
        128, 3, padding="same", dilation_rate=2, use_bias=False, kernel_regularizer=l2_regularizer()
    )(x)
    features = normalization()(features)
    features = layers.ReLU()(features)

    objectness = layers.Conv2D(CNN_ANCHOR_COUNT, 1, name="objectness")(features)

    box = layers.Conv2D(CNN_ANCHOR_COUNT * 4, 1, name="box_deltas")(features)
    box = layers.Reshape((CNN_GRID_SIZE, CNN_GRID_SIZE, CNN_ANCHOR_COUNT, 4), name="box")(box)

    # The gate keeps the class loss out of the trunk, so only the fork and class_logits train
    class_features = StopGradient(name="class_features")(features)
    class_features = layers.Concatenate(name="class_lateral")([class_features, class_lateral_features(lateral)])

    classes = layers.Conv2D(CNN_ANCHOR_COUNT * CNN_CLASS_COUNT, 1, name="class_logits")(class_features)
    classes = layers.Reshape((CNN_GRID_SIZE, CNN_GRID_SIZE, CNN_ANCHOR_COUNT, CNN_CLASS_COUNT), name="classes")(classes)

    return keras.Model(inputs, {"objectness": objectness, "box": box, "classes": classes}, name="dense_objectness")


def class_lateral_features(lateral):
    """Stride-4 lateral input for the class head, gated then pooled onto the stride-8 grid."""
    lateral = StopGradient(name="class_lateral_stop")(lateral)
    lateral = class_fork(lateral)

    return layers.AveragePooling2D(pool_size=2, strides=2, padding="same", name="class_lateral_pool")(lateral)


def class_fork(x):
    """Convolution the class loss is allowed to train, private to the class branch."""
    x = layers.Conv2D(
        config.CNN_CLASS_FORK_FILTERS,
        3,
        padding="same",
        use_bias=False,
        name="class_fork_conv",
        kernel_regularizer=l2_regularizer(),
    )(x)
    x = normalization(name="class_fork_norm")(x)

    return layers.ReLU(name="class_fork_relu")(x)


@keras.saving.register_keras_serializable(package="cnn")
class MaskedBinaryCrossentropy(keras.losses.Loss):
    """BCE over the sampled anchors; anchors labelled -1 are ignored."""

    def __init__(self, **kwargs):
        kwargs.setdefault("reduction", "sum")
        super().__init__(**kwargs)

    def call(self, y_true, y_pred):
        y_true = ops.cast(y_true, "float32")
        logits = ops.cast(y_pred, "float32")

        valid = ops.cast(ops.not_equal(y_true, -1.0), "float32")
        targets = ops.maximum(y_true, 0.0)
        probs = ops.sigmoid(logits)

        loss = -(
            targets * ops.log(ops.clip(probs, 1e-7, 1.0)) + (1.0 - targets) * ops.log(ops.clip(1.0 - probs, 1e-7, 1.0))
        )

        return ops.sum(loss * valid) / ops.maximum(ops.sum(valid), 1.0)


@keras.saving.register_keras_serializable(package="cnn")
class MaskedHuber(keras.losses.Loss):
    """Smooth L1 over box deltas; channel 0 of y_true is the objectness label and only positives count."""

    def __init__(self, delta=1.0, **kwargs):
        kwargs.setdefault("reduction", "sum")
        super().__init__(**kwargs)
        self.delta = delta

    def call(self, y_true, y_pred):
        y_true = ops.cast(y_true, "float32")
        predicted = ops.cast(y_pred, "float32")

        mask = ops.cast(ops.equal(y_true[..., 0:1], 1.0), "float32")
        deltas = y_true[..., 1:]

        error = ops.abs(predicted - deltas)
        huber = ops.where(error < self.delta, 0.5 * ops.square(error), self.delta * (error - 0.5 * self.delta))

        return ops.sum(huber * mask) / ops.maximum(ops.sum(mask) * 4.0, 1.0)

    def get_config(self):
        config_ = super().get_config()
        config_.update({"delta": self.delta})
        return config_


@keras.saving.register_keras_serializable(package="cnn")
class MaskedClassCrossentropy(keras.losses.Loss):
    """Cross-entropy over the class of positive anchors; -1 marks everything else."""

    def __init__(self, label_smoothing=None, **kwargs):
        kwargs.setdefault("reduction", "sum")
        super().__init__(**kwargs)
        self.label_smoothing = float(config.CNN_LABEL_SMOOTHING if label_smoothing is None else label_smoothing)

    def call(self, y_true, y_pred):
        y_true = ops.cast(y_true, "float32")
        logits = ops.cast(y_pred, "float32")

        valid = ops.cast(ops.not_equal(y_true[..., 0], -1.0), "float32")
        indices = ops.cast(ops.maximum(y_true[..., 0], 0.0), "int32")
        weight = y_true[..., 1]

        log_probs = ops.log_softmax(logits, axis=-1)
        picked = ops.take_along_axis(log_probs, indices[..., None], axis=-1)[..., 0]

        class_count = logits.shape[-1] if logits.shape[-1] is not None else config.CLASS_COUNT

        if self.label_smoothing > 0 and class_count > 1:
            # The smoothing mass is spread over the other classes, leaving the target at 1 - smoothing
            others = ops.sum(log_probs, axis=-1) - picked
            loss = -((1.0 - self.label_smoothing) * picked + (self.label_smoothing / (class_count - 1)) * others)
        else:
            loss = -picked

        return ops.sum(loss * valid) / ops.maximum(ops.sum(weight), 1.0)

    def get_config(self):
        config_ = super().get_config()
        config_.update({"label_smoothing": self.label_smoothing})
        return config_


@keras.saving.register_keras_serializable(package="cnn")
class WarmupCosineDecay(keras.optimizers.schedules.LearningRateSchedule):
    """Linear warmup then cosine decay from CNN_LR down to CNN_MIN_LR."""

    def __init__(self, peak_learning_rate, warmup_steps, total_steps, min_learning_rate):
        super().__init__()
        self.peak_learning_rate = float(peak_learning_rate)
        self.warmup_steps = float(warmup_steps)
        self.total_steps = float(total_steps)
        self.min_learning_rate = float(min_learning_rate)

    def __call__(self, step):
        step = ops.cast(step, "float32")

        warmup = self.peak_learning_rate * (step / max(self.warmup_steps, 1.0))
        progress = ops.clip((step - self.warmup_steps) / max(self.total_steps - self.warmup_steps, 1.0), 0.0, 1.0)
        cosine = self.min_learning_rate + 0.5 * (self.peak_learning_rate - self.min_learning_rate) * (
            1.0 + ops.cos(np.pi * progress)
        )

        return ops.where(step < self.warmup_steps, warmup, cosine)

    def get_config(self):
        return {
            "peak_learning_rate": self.peak_learning_rate,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
            "min_learning_rate": self.min_learning_rate,
        }


class DetectionDataset(keras.utils.PyDataset):
    """Padded images with dense anchor targets; the box and class targets carry their own masks."""

    def __init__(self, dataframe, batch_size, augment=True, **kwargs):
        super().__init__(**kwargs)
        self.dataframe = dataframe.reset_index(drop=True)
        self.batch_size = batch_size
        self.augment = augment

    def __len__(self):
        return int(np.ceil(len(self.dataframe) / self.batch_size))

    def __getitem__(self, index):
        rows = self.dataframe.iloc[index * self.batch_size : (index + 1) * self.batch_size]

        images = np.empty((len(rows), config.CNN_IMAGE_SIZE, config.CNN_IMAGE_SIZE, 1), dtype=np.float32)
        objectness = np.empty((len(rows), CNN_GRID_SIZE, CNN_GRID_SIZE, CNN_ANCHOR_COUNT), dtype=np.float32)
        boxes = np.empty((len(rows), CNN_GRID_SIZE, CNN_GRID_SIZE, CNN_ANCHOR_COUNT, 5), dtype=np.float32)
        classes = np.empty((len(rows), CNN_GRID_SIZE, CNN_GRID_SIZE, CNN_ANCHOR_COUNT, 2), dtype=np.float32)

        for i, (_, row) in enumerate(rows.iterrows()):
            image, mask, gt_boxes, class_labels = cnn_prepare_image(row, augment=self.augment)

            labels, deltas, class_targets = cnn_dense_targets(gt_boxes, class_labels, cnn_breast_grid(mask))
            labels = cnn_sample_targets(labels)

            # Class divisor, kept on the same population the objectness loss normalises over
            divisor = labels != -1.0

            objectness[i] = labels
            boxes[i] = np.concatenate([labels[..., None], deltas], axis=-1)
            classes[i, ..., 0] = class_targets
            classes[i, ..., 1] = divisor
            images[i, :, :, 0] = image

        return images, {"objectness": objectness, "box": boxes, "classes": classes}


def validate(model, dataframe, score_threshold=config.CNN_SCORE_THRESHOLD, desc="val", progress=None):
    """NMS plus the shared metrics for the per-epoch validation row."""
    raw = cnn_predict_raw(model, dataframe, score_threshold, desc=desc, progress=progress)

    for entry in raw:
        keep = cnn_nms(entry["boxes"], entry["scores"])

        entry["boxes"] = entry["boxes"][keep]
        entry["scores"] = entry["scores"][keep]
        entry["labels"] = entry["labels"][keep]

    return Detector.detection_metrics(raw, config.CNN_IOU_THRESHOLD, config.CNN_INFERENCE_SCORE_THRESHOLD)


EPOCH_TABLE_COLUMNS = ("Epoch", "Loss", "obj", "box", "cls", "AP50", "mAP50", "P", "R", "F1", "Det")
EPOCH_TABLE_KEYS = (
    "epoch",
    "train_loss",
    "objectness_loss",
    "box_loss",
    "classes_loss",
    "ap50",
    "ap50_macro",
    "precision",
    "recall",
    "f1",
    "det_per_image",
)
EPOCH_TABLE_FORMATS = ("{:.0f}",) + ("{:.4f}",) * 9 + ("{:.1f}",)


class DetectionMetrics(keras.callbacks.Callback):
    """Epoch rows, the best raw weights and whichever progress bars the caller passed in."""

    def __init__(
        self,
        dataframe,
        outer,
        inner,
        train_steps,
        n_epochs,
        tag="",
        history=None,
        best_ap50=0.0,
        best_epoch=0,
        best_weights=None,
        row_suffix=None,
    ):
        super().__init__()
        self.dataframe = dataframe
        self.outer = outer
        self.inner = inner
        self.train_steps = train_steps
        self.n_epochs = n_epochs
        self.tag = tag
        self.table = utility.EpochTable(EPOCH_TABLE_COLUMNS, EPOCH_TABLE_KEYS, EPOCH_TABLE_FORMATS, n_epochs)
        self.fit = utility.FitHistory(history, best_epoch, best_ap50, best_weights)
        self.print_header = True
        self.row_suffix = row_suffix

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

        metrics = validate(
            self.model,
            self.dataframe,
            desc=f"{self.tag}Epoch {epoch + 1}/{self.n_epochs} [Val]",
            progress=self.inner,
        )

        class_columns = {f"ap50_{name}": value for name, value in zip(config.CLASS_NAMES, metrics["ap50_per_class"])}

        # Keras logs the weighted term of every output head
        loss_terms = {f"{head}_loss": logs.get(f"{head}_loss", np.nan) for head in ("objectness", "box", "classes")}

        entry = {
            "epoch": epoch + 1,
            "train_loss": logs.get("loss", np.nan),
            **loss_terms,
            **{key: value for key, value in metrics.items() if key != "ap50_per_class"},
            **class_columns,
        }

        improved = self.fit.add(entry, metrics["ap50"], [weight.copy() for weight in self.model.get_weights()])

        suffix = "" if self.row_suffix is None else self.row_suffix(entry, improved)

        tqdm.write(self.table.row(entry) + (" | Saved best model" if improved else "") + suffix)

        self.outer.update(1)


class ClassLossRamp(keras.callbacks.Callback):
    """Ramp the class term from CNN_CLASS_LOSS_WEIGHT_START to CNN_CLASS_LOSS_WEIGHT over the first epochs."""

    def __init__(self, start=None, end=None, ramp_epochs=None, total_epochs=None):
        super().__init__()
        self.start = config.CNN_CLASS_LOSS_WEIGHT_START if start is None else float(start)
        self.end = config.CNN_CLASS_LOSS_WEIGHT if end is None else float(end)
        self.ramp_epochs = config.CNN_CLASS_LOSS_RAMP_EPOCHS if ramp_epochs is None else int(ramp_epochs)
        self.total_epochs = config.CNN_NUM_EPOCHS if total_epochs is None else int(total_epochs)

    def weight_at(self, epoch):
        last = max(self.total_epochs - 1, 1)
        position = min(max(int(epoch), 1), last)
        progress = (position - 1) / last

        if self.ramp_epochs <= 0:
            return self.end

        ramp = min(progress / (self.ramp_epochs / self.total_epochs), 1.0)

        return self.start + (self.end - self.start) * ramp

    def on_epoch_begin(self, epoch, logs=None):
        set_class_loss_weight(self.model, self.weight_at(epoch + 1))


class FoldCheckpoint(keras.callbacks.Callback):
    """Per-epoch training state so an interrupted fold resumes; removed once the fold completes."""

    def __init__(self, tracker, paths, fingerprint):
        super().__init__()
        self.tracker = tracker
        self.paths = paths
        self.fingerprint = fingerprint

    def on_epoch_end(self, epoch, logs=None):
        if self.tracker.best_epoch == epoch + 1:
            utility.write_atomically(lambda path: self.model.save(path, include_optimizer=False), self.paths["best"])

        utility.write_atomically(lambda path: self.model.save(path), self.paths["progress"])

        utility.write_fold_state(
            self.paths["state"],
            self.fingerprint,
            epoch=epoch + 1,
            best_epoch=self.tracker.best_epoch,
            best_ap50=self.tracker.best_ap50,
            history=self.tracker.history,
        )


def training_config(lesion_type, best_epoch=None, best_val_ap50=None):
    """Pipeline settings a checkpoint depends on, so any consumer can reproduce it."""
    return {
        "lesion_type": lesion_type,
        "image_size": config.CNN_IMAGE_SIZE,
        "feature_stride": config.CNN_FEATURE_STRIDE,
        "anchor_sizes": list(config.CNN_ANCHOR_SIZES),
        "anchor_aspect_ratios": list(config.CNN_ANCHOR_ASPECT_RATIOS),
        "anchor_iou_positive": config.CNN_ANCHOR_IOU_POSITIVE,
        "anchor_iou_negative": config.CNN_ANCHOR_IOU_NEGATIVE,
        "mask_min_fraction": config.CNN_MASK_MIN_FRACTION,
        "augment_flip_probability": config.CNN_AUGMENT_FLIP_PROBABILITY,
        "augment_rotation_degrees": config.CNN_AUGMENT_ROTATION_DEGREES,
        "augment_scale_range": list(config.CNN_AUGMENT_SCALE_RANGE),
        "augment_translate_fraction": config.CNN_AUGMENT_TRANSLATE_FRACTION,
        "augment_gamma_range": list(config.CNN_AUGMENT_GAMMA_RANGE),
        "class_names": list(config.CLASS_NAMES),
        "class_fork_filters": config.CNN_CLASS_FORK_FILTERS,
        "kernel_l2": config.CNN_KERNEL_L2,
        "norm": config.CNN_NORM,
        "group_norm_groups": config.CNN_GROUP_NORM_GROUPS,
        "label_smoothing": config.CNN_LABEL_SMOOTHING,
        "weight_decay_exclude": list(config.CNN_WEIGHT_DECAY_EXCLUDE),
        "negative_ratio": config.CNN_NEGATIVE_RATIO,
        "max_negatives": config.CNN_MAX_NEGATIVES,
        "box_loss_weight": config.CNN_BOX_LOSS_WEIGHT,
        "class_loss_weight": config.CNN_CLASS_LOSS_WEIGHT,
        "class_loss_weight_start": config.CNN_CLASS_LOSS_WEIGHT_START,
        "class_loss_ramp_epochs": config.CNN_CLASS_LOSS_RAMP_EPOCHS,
        "batch_size": config.CNN_BATCH_SIZE,
        "epochs": config.CNN_NUM_EPOCHS,
        "score_threshold": config.CNN_SCORE_THRESHOLD,
        "inference_score_threshold": config.CNN_INFERENCE_SCORE_THRESHOLD,
        "iou_threshold": config.CNN_IOU_THRESHOLD,
        "nms_threshold": config.CNN_NMS_THRESHOLD,
        "nms_containment": config.CNN_NMS_CONTAINMENT,
        "stacker_score_threshold": config.CNN_STACKER_SCORE_THRESHOLD,
        "stacker_top_k": config.CNN_STACKER_TOP_K,
        "best_epoch": best_epoch,
        "best_val_ap50": best_val_ap50,
    }


class CNN(Detector.Detector):
    """Dense objectness CNN detector with a benign/malignant class head and fold-aware training."""

    model_name = MODEL_NAME
    default_epochs = config.CNN_NUM_EPOCHS
    score_threshold = config.CNN_SCORE_THRESHOLD
    inference_score_threshold = config.CNN_INFERENCE_SCORE_THRESHOLD
    iou_threshold = config.CNN_IOU_THRESHOLD
    stacker_score_threshold = config.CNN_STACKER_SCORE_THRESHOLD
    stacker_top_k = config.CNN_STACKER_TOP_K
    stacker_collect_threshold = config.CNN_STACKER_COLLECT_THRESHOLD
    resume_artifact_keys = ("progress", "state", "best")
    epoch_table_columns = EPOCH_TABLE_COLUMNS
    epoch_table_keys = EPOCH_TABLE_KEYS
    epoch_table_formats = EPOCH_TABLE_FORMATS

    def build(self):
        self.model = build_model()
        return self.model

    def prepare_model(self):
        """Build the fresh model whose graph the fold's fingerprint is taken over."""
        self.build()

    def model_signature(self, lesion_type):
        """Input shapes, output names and layer configs, stable across builds."""
        return utility.keras_model_signature(self.model)

    def training_config(self, lesion_type, best_epoch=None, best_val_ap50=None):
        """Pipeline settings a checkpoint depends on, so any consumer can reproduce it."""
        return training_config(lesion_type, best_epoch, best_val_ap50)

    def compile(self, steps_per_epoch):
        """Compile with the warmup + cosine schedule sized for the full run."""
        schedule = WarmupCosineDecay(
            peak_learning_rate=config.CNN_LR,
            warmup_steps=config.CNN_WARMUP_EPOCHS * steps_per_epoch,
            total_steps=config.CNN_NUM_EPOCHS * steps_per_epoch,
            min_learning_rate=config.CNN_MIN_LR,
        )

        optimizer = exclude_from_weight_decay(
            keras.optimizers.AdamW(learning_rate=schedule, weight_decay=config.CNN_WEIGHT_DECAY)
        )

        self.model.compile(
            optimizer=optimizer,
            loss={
                "objectness": MaskedBinaryCrossentropy(),
                "box": MaskedHuber(),
                "classes": MaskedClassCrossentropy(),
            },
            loss_weights={
                "objectness": 1.0,
                "box": config.CNN_BOX_LOSS_WEIGHT,
                "classes": config.CNN_CLASS_LOSS_WEIGHT,
            },
        )
        return schedule

    def dataset(self, dataframe, augment=True):
        return DetectionDataset(dataframe, config.CNN_BATCH_SIZE, augment=augment)

    def prepare(self, row, augment=False):
        """Breast-masked canvas, breast mask and GT boxes; augment is training only."""
        return cnn_prepare_image(row, augment=augment)

    def breast_grid(self, mask):
        return cnn_breast_grid(mask)

    def dense_targets(self, gt_boxes, class_labels, breast):
        return cnn_dense_targets(gt_boxes, class_labels, breast)

    def predict_raw(self, dataframe, score_threshold=None, batch_size=None, desc="predict", progress=None):
        """Every candidate above the score threshold, after NMS, with its ground truth."""
        if self.model is None:
            raise ValueError("Call build() or load() before predicting.")

        raw = cnn_predict_raw(
            self.model, dataframe, score_threshold=score_threshold, batch_size=batch_size, desc=desc, progress=progress
        )

        for entry in raw:
            keep = cnn_nms(entry["boxes"], entry["scores"])

            entry["boxes"] = entry["boxes"][keep]
            entry["scores"] = entry["scores"][keep]
            entry["labels"] = entry["labels"][keep]
            entry["class_scores"] = entry["class_scores"][keep]

        return raw

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
        """Fit the compiled model, validating the best raw weights every epoch."""
        epochs = config.CNN_NUM_EPOCHS if epochs is None else epochs

        dataset = self.dataset(train_df)

        if resume is None:
            # A fresh fold compiles its own warmup and cosine schedule over the fold's step count
            self.compile(steps_per_epoch=len(dataset))

        resume = {} if resume is None else resume
        initial_epoch = int(resume.get("initial_epoch", 0))

        own_bars = outer is None
        if own_bars:
            outer = tqdm(total=max(epochs - initial_epoch, 0), position=0, desc="Epochs")
            inner = tqdm(total=0, position=1, desc="Initializing...")

        metrics = DetectionMetrics(
            val_df,
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

        callbacks = [metrics, ClassLossRamp(total_epochs=epochs)]
        if checkpoint_paths is not None:
            callbacks.append(FoldCheckpoint(metrics, checkpoint_paths, fingerprint))

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

        # The progress model carries the optimizer, so the LR schedule continues
        resume = {
            "initial_epoch": int(state["epoch"]),
            "history": list(state["history"]),
            "best_ap50": float(state["best_ap50"]),
            "best_epoch": int(state["best_epoch"]),
            "best_weights": (
                keras.models.load_model(paths["best"], compile=False).get_weights() if state["best_epoch"] else None
            ),
        }

        self.model = keras.models.load_model(paths["progress"])
        exclude_from_weight_decay(self.model.optimizer)

        tqdm.write(
            f"Fold {fold}: resuming at epoch {resume['initial_epoch'] + 1}/{epochs} "
            f"(best val AP50 {resume['best_ap50']:.4f} at epoch {resume['best_epoch']})"
        )

        return resume

    def load(self, path=None):
        """Restore the model from a checkpoint, defaulting to the one this detector was built with."""
        path = Path(path) if path is not None else self.weights_path
        if path is None:
            raise ValueError("No weights path given.")

        self.weights_path = path
        self.model = keras.models.load_model(path, compile=False)
        return self.model

    def save(self, path, fingerprint=None, epoch=0, ap50=0.0, history=None, lesion_type="mass"):
        """Write the weights; the fold's settings live in the config JSON beside them."""
        self.model.save(path)
        return Path(path)

    def predict_image(self, row, score_threshold=None):
        return cnn_predict_image(self.model, row, score_threshold=score_threshold)

    def attention_map(self, row):
        """Objectness probability and class distribution per anchor, plus the breast gate."""
        if self.model is None:
            raise ValueError("Call build() or load() before predicting.")

        return cnn_attention_map(self.model, row)
