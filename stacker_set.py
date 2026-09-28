"""Set stacker: fuses the three detectors' ranked candidates as a set of tokens."""

from pathlib import Path
import json

import numpy as np
import pandas as pd

import config
import utility
import Detector
import cnn_detector
import stacker

import keras
from keras import layers, ops

MODEL_NAME = "stacker_set"
BASE_MODELS = stacker.BASE_MODELS
BASE_LABELS = stacker.BASE_LABELS

FRAME = float(config.CNN_IMAGE_SIZE)
BASE_TOKEN_FEATURES = 4 + 1 + 2 + len(BASE_MODELS)  # box, score, class evidence, model id
AGREEMENT_FEATURES = 2  # best foreign IoU, and the foreign score that reaches it
TOKEN_FEATURES = BASE_TOKEN_FEATURES
SCORE_INDEX = 4


def token_features(agreement=False):
    """Width of a token: the base block, plus the two agreement channels when enabled."""
    return BASE_TOKEN_FEATURES + (AGREEMENT_FEATURES if agreement else 0)


# Tokens


def candidate_frames(lesion_type, folds, top_k=None, lookup=None):
    """{image_id: {model: frame}}, every fold's ranked candidates cut at top_k."""
    top_k = config.SET_K if top_k is None else int(top_k)

    if lookup is not None:
        return {
            image_id: {model: frame[frame["rank"] <= top_k] for model, frame in found.items()}
            for image_id, found in lookup.items()
        }

    lookup = {}

    for model in BASE_MODELS:
        frames = []

        for fold in folds:
            path = config.fold_paths(model, lesion_type, fold)["candidates"]

            if not path.exists():
                raise FileNotFoundError(f"{model} fold {fold} candidates are missing: {path}")

            frames.append(pd.read_csv(path))

        frame = pd.concat(frames, ignore_index=True)
        frame = frame[frame["rank"] <= top_k]

        for image_id, group in frame.groupby("image_id", sort=False):
            lookup.setdefault(image_id, {})[model] = group.reset_index(drop=True)

    return lookup


def token_block(frame, model):
    """One detector's candidates on one mammogram as token features and boxes."""
    position = BASE_MODELS.index(model)

    width = (frame["x2"] - frame["x1"]).to_numpy(np.float32)
    height = (frame["y2"] - frame["y1"]).to_numpy(np.float32)
    centre_x = 0.5 * (frame["x1"] + frame["x2"]).to_numpy(np.float32)
    centre_y = 0.5 * (frame["y1"] + frame["y2"]).to_numpy(np.float32)

    labels = frame["class_label"].to_numpy(np.int64)
    class_scores = frame["class_score"].to_numpy(np.float32)

    block = np.zeros((len(frame), BASE_TOKEN_FEATURES), np.float32)
    block[:, 0] = centre_x / FRAME
    block[:, 1] = centre_y / FRAME
    block[:, 2] = np.log(np.maximum(width, 1.0) / FRAME)
    block[:, 3] = np.log(np.maximum(height, 1.0) / FRAME)
    block[:, 4] = frame["score"].to_numpy(np.float32)
    block[labels == 0, 5] = class_scores[labels == 0]
    block[labels == 1, 6] = class_scores[labels == 1]
    block[:, 7 + position] = 1.0

    return block, frame[["x1", "y1", "x2", "y2"]].to_numpy(np.float32)


def agreement_features(features, boxes, valid):
    """Per-token agreement with the other detectors, as the best IoU against a foreign owner."""
    owner = np.repeat(np.arange(len(BASE_MODELS)), len(boxes) // len(BASE_MODELS))
    scores = features[:, SCORE_INDEX]

    iou = utility.iou_many_vs_many(boxes, boxes)
    iou[owner[:, None] == owner[None, :]] = 0.0

    best = iou.max(axis=1, keepdims=True)
    band = (iou >= config.CNN_ANCHOR_IOU_POSITIVE) & valid[None, :]
    agree = np.where(band, scores[None, :], 0.0).max(axis=1, keepdims=True)

    return np.concatenate([best, agree], axis=1).astype(np.float32)


def box_deltas(boxes, reference):
    """Boxes as deltas from a reference box, in the form the CNN's anchors already use."""
    width = np.maximum(reference[:, 2] - reference[:, 0], 1e-6)
    height = np.maximum(reference[:, 3] - reference[:, 1], 1e-6)
    centre_x = 0.5 * (reference[:, 0] + reference[:, 2])
    centre_y = 0.5 * (reference[:, 1] + reference[:, 3])

    return np.stack(
        [
            (0.5 * (boxes[:, 0] + boxes[:, 2]) - centre_x) / width,
            (0.5 * (boxes[:, 1] + boxes[:, 3]) - centre_y) / height,
            np.log(np.maximum(boxes[:, 2] - boxes[:, 0], 1e-6) / width),
            np.log(np.maximum(boxes[:, 3] - boxes[:, 1], 1e-6) / height),
        ],
        axis=-1,
    ).astype(np.float32)


def apply_deltas(reference, deltas):
    """Inverse of `box_deltas`, so a zero delta hands back the reference box."""
    width = np.maximum(reference[:, 2] - reference[:, 0], 1e-6)
    height = np.maximum(reference[:, 3] - reference[:, 1], 1e-6)
    centre_x = 0.5 * (reference[:, 0] + reference[:, 2])
    centre_y = 0.5 * (reference[:, 1] + reference[:, 3])

    new_width = np.exp(np.clip(deltas[:, 2], -4.0, 4.0)) * width
    new_height = np.exp(np.clip(deltas[:, 3], -4.0, 4.0)) * height
    new_centre_x = centre_x + deltas[:, 0] * width
    new_centre_y = centre_y + deltas[:, 1] * height

    return np.stack(
        [
            new_centre_x - 0.5 * new_width,
            new_centre_y - 0.5 * new_height,
            new_centre_x + 0.5 * new_width,
            new_centre_y + 0.5 * new_height,
        ],
        axis=-1,
    ).astype(np.float32)


def token_targets(boxes, valid, gt_boxes, gt_labels):
    """Objectness, box and class targets of every token; padded tokens are ignored."""
    count = len(boxes)

    objectness = np.full(count, -1.0, np.float32)
    box = np.zeros((count, 5), np.float32)
    classes = np.zeros((count, 2), np.float32)
    box[:, 0] = -1.0
    classes[:, 0] = -1.0
    classes[:, 1] = valid.astype(np.float32)

    if not len(gt_boxes) or not valid.any():
        return objectness, box, classes

    gt_boxes = np.asarray(gt_boxes, np.float32).reshape(-1, 4)
    gt_labels = np.asarray(gt_labels, np.float32).reshape(-1)

    iou = utility.iou_many_vs_many(boxes, gt_boxes)
    best = iou.argmax(axis=1)
    best_iou = iou[np.arange(count), best]

    # Between the two thresholds a token is neither trained nor penalised
    negative = np.where((best_iou <= config.CNN_ANCHOR_IOU_NEGATIVE) & valid)[0]
    positive = np.where((best_iou >= config.CNN_ANCHOR_IOU_POSITIVE) & valid)[0]

    objectness[negative] = 0.0

    if len(positive):
        objectness[positive] = 1.0
        box[positive, 0] = 1.0
        box[positive, 1:] = box_deltas(gt_boxes[best[positive]], boxes[positive])
        classes[positive, 0] = gt_labels[best[positive]]
        classes[positive, 1] = 1.0

    objectness[~valid] = -1.0
    box[~valid, 0] = -1.0
    classes[~valid] = np.array([-1.0, 0.0], np.float32)

    return objectness, box, classes


def build_tokens(dataframe, lesion_type, folds, top_k=None, agreement=False, lookup=None):
    """Per-mammogram token tensors, boxes, ground truth and pad mask, keyed by image id."""
    top_k = config.SET_K if top_k is None else int(top_k)
    lookup = (
        candidate_frames(lesion_type, folds, top_k)
        if lookup is None
        else candidate_frames(lesion_type, folds, top_k, lookup)
    )
    empty = np.zeros((0, BASE_TOKEN_FEATURES), np.float32)
    tokens = {}

    for row in dataframe.itertuples():
        image_id = utility.mammogram_id(row.full_image)
        found = lookup.get(image_id, {})
        gt_boxes, gt_labels, _ = stacker.ground_truth(row)

        features, boxes, valid = [], [], []

        for model in BASE_MODELS:
            frame = found.get(model)

            if frame is None or not len(frame):
                block, model_boxes = empty, np.zeros((0, 4), np.float32)
            else:
                block, model_boxes = token_block(frame, model)

            missing = top_k - len(block)

            if missing > 0:
                block = np.vstack([block, np.zeros((missing, BASE_TOKEN_FEATURES), np.float32)])
                model_boxes = np.vstack([model_boxes, np.zeros((missing, 4), np.float32)])

            features.append(block)
            boxes.append(model_boxes)
            valid.append(np.array([True] * (top_k - missing) + [False] * missing))

        features = np.concatenate(features)
        boxes = np.concatenate(boxes)
        valid = np.concatenate(valid)

        if agreement:
            features = np.concatenate([features, agreement_features(features, boxes, valid)], axis=1)

        objectness, box_target, class_target = token_targets(boxes, valid, gt_boxes, gt_labels)

        tokens[image_id] = {
            "tokens": features,
            "mask": valid.astype(np.float32)[:, None],
            "boxes": boxes,
            "objectness": objectness,
            "box": box_target,
            "classes": class_target,
            "gt_boxes": np.asarray(gt_boxes, np.float32).reshape(-1, 4),
            "gt_labels": np.asarray(gt_labels, np.int64).reshape(-1),
        }

    return tokens


# Model


@keras.saving.register_keras_serializable(package="stacker_set")
class TokenPrior(keras.layers.Layer):
    """Adds the token's own detector score, as a logit, to its objectness logit."""

    def __init__(self, learned=False, **kwargs):
        super().__init__(**kwargs)
        self.learned = bool(learned)

    def build(self, input_shape):
        self.scale = (
            self.add_weight(name="scale", shape=(), initializer="ones", trainable=True) if self.learned else None
        )
        super().build(input_shape)

    def call(self, inputs):
        tokens, logits = inputs

        score = ops.clip(tokens[..., SCORE_INDEX : SCORE_INDEX + 1], 1e-4, 1.0 - 1e-4)
        prior = ops.log(score) - ops.log(1.0 - score)

        return logits + prior if self.scale is None else logits + self.scale * prior

    def get_config(self):
        return super().get_config() | {"learned": self.learned}


@keras.saving.register_keras_serializable(package="stacker_set")
class SetAttention(keras.layers.Layer):
    """Multi-head attention over the tokens, with padded tokens masked out of the keys."""

    def __init__(self, heads, key_dim, count, **kwargs):
        super().__init__(**kwargs)
        self.heads = int(heads)
        self.key_dim = int(key_dim)
        self.count = int(count)

    def build(self, input_shape):
        width = input_shape[0][-1]

        self.query = layers.Dense(self.heads * self.key_dim, name="query")
        self.key = layers.Dense(self.heads * self.key_dim, name="key")
        self.value = layers.Dense(self.heads * self.key_dim, name="value")
        self.projection = layers.Dense(width, name="projection")

        super().build(input_shape)

    def call(self, inputs):
        tokens, mask = inputs
        shape = (-1, self.count, self.heads, self.key_dim)

        query = ops.transpose(ops.reshape(self.query(tokens), shape), (0, 2, 1, 3))
        key = ops.transpose(ops.reshape(self.key(tokens), shape), (0, 2, 1, 3))
        value = ops.transpose(ops.reshape(self.value(tokens), shape), (0, 2, 1, 3))

        scores = ops.matmul(query, ops.transpose(key, (0, 1, 3, 2))) / np.sqrt(self.key_dim)

        # (batch, count, 1) -> (batch, 1, 1, count), broadcast across queries and heads
        padded = ops.expand_dims(1.0 - ops.transpose(mask, (0, 2, 1)), axis=2)
        scores = scores - 1e9 * padded

        attended = ops.matmul(ops.softmax(scores, axis=-1), value)
        attended = ops.reshape(ops.transpose(attended, (0, 2, 1, 3)), (-1, self.count, self.heads * self.key_dim))

        return self.projection(attended)

    def get_config(self):
        return super().get_config() | {"heads": self.heads, "key_dim": self.key_dim, "count": self.count}


def architecture_defaults():
    """The set stacker's baseline architecture, as build_stacker_set's arguments."""
    return {
        "k": config.SET_K,
        "width": config.SET_WIDTH,
        "heads": config.SET_HEADS,
        "blocks": config.SET_BLOCKS,
        "box_head": "none",
        "norm": config.STACKER_NORM,
        "agreement": False,
        "prior_scale": config.SET_PRIOR_SCALE,
        "dropout": 0.0,
    }


def build_stacker_set(
    k=None,
    width=None,
    heads=None,
    blocks=None,
    box_head=None,
    norm=None,
    agreement=False,
    prior_scale=None,
    dropout=0.0,
):
    """Attention over 3K candidate tokens with a per-token objectness, box and class head."""
    k = config.SET_K if k is None else int(k)
    width = config.SET_WIDTH if width is None else int(width)
    heads = config.SET_HEADS if heads is None else int(heads)
    blocks = config.SET_BLOCKS if blocks is None else int(blocks)
    box_head = "none" if box_head is None else str(box_head)
    norm = config.STACKER_NORM if norm is None else norm
    prior_scale = config.SET_PRIOR_SCALE if prior_scale is None else bool(prior_scale)

    count = len(BASE_MODELS) * k

    tokens = keras.Input((count, token_features(agreement)), name="tokens")
    mask = keras.Input((count, 1), name="mask")

    x = layers.Dense(width)(tokens)
    x = stacker.stacker_normalization(norm)(x)
    x = layers.ReLU()(x)

    if dropout:
        x = layers.Dropout(dropout)(x)

    for block in range(blocks):
        attended = SetAttention(heads, width // heads, count, name=f"attention_{block}")([x, mask])
        x = stacker.stacker_normalization(norm)(layers.Add()([x, attended]))

        hidden = layers.Dense(2 * width, activation="relu")(x)
        hidden = layers.Dense(width)(hidden)
        x = stacker.stacker_normalization(norm)(layers.Add()([x, hidden]))

        if dropout:
            x = layers.Dropout(dropout)(x)

    logits = layers.Dense(1, name="objectness_logits")(x)
    objectness = TokenPrior(name="objectness_prior", learned=prior_scale)([tokens, logits])

    outputs = {
        # Objectness carries no trailing axis, matching the per-token target
        "objectness": layers.Reshape((count,), name="objectness")(objectness),
        "box": layers.Dense(4, name="box")(x),
        "classes": layers.Dense(config.CLASS_COUNT, name="classes")(x),
    }

    return keras.Model([tokens, mask], outputs, name=MODEL_NAME)


# Training


class SetTokenDataset(keras.utils.PyDataset):
    """Token tensors and per-token targets of one fold's mammograms."""

    def __init__(self, dataframe, tokens, batch_size, **kwargs):
        super().__init__(**kwargs)
        self.ids = [utility.mammogram_id(path) for path in dataframe["full_image"]]
        self.tokens = tokens
        self.batch_size = int(batch_size)

    def __len__(self):
        return int(np.ceil(len(self.ids) / self.batch_size))

    def __getitem__(self, index):
        entries = [
            self.tokens[image_id] for image_id in self.ids[index * self.batch_size : (index + 1) * self.batch_size]
        ]

        inputs = {
            "tokens": np.stack([entry["tokens"] for entry in entries]),
            "mask": np.stack([entry["mask"] for entry in entries]),
        }
        targets = {
            "objectness": np.stack([entry["objectness"] for entry in entries]),
            "box": np.stack([entry["box"] for entry in entries]),
            "classes": np.stack([entry["classes"] for entry in entries]),
        }

        return inputs, targets


class SetMetrics(keras.callbacks.Callback):
    """Per-epoch validation row, the best raw weights, and an aligned table on stdout."""

    def __init__(self, detector, val_df, fit, n_epochs, tag=""):
        super().__init__()
        self.detector = detector
        self.val_df = val_df
        self.fit = fit
        self.n_epochs = int(n_epochs)
        self.tag = tag
        self.table = utility.EpochTable(
            cnn_detector.EPOCH_TABLE_COLUMNS,
            cnn_detector.EPOCH_TABLE_KEYS,
            cnn_detector.EPOCH_TABLE_FORMATS,
            self.n_epochs,
        )
        self.print_header = True

    def on_epoch_begin(self, epoch, logs=None):
        if self.print_header:
            self.print_header = False
            print(self.table.header())

            for entry in self.fit.rows:
                print(self.table.row(entry))

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        metrics = self.detector.validate(self.val_df, desc=f"{self.tag}Epoch {epoch + 1}")

        entry = {
            "epoch": epoch + 1,
            "train_loss": logs.get("loss", np.nan),
            **{f"{head}_loss": logs.get(f"{head}_loss", np.nan) for head in ("objectness", "box", "classes")},
            **{key: value for key, value in metrics.items() if key != "ap50_per_class"},
            **{f"ap50_{name}": value for name, value in zip(config.CLASS_NAMES, metrics["ap50_per_class"])},
        }

        improved = self.fit.add(entry, metrics["ap50"], [weight.copy() for weight in self.detector.model.get_weights()])

        print(self.table.row(entry) + (" | Saved best model" if improved else ""))


class StackerSet(Detector.Detector):
    """Set-based stacker: attention over the base detectors' ranked candidates."""

    model_name = MODEL_NAME
    default_epochs = config.SET_NUM_EPOCHS
    score_threshold = config.STACKER_SCORE_THRESHOLD
    inference_score_threshold = config.STACKER_INFERENCE_SCORE_THRESHOLD
    iou_threshold = config.STACKER_IOU_THRESHOLD
    stacker_score_threshold = config.STACKER_SCORE_THRESHOLD
    stacker_top_k = config.STACKER_TOP_K
    stacker_collect_threshold = 0.0
    resume_artifact_keys = ()
    history_decimals = 4

    def __init__(self, architecture=None, lesion_type="mass", tag=None):
        super().__init__()
        self.architecture = architecture_defaults() if architecture is None else dict(architecture)
        self.lesion_type = lesion_type
        self.model_name = MODEL_NAME if tag is None else f"{MODEL_NAME}_{tag}"
        config.FOLD_LAYOUT.setdefault(self.model_name, (config.STACKER_WORKSPACE, ".keras"))
        self.tokens = None

    # model hooks

    def build(self):
        self.model = build_stacker_set(**self.architecture)

        return self.model

    def prepare_model(self):
        self.build()

    def model_signature(self, lesion_type):
        return utility.keras_model_signature(self.model)

    def training_config(self, lesion_type, best_epoch=None, best_val_ap50=None):
        """Pipeline settings the checkpoint depends on, so a consumer can reproduce it."""
        return {
            "lesion_type": lesion_type,
            "base_models": list(BASE_MODELS),
            "base_fingerprints": stacker.base_fingerprints(lesion_type),
            "architecture": dict(self.architecture),
            "image_size": config.CNN_IMAGE_SIZE,
            "class_names": list(config.CLASS_NAMES),
            "anchor_iou_positive": config.CNN_ANCHOR_IOU_POSITIVE,
            "anchor_iou_negative": config.CNN_ANCHOR_IOU_NEGATIVE,
            "iou_threshold": config.STACKER_IOU_THRESHOLD,
            "nms_threshold": config.STACKER_NMS_THRESHOLD,
            "nms_containment": config.STACKER_NMS_CONTAINMENT,
            "score_threshold": config.STACKER_SCORE_THRESHOLD,
            "inference_score_threshold": config.STACKER_INFERENCE_SCORE_THRESHOLD,
            "candidate_max": config.STACKER_CANDIDATE_MAX,
            "box_loss_weight": config.SET_BOX_LOSS_WEIGHT,
            "class_loss_weight": config.SET_CLASS_LOSS_WEIGHT,
            "class_loss_weight_start": config.SET_CLASS_LOSS_WEIGHT_START,
            "class_loss_ramp_epochs": config.SET_CLASS_LOSS_RAMP_EPOCHS,
            "batch_size": config.SET_BATCH_SIZE,
            "epochs": config.SET_NUM_EPOCHS,
            "learning_rate": config.SET_LR,
            "weight_decay": config.SET_WEIGHT_DECAY,
            "best_epoch": best_epoch,
            "best_val_ap50": best_val_ap50,
        }

    def resume_state(self, paths, fold, epochs, fingerprint):
        """No mid-training state: a fold is trained in one call or retrained."""
        return None

    def prepare(self, row):
        entry = self.tokens[utility.mammogram_id(row.full_image)]

        return entry["tokens"], entry["gt_boxes"], entry["gt_labels"]

    def prepare_fold_data(self, trainval_df, fold, lesion_type):
        """Build the token tensors of every trainval mammogram from the exported candidates."""
        self.lesion_type = lesion_type
        self.tokens = build_tokens(
            trainval_df, lesion_type, range(config.N_FOLDS), agreement=self.architecture.get("agreement", False)
        )

    def dataset(self, dataframe):
        return SetTokenDataset(dataframe, self.tokens, config.SET_BATCH_SIZE)

    # training

    def compile(self, steps_per_epoch, epochs=None):
        epochs = config.SET_NUM_EPOCHS if epochs is None else int(epochs)

        schedule = cnn_detector.WarmupCosineDecay(
            peak_learning_rate=config.SET_LR,
            warmup_steps=config.SET_WARMUP_EPOCHS * steps_per_epoch,
            total_steps=epochs * steps_per_epoch,
            min_learning_rate=config.SET_MIN_LR,
        )
        optimizer = cnn_detector.exclude_from_weight_decay(
            keras.optimizers.AdamW(learning_rate=schedule, weight_decay=config.SET_WEIGHT_DECAY)
        )

        # A re-ranking model has no box loss to apply, so the head is trained with weight 0
        box_weight = config.SET_BOX_LOSS_WEIGHT if self.architecture["box_head"] == "delta" else 0.0

        self.model.compile(
            optimizer=optimizer,
            loss={
                "objectness": cnn_detector.MaskedBinaryCrossentropy(),
                "box": cnn_detector.MaskedHuber(),
                "classes": cnn_detector.MaskedClassCrossentropy(),
            },
            loss_weights={"objectness": 1.0, "box": box_weight, "classes": config.SET_CLASS_LOSS_WEIGHT},
        )

        return schedule

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
        """Fit the token model, validating the best raw weights every epoch."""
        epochs = config.SET_NUM_EPOCHS if epochs is None else epochs

        if self.model is None:
            self.build()

        dataset = self.dataset(train_df)
        self.compile(steps_per_epoch=len(dataset), epochs=epochs)

        fit = utility.FitHistory()

        callbacks = [
            SetMetrics(self, val_df, fit, epochs, tag=tag),
            cnn_detector.ClassLossRamp(
                start=config.SET_CLASS_LOSS_WEIGHT_START,
                end=config.SET_CLASS_LOSS_WEIGHT,
                ramp_epochs=config.SET_CLASS_LOSS_RAMP_EPOCHS,
                total_epochs=epochs,
            ),
        ]

        self.model.fit(dataset, epochs=epochs, verbose=0, callbacks=callbacks)

        if fit.best_weights is not None:
            self.model.set_weights(fit.best_weights)

        return Detector.TrainResult(
            history=fit.rows,
            best_epoch=fit.best_epoch,
            best_ap50=fit.best_ap50,
            best_weights=fit.best_weights,
        )

    def fit_fold(self, trainval_df, fold, lesion_type, epochs=None, force=False):
        """Train one fold and write its artifacts, mirroring the detector folds."""
        utility.set_deterministic()

        epochs = self.default_epochs if epochs is None else epochs
        paths = config.fold_paths(self.model_name, lesion_type, fold)

        self.prepare_model()
        fingerprint = self.fingerprint(lesion_type)

        manifest = utility.read_oof_manifest(paths["manifest"])
        entry = manifest["folds"].get(str(fold))

        if entry is not None and not force:
            if entry.get("fingerprint") != fingerprint:
                print(f"Fold {fold}: artifacts came from a different pipeline, retraining")
            elif paths["model"].exists():
                print(
                    f"Fold {fold}: complete (val AP50 {entry['best_val_ap50']:.4f} at epoch "
                    f"{entry['best_epoch']}), skipping"
                )
                utility.write_history_table(
                    pd.read_csv(paths["history"]),
                    cnn_detector.EPOCH_TABLE_COLUMNS,
                    cnn_detector.EPOCH_TABLE_KEYS,
                    cnn_detector.EPOCH_TABLE_FORMATS,
                    entry["epochs"],
                    entry["best_epoch"],
                )

                return entry

        train_df, val_df = utility.oof_split(trainval_df, fold)
        print(f"Fold {fold}: train {len(train_df):,}, val {len(val_df):,}, epochs {epochs}")

        result = self.train(train_df, val_df, epochs=epochs, tag=f"Fold {fold} ")

        fold_config = self.training_config(lesion_type, result.best_epoch, result.best_ap50)
        fold_config["fingerprint"] = fingerprint

        utility.write_atomically(lambda path: self.history_frame(result).to_csv(path, index=False), paths["history"])
        utility.write_atomically(lambda path: path.write_text(json.dumps(fold_config, indent=2)), paths["config"])
        self.model.save(paths["model"])

        entry = {
            "fold": fold,
            "seed": config.RANDOM_STATE,
            "epochs": epochs,
            "best_epoch": int(result.best_epoch),
            "best_val_ap50": float(result.best_ap50),
            "record_count": 0,
            "image_count": int(len(val_df)),
            "exported": False,
            "fingerprint": fingerprint,
        }
        manifest["folds"][str(fold)] = entry
        utility.write_oof_manifest(paths["manifest"], manifest)

        print(
            f"Fold {fold}: val AP50 {entry['best_val_ap50']:.4f} at epoch {entry['best_epoch']} "
            f"over {entry['image_count']} mammograms"
        )

        return entry

    # inference

    def load(self, path=None):
        path = self.weights_path if path is None else path

        if path is None:
            raise ValueError("No weights path given.")

        self.weights_path = Path(path)
        self.model = keras.models.load_model(path, compile=False)

        return self.model

    def save(self, path, fingerprint=None, epoch=0, ap50=0.0, history=None, lesion_type="mass"):
        self.model.save(path)

        return Path(path)

    def predict_raw(self, dataframe, score_threshold=None, desc="predict", progress=None, batch_size=None):
        """Decoded candidates per mammogram, after NMS, with their ground truth."""
        if self.model is None:
            raise ValueError("Call build() or load() before predicting.")

        score_threshold = self.score_threshold if score_threshold is None else score_threshold
        batch_size = config.SET_BATCH_SIZE if batch_size is None else batch_size

        dataframe = dataframe.reset_index(drop=True)
        refine = self.architecture["box_head"] == "delta"
        raw = []

        for start in range(0, len(dataframe), batch_size):
            rows = list(dataframe.iloc[start : start + batch_size].itertuples())
            entries = [self.tokens[utility.mammogram_id(row.full_image)] for row in rows]

            outputs = self.model.predict_on_batch(
                {
                    "tokens": np.stack([entry["tokens"] for entry in entries]),
                    "mask": np.stack([entry["mask"] for entry in entries]),
                }
            )

            scores = 1.0 / (1.0 + np.exp(-outputs["objectness"]))
            class_probs = np.exp(outputs["classes"] - outputs["classes"].max(axis=-1, keepdims=True))
            class_probs = class_probs / class_probs.sum(axis=-1, keepdims=True)

            for row, entry, image_scores, image_deltas, image_classes in zip(
                rows, entries, scores, outputs["box"], class_probs
            ):
                keep = image_scores >= score_threshold

                boxes = entry["boxes"].copy()

                if refine and keep.any():
                    boxes[keep] = apply_deltas(entry["boxes"][keep], image_deltas[keep])

                boxes, image_scores = boxes[keep], image_scores[keep]
                labels = image_classes[keep].argmax(axis=1).astype(np.int64)
                class_scores = image_classes[keep].max(axis=1).astype(np.float32)

                order = cnn_detector.cnn_nms(
                    boxes,
                    image_scores,
                    iou_threshold=config.STACKER_NMS_THRESHOLD,
                    containment_threshold=config.STACKER_NMS_CONTAINMENT,
                )

                raw.append(
                    {
                        "image_id": utility.mammogram_id(row.full_image),
                        "image_shape": (config.CNN_IMAGE_SIZE, config.CNN_IMAGE_SIZE),
                        "boxes": boxes[order],
                        "scores": image_scores[order],
                        "labels": labels[order],
                        "class_scores": class_scores[order],
                        "gt_boxes": entry["gt_boxes"],
                        "gt_labels": entry["gt_labels"],
                    }
                )

        return raw

    def predict_image(self, row, score_threshold=None):
        """One mammogram through the whole pipeline: image, ground truth, predictions."""
        image, _, gt_boxes, gt_labels = cnn_detector.cnn_prepare_image(row)
        entry = self.tokens[utility.mammogram_id(row.full_image)]

        outputs = self.model.predict_on_batch({"tokens": entry["tokens"][None], "mask": entry["mask"][None]})
        scores = (1.0 / (1.0 + np.exp(-outputs["objectness"][0]))).astype(np.float32)
        class_probs = np.exp(outputs["classes"][0] - outputs["classes"][0].max(axis=-1, keepdims=True))
        class_probs = class_probs / class_probs.sum(axis=-1, keepdims=True)

        keep = scores >= (self.score_threshold if score_threshold is None else score_threshold)

        boxes = entry["boxes"].copy()

        if self.architecture["box_head"] == "delta" and keep.any():
            boxes[keep] = apply_deltas(entry["boxes"][keep], outputs["box"][0][keep])

        boxes, scores = boxes[keep], scores[keep]
        labels = class_probs[keep].argmax(axis=1).astype(np.int64)

        if len(boxes):
            order = cnn_detector.cnn_nms(
                boxes,
                scores,
                iou_threshold=config.STACKER_NMS_THRESHOLD,
                containment_threshold=config.STACKER_NMS_CONTAINMENT,
            )
            boxes, scores, labels = boxes[order], scores[order], labels[order]

        return image, gt_boxes, gt_labels, boxes, scores, labels


def run_variant(trainval_df, lesion_type, architecture, tag=None, fold=None, epochs=None, force=False):
    """Build the tokens, then train one set stacker on one fold."""
    detector = StackerSet(architecture, lesion_type, tag=tag)
    detector.prepare_fold_data(trainval_df, fold, lesion_type)

    return detector.fit_fold(trainval_df, fold, lesion_type, epochs=epochs, force=force)
