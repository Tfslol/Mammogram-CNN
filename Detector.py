"""Shared detector base: fingerprinting, the out-of-fold loop and the record export."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import config
import utility


@dataclass
class TrainResult:
    """What `train` returns: the per-epoch rows and the epoch the checkpoint came from."""

    history: object
    best_epoch: int
    best_ap50: float
    best_weights: object = None


def augment_sample(
    image,
    boxes,
    class_labels,
    mask=None,
    flip_probability=0.0,
    rotation_degrees=0.0,
    scale_range=None,
    translate_fraction=0.0,
    gamma_range=None,
):
    """Random flip, affine transform and intensity jitter; boxes, classes and mask follow."""
    height, width = image.shape

    if flip_probability > 0 and np.random.random() < flip_probability:
        image = image[:, ::-1]
        if mask is not None:
            mask = mask[:, ::-1]
        if len(boxes):
            boxes = boxes.copy()
            boxes[:, [0, 2]] = width - boxes[:, [2, 0]]

    scaling = scale_range is not None and tuple(scale_range) != (1.0, 1.0)

    if rotation_degrees > 0 or scaling or translate_fraction > 0:
        angle = np.random.uniform(-rotation_degrees, rotation_degrees) if rotation_degrees > 0 else 0.0
        scale = np.random.uniform(*scale_range) if scaling and scale_range is not None else 1.0

        matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, scale)

        if translate_fraction > 0:
            matrix[0, 2] += np.random.uniform(-translate_fraction, translate_fraction) * width
            matrix[1, 2] += np.random.uniform(-translate_fraction, translate_fraction) * height

        image = cv2.warpAffine(image, matrix, (width, height), flags=cv2.INTER_LINEAR, borderValue=0)
        if mask is not None:
            mask = cv2.warpAffine(mask, matrix, (width, height), flags=cv2.INTER_NEAREST, borderValue=0)

        if len(boxes):
            # Map the four corner pairs as (4, N, 2), then take the enclosing xyxy box
            corners = np.stack(
                [
                    np.stack([boxes[:, 0], boxes[:, 1]], axis=-1),
                    np.stack([boxes[:, 2], boxes[:, 1]], axis=-1),
                    np.stack([boxes[:, 2], boxes[:, 3]], axis=-1),
                    np.stack([boxes[:, 0], boxes[:, 3]], axis=-1),
                ],
                axis=0,
            )
            rotated = (corners.reshape(-1, 2) @ matrix[:, :2].T + matrix[:, 2]).reshape(4, len(boxes), 2)

            boxes = np.stack(
                [
                    rotated[:, :, 0].min(axis=0),
                    rotated[:, :, 1].min(axis=0),
                    rotated[:, :, 0].max(axis=0),
                    rotated[:, :, 1].max(axis=0),
                ],
                axis=-1,
            )
            np.clip(boxes, 0.0, np.array([width, height, width, height], dtype=np.float32), out=boxes)

            # Drop boxes flattened by the transform, keeping the labels aligned
            keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
            boxes = boxes[keep]
            class_labels = class_labels[keep]

    if gamma_range is not None:
        image = np.power(np.clip(image, 0.0, 1.0), np.random.uniform(gamma_range[0], gamma_range[1]))

    return (
        image.astype(np.float32),
        boxes.astype(np.float32),
        class_labels,
        None if mask is None else mask.astype(np.float32),
    )


def detection_metrics(raw, iou_threshold, inference_score_threshold):
    """The shared metric protocol over per-image raw predictions."""
    predictions = [{"boxes": entry["boxes"], "scores": entry["scores"], "labels": entry["labels"]} for entry in raw]
    targets = [{"boxes": entry["gt_boxes"], "labels": entry["gt_labels"]} for entry in raw]

    metrics = utility.evaluate_detector_by_class(
        predictions, targets, iou_threshold=iou_threshold, operating_threshold=inference_score_threshold
    )

    detections = sum(int((prediction["scores"] >= inference_score_threshold).sum()) for prediction in predictions)
    metrics["det_per_image"] = detections / max(len(predictions), 1)

    return metrics


class Detector(ABC):
    """Common pipeline of the three detectors: fingerprint, fold loop, metrics and export."""

    model_name = ""  # key into config.FOLD_LAYOUT, and the artifact stem
    default_epochs = 0  # epoch budget when a fold starts without one
    score_threshold = 0.0  # collection floor the PR curve is built over
    inference_score_threshold = 0.0  # operating point for P/R/F1 and final predictions
    iou_threshold = 0.5  # matching IoU for AP50
    stacker_score_threshold = 0.0  # export floor, separate from the operating point
    stacker_top_k = 0  # per-image cap on the exported candidates
    stacker_collect_threshold = 0.0  # deep floor the ranked candidate feed is collected at
    history_decimals = 4  # rounding of the fold history CSV, None writes the frame as is
    fold_config_extras: dict = {}  # extra keys the fold config JSON carries
    resume_artifact_keys = ()  # mid-training artifacts removed once a fold completes
    epoch_table_columns = ()
    epoch_table_keys = ()
    epoch_table_formats = ()

    def __init__(self, weights_path: str | Path | None = None):
        self.weights_path = Path(weights_path) if weights_path is not None else None
        self.model = None

    # Model-specific hooks

    @abstractmethod
    def model_signature(self, lesion_type) -> dict:
        """Graph description the fingerprint hashes; any change marks every fold stale."""

    @abstractmethod
    def training_config(self, lesion_type, best_epoch=None, best_val_ap50=None) -> dict:
        """Settings a checkpoint depends on, so any consumer can reproduce it."""

    @abstractmethod
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
    ) -> "TrainResult":
        """Fit one fold and return the per-epoch rows and the selected epoch."""

    @abstractmethod
    def predict_raw(self, dataframe, score_threshold=None, desc="predict", progress=None) -> list:
        """Per-image boxes, scores, labels, class scores and ground truth above the floor."""

    @abstractmethod
    def resume_state(self, paths, fold, epochs, fingerprint):
        """Training state to continue an interrupted fold from, or None to start it fresh."""

    @abstractmethod
    def load(self, path=None) -> object:
        """Restore the model from a checkpoint."""

    @abstractmethod
    def save(self, path, fingerprint=None, epoch=0, ap50=0.0, history=None, lesion_type="mass") -> Path:
        """Write the checkpoint in this detector's own format."""

    @abstractmethod
    def predict_image(self, row, score_threshold=None) -> tuple:
        """One mammogram through the whole pipeline: image, ground truth, predictions."""

    @abstractmethod
    def prepare(self, row) -> tuple:
        """Prepared sample for one row; an augmenting detector adds its own train-only flag."""

    def prepare_model(self) -> None:
        """Build the model whose graph the fingerprint hashes. No-op when it is not needed."""
        return None

    def prepare_fold_data(self, trainval_df, fold, lesion_type) -> None:
        """Write anything the fold's training loop reads from disk. No-op by default."""
        return None

    # Shared pipeline

    def fingerprint(self, lesion_type):
        """Short hash of the model graph and settings; a mismatch marks a fold's artifacts stale."""
        return utility.fingerprint(
            self.model_signature(lesion_type), self.training_config(lesion_type), config.RANDOM_STATE
        )

    def validate(self, dataframe, score_threshold=None, desc="val", progress=None):
        """The per-epoch row, with per-class AP50 and P/R/F1 at the operating point."""
        raw = self.predict_raw(dataframe, score_threshold=score_threshold, desc=desc, progress=progress)

        return detection_metrics(raw, self.iou_threshold, self.inference_score_threshold)

    def infer(self, dataframe, score_threshold=None, desc="infer", **kwargs):
        """Final predictions per mammogram, cut at the operating threshold."""
        score_threshold = self.inference_score_threshold if score_threshold is None else score_threshold

        # The pool is collected at the deeper of the two floors
        collection = min(self.score_threshold, score_threshold)

        raw = self.predict_raw(dataframe, score_threshold=collection, desc=desc, **kwargs)

        records = []

        for row, entry in zip(dataframe.reset_index(drop=True).itertuples(), raw):
            keep = np.asarray(entry["scores"]) >= score_threshold

            records.append(
                {
                    "image_id": utility.mammogram_id(row.full_image),
                    "image_shape": entry["image_shape"],
                    "boxes": entry["boxes"][keep],
                    "scores": entry["scores"][keep],
                    "labels": entry["labels"][keep],
                    "class_scores": entry["class_scores"][keep],
                    "gt_boxes": entry["gt_boxes"],
                    "gt_labels": entry["gt_labels"],
                }
            )

        return records

    def detections(self, dataframe, score_threshold=None, top_k=None, rank=False, desc="export", **kwargs):
        """Stacker candidate records: a low floor plus a per-image top-K."""
        score_threshold = self.stacker_score_threshold if score_threshold is None else score_threshold
        top_k = self.stacker_top_k if top_k is None else top_k

        rows = []

        for record in self.infer(dataframe, score_threshold=score_threshold, desc=desc, **kwargs):
            order = np.argsort(-record["scores"])[:top_k]

            rows.append(
                {
                    **record,
                    "boxes": record["boxes"][order],
                    "scores": record["scores"][order],
                    "labels": record["labels"][order],
                    "class_scores": record["class_scores"][order],
                }
            )

        frame = utility.detection_records(rows)

        if rank:
            counts = [len(record["scores"]) for record in rows]
            places = np.concatenate([np.arange(1, count + 1) for count in counts]) if counts else np.zeros(0, np.int64)
            frame.insert(len(utility.DETECTION_RECORD_COLUMNS), "rank", places)

        return frame

    def open_candidate_floor(self):
        """Open whatever score floor lives inside the model, so a deep export can see below it."""
        return None

    def export_folds(self, trainval_df, lesion_type, folds=None, force=False):
        """Load every saved fold model and write its held-out fold's records and candidate feed."""
        folds = range(config.N_FOLDS) if folds is None else folds
        manifest_path = config.oof_manifest_path(self.model_name, lesion_type)
        manifest = utility.read_oof_manifest(manifest_path)

        floor = self.stacker_collect_threshold or self.stacker_score_threshold
        exported = {}

        for fold in folds:
            paths = config.fold_paths(self.model_name, lesion_type, fold)
            entry = manifest["folds"].get(str(fold), {})

            if not Path(paths["model"]).exists():
                tqdm.write(f"Fold {fold}: no checkpoint at {paths['model'].name}, skipped")
                continue

            if not force and paths["candidates"].exists() and entry.get("exported"):
                tqdm.write(f"Fold {fold}: already exported ({entry.get('candidate_count', 0):,}), skipping")
                exported[fold] = entry
                continue

            val_df = utility.oof_split(trainval_df, fold)[1]

            self.load(paths["model"])
            self.open_candidate_floor()

            records = self.detections(val_df, desc=f"Fold {fold} records")
            candidates = self.detections(
                val_df,
                score_threshold=floor,
                top_k=config.STACKER_CANDIDATE_MAX,
                rank=True,
                desc=f"Fold {fold} candidates",
            )

            utility.write_atomically(lambda path: records.to_csv(path, index=False), paths["records"])
            utility.write_atomically(lambda path: candidates.to_csv(path, index=False), paths["candidates"])

            entry = {
                **entry,
                "fold": fold,
                "record_count": int(len(records)),
                "candidate_count": int(len(candidates)),
                "image_count": int(len(val_df)),
                "exported": True,
            }
            manifest["folds"][str(fold)] = entry
            utility.write_oof_manifest(manifest_path, manifest)
            exported[fold] = entry

            tqdm.write(
                f"Fold {fold}: {len(records):,} records, {len(candidates):,} candidates "
                f"({len(candidates) / max(len(val_df), 1):.1f} per mammogram)"
            )

        return exported

    def history_frame(self, result):
        """Per-epoch table stored for the fold, rounded unless a detector opts out."""
        history = pd.DataFrame(result.history)

        return history if self.history_decimals is None else history.round(self.history_decimals)

    def save_fold_checkpoint(self, path, lesion_type, result, fingerprint):
        """Atomic checkpoint write; the file format is the detector's own."""
        utility.write_atomically(
            lambda temporary: self.save(
                temporary,
                fingerprint=fingerprint,
                epoch=int(result.best_epoch),
                ap50=float(result.best_ap50),
                history=result.history,
                lesion_type=lesion_type,
            ),
            path,
        )
        self.weights_path = Path(path)

    def oof_fold(self, trainval_df, fold, lesion_type, epochs=None, force=False, export=True):
        """Train one out-of-fold model, and export its records unless the export is off."""
        utility.set_deterministic()

        epochs = self.default_epochs if epochs is None else epochs

        paths = config.fold_paths(self.model_name, lesion_type, fold)
        # A fold trained without the export is complete on its checkpoint alone
        artifacts = [paths[key] for key in utility.FOLD_ARTIFACT_KEYS if export or key != "records"]

        # The fingerprint describes the graph, so the model has to be built first
        self.prepare_model()
        fingerprint = self.fingerprint(lesion_type)

        manifest = utility.read_oof_manifest(paths["manifest"])
        entry = manifest["folds"].get(str(fold))

        if entry is not None and not force:
            if entry.get("fingerprint") != fingerprint:
                tqdm.write(f"Fold {fold}: artifacts came from a different pipeline, retraining")
            elif all(path.exists() for path in artifacts):
                tqdm.write(
                    f"Fold {fold}: complete (val AP50 {entry['best_val_ap50']:.4f}, "
                    f"{entry.get('record_count', 0)} records), skipping"
                )
                utility.write_history_table(
                    pd.read_csv(paths["history"]),
                    self.epoch_table_columns,
                    self.epoch_table_keys,
                    self.epoch_table_formats,
                    entry["epochs"],
                    entry["best_epoch"],
                )
                return entry

        train_df, val_df = utility.oof_split(trainval_df, fold)

        if set(train_df["split"]) != {"trainval"} or set(val_df["split"]) != {"trainval"}:
            raise ValueError("An out-of-fold split must stay inside the trainval split.")

        self.prepare_fold_data(trainval_df, fold, lesion_type)

        resume = None if force else self.resume_state(paths, fold, epochs, fingerprint)

        tqdm.write(f"Fold {fold}: train {len(train_df):,}, val {len(val_df):,}, epochs {epochs}")

        result = self.train(
            train_df,
            val_df,
            epochs=epochs,
            tag=f"Fold {fold} ",
            resume=resume,
            checkpoint_paths=paths,
            fingerprint=fingerprint,
        )

        records = self.detections(val_df, desc=f"Fold {fold} export") if export else None

        fold_config = self.training_config(lesion_type, result.best_epoch, result.best_ap50)
        fold_config["fingerprint"] = fingerprint
        fold_config.update(self.fold_config_extras)

        utility.write_atomically(lambda path: self.history_frame(result).to_csv(path, index=False), paths["history"])

        if export:
            utility.write_atomically(lambda path: records.to_csv(path, index=False), paths["records"])

        utility.write_atomically(lambda path: path.write_text(json.dumps(fold_config, indent=2)), paths["config"])

        # Checkpoint last, so a fold counts as complete only when the rest is on disk
        self.save_fold_checkpoint(paths["model"], lesion_type, result, fingerprint)

        entry = {
            "fold": fold,
            "seed": config.RANDOM_STATE,
            "epochs": epochs,
            "best_epoch": int(result.best_epoch),
            "best_val_ap50": float(result.best_ap50),
            "record_count": 0 if records is None else int(len(records)),
            "image_count": int(len(val_df)),
            "exported": bool(export),
            "fingerprint": fingerprint,
        }

        manifest["folds"][str(fold)] = entry
        utility.write_oof_manifest(paths["manifest"], manifest)

        # The fold is complete, so the mid-training state is no longer needed
        for key in self.resume_artifact_keys:
            paths[key].unlink(missing_ok=True)

        tqdm.write(
            f"Fold {fold}: val AP50 {entry['best_val_ap50']:.4f} at epoch {entry['best_epoch']}, "
            f"{entry['record_count']} records over {entry['image_count']} mammograms"
        )

        return entry
