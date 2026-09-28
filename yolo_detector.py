"""YOLO detector for mammogram masses (ultralytics), including the dataset export."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from tqdm.auto import tqdm

import config
import utility
import Detector

# This module's row in config.FOLD_LAYOUT
MODEL_NAME = "yolo"

# The fold history is ultralytics' own results.csv, so the reprint reads its columns
EPOCH_TABLE_COLUMNS = ("Epoch", "box", "cls", "reg", "P", "R", "mAP50", "mAP50-95")
EPOCH_TABLE_KEYS = (
    "epoch",
    "train/box_loss",
    "train/cls_loss",
    "train/l1_loss",
    "metrics/precision(B)",
    "metrics/recall(B)",
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
)
EPOCH_TABLE_FORMATS = ("{:.0f}",) + ("{:.4f}",) * 7


def training_config(lesion_type, best_epoch=None, best_val_ap50=None):
    """Pipeline settings a checkpoint depends on, so any consumer can reproduce it."""
    return {
        "lesion_type": lesion_type,
        "image_size": config.YOLO_IMAGE_SIZE,
        "class_names": list(config.CLASS_NAMES),
        "model": config.YOLO_MODEL,
        "epochs": config.YOLO_EPOCHS,
        "batch_size": config.YOLO_BATCH_SIZE,
        "optimizer": config.YOLO_OPTIMIZER,
        "learning_rate": config.YOLO_LR,
        "weight_decay": config.YOLO_WEIGHT_DECAY,
        "cos_lr": config.YOLO_COS_LR,
        "patience": config.YOLO_PATIENCE,
        "augment_flip_probability": config.YOLO_AUGMENT_FLIP_PROBABILITY,
        "augment_rotation_degrees": config.YOLO_AUGMENT_ROTATION_DEGREES,
        "score_threshold": config.YOLO_SCORE_THRESHOLD,
        "inference_score_threshold": config.YOLO_INFERENCE_SCORE_THRESHOLD,
        "iou_threshold": config.YOLO_IOU_THRESHOLD,
        "stacker_score_threshold": config.YOLO_STACKER_SCORE_THRESHOLD,
        "stacker_top_k": config.YOLO_STACKER_TOP_K,
        "best_epoch": best_epoch,
        "best_val_ap50": best_val_ap50,
    }


def data_yaml_text(root) -> str:
    """The dataset description ultralytics trains from, including the class names."""
    names = "\n".join(f"  {index}: {name}" for index, name in enumerate(config.CLASS_NAMES))

    return f"""
path: {Path(root).as_posix()}

train: images/train
val: images/val
test: images/test

names:
{names}
""".strip()


def yolo_prepare_image(row):
    """Breast-masked image, GT boxes and one class label per box at the export size."""
    image = utility.read_image(row.full_image)
    mask = utility.read_image(row.breast_mask)
    image = image * (mask > 0)

    boxes = utility.masks_to_boxes(row.mask_paths)
    class_labels = utility.pathology_labels(row.pathologies)

    image, boxes = utility.resize_with_boxes(image, boxes, max_size=config.YOLO_IMAGE_SIZE)

    return image, boxes, class_labels


class YOLODetector(Detector.Detector):
    """YOLO detector plus its exported PNG/label dataset."""

    model_name = MODEL_NAME
    default_epochs = config.YOLO_EPOCHS
    score_threshold = config.YOLO_SCORE_THRESHOLD
    inference_score_threshold = config.YOLO_INFERENCE_SCORE_THRESHOLD
    iou_threshold = config.YOLO_IOU_THRESHOLD
    stacker_score_threshold = config.YOLO_STACKER_SCORE_THRESHOLD
    stacker_top_k = config.YOLO_STACKER_TOP_K
    stacker_collect_threshold = config.YOLO_STACKER_COLLECT_THRESHOLD
    # The fold history is ultralytics' own results.csv, stored verbatim
    history_decimals = None
    fold_config_extras = {"selection": "ultralytics best.pt by mAP50-95"}
    epoch_table_columns = EPOCH_TABLE_COLUMNS
    epoch_table_keys = EPOCH_TABLE_KEYS
    epoch_table_formats = EPOCH_TABLE_FORMATS

    def __init__(self, root=None, weights_path=None, lesion_type="mass", device=0):
        super().__init__(weights_path)
        self.lesion_type = lesion_type
        self.root = Path(root) if root is not None else config.YOLO_WORKSPACE / lesion_type
        self.device = device

    def model_signature(self, lesion_type):
        """The dataset description the run is trained from, which is this pipeline's graph."""
        return {"data_yaml": data_yaml_text(config.YOLO_WORKSPACE / lesion_type)}

    def training_config(self, lesion_type, best_epoch=None, best_val_ap50=None):
        """Pipeline settings a checkpoint depends on, so any consumer can reproduce it."""
        return training_config(lesion_type, best_epoch, best_val_ap50)

    def prepare_fold_data(self, trainval_df, fold, lesion_type):
        """Write the fold's PNG/label export, which is what ultralytics trains from."""
        self.lesion_type = lesion_type
        self.export_fold(trainval_df, fold)

    # dataset export

    def prepare(self, row):
        """Breast-masked image, GT boxes and class labels at the export size."""
        return yolo_prepare_image(row)

    def yaml_path(self):
        return self.root / "data.yaml"

    def image_path(self, full_image, split="train"):
        return self.root / "images" / split / f"{utility.mammogram_id(full_image)}.png"

    def label_path(self, full_image, split="train"):
        return self.root / "labels" / split / f"{utility.mammogram_id(full_image)}.txt"

    def export(self, dataframe, split_name, progress=None):
        """Write the split as PNGs plus YOLO normalised label files."""
        image_dir = self.root / "images" / split_name
        label_dir = self.root / "labels" / split_name

        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)

        # A labels cache from another fold would describe the old image list
        (label_dir.parent / f"{split_name}.cache").unlink(missing_ok=True)

        expected = {f"{utility.mammogram_id(full_image)}.png" for full_image in dataframe["full_image"]}

        for stale in image_dir.glob("*.png"):
            if stale.name not in expected:
                stale.unlink()

        for stale in label_dir.glob("*.txt"):
            if f"{stale.stem}.png" not in expected:
                stale.unlink()

        local_bar = progress is None
        if local_bar:
            progress = tqdm(total=len(dataframe), desc=f"{split_name} export", leave=False)

        try:
            progress.reset(total=len(dataframe))
            progress.set_description(f"{split_name} export")

            for row in dataframe.itertuples():
                image_path = self.image_path(row.full_image, split_name)
                label_path = self.label_path(row.full_image, split_name)

                if image_path.exists() and label_path.exists():
                    progress.update(1)
                    continue

                image, boxes, class_labels = yolo_prepare_image(row)

                cv2.imwrite(
                    str(self.image_path(row.full_image, split_name)),
                    (image * 255).clip(0, 255).astype(np.uint8),
                )

                height, width = image.shape

                with open(self.label_path(row.full_image, split_name), "w") as f:
                    for (x1, y1, x2, y2), class_label in zip(boxes, class_labels):
                        xc = (x1 + x2) / 2 / width
                        yc = (y1 + y2) / 2 / height
                        w = (x2 - x1) / width
                        h = (y2 - y1) / height

                        f.write(f"{int(class_label)} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

                progress.update(1)
        finally:
            if local_bar:
                progress.close()

    def write_data_yaml(self):
        yaml_path = self.yaml_path()
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        yaml_path.write_text(data_yaml_text(self.root))

        return yaml_path

    def export_fold(self, trainval_df, fold, force=False):
        """Write the train and val splits of one out-of-fold unit."""
        marker = self.root / f"{self.lesion_type}_export.json"
        fingerprint = self.fingerprint(self.lesion_type)

        train_df, val_df = utility.oof_split(trainval_df, fold)

        if not force and marker.exists():
            recorded = json.loads(marker.read_text())
            if recorded.get("fold") == fold and recorded.get("fingerprint") == fingerprint:
                return {split: tuple(value) for split, value in recorded["counts"].items()}

        self.export(train_df, "train")
        self.export(val_df, "val")
        self.write_data_yaml()

        counts = {split: list(value) for split, value in self.counts().items()}

        utility.write_atomically(
            lambda path: path.write_text(
                json.dumps(
                    {"fold": fold, "fingerprint": fingerprint, "counts": counts},
                    indent=2,
                )
            ),
            marker,
        )

        return {split: tuple(value) for split, value in counts.items()}

    def counts(self, splits=("train", "val")):
        return {
            split: (
                len(list((self.root / "images" / split).glob("*.png"))),
                len(list((self.root / "labels" / split).glob("*.txt"))),
            )
            for split in splits
        }

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
        """Train into the fold's run directory, or continue an interrupted run from its last.pt."""
        epochs = config.YOLO_EPOCHS if epochs is None else epochs
        run_dir = (
            Path(checkpoint_paths["run"])
            if checkpoint_paths is not None
            else config.YOLO_PROJECT / config.YOLO_RUN_NAME
        )

        if resume is not None and (run_dir / "weights" / "last.pt").exists():
            last_path = run_dir / "weights" / "last.pt"

            results = YOLO(str(last_path)).train(resume=str(last_path))
        else:
            results = YOLO(config.YOLO_MODEL).train(
                # Dataset & Input Configuration
                data=str(self.yaml_path()),
                imgsz=config.YOLO_IMAGE_SIZE,
                cache=False,
                # Training Scale & Schedule
                epochs=epochs,
                batch=config.YOLO_BATCH_SIZE,
                patience=config.YOLO_PATIENCE,
                pretrained=True,
                # Optimizer & Learning Rate
                optimizer=config.YOLO_OPTIMIZER,
                lr0=config.YOLO_LR,
                weight_decay=config.YOLO_WEIGHT_DECAY,
                cos_lr=config.YOLO_COS_LR,
                # Augmentation: flip and rotation match the CNN and Faster R-CNN
                fliplr=config.YOLO_AUGMENT_FLIP_PROBABILITY,
                degrees=config.YOLO_AUGMENT_ROTATION_DEGREES,
                # Hardware & Execution
                workers=config.YOLO_WORKERS,
                device=self.device,
                amp=True,
                deterministic=True,
                seed=config.RANDOM_STATE,
                # Logging & Output Management
                project=str(run_dir.parent),
                name=run_dir.name,
                exist_ok=True,
                verbose=False,
                plots=True,
            )

        self.weights_path = Path(getattr(results, "save_dir", run_dir)) / "weights" / "best.pt"
        self.model = YOLO(str(self.weights_path))

        history = self.history(run_dir)

        # ultralytics selects the checkpoint itself, so the fold reports its own best epoch
        return Detector.TrainResult(
            history=history,
            best_epoch=self.best_epoch(history),
            best_ap50=float(self.validate(val_df)["ap50"]),
        )

    def history(self, run_dir=None):
        """Ultralytics' own per-epoch metrics, which the fold history records verbatim."""
        run_dir = Path(run_dir) if run_dir is not None else config.YOLO_PROJECT / config.YOLO_RUN_NAME

        return pd.read_csv(run_dir / "results.csv")

    def best_epoch(self, history):
        """The epoch ultralytics kept as best.pt, by its own fitness."""
        column = "fitness" if "fitness" in history.columns else "metrics/mAP50-95(B)"

        return int(history.loc[history[column].idxmax(), "epoch"])

    def resume_state(self, paths, fold, epochs, fingerprint):
        """Ultralytics continues its own run, so the state is the run's results.csv."""
        completed = len(self.history(paths["run"])) if (paths["run"] / "results.csv").exists() else 0

        if not completed:
            return None

        tqdm.write(f"Fold {fold}: resuming from epoch {completed} of {epochs}")

        return {"initial_epoch": completed, "history": [], "best_ap50": 0.0, "best_epoch": 0, "best_weights": None}

    # inference

    def load(self, path=None):
        """Restore the model from a checkpoint, defaulting to the one this detector was built with."""
        path = Path(path) if path is not None else self.weights_path
        if path is None:
            raise ValueError("No weights path given.")

        self.weights_path = path
        self.model = YOLO(str(path))

        return self.model

    def save(self, path, fingerprint=None, epoch=0, ap50=0.0, history=None, lesion_type="mass"):
        """Copy the trained checkpoint; ultralytics owns its own run directory and metadata."""
        shutil.copy2(self.weights_path, path)
        return Path(path)

    def predict_image(self, row, score_threshold=None):
        """Full pipeline for one mammogram: image, GT boxes and labels, then predictions."""
        score_threshold = config.YOLO_INFERENCE_SCORE_THRESHOLD if score_threshold is None else score_threshold

        image, gt_boxes, gt_labels = yolo_prepare_image(row)

        result = self.predict_prepared(image)

        boxes, scores, labels = self.result_arrays(result)

        keep = scores >= score_threshold

        return image, gt_boxes, gt_labels, boxes[keep], scores[keep], labels[keep]

    def predict_prepared(self, image, conf=None, iou=None):
        """Predictions on one prepared image, so no PNG round trip is needed."""
        conf = config.YOLO_SCORE_THRESHOLD if conf is None else conf
        iou = config.YOLO_IOU_THRESHOLD if iou is None else iou

        source = (image * 255).clip(0, 255).astype(np.uint8)
        source = np.repeat(source[:, :, None], 3, axis=2)

        return self.model.predict(source=source, imgsz=config.YOLO_IMAGE_SIZE, conf=conf, iou=iou, verbose=False)[0]

    def evidence(self, row, conf=0.001, iou=0.9):
        """Every box one mammogram makes the detector fire, at a floor low enough to show it all.

        The default floor and IoU sit below the operating point and above the training NMS, so
        the whole prediction field is visible rather than only the boxes that survive.
        """
        image, gt_boxes, gt_labels = yolo_prepare_image(row)
        boxes, scores, labels = self.result_arrays(self.predict_prepared(image, conf=conf, iou=iou))

        return image, gt_boxes, gt_labels, boxes, scores

    @staticmethod
    def result_arrays(result):
        """Boxes, confidences and class indexes out of one ultralytics result."""
        return (
            result.boxes.xyxy.cpu().numpy(),
            result.boxes.conf.cpu().numpy(),
            result.boxes.cls.cpu().numpy().astype(np.int64),
        )

    def records(self, dataframe, conf=None, score_threshold=None, desc="yolo", progress=None):
        """Predictions per mammogram, one record each, with its ground truth."""
        if self.model is None:
            raise ValueError("Call load() or train() first.")

        conf = config.YOLO_SCORE_THRESHOLD if conf is None else conf
        score_threshold = 0.0 if score_threshold is None else score_threshold

        local_bar = progress is None
        if local_bar:
            progress = tqdm(total=len(dataframe), desc=desc, leave=False)
        else:
            # A borrowed bar belongs to the caller, so it is re-based for this pass
            progress.reset(total=len(dataframe))
            progress.set_description(desc)

        records = []

        try:
            for row in dataframe.itertuples():
                image, gt_boxes, gt_labels = yolo_prepare_image(row)

                boxes, scores, labels = self.result_arrays(self.predict_prepared(image, conf=conf))

                keep = scores >= score_threshold

                records.append(
                    {
                        "image_id": utility.mammogram_id(row.full_image),
                        "image_shape": image.shape[:2],
                        "boxes": boxes[keep],
                        "scores": scores[keep],
                        "labels": labels[keep],
                        "class_scores": scores[keep],
                        "gt_boxes": gt_boxes,
                        "gt_labels": gt_labels,
                    }
                )

                progress.update(1)
        finally:
            if local_bar:
                progress.close()

        return records

    def predict_raw(self, dataframe, score_threshold=None, desc="predict", progress=None):
        """Every detection above the score floor, per image, with its ground truth."""
        conf = self.score_threshold if score_threshold is None else score_threshold

        return self.records(dataframe, conf=conf, score_threshold=0.0, desc=desc, progress=progress)
