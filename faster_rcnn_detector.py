"""Faster R-CNN detector for mammogram masses (torchvision MobileNetV3-FPN)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn, FasterRCNN_MobileNet_V3_Large_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm.auto import tqdm

import config
import utility
import Detector

# This module's row in config.FOLD_LAYOUT
MODEL_NAME = "faster_rcnn"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Same shape as the CNN's table, with the RPN box term as the fourth loss column
EPOCH_TABLE_COLUMNS = ("Epoch", "Loss", "obj", "box", "cls", "rpn", "AP50", "mAP50", "P", "R", "F1", "Det")
EPOCH_TABLE_KEYS = (
    "epoch",
    "train_loss",
    "loss_objectness",
    "loss_box_reg",
    "loss_classifier",
    "loss_rpn_box_reg",
    "ap50",
    "ap50_macro",
    "precision",
    "recall",
    "f1",
    "det_per_image",
)
EPOCH_TABLE_FORMATS = ("{:.0f}",) + ("{:.4f}",) * 10 + ("{:.1f}",)


def training_config(lesion_type, best_epoch=None, best_val_ap50=None):
    """Pipeline settings a checkpoint depends on, so any consumer can reproduce it."""
    return {
        "lesion_type": lesion_type,
        "resize": config.FASTER_RCNN_RESIZE,
        "max_size": config.FASTER_RCNN_MAX_SIZE,
        "class_names": list(config.CLASS_NAMES),
        "num_classes": config.FASTER_RCNN_CLASS_COUNT,
        "batch_size": config.FASTER_RCNN_BATCH_SIZE,
        "epochs": config.FASTER_RCNN_NUM_EPOCHS,
        "learning_rate": config.FASTER_RCNN_LR,
        "momentum": config.FASTER_RCNN_MOMENTUM,
        "weight_decay": config.FASTER_RCNN_WEIGHT_DECAY,
        "min_learning_rate": config.FASTER_RCNN_MIN_LR,
        "lr_patience": config.LR_PATIENCE,
        "lr_factor": config.LR_FACTOR,
        "augment_flip_probability": config.FASTER_RCNN_AUGMENT_FLIP_PROBABILITY,
        "augment_rotation_degrees": config.FASTER_RCNN_AUGMENT_ROTATION_DEGREES,
        "augment_scale_range": list(config.FASTER_RCNN_AUGMENT_SCALE_RANGE),
        "augment_translate_fraction": config.FASTER_RCNN_AUGMENT_TRANSLATE_FRACTION,
        "augment_gamma_range": list(config.FASTER_RCNN_AUGMENT_GAMMA_RANGE),
        "score_threshold": config.FASTER_RCNN_SCORE_THRESHOLD,
        "inference_score_threshold": config.FASTER_RCNN_INFERENCE_SCORE_THRESHOLD,
        "iou_threshold": config.FASTER_RCNN_IOU_THRESHOLD,
        "stacker_score_threshold": config.FASTER_RCNN_STACKER_SCORE_THRESHOLD,
        "stacker_top_k": config.FASTER_RCNN_STACKER_TOP_K,
        "best_epoch": best_epoch,
        "best_val_ap50": best_val_ap50,
    }


def frc_augment(
    image,
    boxes,
    class_labels,
    flip_probability=None,
    rotation_degrees=None,
    scale_range=None,
    translate_fraction=None,
    gamma_range=None,
):
    """Random flip, affine transform and intensity jitter; the boxes and classes follow."""
    boxes = boxes.astype(np.float32)

    image, boxes, class_labels, _ = Detector.augment_sample(
        image,
        boxes,
        class_labels,
        flip_probability=(
            config.FASTER_RCNN_AUGMENT_FLIP_PROBABILITY if flip_probability is None else flip_probability
        ),
        rotation_degrees=(
            config.FASTER_RCNN_AUGMENT_ROTATION_DEGREES if rotation_degrees is None else rotation_degrees
        ),
        scale_range=config.FASTER_RCNN_AUGMENT_SCALE_RANGE if scale_range is None else scale_range,
        translate_fraction=(
            config.FASTER_RCNN_AUGMENT_TRANSLATE_FRACTION if translate_fraction is None else translate_fraction
        ),
        gamma_range=config.FASTER_RCNN_AUGMENT_GAMMA_RANGE if gamma_range is None else gamma_range,
    )

    return image, boxes, class_labels


def frc_prepare_image(row, augment=False):
    """Breast-masked image, GT boxes and one class label per box; augment is training only."""
    image = utility.read_image(row.full_image)
    mask = utility.read_image(row.breast_mask)
    image = image * (mask > 0)

    boxes = utility.masks_to_boxes(row.mask_paths)
    class_labels = utility.pathology_labels(row.pathologies)

    image, boxes = utility.resize_with_boxes(image, boxes, max_size=config.FASTER_RCNN_RESIZE)

    if augment:
        image, boxes, class_labels = frc_augment(image, boxes, class_labels)

    return image, boxes, class_labels


class FasterRCNNDataset(Dataset):
    """Breast-masked images with their boxes and classes, as torchvision targets."""

    def __init__(self, dataframe, augment=False):
        self.dataframe = dataframe.reset_index(drop=True)
        self.augment = augment

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]

        image, boxes, class_labels = frc_prepare_image(row, augment=self.augment)

        # Torchvision reserves label 0 for background
        labels = class_labels + 1

        image = torch.from_numpy(image).float().unsqueeze(0).repeat(3, 1, 1)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor(index),
            "area": (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]),
            "iscrowd": torch.zeros(len(boxes), dtype=torch.int64),
        }

        return image, target


def collate_fn(batch):
    return tuple(zip(*batch))


def build_model(num_classes=None, device=None, pretrained=True):
    """Pretrained MobileNetV3-FPN with a fresh box predictor for the lesion classes."""
    device = DEVICE if device is None else device

    num_classes = config.FASTER_RCNN_CLASS_COUNT if num_classes is None else num_classes

    # A checkpoint replaces every weight, so only training needs the pretrained download
    weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT if pretrained else None
    model = fasterrcnn_mobilenet_v3_large_fpn(
        weights=weights, min_size=config.FASTER_RCNN_RESIZE, max_size=config.FASTER_RCNN_MAX_SIZE
    )

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model.to(device)


class FasterRCNN(Detector.Detector):
    """Faster R-CNN detector."""

    model_name = MODEL_NAME
    default_epochs = config.FASTER_RCNN_NUM_EPOCHS
    score_threshold = config.FASTER_RCNN_SCORE_THRESHOLD
    inference_score_threshold = config.FASTER_RCNN_INFERENCE_SCORE_THRESHOLD
    iou_threshold = config.FASTER_RCNN_IOU_THRESHOLD
    stacker_score_threshold = config.FASTER_RCNN_STACKER_SCORE_THRESHOLD
    stacker_top_k = config.FASTER_RCNN_STACKER_TOP_K
    stacker_collect_threshold = config.FASTER_RCNN_STACKER_COLLECT_THRESHOLD
    resume_artifact_keys = ("progress", "state", "best")
    epoch_table_columns = EPOCH_TABLE_COLUMNS
    epoch_table_keys = EPOCH_TABLE_KEYS
    epoch_table_formats = EPOCH_TABLE_FORMATS

    def __init__(self, weights_path=None, device=None):
        super().__init__(weights_path)
        self.device = DEVICE if device is None else device
        self.optimizer = None
        self.scheduler = None
        self.scaler = None

    def prepare_model(self):
        """Build the model whose graph the fold's fingerprint is taken over."""
        self.build()

    def model_signature(self, lesion_type):
        """Parameter and buffer names with their shapes, which is this model's whole graph."""
        return utility.torch_model_signature(self.model)

    def training_config(self, lesion_type, best_epoch=None, best_val_ap50=None):
        """Pipeline settings a checkpoint depends on, so any consumer can reproduce it."""
        return training_config(lesion_type, best_epoch, best_val_ap50)

    def dataset(self, dataframe, augment=False):
        return FasterRCNNDataset(dataframe, augment=augment)

    def loaders(self, dataframe, batch_size=None, shuffle=False, augment=False):
        """One loader over a dataframe; shuffle and augment are training only."""
        batch_size = config.FASTER_RCNN_BATCH_SIZE if batch_size is None else batch_size

        return DataLoader(
            self.dataset(dataframe, augment=augment),
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
            num_workers=0,
        )

    def prepare(self, row, augment=False):
        """Breast-masked image, GT boxes and class labels; augment is training only."""
        return frc_prepare_image(row, augment=augment)

    def build(self, num_classes=None, pretrained=True):
        self.model = build_model(num_classes=num_classes, device=self.device, pretrained=pretrained)
        return self.model

    def summary(self):
        """Parameter counts of the built model, the torch equivalent of model.summary()."""
        counts = {
            "total": sum(parameter.numel() for parameter in self.model.parameters()),
            "trainable": sum(parameter.numel() for parameter in self.model.parameters() if parameter.requires_grad),
        }

        for name, value in counts.items():
            print(f"{name.capitalize():<10}: {value:,}")

        return counts

    def setup_optimizer(self):
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=config.FASTER_RCNN_LR,
            momentum=config.FASTER_RCNN_MOMENTUM,
            weight_decay=config.FASTER_RCNN_WEIGHT_DECAY,
        )

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="max",
            factor=config.LR_FACTOR,
            patience=config.LR_PATIENCE,
            min_lr=config.FASTER_RCNN_MIN_LR,
        )

        self.scaler = torch.amp.GradScaler("cuda", enabled=config.USE_AMP)

        return self.optimizer, self.scheduler, self.scaler

    def train_epoch(self, loader, progress):
        """One epoch of the torchvision multi-task loss; returns the mean of every term."""
        self.model.train()

        totals = {}

        for images, targets in loader:
            images = [image.to(self.device) for image in images]
            targets = [{key: value.to(self.device) for key, value in target.items()} for target in targets]
            self.optimizer.zero_grad()

            with torch.autocast(device_type="cuda", enabled=config.USE_AMP):
                loss_dict = self.model(images, targets)
                loss = sum(loss_dict.values())
                if not torch.isfinite(loss):
                    print(loss_dict)
                    raise RuntimeError("Loss became NaN")

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            totals["loss"] = totals.get("loss", 0.0) + loss.item()
            for key, value in loss_dict.items():
                totals[key] = totals.get(key, 0.0) + value.item()

            progress.update(1)
            progress.set_postfix_str(
                f"Loss: {loss.item():.4f} cls: {loss_dict['loss_classifier'].item():.3f} "
                f"box: {loss_dict['loss_box_reg'].item():.3f} obj: {loss_dict['loss_objectness'].item():.3f} "
                f"rpn: {loss_dict['loss_rpn_box_reg'].item():.3f}"
            )

        return {key: value / len(loader) for key, value in totals.items()}

    def save_checkpoint_state(self, paths, fingerprint, epoch, metrics):
        """Per-epoch state so an interrupted fold resumes; removed once the fold completes."""
        weights = metrics.best_weights
        if metrics.best_epoch == epoch:
            utility.write_atomically(lambda path: torch.save({"model_state_dict": weights}, path), paths["best"])

        progress = {
            "fingerprint": fingerprint,
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
        }
        utility.write_atomically(lambda path: torch.save(progress, path), paths["progress"])

        utility.write_fold_state(
            paths["state"],
            fingerprint,
            epoch=epoch,
            best_epoch=metrics.best_epoch,
            best_ap50=metrics.best_ap50,
            history=metrics.history,
        )

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
        """Fit one model, validating every epoch, and keep the best raw weights."""
        epochs = config.FASTER_RCNN_NUM_EPOCHS if epochs is None else epochs

        train_loader = self.loaders(train_df, shuffle=True, augment=True)

        if resume is None:
            # A fresh fold builds its own optimizer and LR schedule
            self.setup_optimizer()

        resume = {} if resume is None else resume
        initial_epoch = int(resume.get("initial_epoch", 0))

        metrics = utility.FitHistory(
            history=resume.get("history"),
            best_epoch=resume.get("best_epoch", 0),
            best_ap50=resume.get("best_ap50", 0.0),
            best_weights=resume.get("best_weights"),
        )

        table = utility.EpochTable(EPOCH_TABLE_COLUMNS, EPOCH_TABLE_KEYS, EPOCH_TABLE_FORMATS, epochs)

        own_bars = outer is None
        if own_bars:
            outer = tqdm(total=max(epochs - initial_epoch, 0), position=0, desc="Epochs")
            inner = tqdm(total=0, position=1, desc="Initializing...")

        try:
            if initial_epoch < epochs:
                tqdm.write(table.header())

                # Re-print finished epochs so a resumed fold still shows a complete table
                for entry in metrics.history:
                    tqdm.write(table.row(entry))

            for epoch in range(initial_epoch, epochs):
                outer.set_description(f"Epochs {epoch + 1}/{epochs}")

                inner.reset(total=len(train_loader))
                inner.set_description(f"{tag}Epoch {epoch + 1}/{epochs} [Train]")
                losses = self.train_epoch(train_loader, inner)

                row = self.validate(val_df, desc=f"{tag}Epoch {epoch + 1}/{epochs} [Val]", progress=inner)

                self.scheduler.step(row["ap50"])

                class_columns = {
                    f"ap50_{name}": value for name, value in zip(config.CLASS_NAMES, row["ap50_per_class"])
                }

                entry = {
                    "epoch": epoch + 1,
                    "train_loss": losses.pop("loss"),
                    **losses,
                    **{key: value for key, value in row.items() if key != "ap50_per_class"},
                    **class_columns,
                }

                best = {name: value.detach().cpu().clone() for name, value in self.model.state_dict().items()}
                improved = metrics.add(entry, row["ap50"], best)

                tqdm.write(table.row(entry) + (" | Saved best model" if improved else ""))
                outer.update(1)

                if checkpoint_paths is not None:
                    self.save_checkpoint_state(checkpoint_paths, fingerprint, epoch + 1, metrics)
        finally:
            if own_bars:
                inner.close()
                outer.close()

        if metrics.best_weights is not None:
            self.model.load_state_dict(metrics.best_weights)

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

        # The progress checkpoint restores the optimizer and the LR schedule
        checkpoint = torch.load(paths["progress"], map_location=self.device, weights_only=False)

        self.build()
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.setup_optimizer()
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        resume = {
            "initial_epoch": int(state["epoch"]),
            "history": list(state["history"]),
            "best_ap50": float(state["best_ap50"]),
            "best_epoch": int(state["best_epoch"]),
            "best_weights": (
                torch.load(paths["best"], map_location=self.device, weights_only=False)["model_state_dict"]
                if state["best_epoch"]
                else None
            ),
        }

        tqdm.write(
            f"Fold {fold}: resuming at epoch {resume['initial_epoch'] + 1}/{epochs} "
            f"(best val AP50 {resume['best_ap50']:.4f} at epoch {resume['best_epoch']})"
        )

        return resume

    def open_candidate_floor(self):
        """Open the roi_heads score floor, which torchvision applies inside the model."""
        self.model.roi_heads.score_thresh = config.FASTER_RCNN_STACKER_COLLECT_THRESHOLD
        self.model.roi_heads.nms_thresh = 0.5
        self.model.roi_heads.detections_per_img = config.STACKER_CANDIDATE_MAX

    @torch.inference_mode()
    def predict_raw(self, dataframe, score_threshold=None, batch_size=None, desc="predict", progress=None):
        """Every detection above the score threshold, per image, with its ground truth."""
        score_threshold = config.FASTER_RCNN_SCORE_THRESHOLD if score_threshold is None else score_threshold

        if self.model is None:
            raise ValueError("Call build() or load() before predicting.")

        dataframe = dataframe.reset_index(drop=True)
        loader = self.loaders(dataframe, batch_size=batch_size)

        self.model.eval()

        local_bar = progress is None
        if local_bar:
            progress = tqdm(total=len(loader), desc=desc)

        raw = []

        try:
            progress.reset(total=len(loader))
            progress.set_description(desc)

            for images, batch_targets in loader:
                outputs = self.model([image.to(self.device) for image in images])

                for image, output, target in zip(images, outputs, batch_targets):
                    keep = output["scores"].cpu().numpy() >= score_threshold

                    raw.append(
                        {
                            "image_shape": tuple(image.shape[-2:]),
                            "boxes": output["boxes"].cpu().numpy()[keep],
                            "scores": output["scores"].cpu().numpy()[keep],
                            # Torchvision labels background as 0, the shared schema does not
                            "labels": (output["labels"][keep] - 1).cpu().numpy(),
                            "class_scores": output["scores"].cpu().numpy()[keep],
                            "gt_boxes": target["boxes"].numpy(),
                            "gt_labels": (target["labels"] - 1).numpy(),
                        }
                    )

                progress.update(1)
        finally:
            if local_bar:
                progress.close()

        return raw

    def save(self, path, fingerprint=None, epoch=0, ap50=0.0, history=None, lesion_type="mass"):
        """Weights plus the settings and score they belong to."""
        torch.save(
            {
                "epoch": epoch,
                "ap50": ap50,
                "history": history,
                "fingerprint": fingerprint,
                "model_state_dict": self.model.state_dict(),
                "config": self.training_config(lesion_type, epoch, ap50),
            },
            path,
        )
        return Path(path)

    def load(self, path=None):
        """Restore the model (and optimizer, when present) from a checkpoint."""
        path = Path(path) if path is not None else self.weights_path
        if path is None:
            raise ValueError("No checkpoint path given.")

        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        if self.model is None:
            self.build(pretrained=False)

        self.model.load_state_dict(checkpoint["model_state_dict"])

        if self.optimizer is not None and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        self.weights_path = path

        return checkpoint

    @torch.inference_mode()
    def predict_image(self, row, score_threshold=None):
        """Full pipeline for one mammogram: image, GT boxes and labels, then predictions."""
        score_threshold = config.FASTER_RCNN_INFERENCE_SCORE_THRESHOLD if score_threshold is None else score_threshold

        if self.model is None:
            raise ValueError("Call build() or load() before predicting.")

        image, gt_boxes, gt_labels = frc_prepare_image(row)

        tensor = torch.from_numpy(image).float().unsqueeze(0).repeat(3, 1, 1)

        self.model.eval()
        output = self.model([tensor.to(self.device)])[0]

        keep = output["scores"].cpu().numpy() >= score_threshold

        return (
            image,
            gt_boxes,
            gt_labels,
            output["boxes"].cpu().numpy()[keep],
            output["scores"].cpu().numpy()[keep],
            (output["labels"][keep] - 1).cpu().numpy(),
        )

    @torch.inference_mode()
    def proposals(self, row):
        """Region proposals and the finest FPN feature map for one mammogram.

        The proposals come out of the RPN before the ROI head classifies or refines them, so
        they are the candidate set the second stage is allowed to look at.
        """
        if self.model is None:
            raise ValueError("Call build() or load() before predicting.")

        image, gt_boxes, gt_labels = frc_prepare_image(row)
        tensor = torch.from_numpy(image).float().unsqueeze(0).repeat(3, 1, 1).to(self.device)

        self.model.eval()

        images, _ = self.model.transform([tensor], None)
        features = self.model.backbone(images.tensors)
        proposals, _ = self.model.rpn(images, features)

        finest = features[max(features, key=lambda level: features[level].shape[-1])]

        return image, gt_boxes, gt_labels, proposals[0].cpu().numpy(), finest.mean(dim=1)[0].cpu().numpy()
