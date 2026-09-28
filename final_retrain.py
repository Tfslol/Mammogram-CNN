"""Full-trainval retrain of one model, writing a `_full` checkpoint beside the fold artifacts."""

import argparse

import pandas as pd

import config
import utility
import cnn_detector
import faster_rcnn_detector
import yolo_detector
import stacker
import stacker_set

LESION_TYPE = "mass"

# Written beside the fold checkpoints; the fold artifacts are never overwritten
OUTPUTS = {
    "cnn": config.MODELS_ROOT / f"{LESION_TYPE}_cnn_full.keras",
    "faster_rcnn": config.MODELS_ROOT / f"{LESION_TYPE}_faster_rcnn_full.pt",
    "yolo": config.MODELS_ROOT / f"{LESION_TYPE}_yolo_full.pt",
    "stacker_set": config.MODELS_ROOT / f"{LESION_TYPE}_stacker_set_full.keras",
    "stacker_grid": config.MODELS_ROOT / f"{LESION_TYPE}_stacker_grid_full.keras",
}

EPOCH_DEFAULTS = {
    "cnn": config.CNN_NUM_EPOCHS,
    "faster_rcnn": config.FASTER_RCNN_NUM_EPOCHS,
    "yolo": config.YOLO_EPOCHS,
    "stacker_set": config.SET_NUM_EPOCHS,
    "stacker_grid": config.STACKER_NUM_EPOCHS,
}

GRID_ARCHITECTURE = stacker.architecture_defaults() | {"objectness_prior": True}
YOLO_FULL_RUN = config.YOLO_PROJECT / f"{LESION_TYPE}_yolo_full"


def load_splits():
    """All of trainval for training, fold 0 again as the monitored column."""
    metadata = pd.read_csv(config.PROJECT_ROOT / f"{LESION_TYPE}_metadata.csv")
    groups = utility.group_mammograms(metadata)
    trainval_df = groups[groups["split"] == "trainval"].reset_index(drop=True)

    return trainval_df, trainval_df[trainval_df["fold"] == config.VAL_FOLD].reset_index(drop=True)


def retrain_keras(detector, trainval_df, val_df, epochs, output):
    """CNN and Faster R-CNN share a training signature, so one path serves both."""
    utility.set_deterministic()
    detector.prepare_model()
    fingerprint = detector.fingerprint(LESION_TYPE)

    print(f"Training on {len(trainval_df):,} mammograms for {epochs} epochs, fingerprint {fingerprint}", flush=True)

    result = detector.train(trainval_df, val_df, epochs=epochs, tag="Full ")
    detector.save_fold_checkpoint(output, LESION_TYPE, result, fingerprint)

    return result


def retrain_yolo(detector, trainval_df, val_df, epochs, output):
    """YOLO reads its dataset from disk, so all of trainval is exported as the train split."""
    import shutil

    utility.set_deterministic()

    detector.export(trainval_df, "train")
    detector.export(val_df, "val")
    detector.write_data_yaml()

    print(f"Exported {len(trainval_df):,} train and {len(val_df):,} val mammograms", flush=True)

    result = detector.train(trainval_df, val_df, epochs=epochs, tag="Full ", checkpoint_paths={"run": YOLO_FULL_RUN})

    shutil.copy2(YOLO_FULL_RUN / "weights" / "last.pt", output)
    detector.weights_path = output

    # The exported tree holds all of trainval as "train", so the fold marker is cleared
    (detector.root / f"{LESION_TYPE}_export.json").unlink(missing_ok=True)

    return result


def retrain_grid_stacker(trainval_df, val_df, epochs, output):
    """The grid stacker on the pooled out-of-fold candidate feed, over the whole trainval split."""
    utility.set_deterministic()

    detector = stacker.Stacker(GRID_ARCHITECTURE, LESION_TYPE, tag="gridfull")
    detector.prepare_fold_data(trainval_df, config.VAL_FOLD, LESION_TYPE)
    detector.prepare_model()

    print(
        f"Training the grid stacker on {len(trainval_df):,} mammograms for {epochs} epochs, "
        f"{len(detector.grid):,} input grids, {detector.model.count_params():,} parameters",
        flush=True,
    )

    result = detector.train(trainval_df, val_df, epochs=epochs, tag="Full ")
    detector.model.save(output)

    return result


def retrain_set_stacker(trainval_df, val_df, epochs, output):
    """The set stacker on the pooled out-of-fold candidate feed, over the whole trainval split."""
    utility.set_deterministic()

    detector = stacker_set.StackerSet(stacker_set.architecture_defaults(), LESION_TYPE, tag="full")
    detector.tokens = stacker_set.build_tokens(trainval_df, LESION_TYPE, config.OOF_FOLDS)
    detector.prepare_model()

    print(
        f"Training the set stacker on {len(trainval_df):,} mammograms for {epochs} epochs, "
        f"{len(detector.tokens):,} token sets, {detector.model.count_params():,} parameters",
        flush=True,
    )

    result = detector.train(trainval_df, val_df, epochs=epochs, tag="Full ")
    detector.model.save(output)

    return result


def train(model, epochs=None):
    """Train one model on all of trainval and return (checkpoint path, TrainResult)."""
    epochs = EPOCH_DEFAULTS[model] if epochs is None else int(epochs)
    output = OUTPUTS[model]

    trainval_df, val_df = load_splits()
    print(f"{model}: {len(trainval_df):,} trainval mammograms, epochs {epochs}", flush=True)

    if model == "cnn":
        result = retrain_keras(cnn_detector.CNN(), trainval_df, val_df, epochs, output)
    elif model == "faster_rcnn":
        result = retrain_keras(faster_rcnn_detector.FasterRCNN(), trainval_df, val_df, epochs, output)
    elif model == "yolo":
        result = retrain_yolo(yolo_detector.YOLODetector(lesion_type=LESION_TYPE), trainval_df, val_df, epochs, output)
    elif model == "stacker_grid":
        result = retrain_grid_stacker(trainval_df, val_df, epochs, output)
    else:
        result = retrain_set_stacker(trainval_df, val_df, epochs, output)

    return output, result


def load_or_train(model, epochs=None, force=False):
    """The full-trainval checkpoint of one model, trained only when it is not already on disk."""
    output = OUTPUTS[model]

    if output.exists() and not force:
        print(f"{model}: {output.name} is already on disk, not retraining", flush=True)
        return output

    path, result = train(model, epochs)

    print(
        f"{model}: {path.name}, {path.stat().st_size / 1e6:.1f} MB, "
        f"selected epoch {result.best_epoch} (training curve, not held out), "
        f"{len(result.history)} history rows",
        flush=True,
    )

    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=tuple(OUTPUTS))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()

    load_or_train(arguments.model, arguments.epochs, arguments.force)


if __name__ == "__main__":
    main()
