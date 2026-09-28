"""Held-out test evaluation of the pipeline."""

import argparse
import os

import numpy as np
import pandas as pd

import config
import utility
import Detector
import cnn_detector
import faster_rcnn_detector
import yolo_detector
import stacker
import stacker_set

LESION_TYPE = "mass"
BASE = stacker.BASE_MODELS
BUILDERS = {
    "cnn": cnn_detector.CNN,
    "faster_rcnn": faster_rcnn_detector.FasterRCNN,
    "yolo": lambda: yolo_detector.YOLODetector(lesion_type=LESION_TYPE),
}

GRID_TAG = "prior"
GRID_ARCHITECTURE = stacker.architecture_defaults() | {"objectness_prior": True}
GRID_NAME = f"{stacker.MODEL_NAME}_{GRID_TAG}"

SET_TAG = "w128"
SET_ARCHITECTURE = stacker_set.architecture_defaults() | {"width": 128}
SET_NAME = f"{stacker_set.MODEL_NAME}_{SET_TAG}"

FULL_PATHS = {
    "cnn": config.MODELS_ROOT / f"{LESION_TYPE}_cnn_full.keras",
    "faster_rcnn": config.MODELS_ROOT / f"{LESION_TYPE}_faster_rcnn_full.pt",
    "yolo": config.MODELS_ROOT / f"{LESION_TYPE}_yolo_full.pt",
}

# Both full-trainval stackers were trained on the pooled out-of-fold candidate feed
FULL_GRID_STACKER = config.MODELS_ROOT / f"{LESION_TYPE}_stacker_grid_full.keras"
FULL_SET_STACKER = config.MODELS_ROOT / f"{LESION_TYPE}_stacker_set_full.keras"
FULL_GRID_NAME = "stacker_grid_full"
FULL_NAME = "stacker_set_full"

LABELS = stacker.BASE_LABELS | {
    GRID_NAME: "Stacker (grid)",
    SET_NAME: "Stacker (set, 128)",
    FULL_GRID_NAME: "Stacker (grid, full)",
    FULL_NAME: "Stacker (set, full)",
}
THRESHOLDS = {
    "cnn": config.CNN_INFERENCE_SCORE_THRESHOLD,
    "faster_rcnn": config.FASTER_RCNN_INFERENCE_SCORE_THRESHOLD,
    "yolo": config.YOLO_INFERENCE_SCORE_THRESHOLD,
    GRID_NAME: config.STACKER_INFERENCE_SCORE_THRESHOLD,
    SET_NAME: config.STACKER_INFERENCE_SCORE_THRESHOLD,
    FULL_GRID_NAME: config.STACKER_INFERENCE_SCORE_THRESHOLD,
    FULL_NAME: config.STACKER_INFERENCE_SCORE_THRESHOLD,
}

for name in (GRID_NAME, SET_NAME):
    config.FOLD_LAYOUT.setdefault(name, (config.STACKER_WORKSPACE, ".keras"))


def load_test():
    """Test mammograms, their ground truth in the shared canvas frame, and the folds to run."""
    metadata = pd.read_csv(config.PROJECT_ROOT / f"{LESION_TYPE}_metadata.csv")
    groups = utility.group_mammograms(metadata)
    test_df = groups[groups["split"] == "test"].reset_index(drop=True)

    # A short smoke run: FINAL_EVAL_LIMIT=8 checks the plumbing without the full 320 images
    limit = int(os.environ.get("FINAL_EVAL_LIMIT", 0))
    folds = config.OOF_FOLDS[:1] if limit else config.OOF_FOLDS
    test_df = test_df.head(limit).reset_index(drop=True) if limit else test_df

    ids = [utility.mammogram_id(path) for path in test_df["full_image"]]
    ground_truth = {}

    for row in test_df.itertuples():
        boxes, labels, _ = stacker.ground_truth(row)
        ground_truth[utility.mammogram_id(row.full_image)] = {"gt_boxes": boxes, "gt_labels": labels}

    print(
        f"Test mammograms: {len(test_df):,}, "
        f"{sum(len(entry['gt_boxes']) for entry in ground_truth.values())} lesions, folds {list(folds)}",
        flush=True,
    )

    return test_df, ids, ground_truth, folds


def base_feed(detector, test_df, tag, progress=None):
    """One detector's test records and its ranked candidate feed."""
    records = detector.detections(test_df, desc=f"{tag} records", progress=progress)
    candidates = detector.detections(
        test_df,
        score_threshold=detector.stacker_collect_threshold or detector.stacker_score_threshold,
        top_k=config.STACKER_CANDIDATE_MAX,
        rank=True,
        desc=f"{tag} candidates",
        progress=progress,
    )

    return records, candidates


def feed_table(rows):
    """One line per detector pass: the records it produced and the candidates it fed on."""
    table = pd.DataFrame(rows)

    print(
        table.to_string(index=False, formatters={column: "{:,}".format for column in ("records", "candidates")}),
        flush=True,
    )


def lookup_from(frames, ids):
    """{image_id: {model: frame}} from one frame per model."""
    lookup = {image_id: {} for image_id in ids}

    for model, frame in frames.items():
        for image_id, group in frame.groupby("image_id", sort=False):
            lookup[image_id][model] = group

    return lookup


def row_for(model, raw, threshold):
    """One row of the report from a model's raw detections."""
    metrics = Detector.detection_metrics(raw, config.CNN_IOU_THRESHOLD, threshold)

    return {
        "model": LABELS[model],
        "AP50": metrics["ap50"],
        "mAP50": metrics["ap50_macro"],
        "P": metrics["precision"],
        "R": metrics["recall"],
        "F1": metrics["f1"],
        "Det": metrics["det_per_image"],
    }


def pool_folds(by_fold, ids, ground_truth):
    """Every fold's held-out predictions on one mammogram, duplicates suppressed."""
    raw = []

    for image_id in ids:
        entries = [frame[image_id] for frame in by_fold if len(frame[image_id]["boxes"])]

        if entries:
            boxes = np.concatenate([entry["boxes"] for entry in entries])
            scores = np.concatenate([entry["scores"] for entry in entries])
            labels = np.concatenate([entry["labels"] for entry in entries])
            class_scores = np.concatenate([entry["class_scores"] for entry in entries])

            keep = cnn_detector.cnn_nms(
                boxes,
                scores,
                iou_threshold=config.CNN_NMS_THRESHOLD,
                containment_threshold=config.CNN_NMS_CONTAINMENT,
            )
            boxes, scores, labels, class_scores = boxes[keep], scores[keep], labels[keep], class_scores[keep]
        else:
            boxes = np.zeros((0, 4), np.float32)
            scores = np.zeros(0, np.float32)
            labels = np.zeros(0, np.int64)
            class_scores = np.zeros(0, np.float32)

        raw.append(
            {
                "image_shape": (config.CNN_IMAGE_SIZE, config.CNN_IMAGE_SIZE),
                "boxes": boxes,
                "scores": scores,
                "labels": labels,
                "class_scores": class_scores,
                "gt_boxes": ground_truth[image_id]["gt_boxes"],
                "gt_labels": ground_truth[image_id]["gt_labels"],
            }
        )

    return raw


def raw_folds(test_df, ids, ground_truth, folds, progress=None):
    """Per-model detections of the five out-of-fold models on the test split."""
    per_fold = {model: [] for model in list(BASE) + [GRID_NAME, SET_NAME]}

    for fold in folds:
        print(f"=== fold {fold} ===", flush=True)

        records, candidates = {}, {}
        feed = []

        for model in BASE:
            detector = BUILDERS[model]()
            detector.load(config.fold_paths(model, LESION_TYPE, fold)["model"])
            detector.open_candidate_floor()
            records[model], candidates[model] = base_feed(detector, test_df, f"fold {fold} {LABELS[model]}", progress)
            feed.append(
                {
                    "fold": fold,
                    "detector": LABELS[model],
                    "records": len(records[model]),
                    "candidates": len(candidates[model]),
                }
            )

        # Per fold, so a run that takes minutes a fold reports each detector as it finishes
        feed_table(feed)

        grid = stacker.build_grid(test_df, LESION_TYPE, folds, lookup=lookup_from(records, ids))
        tokens = stacker_set.build_tokens(
            test_df, LESION_TYPE, folds, top_k=SET_ARCHITECTURE["k"], lookup=lookup_from(candidates, ids)
        )

        grid_stacker = stacker.Stacker(GRID_ARCHITECTURE, LESION_TYPE, tag=GRID_TAG)
        grid_stacker.grid = grid
        grid_stacker.load(config.fold_paths(GRID_NAME, LESION_TYPE, fold)["model"])

        set_stacker = stacker_set.StackerSet(SET_ARCHITECTURE, LESION_TYPE, tag=SET_TAG)
        set_stacker.tokens = tokens
        set_stacker.load(config.fold_paths(SET_NAME, LESION_TYPE, fold)["model"])

        produced = {
            **{model: stacker.base_raw(records[model], ground_truth, ids) for model in BASE},
            GRID_NAME: grid_stacker.predict_raw(test_df, desc=f"fold {fold} {LABELS[GRID_NAME]}", progress=progress),
            SET_NAME: set_stacker.predict_raw(test_df),
        }

        for model, raw in produced.items():
            per_fold[model].append({image_id: entry for image_id, entry in zip(ids, raw)})

    pooled = {model: pool_folds(by_fold, ids, ground_truth) for model, by_fold in per_fold.items()}

    return pooled, per_fold


def evaluate_folds(test_df, ids, ground_truth, folds):
    """The pooled report: one row per model, with the spread across folds printed beside it."""
    pooled, per_fold = raw_folds(test_df, ids, ground_truth, folds)

    rows = []

    for model, raw in pooled.items():
        ap50 = [
            Detector.detection_metrics(
                [frame[image_id] for image_id in ids], config.CNN_IOU_THRESHOLD, THRESHOLDS[model]
            )["ap50"]
            for frame in per_fold[model]
        ]

        row = row_for(model, raw, THRESHOLDS[model])
        row["fold AP50 mean"] = float(np.mean(ap50))
        row["fold AP50 std"] = float(np.std(ap50))
        rows.append(row)

    return pd.DataFrame(rows).set_index("model")


def raw_full(test_df, ids, ground_truth, folds, progress=None):
    """Per-model detections of the five full-trainval models on the test split."""
    missing = [path.name for path in [*FULL_PATHS.values(), FULL_GRID_STACKER, FULL_SET_STACKER] if not path.exists()]

    if missing:
        raise SystemExit(f"Run final_retrain.py first; missing {', '.join(missing)}")

    records, candidates = {}, {}
    feed = []

    for model in BASE:
        detector = BUILDERS[model]()
        detector.load(FULL_PATHS[model])
        detector.open_candidate_floor()
        records[model], candidates[model] = base_feed(detector, test_df, f"full {LABELS[model]}", progress)
        feed.append({"detector": LABELS[model], "records": len(records[model]), "candidates": len(candidates[model])})

    feed_table(feed)

    grid_stacker = stacker.Stacker(GRID_ARCHITECTURE, LESION_TYPE, tag="gridfull")
    grid_stacker.grid = stacker.build_grid(test_df, LESION_TYPE, folds, lookup=lookup_from(records, ids))
    grid_stacker.load(FULL_GRID_STACKER)

    set_stacker = stacker_set.StackerSet(stacker_set.architecture_defaults(), LESION_TYPE, tag="full")
    set_stacker.tokens = stacker_set.build_tokens(
        test_df, LESION_TYPE, folds, top_k=SET_ARCHITECTURE["k"], lookup=lookup_from(candidates, ids)
    )
    set_stacker.load(FULL_SET_STACKER)

    return {
        **{model: stacker.base_raw(records[model], ground_truth, ids) for model in BASE},
        FULL_GRID_NAME: grid_stacker.predict_raw(test_df, desc=f"full {LABELS[FULL_GRID_NAME]}", progress=progress),
        FULL_NAME: set_stacker.predict_raw(test_df),
    }


def evaluate_full(test_df, ids, ground_truth, folds):
    """The full-trainval report: one row per model."""
    produced = raw_full(test_df, ids, ground_truth, folds)

    rows = [row_for(model, raw, THRESHOLDS[model]) for model, raw in produced.items()]

    return pd.DataFrame(rows).set_index("model")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="score the full-trainval retrain instead")
    arguments = parser.parse_args()

    test_df, ids, ground_truth, folds = load_test()
    table = (
        evaluate_full(test_df, ids, ground_truth, folds)
        if arguments.full
        else evaluate_folds(test_df, ids, ground_truth, folds)
    )

    print()
    print(table.round(4).to_string())


if __name__ == "__main__":
    main()
