"""Train set-stacker variants over the out-of-fold split and report the pooled metrics.

Usage: python set_variants.py <tag> [KEY=VALUE ...] [arch.KEY=VALUE ...]. `KEY=VALUE` overrides
a `config` constant for this process only; `arch.KEY=VALUE` overrides the architecture dict.
"""

import sys
import time

import numpy as np
import pandas as pd

import config
import utility
import Detector
import stacker
import stacker_set

LESION_TYPE = "mass"
FOLDS = config.OOF_FOLDS


def apply_overrides(pairs):
    """Process-local config and architecture overrides from `KEY=VALUE` arguments."""
    config_overrides, arch_overrides = {}, {}

    for pair in pairs:
        key, _, value = pair.partition("=")

        if key.startswith("arch."):
            arch_overrides[key[5:]] = eval(value)
        else:
            config_overrides[key] = eval(value)

    for key, value in config_overrides.items():
        setattr(config, key, value)

    return config_overrides, arch_overrides


def score(name, trainval_df, gt):
    """Pooled out-of-fold metrics, oracle recall and record count of one finished variant."""
    frame = stacker.base_records(LESION_TYPE, name, FOLDS)
    image_ids = [utility.mammogram_id(path) for path in trainval_df["full_image"]]

    raw = stacker.base_raw(frame, gt, image_ids)
    metrics = Detector.detection_metrics(raw, config.CNN_IOU_THRESHOLD, config.STACKER_INFERENCE_SCORE_THRESHOLD)

    return metrics, stacker.oracle_recall(frame, gt, image_ids)


def main(tag, pairs):
    applied, overrides = apply_overrides(pairs)

    metadata = pd.read_csv(config.PROJECT_ROOT / f"{LESION_TYPE}_metadata.csv")
    groups = utility.group_mammograms(metadata)
    trainval_df = groups[groups["split"] == "trainval"].reset_index(drop=True)

    architecture = stacker_set.architecture_defaults() | overrides
    name = f"{stacker_set.MODEL_NAME}_{tag}"
    config.FOLD_LAYOUT.setdefault(name, (config.STACKER_WORKSPACE, ".keras"))

    print(f"=== {tag}: {architecture} overrides {applied} ===", flush=True)

    tokens = stacker_set.build_tokens(
        trainval_df,
        LESION_TYPE,
        FOLDS,
        top_k=architecture["k"],
        agreement=architecture.get("agreement", False),
    )
    gt = {
        image_id: {"gt_boxes": entry["gt_boxes"], "gt_labels": entry["gt_labels"]} for image_id, entry in tokens.items()
    }

    detector = stacker_set.StackerSet(architecture, LESION_TYPE, tag=tag)
    detector.tokens = tokens
    detector.prepare_model()

    started = time.time()

    for fold in FOLDS:
        detector.fit_fold(trainval_df, fold, LESION_TYPE, epochs=config.SET_NUM_EPOCHS, force=True)

    detector.export_folds(trainval_df, LESION_TYPE, folds=FOLDS, force=True)

    metrics, oracle = score(name, trainval_df, gt)
    manifest = utility.read_oof_manifest(config.oof_manifest_path(name, LESION_TYPE))
    per_fold = [manifest["folds"][str(fold)]["best_val_ap50"] for fold in FOLDS]

    print(
        f"RESULT {tag} params={detector.model.count_params():,} "
        f"AP50={metrics['ap50']:.4f} mAP50={metrics['ap50_macro']:.4f} P={metrics['precision']:.4f} "
        f"R={metrics['recall']:.4f} F1={metrics['f1']:.4f} Det={metrics['det_per_image']:.2f} "
        f"oracle={oracle:.4f} foldAP50={[round(value, 4) for value in per_fold]} "
        f"secs={time.time() - started:.0f}",
        flush=True,
    )


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
