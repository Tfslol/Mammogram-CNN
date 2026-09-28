"""Train the stacker baseline and its architecture variants, one job group per process.

Usage: python stack_variants.py <job> [...], where a job is a variant key and `<variant>:all`
walks every fold.
"""

import sys

import pandas as pd

import config
import utility
import stacker

# Architecture overrides against stacker.architecture_defaults()
VARIANTS = {
    "narrow": {"anchor_width": False},
    "stages1": {"stages": 1},
    "stages3": {"stages": 3},
    "classgate": {"class_stop_gradient": True},
    "head64": {"head_filters": 64},
    "doub": {"double": True},
    "noproj": {"projection": False},
    "prior": {"objectness_prior": True},
    "prior_sup": {"objectness_prior": True, "candidate_supervision": True},
    "prior_noproj": {"objectness_prior": True, "projection": False},
}

EPOCHS = 20


def trainval_frame():
    metadata = pd.read_csv(config.PROJECT_ROOT / "mass_metadata.csv")
    groups = utility.group_mammograms(metadata)

    return groups[groups["split"] == "trainval"].reset_index(drop=True)


def main(jobs):
    trainval_df = trainval_frame()

    for job in jobs:
        name, _, scope = job.partition(":")

        if name not in VARIANTS:
            raise SystemExit(f"Unknown job: {job}")

        if scope == "all":
            folds = tuple(range(config.N_FOLDS))
        elif scope:
            folds = tuple(int(part) for part in scope.split(","))
        else:
            folds = (0,)

        for fold in folds:
            print(f"=== {job} fold {fold} ===", flush=True)
            stacker.run_variant(
                trainval_df,
                "mass",
                stacker.architecture_defaults() | VARIANTS[name],
                tag=name,
                fold=fold,
                epochs=EPOCHS,
            )

    print("ALL JOBS DONE", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
