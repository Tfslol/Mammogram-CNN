from pathlib import Path

# Splitting
RANDOM_STATE = 42
TEST_SIZE = 0.20
N_FOLDS = 5
VAL_FOLD = 0

# Out-of-fold artifacts cover trainval only; set to (0,) to run a single fold
OOF_FOLDS = tuple(range(N_FOLDS))

# Absolute root so generated paths do not depend on the working directory
PROJECT_ROOT = Path(__file__).resolve().parent

# Dataset
DATASET_ROOT = PROJECT_ROOT / "dataset"
DOWNLOADS_FOLDER = Path("~/Downloads").expanduser()
ORIGINAL_DATASET_ROOT = DOWNLOADS_FOLDER / "cbis_ddsm"
CSV_FILES = (
    DOWNLOADS_FOLDER / "calc_case_description_test_set.csv",
    DOWNLOADS_FOLDER / "calc_case_description_train_set.csv",
    DOWNLOADS_FOLDER / "mass_case_description_test_set.csv",
    DOWNLOADS_FOLDER / "mass_case_description_train_set.csv",
)

MASK_ALIGN_TOLERANCE = 0.15

MAX_IMAGE_SIDE = 1024

# Lesion classes, shared by every detector; background is never a lesion class
CLASS_NAMES = ("benign", "malignant")
CLASS_LABELS = {"BENIGN": 0, "BENIGN_WITHOUT_CALLBACK": 0, "MALIGNANT": 1}
CLASS_COUNT = len(CLASS_NAMES)

# Output locations
MODELS_ROOT = PROJECT_ROOT / "models"

CNN_WORKSPACE = PROJECT_ROOT / "01_cnn_workspace"
FASTER_RCNN_WORKSPACE = PROJECT_ROOT / "02_faster_rcnn_workspace"
YOLO_WORKSPACE = PROJECT_ROOT / "03_yolo_workspace"
STACKER_WORKSPACE = PROJECT_ROOT / "04_stacker_workspace"

# Training loop
USE_AMP = True

# Read by the Faster R-CNN LR plateau; the CNN runs its own warmup and cosine schedule
LR_PATIENCE = 4
LR_FACTOR = 0.5

# Faster R-CNN
FASTER_RCNN_BATCH_SIZE = 16
FASTER_RCNN_NUM_EPOCHS = 50
FASTER_RCNN_LR = 1e-3
FASTER_RCNN_MIN_LR = 5e-6

FASTER_RCNN_SCORE_THRESHOLD = 0.05
FASTER_RCNN_INFERENCE_SCORE_THRESHOLD = 0.5
FASTER_RCNN_IOU_THRESHOLD = 0.5

FASTER_RCNN_WEIGHT_DECAY = 5e-4
FASTER_RCNN_MOMENTUM = 0.9

FASTER_RCNN_RESIZE = 768
FASTER_RCNN_MAX_SIZE = FASTER_RCNN_RESIZE

# Training augmentation, train split only; mirrors the CNN's.
FASTER_RCNN_AUGMENT_FLIP_PROBABILITY = 0.5
FASTER_RCNN_AUGMENT_ROTATION_DEGREES = 10.0
FASTER_RCNN_AUGMENT_SCALE_RANGE = (0.8, 1.25)
FASTER_RCNN_AUGMENT_TRANSLATE_FRACTION = 0.1
FASTER_RCNN_AUGMENT_GAMMA_RANGE = (0.75, 1.25)

# Torchvision reserves index 0 for background, so the head is one class wider
FASTER_RCNN_CLASS_COUNT = CLASS_COUNT + 1

# Stacker export: a low floor plus a per-image top-K, mirroring the CNN's export
FASTER_RCNN_STACKER_SCORE_THRESHOLD = 0.05
FASTER_RCNN_STACKER_TOP_K = 100

# Candidate feed for the stacker: a deeper floor applied to roi_heads at export time
FASTER_RCNN_STACKER_COLLECT_THRESHOLD = 0.0001

# YOLO
YOLO_MODEL = "yolo26s.pt"

YOLO_IMAGE_SIZE = 768
YOLO_BATCH_SIZE = 12

YOLO_EPOCHS = 50
YOLO_WORKERS = 2

YOLO_PATIENCE = 15

YOLO_OPTIMIZER = "AdamW"
YOLO_LR = 2e-3
YOLO_WEIGHT_DECAY = 5e-4

# Score floor for the PR curve, operating threshold and evaluation IoU.
YOLO_SCORE_THRESHOLD = 0.05
YOLO_INFERENCE_SCORE_THRESHOLD = 0.25
YOLO_IOU_THRESHOLD = 0.5

# Flip and rotation shared with the CNN and Faster R-CNN; mosaic, scale and hsv stay on
YOLO_AUGMENT_FLIP_PROBABILITY = 0.5
YOLO_AUGMENT_ROTATION_DEGREES = 10.0

YOLO_COS_LR = True

# Stacker export: a low floor plus a per-image top-K, mirroring the CNN's export
YOLO_STACKER_SCORE_THRESHOLD = 0.05
YOLO_STACKER_TOP_K = 100

# Candidate feed for the stacker, passed to ultralytics as its conf
YOLO_STACKER_COLLECT_THRESHOLD = 0.0005

# Must stay absolute: ultralytics resolves relative project paths under its runs_dir
YOLO_PROJECT = YOLO_WORKSPACE / "runs"
YOLO_RUN_NAME = "mass_yolo"

# CNN (dense objectness head) on a fixed square canvas, resized on the long side then padded
CNN_IMAGE_SIZE = 768

# Backbone output stride, which is also the anchor grid spacing.
CNN_FEATURE_STRIDE = 8

# Anchor sizes in px on the canvas, from the mass lesion max-side percentiles of the training split
CNN_ANCHOR_SIZES = (24, 32, 48, 80)

# Square anchors: extra ratios measured 95.7% coverage vs 95.2%, not worth 3x anchors.
CNN_ANCHOR_ASPECT_RATIOS = (1.0,)

# Positive at or above the first IoU, negative below the second, -1 in between and outside the breast
CNN_ANCHOR_IOU_POSITIVE = 0.5
CNN_ANCHOR_IOU_NEGATIVE = 0.3
CNN_MASK_MIN_FRACTION = 0.5

# Training augmentation, applied only while building training targets.
CNN_AUGMENT_FLIP_PROBABILITY = 0.5
CNN_AUGMENT_ROTATION_DEGREES = 10.0

# Scale and translation, narrower than YOLO's: a full mammogram has no mosaic to restore context
CNN_AUGMENT_SCALE_RANGE = (0.8, 1.25)
CNN_AUGMENT_TRANSLATE_FRACTION = 0.1

# Gamma range, widened to cover the brightness role YOLO gives hsv_v
CNN_AUGMENT_GAMMA_RANGE = (0.75, 1.25)

CNN_BATCH_SIZE = 4
CNN_NUM_EPOCHS = 50
CNN_LR = 1e-3
CNN_MIN_LR = 1e-5
CNN_WEIGHT_DECAY = 5e-4

# Regularisation: L2 on the trunk and fork conv kernels, 0 disables it
CNN_KERNEL_L2 = 1e-4

# "batch" or "group"; the group count must divide every trunk filter count (32, 64, 128)
CNN_NORM = "group"
CNN_GROUP_NORM_GROUPS = 16

# Class CE target smoothing; 0 disables it
CNN_LABEL_SMOOTHING = 0.0

# AdamW decays every trainable variable by default, including norm scale/shift and biases.
CNN_WEIGHT_DECAY_EXCLUDE = ("bias", "gamma", "beta")

# Class weight ramps from START to the full value over RAMP_EPOCHS, then holds
CNN_CLASS_LOSS_WEIGHT = 21.0
CNN_CLASS_LOSS_WEIGHT_START = 1.0
CNN_CLASS_LOSS_RAMP_EPOCHS = 5

# Stride-4 lateral input for the class head, gated so the class loss cannot reach the trunk
CNN_CLASS_FORK_FILTERS = 64

# Linear warmup then cosine decay down to CNN_MIN_LR
CNN_WARMUP_EPOCHS = 1

# Objectness samples every positive plus CNN_NEGATIVE_RATIO negatives, capped
CNN_NEGATIVE_RATIO = 20
CNN_MAX_NEGATIVES = 2048

# Box term weight, relative to the per-sampled-anchor objectness term
CNN_BOX_LOSS_WEIGHT = 1.0

# Inference / evaluation
CNN_SCORE_THRESHOLD = 0.05
CNN_INFERENCE_SCORE_THRESHOLD = 0.5
CNN_IOU_THRESHOLD = 0.5
CNN_NMS_THRESHOLD = 0.5

# Drop a box when this fraction of its area sits inside a higher scoring kept box
CNN_NMS_CONTAINMENT = 0.5

# Stacker export: a low floor plus a per-image top-K, separate from the operating threshold
CNN_STACKER_SCORE_THRESHOLD = 0.05
CNN_STACKER_TOP_K = 100

# Candidate feed for the stacker: the CNN's decode floor, below its 50th candidate
CNN_STACKER_COLLECT_THRESHOLD = 0.002

# Stacker: fuses the three detectors' out-of-fold candidates on the anchor grid
STACKER_BASE_MODELS = ("cnn", "faster_rcnn", "yolo")

# Width of the fusion trunk per anchor; STACKER_ANCHOR_WIDTH False folds the anchors away
STACKER_FILTERS = 64
STACKER_STAGES = 2
STACKER_HEAD_FILTERS = 32
STACKER_KERNEL_SIZE = 3
STACKER_DOUBLE_CONV = False
STACKER_PROJECTION = True
STACKER_ANCHOR_WIDTH = True

# LayerNorm over each position's channels, the only statistic that survives an almost empty grid
STACKER_NORM = "layer"

# Baseline stacker leaves the class loss ungated so the gate can be measured as a variant
STACKER_CLASS_STOP_GRADIENT = False

# Add the detectors' own best score, as a logit, to the objectness logits
STACKER_OBJECTNESS_PRIOR = False

# Train every anchor that holds a candidate, not just the sampled ones
STACKER_CANDIDATE_SUPERVISION = False

STACKER_BATCH_SIZE = 8
STACKER_NUM_EPOCHS = 30
STACKER_LR = 1e-3
STACKER_MIN_LR = 1e-5
STACKER_WEIGHT_DECAY = 5e-4
STACKER_WARMUP_EPOCHS = 1

# Same loss recipe as the CNN: sampled BCE objectness, Huber box, ramped class CE.
STACKER_NEGATIVE_RATIO = 20
STACKER_MAX_NEGATIVES = 2048
STACKER_BOX_LOSS_WEIGHT = 1.0
STACKER_CLASS_LOSS_WEIGHT = 21.0
STACKER_CLASS_LOSS_WEIGHT_START = 1.0
STACKER_CLASS_LOSS_RAMP_EPOCHS = 5

# Inference and export, on the CNN's anchor grid and its decode
STACKER_SCORE_THRESHOLD = 0.05
STACKER_INFERENCE_SCORE_THRESHOLD = 0.5
STACKER_IOU_THRESHOLD = 0.5
STACKER_NMS_THRESHOLD = 0.5
STACKER_NMS_CONTAINMENT = 0.5
STACKER_TOP_K = 100

# Candidates each base detector offers the stacker, ranked by score with no operating threshold
STACKER_CANDIDATE_MAX = 50

# Set stacker: one token per candidate, K per model, self-attention across the tokens
SET_K = 10  # candidates per base model, so 3 * SET_K tokens per mammogram
SET_WIDTH = 128
SET_HEADS = 4
SET_BLOCKS = 2
SET_NUM_EPOCHS = 20
SET_BATCH_SIZE = 16
SET_LR = 1e-3
SET_MIN_LR = 1e-5
SET_WEIGHT_DECAY = 5e-4
SET_WARMUP_EPOCHS = 1
SET_NEGATIVE_RATIO = 20

# Roughly a sixth of the tokens are positive, so the weights start near 1
SET_BOX_LOSS_WEIGHT = 1.0
SET_CLASS_LOSS_WEIGHT = 2.0
SET_CLASS_LOSS_WEIGHT_START = 0.5
SET_CLASS_LOSS_RAMP_EPOCHS = 3

# SET_PRIOR_SCALE makes the detector-score prior's weight a trained scalar instead of a fixed logit
SET_PRIOR_SCALE = True

# Suppress anchors that hold no candidate; dilated by one cell for the head's 3 x 3 read
STACKER_GATE_EMPTY = False


# Fold artifact paths: one layout per model, by workspace and checkpoint suffix
FOLD_LAYOUT = {
    "cnn": (CNN_WORKSPACE, ".keras"),
    "faster_rcnn": (FASTER_RCNN_WORKSPACE, ".pt"),
    "yolo": (YOLO_WORKSPACE, ".pt"),
    "stacker": (STACKER_WORKSPACE, ".keras"),
}


def fold_paths(model: str, lesion_type: str, fold: int) -> dict:
    """Every path one out-of-fold unit reads or writes."""
    workspace, suffix = FOLD_LAYOUT[model]
    stem = f"{lesion_type}_{model}_fold{fold}"
    paths = {
        "model": MODELS_ROOT / f"{stem}{suffix}",
        "config": MODELS_ROOT / f"{stem}_config.json",
        "history": workspace / f"{stem}_history.csv",
        "records": workspace / f"{lesion_type}_{model}_oof_fold{fold}_records.csv",
        "candidates": workspace / f"{lesion_type}_{model}_oof_fold{fold}_candidates.csv",
        "manifest": oof_manifest_path(model, lesion_type),
    }
    if model == "yolo":
        paths["run"] = YOLO_PROJECT / stem
    else:
        paths["progress"] = MODELS_ROOT / f"{stem}_progress{suffix}"
        paths["state"] = MODELS_ROOT / f"{stem}_state.json"
        paths["best"] = MODELS_ROOT / f"{stem}_best{suffix}"
    return paths


def fold_model_path(model: str, lesion_type: str, fold: int) -> Path:
    return fold_paths(model, lesion_type, fold)["model"]


def oof_manifest_path(model: str, lesion_type: str) -> Path:
    workspace, _ = FOLD_LAYOUT[model]
    return workspace / f"{lesion_type}_{model}_oof_manifest.json"
