"""Shared paths and default hyper-parameters for the UAV-Crack pipeline."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Downloaded dataset. Note the directory name is ``leftImg8bit`` with a capital I,
# unlike the ``leftimg8bit`` spelling used in the official submission guidelines.
DATA_ROOT = PROJECT_ROOT / "UAV-Crack-dataset" / "UAV-Crack-dataset"
IMG_DIR = DATA_ROOT / "leftImg8bit"
GT_DIR = DATA_ROOT / "gtFine"
TRAIN_IMG_DIR = IMG_DIR / "train"
TRAIN_GT_DIR = GT_DIR / "train"
VAL_IMG_DIR = IMG_DIR / "val"

RUN_DIR = PROJECT_ROOT / "runs"
SUBMISSION_DIR = PROJECT_ROOT / "submission"
SPLIT_FILE = RUN_DIR / "split.json"

SCALES = ("UAV-CrackX4", "UAV-CrackX8", "UAV-CrackX16")

IMAGE_HEIGHT = 378
IMAGE_WIDTH = 672
# Encoder downsampling requires both sides to be divisible by 32.
PAD_TO = 32

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Validation split is drawn from the training patches because the official
# ``leftImg8bit/val`` folder ships without ground truth.
DEFAULT_ARGS = dict(
    arch="unet",
    encoder="resnet34",
    epochs=80,
    batch_size=4,
    accum_steps=4,
    lr=3e-4,
    weight_decay=1e-4,
    val_frac=0.1,
    split_group="scene",
    seed=42,
    num_workers=0,
    pos_weight=3.0,
    bce_weight=0.5,
    threshold=0.5,
)
