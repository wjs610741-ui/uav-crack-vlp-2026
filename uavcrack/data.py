"""Dataset, augmentation and train/val splitting for UAV-Crack."""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import defaultdict
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .config import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    SCALES,
    TRAIN_GT_DIR,
    TRAIN_IMG_DIR,
    VAL_IMG_DIR,
)

_MEAN = np.asarray(IMAGENET_MEAN, dtype=np.float32).reshape(1, 1, 3)
_STD = np.asarray(IMAGENET_STD, dtype=np.float32).reshape(1, 1, 3)

# ``937956_DJI_20231015154707_0002_Z.JPG_12_z.jpg`` -> original flight frame.
_ORIGINAL_RE = re.compile(r"_\d+_z\.(?:jpg|png)$", re.IGNORECASE)


def original_id(filename: str) -> str:
    """Strip the patch suffix so patches from the same aerial frame share an id."""
    return _ORIGINAL_RE.sub("", filename)


def imread_unicode(path: str | Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    """Read an image from a path that may contain non-ASCII characters.

    ``cv2.imread`` silently fails on Unicode paths on Windows, so decode from a
    byte buffer instead.
    """
    buf = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(buf, flags)
    if image is None:
        raise OSError(f"failed to decode image: {path}")
    return image


def build_index(scales=SCALES, img_root: Path = TRAIN_IMG_DIR, gt_root: Path = TRAIN_GT_DIR):
    """List every (image, mask) pair under the training scales."""
    items = []
    for scale in scales:
        img_dir = img_root / scale
        gt_dir = gt_root / scale
        for img_path in sorted(img_dir.glob("*.jpg")):
            mask_path = gt_dir / f"{img_path.stem}.png"
            if not mask_path.exists():
                raise FileNotFoundError(f"missing mask for {img_path.name}: {mask_path}")
            items.append(
                {
                    "image": str(img_path),
                    "mask": str(mask_path),
                    "scale": scale,
                    "name": img_path.name,
                }
            )
    return items


def list_val_images(img_dir: Path = VAL_IMG_DIR):
    return sorted(img_dir.glob("*.jpg"))


def split_index(items, val_frac: float = 0.1, seed: int = 42, group: str = "scene"):
    """Split the training patches, stratified by zoom scale.

    ``group="scene"`` holds out exactly one patch per aerial frame, which mirrors
    the official benchmark: its val patches share source frames with training and
    only differ in patch position. This is the default.

    ``group="original"`` holds out every patch of a frame, measuring
    generalisation to unseen scenes (pessimistic).

    ``group="patch"`` is a plain random split. Neighbouring patches of the same
    frame are highly similar, so it overestimates validation scores.
    """
    rng = random.Random(seed)
    by_scale = defaultdict(list)
    for item in items:
        by_scale[item["scale"]].append(item)

    train, val = [], []
    for scale in sorted(by_scale):
        entries = sorted(by_scale[scale], key=lambda d: d["name"])
        if group in ("original", "scene"):
            buckets = defaultdict(list)
            for item in entries:
                buckets[original_id(item["name"])].append(item)
            keys = sorted(buckets)
            if group == "original":
                rng.shuffle(keys)
                n_val = max(1, round(len(keys) * val_frac))
                val_keys = set(keys[:n_val])
                for key in keys:
                    (val if key in val_keys else train).extend(buckets[key])
            else:
                for key in keys:
                    patches = buckets[key]
                    if len(patches) == 1:
                        # No sibling patch, so the frame is only seen in val.
                        val.extend(patches)
                        continue
                    digest = hashlib.sha1(f"{seed}:{key}".encode("utf-8")).hexdigest()
                    held_out = int(digest, 16) % len(patches)
                    val.append(patches[held_out])
                    train.extend(p for i, p in enumerate(patches) if i != held_out)
        elif group == "patch":
            rng.shuffle(entries)
            n_val = max(1, round(len(entries) * val_frac))
            val.extend(entries[:n_val])
            train.extend(entries[n_val:])
        else:
            raise ValueError(f"unknown split group: {group}")

    train.sort(key=lambda d: (d["scale"], d["name"]))
    val.sort(key=lambda d: (d["scale"], d["name"]))
    return train, val


def split_kfold(items, k: int = 5, fold: int = 0, seed: int = 42):
    """Assign patches to ``k`` folds deterministically.

    Folds are patch-level (not frame-level) so that sibling patches of a frame
    stay in training while one is validated, mirroring the official val set.
    """
    if not 0 <= fold < k:
        raise ValueError(f"fold must be in [0, {k}), got {fold}")
    train, val = [], []
    for item in items:
        digest = hashlib.sha1(f"{seed}:{item['name']}".encode("utf-8")).hexdigest()
        assigned = int(digest, 16) % k
        (val if assigned == fold else train).append(item)
    train.sort(key=lambda d: (d["scale"], d["name"]))
    val.sort(key=lambda d: (d["scale"], d["name"]))
    return train, val


def save_split(path: Path, train_items, val_items):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "train": [item["name"] for item in train_items],
        "val": [item["name"] for item in val_items],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_split(path: Path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return set(payload["train"]), set(payload["val"])


def build_transforms(training: bool):
    if not training:
        return None
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Affine(
                scale=(0.9, 1.1),
                translate_percent={"x": (-0.03, 0.03), "y": (-0.03, 0.03)},
                rotate=(-10, 10),
                border_mode=cv2.BORDER_REFLECT_101,
                mask_interpolation=cv2.INTER_NEAREST,
                p=0.5,
            ),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.OneOf([A.GaussNoise(p=1.0), A.GaussianBlur(blur_limit=3, p=1.0)], p=0.15),
        ]
    )


class CrackDataset(Dataset):
    """Paired image/mask dataset. Masks are binarised to {0, 1}."""

    def __init__(self, items, training: bool = False):
        self.items = list(items)
        self.transforms = build_transforms(training)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]
        image = imread_unicode(item["image"], cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = imread_unicode(item["mask"], cv2.IMREAD_GRAYSCALE)
        mask = (mask > 0).astype(np.float32)

        if self.transforms is not None:
            augmented = self.transforms(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]

        image = image.astype(np.float32) / 255.0
        image = (image - _MEAN) / _STD

        x = torch.from_numpy(image.transpose(2, 0, 1))
        y = torch.from_numpy(mask[None].astype(np.float32))
        return x, y


class InferenceDataset(Dataset):
    """Images only; returns the tensor plus the source file name."""

    def __init__(self, paths):
        self.paths = [Path(p) for p in paths]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        image = imread_unicode(path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0
        image = (image - _MEAN) / _STD
        return torch.from_numpy(image.transpose(2, 0, 1)), path.name
