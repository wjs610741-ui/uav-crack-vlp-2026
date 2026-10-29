"""Validate the prediction folder and zip against the submission rules."""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from .config import IMAGE_HEIGHT, IMAGE_WIDTH, SUBMISSION_DIR, VAL_IMG_DIR


def parse_args():
    p = argparse.ArgumentParser(description="Check submission format")
    p.add_argument("--result-dir", default=str(SUBMISSION_DIR / "result"))
    p.add_argument("--zip", default=str(SUBMISSION_DIR / "result.zip"))
    return p.parse_args()


def expected_names(image_dir: Path):
    """Input ``foo.JPG_12_z.jpg`` must become ``foo.JPG_12_z.png``."""
    return {f"{p.stem}.png" for p in sorted(image_dir.glob("*.jpg"))}


def check_result_dir(result_dir: Path, expected: set[str]):
    problems = []
    masks = sorted(result_dir.glob("*.png"))
    names = {p.name for p in masks}

    if len(masks) != len(expected):
        problems.append(f"expected {len(expected)} masks, found {len(masks)}")
    missing = expected - names
    extra = names - expected
    if missing:
        problems.append(f"missing {len(missing)} predictions, e.g. {sorted(missing)[:3]}")
    if extra:
        problems.append(f"{len(extra)} unexpected files, e.g. {sorted(extra)[:3]}")

    zeros = 0
    for path in masks:
        with Image.open(path) as img:
            if img.size != (IMAGE_WIDTH, IMAGE_HEIGHT):
                problems.append(f"{path.name}: size {img.size}, expected {(IMAGE_WIDTH, IMAGE_HEIGHT)}")
                continue
            if img.mode != "L":
                problems.append(f"{path.name}: mode {img.mode}, expected L (single channel)")
            array = np.array(img)
            values = np.unique(array)
            if not np.isin(values, [0, 1]).all():
                problems.append(f"{path.name}: values {values[:5]}, expected only 0/1")
            if array.sum() == 0:
                zeros += 1

    if zeros:
        print(f"note: {zeros} masks are entirely background")

    return problems


def check_zip(zip_path: Path, expected: set[str]):
    problems = []
    if not zip_path.exists():
        return [f"zip not found: {zip_path}"]

    with zipfile.ZipFile(zip_path) as zf:
        if zf.testzip() is not None:
            problems.append(f"zip CRC check failed at {zf.testzip()}")
        entries = [info.filename for info in zf.infolist() if not info.is_dir()]
        if len(entries) != len(set(entries)):
            problems.append("zip contains duplicate entry names")
    expected_entries = {f"result/{name}" for name in expected}
    if set(entries) != expected_entries:
        missing = expected_entries - set(entries)
        extra = set(entries) - expected_entries
        if missing:
            problems.append(f"zip missing {len(missing)} entries, e.g. {sorted(missing)[:3]}")
        if extra:
            problems.append(f"zip has {len(extra)} unexpected entries, e.g. {sorted(extra)[:3]}")
    bad_prefix = [e for e in entries if not e.startswith("result/")]
    if bad_prefix:
        problems.append(f"entries outside result/: {bad_prefix[:3]}")
    return problems


def main():
    args = parse_args()
    expected = expected_names(Path(VAL_IMG_DIR))
    print(f"expected masks: {len(expected)}")

    problems = check_result_dir(Path(args.result_dir), expected)
    problems += check_zip(Path(args.zip), expected)

    if problems:
        print("\nFAILED:")
        for problem in problems:
            print(f"  - {problem}")
        sys.exit(1)

    print("\nOK: result/ and result.zip satisfy the submission requirements")


if __name__ == "__main__":
    main()
