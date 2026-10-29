"""Zip the predicted masks into the exact layout the platform expects."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

from .config import SUBMISSION_DIR


def parse_args():
    p = argparse.ArgumentParser(description="Package result/ into result.zip")
    p.add_argument("--result-dir", default=str(SUBMISSION_DIR / "result"))
    p.add_argument("--out", default=str(SUBMISSION_DIR / "result.zip"))
    return p.parse_args()


def main():
    args = parse_args()
    result_dir = Path(args.result_dir)
    masks = sorted(result_dir.glob("*.png"))
    if not masks:
        raise FileNotFoundError(f"no PNG masks in {result_dir}; run infer.py first")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # zipfile always writes forward slashes, unlike Compress-Archive on Windows.
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for mask in masks:
            zf.write(mask, arcname=f"result/{mask.name}")

    print(f"packed {len(masks)} masks -> {out}  ({out.stat().st_size / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()
