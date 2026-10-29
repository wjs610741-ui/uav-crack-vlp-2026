"""Run inference on the official val images and write binary 0/1 PNG masks."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from .config import SUBMISSION_DIR, VAL_IMG_DIR
from .data import InferenceDataset, list_val_images
from .model import CrackSegModel


def parse_args():
    p = argparse.ArgumentParser(description="Predict binary masks for the 300 val images")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--image-dir", default=str(VAL_IMG_DIR))
    p.add_argument("--out-dir", default=str(SUBMISSION_DIR / "result"))
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--no-tta", action="store_true")
    return p.parse_args()


@torch.no_grad()
def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = CrackSegModel(ckpt["arch"], ckpt["encoder"], encoder_weights=None).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    paths = list_val_images(Path(args.image_dir))
    if not paths:
        raise FileNotFoundError(f"no .jpg images under {args.image_dir}")

    loader = DataLoader(
        InferenceDataset(paths),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    positive_pixels = 0
    total_pixels = 0
    for images, names in loader:
        images = images.to(device)
        prob = model.predict_proba(images, tta=not args.no_tta).cpu().numpy()
        binary = (prob > args.threshold).astype(np.uint8)

        for mask, name in zip(binary, names):
            mask = mask[0]
            Image.fromarray(mask, mode="L").save(out_dir / f"{Path(name).stem}.png")
            positive_pixels += int(mask.sum())
            total_pixels += mask.size

    ratio = positive_pixels / max(1, total_pixels)
    print(f"wrote {len(paths)} masks to {out_dir}")
    print(f"positive pixel ratio: {ratio:.4f}  (threshold={args.threshold}, tta={not args.no_tta})")


if __name__ == "__main__":
    run(parse_args())
