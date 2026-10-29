"""Average predictions from several checkpoints and write binary PNG masks.

Models are streamed one at a time so that large ensembles do not exhaust GPU
memory (which on Windows WDDM causes silent shared-memory spilling and a 10x
slowdown).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from .config import SUBMISSION_DIR, VAL_IMG_DIR
from .data import InferenceDataset, list_val_images
from .model import CrackSegModel


def parse_args():
    p = argparse.ArgumentParser(description="Ensemble inference on the val images")
    p.add_argument("--ckpts", nargs="+", required=True)
    p.add_argument("--image-dir", default=str(VAL_IMG_DIR))
    p.add_argument("--out-dir", default=str(SUBMISSION_DIR / "result"))
    p.add_argument("--threshold", type=float, default=0.55)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--no-tta", action="store_true")
    p.add_argument("--ms-scales", default="1.0", help="comma separated multi-scale TTA factors")
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scales = tuple(float(s) for s in args.ms_scales.split(","))

    paths = list_val_images(Path(args.image_dir))
    loader = DataLoader(
        InferenceDataset(paths),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.png"):
        old.unlink()

    accumulator = None
    names_all: list[str] = []
    total_models = len(args.ckpts)
    started = time.time()

    for index, path in enumerate(args.ckpts, 1):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        model = CrackSegModel(ckpt["arch"], ckpt["encoder"], encoder_weights=None).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()

        batches = []
        for images, names in loader:
            prob = model.predict_proba(images.to(device), tta=not args.no_tta, scales=scales)
            batches.append(prob.cpu())
            if index == 1:
                names_all.extend(names)

        model_probs = torch.cat(batches, dim=0)
        accumulator = model_probs if accumulator is None else accumulator + model_probs

        del model, batches, model_probs
        torch.cuda.empty_cache()

        elapsed = time.time() - started
        eta = elapsed / index * (total_models - index)
        print(
            f"[{index}/{total_models}] {path} (fold={ckpt.get('fold')}, epoch={ckpt.get('epoch')})"
            f"  elapsed={elapsed:.0f}s  eta={eta:.0f}s",
            flush=True,
        )

    accumulator /= total_models

    positive = 0
    total = 0
    for i, name in enumerate(names_all):
        mask = (accumulator[i, 0].numpy() > args.threshold).astype(np.uint8)
        Image.fromarray(mask, mode="L").save(out_dir / f"{Path(name).stem}.png")
        positive += int(mask.sum())
        total += mask.size

    print(f"wrote {len(names_all)} masks to {out_dir}")
    print(f"positive pixel ratio: {positive / max(1, total):.4f}  (threshold={args.threshold})")


if __name__ == "__main__":
    main()
