"""Evaluate a checkpoint on the local validation split and sweep thresholds."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .config import SPLIT_FILE
from .data import CrackDataset, build_index, load_split
from .metrics import SegMetrics, format_metrics
from .model import CrackSegModel


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a UAV-Crack checkpoint")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--split-file", default=str(SPLIT_FILE))
    p.add_argument("--scales", nargs="+", default=["UAV-CrackX4", "UAV-CrackX8", "UAV-CrackX16"])
    p.add_argument("--thresholds", default="0.3,0.4,0.5,0.6,0.7")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--tta", action="store_true")
    p.add_argument("--ms-scales", default="1.0", help="comma separated multi-scale TTA factors")
    return p.parse_args()


@torch.no_grad()
def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = CrackSegModel(ckpt["arch"], ckpt["encoder"], encoder_weights=None).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    if ckpt.get("val_names"):
        val_names = set(ckpt["val_names"])
        print(f"[split] using {len(val_names)} val names stored in the checkpoint")
    else:
        _, val_names = load_split(Path(args.split_file))
    items = [it for it in build_index(args.scales) if it["name"] in val_names]
    loader = DataLoader(
        CrackDataset(items, training=False),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    thresholds = [float(t) for t in args.thresholds.split(",")]
    meters = {t: SegMetrics() for t in thresholds}

    for images, masks in loader:
        images = images.to(device)
        scales = tuple(float(s) for s in args.ms_scales.split(","))
        prob = model.predict_proba(images, tta=args.tta, scales=scales).cpu()
        for threshold, meter in meters.items():
            meter.update(prob > threshold, masks > 0.5)

    print(f"checkpoint: {args.ckpt}  (epoch {ckpt.get('epoch')}, tta={args.tta})")
    best = None
    for threshold in thresholds:
        metrics = meters[threshold].compute()
        print(f"threshold={threshold:.2f}  {format_metrics(metrics)}")
        if best is None or metrics["Crack_F1"] > best[1]["Crack_F1"]:
            best = (threshold, metrics)

    print(f"\nbest threshold by Crack F1: {best[0]:.2f}  {format_metrics(best[1])}")
    return best


if __name__ == "__main__":
    evaluate(parse_args())
