"""Evaluate a weighted ensemble on a shared, clean validation fold."""

from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from .data import CrackDataset, build_index
from .metrics import SegMetrics, format_metrics
from .model import CrackSegModel


def parse_args():
    p = argparse.ArgumentParser(description="Weighted ensemble evaluation")
    p.add_argument("--ckpts", nargs="+", required=True)
    p.add_argument("--weights", default=None, help="comma separated weights, default all 1.0")
    p.add_argument("--thresholds", default="0.45,0.50,0.55,0.60")
    p.add_argument("--batch-size", type=int, default=6)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--tta", action="store_true")
    p.add_argument("--splits-from", default=None, help="checkpoint whose val_names to use")
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = (
        [float(w) for w in args.weights.split(",")]
        if args.weights
        else [1.0] * len(args.ckpts)
    )
    if len(weights) != len(args.ckpts):
        raise ValueError("number of weights must match number of checkpoints")

    models, val_names = [], None
    for path in args.ckpts:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        model = CrackSegModel(ckpt["arch"], ckpt["encoder"], encoder_weights=None).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        models.append(model)
        names = set(ckpt["val_names"])
        val_names = names if val_names is None else (val_names & names)
        print(f"[load] {path} arch={ckpt['arch']} encoder={ckpt['encoder']}")

    if not val_names:
        raise RuntimeError("checkpoints share no validation images")

    items = [it for it in build_index() if it["name"] in val_names]
    loader = DataLoader(
        CrackDataset(items, training=False),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    thresholds = [float(t) for t in args.thresholds.split(",")]
    meters = {t: SegMetrics() for t in thresholds}
    weight_sum = sum(weights)

    for images, masks in loader:
        images = images.to(device)
        prob = None
        for weight, model in zip(weights, models):
            current = model.predict_proba(images, tta=args.tta) * weight
            prob = current if prob is None else prob + current
        prob = (prob / weight_sum).cpu()
        for threshold, meter in meters.items():
            meter.update(prob > threshold, masks > 0.5)

    print(f"shared val images: {len(items)}  weights={weights}")
    best = None
    for threshold in thresholds:
        metrics = meters[threshold].compute()
        print(f"threshold={threshold:.2f}  {format_metrics(metrics)}")
        if best is None or metrics["Crack_IoU"] > best[1]["Crack_IoU"]:
            best = (threshold, metrics)
    print(f"best threshold by Crack IoU: {best[0]:.2f}  {format_metrics(best[1])}")


if __name__ == "__main__":
    main()
