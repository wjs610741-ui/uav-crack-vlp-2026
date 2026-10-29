"""Out-of-fold evaluation across k-fold checkpoints.

Each checkpoint predicts exactly the fold it did not train on, so together the
checkpoints cover every training patch once. Use the resulting sweep to pick a
decision threshold for the ensemble submission.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data import CrackDataset, build_index
from .metrics import SegMetrics, format_metrics
from .model import CrackSegModel


def parse_args():
    p = argparse.ArgumentParser(description="Out-of-fold threshold sweep")
    p.add_argument("--ckpts", nargs="+", required=True)
    p.add_argument("--scales", nargs="+", default=["UAV-CrackX4", "UAV-CrackX8", "UAV-CrackX16"])
    p.add_argument("--thresholds", default="0.40,0.45,0.50,0.55,0.60,0.65")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--tta", action="store_true")
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    thresholds = [float(t) for t in args.thresholds.split(",")]
    meters = {t: SegMetrics() for t in thresholds}
    all_items = build_index(args.scales)
    covered = set()

    for ckpt_path in args.ckpts:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = CrackSegModel(ckpt["arch"], ckpt["encoder"], encoder_weights=None).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()

        val_names = set(ckpt["val_names"])
        covered |= val_names
        items = [it for it in all_items if it["name"] in val_names]
        loader = DataLoader(
            CrackDataset(items, training=False),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
        for images, masks in loader:
            prob = model.predict_proba(images.to(device), tta=args.tta).cpu()
            for threshold, meter in meters.items():
                meter.update(prob > threshold, masks > 0.5)

        print(f"[oof] {ckpt_path}: fold={ckpt.get('fold')} epoch={ckpt.get('epoch')} images={len(items)}")
        del model
        torch.cuda.empty_cache()

    print(f"[oof] covered {len(covered)}/{len(all_items)} training patches")
    best = None
    for threshold in thresholds:
        metrics = meters[threshold].compute()
        print(f"threshold={threshold:.2f}  {format_metrics(metrics)}")
        if best is None or metrics["Crack_F1"] > best[1]["Crack_F1"]:
            best = (threshold, metrics)

    print(f"\nbest OOF threshold by Crack F1: {best[0]:.2f}  {format_metrics(best[1])}")


if __name__ == "__main__":
    main()
