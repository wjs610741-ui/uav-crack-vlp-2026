"""Train the UAV-Crack segmentation baseline."""

from __future__ import annotations

import os

# Pretrained encoder weights are hosted on Hugging Face; use the mirror when it
# is reachable. Set HF_ENDPOINT yourself to override.
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import argparse
import csv
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import DEFAULT_ARGS, RUN_DIR, SPLIT_FILE
from .data import (
    CrackDataset,
    build_index,
    load_split,
    save_split,
    split_index,
    split_kfold,
)
from .losses import BCEDiceLoss
from .metrics import SegMetrics, format_metrics
from .model import CrackSegModel


def parse_args():
    d = DEFAULT_ARGS
    p = argparse.ArgumentParser(description="Train a UAV-Crack segmentation model")
    p.add_argument("--arch", default=d["arch"], help="smp architecture, e.g. unet / deeplabv3plus")
    p.add_argument("--encoder", default=d["encoder"], help="smp encoder, e.g. resnet34 / resnet50")
    p.add_argument("--epochs", type=int, default=d["epochs"])
    p.add_argument("--batch-size", type=int, default=d["batch_size"])
    p.add_argument("--accum-steps", type=int, default=d["accum_steps"])
    p.add_argument("--lr", type=float, default=d["lr"])
    p.add_argument("--weight-decay", type=float, default=d["weight_decay"])
    p.add_argument("--val-frac", type=float, default=d["val_frac"])
    p.add_argument("--split-group", default=d["split_group"], choices=["patch", "original"])
    p.add_argument("--seed", type=int, default=d["seed"])
    p.add_argument("--split-seed", type=int, default=None,
                   help="seed for fold assignment; defaults to --seed")
    p.add_argument("--num-workers", type=int, default=d["num_workers"])
    p.add_argument("--pos-weight", type=float, default=d["pos_weight"])
    p.add_argument("--bce-weight", type=float, default=d["bce_weight"])
    p.add_argument("--scales", nargs="+", default=["UAV-CrackX4", "UAV-CrackX8", "UAV-CrackX16"])
    p.add_argument("--run-dir", default=str(RUN_DIR))
    p.add_argument("--split-file", default=str(SPLIT_FILE))
    p.add_argument("--resplit", action="store_true", help="ignore the cached split file")
    p.add_argument("--full-data", action="store_true", help="train on every patch with a fixed epoch budget")
    p.add_argument("--kfold", type=int, default=0, help="enable k-fold cross validation")
    p.add_argument("--fold", type=int, default=0, help="which fold to validate on")
    p.add_argument("--warmup-epochs", type=int, default=3)
    p.add_argument("--encoder-lr-scale", type=float, default=0.33)
    p.add_argument("--ema-decay", type=float, default=0.0, help="weight EMA decay; 0 disables")
    p.add_argument("--max-steps", type=int, default=0, help="stop each epoch early (smoke test)")
    p.add_argument("--resume", default=None, help="checkpoint to resume from")
    return p.parse_args()


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ModelEma:
    """Exponential moving average of model weights, evaluated instead of the raw net."""

    def __init__(self, model, decay: float = 0.999):
        import copy

        self.module = copy.deepcopy(model).eval()
        for param in self.module.parameters():
            param.requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update(self, model):
        decay = self.decay
        for ema_param, param in zip(self.module.parameters(), model.parameters()):
            ema_param.mul_(decay).add_(param.detach(), alpha=1.0 - decay)
        for ema_buf, buf in zip(self.module.buffers(), model.buffers()):
            ema_buf.copy_(buf)


def build_loaders(args, split_path: Path):
    items = build_index(args.scales)
    if args.full_data:
        train_items, val_items = items, []
        save_split(split_path, train_items, val_items)
        print(f"[split] full-data: train={len(train_items)} (no holdout)")
        train_loader = DataLoader(
            CrackDataset(train_items, training=True),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        return train_loader, None, train_items, val_items
    split_seed = args.split_seed if args.split_seed is not None else args.seed
    if args.kfold > 0:
        train_items, val_items = split_kfold(
            items, k=args.kfold, fold=args.fold, seed=split_seed
        )
        save_split(split_path, train_items, val_items)
        print(f"[split] kfold={args.kfold} fold={args.fold} -> {split_path}")
    elif split_path.exists() and not args.resplit:
        train_names, val_names = load_split(split_path)
        train_items = [it for it in items if it["name"] in train_names]
        val_items = [it for it in items if it["name"] in val_names]
        if not train_items or not val_items:
            raise RuntimeError("cached split does not match the dataset; pass --resplit")
        print(f"[split] loaded {split_path}")
    else:
        train_items, val_items = split_index(
            items, val_frac=args.val_frac, seed=split_seed, group=args.split_group
        )
        save_split(split_path, train_items, val_items)
        print(f"[split] created {split_path}")

    print(f"[split] train={len(train_items)}  val={len(val_items)}")
    train_loader = DataLoader(
        CrackDataset(train_items, training=True),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        CrackDataset(val_items, training=False),
        batch_size=max(1, args.batch_size * 2),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, train_items, val_items


@torch.no_grad()
def evaluate(model, loader, device, threshold: float = 0.5):
    model.eval()
    metrics = SegMetrics()
    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(images)
        pred = torch.sigmoid(logits.float()) > threshold
        metrics.update(pred.cpu(), masks > 0.5)
    return metrics.compute()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; install a cu128 PyTorch build for Blackwell GPUs")
    seed_everything(args.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda")

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train_log.csv"

    split_path = Path(args.split_file)
    if args.split_file == str(SPLIT_FILE):
        if args.kfold > 0:
            split_path = run_dir / f"split_kfold{args.kfold}_f{args.fold}.json"
        else:
            split_path = run_dir / f"split_{args.split_group}.json"
    train_loader, val_loader, _, val_items = build_loaders(args, split_path)
    val_names = [item["name"] for item in val_items]

    model = CrackSegModel(args.arch, args.encoder).to(device)
    ema = ModelEma(model, args.ema_decay) if args.ema_decay > 0 else None
    criterion = BCEDiceLoss(pos_weight=args.pos_weight, bce_weight=args.bce_weight)

    # The pretrained encoder starts from a lower learning rate than the decoder.
    encoder_params = list(model.net.encoder.parameters())
    decoder_params = [
        p for name, p in model.named_parameters() if not name.startswith("net.encoder.")
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": args.lr * args.encoder_lr_scale},
            {"params": decoder_params, "lr": args.lr},
        ],
        weight_decay=args.weight_decay,
    )

    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return (epoch + 1) / max(1, args.warmup_epochs)
        progress = (epoch - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
        return 0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    start_epoch = 0
    best_iou = -1.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        start_epoch = ckpt.get("epoch", -1) + 1
        best_iou = ckpt.get("metrics", {}).get("Crack_IoU", -1.0)
        print(f"[resume] {args.resume} from epoch {start_epoch}, best IoU {best_iou:.2f}")

    header = ["epoch", "lr", "loss", "mIoU", "Crack_F1", "Crack_IoU", "Crack_Precision", "Crack_Recall", "aAcc", "seconds"]
    if not log_path.exists():
        with log_path.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(header)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        started = time.time()
        running_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        progress = tqdm(train_loader, desc=f"epoch {epoch + 1}/{args.epochs}")
        for step, (images, masks) in enumerate(progress):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(images)
                loss = criterion(logits, masks)

            (loss / args.accum_steps).backward()
            running_loss += loss.item()

            if (step + 1) % args.accum_steps == 0 or (step + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if ema is not None:
                    ema.update(model)

            progress.set_postfix(loss=f"{running_loss / (step + 1):.4f}")
            if args.max_steps and step + 1 >= args.max_steps:
                break

        scheduler.step()
        eval_model = ema.module if ema is not None else model
        if val_loader is not None:
            metrics = evaluate(eval_model, val_loader, device, threshold=DEFAULT_ARGS["threshold"])
        else:
            metrics = {}
        elapsed = time.time() - started
        if metrics:
            print(f"[epoch {epoch + 1}] loss={running_loss / max(1, step + 1):.4f}  {format_metrics(metrics)}  {elapsed:.1f}s")
        else:
            print(f"[epoch {epoch + 1}] loss={running_loss / max(1, step + 1):.4f}  {elapsed:.1f}s")

        metric_values = (
            [f"{metrics[k]:.4f}" for k in header[3:-1]] if metrics else [""] * (len(header) - 4)
        )
        with log_path.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                [epoch + 1, optimizer.param_groups[-1]["lr"], running_loss / max(1, step + 1)]
                + metric_values
                + [f"{elapsed:.1f}"]
            )

        state = {
            "model": eval_model.state_dict(),
            "arch": args.arch,
            "encoder": args.encoder,
            "epoch": epoch,
            "metrics": metrics,
            "args": vars(args),
            "val_names": val_names,
            "fold": args.fold if args.kfold > 0 else None,
            "ema": ema is not None,
        }
        torch.save(state, run_dir / "last.pth")
        if metrics and metrics["Crack_IoU"] > best_iou:
            best_iou = metrics["Crack_IoU"]
            torch.save(state, run_dir / "best.pth")
            print(f"[best] Crack IoU {best_iou:.2f} -> {run_dir / 'best.pth'}")

    if val_loader is None:
        torch.save(state, run_dir / "final.pth")
        print(f"[done] full-data model saved to {run_dir / 'final.pth'}")
    else:
        print(f"[done] best Crack IoU {best_iou:.2f}")


if __name__ == "__main__":
    main()
