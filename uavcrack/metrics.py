"""Pixel-level segmentation metrics matching the VLP leaderboard."""

from __future__ import annotations

import torch


class SegMetrics:
    """Accumulates TP/FP/FN/TN over the dataset and derives the six metrics."""

    def __init__(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        pred = pred.bool()
        target = target.bool()
        self.tp += int((pred & target).sum())
        self.fp += int((pred & ~target).sum())
        self.fn += int((~pred & target).sum())
        self.tn += int((~pred & ~target).sum())

    def compute(self) -> dict:
        tp, fp, fn, tn = self.tp, self.fp, self.fn, self.tn
        eps = 1e-9
        crack_iou = tp / (tp + fp + fn + eps)
        bg_iou = tn / (tn + fp + fn + eps)
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        total = tp + fp + fn + tn + eps
        return {
            "mIoU": 0.5 * (crack_iou + bg_iou) * 100.0,
            "Crack_F1": f1 * 100.0,
            "Crack_IoU": crack_iou * 100.0,
            "Crack_Precision": precision * 100.0,
            "Crack_Recall": recall * 100.0,
            "aAcc": (tp + tn) / total * 100.0,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    def reset(self):
        self.tp = self.fp = self.fn = self.tn = 0


def format_metrics(metrics: dict) -> str:
    keys = ["mIoU", "Crack_F1", "Crack_IoU", "Crack_Precision", "Crack_Recall", "aAcc"]
    return "  ".join(f"{key}={metrics[key]:.2f}" for key in keys)
