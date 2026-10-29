"""BCE + soft Dice loss for heavily imbalanced crack masks."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BCEDiceLoss(nn.Module):
    def __init__(self, pos_weight: float = 3.0, bce_weight: float = 0.5):
        super().__init__()
        self.pos_weight = pos_weight
        self.bce_weight = bce_weight

    def forward(self, logits, target):
        pos_weight = torch.tensor(self.pos_weight, device=logits.device, dtype=logits.dtype)
        bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)

        prob = torch.sigmoid(logits)
        dims = (1, 2, 3)
        intersection = (prob * target).sum(dims)
        union = prob.sum(dims) + target.sum(dims)
        dice_loss = 1.0 - (2.0 * intersection + 1.0) / (union + 1.0)

        return self.bce_weight * bce + (1.0 - self.bce_weight) * dice_loss.mean()
