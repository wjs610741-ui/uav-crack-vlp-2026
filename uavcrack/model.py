"""Model wrapper handling the 378 -> 384 padding required by the encoder."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import PAD_TO


class CrackSegModel(nn.Module):
    """segmentation-models-pytorch backbone plus automatic pad/crop.

    The benchmark images are 672x378, and 378 is not divisible by 32, so the
    input is reflect-padded to 672x384, forwarded, then cropped back to the
    original size to keep the loss aligned with the mask.
    """

    def __init__(self, arch: str = "unet", encoder: str = "resnet34", encoder_weights="imagenet"):
        super().__init__()
        import segmentation_models_pytorch as smp

        self.net = smp.create_model(
            arch,
            encoder_name=encoder,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=1,
        )

    def forward(self, x):
        height, width = x.shape[-2:]
        pad_h = (-height) % PAD_TO
        pad_w = (-width) % PAD_TO
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        logits = self.net(x)
        return logits[..., :height, :width]

    @torch.no_grad()
    def predict_proba(self, x, tta: bool = True, scales=(1.0,)):
        """Sigmoid probabilities averaged over scales and, optionally, four flips."""
        height, width = x.shape[-2:]
        views = [(False, False), (True, False), (False, True), (True, True)] if tta else [(False, False)]
        prob_sum = None
        for scale in scales:
            if scale == 1.0:
                scaled = x
            else:
                new_h = max(PAD_TO, int(round(height * scale / PAD_TO)) * PAD_TO)
                new_w = max(PAD_TO, int(round(width * scale / PAD_TO)) * PAD_TO)
                scaled = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)

            for flip_h, flip_w in views:
                view = scaled
                if flip_h:
                    view = torch.flip(view, dims=[3])
                if flip_w:
                    view = torch.flip(view, dims=[2])
                prob = torch.sigmoid(self(view).float())
                if flip_h:
                    prob = torch.flip(prob, dims=[3])
                if flip_w:
                    prob = torch.flip(prob, dims=[2])
                if scaled.shape[-2:] != (height, width):
                    prob = F.interpolate(
                        prob, size=(height, width), mode="bilinear", align_corners=False
                    )
                prob_sum = prob if prob_sum is None else prob_sum + prob

        return prob_sum / (len(scales) * len(views))
