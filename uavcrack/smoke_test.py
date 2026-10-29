"""Quick diagnostics: dataset integrity, split sizes, and a forward pass."""

from __future__ import annotations

import torch

from .data import CrackDataset, build_index, split_index
from .model import CrackSegModel


def main():
    print(f"torch {torch.__version__}, cuda={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(
            f"device={torch.cuda.get_device_name(0)} "
            f"capability={torch.cuda.get_device_capability(0)} "
            f"bf16={torch.cuda.is_bf16_supported()}"
        )

    items = build_index()
    print(f"indexed {len(items)} image/mask pairs")
    train_items, val_items = split_index(items, group="scene")
    print(f"scene split: train={len(train_items)} val={len(val_items)}")

    dataset = CrackDataset(train_items[:4], training=True)
    images = torch.stack([dataset[i][0] for i in range(4)])
    masks = torch.stack([dataset[i][1] for i in range(4)])
    print(f"batch {tuple(images.shape)}  mask {tuple(masks.shape)}")
    print(f"mask values {torch.unique(masks).tolist()} (must be [0.0, 1.0])")

    model = CrackSegModel(encoder_weights=None).cuda().eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(images.cuda())
    print(f"logits {tuple(logits.shape)} {logits.dtype}")
    assert logits.shape[:2] == (4, 1) and logits.shape[-2:] == (378, 672)
    print(f"peak GPU memory: {torch.cuda.max_memory_allocated() / 1024 / 1024:.0f} MB")


if __name__ == "__main__":
    main()
