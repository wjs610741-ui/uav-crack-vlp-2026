"""Export a trained checkpoint to ONNX (required by the competition rules)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .config import IMAGE_HEIGHT, IMAGE_WIDTH
from .model import CrackSegModel


def parse_args():
    p = argparse.ArgumentParser(description="Export a checkpoint to ONNX")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--verify", action="store_true", help="compare against onnxruntime if installed")
    return p.parse_args()


def main():
    args = parse_args()
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = CrackSegModel(ckpt["arch"], ckpt["encoder"], encoder_weights=None)
    model.load_state_dict(ckpt["model"])
    model.eval()

    dummy = torch.randn(1, 3, IMAGE_HEIGHT, IMAGE_WIDTH)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        input_names=["image"],
        output_names=["logits"],
        opset_version=args.opset,
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"exported {args.ckpt} -> {out_path}  ({out_path.stat().st_size / 1024 / 1024:.1f} MB)")

    if args.verify:
        try:
            import onnxruntime as ort
        except ImportError:
            print("onnxruntime not installed; run `pip install onnxruntime` to verify")
            return
        session = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
        reference = model(dummy).detach().numpy()
        exported = session.run(None, {"image": dummy.numpy()})[0]
        diff = np.abs(reference - exported).max()
        print(f"max |pytorch - onnx| = {diff:.6e}")
        # Logits are order 10-20, so 5e-4 is well within float32 export noise.
        if diff > 5e-4:
            raise SystemExit("ONNX export does not match PyTorch output")


if __name__ == "__main__":
    main()
