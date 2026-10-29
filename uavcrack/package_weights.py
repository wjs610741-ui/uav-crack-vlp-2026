"""Bundle checkpoints, the ONNX export and a manifest for the weight upload."""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Package model weights for sharing")
    p.add_argument("--ckpts", nargs="+", required=True)
    p.add_argument("--onnx", nargs="+", default=None)
    p.add_argument("--extra", nargs="+", default=None, help="additional documentation files")
    p.add_argument("--out", default="weights/uavcrack_weights.zip")
    return p.parse_args()


def describe(path: Path):
    import torch

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    return {
        "file": f"{path.parent.name}_{path.name}",
        "arch": ckpt.get("arch"),
        "encoder": ckpt.get("encoder"),
        "fold": ckpt.get("fold"),
        "epoch": ckpt.get("epoch"),
        "ema": ckpt.get("ema", False),
        "metrics": {
            key: round(value, 4)
            for key, value in (ckpt.get("metrics") or {}).items()
            if isinstance(value, float)
        },
    }


def main():
    args = parse_args()
    ckpts = [Path(p) for p in args.ckpts]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    manifest = {"checkpoints": [describe(p) for p in ckpts]}
    if args.onnx:
        manifest["onnx"] = [Path(p).name for p in args.onnx]

    manifest_path = out.parent / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_STORED) as zf:
        for path in ckpts:
            # Several runs share the file name best.pth/final.pth, so keep the
            # run directory name to avoid collisions inside the archive.
            zf.write(path, arcname=f"checkpoints/{path.parent.name}_{path.name}")
        for onnx_path in args.onnx or []:
            onnx_path = Path(onnx_path)
            zf.write(onnx_path, arcname=f"onnx/{onnx_path.name}")
        zf.write(manifest_path, arcname="manifest.json")
        for extra in args.extra or []:
            extra = Path(extra)
            zf.write(extra, arcname=f"docs/{extra.name}")

    print(f"packed {len(ckpts)} checkpoints -> {out}  ({out.stat().st_size / 1024 / 1024:.1f} MB)")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
