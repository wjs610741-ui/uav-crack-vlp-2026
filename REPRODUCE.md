# Reproducing the CHD-UNetR34 submission

Leaderboard result: **mIoU 82.58**, Crack F1 80.65, Crack IoU 67.58,
precision 79.63, recall 81.70, aAcc 97.70 (2026-10-29, rank 3/82).

The submission is a probability average of **51 models** with four-way flip TTA,
binarised at threshold **0.56**.

## 1. Environment

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128
$env:HF_ENDPOINT = "https://hf-mirror.com"   # optional, for smp encoder weights
```

Blackwell GPUs (RTX 50 series) need the cu128 wheels; `requirements.txt` pins
`torch==2.7.1` / `torchvision==0.22.1`.

## 2. Dataset

Place the official UAV-CrackX archive at:

```
UAV-Crack-dataset/UAV-Crack-dataset/
├── gtFine/train/UAV-CrackX{4,8,16}/        400 masks each
├── leftImg8bit/train/UAV-CrackX{4,8,16}/   400 images each
└── leftImg8bit/val/                        300 images
```

## 3. Models

All checkpoints are distributed in `uavcrack_weights_part1.zip` …
`uavcrack_weights_part5.zip` under `checkpoints/`. Each file is named
`<run-directory>_<best|final>.pth`.

| Group | Run directories | Architecture | Command |
|---|---|---|---|
| A | `fold0` … `fold4` | UNet + ResNet34 | `--arch unet --encoder resnet34 --kfold 5 --fold N --epochs 50 --batch-size 8 --accum-steps 2` |
| B | `uplus_fold0` … `uplus_fold4` | UNet++ + ResNet34 | `--arch unetplusplus --encoder resnet34 --kfold 5 --fold N --epochs 50 --batch-size 8 --accum-steps 2` |
| C | `full_unet_r34_s42`, `full_unetpp_r34_s42`, `full_unet_r50_s42`, `full_unetpp_r50_s42`, `full_unet_r34_s43`, `full_unetpp_r34_s43` | UNet/UNet++ + ResNet34/50 | `--full-data --epochs 50 --ema-decay 0.999` (batch 8 for R34, 4 for R50) |
| D | `het_seg_b2_f0` … `f4` | SegFormer + MiT-B2 | `--arch segformer --encoder mit_b2 --lr 1e-4 --encoder-lr-scale 1.0 --ema-decay 0.999 --kfold 5 --fold N --epochs 60 --batch-size 8` |
| E | `seed2_seg_f0` … `f4` | SegFormer + MiT-B2 | same as D with `--seed 43 --split-seed 42` |
| F | `het_convnext_f0` … `f4` | UNet + ConvNeXt-Tiny | `--arch unet --encoder tu-convnext_tiny --ema-decay 0.999 --kfold 5 --fold N --epochs 50 --batch-size 4` |
| G | `het_convnext_small_f0` … `f4` | UNet + ConvNeXt-Small | `--arch unet --encoder tu-convnext_small --ema-decay 0.999 --kfold 5 --fold N --epochs 50 --batch-size 4` |
| H | `seed2_small_f0` … `f4` | UNet + ConvNeXt-Small | same as G with `--seed 43 --split-seed 42` |
| I | `het_convnext_base_f0` … `f4` | UNet + ConvNeXt-Base | `--arch unet --encoder tu-convnext_base --ema-decay 0.999 --kfold 5 --fold N --epochs 50 --batch-size 4` |
| J | `seed2_base_f0` … `f4` | UNet + ConvNeXt-Base | same as I with `--seed 43 --split-seed 42` |

All commands also use `--num-workers 4` and a distinct `--run-dir`. Groups E, H
and J keep the same fold assignment as their first-seed counterparts
(`--split-seed 42`) and vary only the model initialisation and augmentation RNG.

## 4. Inference

```powershell
python -m uavcrack.ensemble_infer --ckpts <51 checkpoint paths> `
    --threshold 0.56 --batch-size 4 --out-dir submission/result
python -m uavcrack.package_result
python -m uavcrack.check_submission
```

`ensemble_infer` streams one model at a time, averages sigmoid probabilities
with four-way flip TTA, and writes 300 single-channel PNG masks with values 0/1.

Predicted crack ratio on the official val set: **6.03%** (training GT: 6.17%).

## 5. ONNX

`uavcrack_onnx.zip` contains seven exports covering every architecture family,
all with fixed input `(1, 3, 378, 672)` and output logits `(1, 1, 378, 672)`.
Exports were verified against PyTorch (maximum absolute logit difference
1.7e-5).

## 6. Notes

- The dataset is **not** included (CC BY-NC-SA 3.0, no redistribution).
- Heterogeneous backbones matter more than model count: on the fold-0
  validation split, adding SegFormer lifted the CNN pair from 81.38 to 81.53,
  ConvNeXt-Tiny to 81.71, and ConvNeXt-Small to 81.89.
- `uavcrack/ensemble_eval.py` reproduces the fold ablations; `oof_eval.py`
  reproduces the out-of-fold threshold sweeps.
