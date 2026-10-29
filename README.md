# UAV-Crack Segmentation

**2026 VLP 挑战赛参赛作品**

Binary crack segmentation for the [VLP UAV-Crack benchmark](http://vlp.chd.edu.cn/tasks/4).

The pipeline trains a diverse ensemble of segmentation models on the 1,200
annotated patches, predicts binary masks for the 300 official validation images,
and packages them into the `result.zip` layout required by the platform.

## Dataset layout

```
UAV-Crack-dataset/UAV-Crack-dataset/
├── gtFine/train/UAV-CrackX{4,8,16}/        400 binary masks each
├── leftImg8bit/train/UAV-CrackX{4,8,16}/   400 input images each
└── leftImg8bit/val/                        300 input images (no ground truth)
```

All patches are 672×378. Masks contain only pixel values 0 (road) and 1 (crack).
The official validation split ships without labels, so local validation uses a
scene split that holds out one patch per source frame, keeping images and
training patches from the same flight while avoiding patch overlap.

## Environment

RTX 50-series cards are Blackwell (sm_120) and require CUDA 12.8 wheels.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128
$env:HF_ENDPOINT = "https://hf-mirror.com"   # optional, for smp encoder weights
```

## Usage

```powershell
# single model, 5-fold cross validation
python -m uavcrack.train --arch unet --encoder resnet34 --kfold 5 --fold 0 `
    --epochs 50 --batch-size 8 --accum-steps 2 --num-workers 4 --run-dir runs/fold0

# threshold sweep on the local validation split
python -m uavcrack.evaluate --ckpt runs/fold0/best.pth --tta

# ensemble inference over the 300 official val images
python -m uavcrack.ensemble_infer --ckpts <checkpoint paths> `
    --threshold 0.56 --batch-size 4 --out-dir submission/result

# package and validate
python -m uavcrack.package_result
python -m uavcrack.check_submission
```

Upload `submission/result.zip` on the platform's Submit Task page.

## Implementation notes

- **Padding**: 378 is not divisible by 32, so inputs are reflect-padded to 672×384
  and cropped back after the forward pass (`uavcrack/model.py`).
- **Loss**: `BCEWithLogits(pos_weight=3)` + soft Dice, which suits masks where
  cracks cover roughly 6% of the pixels.
- **Augmentation**: flips, small affine transforms, brightness/contrast and mild
  noise. Elastic deformation is avoided because it distorts thin cracks.
- **Unicode paths**: images are decoded from byte buffers rather than
  `cv2.imread`, which fails on non-ASCII paths on Windows.
- **Submission format**: masks are single-channel PNGs with values exactly 0/1,
  named after the input jpg, inside a `result/` folder in the zip.

## Results

Leaderboard entry `CHD-UNetR34`: **82.58 mIoU**, 3rd out of 82 methods
(2026-10-29). Crack F1 80.65, Crack IoU 67.58, precision 79.63, recall 81.70,
aAcc 97.70.

The submission is a **51-model ensemble** at threshold **0.56** with four-way
flip TTA:

| Group | Count | Architecture | Training |
|---|---:|---|---|
| A | 5 | UNet + ResNet34 | 5-fold CV |
| B | 5 | UNet++ + ResNet34 | 5-fold CV |
| C | 6 | UNet/UNet++ + ResNet34/ResNet50 | full data, EMA |
| D | 5 | SegFormer + MiT-B2 | 5-fold CV, EMA |
| E | 5 | SegFormer + MiT-B2 | 5-fold CV, EMA, second seed |
| F | 5 | UNet + ConvNeXt-Tiny | 5-fold CV, EMA |
| G | 5 | UNet + ConvNeXt-Small | 5-fold CV, EMA |
| H | 5 | UNet + ConvNeXt-Small | 5-fold CV, EMA, second seed |
| I | 5 | UNet + ConvNeXt-Base | 5-fold CV, EMA |
| J | 5 | UNet + ConvNeXt-Base | 5-fold CV, EMA, second seed |

## Reproducibility

See [REPRODUCE.md](REPRODUCE.md) for the exact commands used for every model
group and for the ensemble inference.

- Model weights (Baidu Netdisk, permanent share):
  <https://pan.baidu.com/s/1wG8yhJtFOOf-ki3Gl2DRjQ?pwd=dzdn> — extraction code `dzdn`

  The folder contains all 51 checkpoints and 7 ONNX exports:

  | File | Size |
  |---|---:|
  | `uavcrack_weights_part1.zip` … `part5.zip` | 1.7–1.8 GB each |
  | `uavcrack_onnx.zip` | 1.09 GB |

  Extract the five parts into the same directory to obtain `checkpoints/`.
  Each part also contains `manifest.json` with the architecture, fold, epoch and
  validation metrics of every model.
- ONNX export (required by the competition rules for PyTorch models):

  ```powershell
  python -m uavcrack.export_onnx --ckpt runs/fold0/best.pth --out weights/model.onnx --verify
  ```

## License

Released under the MIT License, see [LICENSE](LICENSE).

The UAV-Crack dataset is distributed under CC BY-NC-SA 3.0. It is **not**
included in this repository and was used for non-commercial competition
purposes only.
