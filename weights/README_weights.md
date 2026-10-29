# CHD-UNetR34 — 2026 VLP 挑战赛参赛作品

UAV-Crack binary crack segmentation. Best leaderboard score: **82.58 mIoU**
(2026-10-29, rank 3/82).

## Download

Baidu Netdisk (permanent): <https://pan.baidu.com/s/1wG8yhJtFOOf-ki3Gl2DRjQ?pwd=dzdn>
extraction code `dzdn`.

The folder contains five weight parts and one ONNX bundle:

| File | Size | Content |
|---|---:|---|
| `uavcrack_weights_part1.zip` … `part5.zip` | 1.7–1.8 GB each | 51 checkpoints in total |
| `uavcrack_onnx.zip` | 1.09 GB | seven ONNX exports |

Extract the five parts into the same directory to obtain `checkpoints/`.

## Ensemble

| Group | Count | Architecture | Training |
|---|---:|---|---|
| A | 5 | UNet + ResNet34 | 5-fold CV |
| B | 5 | UNet++ + ResNet34 | 5-fold CV |
| C | 6 | UNet/UNet++ + ResNet34/ResNet50 | full data, EMA |
| D | 5 | SegFormer + MiT-B2 | 5-fold CV, EMA |
| E | 5 | SegFormer + MiT-B2 | second seed |
| F | 5 | UNet + ConvNeXt-Tiny | 5-fold CV, EMA |
| G | 5 | UNet + ConvNeXt-Small | 5-fold CV, EMA |
| H | 5 | UNet + ConvNeXt-Small | second seed |
| I | 5 | UNet + ConvNeXt-Base | 5-fold CV, EMA |
| J | 5 | UNet + ConvNeXt-Base | second seed |

Inference averages the sigmoid probabilities of all 51 models with four-way flip
TTA and binarises at threshold **0.56**. Predicted crack ratio on the official
300 val images: 6.03% (training ground truth averages 6.17%).

## Checkpoints

`checkpoints/` contains every `.pth` used by the ensemble. `manifest.json` lists
each model's architecture, encoder, fold, epoch and validation metrics.

```python
from uavcrack.model import CrackSegModel
import torch

ckpt = torch.load("checkpoints/fold0_best.pth", map_location="cpu", weights_only=False)
model = CrackSegModel(ckpt["arch"], ckpt["encoder"], encoder_weights=None)
model.load_state_dict(ckpt["model"])
model.eval()
```

## ONNX

`onnx/` contains one export per architecture family, all with fixed input
`(1, 3, 378, 672)` and output logits `(1, 1, 378, 672)`. Apply sigmoid and
threshold 0.56 to obtain the binary mask.

```python
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("onnx/unet_r34_s42.onnx", providers=["CPUExecutionProvider"])
logits = session.run(None, {"image": image})[0]   # float32, (1,3,378,672)
mask = (1 / (1 + np.exp(-logits)) > 0.56).astype(np.uint8)
```

Exports were verified against PyTorch; the maximum absolute logit difference is
below 1.7e-5.

## Data licence

UAV-Crack is distributed under CC BY-NC-SA 3.0 and is **not** included in this
package. It was used for non-commercial competition purposes only.
