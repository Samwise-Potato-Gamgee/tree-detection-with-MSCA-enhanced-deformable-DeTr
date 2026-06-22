# Tree Detection with MSCA-Enhanced Deformable DETR

Off-road tree detection using Deformable DETR enhanced with **MSCA (Multi-Scale Coupled Attention)** and **CLAHE preprocessing**. Trained and evaluated on the RPTU forest real dataset.

---

## Overview

This project enhances Deformable DETR for tree detection in off-road forest scenes. The key contributions are:

- **MSCA (Multi-Scale Coupled Attention)** — a plug-in attention module inserted between the ResNet-50 backbone and the Deformable DETR transformer. It couples channel and spatial attention across all backbone scales simultaneously, improving feature discrimination in cluttered forest scenes.
- **CLAHE preprocessing** — Contrast Limited Adaptive Histogram Equalization applied online during training to handle extreme lighting variation (sun glare, deep shadows) common in forest imagery.
- **Fine-tuning from COCO** — the full Deformable DETR model (backbone + transformer) is initialized from COCO pretrained weights and fine-tuned on the trees dataset.

### Architecture

```
Input Image
    ↓
CLAHE (L channel of LAB colorspace)
    ↓
ResNet-50 backbone (pretrained on COCO)
    ↓
MSCA Module
  ├── MSCCA: Multi-Scale Coupled Channel Attention
  └── MSCSA: Multi-Scale Coupled Spatial Attention
    ↓
Deformable Transformer Encoder / Decoder
    ↓
Detection Head (bounding boxes + scores)
```

---

## Project Structure

```
master-project/
├── deformable_detr/              # Deformable DETR (modified)
│   ├── models/
│   │   ├── backbone.py           # Modified: MSCA injected after ResNet
│   │   ├── deformable_detr.py    # Modified: dynamic num_classes support
│   │   └── ops/                  # Custom CUDA multi-scale deformable attention
│   ├── datasets/
│   ├── engine.py
│   └── util/
├── msca_module/
│   ├── __init__.py
│   └── msca.py                   # MSCCA + MSCSA + MSCA implementation
├── experiments/
│   └── tree_detection/
│       ├── config.yaml           # All hyperparameters
│       ├── train.py              # Training script
│       └── evaluate.py           # Evaluation script
├── scripts/
│   └── download_swin_weights.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Setup

### 1. Clone the repository

```bash
git clone <your-repo-url> master-project
cd master-project
```

### 2. Create conda environment

```bash
conda create --prefix ./env python=3.11 -y
conda activate ./env
```

### 3. Install PyTorch (CUDA 12.4)

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Compile Deformable DETR CUDA ops

This must be compiled on each machine before training. Requires CUDA toolkit on PATH.

```bash
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

cd deformable_detr/models/ops
sh make.sh
python test.py  # verify compilation
cd ../../..
```

### 6. Download COCO pretrained weights (for fine-tuning)

Download `r50_deformable_detr-checkpoint.pth` from the official Deformable DETR release and place it in `pretrained/`:

```bash
mkdir -p pretrained
# Download from: https://github.com/fundamentalvision/Deformable-DETR
# Place at: pretrained/deformable_detr_r50_coco.pth
```

---

## Dataset Structure

Each dataset split must follow this structure:

```
<coco_path>/
├── train/
│   ├── images/
│   └── coco_annotations_clean.json
├── val/
│   ├── images/
│   └── coco_annotations_clean.json
└── test/
    ├── images/
    └── coco_annotations_clean.json
```

Annotation files must follow COCO instance format with `category_id = 1` for trees.

Multiple dataset paths can be combined with commas:

```
--coco_path /path/to/part_1/tree,/path/to/part_2/tree
```

---

## Training

**Baseline — ResNet-50, no MSCA, no CLAHE:**

```bash
python experiments/tree_detection/train.py \
  --coco_path /path/to/dataset \
  --output_dir runs/baseline \
  --backbone resnet50 \
  --no_clahe
```

**Full model — ResNet-50 + MSCA + CLAHE + COCO pretrained:**

```bash
python experiments/tree_detection/train.py \
  --coco_path /path/to/dataset \
  --output_dir runs/msca_clahe \
  --backbone resnet50 \
  --use_msca \
  --pretrained pretrained/deformable_detr_r50_coco.pth \
  --epochs 100
```

**Multi-GPU training (4 GPUs):**

```bash
python -m torch.distributed.launch --nproc_per_node=4 \
  experiments/tree_detection/train.py \
  --coco_path /path/to/dataset \
  --output_dir runs/msca_clahe \
  --backbone resnet50 \
  --use_msca \
  --pretrained pretrained/deformable_detr_r50_coco.pth \
  --epochs 100 \
  --batch_size 2 \
  --num_workers 4 \
  --device cuda
```

**Resume from checkpoint:**

```bash
python -m torch.distributed.launch --nproc_per_node=4 \
  experiments/tree_detection/train.py \
  --coco_path /path/to/dataset \
  --output_dir runs/msca_clahe \
  --backbone resnet50 \
  --use_msca \
  --pretrained pretrained/deformable_detr_r50_coco.pth \
  --resume runs/msca_clahe/checkpoint_epoch_40.pth \
  --epochs 100
```

### Key training arguments

| Argument | Default | Description |
|---|---|---|
| `--coco_path` | required | Dataset root (comma-separated for multiple) |
| `--output_dir` | required | Where to save checkpoints and logs |
| `--backbone` | resnet50 | Backbone architecture |
| `--use_msca` | False | Enable MSCA attention module |
| `--no_clahe` | False | Disable CLAHE preprocessing |
| `--pretrained` | None | COCO pretrained Deformable DETR weights |
| `--resume` | None | Resume from checkpoint |
| `--epochs` | 50 | Number of training epochs |
| `--batch_size` | 2 | Per-GPU batch size |

---

## Evaluation

```bash
python experiments/tree_detection/evaluate.py \
  --checkpoint runs/msca_clahe/checkpoint_epoch_100.pth \
  --coco_path /path/to/dataset \
  --backbone resnet50 \
  --use_msca \
  --split test \
  --device cuda
```

### Key evaluation arguments

| Argument | Default | Description |
|---|---|---|
| `--checkpoint` | required | Path to trained checkpoint |
| `--coco_path` | required | Dataset root |
| `--backbone` | resnet50 | Must match training backbone |
| `--use_msca` | False | Must match training setting |
| `--no_clahe` | False | Must match training setting |
| `--split` | test | Evaluation split (train/val/test) |

---

## Checkpointing

The training script saves:
- `checkpoint.pth` — latest epoch (overwritten every epoch)
- `checkpoint_best.pth` — best val mAP (single GPU mode only)
- `checkpoint_epoch_N.pth` — saved every 10 epochs
- `log.txt` — one line per epoch with loss and mAP

In distributed training, run `evaluate.py` after training for full COCO evaluation.

---

## MSCA Module

The MSCA module (`msca_module/msca.py`) contains two sub-modules:

**MSCCA — Multi-Scale Coupled Channel Attention:**
Global average pools each scale, projects to a shared bottleneck via a shared FC layer (the coupling), then scales back per-channel with sigmoid gating.

**MSCSA — Multi-Scale Coupled Spatial Attention:**
Resizes all scales to a common resolution, concatenates and applies depthwise convolution, resizes attention maps back and applies sigmoid gating.

```python
from msca_module import MSCA
import torch

msca = MSCA(channels_per_scale=[512, 1024, 2048], reduction=16, common_size=32)
feats = [torch.randn(1, 512, 64, 64), torch.randn(1, 1024, 32, 32), torch.randn(1, 2048, 16, 16)]
out = msca(feats)
# out: same shapes as input, recalibrated
```

---

## Notes

- **Distributed evaluation** is disabled during multi-GPU training to avoid NCCL timeouts. Run `evaluate.py` separately after training.
- **Category IDs**: annotations use `category_id=1` (1-indexed). The dataset class remaps to 0-indexed internally.
- **CLAHE** is applied to the L channel of LAB colorspace online during data loading.
- The CUDA ops must be recompiled on each machine — compiled `.so` files are not portable.
