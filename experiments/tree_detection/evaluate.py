"""
Tree Detection — evaluate a checkpoint on train / val / test split.

Usage:
  python evaluate.py \\
    --coco_path /data/trees \\
    --checkpoint /runs/tree/checkpoint_best.pth \\
    --backbone swin_t \\
    --use_msca \\
    --split test
"""

import argparse
import math
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _SCRIPT_DIR.parent.parent
DDETR_ROOT = PROJECT_ROOT / "deformable_detr"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(DDETR_ROOT))

import util.misc as utils
from datasets import get_coco_api_from_dataset
from engine import evaluate as detr_evaluate
from models import build_model

from train import (
    CLAHETransform,
    TreeCocoDetection,
    build_detr_args,
    make_tree_transforms,
)


def parse_args():
    p = argparse.ArgumentParser("Tree Detection — Evaluator")
    p.add_argument("--coco_path", required=True, type=Path)
    p.add_argument("--checkpoint", required=True, type=str)
    p.add_argument("--backbone", default=None, choices=["resnet50", "swin_t"],
                   help="Backbone used at training time (overrides config)")
    p.add_argument("--pretrained_backbone", default=None, type=str)
    p.add_argument("--use_msca", action="store_true")
    p.add_argument("--no_clahe", action="store_true")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main():
    cli = parse_args()

    cfg_path = Path(__file__).parent / "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    use_clahe = not cli.no_clahe and cfg["clahe"]["use_clahe"]

    # Build a minimal cli-like namespace for build_detr_args
    import argparse as _ap
    dummy_cli = _ap.Namespace(
        backbone=cli.backbone,
        pretrained_backbone=cli.pretrained_backbone,
        use_msca=cli.use_msca,
        coco_path=cli.coco_path,
        output_dir=Path("."),
        resume=None,
        device=cli.device,
    )
    args = build_detr_args(
        cfg, dummy_cli,
        epochs=cfg["training"]["epochs"],
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["training"]["num_workers"],
    )

    device = torch.device(cli.device)

    model, criterion, postprocessors = build_model(args)

    ckpt = torch.load(cli.checkpoint, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model", ckpt)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)

    # Remap predicted 0-indexed labels back to COCO category_ids (1-indexed)
    # before the COCO evaluator sees them, to match GT annotations (category_id=1)
    _orig_bbox_pp = postprocessors['bbox']
    def _remapped_bbox_pp(outputs, target_sizes):
        results = _orig_bbox_pp(outputs, target_sizes)
        for output in results:
            output['labels'] = output['labels'] + 1
        return results
    postprocessors['bbox'] = _remapped_bbox_pp

    val_xf = make_tree_transforms(cli.split, use_clahe, cfg["clahe"])
    dataset = TreeCocoDetection(cli.coco_path, cli.split, val_xf, use_clahe=False)
    sampler = torch.utils.data.SequentialSampler(dataset)
    data_loader = DataLoader(
        dataset,
        batch_size=cfg["training"]["batch_size"],
        sampler=sampler,
        drop_last=False,
        collate_fn=utils.collate_fn,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=(cli.device != "cpu"),
    )
    base_ds = get_coco_api_from_dataset(dataset)

    test_stats, coco_evaluator = detr_evaluate(
        model, criterion, postprocessors, data_loader, base_ds, device, "."
    )

    bbox_stats = coco_evaluator.coco_eval["bbox"].stats if coco_evaluator else [0] * 12
    backbone_label = args.backbone
    msca_label = "ENABLED" if cli.use_msca else "DISABLED"
    clahe_label = "ENABLED" if use_clahe else "DISABLED"

    line = "─" * 40
    print(line)
    print("  Tree Detection — Evaluation Results")
    print(line)
    print(f"  Split      : {cli.split}")
    print(f"  Checkpoint : {cli.checkpoint}")
    print(f"  Backbone   : {backbone_label}")
    print(f"  MSCA       : {msca_label}")
    print(f"  CLAHE      : {clahe_label}")
    print(line)
    print(f"  mAP @ IoU=0.50:0.95   : {bbox_stats[0]:.4f}")
    print(f"  mAP @ IoU=0.50        : {bbox_stats[1]:.4f}")
    print(f"  mAP @ IoU=0.75        : {bbox_stats[2]:.4f}")
    print(f"  mAR @ max=100         : {bbox_stats[8]:.4f}")
    print(line)


if __name__ == "__main__":
    main()
