"""
Tree Detection — fine-tune Deformable DETR (+ optional MSCA) on a COCO-format
tree dataset.  Uses the existing Deformable DETR engine for the training loop.

Supported backbones: resnet50, swin_t

Dataset structure expected at --coco_path:
  coco_path/
    train/images/   train/coco_annotations_clean.json
    val/images/     val/coco_annotations_clean.json
    test/images/    test/coco_annotations_clean.json
"""

import argparse
import math
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader

# ── locate deformable_detr repo and project root ──────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _SCRIPT_DIR.parent.parent          # master-project/
DDETR_ROOT = PROJECT_ROOT / "deformable_detr"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(DDETR_ROOT))

import util.misc as utils
import datasets.transforms as T
from datasets import get_coco_api_from_dataset
from datasets.coco import CocoDetection, ConvertCocoPolysToMask
from engine import evaluate, train_one_epoch
from models import build_model


# ── CLAHE transform ───────────────────────────────────────────────────────────

class CLAHETransform:
    """Apply CLAHE to the L channel of LAB colorspace.
    Accepts (PIL Image, target) and returns (PIL Image, target).
    """

    def __init__(self, clip_limit: float = 2.0, tile_grid_size: int = 8):
        self.clahe = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=(tile_grid_size, tile_grid_size)
        )

    def __call__(self, img, target):
        img_np = np.array(img)
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        lab[:, :, 0] = self.clahe.apply(lab[:, :, 0])
        img_np = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return Image.fromarray(img_np), target


# ── Dataset ───────────────────────────────────────────────────────────────────

class TreeCocoDetection(CocoDetection):
    """CocoDetection subclass for tree dataset path conventions."""

    def _find_ann_file(self, split_dir: Path) -> Path:
        # Try both annotation filename variants
        for name in ['coco_annotations_clean.json', 'coco_annotations_cleaned.json']:
            ann_file = split_dir / name
            if ann_file.exists():
                return ann_file
        raise FileNotFoundError(
            f"No annotation file found in {split_dir}. "
            f"Expected 'coco_annotations_clean.json' or 'coco_annotations_cleaned.json'."
        )

    def __init__(self, coco_path: Path, split: str, transforms, use_clahe: bool = True,
                 clip_limit: float = 2.0, tile_grid_size: int = 8):
        img_folder = coco_path / split / "images"
        ann_file = self._find_ann_file(coco_path / split)
        super().__init__(
            img_folder=str(img_folder),
            ann_file=str(ann_file),
            transforms=transforms,
            return_masks=False,
        )
        self.clahe_transform = CLAHETransform(clip_limit, tile_grid_size) if use_clahe else None

    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)
        # Remap category_ids to 0-indexed class labels
        # COCO category_id=1 (Tree) -> class index 0
        if 'labels' in target:
            target['labels'] = target['labels'] - 1
        return img, target


# ── Transform factory ─────────────────────────────────────────────────────────

def make_tree_transforms(image_set: str, use_clahe: bool, cfg_clahe: dict):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    clahe_xf = (
        CLAHETransform(cfg_clahe["clip_limit"], cfg_clahe["tile_grid_size"])
        if use_clahe else None
    )

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == "train":
        spatial_xf = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1333),
                ])
            ),
        ])
    else:
        spatial_xf = T.RandomResize([800], max_size=1333)

    def combined(img, target):
        if clahe_xf is not None:
            img, target = clahe_xf(img, target)
        img, target = spatial_xf(img, target)
        img, target = normalize(img, target)
        return img, target

    return combined


# ── Dataset builder (supports multiple paths via ConcatDataset) ───────────────

def build_split_dataset(coco_paths, split, transforms, use_clahe, clahe_cfg):
    datasets = []
    for coco_path in coco_paths:
        dataset = TreeCocoDetection(
            coco_path=Path(coco_path),
            split=split,
            transforms=transforms,
            use_clahe=use_clahe,
            clip_limit=clahe_cfg["clip_limit"],
            tile_grid_size=clahe_cfg["tile_grid_size"],
        )
        datasets.append(dataset)
    if len(datasets) == 1:
        return datasets[0]
    return ConcatDataset(datasets)


# ── Pretrained-weight loader ──────────────────────────────────────────────────

def load_pretrained_weights(model, ckpt_path: str):
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model", checkpoint)

    model_state = model.state_dict()
    skip_keys, load_keys = [], []
    filtered = {}
    for k, v in state_dict.items():
        if k in model_state and model_state[k].shape == v.shape:
            filtered[k] = v
            load_keys.append(k)
        else:
            skip_keys.append(k)

    model.load_state_dict(filtered, strict=False)
    print(f"[Pretrained] Loaded {len(load_keys)} keys, skipped {len(skip_keys)} keys.")
    if skip_keys:
        print(f"  Skipped: {skip_keys[:10]}{'...' if len(skip_keys) > 10 else ''}")


# ── Args builder ──────────────────────────────────────────────────────────────

def build_detr_args(cfg, cli_args, epochs, batch_size, num_workers):
    """Construct an argparse.Namespace that Deformable DETR's engine expects."""
    # CLI overrides config; config provides defaults
    backbone = getattr(cli_args, 'backbone', None) or cfg["model"].get("backbone", "resnet50")
    pretrained_backbone = getattr(cli_args, 'pretrained_backbone', None) or cfg["model"].get("pretrained_backbone")

    return argparse.Namespace(
        # backbone
        backbone=backbone,
        pretrained_backbone=pretrained_backbone,
        dilation=False,
        position_embedding="sine",
        position_embedding_scale=2 * math.pi,
        hidden_dim=cfg["model"]["hidden_dim"],
        num_feature_levels=cfg["model"]["num_feature_levels"],
        # transformer
        enc_layers=cfg["model"]["enc_layers"],
        dec_layers=cfg["model"]["dec_layers"],
        dim_feedforward=cfg["model"]["dim_feedforward"],
        dropout=0.1,
        nheads=cfg["model"]["nheads"],
        num_queries=cfg["model"]["num_queries"],
        dec_n_points=cfg["model"]["dec_n_points"],
        enc_n_points=cfg["model"]["enc_n_points"],
        # model flags
        masks=False,
        aux_loss=True,
        with_box_refine=cfg["model"]["with_box_refine"],
        two_stage=cfg["model"]["two_stage"],
        num_classes=cfg["model"]["num_classes"],
        # MSCA
        use_msca=cli_args.use_msca,
        msca_reduction=cfg["msca"]["reduction"],
        msca_common_size=cfg["msca"]["common_size"],
        # loss weights
        set_cost_class=2.0,
        set_cost_bbox=5.0,
        set_cost_giou=2.0,
        mask_loss_coef=1.0,
        dice_loss_coef=1.0,
        cls_loss_coef=2.0,
        bbox_loss_coef=5.0,
        giou_loss_coef=2.0,
        focal_alpha=0.25,
        # dataset (unused by build_model, only matters if we call build_dataset)
        dataset_file="coco",
        coco_path=str(cli_args.coco_path),
        remove_difficult=False,
        cache_mode=False,
        # training
        lr=cfg["training"]["lr"],
        lr_backbone=cfg["training"]["lr_backbone"],
        lr_backbone_names=["backbone.0"],
        lr_linear_proj_names=["reference_points", "sampling_offsets"],
        lr_linear_proj_mult=0.1,
        weight_decay=cfg["training"]["weight_decay"],
        epochs=epochs,
        lr_drop=cfg["training"]["lr_drop"],
        lr_drop_epochs=None,
        clip_max_norm=cfg["training"]["clip_max_norm"],
        batch_size=batch_size,
        num_workers=num_workers,
        seed=cfg["training"]["seed"],
        sgd=False,
        device=cli_args.device,
        output_dir=str(cli_args.output_dir),
        resume=cli_args.resume or "",
        start_epoch=0,
        eval=False,
        frozen_weights=None,
        distributed=False,
        override_resumed_lr_drop=False,
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser("Tree Detection — Deformable DETR trainer")
    p.add_argument('--coco_path', type=str, required=True,
                   help='One or more dataset roots separated by commas. '
                        'Each must contain train/val/test folders with images/ and coco_annotations_clean.json. '
                        'Example: /path/to/part_1/tree,/path/to/part_2/tree')
    p.add_argument("--output_dir", required=True, type=Path)
    p.add_argument("--backbone", default=None, choices=["resnet50", "swin_t"],
                   help="Backbone to use (overrides config)")
    p.add_argument("--pretrained_backbone", default=None, type=str,
                   help="Path to Swin-T ImageNet pretrained .pth file")
    p.add_argument("--use_msca", action="store_true")
    p.add_argument("--no_clahe", action="store_true")
    p.add_argument("--pretrained", default=None, type=str,
                   help="Path to COCO-pretrained Deformable DETR .pth for fine-tuning")
    p.add_argument("--resume", default=None, type=str,
                   help="Resume from checkpoint")
    p.add_argument("--epochs", default=None, type=int)
    p.add_argument("--batch_size", default=None, type=int)
    p.add_argument("--device", default="cuda")
    p.add_argument("--num_workers", default=None, type=int)
    p.add_argument('--local-rank', type=int, default=0,
                   help='Local rank for distributed training (set automatically by torch.distributed.launch)')
    return p.parse_args()


def main():
    cli = parse_args()
    coco_paths = [p.strip() for p in cli.coco_path.split(',')]

    cfg_path = Path(__file__).parent / "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    epochs = cli.epochs if cli.epochs is not None else cfg["training"]["epochs"]
    batch_size = cli.batch_size if cli.batch_size is not None else cfg["training"]["batch_size"]
    num_workers = cli.num_workers if cli.num_workers is not None else cfg["training"]["num_workers"]
    use_clahe = not cli.no_clahe and cfg["clahe"]["use_clahe"]

    args = build_detr_args(cfg, cli, epochs, batch_size, num_workers)

    # Derive W&B-friendly run name from config
    msca_tag = "msca" if cli.use_msca else "no_msca"
    clahe_tag = "clahe" if use_clahe else "no_clahe"
    run_name = f"{args.backbone}_{msca_tag}_{clahe_tag}"

    # Print startup banner
    msca_label = "ENABLED" if cli.use_msca else "DISABLED"
    clahe_label = "ENABLED" if use_clahe else "DISABLED"
    print("=" * 60)
    print("  Tree Detection — Deformable DETR")
    print("=" * 60)
    print(f"  Backbone    : {args.backbone}")
    print(f"  MSCA        : {msca_label}")
    print(f"  CLAHE       : {clahe_label}")
    print(f"  Dataset     : {coco_paths[0]}")
    for _p in coco_paths[1:]:
        print(f"                {_p}")
    print(f"  Output      : {cli.output_dir}")
    print(f"  Classes     : {cfg['model']['num_classes']} (Tree)")
    print(f"  Epochs      : {epochs}")
    print(f"  Batch size  : {batch_size}")
    print(f"  Pretrained  : {cli.pretrained or 'None'}")
    print(f"  Device      : {cli.device}")
    print(f"  Run name    : {run_name}")
    print("=" * 60)

    cli.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(cli.device)

    # Initialize distributed training if launched with torch.distributed.launch
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        torch.distributed.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=rank,
        )
        torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
        distributed = True
    else:
        rank = 0
        distributed = False

    seed = cfg["training"]["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Build model
    model, criterion, postprocessors = build_model(args)
    model.to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    if cli.pretrained:
        load_pretrained_weights(model, cli.pretrained)

    # Datasets
    clahe_cfg = cfg["clahe"]
    train_xf = make_tree_transforms("train", use_clahe, clahe_cfg)
    val_xf = make_tree_transforms("val", use_clahe, clahe_cfg)

    dataset_train = build_split_dataset(coco_paths, "train", train_xf,
                                        use_clahe=False, clahe_cfg=clahe_cfg)
    dataset_val = build_split_dataset(coco_paths, "val", val_xf,
                                      use_clahe=False, clahe_cfg=clahe_cfg)

    if distributed:
        sampler_train = torch.utils.data.distributed.DistributedSampler(dataset_train)
        sampler_val = torch.utils.data.distributed.DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, batch_size, drop_last=True)

    data_loader_train = DataLoader(
        dataset_train, batch_sampler=batch_sampler_train,
        collate_fn=utils.collate_fn, num_workers=num_workers, pin_memory=True)
    data_loader_val = DataLoader(
        dataset_val, batch_size, sampler=sampler_val,
        drop_last=False, collate_fn=utils.collate_fn,
        num_workers=num_workers, pin_memory=True)

    base_ds = get_coco_api_from_dataset(dataset_val)
    if base_ds is None:
        # ConcatDataset — use first sub-dataset's COCO API
        ds = dataset_val.datasets[0]
        for _ in range(10):
            if hasattr(ds, 'dataset'):
                ds = ds.dataset
            else:
                break
        base_ds = ds.coco

    # Optimiser
    def match_name_keywords(n, keywords):
        return any(kw in n for kw in keywords)

    param_dicts = [
        {
            "params": [p for n, p in model.named_parameters()
                       if not match_name_keywords(n, args.lr_backbone_names)
                       and not match_name_keywords(n, args.lr_linear_proj_names)
                       and p.requires_grad],
            "lr": args.lr,
        },
        {
            "params": [p for n, p in model.named_parameters()
                       if match_name_keywords(n, args.lr_backbone_names) and p.requires_grad],
            "lr": args.lr_backbone,
        },
        {
            "params": [p for n, p in model.named_parameters()
                       if match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
            "lr": args.lr * args.lr_linear_proj_mult,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    # Resume
    start_epoch = 0
    best_map = 0.0
    if cli.resume:
        ckpt = torch.load(cli.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"], strict=False)
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "lr_scheduler" in ckpt:
            lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
        start_epoch = ckpt.get("epoch", -1) + 1
        best_map = ckpt.get("best_map", 0.0)
        print(f"[Resume] Epoch {start_epoch}, best mAP so far: {best_map:.4f}")

    log_path = cli.output_dir / "log.txt"

    print("Start training")
    for epoch in range(start_epoch, epochs):
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch, args.clip_max_norm)
        lr_scheduler.step()

        if not distributed or rank == 0:
            train_loss = train_stats.get("loss", 0.0)

            # Checkpoint: always save latest
            ckpt_data = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch,
                "best_map": best_map,
                "args": args,
            }
            torch.save(ckpt_data, cli.output_dir / "checkpoint.pth")

            # Save periodic checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                torch.save(ckpt_data, cli.output_dir / f"checkpoint_epoch_{epoch + 1}.pth")
                print(f"[Rank 0] Saved periodic checkpoint at epoch {epoch + 1}")

            # Append log line
            log_line = f"epoch={epoch}  train_loss={train_loss:.4f}\n"
            with open(log_path, "a") as f:
                f.write(log_line)

            print(f"Epoch {epoch}/{epochs - 1} — loss {train_loss:.4f}")

    print("Training complete.")

    # Destroy process group after ALL epochs complete
    if distributed:
        torch.cuda.empty_cache()
        try:
            torch.distributed.barrier()
        except Exception:
            pass
        try:
            torch.distributed.destroy_process_group()
        except Exception:
            pass
        if rank != 0:
            sys.exit(0)

    # Evaluation — only in single GPU mode
    # In distributed mode, run evaluate.py separately after training
    if not distributed:
        test_stats, coco_evaluator = evaluate(
            model, criterion, postprocessors, data_loader_val, base_ds, device, str(cli.output_dir))
        if coco_evaluator is not None:
            bbox_stats = coco_evaluator.coco_eval["bbox"].stats
            val_map5095 = float(bbox_stats[0]) if len(bbox_stats) > 0 else 0.0
            val_map50 = float(bbox_stats[1]) if len(bbox_stats) > 1 else 0.0
            print(f"\nmAP@50:95 : {val_map5095:.4f}")
            print(f"mAP@50    : {val_map50:.4f}")

            if val_map5095 > best_map:
                best_map = val_map5095
                ckpt_data = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "epoch": epochs - 1,
                    "best_map": best_map,
                    "args": args,
                }
                torch.save(ckpt_data, cli.output_dir / "checkpoint_best.pth")
                print(f"  [*] New best mAP@50:95 = {best_map:.4f}")

            with open(log_path, "a") as f:
                f.write(f"final_val_mAP50_95={val_map5095:.4f}  final_val_mAP50={val_map50:.4f}\n")
    else:
        print("\nDistributed training complete. Run evaluate.py for full evaluation.")

    # Clean shutdown
    if distributed and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
