"""
Swin-T backbone wrapper for Deformable DETR.

SwinTransformerBackbone wraps timm's swin_tiny_patch4_window7_224 with
features_only=True to extract 4 feature maps at strides [4, 8, 16, 32]
with channels [96, 192, 384, 768].

It is self-contained (includes position encoding) and returns the same
(list[NestedTensor], list[pos]) interface as Joiner, so it can be used
as a drop-in replacement for the ResNet Joiner in build_backbone().
"""

import math
import os
import sys
from typing import Dict, List, Optional

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add PROJECT_ROOT to sys.path so msca_module can be imported
_SWIN_DIR = os.path.dirname(os.path.abspath(__file__))   # models/
_DDETR_ROOT = os.path.dirname(_SWIN_DIR)                  # deformable_detr/
PROJECT_ROOT = os.path.dirname(_DDETR_ROOT)               # master-project/
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from util.misc import NestedTensor
from msca_module import MSCA
from .position_encoding import PositionEmbeddingSine


class SwinTransformerBackbone(nn.Module):
    """
    Swin-T feature extractor + position encoding in one module.

    Interface mirrors Joiner — forward() returns (out, pos) where
    out is a list[NestedTensor] and pos is a list of position tensors.

    Attributes
    ----------
    num_channels : list[int]  [96, 192, 384, 768]
    strides      : list[int]  [4, 8, 16, 32]
    """

    num_channels: List[int] = [96, 192, 384, 768]
    strides: List[int] = [4, 8, 16, 32]

    def __init__(
        self,
        position_embedding: Optional[nn.Module] = None,
        pretrained_path: Optional[str] = None,
        use_msca: bool = False,
        msca_reduction: int = 16,
        msca_common_size: int = 32,
        hidden_dim: int = 256,
    ):
        super().__init__()

        self.swin = timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=False,
            features_only=True,
            out_indices=(0, 1, 2, 3),
            img_size=None,
            strict_img_size=False,
            dynamic_img_size=True,
        )

        if pretrained_path is not None:
            self._load_pretrained(pretrained_path)

        # Build default sine position embedding when none is provided
        if position_embedding is None:
            position_embedding = PositionEmbeddingSine(hidden_dim // 2, normalize=True)
        self.position_embedding = position_embedding

        self.msca = None
        if use_msca:
            self.msca = MSCA(
                channels_per_scale=self.num_channels,
                reduction=msca_reduction,
                common_size=msca_common_size,
            )

        msca_label = "ENABLED" if use_msca else "DISABLED"
        print(f"[Backbone] Swin-T loaded | MSCA: {msca_label}")

    def _load_pretrained(self, pretrained_path: str) -> None:
        ckpt = torch.load(pretrained_path, map_location="cpu")
        state_dict = ckpt.get("model", ckpt)
        missing, unexpected = self.swin.load_state_dict(state_dict, strict=False)
        print(f"[Swin-T] Pretrained loaded from {pretrained_path}")
        if missing:
            print(f"  Missing  ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")
        if unexpected:
            print(f"  Unexpected ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")

    # ------------------------------------------------------------------
    # Allow backbone[1] access (used by DeformableDETR for extra levels)
    # ------------------------------------------------------------------
    def __getitem__(self, idx: int) -> nn.Module:
        if idx == 1:
            return self.position_embedding
        raise IndexError(f"SwinTransformerBackbone: index {idx} is out of range")

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors  # [B, 3, H, W]

        # Pad to make H and W divisible by 32
        _, _, H, W = x.shape
        pad_h = (32 - H % 32) % 32
        pad_w = (32 - W % 32) % 32
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))

        # timm Swin outputs BHWC — permute each stage to BCHW
        raw_feats = self.swin(x)
        feature_maps = [f.permute(0, 3, 1, 2).contiguous() for f in raw_feats]

        if self.msca is not None:
            feature_maps = self.msca(feature_maps)

        # Pack into NestedTensors, propagating the input mask
        m = tensor_list.mask
        out: List[NestedTensor] = []
        for feat in feature_maps:
            if m is not None:
                mask = F.interpolate(
                    m[None].float(), size=feat.shape[-2:]
                ).to(torch.bool)[0]
            else:
                mask = torch.zeros(
                    feat.shape[0], feat.shape[2], feat.shape[3],
                    dtype=torch.bool, device=feat.device,
                )
            out.append(NestedTensor(feat, mask))

        pos = [self.position_embedding(nt).to(nt.tensors.dtype) for nt in out]
        return out, pos
