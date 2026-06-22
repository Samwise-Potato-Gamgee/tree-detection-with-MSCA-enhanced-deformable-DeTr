import torch
import torch.nn as nn
import torch.nn.functional as F


class MSCCA(nn.Module):
    """Multi-Scale Coupled Channel Attention.

    All scales are projected to a shared bottleneck via scale-specific input
    projections, coupled through a single shared FC, then gated back per scale.
    """

    def __init__(self, channels_per_scale: list, reduction: int = 16):
        super().__init__()
        self.channels_per_scale = channels_per_scale
        bottleneck_dim = channels_per_scale[0] // reduction
        self.input_projs = nn.ModuleList([
            nn.Linear(c, bottleneck_dim) for c in channels_per_scale
        ])
        # Shared FC — same weights used for every scale's bottleneck
        self.shared_fc = nn.Linear(bottleneck_dim, bottleneck_dim)
        self.output_projs = nn.ModuleList([
            nn.Linear(bottleneck_dim, c) for c in channels_per_scale
        ])
        self.relu = nn.ReLU(inplace=True)

    def forward(self, feature_maps: list) -> list:
        # Project each scale to the shared bottleneck dimension
        bottlenecks = []
        for i, x in enumerate(feature_maps):
            desc = x.mean(dim=[2, 3])                          # [B, C_i]
            desc = self.relu(self.input_projs[i](desc))        # [B, bottleneck]
            bottlenecks.append(desc)

        # Couple: sum across scales so each scale sees global multi-scale context
        coupled = sum(bottlenecks)                             # [B, bottleneck]

        # Apply shared FC (same weights regardless of scale)
        coupled = self.shared_fc(coupled)                      # [B, bottleneck]

        # Scale-specific output projections and sigmoid gating
        outputs = []
        for i, x in enumerate(feature_maps):
            attn = torch.sigmoid(self.output_projs[i](coupled))  # [B, C_i]
            outputs.append(x * attn.unsqueeze(-1).unsqueeze(-1))
        return outputs


class MSCSA(nn.Module):
    """Multi-Scale Coupled Spatial Attention.

    Resizes all scales to a common resolution, concatenates, applies a
    depthwise conv to produce coupled spatial attention, then resizes back.
    """

    def __init__(self, channels_per_scale: list, common_size: int = 32):
        super().__init__()
        self.common_size = common_size
        self.channels_per_scale = channels_per_scale
        total_channels = sum(channels_per_scale)
        self.dw_conv = nn.Conv2d(
            total_channels, total_channels,
            kernel_size=3, padding=1, groups=total_channels
        )

    def forward(self, feature_maps: list) -> list:
        # Resize all scales to common_size x common_size
        resized = [
            F.interpolate(x, size=(self.common_size, self.common_size),
                          mode='bilinear', align_corners=False)
            for x in feature_maps
        ]

        # Concat and apply depthwise conv: [B, sum(C_i), H, W]
        cat = torch.cat(resized, dim=1)
        attn_common = self.dw_conv(cat)

        # Split, resize to original resolution, sigmoid gate
        outputs = []
        ch_start = 0
        for x, c in zip(feature_maps, self.channels_per_scale):
            attn = attn_common[:, ch_start:ch_start + c, :, :]
            attn = F.interpolate(attn, size=x.shape[-2:], mode='bilinear', align_corners=False)
            attn = torch.sigmoid(attn)
            outputs.append(x * attn)
            ch_start += c
        return outputs


class MSCA(nn.Module):
    """Multi-Scale Coupled Attention: MSCCA followed by MSCSA."""

    def __init__(self, channels_per_scale: list, reduction: int = 16, common_size: int = 32):
        super().__init__()
        self.mscca = MSCCA(channels_per_scale, reduction)
        self.mscsa = MSCSA(channels_per_scale, common_size)

    def forward(self, feature_maps: list) -> list:
        feature_maps = self.mscca(feature_maps)
        feature_maps = self.mscsa(feature_maps)
        return feature_maps
