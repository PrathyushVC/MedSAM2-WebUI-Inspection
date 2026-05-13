"""
Vision transformer image encoder for histology patch spots.

Implements a lightweight ViT that can also wrap a timm/torchvision pretrained
backbone when one is specified via `pretrained_backbone`.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """Splits an image into non-overlapping patches and linearly projects each."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
    ):
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, C, H, W) → (B, N, E)
        return self.proj(x).flatten(2).transpose(1, 2)


class ImageEncoder(nn.Module):
    """
    ViT-based encoder that maps a (B, 3, H, W) image patch to (B, embed_dim).

    When `pretrained_backbone` is set (e.g. 'vit_base_patch16_224') the timm
    model is used as a drop-in feature extractor and its output is projected to
    `embed_dim`. Otherwise a lightweight ViT is trained from scratch.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 512,
        depth: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        pretrained_backbone: Optional[str] = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        if pretrained_backbone is not None:
            self._build_from_backbone(pretrained_backbone, embed_dim)
        else:
            self._build_vit(
                img_size, patch_size, in_chans, embed_dim,
                depth, num_heads, mlp_ratio, dropout,
            )

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def _build_vit(
        self, img_size, patch_size, in_chans, embed_dim,
        depth, num_heads, mlp_ratio, dropout,
    ):
        self._mode = "vit"

        self.patch_embed = PatchEmbedding(img_size, patch_size, in_chans, embed_dim)
        N = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, N + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)

        self._init_weights()

    def _build_from_backbone(self, backbone_name: str, embed_dim: int):
        try:
            import timm
        except ImportError as e:
            raise ImportError(
                "timm is required for pretrained_backbone. "
                "Install with: pip install timm"
            ) from e

        self._mode = "backbone"
        self.backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0)
        backbone_dim = self.backbone.num_features
        self.proj = (
            nn.Linear(backbone_dim, embed_dim)
            if backbone_dim != embed_dim
            else nn.Identity()
        )

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) float32 in [0, 1]
        Returns:
            (B, embed_dim) image embeddings
        """
        if self._mode == "backbone":
            return self.proj(self.backbone(x))

        B = x.shape[0]
        tokens = self.patch_embed(x)                      # (B, N, E)
        cls = self.cls_token.expand(B, -1, -1)            # (B, 1, E)
        tokens = torch.cat([cls, tokens], dim=1)          # (B, N+1, E)
        tokens = self.pos_drop(tokens + self.pos_embed)
        tokens = self.transformer(tokens)
        tokens = self.norm(tokens)
        return tokens[:, 0]                               # CLS token (B, E)
