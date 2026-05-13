"""
Cross-modal attention fusion module.

Handles missing modalities by substituting a learnable mask token for any
dropped modality before fusion, so the fused representation is always defined
regardless of which modalities are available.

Dropout scenarios handled:
  has_image & has_omics  → both real embeddings attend each other
  has_image & ~has_omics → image embedding + omics mask token
  ~has_image & has_omics → image mask token + omics embedding
  ~has_image & ~has_omics → both mask tokens (both-dropped; excluded from loss)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _CrossAttentionBlock(nn.Module):
    """Single bidirectional cross-attention + FFN block."""

    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        dim_ff = int(embed_dim * mlp_ratio)

        self.img_norm = nn.LayerNorm(embed_dim)
        self.omics_norm = nn.LayerNorm(embed_dim)

        # Image queries, omics keys/values
        self.img_cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        # Omics queries, image keys/values
        self.omics_cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )

        self.img_ffn = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, embed_dim),
            nn.Dropout(dropout),
        )
        self.omics_ffn = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        img: torch.Tensor,    # (B, 1, D)
        omics: torch.Tensor,  # (B, 1, D)
    ):
        img_n = self.img_norm(img)
        omics_n = self.omics_norm(omics)

        img_cross, _ = self.img_cross_attn(img_n, omics_n, omics_n)
        img = img + img_cross
        img = img + self.img_ffn(img)

        omics_cross, _ = self.omics_cross_attn(omics_n, img_n, img_n)
        omics = omics + omics_cross
        omics = omics + self.omics_ffn(omics)

        return img, omics


class CrossModalFusion(nn.Module):
    """
    Stacked bidirectional cross-attention between image and omics modalities.

    Replaces dropped modality embeddings with learned mask tokens before fusion
    so the rest of the network always receives a valid (B, embed_dim) tensor.

    Args:
        embed_dim:  Shared embedding dimension.
        num_heads:  Number of attention heads per block.
        depth:      Number of cross-attention blocks.
        mlp_ratio:  FFN expansion factor.
        dropout:    Dropout probability.
    """

    def __init__(
        self,
        embed_dim: int = 512,
        num_heads: int = 8,
        depth: int = 2,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # Learnable mask tokens substituted for dropped modalities
        self.image_mask_token = nn.Parameter(torch.zeros(embed_dim))
        self.omics_mask_token = nn.Parameter(torch.zeros(embed_dim))

        self.blocks = nn.ModuleList([
            _CrossAttentionBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        # Project concatenated [img_out, omics_out] → embed_dim
        self.fusion_proj = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
        )

        nn.init.normal_(self.image_mask_token, std=0.02)
        nn.init.normal_(self.omics_mask_token, std=0.02)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def _apply_mask_tokens(
        self,
        img_feat: torch.Tensor,
        omics_feat: torch.Tensor,
        drop_image: torch.Tensor,
        drop_omics: torch.Tensor,
    ):
        """Replace dropped modality features with learned mask tokens."""
        img_out = img_feat.clone()
        omics_out = omics_feat.clone()
        img_out[drop_image] = self.image_mask_token.to(dtype=img_feat.dtype)
        omics_out[drop_omics] = self.omics_mask_token.to(dtype=omics_feat.dtype)
        return img_out, omics_out

    def forward(
        self,
        img_feat: torch.Tensor,
        omics_feat: torch.Tensor,
        drop_image: torch.Tensor,
        drop_omics: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            img_feat:   (B, D) image embeddings (always encoded from real data).
            omics_feat: (B, D) omics embeddings (always encoded from real data).
            drop_image: (B,) bool — True means image modality is masked for this spot.
            drop_omics: (B,) bool — True means omics modality is masked for this spot.
        Returns:
            fused: (B, D) fused representation.
        """
        img_m, omics_m = self._apply_mask_tokens(img_feat, omics_feat, drop_image, drop_omics)

        # Add sequence dimension for MultiheadAttention
        img_m = img_m.unsqueeze(1)      # (B, 1, D)
        omics_m = omics_m.unsqueeze(1)  # (B, 1, D)

        for block in self.blocks:
            img_m, omics_m = block(img_m, omics_m)

        img_m = img_m.squeeze(1)    # (B, D)
        omics_m = omics_m.squeeze(1)  # (B, D)

        fused = self.fusion_proj(torch.cat([img_m, omics_m], dim=-1))
        return fused
