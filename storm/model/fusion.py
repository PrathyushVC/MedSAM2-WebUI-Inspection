from __future__ import annotations

import torch
import torch.nn as nn


class _CrossAttentionBlock(nn.Module):
    """Single bidirectional cross-attention block with pre-norm FFN residuals.

    Args:
        embed_dim: Token embedding dimension.
        num_heads: Number of attention heads.
        mlp_ratio: FFN hidden width multiplier.
        dropout: Dropout probability.
    """

    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        dim_ff = int(embed_dim * mlp_ratio)

        self.img_norm = nn.LayerNorm(embed_dim)
        self.omics_norm = nn.LayerNorm(embed_dim)
        self.img_cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.omics_cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

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

    def forward(self, img: torch.Tensor, omics: torch.Tensor):
        img_n, omics_n = self.img_norm(img), self.omics_norm(omics)
        img_cross, _ = self.img_cross_attn(img_n, omics_n, omics_n)
        img = img + img_cross + self.img_ffn(img + img_cross)
        omics_cross, _ = self.omics_cross_attn(omics_n, img_n, img_n)
        omics = omics + omics_cross + self.omics_ffn(omics + omics_cross)
        return img, omics


class CrossModalFusion(nn.Module):
    """Stacked bidirectional cross-attention fusion for image and omics modalities.

    Spots with a dropped modality have their embedding replaced by a learnable
    mask token before fusion, so a valid fused representation is always produced.
    The per-spot scenario (both present, image-only, omics-only, or neither) is
    determined externally via the boolean dropout flags passed to ``forward``.

    Args:
        embed_dim: Shared embedding dimension for both modalities.
        num_heads: Attention heads per cross-attention block.
        depth: Number of stacked cross-attention blocks.
        mlp_ratio: FFN hidden width multiplier.
        dropout: Dropout probability.
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
        self.image_mask_token = nn.Parameter(torch.zeros(embed_dim))
        self.omics_mask_token = nn.Parameter(torch.zeros(embed_dim))
        self.blocks = nn.ModuleList([
            _CrossAttentionBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.fusion_proj = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
        )
        nn.init.normal_(self.image_mask_token, std=0.02)
        nn.init.normal_(self.omics_mask_token, std=0.02)

    def forward(
        self,
        img_feat: torch.Tensor,
        omics_feat: torch.Tensor,
        drop_image: torch.Tensor,
        drop_omics: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse image and omics embeddings with modality-dropout awareness.

        Dropped modalities are replaced with learned mask tokens before
        attention, so the network learns to reconstruct missing information
        from whichever modality is present.

        Args:
            img_feat: Image embeddings of shape ``(B, D)``, always computed
                from real data regardless of dropout.
            omics_feat: Omics embeddings of shape ``(B, D)``, same caveat.
            drop_image: Boolean tensor of shape ``(B,)``; ``True`` means the
                image modality is masked for that spot.
            drop_omics: Boolean tensor of shape ``(B,)``; ``True`` means the
                omics modality is masked for that spot.

        Returns:
            Fused representation of shape ``(B, D)``.
        """
        img_m = img_feat.clone()
        omics_m = omics_feat.clone()
        img_m[drop_image] = self.image_mask_token.to(dtype=img_feat.dtype)
        omics_m[drop_omics] = self.omics_mask_token.to(dtype=omics_feat.dtype)

        img_m = img_m.unsqueeze(1)
        omics_m = omics_m.unsqueeze(1)
        for block in self.blocks:
            img_m, omics_m = block(img_m, omics_m)

        return self.fusion_proj(torch.cat([img_m.squeeze(1), omics_m.squeeze(1)], dim=-1))
