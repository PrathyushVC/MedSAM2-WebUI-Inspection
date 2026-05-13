"""
Full STORM model: image encoder + omics encoder + cross-modal fusion.

The forward pass always encodes both modalities from their real inputs so that:
  - Clean embeddings are available for the contrastive loss.
  - The fusion module receives proper ground-truth features to substitute with
    mask tokens for the modality-dropout-aware fused representation.
  - Cross-modal prediction heads (omics from image, image-feat from omics) are
    trained from the fused representation.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from .image_encoder import ImageEncoder
from .omics_encoder import OmicsEncoder
from .fusion import CrossModalFusion


class STORMModel(nn.Module):
    """
    Dual-encoder + cross-modal fusion model for spatial transcriptomics.

    Args:
        num_genes:                Number of genes in the expression panel.
        embed_dim:                Shared embedding dimension for all modules.
        img_size:                 Input image patch size (pixels, square assumed).
        patch_size:               ViT patch size (only used when no backbone).
        img_encoder_depth:        ViT depth for image encoder.
        img_encoder_heads:        ViT attention heads for image encoder.
        omics_encoder_depth:      Transformer depth for omics encoder.
        omics_encoder_heads:      Attention heads for omics encoder.
        omics_hidden_dim:         MLP width inside omics encoder (bulk mode).
        fusion_depth:             Cross-attention blocks in fusion module.
        fusion_heads:             Attention heads in fusion module.
        dropout:                  Shared dropout probability.
        pretrained_image_backbone: timm model name for a pretrained image encoder.
        use_gene_tokens:          If True, each gene is its own transformer token.
    """

    def __init__(
        self,
        num_genes: int,
        embed_dim: int = 512,
        img_size: int = 224,
        patch_size: int = 16,
        img_encoder_depth: int = 6,
        img_encoder_heads: int = 8,
        omics_encoder_depth: int = 4,
        omics_encoder_heads: int = 8,
        omics_hidden_dim: int = 1024,
        fusion_depth: int = 2,
        fusion_heads: int = 8,
        dropout: float = 0.1,
        pretrained_image_backbone: Optional[str] = None,
        use_gene_tokens: bool = False,
    ):
        super().__init__()
        self.num_genes = num_genes
        self.embed_dim = embed_dim

        # ------------------------------------------------------------------
        # Modality encoders
        # ------------------------------------------------------------------
        self.image_encoder = ImageEncoder(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=img_encoder_depth,
            num_heads=img_encoder_heads,
            dropout=dropout,
            pretrained_backbone=pretrained_image_backbone,
        )
        self.omics_encoder = OmicsEncoder(
            num_genes=num_genes,
            embed_dim=embed_dim,
            hidden_dim=omics_hidden_dim,
            depth=omics_encoder_depth,
            num_heads=omics_encoder_heads,
            dropout=dropout,
            use_gene_tokens=use_gene_tokens,
        )

        # ------------------------------------------------------------------
        # Fusion
        # ------------------------------------------------------------------
        self.fusion = CrossModalFusion(
            embed_dim=embed_dim,
            num_heads=fusion_heads,
            depth=fusion_depth,
            dropout=dropout,
        )

        # ------------------------------------------------------------------
        # Projection heads for contrastive loss (separate from backbone)
        # ------------------------------------------------------------------
        self.img_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.omics_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # ------------------------------------------------------------------
        # Cross-modal prediction heads (trained via reconstruction losses)
        # ------------------------------------------------------------------
        # Predict gene expression from fused representation
        self.omics_predictor = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.LayerNorm(embed_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, num_genes),
        )
        # Predict image embedding from fused representation
        self.image_feat_predictor = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.LayerNorm(embed_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        images: torch.Tensor,
        omics: torch.Tensor,
        drop_image: torch.Tensor,
        drop_omics: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            images:     (B, 3, H, W) float32 image patches.
            omics:      (B, G) float32 log-normalised gene expression.
            drop_image: (B,) bool — image modality dropped (masked) for this spot.
            drop_omics: (B,) bool — omics modality dropped (masked) for this spot.

        Returns:
            Dictionary containing:
              img_embed:      (B, D) clean image embeddings (pre-masking).
              omics_embed:    (B, D) clean omics embeddings (pre-masking).
              img_proj:       (B, D) image projection for contrastive loss.
              omics_proj:     (B, D) omics projection for contrastive loss.
              fused:          (B, D) fused representation (mask-token aware).
              pred_omics:     (B, G) gene expression predicted from fused.
              pred_img_feat:  (B, D) image features predicted from fused.
        """
        # Always encode from real data — clean embeddings used for contrastive
        img_embed = self.image_encoder(images)    # (B, D)
        omics_embed = self.omics_encoder(omics)   # (B, D)

        # Contrastive projection heads
        img_proj = self.img_proj(img_embed)
        omics_proj = self.omics_proj(omics_embed)

        # Fuse with mask-token substitution for dropped modalities
        fused = self.fusion(img_embed, omics_embed, drop_image, drop_omics)

        # Cross-modal predictions from fused representation
        pred_omics = self.omics_predictor(fused)         # (B, G)
        pred_img_feat = self.image_feat_predictor(fused) # (B, D)

        return {
            "img_embed": img_embed,
            "omics_embed": omics_embed,
            "img_proj": img_proj,
            "omics_proj": omics_proj,
            "fused": fused,
            "pred_omics": pred_omics,
            "pred_img_feat": pred_img_feat,
        }

    def encode_image_only(self, images: torch.Tensor) -> torch.Tensor:
        """Inference helper: encode images with omics masked out entirely."""
        B = images.shape[0]
        device = images.device
        dummy_omics = torch.zeros(B, self.num_genes, device=device)
        drop_image = torch.zeros(B, dtype=torch.bool, device=device)
        drop_omics = torch.ones(B, dtype=torch.bool, device=device)
        out = self.forward(images, dummy_omics, drop_image, drop_omics)
        return out["fused"]

    def encode_omics_only(self, omics: torch.Tensor) -> torch.Tensor:
        """Inference helper: encode omics with image masked out entirely."""
        B = omics.shape[0]
        device = omics.device
        dummy_images = torch.zeros(B, 3, 224, 224, device=device)
        drop_image = torch.ones(B, dtype=torch.bool, device=device)
        drop_omics = torch.zeros(B, dtype=torch.bool, device=device)
        out = self.forward(dummy_images, omics, drop_image, drop_omics)
        return out["fused"]
