from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from .fusion import CrossModalFusion
from .image_encoder import ImageEncoder
from .omics_encoder import OmicsEncoder


class STORMModel(nn.Module):
    """Dual-encoder cross-modal fusion model for spatial transcriptomics.

    Both modalities are always encoded from their real inputs on each forward
    pass. The clean embeddings drive the contrastive projection heads; the
    fusion module then receives mask-token substitutions for whichever
    modality was designated as dropped, training robustness to missing data.

    Args:
        num_genes: Number of genes in the expression panel.
        embed_dim: Shared embedding dimension across all sub-modules.
        img_size: Square spatial resolution of input image patches in pixels.
        patch_size: ViT patch size; must evenly divide ``img_size``.
        img_encoder_depth: Number of transformer layers in the image encoder.
        img_encoder_heads: Attention heads in the image encoder.
        omics_encoder_depth: Number of transformer layers in the omics encoder.
        omics_encoder_heads: Attention heads in the omics encoder.
        omics_hidden_dim: MLP bottleneck width in the omics encoder (bulk mode).
        fusion_depth: Number of cross-attention blocks in the fusion module.
        fusion_heads: Attention heads in the fusion module.
        dropout: Dropout probability shared across all sub-modules.
        pretrained_image_backbone: Optional timm model name for the image encoder.
        use_gene_tokens: When ``True``, each gene becomes its own omics token.
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
        self.fusion = CrossModalFusion(
            embed_dim=embed_dim,
            num_heads=fusion_heads,
            depth=fusion_depth,
            dropout=dropout,
        )
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
        self.omics_predictor = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.LayerNorm(embed_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, num_genes),
        )
        self.image_feat_predictor = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.LayerNorm(embed_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
        )

    def forward(
        self,
        images: torch.Tensor,
        omics: torch.Tensor,
        drop_image: torch.Tensor,
        drop_omics: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Run a full forward pass with modality-dropout masking.

        Args:
            images: Float32 image patches of shape ``(B, 3, H, W)``.
            omics: Float32 log-normalised expression matrix of shape ``(B, G)``.
            drop_image: Boolean mask of shape ``(B,)``; ``True`` means the image
                modality is dropped (replaced by mask token in fusion) for
                that spot.
            drop_omics: Boolean mask of shape ``(B,)``; same semantics as
                ``drop_image`` for the omics modality.

        Returns:
            Dictionary with keys:

            - ``img_embed``: ``(B, D)`` clean image embeddings.
            - ``omics_embed``: ``(B, D)`` clean omics embeddings.
            - ``img_proj``: ``(B, D)`` image projections for contrastive loss.
            - ``omics_proj``: ``(B, D)`` omics projections for contrastive loss.
            - ``fused``: ``(B, D)`` mask-token-aware fused representation.
            - ``pred_omics``: ``(B, G)`` gene expression predicted from fused.
            - ``pred_img_feat``: ``(B, D)`` image features predicted from fused.
        """
        img_embed = self.image_encoder(images)
        omics_embed = self.omics_encoder(omics)
        fused = self.fusion(img_embed, omics_embed, drop_image, drop_omics)
        return {
            "img_embed": img_embed,
            "omics_embed": omics_embed,
            "img_proj": self.img_proj(img_embed),
            "omics_proj": self.omics_proj(omics_embed),
            "fused": fused,
            "pred_omics": self.omics_predictor(fused),
            "pred_img_feat": self.image_feat_predictor(fused),
        }

    def encode_image_only(self, images: torch.Tensor) -> torch.Tensor:
        """Return fused embeddings for a batch where omics is entirely absent.

        Args:
            images: Float32 image patches of shape ``(B, 3, H, W)``.

        Returns:
            Fused embeddings of shape ``(B, D)``.
        """
        B, device = images.shape[0], images.device
        dummy_omics = torch.zeros(B, self.num_genes, device=device)
        out = self.forward(
            images, dummy_omics,
            drop_image=torch.zeros(B, dtype=torch.bool, device=device),
            drop_omics=torch.ones(B, dtype=torch.bool, device=device),
        )
        return out["fused"]

    def encode_omics_only(self, omics: torch.Tensor) -> torch.Tensor:
        """Return fused embeddings for a batch where the image is entirely absent.

        Args:
            omics: Float32 expression matrix of shape ``(B, G)``.

        Returns:
            Fused embeddings of shape ``(B, D)``.
        """
        B, device = omics.shape[0], omics.device
        dummy_images = torch.zeros(B, 3, 224, 224, device=device)
        out = self.forward(
            dummy_images, omics,
            drop_image=torch.ones(B, dtype=torch.bool, device=device),
            drop_omics=torch.zeros(B, dtype=torch.bool, device=device),
        )
        return out["fused"]
