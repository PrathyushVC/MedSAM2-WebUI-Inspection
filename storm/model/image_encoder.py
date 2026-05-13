from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from monai.networks.nets import ViT


class ImageEncoder(nn.Module):
    """Vision transformer encoder for histology patch spots.

    Uses MONAI's ViT as the default backbone. When ``pretrained_backbone`` is
    supplied, a timm model is used instead and its output is projected linearly
    to ``embed_dim``.

    Args:
        img_size: Square spatial resolution of the input patch in pixels.
        patch_size: ViT patch size; ``img_size`` must be divisible by this.
        in_chans: Number of input image channels.
        embed_dim: Output embedding dimension.
        depth: Number of transformer encoder layers.
        num_heads: Number of attention heads per layer.
        mlp_ratio: FFN hidden width as a multiple of ``embed_dim``.
        dropout: Dropout probability applied throughout.
        pretrained_backbone: Optional timm model name (e.g.
            ``"vit_base_patch16_224"``). When set, MONAI ViT is bypassed and
            the timm model is used as a frozen or fine-tunable feature extractor.
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
            self._mode = "backbone"
            try:
                import timm
            except ImportError as exc:
                raise ImportError("timm is required when pretrained_backbone is set") from exc
            self.backbone = timm.create_model(pretrained_backbone, pretrained=True, num_classes=0)
            backbone_dim = self.backbone.num_features
            self.proj = (
                nn.Linear(backbone_dim, embed_dim)
                if backbone_dim != embed_dim
                else nn.Identity()
            )
        else:
            self._mode = "vit"
            self.vit = ViT(
                in_channels=in_chans,
                img_size=(img_size, img_size),
                patch_size=(patch_size, patch_size),
                hidden_size=embed_dim,
                mlp_dim=int(embed_dim * mlp_ratio),
                num_layers=depth,
                num_heads=num_heads,
                pos_embed="conv",
                classification=True,
                num_classes=embed_dim,
                dropout_rate=dropout,
                spatial_dims=2,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a batch of image patches to fixed-size embeddings.

        Args:
            x: Float32 tensor of shape ``(B, C, H, W)`` with values in ``[0, 1]``.

        Returns:
            Embedding tensor of shape ``(B, embed_dim)``.
        """
        if self._mode == "backbone":
            return self.proj(self.backbone(x))
        out, _ = self.vit(x)
        return out
