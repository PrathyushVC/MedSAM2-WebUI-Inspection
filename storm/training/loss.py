"""
STORM loss functions.

Loss components and when they apply
-------------------------------------
contrastive     : NT-Xent (CLIP-style) between image and omics projections.
                  Applied only where BOTH modalities are present as inputs
                  (has_both mask). Needs ≥ 2 such spots per batch.

omics_recon     : MSE between pred_omics and ground-truth gene expression.
                  Applied where ONLY image is present (image-only spots) so
                  the model must hallucinate omics from image information alone.

omics_recon_reg : Same prediction, but applied on both-present spots as a
                  soft regulariser (weight typically << w_omics_recon).

img_feat_recon  : Cosine similarity between pred_img_feat and the detached
                  clean image embedding. Applied where ONLY omics is present.

Total loss = w_contrastive * contrastive
           + w_omics_recon * omics_recon
           + w_omics_recon_reg * omics_recon_reg
           + w_img_feat_recon * img_feat_recon
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Individual loss modules
# ---------------------------------------------------------------------------

class NTXentLoss(nn.Module):
    """
    Symmetric NT-Xent (InfoNCE) contrastive loss between two sets of embeddings.

    The temperature is a learnable log-scalar clamped to [0.01, 100] to prevent
    training instability.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.log_temp = nn.Parameter(torch.tensor(temperature).log())

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temp.exp().clamp(min=0.01, max=100.0)

    def forward(
        self,
        img_proj: torch.Tensor,
        omics_proj: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            img_proj:   (N, D) image projections for N matched spots.
            omics_proj: (N, D) omics projections for the same N spots.
        Returns:
            Scalar contrastive loss.
        """
        assert img_proj.shape == omics_proj.shape
        img_proj = F.normalize(img_proj, dim=-1)
        omics_proj = F.normalize(omics_proj, dim=-1)

        logits = torch.matmul(img_proj, omics_proj.T) / self.temperature  # (N, N)
        labels = torch.arange(len(logits), device=logits.device)

        loss_i2o = F.cross_entropy(logits, labels)
        loss_o2i = F.cross_entropy(logits.T, labels)
        return (loss_i2o + loss_o2i) / 2.0


class OmicsReconLoss(nn.Module):
    """
    MSE reconstruction loss for gene expression.
    Only computes on spots indicated by the boolean mask.
    """

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred:   (B, G) predicted expression values.
            target: (B, G) ground-truth expression values.
            mask:   (B,) bool — True ⟹ include this spot in the loss.
        Returns:
            Scalar MSE loss (zero-gradient zero if no spots selected).
        """
        if not mask.any():
            return pred.sum() * 0.0
        return F.mse_loss(pred[mask], target[mask], reduction="mean")


class ImageFeatReconLoss(nn.Module):
    """
    Cosine-distance loss for reconstructing image features from the other modality.
    Target embeddings are detached so gradients only flow through the predictor.
    """

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred:   (B, D) predicted image feature vectors.
            target: (B, D) clean image embeddings from the image encoder.
            mask:   (B,) bool — True ⟹ include this spot.
        Returns:
            Scalar cosine-distance loss in [0, 2].
        """
        if not mask.any():
            return pred.sum() * 0.0
        pred_n = F.normalize(pred[mask], dim=-1)
        tgt_n = F.normalize(target[mask].detach(), dim=-1)
        # 1 − cosine similarity ∈ [0, 2]; 0 when perfect alignment
        return (1.0 - (pred_n * tgt_n).sum(dim=-1)).mean()


# ---------------------------------------------------------------------------
# Combined STORM loss
# ---------------------------------------------------------------------------

class STORMLoss(nn.Module):
    """
    Weighted sum of all STORM training objectives.

    Args:
        w_contrastive:      Weight for NT-Xent loss (both-present spots).
        w_omics_recon:      Weight for omics reconstruction (image-only spots).
        w_omics_recon_reg:  Weight for omics reconstruction regulariser (both spots).
        w_img_feat_recon:   Weight for image-feature reconstruction (omics-only spots).
        temperature:        Initial contrastive temperature.
    """

    def __init__(
        self,
        w_contrastive: float = 1.0,
        w_omics_recon: float = 0.5,
        w_omics_recon_reg: float = 0.1,
        w_img_feat_recon: float = 0.5,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.w_contrastive = w_contrastive
        self.w_omics_recon = w_omics_recon
        self.w_omics_recon_reg = w_omics_recon_reg
        self.w_img_feat_recon = w_img_feat_recon

        self.contrastive = NTXentLoss(temperature)
        self.omics_recon = OmicsReconLoss()
        self.img_feat_recon = ImageFeatReconLoss()

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        omics_gt: torch.Tensor,
        drop_image: torch.Tensor,
        drop_omics: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute the total STORM loss.

        Args:
            outputs:    Dict returned by STORMModel.forward().
            omics_gt:   (B, G) ground-truth gene expression (from the DataLoader,
                        always available regardless of dropout).
            drop_image: (B,) bool dropout flags for the image modality.
            drop_omics: (B,) bool dropout flags for the omics modality.

        Returns:
            total_loss:  Scalar tensor (with gradient).
            components:  Dict of named scalar values for logging.
        """
        has_image = ~drop_image
        has_omics = ~drop_omics
        has_both = has_image & has_omics
        only_image = has_image & ~has_omics   # omics was dropped → must predict it
        only_omics = ~has_image & has_omics   # image was dropped → must predict it
        # has_neither = drop_image & drop_omics → excluded from all losses

        device = omics_gt.device
        total = torch.zeros(1, device=device, requires_grad=False).squeeze()
        components: Dict[str, float] = {}

        # ------------------------------------------------------------------
        # 1. Contrastive loss — spots where both modalities fed as real input
        # ------------------------------------------------------------------
        if has_both.sum() > 1:
            loss_c = self.contrastive(
                outputs["img_proj"][has_both],
                outputs["omics_proj"][has_both],
            )
            components["contrastive"] = loss_c.item()
            total = total + self.w_contrastive * loss_c
        else:
            components["contrastive"] = 0.0

        # ------------------------------------------------------------------
        # 2. Omics reconstruction — image-only spots
        # ------------------------------------------------------------------
        if only_image.any() and self.w_omics_recon > 0:
            loss_or = self.omics_recon(outputs["pred_omics"], omics_gt, only_image)
            components["omics_recon"] = loss_or.item()
            total = total + self.w_omics_recon * loss_or
        else:
            components["omics_recon"] = 0.0

        # ------------------------------------------------------------------
        # 3. Omics reconstruction regulariser — both-present spots
        # ------------------------------------------------------------------
        if has_both.any() and self.w_omics_recon_reg > 0:
            loss_or_reg = self.omics_recon(outputs["pred_omics"], omics_gt, has_both)
            components["omics_recon_reg"] = loss_or_reg.item()
            total = total + self.w_omics_recon_reg * loss_or_reg
        else:
            components["omics_recon_reg"] = 0.0

        # ------------------------------------------------------------------
        # 4. Image feature reconstruction — omics-only spots
        # ------------------------------------------------------------------
        if only_omics.any() and self.w_img_feat_recon > 0:
            loss_if = self.img_feat_recon(
                outputs["pred_img_feat"],
                outputs["img_embed"],  # detached inside the loss
                only_omics,
            )
            components["img_feat_recon"] = loss_if.item()
            total = total + self.w_img_feat_recon * loss_if
        else:
            components["img_feat_recon"] = 0.0

        components["total"] = total.item()
        return total, components
