from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXentLoss(nn.Module):
    """Symmetric NT-Xent (InfoNCE) contrastive loss for matched embedding pairs.

    The temperature is a learnable log-scalar, clamped to ``[0.01, 100]`` to
    avoid numerical instability at the extremes.

    Args:
        temperature: Initial contrastive temperature value.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.log_temp = nn.Parameter(torch.tensor(temperature).log())

    @property
    def temperature(self) -> torch.Tensor:
        """Clamped temperature derived from the learnable log parameter."""
        return self.log_temp.exp().clamp(min=0.01, max=100.0)

    def forward(self, img_proj: torch.Tensor, omics_proj: torch.Tensor) -> torch.Tensor:
        """Compute the symmetric contrastive loss between two embedding sets.

        Args:
            img_proj: L2-normalised image projections of shape ``(N, D)``.
            omics_proj: L2-normalised omics projections of shape ``(N, D)``,
                matched one-to-one with ``img_proj``.

        Returns:
            Scalar contrastive loss.
        """
        img_proj = F.normalize(img_proj, dim=-1)
        omics_proj = F.normalize(omics_proj, dim=-1)
        logits = torch.matmul(img_proj, omics_proj.T) / self.temperature
        labels = torch.arange(len(logits), device=logits.device)
        return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2.0


class OmicsReconLoss(nn.Module):
    """Masked MSE loss for reconstructing gene expression profiles.

    Only spots selected by the boolean ``mask`` contribute to the loss,
    enabling scenario-specific supervision (e.g. only image-only spots).
    """

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute masked mean-squared error over expression profiles.

        Args:
            pred: Predicted expression values of shape ``(B, G)``.
            target: Ground-truth expression values of shape ``(B, G)``.
            mask: Boolean tensor of shape ``(B,)``; ``True`` = include this spot.

        Returns:
            Scalar MSE loss; differentiably zero when no spots are selected.
        """
        if not mask.any():
            return pred.sum() * 0.0
        return F.mse_loss(pred[mask], target[mask], reduction="mean")


class ImageFeatReconLoss(nn.Module):
    """Masked cosine-distance loss for reconstructing image feature vectors.

    The target embeddings are detached so gradients flow only through the
    predictor, not back into the image encoder.
    """

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute masked cosine-distance loss between predicted and target features.

        Args:
            pred: Predicted feature vectors of shape ``(B, D)``.
            target: Clean image embeddings of shape ``(B, D)`` from the image
                encoder; detached inside this method.
            mask: Boolean tensor of shape ``(B,)``; ``True`` = include this spot.

        Returns:
            Scalar loss in ``[0, 2]``; zero at perfect alignment.
        """
        if not mask.any():
            return pred.sum() * 0.0
        pred_n = F.normalize(pred[mask], dim=-1)
        tgt_n = F.normalize(target[mask].detach(), dim=-1)
        return (1.0 - (pred_n * tgt_n).sum(dim=-1)).mean()


class STORMLoss(nn.Module):
    """Combined STORM training objective.

    Routes each loss component to the appropriate subset of spots based on
    which modalities are available as inputs:

    - **contrastive**: NT-Xent between image and omics projections, applied
      only where both modalities are present (needs ≥ 2 such spots).
    - **omics_recon**: MSE between predicted and actual gene expression, applied
      to image-only spots (the model must hallucinate omics from image alone).
    - **omics_recon_reg**: Same prediction on both-present spots as a soft
      regulariser; weight is typically much smaller than ``w_omics_recon``.
    - **img_feat_recon**: Cosine distance between predicted and actual image
      features, applied to omics-only spots.

    Spots where both modalities are dropped contribute no gradient.

    Args:
        w_contrastive: Weight for the NT-Xent loss term.
        w_omics_recon: Weight for the omics reconstruction term.
        w_omics_recon_reg: Weight for the omics reconstruction regulariser.
        w_img_feat_recon: Weight for the image feature reconstruction term.
        temperature: Initial contrastive temperature.
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
        """Compute the total STORM loss and its named components.

        Args:
            outputs: Output dict from ``STORMModel.forward()``.
            omics_gt: Ground-truth expression matrix of shape ``(B, G)``.
                Always available from the DataLoader regardless of dropout.
            drop_image: Boolean dropout flags of shape ``(B,)`` for images.
            drop_omics: Boolean dropout flags of shape ``(B,)`` for omics.

        Returns:
            A tuple ``(total_loss, components)`` where ``total_loss`` is a
            differentiable scalar and ``components`` is a dict of named float
            values suitable for logging.
        """
        has_image = ~drop_image
        has_omics = ~drop_omics
        has_both = has_image & has_omics
        only_image = has_image & ~has_omics
        only_omics = ~has_image & has_omics

        total = torch.zeros(1, device=omics_gt.device).squeeze()
        components: Dict[str, float] = {}

        if has_both.sum() > 1:
            loss_c = self.contrastive(outputs["img_proj"][has_both], outputs["omics_proj"][has_both])
            components["contrastive"] = loss_c.item()
            total = total + self.w_contrastive * loss_c
        else:
            components["contrastive"] = 0.0

        if only_image.any() and self.w_omics_recon > 0:
            loss_or = self.omics_recon(outputs["pred_omics"], omics_gt, only_image)
            components["omics_recon"] = loss_or.item()
            total = total + self.w_omics_recon * loss_or
        else:
            components["omics_recon"] = 0.0

        if has_both.any() and self.w_omics_recon_reg > 0:
            loss_reg = self.omics_recon(outputs["pred_omics"], omics_gt, has_both)
            components["omics_recon_reg"] = loss_reg.item()
            total = total + self.w_omics_recon_reg * loss_reg
        else:
            components["omics_recon_reg"] = 0.0

        if only_omics.any() and self.w_img_feat_recon > 0:
            loss_if = self.img_feat_recon(outputs["pred_img_feat"], outputs["img_embed"], only_omics)
            components["img_feat_recon"] = loss_if.item()
            total = total + self.w_img_feat_recon * loss_if
        else:
            components["img_feat_recon"] = 0.0

        components["total"] = total.item()
        return total, components
