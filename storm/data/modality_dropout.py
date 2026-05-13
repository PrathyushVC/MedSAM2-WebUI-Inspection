"""
Per-spot modality dropout for multimodal training.

Two tiers of dropout are applied every training step:

  Batch-level (applied first, mutually exclusive):
    With probability p_batch_drop_image, ALL spots in the batch lose their image.
    With probability p_batch_drop_omics, ALL spots in the batch lose their omics.
    If both would fire, one is randomly suppressed (batch-level never drops both).

  Spot-level (independent Bernoulli per spot):
    drop_image[i] ~ Bernoulli(p_drop_image)
    drop_omics[i] ~ Bernoulli(p_drop_omics)
    drop_both_extra[i] ~ Bernoulli(p_drop_both)  — forces BOTH flags True

The resulting boolean tensors (drop_image, drop_omics) are passed to the model
and to the loss function. Spots where BOTH flags are True are fully excluded from
the loss via the `has_neither` mask computed in ModalityDropout.get_masks().
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch


@dataclass
class ModalityDropoutConfig:
    """Configures all dropout probabilities.

    Attributes:
        p_drop_image:       Per-spot probability of masking the image modality.
        p_drop_omics:       Per-spot probability of masking the omics modality.
        p_drop_both:        Additional per-spot probability of masking BOTH
                            modalities simultaneously (stacked on top of the
                            independent per-modality draws).
        p_batch_drop_image: Probability of dropping image for the ENTIRE batch.
        p_batch_drop_omics: Probability of dropping omics for the ENTIRE batch.
        seed:               Optional RNG seed for reproducibility.
    """
    p_drop_image: float = 0.15
    p_drop_omics: float = 0.15
    p_drop_both: float = 0.05
    p_batch_drop_image: float = 0.05
    p_batch_drop_omics: float = 0.05
    seed: Optional[int] = None

    def __post_init__(self):
        for attr in ("p_drop_image", "p_drop_omics", "p_drop_both",
                     "p_batch_drop_image", "p_batch_drop_omics"):
            v = getattr(self, attr)
            assert 0.0 <= v <= 1.0, f"{attr} must be in [0, 1], got {v}"


class ModalityDropout:
    """
    Stateful sampler that generates per-spot modality dropout masks each step.

    Usage::

        dropout = ModalityDropout(ModalityDropoutConfig(p_drop_image=0.2))
        drop_img, drop_omics = dropout.sample(batch_size=32, training=True)
        has_image, has_omics, has_both, has_neither = dropout.get_masks(drop_img, drop_omics)
    """

    def __init__(self, config: ModalityDropoutConfig):
        self.config = config
        seed = config.seed if config.seed is not None else None
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def sample(
        self,
        batch_size: int,
        training: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample dropout flags for a batch.

        During evaluation no dropout is applied and both tensors are all-False.

        Returns:
            drop_image: (B,) bool — True ⟹ image is masked for this spot.
            drop_omics: (B,) bool — True ⟹ omics is masked for this spot.
        """
        if not training:
            zeros = torch.zeros(batch_size, dtype=torch.bool)
            return zeros, zeros.clone()

        cfg = self.config

        # ---- Batch-level dropout (mutually exclusive) -----------------
        fire_img = self._rng.random() < cfg.p_batch_drop_image
        fire_omics = self._rng.random() < cfg.p_batch_drop_omics

        if fire_img and fire_omics:
            # Resolve conflict: randomly keep one dropout
            if self._rng.random() < 0.5:
                fire_img = False
            else:
                fire_omics = False

        if fire_img:
            return (
                torch.ones(batch_size, dtype=torch.bool),
                torch.zeros(batch_size, dtype=torch.bool),
            )
        if fire_omics:
            return (
                torch.zeros(batch_size, dtype=torch.bool),
                torch.ones(batch_size, dtype=torch.bool),
            )

        # ---- Spot-level dropout (independent per spot) ----------------
        drop_img = torch.from_numpy(
            self._rng.random(batch_size) < cfg.p_drop_image
        )
        drop_omics = torch.from_numpy(
            self._rng.random(batch_size) < cfg.p_drop_omics
        )
        # Extra both-drop: force both flags True for some spots
        drop_both_extra = torch.from_numpy(
            self._rng.random(batch_size) < cfg.p_drop_both
        )
        drop_img = drop_img | drop_both_extra
        drop_omics = drop_omics | drop_both_extra

        return drop_img, drop_omics

    @staticmethod
    def get_masks(
        drop_image: torch.Tensor,
        drop_omics: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Derive availability masks from dropout flags.

        Returns:
            has_image:   (B,) bool — image available as input for this spot.
            has_omics:   (B,) bool — omics available as input for this spot.
            has_both:    (B,) bool — both modalities available.
            has_neither: (B,) bool — no modality available (excluded from loss).
        """
        has_image = ~drop_image
        has_omics = ~drop_omics
        has_both = has_image & has_omics
        has_neither = drop_image & drop_omics
        return has_image, has_omics, has_both, has_neither

    def describe(self, drop_image: torch.Tensor, drop_omics: torch.Tensor) -> dict:
        """Return a summary dict of spot counts per scenario (for logging)."""
        has_image, has_omics, has_both, has_neither = self.get_masks(drop_image, drop_omics)
        B = len(drop_image)
        return {
            "n_both":    has_both.sum().item(),
            "n_img_only": (has_image & ~has_omics).sum().item(),
            "n_omics_only": (~has_image & has_omics).sum().item(),
            "n_neither":  has_neither.sum().item(),
            "batch_size": B,
        }
