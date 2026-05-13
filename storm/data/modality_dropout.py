from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch


@dataclass
class ModalityDropoutConfig:
    """Probability configuration for per-spot and batch-level modality dropout.

    Dropout is applied in two tiers. First, batch-level: with probability
    ``p_batch_drop_image`` the entire batch loses its image modality (and
    similarly for omics). Both cannot fire simultaneously at batch level; if
    they would, one is randomly suppressed. If neither batch-level flag fires,
    spot-level Bernoulli draws are made independently per spot.

    Attributes:
        p_drop_image: Per-spot probability of masking the image modality.
        p_drop_omics: Per-spot probability of masking the omics modality.
        p_drop_both: Additional per-spot probability of forcing both modalities
            masked simultaneously. Stacked on top of the independent draws.
        p_batch_drop_image: Probability of dropping image for the entire batch.
        p_batch_drop_omics: Probability of dropping omics for the entire batch.
        seed: Optional RNG seed for reproducibility during debugging.
    """

    p_drop_image: float = 0.15
    p_drop_omics: float = 0.15
    p_drop_both: float = 0.05
    p_batch_drop_image: float = 0.05
    p_batch_drop_omics: float = 0.05
    seed: Optional[int] = None

    def __post_init__(self):
        for field in ("p_drop_image", "p_drop_omics", "p_drop_both",
                      "p_batch_drop_image", "p_batch_drop_omics"):
            v = getattr(self, field)
            if not 0.0 <= v <= 1.0:
                raise ValueError(f"{field} must be in [0, 1], got {v}")


class ModalityDropout:
    """Stateful per-step sampler for modality dropout masks.

    A fresh pair of boolean masks is drawn on each call to ``sample``, so every
    pass over the same batch of spots sees a different combination of dropped
    modalities. During evaluation ``sample`` returns all-False masks, leaving
    both modalities intact.

    Args:
        config: Dropout probability configuration.

    Example:
        >>> dropout = ModalityDropout(ModalityDropoutConfig(p_drop_image=0.2))
        >>> drop_img, drop_omics = dropout.sample(batch_size=32, training=True)
        >>> has_image, has_omics, has_both, has_neither = dropout.get_masks(drop_img, drop_omics)
    """

    def __init__(self, config: ModalityDropoutConfig):
        self.config = config
        self._rng = np.random.default_rng(config.seed)

    def sample(
        self, batch_size: int, training: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample dropout flags for a single training step.

        Args:
            batch_size: Number of spots in the batch.
            training: When ``False``, returns all-False masks (no dropout).

        Returns:
            A tuple ``(drop_image, drop_omics)`` of boolean tensors each of
            shape ``(B,)``.  ``True`` at position ``i`` means that modality is
            masked for spot ``i``.
        """
        if not training:
            zeros = torch.zeros(batch_size, dtype=torch.bool)
            return zeros, zeros.clone()

        cfg = self.config
        fire_img = self._rng.random() < cfg.p_batch_drop_image
        fire_omics = self._rng.random() < cfg.p_batch_drop_omics

        if fire_img and fire_omics:
            if self._rng.random() < 0.5:
                fire_img = False
            else:
                fire_omics = False

        if fire_img:
            return torch.ones(batch_size, dtype=torch.bool), torch.zeros(batch_size, dtype=torch.bool)
        if fire_omics:
            return torch.zeros(batch_size, dtype=torch.bool), torch.ones(batch_size, dtype=torch.bool)

        drop_img = torch.from_numpy(self._rng.random(batch_size) < cfg.p_drop_image)
        drop_omics = torch.from_numpy(self._rng.random(batch_size) < cfg.p_drop_omics)
        drop_both_extra = torch.from_numpy(self._rng.random(batch_size) < cfg.p_drop_both)
        return drop_img | drop_both_extra, drop_omics | drop_both_extra

    @staticmethod
    def get_masks(
        drop_image: torch.Tensor,
        drop_omics: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Derive availability masks from a pair of dropout flag tensors.

        Args:
            drop_image: Boolean tensor of shape ``(B,)``; ``True`` = image dropped.
            drop_omics: Boolean tensor of shape ``(B,)``; ``True`` = omics dropped.

        Returns:
            A tuple ``(has_image, has_omics, has_both, has_neither)`` of boolean
            tensors each of shape ``(B,)``.
        """
        has_image = ~drop_image
        has_omics = ~drop_omics
        return has_image, has_omics, has_image & has_omics, drop_image & drop_omics

    def describe(self, drop_image: torch.Tensor, drop_omics: torch.Tensor) -> dict:
        """Summarise spot counts per dropout scenario for logging.

        Args:
            drop_image: Boolean tensor of shape ``(B,)``.
            drop_omics: Boolean tensor of shape ``(B,)``.

        Returns:
            Dictionary with keys ``n_both``, ``n_img_only``, ``n_omics_only``,
            ``n_neither``, and ``batch_size``.
        """
        has_image, has_omics, has_both, has_neither = self.get_masks(drop_image, drop_omics)
        return {
            "n_both": has_both.sum().item(),
            "n_img_only": (has_image & ~has_omics).sum().item(),
            "n_omics_only": (~has_image & has_omics).sum().item(),
            "n_neither": has_neither.sum().item(),
            "batch_size": len(drop_image),
        }
