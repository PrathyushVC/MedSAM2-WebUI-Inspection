from __future__ import annotations

import re
from typing import List, Optional

import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

_NO_DECAY_RE = r"bias$|norm\.weight$|norm\.bias$|layer_norm\.|cls_token|pos_embed|mask_token|log_temp"


def build_optimizer(
    model: nn.Module,
    lr: float = 1e-4,
    weight_decay: float = 0.05,
    betas: tuple = (0.9, 0.999),
    eps: float = 1e-8,
    no_decay_pattern: str = _NO_DECAY_RE,
    frozen_modules: Optional[List[str]] = None,
) -> AdamW:
    """Construct an AdamW optimiser with selective weight decay.

    Biases, layer-norm parameters, positional embeddings, and special learnable
    parameters (CLS tokens, mask tokens, temperature) are excluded from weight
    decay. All other parameters receive ``weight_decay``.

    Args:
        model: The model to optimise.
        lr: Base learning rate.
        weight_decay: L2 coefficient for parameters not matched by
            ``no_decay_pattern``.
        betas: AdamW beta coefficients ``(beta1, beta2)``.
        eps: AdamW numerical stability epsilon.
        no_decay_pattern: Regular expression matched against parameter names;
            matching parameters receive ``weight_decay=0``.
        frozen_modules: Optional list of module name prefixes to freeze before
            collecting parameter groups.

    Returns:
        Configured ``AdamW`` optimiser.
    """
    if frozen_modules:
        for name, param in model.named_parameters():
            if any(name.startswith(p) for p in frozen_modules):
                param.requires_grad_(False)

    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (no_decay if re.search(no_decay_pattern, name) else decay).append(param)

    return AdamW(
        [{"params": decay, "weight_decay": weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=lr, betas=betas, eps=eps,
    )


def build_scheduler(
    optimizer: AdamW,
    warmup_epochs: int,
    total_epochs: int,
    min_lr_ratio: float = 0.01,
) -> SequentialLR:
    """Construct a linear-warmup then cosine-annealing LR schedule.

    Args:
        optimizer: The AdamW optimiser whose LR groups are scheduled.
        warmup_epochs: Number of epochs for the linear warmup phase.
        total_epochs: Total training epochs (including warmup).
        min_lr_ratio: Minimum LR as a fraction of the peak LR; the cosine
            schedule anneals to ``lr * min_lr_ratio``.

    Returns:
        A ``SequentialLR`` scheduler that transitions from linear warmup to
        cosine annealing at epoch ``warmup_epochs``.
    """
    base_lr = optimizer.param_groups[0]["lr"]
    warmup = LinearLR(optimizer, start_factor=min_lr_ratio, end_factor=1.0, total_iters=max(warmup_epochs, 1))
    cosine = CosineAnnealingLR(optimizer, T_max=max(total_epochs - warmup_epochs, 1), eta_min=base_lr * min_lr_ratio)
    return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])
