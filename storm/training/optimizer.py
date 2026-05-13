"""
Optimizer and LR-scheduler construction for STORM.

Weight decay is applied only to weight matrices; biases, LayerNorm parameters,
and special learnable parameters (CLS tokens, positional embeddings, mask tokens)
are excluded from decay.
"""

from __future__ import annotations

import re
from typing import List, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR


_NO_DECAY_PATTERN = (
    r"bias$"
    r"|norm\.weight$"
    r"|norm\.bias$"
    r"|layer_norm\."
    r"|cls_token"
    r"|pos_embed"
    r"|mask_token"
    r"|log_temp"
)


def build_optimizer(
    model: nn.Module,
    lr: float = 1e-4,
    weight_decay: float = 0.05,
    betas: tuple = (0.9, 0.999),
    eps: float = 1e-8,
    no_decay_pattern: str = _NO_DECAY_PATTERN,
    frozen_modules: Optional[List[str]] = None,
) -> AdamW:
    """
    AdamW with selective weight decay.

    Args:
        model:            The model whose parameters to optimise.
        lr:               Base learning rate.
        weight_decay:     L2 penalty for parameters not matched by no_decay_pattern.
        betas:            AdamW beta coefficients.
        eps:              AdamW epsilon.
        no_decay_pattern: Regex matched against parameter names; matching params
                          get weight_decay=0.
        frozen_modules:   List of module name prefixes to keep frozen (no grad).
    """
    if frozen_modules:
        for name, param in model.named_parameters():
            if any(name.startswith(prefix) for prefix in frozen_modules):
                param.requires_grad_(False)

    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if re.search(no_decay_pattern, name):
            no_decay.append(param)
        else:
            decay.append(param)

    param_groups = [
        {"params": decay,    "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return AdamW(param_groups, lr=lr, betas=betas, eps=eps)


def build_scheduler(
    optimizer: AdamW,
    warmup_epochs: int,
    total_epochs: int,
    min_lr_ratio: float = 0.01,
) -> torch.optim.lr_scheduler.LRScheduler:
    """
    Linear warmup followed by cosine annealing to min_lr_ratio * base_lr.

    Args:
        optimizer:      The AdamW optimiser.
        warmup_epochs:  Number of epochs for linear warmup.
        total_epochs:   Total training epochs.
        min_lr_ratio:   Fraction of base LR to anneal to.
    """
    warmup = LinearLR(
        optimizer,
        start_factor=min_lr_ratio,
        end_factor=1.0,
        total_iters=max(warmup_epochs, 1),
    )
    cosine = CosineAnnealingLR(
        optimizer,
        T_max=max(total_epochs - warmup_epochs, 1),
        eta_min=optimizer.param_groups[0]["lr"] * min_lr_ratio,
    )
    return SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_epochs],
    )
