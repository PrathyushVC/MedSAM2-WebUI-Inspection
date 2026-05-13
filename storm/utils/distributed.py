"""
Minimal distributed utilities for STORM.

These functions are safe to call in both single-GPU and DDP settings; they
are no-ops or identity operations when torch.distributed is not initialised.
"""

from __future__ import annotations

import os
from typing import Optional

import torch
import torch.distributed as dist


def is_dist_available() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_world_size() -> int:
    return dist.get_world_size() if is_dist_available() else 1


def get_rank() -> int:
    return dist.get_rank() if is_dist_available() else 0


def is_primary_rank() -> bool:
    return get_rank() == 0


def barrier():
    if is_dist_available():
        dist.barrier()


def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    """Average a scalar tensor across all ranks."""
    if not is_dist_available():
        return tensor
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= get_world_size()
    return rt


def setup_distributed(local_rank: int, backend: str = "nccl") -> int:
    """
    Initialise torch.distributed from torchrun / SLURM environment variables.
    Returns the global rank.
    """
    dist.init_process_group(backend=backend)
    torch.cuda.set_device(local_rank)
    return get_rank()


def get_local_rank() -> int:
    """Read LOCAL_RANK from the environment (set by torchrun)."""
    return int(os.environ.get("LOCAL_RANK", 0))
