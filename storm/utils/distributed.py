from __future__ import annotations

import os

import torch
import torch.distributed as dist


def is_dist_available() -> bool:
    """Return ``True`` when ``torch.distributed`` is available and initialised."""
    return dist.is_available() and dist.is_initialized()


def get_world_size() -> int:
    """Return the number of processes in the distributed group, or 1 if single-GPU."""
    return dist.get_world_size() if is_dist_available() else 1


def get_rank() -> int:
    """Return the global rank of this process, or 0 if single-GPU."""
    return dist.get_rank() if is_dist_available() else 0


def is_primary_rank() -> bool:
    """Return ``True`` for the rank-0 process (or always for single-GPU jobs)."""
    return get_rank() == 0


def barrier():
    """Block until all processes in the group reach this point."""
    if is_dist_available():
        dist.barrier()


def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    """Average a scalar tensor across all ranks.

    Args:
        tensor: Scalar tensor to reduce.

    Returns:
        Tensor with value equal to the mean across all ranks.
    """
    if not is_dist_available():
        return tensor
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    return rt / get_world_size()


def get_local_rank() -> int:
    """Read ``LOCAL_RANK`` from the environment, defaulting to 0.

    This is set automatically by ``torchrun`` and most SLURM launchers.
    """
    return int(os.environ.get("LOCAL_RANK", 0))
