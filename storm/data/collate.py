"""
Custom collate function for spatial spot batches.

Handles optional fields (coords, label) gracefully and stacks everything
into tensors ready for the model forward pass.
"""

from __future__ import annotations

from typing import Dict, List

import torch
from torch.utils.data.dataloader import default_collate


def spatial_collate_fn(batch: List[dict]) -> Dict[str, torch.Tensor]:
    """
    Collate a list of spot dicts into a batch dict.

    Required keys in each sample: "image", "omics", "spot_idx".
    Optional keys forwarded if present: "coords", "label".
    """
    keys = batch[0].keys()
    out: dict = {}

    # Always-present tensors
    out["image"] = torch.stack([s["image"] for s in batch])      # (B, 3, H, W)
    out["omics"] = torch.stack([s["omics"] for s in batch])      # (B, G)
    out["spot_idx"] = torch.tensor([s["spot_idx"] for s in batch])

    # Optional tensors
    if "coords" in keys:
        out["coords"] = torch.stack([s["coords"] for s in batch])  # (B, 2)
    if "label" in keys:
        out["label"] = torch.tensor([s["label"] for s in batch], dtype=torch.long)

    return out
