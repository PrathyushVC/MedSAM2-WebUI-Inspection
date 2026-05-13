from __future__ import annotations

from typing import Dict, List

import torch


def spatial_collate_fn(batch: List[dict]) -> Dict[str, torch.Tensor]:
    """Collate a list of spot sample dicts into a batched dict.

    Handles optional fields (``coords``, ``label``) gracefully; they are
    included in the output only if present in every sample.

    Args:
        batch: List of dicts, each containing at minimum ``"image"`` (float
            tensor, ``C×H×W``), ``"omics"`` (float tensor, ``G``), and
            ``"spot_idx"`` (int).

    Returns:
        Dict with stacked tensors keyed by field name.
    """
    keys = batch[0].keys()
    out: dict = {
        "image": torch.stack([s["image"] for s in batch]),
        "omics": torch.stack([s["omics"] for s in batch]),
        "spot_idx": torch.tensor([s["spot_idx"] for s in batch]),
    }
    if "coords" in keys:
        out["coords"] = torch.stack([s["coords"] for s in batch])
    if "label" in keys:
        out["label"] = torch.tensor([s["label"] for s in batch], dtype=torch.long)
    return out
