"""
SpatialSpotDataset — loads histology image patches and gene expression vectors
for spatial transcriptomics experiments.

Supported file formats
----------------------
HDF5  (.h5):   datasets "expr" (N×G), "patches" (N×H×W×3), optionally
               "coords" (N×2) and "labels" (N,).
NPZ   (.npz):  arrays with the same keys.
Directory:     expr.npy, patches.npy, optionally coords.npy / labels.npy.

All splits (train / val / test) are carved out of the same file via a
deterministic random permutation seeded by `seed`.
"""

from __future__ import annotations

import os
from typing import Callable, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class SpatialSpotDataset(Dataset):
    """
    Args:
        data_path:    Path to .h5, .npz, or directory.
        img_size:     Side length (pixels) to resize patches to; 0 = no resize.
        transform:    torchvision-compatible callable applied to the CHW float
                      tensor AFTER converting from uint8.
        split:        "train", "val", or "test".
        split_ratio:  Fraction of spots used for training (rest split 50/50
                      between val and test).
        seed:         Random seed for the train/val/test split.
        gene_filter:  Optional array of gene indices to retain (sub-panel).
    """

    def __init__(
        self,
        data_path: str,
        img_size: int = 224,
        transform: Optional[Callable] = None,
        split: str = "train",
        split_ratio: float = 0.85,
        seed: int = 42,
        gene_filter: Optional[np.ndarray] = None,
    ):
        self.img_size = img_size
        self.transform = transform

        self._load(data_path)

        if gene_filter is not None:
            self.expr = self.expr[:, gene_filter]

        self.indices = self._split_indices(split, split_ratio, seed)

    # ------------------------------------------------------------------
    # Loading helpers
    # ------------------------------------------------------------------

    def _load(self, path: str):
        if os.path.isfile(path) and path.endswith(".h5"):
            self._load_h5(path)
        elif os.path.isfile(path) and path.endswith(".npz"):
            self._load_npz(path)
        elif os.path.isdir(path):
            self._load_dir(path)
        else:
            raise ValueError(
                f"Unsupported data path: {path!r}. "
                "Expected .h5, .npz, or a directory."
            )

    def _load_h5(self, path: str):
        import h5py
        with h5py.File(path, "r") as f:
            self.expr = f["expr"][:]
            self.patches = f["patches"][:]
            self.coords = f["coords"][:] if "coords" in f else None
            self.labels = f["labels"][:] if "labels" in f else None

    def _load_npz(self, path: str):
        data = np.load(path, allow_pickle=False)
        self.expr = data["expr"]
        self.patches = data["patches"]
        self.coords = data["coords"] if "coords" in data else None
        self.labels = data["labels"] if "labels" in data else None

    def _load_dir(self, path: str):
        self.expr = np.load(os.path.join(path, "expr.npy"))
        self.patches = np.load(os.path.join(path, "patches.npy"))
        coords_p = os.path.join(path, "coords.npy")
        labels_p = os.path.join(path, "labels.npy")
        self.coords = np.load(coords_p) if os.path.exists(coords_p) else None
        self.labels = np.load(labels_p) if os.path.exists(labels_p) else None

    def _split_indices(self, split: str, ratio: float, seed: int) -> np.ndarray:
        N = len(self.expr)
        rng = np.random.default_rng(seed)
        idx = rng.permutation(N)
        n_train = int(N * ratio)
        n_val = (N - n_train) // 2
        if split == "train":
            return idx[:n_train]
        elif split == "val":
            return idx[n_train : n_train + n_val]
        elif split == "test":
            return idx[n_train + n_val :]
        else:  # "all"
            return idx

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def num_genes(self) -> int:
        return self.expr.shape[1]

    @property
    def num_spots(self) -> int:
        return len(self.indices)

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict:
        real_idx = int(self.indices[idx])

        # Gene expression: (G,) float32
        expr = torch.from_numpy(self.expr[real_idx].astype(np.float32))

        # Image patch: uint8 HWC or CHW → float32 CHW in [0, 1]
        patch = self.patches[real_idx]
        if patch.ndim == 3 and patch.shape[2] == 3:
            patch = patch.transpose(2, 0, 1)               # HWC → CHW
        patch = torch.from_numpy(patch.astype(np.float32)) / 255.0

        if self.img_size > 0 and patch.shape[-1] != self.img_size:
            import torch.nn.functional as F
            patch = F.interpolate(
                patch.unsqueeze(0),
                size=(self.img_size, self.img_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

        if self.transform is not None:
            patch = self.transform(patch)

        item: dict = {"image": patch, "omics": expr, "spot_idx": real_idx}

        if self.coords is not None:
            item["coords"] = torch.from_numpy(
                self.coords[real_idx].astype(np.float32)
            )
        if self.labels is not None:
            item["label"] = int(self.labels[real_idx])

        return item

    # ------------------------------------------------------------------
    # DataLoader factory
    # ------------------------------------------------------------------

    def get_loader(
        self,
        batch_size: int = 64,
        num_workers: int = 4,
        shuffle: bool = True,
        pin_memory: bool = True,
        drop_last: bool = True,
    ) -> DataLoader:
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
        )


# ---------------------------------------------------------------------------
# Synthetic dataset for unit tests / smoke tests (no real data needed)
# ---------------------------------------------------------------------------

class SyntheticSpatialDataset(Dataset):
    """
    Generates random (image, omics) pairs in-memory.
    Useful for debugging and CI smoke tests without real data.
    """

    def __init__(
        self,
        num_spots: int = 1000,
        num_genes: int = 500,
        img_size: int = 64,
        seed: int = 0,
    ):
        rng = np.random.default_rng(seed)
        self._images = rng.random((num_spots, 3, img_size, img_size), dtype=np.float32)
        self._omics = rng.random((num_spots, num_genes), dtype=np.float32)
        self._labels = rng.integers(0, 5, size=num_spots)
        self.num_genes = num_genes

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, idx: int) -> dict:
        return {
            "image": torch.from_numpy(self._images[idx]),
            "omics": torch.from_numpy(self._omics[idx]),
            "label": int(self._labels[idx]),
            "spot_idx": idx,
        }

    def get_loader(self, batch_size: int = 64, **kw) -> DataLoader:
        return DataLoader(self, batch_size=batch_size, shuffle=kw.get("shuffle", True))
