from __future__ import annotations

import os
from typing import Callable, Optional

import numpy as np
import torch
from monai.transforms import Compose, NormalizeIntensity, RandFlip, RandRotate90, Resize
from torch.utils.data import DataLoader, Dataset


def default_train_transform(img_size: int) -> Compose:
    """Build a standard training augmentation pipeline using MONAI transforms.

    Applies spatial flips, 90-degree rotations, and per-channel intensity
    normalisation. All transforms operate on channel-first float tensors.

    Args:
        img_size: Target square spatial resolution in pixels.

    Returns:
        A ``monai.transforms.Compose`` object ready to be called on a
        ``(C, H, W)`` float tensor.
    """
    return Compose([
        Resize(spatial_size=(img_size, img_size)),
        RandFlip(spatial_axis=0, prob=0.5),
        RandFlip(spatial_axis=1, prob=0.5),
        RandRotate90(prob=0.5, max_k=3, spatial_axes=(0, 1)),
        NormalizeIntensity(channel_wise=True),
    ])


def default_val_transform(img_size: int) -> Compose:
    """Build a deterministic validation preprocessing pipeline using MONAI transforms.

    Args:
        img_size: Target square spatial resolution in pixels.

    Returns:
        A ``monai.transforms.Compose`` object.
    """
    return Compose([
        Resize(spatial_size=(img_size, img_size)),
        NormalizeIntensity(channel_wise=True),
    ])


class SpatialSpotDataset(Dataset):
    """Dataset of histology image patches and gene expression profiles for spatial
    transcriptomics spots.

    Supported data formats:

    - **HDF5** (``.h5``): datasets ``expr`` (N×G), ``patches`` (N×H×W×3),
      optionally ``coords`` (N×2) and ``labels`` (N,).
    - **NPZ** (``.npz``): same array keys.
    - **Directory**: ``expr.npy``, ``patches.npy``, optionally ``coords.npy``
      and ``labels.npy``.

    The train / val / test split is derived via a seeded random permutation, so
    all three splits are carved from the same file without needing separate files.

    Args:
        data_path: Path to a ``.h5`` file, a ``.npz`` file, or a directory.
        img_size: Target square spatial resolution; patches are resized if
            their stored size differs.  Pass ``0`` to skip resizing.
        transform: Callable applied to the ``(C, H, W)`` float patch tensor.
            Defaults to ``default_train_transform`` or ``default_val_transform``
            depending on ``split``, unless explicitly overridden.
        split: One of ``"train"``, ``"val"``, ``"test"``, or ``"all"``.
        split_ratio: Fraction of spots used for training; the remainder is split
            evenly between val and test.
        seed: RNG seed for the split permutation.
        gene_filter: Optional integer array of gene indices to retain, allowing
            a subset of the stored panel to be used.
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
        self._load(data_path)

        if gene_filter is not None:
            self.expr = self.expr[:, gene_filter]

        self.indices = self._split_indices(split, split_ratio, seed)

        if transform is not None:
            self.transform = transform
        elif split == "train":
            self.transform = default_train_transform(img_size) if img_size > 0 else None
        else:
            self.transform = default_val_transform(img_size) if img_size > 0 else None

    def _load(self, path: str):
        if os.path.isfile(path) and path.endswith(".h5"):
            self._load_h5(path)
        elif os.path.isfile(path) and path.endswith(".npz"):
            self._load_npz(path)
        elif os.path.isdir(path):
            self._load_dir(path)
        else:
            raise ValueError(f"Unsupported data source: {path!r}. Expected .h5, .npz, or directory.")

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
        coords_p, labels_p = os.path.join(path, "coords.npy"), os.path.join(path, "labels.npy")
        self.coords = np.load(coords_p) if os.path.exists(coords_p) else None
        self.labels = np.load(labels_p) if os.path.exists(labels_p) else None

    def _split_indices(self, split: str, ratio: float, seed: int) -> np.ndarray:
        N = len(self.expr)
        rng = np.random.default_rng(seed)
        idx = rng.permutation(N)
        n_train = int(N * ratio)
        n_val = (N - n_train) // 2
        mapping = {
            "train": idx[:n_train],
            "val": idx[n_train: n_train + n_val],
            "test": idx[n_train + n_val:],
            "all": idx,
        }
        if split not in mapping:
            raise ValueError(f"split must be one of {list(mapping)}, got {split!r}")
        return mapping[split]

    @property
    def num_genes(self) -> int:
        """Number of genes in the (possibly filtered) expression panel."""
        return self.expr.shape[1]

    @property
    def num_spots(self) -> int:
        """Number of spots in the active split."""
        return len(self.indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict:
        real_idx = int(self.indices[idx])
        expr = torch.from_numpy(self.expr[real_idx].astype(np.float32))

        patch = self.patches[real_idx]
        if patch.ndim == 3 and patch.shape[2] == 3:
            patch = patch.transpose(2, 0, 1)
        patch = torch.from_numpy(patch.astype(np.float32)) / 255.0

        if self.transform is not None:
            patch = self.transform(patch)

        item = {"image": patch, "omics": expr, "spot_idx": real_idx}
        if self.coords is not None:
            item["coords"] = torch.from_numpy(self.coords[real_idx].astype(np.float32))
        if self.labels is not None:
            item["label"] = int(self.labels[real_idx])
        return item

    def get_loader(
        self,
        batch_size: int = 64,
        num_workers: int = 4,
        shuffle: bool = True,
        pin_memory: bool = True,
        drop_last: bool = True,
    ) -> DataLoader:
        """Construct a DataLoader for this dataset split.

        Args:
            batch_size: Number of spots per batch.
            num_workers: Parallel data-loading workers.
            shuffle: Whether to shuffle between epochs.
            pin_memory: Pin tensors to CUDA pinned memory for faster transfer.
            drop_last: Drop the final incomplete batch.

        Returns:
            A configured ``torch.utils.data.DataLoader``.
        """
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
        )


class SyntheticSpatialDataset(Dataset):
    """In-memory random dataset for smoke tests and unit tests.

    Generates Gaussian random image patches and expression profiles so that
    the full training pipeline can be validated without any real data files.

    Args:
        num_spots: Number of synthetic spots.
        num_genes: Gene panel width.
        img_size: Square spatial resolution of generated patches.
        seed: NumPy RNG seed.
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

    def get_loader(self, batch_size: int = 64, shuffle: bool = True) -> DataLoader:
        """Construct a DataLoader for the synthetic dataset.

        Args:
            batch_size: Spots per batch.
            shuffle: Whether to shuffle each epoch.

        Returns:
            A configured ``torch.utils.data.DataLoader``.
        """
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle)
