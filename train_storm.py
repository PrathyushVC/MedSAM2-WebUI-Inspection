"""Entry point for STORM multimodal training.

Single-GPU usage::

    python train_storm.py --config storm/configs/base.yaml

Multi-GPU with torchrun::

    torchrun --nproc_per_node=4 train_storm.py --config storm/configs/base.yaml

Override individual config values with ``--set``::

    python train_storm.py --config storm/configs/base.yaml \\
        --set training.max_epochs=50 --set data.data_path=my_data.h5
"""

from __future__ import annotations

import argparse
import logging
import os
import random
from typing import List, Tuple

import numpy as np
import torch
import yaml

from storm.data.dataset import SpatialSpotDataset, SyntheticSpatialDataset
from storm.data.modality_dropout import ModalityDropoutConfig
from storm.model.storm_model import STORMModel
from storm.training.trainer import STORMTrainer, setup_distributed
from storm.utils.distributed import get_local_rank, is_primary_rank
from storm.utils.logging import setup_logging


def load_config(path: str) -> dict:
    """Load a YAML config file into a nested dict.

    Args:
        path: Path to the YAML file.

    Returns:
        Parsed config dict.
    """
    with open(path) as f:
        return yaml.safe_load(f)


def apply_overrides(cfg: dict, overrides: List[str]) -> dict:
    """Apply a list of ``KEY=VALUE`` strings to a nested config dict.

    Keys use dot notation to address nested fields (e.g.
    ``training.max_epochs=100``). Values are parsed as YAML scalars so
    integers, floats, booleans, and lists are handled correctly.

    Args:
        cfg: Mutable config dict to update in-place.
        overrides: List of ``"section.key=value"`` strings.

    Returns:
        The updated ``cfg`` dict (same object).
    """
    for override in overrides:
        key, _, value = override.partition("=")
        parts = key.strip().split(".")
        node = cfg
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = yaml.safe_load(value)
    return cfg


def set_seeds(seed: int):
    """Set random seeds for Python, NumPy, and PyTorch.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_datasets(cfg: dict) -> Tuple:
    """Construct train and validation datasets from the config.

    When ``data_path`` is ``"__synthetic__"``, in-memory random data is
    generated using ``SyntheticSpatialDataset`` (no real files needed).

    Args:
        cfg: Full training config dict.

    Returns:
        Tuple of ``(train_dataset, val_dataset)``.
    """
    data_cfg = cfg["data"]
    data_path = data_cfg["data_path"]
    gene_filter = data_cfg.get("gene_filter")
    if gene_filter is not None:
        gene_filter = np.array(gene_filter, dtype=np.int64)

    if data_path == "__synthetic__":
        num_genes = cfg["model"]["num_genes"]
        return (
            SyntheticSpatialDataset(num_spots=2000, num_genes=num_genes, img_size=64),
            SyntheticSpatialDataset(num_spots=400, num_genes=num_genes, img_size=64),
        )

    shared = dict(
        img_size=data_cfg.get("img_size", 224),
        split_ratio=data_cfg.get("split_ratio", 0.85),
        seed=data_cfg.get("seed", 42),
        gene_filter=gene_filter,
    )
    return (
        SpatialSpotDataset(data_path, split="train", **shared),
        SpatialSpotDataset(data_path, split="val", **shared),
    )


def main():
    parser = argparse.ArgumentParser(description="Train the STORM multimodal model")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument(
        "--set", nargs="*", default=[], metavar="KEY=VALUE",
        help="Override config values, e.g. --set training.max_epochs=100",
    )
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Run 2 epochs on synthetic data to verify the pipeline",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.set:
        cfg = apply_overrides(cfg, args.set)
    if args.smoke_test:
        cfg["data"]["data_path"] = "__synthetic__"
        cfg["training"]["max_epochs"] = 2
        cfg["model"].setdefault("num_genes", 500)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_distributed = "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1
    if is_distributed:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        setup_distributed(local_rank, backend=backend)

    train_cfg = cfg["training"]
    set_seeds(train_cfg.get("seed", 42))

    setup_logging(
        log_dir=train_cfg.get("log_dir") if is_primary_rank() else None,
        rank=local_rank,
    )
    logger = logging.getLogger(__name__)

    if is_primary_rank():
        logger.info("Config:\n%s", yaml.dump(cfg, default_flow_style=False))

    train_ds, val_ds = build_datasets(cfg)
    data_cfg = cfg["data"]

    if is_distributed:
        from torch.utils.data.distributed import DistributedSampler
        train_sampler = DistributedSampler(train_ds, shuffle=True)
        val_sampler = DistributedSampler(val_ds, shuffle=False)
    else:
        train_sampler = val_sampler = None

    loader_kw = dict(
        batch_size=data_cfg.get("batch_size", 64),
        num_workers=data_cfg.get("num_workers", 4),
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    train_loader = train_ds.get_loader(**loader_kw, shuffle=(train_sampler is None))
    val_loader = val_ds.get_loader(**{**loader_kw, "shuffle": False, "drop_last": False})

    num_genes = cfg["model"].get("num_genes") or getattr(train_ds, "num_genes", None)
    if num_genes is None:
        raise RuntimeError("Cannot determine num_genes; set model.num_genes in config.")
    cfg["model"]["num_genes"] = num_genes

    model = STORMModel(**cfg["model"])
    if is_primary_rank():
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info("Trainable parameters: %.1f M", n_params / 1e6)

    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

    trainer = STORMTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_cfg=cfg.get("loss"),
        optim_cfg=cfg.get("optimizer"),
        scheduler_cfg=dict(
            warmup_epochs=train_cfg.get("warmup_epochs", 10),
            total_epochs=train_cfg.get("max_epochs", 100),
            min_lr_ratio=train_cfg.get("min_lr_ratio", 0.01),
        ),
        dropout_cfg=ModalityDropoutConfig(**cfg.get("dropout", {})),
        max_epochs=train_cfg.get("max_epochs", 100),
        warmup_epochs=train_cfg.get("warmup_epochs", 10),
        device=device,
        use_amp=train_cfg.get("use_amp", True),
        amp_dtype=train_cfg.get("amp_dtype", "bfloat16"),
        checkpoint_dir=train_cfg.get("checkpoint_dir", "checkpoints/storm"),
        save_every=train_cfg.get("save_every", 10),
        log_every=train_cfg.get("log_every", 50),
        clip_grad_norm=train_cfg.get("clip_grad_norm", 1.0),
        local_rank=local_rank,
        is_distributed=is_distributed,
    )
    trainer.run()


if __name__ == "__main__":
    main()
