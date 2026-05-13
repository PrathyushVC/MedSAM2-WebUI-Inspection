"""
STORM training entry point.

Single-GPU usage:
    python train_storm.py --config storm/configs/base.yaml

Multi-GPU (torchrun):
    torchrun --nproc_per_node=4 train_storm.py --config storm/configs/base.yaml

Override any config key via --set:
    python train_storm.py --config storm/configs/base.yaml \\
        --set training.max_epochs=50 \\
        --set data.data_path=my_data.h5
"""

from __future__ import annotations

import argparse
import logging
import os
import random

import numpy as np
import torch
from typing import List

import yaml

from storm.data.dataset import SpatialSpotDataset, SyntheticSpatialDataset
from storm.data.modality_dropout import ModalityDropoutConfig
from storm.model.storm_model import STORMModel
from storm.training.trainer import STORMTrainer, setup_distributed
from storm.utils.distributed import get_local_rank, is_dist_available, is_primary_rank
from storm.utils.logging import setup_logging


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def apply_overrides(cfg: dict, overrides: List[str]) -> dict:
    """Apply --set key.sub_key=value overrides to a nested dict."""
    for override in overrides:
        key, _, value = override.partition("=")
        parts = key.strip().split(".")
        node = cfg
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        # Try to parse the value as YAML (handles ints, floats, booleans, lists)
        node[parts[-1]] = yaml.safe_load(value)
    return cfg


def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------

def build_datasets(cfg: dict):
    data_cfg = cfg["data"]
    data_path = data_cfg["data_path"]
    gene_filter = data_cfg.get("gene_filter", None)
    if gene_filter is not None:
        gene_filter = np.array(gene_filter, dtype=np.int64)

    common_kw = dict(
        img_size=data_cfg.get("img_size", 224),
        split_ratio=data_cfg.get("split_ratio", 0.85),
        seed=data_cfg.get("seed", 42),
        gene_filter=gene_filter,
    )

    if data_path == "__synthetic__":
        num_genes = cfg["model"]["num_genes"]
        train_ds = SyntheticSpatialDataset(num_spots=2000, num_genes=num_genes, img_size=64)
        val_ds = SyntheticSpatialDataset(num_spots=400, num_genes=num_genes, img_size=64)
        return train_ds, val_ds

    train_ds = SpatialSpotDataset(data_path, split="train", **common_kw)
    val_ds = SpatialSpotDataset(data_path, split="val", **common_kw)
    return train_ds, val_ds


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train STORM multimodal model")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument(
        "--set", nargs="*", default=[],
        metavar="KEY=VALUE",
        help="Override config values (e.g. --set training.max_epochs=100)",
    )
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Run a quick sanity check with synthetic data (2 epochs, no real data needed)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.set:
        cfg = apply_overrides(cfg, args.set)

    if args.smoke_test:
        cfg["data"]["data_path"] = "__synthetic__"
        cfg["training"]["max_epochs"] = 2
        cfg["model"]["num_genes"] = cfg["model"].get("num_genes", 500)

    # Distributed setup
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_distributed = "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1
    if is_distributed:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        setup_distributed(local_rank, backend=backend)

    train_cfg = cfg["training"]
    set_seeds(train_cfg.get("seed", 42))

    # Logging (primary rank only writes to file)
    setup_logging(
        log_dir=train_cfg.get("log_dir") if is_primary_rank() else None,
        rank=local_rank,
    )
    logger = logging.getLogger(__name__)

    if is_primary_rank():
        logger.info("Config:\n" + yaml.dump(cfg, default_flow_style=False))

    # Dataset & loaders
    train_ds, val_ds = build_datasets(cfg)
    data_cfg = cfg["data"]

    if is_distributed:
        from torch.utils.data.distributed import DistributedSampler
        train_sampler = DistributedSampler(train_ds, shuffle=True)
        val_sampler = DistributedSampler(val_ds, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None

    loader_kw = dict(
        batch_size=data_cfg.get("batch_size", 64),
        num_workers=data_cfg.get("num_workers", 4),
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    train_loader = train_ds.get_loader(
        **loader_kw,
        shuffle=(train_sampler is None),
    )
    val_loader = val_ds.get_loader(
        **{**loader_kw, "shuffle": False, "drop_last": False},
    )

    # Infer num_genes from dataset if not explicitly configured
    num_genes = cfg["model"].get("num_genes") or getattr(train_ds, "num_genes", None)
    assert num_genes is not None, "Could not determine num_genes; set model.num_genes in config."
    cfg["model"]["num_genes"] = num_genes

    # Model
    model = STORMModel(**cfg["model"])
    if is_primary_rank():
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model parameters: {n_params / 1e6:.1f} M")

    # Dropout config
    dropout_cfg = ModalityDropoutConfig(**cfg.get("dropout", {}))

    # Device
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

    # Trainer
    trainer = STORMTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_cfg=cfg.get("loss", {}),
        optim_cfg=cfg.get("optimizer", {}),
        scheduler_cfg={
            "warmup_epochs": train_cfg.get("warmup_epochs", 10),
            "total_epochs": train_cfg.get("max_epochs", 100),
            "min_lr_ratio": train_cfg.get("min_lr_ratio", 0.01),
        },
        dropout_cfg=dropout_cfg,
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
