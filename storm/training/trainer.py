from __future__ import annotations

import contextlib
import logging
import os
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from storm.data.modality_dropout import ModalityDropout, ModalityDropoutConfig
from storm.model.storm_model import STORMModel
from storm.training.loss import STORMLoss
from storm.training.optimizer import build_optimizer, build_scheduler
from storm.utils.logging import MetricLogger

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def _maybe_autocast(enabled: bool, dtype: torch.dtype):
    if enabled:
        with torch.cuda.amp.autocast(dtype=dtype):
            yield
    else:
        yield


class STORMTrainer:
    """End-to-end training harness for the STORM model.

    Modality dropout masks are sampled fresh on every step so the same batch of
    spots is seen under different missing-data scenarios across epochs. Dropout
    scenario counts are logged at debug level to verify sampling rates.

    Distributed training (DDP) is supported; call ``setup_distributed`` before
    constructing this class when running under ``torchrun``.

    Args:
        model: Initialised ``STORMModel`` (moved to ``device`` internally).
        train_loader: DataLoader whose batches contain ``"image"`` and ``"omics"`` keys.
        val_loader: Optional validation DataLoader; validation is skipped when absent.
        loss_cfg: Keyword arguments forwarded to ``STORMLoss``.
        optim_cfg: Keyword arguments forwarded to ``build_optimizer``.
        scheduler_cfg: Keyword arguments forwarded to ``build_scheduler``.
        dropout_cfg: Controls per-spot and batch-level masking rates.
        max_epochs: Total training epochs.
        warmup_epochs: Epochs of linear LR warmup before cosine annealing.
        device: Torch device string (e.g. ``"cuda:0"`` or ``"cpu"``).
        use_amp: Enable automatic mixed precision.
        amp_dtype: AMP dtype; ``"bfloat16"`` (recommended on Ampere) or ``"float16"``.
        checkpoint_dir: Directory for saving and resuming checkpoints.
        save_every: Save a numbered checkpoint every N epochs; ``0`` disables this.
        log_every: Log step-level metrics every N optimiser steps.
        clip_grad_norm: Max gradient norm; ``None`` disables clipping.
        local_rank: GPU rank for DDP; ``0`` for single-GPU jobs.
        is_distributed: Set ``True`` when running DDP.
    """

    def __init__(
        self,
        model: STORMModel,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        loss_cfg: Optional[Dict[str, Any]] = None,
        optim_cfg: Optional[Dict[str, Any]] = None,
        scheduler_cfg: Optional[Dict[str, Any]] = None,
        dropout_cfg: Optional[ModalityDropoutConfig] = None,
        max_epochs: int = 100,
        warmup_epochs: int = 10,
        device: str = "cuda",
        use_amp: bool = True,
        amp_dtype: str = "bfloat16",
        checkpoint_dir: str = "checkpoints/storm",
        save_every: int = 10,
        log_every: int = 50,
        clip_grad_norm: Optional[float] = 1.0,
        local_rank: int = 0,
        is_distributed: bool = False,
    ):
        self.device = torch.device(device)
        self.local_rank = local_rank
        self.is_distributed = is_distributed
        self.is_primary = (local_rank == 0)

        self.model = model.to(self.device)
        if is_distributed:
            self.model = DDP(self.model, device_ids=[local_rank], find_unused_parameters=False)

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.max_epochs = max_epochs
        self.save_every = save_every
        self.log_every = log_every
        self.clip_grad_norm = clip_grad_norm

        self.criterion = STORMLoss(**(loss_cfg or {})).to(self.device)
        self.optimizer = build_optimizer(self._raw_model, **(optim_cfg or {}))

        sched_kw = dict(warmup_epochs=warmup_epochs, total_epochs=max_epochs)
        if scheduler_cfg:
            sched_kw.update(scheduler_cfg)
        self.scheduler = build_scheduler(self.optimizer, **sched_kw)

        self._amp_dtype = torch.bfloat16 if amp_dtype == "bfloat16" else torch.float16
        self.use_amp = use_amp and torch.cuda.is_available()
        self.scaler = GradScaler(enabled=(self.use_amp and self._amp_dtype == torch.float16))

        self.dropout = ModalityDropout(dropout_cfg or ModalityDropoutConfig())

        self.checkpoint_dir = checkpoint_dir
        if self.is_primary:
            os.makedirs(checkpoint_dir, exist_ok=True)

        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")

    @property
    def _raw_model(self) -> STORMModel:
        return self.model.module if isinstance(self.model, DDP) else self.model

    def _step(
        self, batch: dict, training: bool
    ) -> Tuple[torch.Tensor, Dict[str, float], dict]:
        """Run one forward pass and return the loss, components, and scenario counts.

        Args:
            batch: Batch dict with keys ``"image"`` and ``"omics"``.
            training: Whether to apply modality dropout and compute gradients.

        Returns:
            A tuple ``(loss, components, scenario_counts)``.
        """
        images = batch["image"].to(self.device, non_blocking=True)
        omics = batch["omics"].to(self.device, non_blocking=True)
        B = images.shape[0]

        drop_image, drop_omics = self.dropout.sample(B, training=training)
        drop_image = drop_image.to(self.device)
        drop_omics = drop_omics.to(self.device)

        with _maybe_autocast(self.use_amp, self._amp_dtype):
            outputs = self.model(images, omics, drop_image, drop_omics)
            loss, components = self.criterion(outputs, omics, drop_image, drop_omics)

        return loss, components, self.dropout.describe(drop_image.cpu(), drop_omics.cpu())

    def train_epoch(self) -> Dict[str, float]:
        """Run one full training epoch and return aggregated metrics.

        Returns:
            Dict of epoch-averaged loss components and ``"epoch_time_s"``.
        """
        self.model.train()
        agg: Dict[str, float] = defaultdict(float)
        n_steps = 0
        t0 = time.time()

        for batch in self.train_loader:
            self.optimizer.zero_grad(set_to_none=True)
            loss, components, scenario_counts = self._step(batch, training=True)

            if not loss.isfinite():
                logger.warning("Non-finite loss at step %d; skipping update.", self.global_step)
                continue

            if self.use_amp and self._amp_dtype == torch.float16:
                self.scaler.scale(loss).backward()
                if self.clip_grad_norm is not None:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.clip_grad_norm is not None:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                self.optimizer.step()

            for k, v in components.items():
                agg[k] += v
            for k, v in scenario_counts.items():
                agg[f"scenario/{k}"] += v
            n_steps += 1
            self.global_step += 1

            if self.is_primary and self.global_step % self.log_every == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                parts = [f"step={self.global_step}", f"lr={lr:.2e}"]
                parts += [f"{k}={v:.4f}" for k, v in components.items()]
                logger.info("  ".join(parts))

        self.scheduler.step()
        metrics = {k: v / max(n_steps, 1) for k, v in agg.items()}
        metrics["epoch_time_s"] = time.time() - t0
        return metrics

    @torch.no_grad()
    def val_epoch(self) -> Dict[str, float]:
        """Run one full validation epoch and return aggregated metrics.

        Returns:
            Dict of epoch-averaged loss components, or an empty dict if no
            validation loader was provided.
        """
        if self.val_loader is None:
            return {}
        self.model.eval()
        agg: Dict[str, float] = defaultdict(float)
        n_steps = 0
        for batch in self.val_loader:
            _, components, scenario_counts = self._step(batch, training=False)
            for k, v in components.items():
                agg[k] += v
            for k, v in scenario_counts.items():
                agg[f"scenario/{k}"] += v
            n_steps += 1
        return {k: v / max(n_steps, 1) for k, v in agg.items()}

    def run(self):
        """Train for ``max_epochs``, validating and checkpointing after each epoch."""
        logger.info(
            "Starting STORM training: %d epochs, device=%s, amp=%s",
            self.max_epochs, self.device, self.use_amp,
        )
        self._try_load_checkpoint()

        for epoch in range(self.epoch, self.max_epochs):
            self.epoch = epoch
            if self.is_distributed:
                self.train_loader.sampler.set_epoch(epoch)

            train_metrics = self.train_epoch()
            val_metrics = self.val_epoch()

            if self.is_primary:
                self._log_epoch(epoch, train_metrics, val_metrics)
                self._save_checkpoint(epoch, val_metrics)

            if self.is_distributed:
                dist.barrier()

        logger.info("Training complete.")

    def _checkpoint_path(self, name: str = "latest") -> str:
        return os.path.join(self.checkpoint_dir, f"storm_{name}.pt")

    def _save_checkpoint(self, epoch: int, val_metrics: dict):
        state = {
            "epoch": epoch + 1,
            "global_step": self.global_step,
            "model": self._raw_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "best_val_loss": self.best_val_loss,
        }
        torch.save(state, self._checkpoint_path("latest"))

        val_loss = val_metrics.get("total", float("inf"))
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            torch.save(state, self._checkpoint_path("best"))
            logger.info("New best val loss %.4f — checkpoint saved.", val_loss)

        if self.save_every > 0 and (epoch + 1) % self.save_every == 0:
            torch.save(state, self._checkpoint_path(f"epoch{epoch + 1:04d}"))

    def _try_load_checkpoint(self):
        path = self._checkpoint_path("latest")
        if not os.path.exists(path):
            return
        ckpt = torch.load(path, map_location=self.device)
        self._raw_model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        self.scaler.load_state_dict(ckpt["scaler"])
        self.epoch = ckpt["epoch"]
        self.global_step = ckpt["global_step"]
        self.best_val_loss = ckpt.get("best_val_loss", float("inf"))
        logger.info("Resumed from checkpoint at epoch %d.", self.epoch)

    def _log_epoch(self, epoch: int, train: dict, val: dict):
        def _fmt(d: dict, prefix: str) -> str:
            return "  ".join(
                f"{prefix}/{k}={v:.4f}" for k, v in d.items()
                if not k.startswith("scenario/") and not k.endswith("_s")
            )
        logger.info(
            "[Epoch %d/%d]  %s  %s",
            epoch + 1, self.max_epochs,
            _fmt(train, "train"),
            _fmt(val, "val") if val else "no_val",
        )
        for k, v in train.items():
            if k.startswith("scenario/"):
                logger.debug("  %s=%.1f", k, v)


def setup_distributed(local_rank: int, backend: str = "nccl") -> int:
    """Initialise ``torch.distributed`` from torchrun environment variables.

    Args:
        local_rank: Local GPU index for this process.
        backend: Distributed backend; ``"nccl"`` for GPU, ``"gloo"`` for CPU.

    Returns:
        Global rank of this process.
    """
    dist.init_process_group(backend=backend)
    torch.cuda.set_device(local_rank)
    return dist.get_rank()
