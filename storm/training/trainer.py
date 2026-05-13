"""
STORMTrainer — training loop with per-step modality dropout.

Key design points
-----------------
* Modality dropout masks are sampled fresh every step (not baked into the
  DataLoader) so each pass over the same batch sees a different combination of
  dropped modalities.
* The trainer keeps a running tally of per-scenario spot counts for logging so
  you can monitor whether dropout probabilities are behaving as expected.
* Checkpoint state includes the model, optimizer, scheduler, scaler, epoch, and
  step counters, making training fully resumable.
* Distributed training (DDP) is supported via standard torch.distributed; call
  setup_distributed() before constructing the trainer in a multi-GPU job.
"""

from __future__ import annotations

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


class STORMTrainer:
    """
    End-to-end training harness for the STORM model.

    Args:
        model:              STORMModel instance.
        train_loader:       DataLoader yielding dicts with "image", "omics" keys.
        val_loader:         Optional validation DataLoader.
        loss_cfg:           Keyword arguments forwarded to STORMLoss.
        optim_cfg:          Keyword arguments forwarded to build_optimizer.
        scheduler_cfg:      Keyword arguments forwarded to build_scheduler.
        dropout_cfg:        ModalityDropoutConfig; controls per-spot masking rates.
        max_epochs:         Total number of training epochs.
        warmup_epochs:      Epochs of linear LR warmup.
        device:             Torch device string or device object.
        use_amp:            Enable automatic mixed precision (bfloat16).
        amp_dtype:          "bfloat16" or "float16" when use_amp=True.
        checkpoint_dir:     Directory to save/load checkpoints.
        save_every:         Save a numbered checkpoint every N epochs (0=disabled).
        log_every:          Log metrics every N steps.
        clip_grad_norm:     Max gradient norm; None to disable clipping.
        local_rank:         Local GPU rank for DDP (0 = single-GPU / rank-0).
        is_distributed:     True when running DDP.
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

        # Loss
        self.criterion = STORMLoss(**(loss_cfg or {})).to(self.device)

        # Optimizer and scheduler
        optim_cfg = optim_cfg or {}
        self.optimizer = build_optimizer(self._raw_model, **optim_cfg)

        total_epochs = max_epochs
        sched_cfg = dict(
            warmup_epochs=warmup_epochs,
            total_epochs=total_epochs,
        )
        if scheduler_cfg:
            sched_cfg.update(scheduler_cfg)
        self.scheduler = build_scheduler(self.optimizer, **sched_cfg)

        # AMP
        self.use_amp = use_amp and torch.cuda.is_available()
        self._amp_dtype = torch.bfloat16 if amp_dtype == "bfloat16" else torch.float16
        self.scaler = GradScaler(enabled=(self.use_amp and self._amp_dtype == torch.float16))

        # Modality dropout sampler
        self.dropout = ModalityDropout(dropout_cfg or ModalityDropoutConfig())

        # Checkpoint
        self.checkpoint_dir = checkpoint_dir
        if self.is_primary:
            os.makedirs(checkpoint_dir, exist_ok=True)

        # State
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def _raw_model(self) -> STORMModel:
        return self.model.module if isinstance(self.model, DDP) else self.model

    # ------------------------------------------------------------------
    # Core training step
    # ------------------------------------------------------------------

    def _step(self, batch: dict, training: bool) -> Tuple[torch.Tensor, Dict[str, float], dict]:
        images = batch["image"].to(self.device, non_blocking=True)   # (B, 3, H, W)
        omics = batch["omics"].to(self.device, non_blocking=True)    # (B, G)

        B = images.shape[0]

        # Sample per-spot modality dropout flags
        drop_image, drop_omics = self.dropout.sample(B, training=training)
        drop_image = drop_image.to(self.device)
        drop_omics = drop_omics.to(self.device)

        ctx = (
            torch.cuda.amp.autocast(dtype=self._amp_dtype)
            if self.use_amp
            else torch.no_grad() if not training else _null_context()
        )

        with ctx:
            outputs = self.model(images, omics, drop_image, drop_omics)
            loss, components = self.criterion(outputs, omics, drop_image, drop_omics)

        scenario_counts = self.dropout.describe(drop_image.cpu(), drop_omics.cpu())
        return loss, components, scenario_counts

    # ------------------------------------------------------------------
    # Epoch loops
    # ------------------------------------------------------------------

    def train_epoch(self) -> Dict[str, float]:
        self.model.train()
        agg = defaultdict(float)
        n_steps = 0
        t0 = time.time()

        for batch in self.train_loader:
            self.optimizer.zero_grad(set_to_none=True)

            loss, components, scenario_counts = self._step(batch, training=True)

            if not loss.isfinite():
                logger.warning(f"Non-finite loss at step {self.global_step}; skipping.")
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

            # Accumulate metrics
            for k, v in components.items():
                agg[k] += v
            for k, v in scenario_counts.items():
                agg[f"scenario/{k}"] += v
            n_steps += 1
            self.global_step += 1

            if self.is_primary and self.global_step % self.log_every == 0:
                step_lr = self.optimizer.param_groups[0]["lr"]
                log_parts = [f"step={self.global_step}", f"lr={step_lr:.2e}"]
                log_parts += [f"{k}={v:.4f}" for k, v in components.items()]
                logger.info("  ".join(log_parts))

        self.scheduler.step()

        epoch_metrics = {k: v / max(n_steps, 1) for k, v in agg.items()}
        epoch_metrics["epoch_time_s"] = time.time() - t0
        return epoch_metrics

    @torch.no_grad()
    def val_epoch(self) -> Dict[str, float]:
        if self.val_loader is None:
            return {}
        self.model.eval()
        agg = defaultdict(float)
        n_steps = 0

        for batch in self.val_loader:
            _, components, scenario_counts = self._step(batch, training=False)
            for k, v in components.items():
                agg[k] += v
            for k, v in scenario_counts.items():
                agg[f"scenario/{k}"] += v
            n_steps += 1

        return {k: v / max(n_steps, 1) for k, v in agg.items()}

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    def run(self):
        """Train for max_epochs and run validation after each epoch."""
        logger.info(
            f"Starting STORM training: {self.max_epochs} epochs, "
            f"device={self.device}, amp={self.use_amp}"
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

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

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
            logger.info(f"  → New best val loss: {val_loss:.4f} (saved checkpoint)")

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
        logger.info(f"Resumed from checkpoint at epoch {self.epoch}")

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log_epoch(self, epoch: int, train: dict, val: dict):
        def _fmt(d: dict, prefix: str) -> str:
            parts = [f"{prefix}/{k}={v:.4f}" for k, v in d.items()
                     if not k.startswith("scenario/") and not k.endswith("_s")]
            return "  ".join(parts)

        logger.info(
            f"[Epoch {epoch + 1}/{self.max_epochs}]  "
            f"{_fmt(train, 'train')}  "
            f"{_fmt(val, 'val') if val else 'no_val'}"
        )

        # Scenario distribution
        for k, v in train.items():
            if k.startswith("scenario/"):
                logger.debug(f"  {k}={v:.1f}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

import contextlib


@contextlib.contextmanager
def _null_context():
    yield


def setup_distributed(local_rank: int, backend: str = "nccl") -> int:
    """Initialise torch.distributed from environment variables (torchrun-style)."""
    dist.init_process_group(backend=backend)
    torch.cuda.set_device(local_rank)
    return dist.get_rank()
