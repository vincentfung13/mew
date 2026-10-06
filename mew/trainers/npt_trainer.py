import os
import logging

from omegaconf import DictConfig

import torch

from mew.nn.utils import build_model
from mew.nn.functionals import cross_entropy
from mew.optimizers.adamw import AdamW
from mew.optimizers.lr_scheduling import CosineAnnealingScheduler
from mew.optimizers.utils import clip_gradients
from mew.data_loaders.numpy_batch_loader import NumpyBatchLoader
from mew.trainers.utils import (
    save_checkpoint,
    load_checkpoint,
    log_gradient_norm_and_weight_norm,
)
from mew.trainers.dist_context import DistContext
from mew.perf.utils import module_flops_per_token
from mew.perf.throughput_meter import ThroughputMeter

LOGGER = logging.getLogger(__name__)


class TrainStep:
    def __init__(self, cfg: DictConfig, dist_context: DistContext):
        self.dist_context = dist_context

        # Init model
        LOGGER.info("Init LM model and optimizer")
        self.model = build_model(
            cfg=cfg,
            device=dist_context.device,
        )
        self.model_flops_per_token = (
            module_flops_per_token(module=self.model, seq_len=cfg.data.seq_len) * 3
        )
        LOGGER.info(f"Model initialized, FLOPs/token: {self.model_flops_per_token}")

        # Init optimizer and lr scheduler
        self.optim = AdamW(
            params=self.model.parameters(),
            lr=cfg.optim.lr,
            weight_decay=cfg.optim.weight_decay,
            betas=cfg.optim.betas,
            eps=cfg.optim.eps,
        )
        self.lr_scheduler = CosineAnnealingScheduler(
            optimizer=self.optim,
            max_learning_rate=cfg.optim.lr,
            min_learning_rate=cfg.optim.min_lr,
            warmup_iters=cfg.optim.warmup_iters,
            cosine_cycle_iters=cfg.optim.cosine_cycle_iters,
        )

        # AMP
        self.amp_dtype = None
        # Only supporting bf16 for now
        if cfg.trainer.amp.enable:
            try:
                self.amp_dtype = {
                    "bf16": torch.bfloat16,
                }[cfg.trainer.amp.dtype]
            except KeyError as error:
                raise ValueError(
                    f"Unsupported AMP dtype: {cfg.trainer.amp.dtype}"
                ) from error

        self.cfg = cfg

    def forward(
        self,
        data: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        with self._autocast():
            logits = self.model(data)
            loss = cross_entropy(logits.float(), target)
        return loss

    def backward(
        self,
        loss: torch.Tensor,
    ):
        # Handle gradient accmulation
        scaled_loss = loss / self.cfg.optim.grad_accumulation_steps
        scaled_loss.backward()

    def optim_and_lr_scheduler_step(self):
        if self.cfg.optim.grad_clip_norm > 0:
            clip_gradients(
                parameters=self.model.parameters(),
                max_l2_norm=self.cfg.optim.grad_clip_norm,
            )
        self.optim.step()
        self.lr_scheduler.step()
        self.optim.zero_grad()

    def _autocast(self):
        return torch.autocast(
            device_type=self.dist_context.device.type,
            dtype=self.amp_dtype,
            enabled=self.cfg.trainer.amp.enable,
        )


class NPTTrainer:
    def __init__(self, cfg: DictConfig, dist_context: DistContext, wandb=None):
        self.dist_context = dist_context

        # Init dataloader
        LOGGER.info(
            "Init train dataloader from %s with seq_len=%d",
            cfg.data.train_file,
            cfg.data.seq_len,
        )
        self.train_data_loader = NumpyBatchLoader(
            data=cfg.data.train_file,
            seq_len=cfg.data.seq_len,
            batch_size=cfg.data.batch_size,
            is_training=True,
        )
        LOGGER.info(
            "Starting running validation, init val dataloader from %s with seq_len=%d",
            cfg.data.val_file,
            cfg.data.seq_len,
        )
        self.val_data_loader = NumpyBatchLoader(
            data=cfg.data.val_file,
            seq_len=cfg.data.seq_len,
            batch_size=cfg.data.batch_size,
            is_training=False,
        )

        # Init train step
        self.train_step = TrainStep(cfg=cfg, dist_context=dist_context)
        self.start_step = 0
        if cfg.trainer.resume:
            LOGGER.info(
                "Resume training from checkpoint: %s",
                cfg.trainer.resume_checkpoint_path,
            )
            self.start_step = load_checkpoint(
                src=cfg.trainer.resume_checkpoint_path,
                model=self.train_step.model,
                optimizer=self.train_step.optim,
                lr_scheduler=self.train_step.lr_scheduler,
            )

        # Throughput meter for MFU logging
        self.throughput_meter = ThroughputMeter(
            model_flops_per_token=self.train_step.model_flops_per_token,
            device_peak_tflops=cfg.trainer.perf.peak_tflops,
            device=dist_context.device,
        )

        self.wandb = wandb
        self.cfg = cfg

    def train(self):
        # Start profiling clock once
        self.throughput_meter.start()

        # Main training loop
        for step in range(self.start_step + 1, self.cfg.trainer.total_steps + 1):
            # Get batch
            data, target = self.train_data_loader.get_batch(
                device=self.dist_context.device,
            )

            # Forward and backward
            loss = self.train_step.forward(data=data, target=target)
            self.train_step.backward(loss)

            # Optional: log grad and weight norm (adds overhead to training)
            if (
                self.cfg.trainer.log_grads_and_weights_norm
                and step % self.cfg.trainer.log_freq == 0
                and step % self.cfg.optim.grad_accumulation_steps == 0
            ):
                log_gradient_norm_and_weight_norm(
                    wandb=self.wandb,
                    model=self.train_step.model,
                    step=step,
                )

            # Optim and lr scheduler step
            if step % self.cfg.optim.grad_accumulation_steps == 0:
                self.train_step.optim_and_lr_scheduler_step()

            # step throughput_meter
            self.throughput_meter.step(data.numel())

            # log progress
            if step % self.cfg.trainer.log_freq == 0:
                with torch.no_grad():
                    # Run mini val
                    val_data, val_target = self.val_data_loader.get_batch(
                        device=self.dist_context.device
                    )
                    val_loss = self.train_step.forward(data=val_data, target=val_target)

                # Create item for logging
                log_item = {
                    "train/loss": loss.item(),
                    "val/loss": val_loss.item(),
                    "train/lr": self.train_step.optim.param_groups[0]["lr"],
                }
                # retrieve throughput stats
                for key, val in self.throughput_meter.report().items():
                    log_item["perf/" + key] = val

                if self.wandb is not None:
                    self.wandb.log(log_item, step=step)

                LOGGER.info(
                    "Step [%d/%d] %s",
                    step,
                    self.cfg.trainer.total_steps,
                    " | ".join(
                        f"{k}={_logging_format(k, v)}" for k, v in log_item.items()
                    ),
                )

                # restart throughput meter
                self.throughput_meter.start()

            # Save checkpoint
            if step % self.cfg.trainer.save_freq == 0:
                if not os.path.isdir(self.cfg.save_dir):
                    os.makedirs(self.cfg.save_dir)
                ckpt_path = os.path.join(
                    self.cfg.save_dir,
                    f"checkpoint_step_{step}.pt",
                )
                save_checkpoint(
                    model=self.train_step.model,
                    optimizer=self.train_step.optim,
                    iteration=step,
                    output_path=ckpt_path,
                    lr_scheduler=self.train_step.lr_scheduler,
                )
                LOGGER.info(f"Checkpoint saved to {ckpt_path}")


def _logging_format(key: str, value: float) -> str:
    # Human-readable formatting for the log line; wandb gets the raw values.
    if key.endswith("mfu"):
        return f"{value:.1%}"
    if key.endswith("tokens_per_s"):
        return f"{value:,.0f}"
    if key.endswith("_gib"):
        return f"{value:.2f}"
    if key.endswith("lr"):
        return f"{value:.2e}"
    return f"{value:.4f}"
