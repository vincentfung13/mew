import os
import logging

from omegaconf import DictConfig

import torch

from mew.tokenization.bpe import BPETokenizer
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

LOGGER = logging.getLogger(__name__)


torch.autograd.set_detect_anomaly(True)


class NPTTrainer:
    def __init__(self, cfg: DictConfig, wandb=None):
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
            "Starting running evalidation, init val dataloader from %s with seq_len=%d",
            cfg.data.val_file,
            cfg.data.seq_len,
        )
        self.val_data_loader = NumpyBatchLoader(
            data=cfg.data.val_file,
            seq_len=cfg.data.seq_len,
            batch_size=cfg.data.batch_size,
            is_training=False,
        )

        # Init model
        LOGGER.info("Init LM model and optimizer")
        self.model = build_model(
            cfg=cfg,
            device=cfg.device,
        )

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
        self.tokenizer = BPETokenizer.from_dir(cfg.data.tokenizer_path)

        self.wandb = wandb

        # Load checkpoint
        if cfg.trainer.resume:
            LOGGER.info(
                "Resume training from checkpoint: %s",
                cfg.trainer.resume_checkpoint_path,
            )
            load_checkpoint(
                src=cfg.trainer.resume_checkpoint_path,
                model=self.model,
                optimizer=self.optim,
                lr_scheduler=self.lr_scheduler,
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

    def train(self):
        # Main training loop
        for step in range(1, self.cfg.trainer.total_steps + 1):
            # Get batch
            data, target = self.train_data_loader.get_batch(
                device=self.cfg.device,
            )

            # Forward
            with self._autocast():
                logits = self.model(data)
                loss = cross_entropy(logits.float(), target)
                scaled_loss = loss / self.cfg.optim.grad_accumulation_steps
            scaled_loss.backward()

            # log weight norm and gradient norm
            if step % self.cfg.trainer.log_freq == 0:
                log_gradient_norm_and_weight_norm(
                    wandb=self.wandb,
                    model=self.model,
                    step=step,
                )

            # Backward
            if step % self.cfg.optim.grad_accumulation_steps == 0:
                if self.cfg.optim.grad_clip_norm > 0:
                    clip_gradients(
                        parameters=self.model.parameters(),
                        max_l2_norm=self.cfg.optim.grad_clip_norm,
                    )
                self.optim.step()
                self.lr_scheduler.step()
                self.optim.zero_grad()

            # log progress
            if step % self.cfg.trainer.log_freq == 0:
                with torch.no_grad(), self._autocast():
                    # Run mini val
                    val_data, val_target = self.val_data_loader.get_batch(
                        device=self.cfg.device
                    )
                    val_logits = self.model(val_data)
                    val_loss = cross_entropy(val_logits.float(), val_target)

                LOGGER.info(
                    "Step [%d/%d], Train Loss: %.4f, Val Loss: %.4f LR: %.6f",
                    step,
                    self.cfg.trainer.total_steps,
                    loss.item(),
                    val_loss.item(),
                    self.optim.param_groups[0]["lr"],
                )
                if self.wandb is not None:
                    self.wandb.log(
                        {
                            "train/loss": loss.item(),
                            "val/loss": val_loss.item(),
                            "train/lr": self.optim.param_groups[0]["lr"],
                        },
                        step=step,
                    )

            # Save checkpoint
            if step % self.cfg.trainer.save_freq == 0:
                if not os.path.isdir(self.cfg.save_dir):
                    os.makedirs(self.cfg.save_dir)
                ckpt_path = os.path.join(
                    self.cfg.save_dir,
                    f"checkpoint_step_{step}.pt",
                )
                save_checkpoint(
                    model=self.model,
                    optimizer=self.optim,
                    iteration=step,
                    output_path=ckpt_path,
                    lr_scheduler=self.lr_scheduler,
                )
                LOGGER.info(f"Checkpoint saved to {ckpt_path}")

    def _autocast(self):
        return torch.autocast(
            device_type=torch.device(self.cfg.device).type,
            dtype=self.amp_dtype,
            enabled=self.cfg.trainer.amp.enable,
        )
