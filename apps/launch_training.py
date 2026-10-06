import os
import random

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from omegaconf import OmegaConf

from mew.perf.gpu_specs import load_gpu_specs, peak_tflops_per_second
from mew.trainers.npt_trainer import NPTTrainer
from mew.trainers.dist_context import DistContext

AMP_DTYPES = {"bf16": torch.bfloat16}


def seed_everything(seed: int) -> None:
    # The data loader samples batch offsets with the global numpy RNG,
    # and model weights are initialized with the global torch RNG.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_peak_tflops(cfg: DictConfig, dist_context: DistContext) -> None:
    # Fill trainer.perf.peak_tflops from the GPU model (cfgs/gpu_specs.yaml)
    if cfg.trainer.perf.peak_tflops is not None:
        return
    dtype = (
        AMP_DTYPES[cfg.trainer.amp.dtype] if cfg.trainer.amp.enable else torch.float32
    )
    gpu_specs = load_gpu_specs(cfg.gpu_specs)
    cfg.trainer.perf.peak_tflops = peak_tflops_per_second(
        dist_context.device, dtype, gpu_specs
    )


def _validate_cfg(cfg: DictConfig):
    assert cfg.device == "cuda", "Training needs to be run on GPUs!"
    assert cfg.task_name in ["npt_training"], f"Unknown task_name: {cfg.task_name}!"
    if cfg.trainer.log_grads_and_weights_norm:
        assert (
            cfg.wandb.enable
        ), "wandb is needed when log_grads_and_weights_norm is True!"


@hydra.main(version_base=None, config_path="cfgs", config_name="training")
def main(cfg: DictConfig) -> None:
    _validate_cfg(cfg)

    # For distributed training
    dist_context = DistContext.from_env()
    torch.cuda.set_device(dist_context.device)

    # Seed before the trainer builds the model and data loaders
    if cfg.get("seed") is not None:
        seed_everything(cfg.seed)

    # Resolve before saving, so the saved config records the peak actually used
    resolve_peak_tflops(cfg, dist_context)

    # Copy tokenizer to save dir
    os.system(f"cp -r {cfg.data.tokenizer_path} {cfg.save_dir}/tokenizer")

    # Save a copy of the config
    OmegaConf.save(config=cfg, f=f"{cfg.save_dir}/training_cfg.yaml")

    # Init wandb for experiment tracking
    wandb = None
    if cfg.wandb.enable:
        container = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

        import wandb

        wandb.init(project=cfg.wandb.project, name=cfg.run_name, config=container)

    # Launch training job
    trainer = NPTTrainer(cfg, dist_context=dist_context, wandb=wandb)
    trainer.train()


if __name__ == "__main__":
    main()
