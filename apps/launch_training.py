import os
import random

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from omegaconf import OmegaConf

from mew.perf.gpu_specs import GPUSpec, peak_flops_per_second

AMP_DTYPES = {"bf16": torch.bfloat16}


def seed_everything(seed: int) -> None:
    # The data loader samples batch offsets with the global numpy RNG,
    # and model weights are initialized with the global torch RNG.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_gpu_specs(raw) -> list[GPUSpec]:
    # Convert the cfg.gpu_specs entries (cfgs/gpu_specs.yaml) into GPUSpecs,
    # naming the offending entry if one is malformed.
    specs = []
    for index, entry in enumerate(OmegaConf.to_container(raw)):
        try:
            specs.append(GPUSpec(**entry))
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid gpu_specs entry {index}: {error}") from error
    return specs


def resolve_peak_tflops(cfg: DictConfig) -> None:
    # Fill trainer.perf.peak_tflops from the GPU model (cfgs/gpu_specs.yaml)
    # unless it was set explicitly. Left null on non-CUDA devices, where MFU is
    # undefined.
    if cfg.trainer.perf.peak_tflops is not None:
        return
    if torch.device(cfg.device).type != "cuda":
        return
    dtype = (
        AMP_DTYPES[cfg.trainer.amp.dtype] if cfg.trainer.amp.enable else torch.float32
    )
    gpu_specs = load_gpu_specs(cfg.gpu_specs)
    cfg.trainer.perf.peak_tflops = (
        peak_flops_per_second(cfg.device, dtype, gpu_specs) / 1e12
    )


@hydra.main(version_base=None, config_path="cfgs", config_name="training")
def main(cfg: DictConfig) -> None:
    # Seed before the trainer builds the model and data loaders
    if cfg.get("seed") is not None:
        seed_everything(cfg.seed)

    # Resolve before saving, so the saved config records the peak actually used
    resolve_peak_tflops(cfg)

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
    if cfg.task_name == "npt_training":
        from mew.trainers.npt_trainer import NPTTrainer

        trainer = NPTTrainer(cfg, wandb)
        trainer.train()
    else:
        raise ValueError(f"Unknown task_name: {cfg.task_name}")


if __name__ == "__main__":
    main()
