import os
import random

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from omegaconf import OmegaConf


def seed_everything(seed: int) -> None:
    # The data loader samples batch offsets with the global numpy RNG,
    # and model weights are initialized with the global torch RNG.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@hydra.main(version_base=None, config_path="cfgs", config_name="training")
def main(cfg: DictConfig) -> None:
    # Seed before the trainer builds the model and data loaders
    if cfg.get("seed") is not None:
        seed_everything(cfg.seed)

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
