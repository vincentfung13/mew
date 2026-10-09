import os

import hydra
import torch
import torch.distributed as dist
from omegaconf import DictConfig
from omegaconf import OmegaConf

from mew.perf.gpu_specs import load_gpu_specs, peak_tflops_per_second
from mew.trainers.npt_trainer import NPTTrainer
from mew.parallel.dist_context import DistContext

AMP_DTYPES = {"bf16": torch.bfloat16}


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
    assert cfg.seed is not None, "Please always set your seed for reproducible runs"

    if cfg.trainer.log_grads_and_weights_norm:
        assert (
            cfg.wandb.enable
        ), "wandb is needed when log_grads_and_weights_norm is True!"

    if cfg.parallel.backend == "nccl":
        assert torch.cuda.is_available(), "NCCL needs machines with GPUs!"

    if cfg.parallel.backend is not None:
        assert (
            dist.is_torchelastic_launched()
        ), "parallel.backend is set but script's not started with torchrun"


@hydra.main(version_base=None, config_path="cfgs", config_name="training")
def main(cfg: DictConfig) -> None:
    _validate_cfg(cfg)

    dist_context = DistContext.set_up_dist_env(dist_backend=cfg.parallel.backend)

    # Seed weight init before the trainer builds the model. Data sampling is
    # seeded per rank inside the trainer from [seed, rank].
    torch.manual_seed(cfg.seed)

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
    try:
        trainer = NPTTrainer(cfg, dist_context=dist_context, wandb=wandb)
        trainer.train()
    finally:
        dist_context.shutdown()


if __name__ == "__main__":
    main()
