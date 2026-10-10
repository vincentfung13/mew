import logging
import os
import shutil

import hydra
import torch
import torch.distributed as dist
from hydra.core.hydra_config import HydraConfig
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


def sync_run_id(cfg: DictConfig, dist_context: DistContext) -> None:
    # Each rank resolves ${now:...} on its own, so ranks that start in different
    # seconds disagree on run_id, and hence on save_dir. Use rank 0's everywhere.
    if dist_context.world_size == 1:
        return
    run_id = [cfg.run_id]
    dist.broadcast_object_list(run_id, src=0)
    if run_id[0] == cfg.run_id:
        return

    # Hydra created this rank's output dir and opened its log file before main()
    # ran, so move the log into rank 0's dir and drop the stray dir
    stray_dir = HydraConfig.get().runtime.output_dir
    cfg.run_id = run_id[0]
    os.makedirs(cfg.save_dir, exist_ok=True)
    _move_file_logs(src_dir=stray_dir, dst_dir=cfg.save_dir)
    _remove_stray_dir(stray_dir)


def _move_file_logs(src_dir: str, dst_dir: str) -> None:
    root = logging.getLogger()
    for handler in list(root.handlers):
        if not isinstance(handler, logging.FileHandler):
            continue
        if os.path.dirname(handler.baseFilename) != os.path.abspath(src_dir):
            continue
        src_path = handler.baseFilename
        dst_path = os.path.join(os.path.abspath(dst_dir), os.path.basename(src_path))
        root.removeHandler(handler)
        handler.close()
        # Keep what was logged before the move
        with open(src_path) as src, open(dst_path, "a") as dst:
            dst.write(src.read())
        os.remove(src_path)
        new_handler = logging.FileHandler(dst_path)
        new_handler.setFormatter(handler.formatter)
        new_handler.setLevel(handler.level)
        root.addHandler(new_handler)


def _remove_stray_dir(stray_dir: str) -> None:
    # Only remove what Hydra created. Several ranks may share one stray dir, so
    # the dir itself goes once the last rank has moved its log out.
    shutil.rmtree(os.path.join(stray_dir, ".hydra"), ignore_errors=True)
    try:
        os.rmdir(stray_dir)
    except OSError:
        pass


def _quiet_console(level: int) -> None:
    # Raise only the console handlers' level: the console shows rank 0's lines
    # once, plus warnings and errors from every rank, while each rank's own log
    # file keeps its full INFO output (e.g. its local loss and MFU).
    # FileHandler subclasses StreamHandler, so exclude it explicitly.
    for handler in logging.getLogger().handlers:
        if isinstance(handler, logging.StreamHandler) and not isinstance(
            handler, logging.FileHandler
        ):
            handler.setLevel(level)


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
    sync_run_id(cfg, dist_context)

    if not dist_context.is_main:
        _quiet_console(logging.WARNING)

    # Seed weight init before the trainer builds the model. Data sampling is
    # seeded per rank inside the trainer from [seed, rank].
    torch.manual_seed(cfg.seed)

    # Resolve before saving, so the saved config records the peak actually used
    resolve_peak_tflops(cfg, dist_context)

    # Files in save_dir and the wandb run are written once, by rank 0. Other
    # ranks pass wandb=None to the trainer.
    if dist_context.is_main:
        # Copy tokenizer to save dir
        os.system(f"cp -r {cfg.data.tokenizer_path} {cfg.save_dir}/tokenizer")

        # Save a copy of the config
        OmegaConf.save(config=cfg, f=f"{cfg.save_dir}/training_cfg.yaml")

    # Init wandb for experiment tracking
    wandb = None
    if cfg.wandb.enable and dist_context.is_main:
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
