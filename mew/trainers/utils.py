import torch

from mew.data_loaders.numpy_batch_loader import NumpyBatchLoader


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    output_path: str,
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler = None,
    train_batch_spawned: int | None = None,
    val_batch_spawned: int | None = None,
):
    state_dicts = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
    }
    if lr_scheduler is not None:
        state_dicts["lr_scheduler"] = lr_scheduler.state_dict()
    if train_batch_spawned is not None:
        state_dicts["train_batch_spawned"] = train_batch_spawned
    if val_batch_spawned is not None:
        state_dicts["val_batch_spawned"] = val_batch_spawned
    torch.save(state_dicts, output_path)


def load_checkpoint(
    src: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer = None,
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler = None,
    train_data_loader: NumpyBatchLoader | None = None,
    val_data_loader: NumpyBatchLoader | None = None,
) -> int:
    state_dicts = torch.load(src)
    model.load_state_dict(state_dicts["model"], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(state_dicts["optimizer"])
    if lr_scheduler is not None:
        lr_scheduler.load_state_dict(state_dicts["lr_scheduler"])
    if train_data_loader is not None and "train_batch_spawned" in state_dicts:
        train_data_loader.resume(state_dicts["train_batch_spawned"])
    if val_data_loader is not None and "val_batch_spawned" in state_dicts:
        val_data_loader.resume(state_dicts["val_batch_spawned"])
    return state_dicts["iteration"]


def log_gradient_norm_and_weight_norm(wandb, model: torch.nn.Module, step: int):
    if wandb is None:
        return
    metrics = {}
    for name, p in model.named_parameters():
        w = p.data
        w_norm = (w.float().pow(2).sum()).sqrt().item()
        if p.grad is not None:
            g_norm = (p.grad.data.float().pow(2).sum()).sqrt().item()
        else:
            g_norm = 0.0
        metrics[f"param/{name}/weight_norm"] = w_norm
        metrics[f"param/{name}/grad_norm"] = g_norm
    wandb.log(metrics, step=step)
