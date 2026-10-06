# Performance/Profiling-Related Utils
import torch


def module_flops_per_token(
    module: torch.nn.Module,
    seq_len: int,
) -> int:
    if hasattr(module, "flops_per_token"):
        # Directly use the result and stop iterating
        flops_per_token = module.flops_per_token(seq_len)
    else:
        # If the module has self-defined params but no flops_per_token method
        # raise an exception
        params = [item[0] for item in module.named_parameters(recurse=False)]
        cls = type(module)
        cls_name = f"{cls.__module__}.{cls.__qualname__}"
        if len(params) > 0:
            raise TypeError(
                f'{cls_name} has named params {params} but no "flops_per_token" method.'
            )

        # Loop through it's module to get the total flops per token
        flops_per_token = 0
        for child_module in module.children():
            flops_per_token += module_flops_per_token(
                module=child_module, seq_len=seq_len
            )
    return flops_per_token


def compute_mfu(
    tokens_per_s: float, model_flops_per_token: int, peak_tflops: float
) -> float:
    mfu = tokens_per_s * model_flops_per_token / (peak_tflops * 1e12)
    return mfu


def peak_memory_stats(device: torch.device) -> dict[str, float]:
    if device.type != "cuda":
        return {}

    peak_mem_allocated_gib = torch.cuda.max_memory_allocated(device) / 2**30
    peak_mem_reserved_gib = torch.cuda.max_memory_reserved(device) / 2**30
    return {
        "peak_mem_allocated_gib": peak_mem_allocated_gib,
        "peak_mem_reserved_gib": peak_mem_reserved_gib,
    }
