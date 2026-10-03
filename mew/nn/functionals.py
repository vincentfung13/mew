import math
from einops import einsum, reduce, rearrange

import torch


def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    # x -> (... dim)

    # Compute exp(x) - subtract max first for stability
    dim_max = x.max(dim=dim, keepdim=True).values
    x = x - dim_max
    exp_x = x.exp()

    # Compute exp_sum
    exp_sum = exp_x.sum(dim=dim, keepdim=True)

    return exp_x / exp_sum


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    # logits -> (batch ... vocab_size)
    # targets -> (batch ...)
    # Compute log_softmax directly for numerical stability
    # log_softmax(x) = x - log(sum(exp(x - max(x))))
    max_logits = reduce(logits, "... vocab_size -> ... ()", reduction="max")
    logits_shifted = logits - max_logits  # (batch ... vocab_size)
    log_exp_sum = reduce(
        logits_shifted.exp(), "... vocab_size -> ... ()", reduction="sum"
    ).log()
    log_softmax = logits_shifted - log_exp_sum

    # Directly gather log probabilities for target classes using einops for clarity
    # Flatten all batch dimensions for gather operation
    log_softmax_flat = rearrange(log_softmax, "... vocab_size -> (...) vocab_size")
    targets_flat = rearrange(targets, "... -> (...) 1")

    # Gather the log probabilities for the target classes
    log_probs_flat = log_softmax_flat.gather(1, targets_flat).squeeze(1)

    # Reshape back to original batch dimensions
    original_shape = logits.shape[:-1]
    log_probs = log_probs_flat.view(original_shape)
    loss = -reduce(log_probs, "... -> ()", reduction="mean")

    return loss


def scaled_dot_product(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor = None,
) -> torch.Tensor:
    # q -> (batch ... seq_len_q d_head)
    # k -> (batch ... seq_len_k d_head)
    # v -> (batch ... seq_len_k d_head)
    # mask -> (seq_len seq_len)
    # returns output -> (batch ... seq_len_q d_head)
    assert q.size(-3) == k.size(-3), "GQA is not supported for naive SDPA"
    qk_dot = einsum(
        q,
        k,
        "batch ... seq_len_q d_head, batch ... seq_len_k d_head -> batch ... seq_len_q seq_len_k",
    )
    qk_dot = qk_dot / math.sqrt(q.size()[-1])

    # Before softmax, apply mask if specified
    if mask is not None:
        # Set very negative value (e.g., -1e9) to masked positions
        qk_dot = qk_dot.masked_fill(~mask, float("-inf"))
    attn_weights = softmax(qk_dot, dim=-1)  # (batch ... seq_len_q seq_len_k)

    # Reduce with V
    aggregated_v = einsum(
        attn_weights,
        v,
        "batch ... seq_len_q seq_len_k, batch ... seq_len_k d_head -> batch ... seq_len_q d_head",
    )
    return aggregated_v
