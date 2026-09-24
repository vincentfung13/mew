import math
import torch
import triton

from mew.nn.flash_attention.kernels.flash_foward import flash_fwd_kernel


class FlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        Q: torch.Tensor,  # (batch, n_q, d)
        K: torch.Tensor,  # (batch, n_kv, d)
        V: torch.Tensor,  # (batch, n_kv, d)
        is_causal: bool = False,
        cfg: dict = {"Q_TILE_SIZE": 16, "K_TILE_SIZE": 16},
    ):
        batch, n_q, d = Q.size()
        batch, n_kv, _ = K.size()
        scale = math.sqrt(d)

        # Init output buffer
        O = torch.empty_like(Q, dtype=torch.float32)
        L = torch.empty((batch, n_q), dtype=torch.float32)

        # Launch triton kernel (launch grid is [n_queries, batch])
        flash_fwd_kernel[
            (
                triton.cdiv(n_q, cfg["Q_TILE_SIZE"]),
                triton.cdiv(batch, cfg["K_TILE_SIZE"]),
            )
        ](
            Q,
            K,
            V,
            O,
            L,
            Q.stride(0),
            Q.stride(1),
            Q.stride(2),
            K.stride(0),
            K.stride(1),
            K.stride(2),
            V.stride(0),
            V.stride(1),
            V.stride(2),
            O.stride(0),
            O.stride(1),
            O.stride(2),
            L.stride(0),
            L.stride(1),
            N_QUERIES=n_q,
            NKEYS=n_kv,
            scale=scale,
            D=d,
            Q_TILE_SIZE=cfg["Q_TILE_SIZE"],
            K_TILE_SIZE=cfg["K_TILE_SIZE"],
        )
        ctx.save_for_backward(L, Q, K, V, O)
        return O

    @staticmethod
    def backward(ctx):
        raise NotImplementedError
