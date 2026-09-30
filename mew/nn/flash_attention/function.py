import math
import torch
import triton

from mew.nn.flash_attention.kernels.forward import flash_fwd_kernel
from mew.nn.flash_attention.kernels.backward import (
    flash_bwd_kernel_dkv,
    flash_bwd_kernel_dq,
)


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
        assert torch.cuda.is_available()
        batch, n_q, d = Q.size()
        batch, n_kv, _ = K.size()
        scale = math.sqrt(d)

        # Init output buffer
        O_acc = torch.empty_like(Q, dtype=torch.float32, device="cuda")
        L = torch.empty((batch, n_q), dtype=torch.float32, device="cuda")

        # Launch triton kernel (launch grid is [n_queries, batch])
        flash_fwd_kernel[(triton.cdiv(n_q, cfg["Q_TILE_SIZE"]), batch)](
            Q,
            K,
            V,
            O_acc,
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
            O_acc.stride(0),
            O_acc.stride(1),
            O_acc.stride(2),
            L.stride(0),
            L.stride(1),
            N_QUERIES=n_q,
            N_KEYS=n_kv,
            scale=scale,
            dim=d,
            Q_TILE_SIZE=cfg["Q_TILE_SIZE"],
            K_TILE_SIZE=cfg["K_TILE_SIZE"],
            IS_CAUSAL=is_causal,
        )

        ctx.is_causal = is_causal
        ctx.scale = scale
        ctx.Q_TILE_SIZE = cfg["Q_TILE_SIZE"]
        ctx.K_TILE_SIZE = cfg["K_TILE_SIZE"]
        ctx.save_for_backward(L, Q, K, V, O_acc)

        return O_acc

    @staticmethod
    def backward(ctx, dO: torch.Tensor):
        # Retrieve saved tensors from forward
        L, Q, K, V, O = ctx.saved_tensors
        batch, n_q, d = Q.size()
        batch, n_kv, _ = K.size()
        scale = ctx.scale

        # Init result pointers
        dQ = torch.zeros_like(Q, dtype=torch.float32, device="cuda")
        dK = torch.zeros_like(K, dtype=torch.float32, device="cuda")
        dV = torch.zeros_like(V, dtype=torch.float32, device="cuda")

        # Pre-compute row sum for dO to simplify softmax grad
        # dO -> (batch, n_q, dim)
        D = (O * dO).sum(axis=-1)

        # Launch triton kernels - two outer loops
        flash_bwd_kernel_dq[(triton.cdiv(n_q, ctx.Q_TILE_SIZE), batch)](
            Q,
            K,
            V,
            L,
            D,  # (batch, n_q)
            dO,  # (batch, n_q, d)
            dQ,
            Q.stride(0),
            Q.stride(1),
            Q.stride(2),
            K.stride(0),
            K.stride(1),
            K.stride(2),
            V.stride(0),
            V.stride(1),
            V.stride(2),
            L.stride(0),
            L.stride(1),
            dO.stride(0),
            dO.stride(1),
            dO.stride(2),
            N_QUERIES=n_q,
            N_KEYS=n_kv,
            scale=scale,
            dim=d,
            Q_TILE_SIZE=ctx.Q_TILE_SIZE,
            K_TILE_SIZE=ctx.K_TILE_SIZE,
            IS_CAUSAL=ctx.is_causal,
        )

        flash_bwd_kernel_dkv[(triton.cdiv(n_kv, ctx.K_TILE_SIZE), batch)](
            Q,
            K,
            V,
            L,
            D,  # (batch, n_q)
            dO,  # (batch, n_q, d)
            dK,
            dV,
            Q.stride(0),
            Q.stride(1),
            Q.stride(2),
            K.stride(0),
            K.stride(1),
            K.stride(2),
            V.stride(0),
            V.stride(1),
            V.stride(2),
            L.stride(0),
            L.stride(1),
            dO.stride(0),
            dO.stride(1),
            dO.stride(2),
            N_QUERIES=n_q,
            N_KEYS=n_kv,
            scale=scale,
            dim=d,
            Q_TILE_SIZE=ctx.Q_TILE_SIZE,
            K_TILE_SIZE=ctx.K_TILE_SIZE,
            IS_CAUSAL=ctx.is_causal,
        )

        return dQ, dK, dV, None, None
