import math
import torch


class FlashAttention(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(
        ctx,
        Q: torch.Tensor,  # (batch, n_q_heads, n_q, d_head)
        K: torch.Tensor,  # (batch, n_kv_heads, n_kv, d_head)
        V: torch.Tensor,  # (batch, n_kv_heads, n_kv, d_head)
        is_causal: bool = False,
    ):
        batch, n_q_heads, n_q, d = Q.size()
        batch, n_kv_heads, n_kv, _ = K.size()
        # On GPU, exp(x) is computed as exp2(x * log2e)
        # exp(x) = e^x = (2^(log₂ e))^x = 2^(x · log₂ e) = exp2(x · log₂ e)
        # so we fold log2e into the scale and use tl.math.exp2 in the kernel
        qk_scale = (1.0 / math.sqrt(d)) * math.log2(math.e)
        if is_causal:
            assert n_q == n_kv, "FlashAttention only support n_q == n_kv currently"
        assert torch.cuda.is_available()

        # Init output buffer:
        # O should match the dtype of Q (avoid fp16 <-> fp32 mismatch);
        # L always needs to be fp32, because backward uses it as an exponent offset in exp(S - L)
        # where a rounding error in L turns into a relative error in every recomputed probability.
        O = torch.empty_like(Q, device="cuda")
        L = torch.empty((batch, n_q_heads, n_q), dtype=torch.float32, device="cuda")

        # Launch triton kernel (launch grid is [n_queries, batch * n_q_heads])
        import triton
        from mew.nn.flash_attention.kernels.forward import flash_fwd_kernel

        flash_fwd_kernel[
            lambda META: (triton.cdiv(n_q, META["Q_TILE_SIZE"]), batch * n_q_heads)
        ](
            Q,
            K,
            V,
            O,
            L,
            Q.stride(0),
            Q.stride(1),
            Q.stride(2),
            Q.stride(3),
            K.stride(0),
            K.stride(1),
            K.stride(2),
            K.stride(3),
            V.stride(0),
            V.stride(1),
            V.stride(2),
            V.stride(3),
            O.stride(0),
            O.stride(1),
            O.stride(2),
            O.stride(3),
            L.stride(0),
            L.stride(1),
            L.stride(2),
            N_QUERIES=n_q,
            N_KEYS=n_kv,
            N_Q_HEADS=n_q_heads,
            N_KV_HEADS=n_kv_heads,
            qk_scale=qk_scale,
            dim=d,
            IS_CAUSAL=is_causal,
        )

        ctx.is_causal = is_causal
        ctx.qk_scale = qk_scale
        ctx.sm_scale = 1.0 / math.sqrt(d)
        ctx.save_for_backward(L, Q, K, V, O)

        return O

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, dO: torch.Tensor):
        # Retrieve saved tensors from forward
        L, Q, K, V, O = ctx.saved_tensors
        batch, n_q_heads, n_q, d = Q.size()
        batch, n_kv_heads, n_kv, _ = K.size()
        qk_scale = ctx.qk_scale
        sm_scale = ctx.sm_scale

        # Init result pointers
        dQ = torch.empty_like(Q, device="cuda")
        dK = torch.empty_like(K, device="cuda")
        dV = torch.empty_like(V, device="cuda")

        # Pre-compute row sum for dO to simplify softmax grad
        # dO -> (batch, n_q_heads, n_q, dim)
        D = (O.float() * dO).sum(axis=-1)

        # Launch triton kernels - two outer loops
        import triton
        from mew.nn.flash_attention.kernels.backward import (
            flash_bwd_kernel_dkv,
            flash_bwd_kernel_dq,
        )

        flash_bwd_kernel_dq[
            lambda META: (triton.cdiv(n_q, META["Q_TILE_SIZE"]), batch * n_q_heads)
        ](
            Q,
            K,
            V,
            L,  # (batch, n_q_heads, n_q)
            D,  # (batch, n_q_heads, n_q)
            dO,  # (batch, n_q_heads, n_q, d)
            dQ,
            Q.stride(0),
            Q.stride(1),
            Q.stride(2),
            Q.stride(3),
            K.stride(0),
            K.stride(1),
            K.stride(2),
            K.stride(3),
            V.stride(0),
            V.stride(1),
            V.stride(2),
            V.stride(3),
            L.stride(0),
            L.stride(1),
            L.stride(2),
            D.stride(0),
            D.stride(1),
            D.stride(2),
            dO.stride(0),
            dO.stride(1),
            dO.stride(2),
            dO.stride(3),
            N_QUERIES=n_q,
            N_KEYS=n_kv,
            N_Q_HEADS=n_q_heads,
            N_KV_HEADS=n_kv_heads,
            qk_scale=qk_scale,
            sm_scale=sm_scale,
            dim=d,
            IS_CAUSAL=ctx.is_causal,
        )

        flash_bwd_kernel_dkv[
            lambda META: (triton.cdiv(n_kv, META["K_TILE_SIZE"]), batch * n_kv_heads)
        ](
            Q,
            K,
            V,
            L,
            D,  # (batch, n_q_heads, n_q)
            dO,  # (batch, n_q_heads, n_q, d)
            dK,
            dV,
            Q.stride(0),
            Q.stride(1),
            Q.stride(2),
            Q.stride(3),
            K.stride(0),
            K.stride(1),
            K.stride(2),
            K.stride(3),
            V.stride(0),
            V.stride(1),
            V.stride(2),
            V.stride(3),
            L.stride(0),
            L.stride(1),
            L.stride(2),
            D.stride(0),
            D.stride(1),
            D.stride(2),
            dO.stride(0),
            dO.stride(1),
            dO.stride(2),
            dO.stride(3),
            N_QUERIES=n_q,
            N_KEYS=n_kv,
            N_Q_HEADS=n_q_heads,
            N_KV_HEADS=n_kv_heads,
            qk_scale=qk_scale,
            sm_scale=sm_scale,
            dim=d,
            IS_CAUSAL=ctx.is_causal,
        )

        return dQ, dK, dV, None
