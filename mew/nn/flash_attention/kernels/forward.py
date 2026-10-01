import triton
import triton.language as tl


@triton.jit
def flash_fwd_kernel(
    # pointers to:
    # Q -> (batch, n_q_heads, n_q, d)
    # K -> (batch, n_kv_heads, n_kv, d)
    # V -> (batch, n_kv_heads, n_kv, d)
    Q_ptr,
    K_ptr,
    V_ptr,
    # O -> (batch, n_q, d)
    # L -> (batch, n_q)
    O_ptr,
    L_ptr,
    stride_qb,
    stride_qh,
    stride_qq,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_kq,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vq,
    stride_vd,
    stride_ob,
    stride_oh,
    stride_oq,
    stride_od,
    stride_lb,
    stride_lh,
    stride_lq,
    N_QUERIES,
    N_KEYS,
    N_Q_HEADS,
    N_KV_HEADS,
    scale,
    dim: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    query_tile_ind = tl.program_id(0)
    batch_head_ind = tl.program_id(1)

    # Compute inds for q
    batch_ind = batch_head_ind // N_Q_HEADS
    head_ind_q = batch_head_ind % N_Q_HEADS

    # Compute head ind for kv (for non-GQA, head_ind_q == head_ind_kv)
    head_ind_kv = head_ind_q // (N_Q_HEADS // N_KV_HEADS)

    # Init input blk ptrs
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_ind * stride_qb + head_ind_q * stride_qh,
        shape=(N_QUERIES, dim),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_ind * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, dim),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_ind * stride_kb + head_ind_kv * stride_kh,
        shape=(N_KEYS, dim),
        strides=(stride_kq, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, dim),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_ind * stride_vb + head_ind_kv * stride_vh,
        shape=(N_KEYS, dim),
        strides=(stride_vq, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, dim),
        order=(1, 0),
    )

    # Init output buffer ptrs
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_ind * stride_ob + head_ind_q * stride_oh,
        shape=(N_QUERIES, dim),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_ind * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, dim),
        order=(1, 0),
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_ind * stride_lb + head_ind_q * stride_lh,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_ind * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    # Init running max/log_exp_sum/output/log_exp_sum tensor
    M = tl.full((Q_TILE_SIZE,), float("-inf"), dtype=tl.float32)
    O_acc = tl.zeros((Q_TILE_SIZE, dim), dtype=tl.float32)
    L = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)

    # Load q tile
    Q = tl.load(Q_block_ptr, boundary_check=(0,), padding_option="zero")
    Q_offsets = query_tile_ind * Q_TILE_SIZE + tl.arange(
        0, Q_TILE_SIZE
    )  # (Q_TILE_SIZE,)
    Q_is_valid = Q_offsets[:, None] < N_QUERIES

    # Load K, V tile by tile
    for j in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        # Load the i_th tile (no need for boundary check on the 1st dim)
        K_j = tl.load(
            K_block_ptr, boundary_check=(0,), padding_option="zero"
        )  # (K_TILE_SIZE, dim)
        V_j = tl.load(
            V_block_ptr, boundary_check=(0,), padding_option="zero"
        )  # (K_TILE_SIZE, dim)
        K_offsets = j * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)  # (K_TILE_SIZE, )
        K_is_valid = K_offsets[None, :] < N_KEYS

        # To handle partial tile, each (q_ind, k_ind/v_ind) is valid
        QK_is_valid = Q_is_valid & K_is_valid

        # Compute dot product
        S_ij = tl.dot(Q, tl.trans(K_j)) / scale  # (Q_TILE_SIZE, K_TILE_SIZE)

        if IS_CAUSAL:
            # Upper triangular mask
            mask = (
                Q_offsets[:, None] >= K_offsets[None, :]
            )  # (Q_TILE_SIZE, K_TILE_SIZE)
            # Apply causal mask
            S_ij = tl.where(mask & QK_is_valid, S_ij, -1e6)
        else:
            S_ij = tl.where(QK_is_valid, S_ij, -1e6)

        # Compute local max, calibrate prev results
        _M = tl.maximum(M, tl.max(S_ij, axis=1))

        # Normalize by local max logits
        S_ij -= _M[:, None]  # (Q_TILE_SIZE, K_TILE_SIZE)

        # compute and aggregate exp sum (for backward)
        P_ij = tl.exp(S_ij)  # (Q_TILE_SIZE, K_TILE_SIZE)
        M_calibration = tl.exp(M - _M)
        L = M_calibration * L + tl.sum(P_ij, axis=1)

        # compute, aggregate, calibrate output
        O_acc = M_calibration[:, None] * O_acc + tl.dot(P_ij.to(V_j.dtype), V_j)

        # assign new max
        M = _M

        # Advance kv pointer
        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

    # Softmax normalization
    O_acc /= L[:, None]

    # Store log sum + row_max for backward
    # rationale of this logsum trick: when we recompute softmax in backward, we do:
    #   exp(S_ij - M_i - log(L))
    # whic automatically reduces to exp(S_ij - M_i) / L
    L = M + tl.log(L)

    tl.store(O_block_ptr, O_acc.to(O_ptr.dtype.element_ty), boundary_check=(0,))
    tl.store(L_block_ptr, L, boundary_check=(0,))
