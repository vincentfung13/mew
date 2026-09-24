import triton
import triton.language as tl


@triton.jit
def flash_fwd_kernel(
    # pointers to:
    # Q -> (batch, n_q, d)
    # K -> (batch, n_kv, d)
    # V -> (batch, n_kv, d)
    Q_ptr,
    K_ptr,
    V_ptr,
    # O -> (batch, n_q, d)
    # L -> (batch, n_q)
    O_ptr,
    L_ptr,
    stride_qb,
    stride_qq,
    stride_qd,
    stride_kb,
    stride_kq,
    stride_kd,
    stride_vb,
    stride_vq,
    stride_vd,
    stride_ob,
    stride_oq,
    stride_od,
    stride_lb,
    stride_lq,
    N_QUERIES,
    N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
):
    query_tile_ind = tl.program_id(0)
    batch_ind = tl.program_id(1)

    # Init input blk ptrs
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_ind * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_ind * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_ind * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kq, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_ind * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vq, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    # Init output buffer ptrs
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_ind * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_ind * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_ind * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_ind * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    # Init running max/log_exp_sum/output/log_exp_sum tensor
    M = tl.full((Q_TILE_SIZE,), float("-inf"), dtype=tl.float32)
    O_acc = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    L = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)

    # Load q tile
    Q = tl.load(Q_block_ptr, boundary_check=(0,), padding_option="zero")

    # Load K, V tile by tile
    for j in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        # Load the i_th tile (no need for boundary check on the 1st dim)
        K_j = tl.load(
            K_block_ptr, boundary_check=(0,), padding_option="zero"
        )  # (K_TILE_SIZE, D)
        V_j = tl.load(
            V_block_ptr, boundary_check=(0,), padding_option="zero"
        )  # (K_TILE_SIZE, D)

        # Compute dot product
        S_ij = tl.dot(Q, tl.trans(K_j)) / scale  # (Q_TILE_SIZE, K_TILE_SIZE)

        # Compute local max, calibrate prev results
        _M = tl.maximum(M, tl.max(S_ij, axis=1))

        # compute and aggregate exp sum (for backward)
        P_ij = tl.exp(S_ij - _M[:, None])  # (Q_TILE_SIZE, K_TILE_SIZE)
        M_calibration = tl.exp(M - _M)
        L = M_calibration * L + tl.sum(P_ij, axis=1)

        # compute, aggregate, calibrate output
        O_acc = M_calibration[:, None] * O_acc + tl.dot(P_ij, V_j)

        # assign new max
        M = _M

        # Advance kv pointer
        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

    # Softmax normalization
    O_acc /= L[:, None]

    # Store log sum for backward
    L = M + tl.log(L)

    tl.store(O_block_ptr, O_acc)
    tl.store(L_block_ptr, L)
