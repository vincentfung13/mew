import pytest
import torch
from einops import einsum

from .adapters import get_flashattention_autograd_function_triton

DTYPES = [torch.float32, torch.float16, torch.bfloat16]
DTYPE_IDS = ["fp32", "fp16", "bf16"]

# Tolerances against an fp32 reference computed from the same (rounded) inputs.
# Half precision rounds the tl.dot operands (P, dS, dO), and backward gradients
# sum over more terms than the forward output, so they get looser bounds.
FORWARD_TOL = {torch.float32: 1e-2, torch.float16: 1e-2, torch.bfloat16: 2e-2}
BACKWARD_TOL = {torch.float32: 1e-2, torch.float16: 2e-2, torch.bfloat16: 5e-2}


def _attention_and_lse(q, k, v, is_causal=False):
    """
    Reference attention. q is (B, H_q, N, D), k/v are (B, H_kv, N, D);
    output is (B, H_q, N, D) and L is (B, H_q, N).

    For GQA (H_kv < H_q), query head h attends to kv head h // (H_q // H_kv),
    matching the "(num_groups num_grouped_queries)" grouping in mew.
    """
    group_size = q.shape[1] // k.shape[1]
    k = k.repeat_interleave(group_size, dim=1)
    v = v.repeat_interleave(group_size, dim=1)
    n_queries = q.shape[-2]
    n_keys = k.shape[-2]
    d = q.shape[-1]
    scale = 1 / (d**0.5)
    S = einsum(q, k, "... q d, ... k d -> ... q k") * scale
    if is_causal:
        S = torch.where(
            torch.arange(n_queries, device=S.device)[:, None]
            >= torch.arange(n_keys, device=S.device)[None, :],
            S,
            -1e6,
        )
    P = torch.softmax(S, dim=-1)
    o = einsum(P, v, "... q k, ... k d -> ... q d")
    L = torch.logsumexp(S, dim=-1)
    return o, L


def _make_attn_inputs(
    device=None,
    non_contiguous=False,
    n_kv_heads=None,
    dtype=torch.float32,
    compute_dtype=None,
):
    """
    Returns (q, k, v, do, leaves). q is (B, H_q, N, D) and k/v are
    (B, H_kv, N, D), as fed to the kernel; leaves are the tensors that
    receive .grad. n_kv_heads defaults to H_q (plain multi-head attention).

    If non_contiguous, the leaves are allocated as (B, N, H, D) and q/k/v are
    their .transpose(1, 2) views, so the kernel sees non-contiguous strides.

    Values are rounded to ``dtype`` and stored as ``compute_dtype`` (defaults
    to ``dtype``). Calling with dtype=fp16, compute_dtype=fp32 yields the exact
    same values as the fp16 inputs, held in fp32 for an fp32 reference.
    """
    if compute_dtype is None:
        compute_dtype = dtype

    def rounded_randn(*shape):
        x = torch.randn(*shape, device=device).to(dtype)
        return x.to(compute_dtype)

    torch.random.manual_seed(0)
    batch_size = 2
    n_queries = 128
    n_keys = 128
    n_heads = 4
    if n_kv_heads is None:
        n_kv_heads = n_heads
    D = 64
    shape_q = (batch_size, n_heads, n_queries, D)
    shape_k = (batch_size, n_kv_heads, n_keys, D)

    def make(shape):
        if non_contiguous:
            b, h, n, d = shape
            leaf = rounded_randn(b, n, h, d).requires_grad_()
            return leaf, leaf.transpose(1, 2)
        leaf = rounded_randn(*shape).requires_grad_()
        return leaf, leaf

    q_leaf, q = make(shape_q)
    k_leaf, k = make(shape_k)
    v_leaf, v = make(shape_k)
    do = rounded_randn(*shape_q)

    if non_contiguous:
        assert not q.is_contiguous()
    return q, k, v, do, (q_leaf, k_leaf, v_leaf)


def _test_flash_forward_pass(
    impl,
    device="cpu",
    is_causal=False,
    non_contiguous=False,
    n_kv_heads=None,
    dtype=torch.float32,
):
    q, k, v, _do, _ = _make_attn_inputs(device, non_contiguous, n_kv_heads, dtype)
    o = impl(q, k, v, is_causal)
    assert o.shape == q.shape
    assert o.dtype == dtype, f"Output should match the input dtype {dtype}"

    # Extract L from the saved tensors
    assert (
        o.grad_fn.saved_tensors is not None
    ), "No saved tensors found in the output tensor. Make sure your autograd forward is saving them using ctx.save_for_backward."
    expected_l_shape = tuple(q.shape[:3])  # (B, H, N_q)
    maybe_ls = [t for t in o.grad_fn.saved_tensors if t.shape == expected_l_shape]

    assert (
        len(maybe_ls) == 1
    ), f"Expected one tensor of shape {expected_l_shape} in saved tensors, but found {len(maybe_ls)}. The tests require you to save exactly one tensor of this shape, corresponding to the log-sum-exp of the attention scores."
    lse = maybe_ls[0]
    assert lse.dtype == torch.float32, "The log-sum-exp should be kept in fp32"

    # fp32 reference on the same rounded input values.
    q_ref, k_ref, v_ref, _, _ = _make_attn_inputs(
        device, non_contiguous, n_kv_heads, dtype, compute_dtype=torch.float32
    )
    o_ref, lse_ref = _attention_and_lse(q_ref, k_ref, v_ref, is_causal)

    tol = FORWARD_TOL[dtype]
    torch.testing.assert_close(o.float(), o_ref, rtol=tol, atol=tol)
    torch.testing.assert_close(lse, lse_ref, rtol=tol, atol=tol)


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="A GPU must be available to run Triton kernels",
)
@pytest.mark.parametrize("is_causal", [False, True])
@pytest.mark.parametrize("non_contiguous", [False, True])
# n_kv_heads: 4 = MHA (H_q = 4), 2 = GQA (2 query heads per kv head), 1 = MQA
@pytest.mark.parametrize("n_kv_heads", [4, 2, 1])
@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
def test_flash_forward_pass_triton(is_causal, non_contiguous, n_kv_heads, dtype):
    _test_flash_forward_pass(
        get_flashattention_autograd_function_triton().apply,
        device="cuda",
        is_causal=is_causal,
        non_contiguous=non_contiguous,
        n_kv_heads=n_kv_heads,
        dtype=dtype,
    )


def flash_backward_results(
    impl,
    is_causal,
    device=None,
    non_contiguous=False,
    n_kv_heads=None,
    dtype=torch.float32,
    compute_dtype=None,
    sum_loss=False,
):
    """
    Run forward + backward and return the leaves' gradients.

    With sum_loss, backpropagate o.sum() instead of a random dO; autograd then
    passes an expanded dO whose strides are all zero.
    """
    q, k, v, do, leaves = _make_attn_inputs(
        device, non_contiguous, n_kv_heads, dtype, compute_dtype
    )
    o = impl(q, k, v, is_causal)
    if sum_loss:
        o.sum().backward()
    else:
        o.backward(do)
    return tuple(t.grad for t in leaves)


def _assert_grads_close(expected, actual, dtype):
    tol = BACKWARD_TOL[dtype]
    for name, grad_expected, grad in zip(("dq", "dk", "dv"), expected, actual):
        assert grad.dtype == dtype, f"{name} should match the input dtype {dtype}"
        torch.testing.assert_close(
            grad.float(),
            grad_expected,
            rtol=tol,
            atol=tol,
            msg=lambda msg, name=name: f"{name} mismatch:\n{msg}",
        )


def _reference_impl(*args):
    return _attention_and_lse(*args)[0]


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="A GPU must be available to run Triton kernels",
)
@pytest.mark.parametrize("is_causal", [False, True])
@pytest.mark.parametrize("non_contiguous", [False, True])
@pytest.mark.parametrize("n_kv_heads", [4, 2, 1])
@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
def test_flash_backward_triton(is_causal, non_contiguous, n_kv_heads, dtype):
    # fp32 reference on the same rounded input values.
    expected = flash_backward_results(
        _reference_impl,
        is_causal,
        device="cuda",
        non_contiguous=non_contiguous,
        n_kv_heads=n_kv_heads,
        dtype=dtype,
        compute_dtype=torch.float32,
    )
    actual = flash_backward_results(
        get_flashattention_autograd_function_triton().apply,
        is_causal,
        device="cuda",
        non_contiguous=non_contiguous,
        n_kv_heads=n_kv_heads,
        dtype=dtype,
    )

    _assert_grads_close(expected, actual, dtype)


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="A GPU must be available to run Triton kernels",
)
@pytest.mark.parametrize("is_causal", [False, True])
@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
def test_flash_backward_triton_zero_stride_grad_output(is_causal, dtype):
    # o.sum().backward() hands the kernel a dO with all-zero strides.
    expected = flash_backward_results(
        _reference_impl,
        is_causal,
        device="cuda",
        dtype=dtype,
        compute_dtype=torch.float32,
        sum_loss=True,
    )
    actual = flash_backward_results(
        get_flashattention_autograd_function_triton().apply,
        is_causal,
        device="cuda",
        dtype=dtype,
        sum_loss=True,
    )

    _assert_grads_close(expected, actual, dtype)
