"""
Model-level parity tests for the attention backends selected by `attn_impl`
(naive | torch_sdpa | flash_triton).

test_attention.py covers the FlashAttention kernels in isolation. These tests
cover the integration: each backend, wired into a full TransformerLM, must
produce the same logits, loss and parameter gradients as an fp32 reference,
and the flash path must receive consistent dtypes under bf16 autocast.

The reference is TransformerLM(attn_impl="torch_sdpa") in fp32, pinned to the
MATH SDPA backend (a plain PyTorch implementation that supports GQA).
test_naive_matches_reference cross-checks that reference against the naive
path, so a bug in the torch_sdpa wiring cannot silently pass.
"""

import pytest
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from mew.nn.functionals import cross_entropy
from mew.nn.lm import TransformerLM

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)

NUM_HEADS = 4
BATCH_SIZE = 4
SEQ_LEN = 128
MODEL_KWARGS = dict(
    d_model=128,  # d_head = 32
    d_ff=344,
    num_heads=NUM_HEADS,
    vocab_size=512,
    context_len=SEQ_LEN,
    num_transformer_layers=2,
    rope_theta=10000.0,
)

PRECISIONS = ["fp32", "bf16_autocast"]

# Bounds on the relative L2 error ||actual - expected|| / ||expected||, applied
# per tensor (logits, loss, and each parameter gradient).
# - naive: same math as the reference, so only fp32 reduction-order noise.
# - fp32: the Triton kernels run tl.dot in TF32 (10-bit mantissa), so the fp32
#   flash path is close to, but not bit-exact with, the reference.
# - bf16_autocast: every matmul input is rounded to bf16 (8-bit mantissa).
NAIVE_REL_TOL = 1e-4
REL_TOL = {"fp32": 5e-3, "bf16_autocast": 5e-2}

GQA_KV_HEADS = [
    pytest.param(NUM_HEADS, id="mha"),
    pytest.param(2, id="gqa2"),
    pytest.param(1, id="mqa"),
]


def _make_batch(device: str, seed: int = 0):
    gen = torch.Generator().manual_seed(seed)
    shape = (BATCH_SIZE, SEQ_LEN)
    tokens = torch.randint(0, MODEL_KWARGS["vocab_size"], shape, generator=gen)
    targets = torch.randint(0, MODEL_KWARGS["vocab_size"], shape, generator=gen)
    return tokens.to(device), targets.to(device)


def _build_model(attn_impl: str, num_kv_heads: int, device: str, state_dict=None):
    torch.manual_seed(0)
    model = TransformerLM(
        **MODEL_KWARGS, num_kv_heads=num_kv_heads, attn_impl=attn_impl
    ).to(device)
    if state_dict is not None:
        model.load_state_dict(state_dict, strict=True)
    return model


def _forward_backward(model, tokens, targets, precision: str):
    """
    One training-style step without the optimizer: autocast covers only the
    forward pass (as in the trainer); the loss is computed on fp32 logits.
    """
    model.zero_grad(set_to_none=True)
    with torch.autocast(
        device_type=tokens.device.type,
        dtype=torch.bfloat16,
        enabled=precision == "bf16_autocast",
    ):
        logits = model(tokens)
    loss = cross_entropy(logits.float(), targets)
    loss.backward()
    grads = {
        name: p.grad.detach().float().clone() for name, p in model.named_parameters()
    }
    return logits.detach().float(), loss.detach().float(), grads


def _reference(num_kv_heads: int, device: str, tokens, targets):
    model = _build_model("torch_sdpa", num_kv_heads, device)
    with sdpa_kernel(SDPBackend.MATH):
        logits, loss, grads = _forward_backward(model, tokens, targets, "fp32")
    return model.state_dict(), (logits, loss, grads)


def _rel_err(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return ((actual - expected).norm() / expected.norm().clamp_min(1e-12)).item()


def _assert_matches(actual, expected, tol: float):
    """Compare (logits, loss, grads) tuples and report every offending tensor."""
    logits, loss, grads = actual
    ref_logits, ref_loss, ref_grads = expected
    assert grads.keys() == ref_grads.keys()

    errors = {"logits": _rel_err(logits, ref_logits), "loss": _rel_err(loss, ref_loss)}
    errors.update(
        {f"grad[{name}]": _rel_err(grads[name], ref_grads[name]) for name in grads}
    )
    failures = [f"{k}: rel err {v:.2e}" for k, v in errors.items() if not v <= tol]
    assert not failures, f"exceeds rel tol {tol:.0e}:\n  " + "\n  ".join(failures)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=requires_cuda)])
@pytest.mark.parametrize(
    "num_kv_heads",
    [
        pytest.param(NUM_HEADS, id="mha"),
        pytest.param(
            2,
            id="gqa2",
            marks=pytest.mark.xfail(
                raises=AssertionError,
                reason="naive scaled_dot_product does not support GQA yet; "
                "remove this marker once it does",
                strict=False,
            ),
        ),
    ],
)
def test_naive_matches_reference(device, num_kv_heads):
    tokens, targets = _make_batch(device)
    state_dict, expected = _reference(num_kv_heads, device, tokens, targets)

    model = _build_model("naive", num_kv_heads, device, state_dict)
    actual = _forward_backward(model, tokens, targets, "fp32")

    _assert_matches(actual, expected, NAIVE_REL_TOL)


@requires_cuda
@pytest.mark.parametrize("precision", PRECISIONS)
@pytest.mark.parametrize("num_kv_heads", GQA_KV_HEADS)
@pytest.mark.parametrize("attn_impl", ["torch_sdpa", "flash_triton"])
def test_attn_impl_matches_reference(attn_impl, num_kv_heads, precision):
    device = "cuda"
    tokens, targets = _make_batch(device)
    state_dict, expected = _reference(num_kv_heads, device, tokens, targets)

    model = _build_model(attn_impl, num_kv_heads, device, state_dict)
    actual = _forward_backward(model, tokens, targets, precision)

    _assert_matches(actual, expected, REL_TOL[precision])


@requires_cuda
@pytest.mark.parametrize("num_kv_heads", GQA_KV_HEADS)
def test_flash_dtypes_under_bf16_autocast(monkeypatch, num_kv_heads):
    """
    Autocast does not see custom autograd.Functions, and RoPE multiplies q/k by
    fp32 sin/cos buffers, so q/k can arrive as fp32 while v is bf16. Whatever
    the fix (an explicit cast in the layer, or torch.amp.custom_fwd), the
    forward kernel should receive bf16 Q, K, V and write a bf16 O.

    The check is made at the kernel launch rather than at FlashAttention.apply,
    because a custom_fwd(cast_inputs=...) cast happens inside apply.
    """
    import mew.nn.flash_attention.function as function_module
    import mew.nn.flash_attention.kernels.forward as forward_module

    real_kernel = forward_module.flash_fwd_kernel
    launches = []

    class RecordingKernel:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                q, k, v, o = args[:4]
                launches.append((q.dtype, k.dtype, v.dtype, o.dtype))
                return real_kernel[grid](*args, **kwargs)

            return launch

    # Patch the kernel module (lazy imports inside forward) and, if present,
    # the module-level name in function.py (eager import at the top).
    monkeypatch.setattr(forward_module, "flash_fwd_kernel", RecordingKernel())
    monkeypatch.setattr(
        function_module, "flash_fwd_kernel", RecordingKernel(), raising=False
    )

    device = "cuda"
    tokens, targets = _make_batch(device)
    model = _build_model("flash_triton", num_kv_heads, device)
    _forward_backward(model, tokens, targets, "bf16_autocast")

    assert len(launches) == MODEL_KWARGS["num_transformer_layers"]
    for q_dtype, k_dtype, v_dtype, o_dtype in launches:
        assert (q_dtype, k_dtype, v_dtype, o_dtype) == (torch.bfloat16,) * 4, (
            f"flash_fwd_kernel got Q={q_dtype}, K={k_dtype}, V={v_dtype}, "
            f"O={o_dtype}; expected all bfloat16 under autocast"
        )
