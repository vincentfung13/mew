# Profiling

This toolkit has two entry points that answer different questions:

| Entry point | Question | How |
|---|---|---|
| `profile_module.py` | *Where* do time and memory go inside an `nn.Module`? | Per-stage timing (forward, backward, optimizer), NVTX ranges for Nsight Systems, CUDA memory snapshots, optional `torch.compile`. |
| `bench_function.py` | *How fast* is a function, compared across implementations and shapes? | `triton.testing.do_bench` with CUDA events and L2 flushing, swept with `triton.testing.perf_report` into a CSV and a plot. |

Benchmarking finds what is slow. Profiling explains why. A typical loop is to spot
a regression or gap with `bench_function`, then reproduce that configuration with
`profile_module` (or Nsight Compute) to diagnose it.

## Profile a module

`profile_module` supports four targets: `lm`, `attention`, `rmsnorm`, and `ffn`.
Each target defines its module, representative inputs, and backward objective in
`cases.py`. Timing, NVTX annotation, and memory capture are shared in
`profile_module.py`.

Run one target directly with Hydra overrides:

```bash
uv run python -m profiling.profile_module \
    case=attention \
    profiling.protocol=full_training_step \
    case.batch_size=8 \
    case.seq_len=1024 \
    case.d_model=2048 \
    case.num_heads=16
```

Three execution protocols are available through `profiling.protocol`:

- `forward_only`: runs `exec_steps` forwards under `torch.no_grad()`.
- `full_training_step`: runs `exec_steps` iterations of gradient reset, forward,
  loss, backward, and optimizer update.
- `repeat_backward_on_same_graph`: resets gradients once, runs `exec_steps`
  forwards, and then backpropagates through the final forward graph
  `exec_steps` times without updating parameters. The graph is retained between
  backward calls and released after the final call.

Warmup follows the same ordering as the selected protocol. Timings include each
individual operation. The repeated-backward protocol additionally reports
aggregate `forward_phase` and `backward_phase` timings.

Run a focused sweep without Nsight Systems capture:

```bash
USE_NSYS=0 ./profiling/examples/run_lm_sweep.sh
USE_NSYS=0 ./profiling/examples/run_attention_sweep.sh
```

Enable `torch.compile` for either sweep with environment variables:

```bash
TORCH_COMPILE=1 \
TORCH_COMPILE_MODE=reduce-overhead \
USE_NSYS=0 \
./profiling/examples/run_attention_sweep.sh
```

`TORCH_COMPILE` defaults to `0`. `TORCH_COMPILE_MODE` defaults to `default` and
also accepts `reduce-overhead`, `max-autotune`, and
`max-autotune-no-cudagraphs`. With at least one warmup step, initial compilation
occurs during warmup and is excluded from measured iterations. Per-module NVTX
hooks are disabled for compiled runs because they can introduce graph breaks;
the outer stage ranges remain enabled.

The attention example sweeps `d_model` over `16`, `32`, `64`, and `128` and
`seq_len` over `256`, `1024`, `4096`, `8192`, and `16384`, always with one
attention head and batch size eight. The batch size is included in every output
directory and artifact name. The longest sequences may require substantial GPU
memory because attention memory grows quadratically with sequence length.

Each target owns its parameters in `configs/case/<target>.yaml`. The shared
`configs/profile_module.yaml` contains only execution settings such as AMP,
timing, NVTX, memory capture, and output paths.

## Benchmark a function

`bench_function` times a single function across interchangeable *providers*.
The `attention` function compares:

- `flash_triton`: the Triton FlashAttention kernels in `mew/nn/flash_attention/`.
- `reference`: the eager `mew.nn.functionals.scaled_dot_product`.
- `torch_sdpa`: `torch.nn.functional.scaled_dot_product_attention`, as an
  external baseline.

All three accept `(batch, heads, seq_len, d_head)` inputs. `flash_triton` and
`torch_sdpa` also support GQA/MQA through `function.num_kv_heads`; `reference`
is MHA-only, so GQA runs drop it and check against `torch_sdpa` instead:

```bash
uv run python -m profiling.bench_function \
    bench.mode=fwd_bwd \
    bench.metric=tflops \
    bench.sweep.x_name=seq_len \
    'bench.sweep.x_vals=[512,1024,2048,4096]' \
    'bench.providers=[flash_triton,torch_sdpa]' \
    bench.reference_provider=torch_sdpa \
    function.num_heads=16 \
    function.num_kv_heads=4 \
    bench.output_dir=benchmarks/attention_gqa_fwd_bwd
```

Three modes are available through `bench.mode`:

- `fwd`: times the forward pass. Inputs do not require grad, so no graph is built.
- `bwd`: runs the forward once outside the timed region, then times
  `output.backward(...)` on the retained graph.
- `fwd_bwd`: times forward and backward together.

In the backward modes, `do_bench` resets input gradients between repetitions
(`grad_to_none`), so timings exclude gradient accumulation.

`bench.metric` selects the y-axis: `ms` (median, with a 20th–80th percentile
band) or `tflops`. Throughput is computed from the forward FLOPs
(`4·B·H·N²·D`, halved for causal), with backward counted as 2.5× forward. Use
`tflops` to compare across shapes and against the GPU's peak.

`bench.sweep.x_name` may be any parameter of the function config, such as
`seq_len`, `d_head`, or `batch_size`; the other parameters stay fixed at their
configured values. Each run writes `<function>_<mode>_<dtype>_<metric>.csv` and
`.png` to `bench.output_dir` and prints the table.

Before timing, each provider's forward output is compared against
`bench.reference_provider` with the function's `atol`/`rtol`, and the run fails
if they disagree. Set `bench.check_correctness=false` to skip this.

Resource failures only invalidate the affected point:

- If the reference runs out of CUDA memory (or a Triton kernel exceeds the GPU's
  shared memory or registers), the correctness check for that point is skipped
  with a warning, and the provider is still timed.
- If the provider itself hits either error, during the check or during timing,
  that `(provider, x)` point is recorded as `NaN` with a warning naming the
  provider, and the sweep continues.

Any other error (a compile error, a failed correctness check) stops the run.

Notes:

- `dtype` (`fp32`, `fp16`, `bf16`) sets the input dtype directly; no autocast is
  applied. `tl.dot` uses TF32 for fp32 inputs by default, so compare against an
  fp32 PyTorch baseline with that in mind.
- `do_bench` budgets (`bench.warmup_ms`, `bench.rep_ms`) are measured in
  milliseconds of runtime, not iterations. One extra untimed call runs before
  each measurement so compilation and autotuning do not distort it.
- The eager `reference` materializes the full `N × N` score matrix, so it is the
  first provider to run out of memory at long sequence lengths. Drop it from
  `bench.providers` and set `bench.reference_provider=torch_sdpa` to check large
  sizes against PyTorch instead.

Run the single-head attention sweep (batch size 1, one head, `d_head` 16–128,
causal and non-causal, `seq_len` from 128 to 65536, fp32 and bf16, all three
modes; 48 runs in total):

```bash
./profiling/examples/run_attention_bench_sweep.sh
METRIC=tflops ./profiling/examples/run_attention_bench_sweep.sh
```

Results go to `benchmarks/attention_b1_h1_d<D>_<causal|noncausal>_<dtype>_<mode>/`,
together with a `bench.log` of the run's output. A run that fails does not stop
the sweep: failed runs are listed at the end, and the script exits non-zero if
any run failed. The
sweep checks correctness against `torch_sdpa`, so long sequences are still
validated after the eager `reference` runs out of memory.

Each function owns its parameters in `configs/function/<name>.yaml`. The shared
`configs/bench_function.yaml` contains the mode, metric, providers, sweep axis,
`do_bench` budgets, and output path. New functions are registered in
`functions.py`.

## Analyze a memory snapshot

Use the repository's [`pytorch-memory-report`](../skills/pytorch-memory-report/)
skill to turn the `.pkl` file produced by memory profiling into an interactive
HTML report and interpret its main optimization opportunities. The skill also
documents attribution limitations and how to distinguish measured findings from
hypotheses.

To render the report directly, run its bundled script from the repository root:

```bash
uv run skills/pytorch-memory-report/scripts/render_memory_report.py \
    profiles/<run-name>/<run-name>.pkl \
    --output profiles/<run-name>/memory-report.html
```

Memory snapshots and Nsight reports are named after their output directory. The
directory includes `eager` or `compile_<mode>` so compiled and eager artifacts
cannot overwrite one another. For example,
`profiling.output_dir=profiles/attention_b8_d64_s4096_h1_compile_reduce_overhead_fp32_full_training_step`
writes:

```text
profiles/attention_b8_d64_s4096_h1_compile_reduce_overhead_fp32_full_training_step/
├── attention_b8_d64_s4096_h1_compile_reduce_overhead_fp32_full_training_step.pkl
└── attention_b8_d64_s4096_h1_compile_reduce_overhead_fp32_full_training_step.nsys-rep
```

The `.nsys-rep` file is produced only when `USE_NSYS=1`. Nsight Systems adds
the extension to the output prefix automatically.

Only load snapshots from trusted sources. PyTorch memory snapshots are pickle
files and can execute arbitrary code when loaded.
