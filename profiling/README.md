# Profiling

This toolkit has two entry points that answer different questions:

| Entry point | Question | How |
|---|---|---|
| `profile_module.py` | *Where* do time and memory go in a training step of the model? | Runs the trainer's own `TrainStep` on a synthetic batch: per-stage timing (forward, backward, optimizer step), achieved TFLOP/s and MFU, peak memory, NVTX ranges for Nsight Systems, CUDA memory snapshots. |
| `bench_function.py` | *How fast* is a function, compared across implementations and shapes? | `triton.testing.do_bench` with CUDA events and L2 flushing, swept with `triton.testing.perf_report` into a CSV and a plot. |

Benchmarking finds what is slow. Profiling explains why. A typical loop is to spot
a regression or gap with `bench_function`, then reproduce that configuration with
`profile_module` (or Nsight Compute) to diagnose it. `profile_module` is
model-level only; layer-level comparisons belong in `bench_function`.

## Profile the training step

`profile_module` profiles exactly what the trainer runs. Its config composes
`apps/cfgs/training.yaml` (through `hydra.searchpath`), so the model, data shape,
AMP setting, optimizer and GPU peak table are the training ones, and the step is
`mew.trainers.npt_trainer.TrainStep`, the same object `NPTTrainer` uses. Each
step runs forward, backward and, on gradient-accumulation boundaries, the
optimizer and LR-scheduler step, exactly as in training. The input is a fixed
synthetic batch already on the GPU, so there is no data loading, validation,
logging or checkpointing in the timed region.

Run from the repository root (the search path is relative to the working
directory) and override the training keys directly:

```bash
uv run python -m profiling.profile_module \
    model.attn_impl=flash_triton \
    model.num_kv_heads=4 \
    data.batch_size=64 \
    trainer.amp.enable=true
```

Only the profiling settings live under `profiling.*`:

- `warmup_steps`, `exec_steps`: untimed warmup steps, then timed steps.
- `nvtx.annotate_modules`: per-module NVTX ranges; `nvtx.use_cudart_range`
  limits Nsight capture to the timed steps.
- `memory_profiling.enable`: records allocation history over the timed steps and
  dumps a `.pkl` snapshot.
- `output_dir`: where the log, `metrics.json`, the snapshot and Nsight reports go.

### Output

The run logs a table with one row per stage and writes the same numbers to
`<output_dir>/metrics.json`, together with peak memory and the configuration:

```text
stage             count    mean (s)    var (s^2)     p95 (s)     p99 (s)      tokens/s   TFLOP/s      MFU
forward              10    ...
backward             10    ...
optimizer_step       10    ...                                                         -         -        -
total                10    ...
peak_mem_allocated_gib: ...
peak_mem_reserved_gib: ...
```

- **TFLOP/s and MFU** count model FLOPs from `mew.perf.utils.module_flops_per_token`:
  forward ×1, backward ×2, total ×3 per token. The optimizer step does no model
  FLOPs, so it has none. The peak comes from `trainer.perf.peak_tflops` if set,
  otherwise from the GPU table in `apps/cfgs/gpu_specs.yaml` for the AMP dtype.
  On CPU there is no peak, so MFU is omitted.
- **tokens/s** is reported for whole steps only (`total`).
- **Peak memory** covers the timed steps only (warmup excluded).

The `total` MFU uses the same FLOP count and peak as the trainer's `perf/mfu`,
so it is the compute-only upper bound for training at that configuration. A
large gap between the two is overhead in the training loop (data loading,
logging, validation), not model compute.

Every stage boundary synchronizes the GPU so stages can be timed separately.
Per-module NVTX hooks and memory-history recording add CPU overhead to every
step, and the run logs a warning when either is on. For clean MFU numbers,
disable both:

```bash
uv run python -m profiling.profile_module \
    profiling.nvtx.annotate_modules=false \
    profiling.memory_profiling.enable=false
```

### Sweeps and Nsight Systems

`examples/run_lm_sweep.sh` profiles a list of model sizes, each with and without
bf16 AMP, and captures each run with Nsight Systems by default:

```bash
./profiling/examples/run_lm_sweep.sh
USE_NSYS=0 ./profiling/examples/run_lm_sweep.sh
```

Each run writes to `profiles/lm_<name>_b<batch>_s<seq>_<amp_bf16|fp32>/`.
Distributed (DDP/FSDP) profiling is not supported yet; the profiler raises if
launched with more than one process.

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

Memory snapshots and Nsight reports are named after their output directory. For
example, `profiling.output_dir=profiles/lm_xl_b4_s256_amp_bf16` writes:

```text
profiles/lm_xl_b4_s256_amp_bf16/
├── metrics.json
├── profile_module.log
├── lm_xl_b4_s256_amp_bf16.pkl
└── lm_xl_b4_s256_amp_bf16.nsys-rep
```

The `.nsys-rep` file is produced only when `USE_NSYS=1`. Nsight Systems adds
the extension to the output prefix automatically.

Only load snapshots from trusted sources. PyTorch memory snapshots are pickle
files and can execute arbitrary code when loaded.
