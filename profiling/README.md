# Profiling

The profiler supports four targets: `lm`, `attention`, `rmsnorm`, and `ffn`.
Each target defines its module, representative inputs, and backward objective in
`cases.py`; timing, NVTX annotation, and memory capture remain shared in `run.py`.

Run one target directly with Hydra overrides:

```bash
uv run python -m profiling.run \
    case=attention \
    case.batch_size=8 \
    case.seq_len=1024 \
    case.d_model=2048 \
    case.num_heads=16
```

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
`configs/profiling.yaml` contains only execution settings such as AMP, timing,
NVTX, memory capture, and output paths.

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
`profiling.output_dir=profiles/attention_b8_d64_s4096_h1_compile_reduce_overhead_fp32_full_step`
writes:

```text
profiles/attention_b8_d64_s4096_h1_compile_reduce_overhead_fp32_full_step/
├── attention_b8_d64_s4096_h1_compile_reduce_overhead_fp32_full_step.pkl
└── attention_b8_d64_s4096_h1_compile_reduce_overhead_fp32_full_step.nsys-rep
```

The `.nsys-rep` file is produced only when `USE_NSYS=1`. Nsight Systems adds
the extension to the output prefix automatically.

Only load snapshots from trusted sources. PyTorch memory snapshots are pickle
files and can execute arbitrary code when loaded.
