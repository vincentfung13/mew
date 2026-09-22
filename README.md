# mew 

<p align="center">
    <img src="resources/logo.jpg" width="30%" alt="mew logo">
</p>

A custom implementation of a GPT-like language model developed entirely from scratch. This project provides the fundamental building blocks for training and running inference on an autoregressive neural probabilistic language model.

## Features

- **Custom Tokenization**: Byte-Pair Encoding (BPE) implementation.
- **Efficient Data Loading**: Memory-optimized Numpy batch loaders capable of handling massive memmap files.
- **Neural Network Architecture**: Custom Transformer blocks, Rotary Positional Embeddings (RoPE), and linear layers built with PyTorch.
- **Optimizers**: Custom AdamW optimizer with learning rate scheduling.
- **Generators**: Autoregressive text generation logic.
- **Trainer**: Training loops with experiment tracking (W&B integration).
- **Configuration Management**: Hydra-based configuration for easy parameter sweeping and experiment management.
- **Performance Profiling**: Configurable timing, CUDA memory, NVTX/Nsight Systems, AMP, and `torch.compile` profiling for full models and individual layers.

## Project Structure

The repository is organized into a core library, an application layer, and supporting performance tooling:

### 1. `@mew/` (Core Library)
The core engine behind the language model:
- `mew/data_loaders/`: Efficient batching and data loading logic (`numpy_batch_loader`).
- `mew/generators/`: Text generation utilities (`conditional_generator`).
- `mew/nn/`: Neural network architectures, modules, and layers (Transformers, RoPE).
- `mew/optimizers/`: Custom optimizers and schedulers (AdamW, LR scheduling).
- `mew/tokenization/`: BPE tokenizer and text processing tools.
- `mew/trainers/`: Implementations of the training loops (e.g., `NPTTrainer`).

### 2. `@apps/` (Application Layer)
High-level scripts and configurations:
- `apps/cfgs/`: Hydra configuration files (`training.yaml`, `inference.yaml`, `tokenization.yaml`).
- `apps/launch_training.py`: Entry point for launching model training.
- `apps/tokenization.py`: Entry point for running the data tokenization pipelines.

### 3. `@profiling/` (Performance Toolkit)
Standalone profiling workloads and configurations:
- `profiling/run.py`: Hydra entry point for timing, CUDA memory snapshots, and NVTX-annotated runs.
- `profiling/cases.py`: Workload definitions for a full language model, attention, RMSNorm, and feed-forward layers.
- `profiling/protocols.py`: Forward-only, full-training-step, and repeated-backward execution protocols.
- `profiling/configs/`: Shared execution settings and per-target Hydra configs.
- `profiling/examples/`: Full-model and attention sweep scripts, optionally captured with Nsight Systems.

The repository also includes `skills/pytorch-memory-report/`, which renders an interactive HTML report from a trusted PyTorch CUDA memory snapshot.

## Setup and Installation

This project strictly uses **`uv`** for fast and reliable Python package management. 

1. Ensure you have `uv` installed.
2. Install the project and its dependencies:
   ```bash
   uv sync
   ```

By default this resolves packages against the public PyPI index. If you're on a network where only an internal mirror is reachable, copy `uv.toml.example` to `uv.toml` (gitignored) and fill in your mirror's URL — uv picks it up automatically, no other changes needed. Note that switching indexes and re-running `uv sync`/`uv lock` may change `uv.lock`, since the resolved package graph can differ between indexes.

## Usage

You can run the application scripts using `uv run`. 

**Tokenization:**
```bash
# 1. Train a BPE tokenizer
uv run apps/tokenization.py \
    task_name=train_bpe \
    training.vocab_size=10000 \
    training.input_path='./data/TinyStoriesV2-GPT4-train.txt' \
    training.save_dir='./data/tinystories_bpe_tokenizer'

# 2. Tokenize the training and vaidation file
uv run apps/tokenization.py \
    task_name=tokenize_file \
    file_tokenization.input_path='./data/TinyStoriesV2-GPT4-train.txt' \
    file_tokenization.tokenizer_path='./data/tinystories_bpe_tokenizer' \
    file_tokenization.save_path='./data/tiny_stories_train.tokens.uint16.npy' \
    file_tokenization.num_workers=12

uv run apps/tokenization.py \
    task_name=tokenize_file \
    file_tokenization.input_path='./data/TinyStoriesV2-GPT4-valid.txt' \
    file_tokenization.tokenizer_path='./data/tinystories_bpe_tokenizer' \
    file_tokenization.save_path='./data/tiny_stories_val.tokens.uint16.npy' \
    file_tokenization.num_workers=12
```

**Training:**
```bash
EXP_PREFIX="tinystories_ablation_$(date +%Y%m%d_%H%M%S)"
uv run apps/launch_training.py \
    run_name="${EXP_PREFIX}_mha" \
    trainer.total_steps=10000 \
    data.train_file='./data/tiny_stories_train.tokens.uint16.npy' \
    data.val_file='./data/tiny_stories_val.tokens.uint16.npy' \
    data.batch_size=128 \
    data.seq_len=256 \
    data.tokenizer_path='./data/tinystories_bpe_tokenizer' \
    model.d_model=512 \
    model.d_ff=1344 \
    model.num_heads=16 \
    model.num_groups=null
```
*Note: The launch scripts use Hydra, so you can override configurations via the CLI (e.g., `uv run apps/launch_training.py wandb.enable=True`).*

**Profiling:**

Run a single profiling case with Hydra overrides:

```bash
uv run python -m profiling.run \
    case=attention \
    profiling.protocol=full_training_step \
    case.batch_size=8 \
    case.seq_len=1024 \
    case.d_model=2048 \
    case.num_heads=16
```

Supported cases are `lm`, `attention`, `rmsnorm`, and `ffn`. Set `torch_compile.enable=true` to profile a compiled workload, or choose `forward_only`, `full_training_step`, or `repeat_backward_on_same_graph` with `profiling.protocol`. CUDA memory profiling is enabled by default and writes a `.pkl` snapshot under `profiling.output_dir`.

Run the example sweeps without Nsight Systems capture:

```bash
USE_NSYS=0 ./profiling/examples/run_lm_sweep.sh
USE_NSYS=0 ./profiling/examples/run_attention_sweep.sh
```

See [profiling/README.md](profiling/README.md) for Nsight Systems capture, AMP and `torch.compile` options, artifact naming, and memory-report instructions. Only open memory snapshots from trusted sources because they use Python's pickle format.

## Development Guidelines

- **Formatting**: Always format the code using `black`.
- **Linting**: Check for lint errors using `flake8`, but ignore the "line too long" error (`E501`).

```bash
uvx black mew/ apps/ tests/
uvx flake8 --ignore=E501 mew/ apps/ tests/
uv run pytest
```

See [AGENTS.md](AGENTS.md) for more details regarding instructions for AI agents and code contributors.
