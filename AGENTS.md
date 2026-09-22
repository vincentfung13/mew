# Agent Instructions

This repository contains a custom implementation of a GPT-like language model developed from scratch.

## 0. Purpose: This Is a Learning Project

This repo exists so the user can learn how a GPT-like model works by implementing it themselves, by hand. This is the most important instruction in this file and overrides convenience.

- **Do not write or edit the core implementation for the user.** This includes model/layer code (`mew/nn/`), tokenization logic (`mew/tokenization/`), optimizers (`mew/optimizers/`), data loaders (`mew/data_loaders/`), generation logic (`mew/generators/`), and training loops (`mew/trainers/`).
- **Instead, give suggestions, hints, and explanations.** Point to the relevant concept, paper, algorithm, or a similar pattern already in the codebase, and let the user write the code. Ask Socratic questions if the user seems stuck, rather than supplying the answer outright.
- **Debugging is the exception.** If the user's own code has a bug, you may read the code, help diagnose the root cause, and explain the fix. Prefer explaining the bug and letting the user apply the fix; only write the fix directly if the user asks you to or it's a trivial one-line correction to code they already wrote.
- **Non-core work is fine to implement directly**, e.g. Hydra configs, scripts under `apps/`, profiling tools under `profiling/`, reusable agent skills under `skills/`, formatting/lint fixes, tests, documentation, or plumbing that isn't itself the learning exercise. When unsure whether something counts as "core," ask.

## 1. Background

The codebase provides the core building blocks to train and run inference on a neural probabilistic language model. It includes custom tokenization (BPE), data loading, neural network layers (Transformers, RoPE), optimizers (AdamW with learning rate scheduling), text generation, and training loops. It also includes a standalone toolkit for profiling full models and individual layers. The project allows users to understand, experiment with, and measure the fundamental components of modern generative AI models.

## 2. High-Level Design and Modules

The architecture is separated into the core library, application entry points, and supporting tooling:

- **`@mew/`** **(Core Library):**
  - `mew/data_loaders/`: Handles batching and loading data for training (e.g., `numpy_batch_loader`).
  - `mew/generators/`: Contains logic for autoregressive text generation (e.g., `conditional_generator`).
  - `mew/nn/`: Implements the neural network architecture, including Transformer blocks, linear layers, and rotary positional embeddings (RoPE).
  - `mew/optimizers/`: Provides optimization algorithms like AdamW and custom learning rate scheduling.
  - `mew/tokenization/`: Contains the custom Byte-Pair Encoding (BPE) tokenizer and text processing utilities.
  - `mew/trainers/`: Implements the training loops and utilities for training the language model (e.g., `NPTTrainer`).
- **`@apps/`** **(Application Layer):**
  - Contains high-level scripts to execute workflows using the `mew` library.
  - `apps/cfgs/`: Stores Hydra configurations for tokenization, training, and inference.
  - `apps/launch_training.py` & `apps/tokenization.py`: Entry points for launching model training and running the data tokenization pipelines.
- **`@profiling/`** **(Performance Toolkit):**
  - `profiling/run.py`: Hydra entry point for timing, CUDA memory snapshots, NVTX annotation, AMP, and optional `torch.compile`.
  - `profiling/cases.py`: Defines workloads for `lm`, `attention`, `rmsnorm`, and `ffn`.
  - `profiling/protocols.py`: Defines `forward_only`, `full_training_step`, and `repeat_backward_on_same_graph` execution semantics.
  - `profiling/configs/`: Keeps execution settings separate from per-case model and input parameters.
  - `profiling/examples/`: Provides LM and attention sweeps with optional Nsight Systems capture.
- **`@skills/`** **(Reusable Analysis Workflows):**
  - `skills/pytorch-memory-report/`: Renders and interprets trusted PyTorch CUDA memory snapshots as interactive HTML reports.

## 3. Package Management

- **Always use** **`uv`** for package management and running the code.
- Example: Use `uv run <script.py>` to execute code or `uv pip install <package>` for managing dependencies to ensure a fast, reliable, and reproducible Python environment.

## 4. Code Formatting and Linting

- **Always format the code with** **`black`.**
- **Check for lint errors with** **`flake8`**, but strictly ignore the "line too long" error (`E501`).
- **Scope:** Only apply `uvx black` formatting and `uvx flake8` linting to `@mew/`, `@apps/`, and `@tests/`. Do not run them on other directories or files in the repository.
- **Always run** styling and lint checks after making changes under `@mew/`, `@apps/`, or `@tests/`, e.g.:
  ```bash
  uvx black mew/ apps/ tests/
  uvx flake8 mew/ apps/ tests/
  ```

## 5. Testing

- **Always run the test suite** (`uv run pytest`) after making any code change, to make sure nothing regresses.
- Profiling cases and protocols have CPU-compatible tests under `tests/basics/test_profiling_cases.py` and `tests/basics/test_profiling_protocols.py`; CUDA and Nsight behavior still requires an appropriate GPU environment for end-to-end validation.

## 6. Profiling Conventions

- Run the profiler from the repository root with `uv run python -m profiling.run` and use Hydra overrides for the case and execution settings.
- Keep target-specific dimensions in `profiling/configs/case/<target>.yaml`; keep AMP, compilation, timing, NVTX, memory capture, protocol, and output settings in `profiling/configs/profiling.yaml`.
- Preserve the distinction among the three protocols: forward-only disables gradients, a full training step resets gradients and updates parameters on every iteration, and repeated backward reuses only the final forward graph without optimizer updates.
- Avoid attaching per-module NVTX hooks to `torch.compile` runs because those hooks can introduce graph breaks. Outer stage ranges should remain available.
- Profiling artifacts belong under the configured output directory. Do not commit generated `.pkl` memory snapshots or `.nsys-rep` files unless the user explicitly requests it.
- Treat PyTorch memory snapshots as untrusted pickle data unless their provenance is known. Never load or render a snapshot from an untrusted source.
- When interpreting memory reports, separate measured observations from hypotheses and do not infer a leak from allocation traffic alone.

