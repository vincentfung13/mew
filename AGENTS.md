# Agent Instructions

This repository contains a custom implementation of a GPT-like language model developed from scratch.

## 1. Background
The codebase provides the core building blocks to train and run inference on a neural probabilistic language model. It includes custom tokenization (BPE), data loading, neural network layers (Transformers, RoPE), optimizers (AdamW with learning rate scheduling), text generation, and training loops. The project allows users to understand and experiment with the fundamental components of modern generative AI models.

## 2. High-Level Design and Modules
The architecture is cleanly separated into two main packages:

*   **`@mew/` (Core Library):**
    *   `mew/data_loaders/`: Handles batching and loading data for training (e.g., `numpy_batch_loader`).
    *   `mew/generators/`: Contains logic for autoregressive text generation (e.g., `conditional_generator`).
    *   `mew/nn/`: Implements the neural network architecture, including Transformer blocks, linear layers, and rotary positional embeddings (RoPE).
    *   `mew/optimizers/`: Provides optimization algorithms like AdamW and custom learning rate scheduling.
    *   `mew/tokenization/`: Contains the custom Byte-Pair Encoding (BPE) tokenizer and text processing utilities.
    *   `mew/trainers/`: Implements the training loops and utilities for training the language model (e.g., `NPTTrainer`).

*   **`@apps/` (Application Layer):**
    *   Contains high-level scripts to execute workflows using the `mew` library.
    *   `apps/cfgs/`: Stores Hydra configurations for tokenization, training, and inference.
    *   `apps/launch_training.py` & `apps/tokenization.py`: Entry points for launching model training and running the data tokenization pipelines.

## 3. Package Management
*   **Always use `uv`** for package management and running the code.
*   Example: Use `uv run <script.py>` to execute code or `uv pip install <package>` for managing dependencies to ensure a fast, reliable, and reproducible Python environment.

## 4. Code Formatting and Linting
*   **Always format the code with `black`.**
*   **Check for lint errors with `flake8`**, but strictly ignore the "line too long" error (`E501`).
*   **Scope:** Only apply `uvx black` formatting and `uvx flake8` linting to the core packages `@mew/` and `@apps/`. Do not run them on other directories or files in the repository.
