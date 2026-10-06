#!/bin/bash

# Profiles the training step of language models across sizes, with and without
# bf16 AMP. Every other setting (optimizer, attention implementation, ...) comes
# from apps/cfgs/training.yaml.
#
# Set USE_NSYS=0 to collect timing and memory data without Nsight Systems.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/profiling_common.sh"

LM_SPECS=(
    # name batch_size seq_len d_model d_ff num_heads num_layers vocab_size
    # "small 4 256 768 3072 12 12 10000"
    # "medium 4 256 1024 4096 16 24 10000"
    # "large 4 256 1280 5120 20 36 10000"
    "xl 4 256 2560 10240 20 36 10000"
)

AMP_ENABLES=(true false)

echo "Starting full-LM profiling sweep..."

for spec in "${LM_SPECS[@]}"; do
    read -r name batch_size seq_len d_model d_ff num_heads num_layers vocab_size <<< "$spec"
    run_all_precisions "${name}_b${batch_size}_s${seq_len}" \
        data.batch_size="$batch_size" data.seq_len="$seq_len" \
        model.d_model="$d_model" model.d_ff="$d_ff" \
        model.num_heads="$num_heads" model.num_transformer_layers="$num_layers" \
        model.vocab_size="$vocab_size"
done

echo "Full-LM profiling sweep completed."
