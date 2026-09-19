#!/bin/bash

# Profiling example for single-head causal attention across widths and sequence lengths.
#
# The 16,384-token cases require substantial GPU memory because this attention
# implementation materializes tensors that scale quadratically with sequence length.
# Set USE_NSYS=0 to collect timing and memory data without Nsight Systems.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/profiling_common.sh"

D_MODELS=(16 32 64 128)
SEQ_LENS=(256 1024 4096 8192 16384)
NUM_HEADS=1
BATCH_SIZE=8

AMP_ENABLES=(true false)
FORWARD_ONLY_FLAGS=(true false)

echo "Starting self-attention profiling sweep..."

for d_model in "${D_MODELS[@]}"; do
    for seq_len in "${SEQ_LENS[@]}"; do
        case_name="b${BATCH_SIZE}_d${d_model}_s${seq_len}_h${NUM_HEADS}"
        run_all_modes attention "$case_name" \
            case.batch_size="$BATCH_SIZE" \
            case.seq_len="$seq_len" \
            case.d_model="$d_model" \
            case.num_heads="$NUM_HEADS" \
            case.num_groups=null
    done
done

echo "Self-attention profiling sweep completed."
