#!/bin/bash

# Shared helpers for the profiling sweep examples. This file does not run a sweep.

USE_NSYS=${USE_NSYS:-1}
AMP_DTYPE=${AMP_DTYPE:-bf16}
OUTPUT_DIR=${OUTPUT_DIR:-profiles}
TORCH_COMPILE=${TORCH_COMPILE:-0}
TORCH_COMPILE_MODE=${TORCH_COMPILE_MODE:-default}

case "$TORCH_COMPILE" in
    1 | true)
        torch_compile_enable=true
        ;;
    0 | false)
        torch_compile_enable=false
        ;;
    *)
        echo "TORCH_COMPILE must be one of: 0, 1, false, true" >&2
        exit 2
        ;;
esac

case "$TORCH_COMPILE_MODE" in
    default | reduce-overhead | max-autotune | max-autotune-no-cudagraphs) ;;
    *)
        echo "Unsupported TORCH_COMPILE_MODE: $TORCH_COMPILE_MODE" >&2
        exit 2
        ;;
esac

mkdir -p "$OUTPUT_DIR"

run_profiling() {
    local target=$1
    local case_name=$2
    local amp_enable=$3
    local protocol=$4
    local amp_dtype=$5
    shift 5
    local case_overrides=("$@")

    local precision_tag="fp32"
    if [ "$amp_enable" = "true" ]; then
        precision_tag="amp_${amp_dtype}"
    fi

    local compile_tag="eager"
    if [ "$torch_compile_enable" = "true" ]; then
        compile_tag="compile_${TORCH_COMPILE_MODE//-/_}"
    fi

    local run_name="${target}_${case_name}_${compile_tag}_${precision_tag}_${protocol}"
    local run_output_dir="${OUTPUT_DIR}/${run_name}"
    # Nsight adds the .nsys-rep extension to this output prefix.
    local report_path="${run_output_dir}/${run_name}"

    mkdir -p "$run_output_dir"

    echo ""
    echo "=========================================="
    echo "Profiling: $run_name"
    echo "Output dir: $run_output_dir"
    echo "=========================================="

    if [ "$USE_NSYS" -eq 1 ]; then
        uv run nsys profile \
            --cuda-memory-usage=true \
            --capture-range=cudaProfilerApi \
            --capture-range-end=stop \
            --output="$report_path" \
            --force-overwrite=true \
            -- python -m profiling.run \
            case="$target" \
            "${case_overrides[@]}" \
            profiling.nvtx.annotate_modules=true \
            profiling.nvtx.use_cudart_range=true \
            profiling.output_dir="$run_output_dir" \
            profiling.memory_profiling.enable=true \
            profiling.protocol="$protocol" \
            torch_compile.enable="$torch_compile_enable" \
            torch_compile.mode="$TORCH_COMPILE_MODE" \
            amp.enable="$amp_enable" \
            amp.dtype="$amp_dtype"
    else
        uv run python -m profiling.run \
            case="$target" \
            "${case_overrides[@]}" \
            profiling.nvtx.annotate_modules=true \
            profiling.nvtx.use_cudart_range=false \
            profiling.output_dir="$run_output_dir" \
            profiling.memory_profiling.enable=true \
            profiling.protocol="$protocol" \
            torch_compile.enable="$torch_compile_enable" \
            torch_compile.mode="$TORCH_COMPILE_MODE" \
            amp.enable="$amp_enable" \
            amp.dtype="$amp_dtype"
    fi
}

run_all_modes() {
    local target=$1
    local case_name=$2
    shift 2
    local case_overrides=("$@")

    for amp_enable in "${AMP_ENABLES[@]}"; do
        for protocol in "${PROTOCOLS[@]}"; do
            run_profiling \
                "$target" \
                "$case_name" \
                "$amp_enable" \
                "$protocol" \
                "$AMP_DTYPE" \
                "${case_overrides[@]}"
        done
    done
}
