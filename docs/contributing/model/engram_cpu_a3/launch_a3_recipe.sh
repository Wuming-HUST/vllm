#!/usr/bin/env bash
set -euo pipefail
# Source: vllm-ascend v0.30.0 tutorial, section 5.1 A3.
# Additions: backend selection, explicit Engram overlap and kernel selection.
# Recipe source SHA256: 075395eebba88416667f57c5127693f99b35baf78764a116dd957cbc7e3c5e2c
BACKEND="${1:-device}"
case "$BACKEND" in device|cpu) ;; *) echo "backend must be device or cpu" >&2; exit 2 ;; esac
: "${MODEL_PATH:?Set MODEL_PATH to the mounted W8A8 checkpoint}"
: "${ENGRAM_PYTHON:?Set ENGRAM_PYTHON to the deployed virtualenv python}"

CPU_KERNEL="${CPU_KERNEL:-auto}"
case "$CPU_KERNEL" in auto|scalar|neon|sve2) ;; *) echo "invalid CPU_KERNEL" >&2; exit 2 ;; esac
ENGRAM_CHUNK_TOKENS="${ENGRAM_CHUNK_TOKENS:-8192}"
ENGRAM_MAX_SEQS="${ENGRAM_MAX_SEQS:-32}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
for value in "$ENGRAM_CHUNK_TOKENS" "$ENGRAM_MAX_SEQS" "$OMP_NUM_THREADS"; do
    if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
        echo "chunk, sequence and thread limits must be positive integers" >&2
        exit 2
    fi
done
case "${ENGRAM_PREFIX_CACHE:-off}" in
    off) PREFIX_FLAG=--no-enable-prefix-caching ;;
    on) PREFIX_FLAG=--enable-prefix-caching ;;
    *) echo "ENGRAM_PREFIX_CACHE must be off or on" >&2; exit 2 ;;
esac
case "${ENGRAM_DRY_RUN:-0}" in 0|1) ;; *) echo "ENGRAM_DRY_RUN must be 0 or 1" >&2; exit 2 ;; esac

export VLLM_USE_V2_MODEL_RUNNER=0
export VLLM_ENGINE_READY_TIMEOUT_S=36000
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

export HCCL_BUFFSIZE=1024
export HCCL_OP_EXPANSION_MODE="AIV"

# Ensure the model path matches the directory recorded during download
command=( "$ENGRAM_PYTHON" -m vllm.entrypoints.openai.api_server --model "$MODEL_PATH" \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 150000 \
    --max-num-batched-tokens "$ENGRAM_CHUNK_TOKENS" \
    --served-model-name dsv41 \
    --gpu-memory-utilization 0.9 \
    --max-num-seqs "$ENGRAM_MAX_SEQS" \
    --data-parallel-size 4 \
    --tensor-parallel-size 4 \
    --enable-expert-parallel \
    --tokenizer-mode deepseek_v41 \
    --reasoning-parser deepseek_v41 \
    --tool-call-parser deepseek_v41 \
    --enable-auto-tool-choice \
    --safetensors-load-strategy lazy \
    --quantization ascend \
    --block-size 128 \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --async-scheduling \
    --enable-chunked-prefill \
    "$PREFIX_FLAG" \
    --engram-config "{\"cpu_offload\":true,\"dp_shared_memory\":true,\"lookup_backend\":\"$BACKEND\",\"cpu_kernel\":\"$CPU_KERNEL\",\"cpu_prefetch_distance\":0}" \
    --additional-config '{
        "ascend_compilation_config":{
            "enable_npugraph_ex":true,
            "enable_static_kernel":false
        },
        "enable_cpu_binding":true,
        "multistream_engram_overlap":true,
        "enable_fused_mc2":1,
        "enable_dsa_cp":true,
        "enable_flashcomm1":true,
        "enable_shared_expert_dp":true
    }' )

# This records requested settings, not the runtime-resolved worker configuration.
if [[ -n "${ENGRAM_RUN_DIR:-}" ]]; then
    mkdir -p "$ENGRAM_RUN_DIR"
    if ! mkdir "$ENGRAM_RUN_DIR/launch"; then
        echo "launch record already exists; use a new run directory" >&2
        exit 2
    fi
    printf '%q ' "${command[@]}" > "$ENGRAM_RUN_DIR/launch/command.sh"
    printf '\n' >> "$ENGRAM_RUN_DIR/launch/command.sh"
    printf 'OMP_NUM_THREADS=%s\nCPU_KERNEL=%s\nPREFIX_CACHE=%s\nCHUNK_TOKENS=%s\nMAX_SEQS=%s\nDRY_RUN=%s\n' \
        "$OMP_NUM_THREADS" "$CPU_KERNEL" "${ENGRAM_PREFIX_CACHE:-off}" \
        "$ENGRAM_CHUNK_TOKENS" "$ENGRAM_MAX_SEQS" "${ENGRAM_DRY_RUN:-0}" \
        > "$ENGRAM_RUN_DIR/launch/requested-settings.txt"
    cp "$0" "$ENGRAM_RUN_DIR/launch/launcher.sh"
fi
if [[ "${ENGRAM_DRY_RUN:-0}" == 1 ]]; then
    printf '%q ' "${command[@]}"
    printf '\n'
    exit 0
fi
exec "${command[@]}"
