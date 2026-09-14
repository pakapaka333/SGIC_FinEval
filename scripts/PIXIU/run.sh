#!/bin/bash
# This script assumes the following environment variables are set:
# MODEL_NAME, MODEL_TYPE, OUTPUT_DIR, PROJECT_ROOT, JOB_ID
set -euo pipefail


# Set environment variables
cd $PROJECT_ROOT
set -a; source .env; set +a;
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="${PROJECT_ROOT}/benchmarks/PIXIU/src/metrics/BARTScore:${PYTHONPATH:-}"
export VLLM_WORKER_MULTIPROC_METHOD="spawn"


# Run evaluation
OPTIONAL_ARGS=()
if [ "$MODEL_TYPE" == "base" ]; then
    echo "Using base model type"
    TASKS="flare_fpb,flare_cfa,flare_finqa"
    EXTRA_MODEL_ARGS=""
    OPTIONAL_ARGS+=(--num_fewshot 4)
    OPTIONAL_ARGS+=(--stop_words $'\n\n')

elif [ "$MODEL_TYPE" == "inst" ]; then
    echo "Using inst model type"
    TASKS="flare_fpb_inst,flare_cfa_inst,flare_finqa_inst"
    EXTRA_MODEL_ARGS=""
    OPTIONAL_ARGS+=(--apply_chat_template)

elif [ "$MODEL_TYPE" == "reasoning" ]; then
    echo "Using reasoning model type"
    TASKS="flare_fpb_inst,flare_cfa_inst,flare_finqa_inst"
    EXTRA_MODEL_ARGS=",max_gen_toks=32768"
    OPTIONAL_ARGS+=(--apply_chat_template)

else
    echo "Invalid model type: $MODEL_TYPE" >&2
    exit 1
fi
BATCH_SIZE=4096


OUTPUT_PATH="${OUTPUT_DIR}/results_${JOB_ID}.json"

cd ${PROJECT_ROOT}/benchmarks/PIXIU

uv run --isolated --locked --project "${PROJECT_ROOT}/scripts/PIXIU/pyproject.toml" \
    python "${PROJECT_ROOT}/benchmarks/PIXIU/src/eval.py" \
        --model "hf-causal-vllm" \
        --model_args "pretrained=${MODEL_NAME},dtype=bfloat16${EXTRA_MODEL_ARGS}" \
        --tasks "${TASKS}" \
        --write_out \
        --device cuda \
        --batch_size "${BATCH_SIZE}" \
        --output_path "${OUTPUT_PATH}" \
        --output_base_path "${OUTPUT_DIR}/write_out_info_${JOB_ID}" \
        --no_cache \
        "${OPTIONAL_ARGS[@]}"

cd ${PROJECT_ROOT}

# Aggregate results
uv run --isolated --locked --project "${PROJECT_ROOT}/scripts/aggregate_results/pyproject.toml" \
    python "${PROJECT_ROOT}/scripts/aggregate_results/aggregate_results.py" \
        --config_path "${PROJECT_ROOT}/scripts/aggregate_results/aggregate_config.jsonl" \
        --results_dir "${OUTPUT_DIR}/.."