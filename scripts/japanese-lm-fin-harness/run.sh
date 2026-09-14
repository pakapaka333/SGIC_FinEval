#!/bin/bash
# This script assumes the following environment variables are set:
# MODEL_NAME, MODEL_TYPE, OUTPUT_DIR, PROJECT_ROOT, JOB_ID
set -euo pipefail


# Set environment variables
cd $PROJECT_ROOT
set -a; source .env; set +a;
export PATH="$HOME/.local/bin:$PATH"


# Run evaluation
OPTIONAL_ARGS=()
if [ "$MODEL_TYPE" == "base" ]; then
    echo "Using base model type"
    TASKS="chabsa,cma_basics,cpa_audit,fp2,security_sales_1"
    EXTRA_MODEL_ARGS=""
    OPTIONAL_ARGS+=(--num_fewshot 4)
    OPTIONAL_ARGS+=(--gen_kwargs $'stop=\n\n')

elif [ "$MODEL_TYPE" == "inst" ]; then
    echo "Using inst model type"
    TASKS="chabsa,cma_basics,cpa_audit,fp2,security_sales_1"
    EXTRA_MODEL_ARGS=""
    OPTIONAL_ARGS+=(--apply_chat_template)

elif [ "$MODEL_TYPE" == "reasoning" ]; then
    echo "Using reasoning model type"
    TASKS="chabsa_gen,cma_basics_gen,cpa_audit_gen,fp2_gen,security_sales_1_gen"
    EXTRA_MODEL_ARGS=",max_gen_toks=32768"
    OPTIONAL_ARGS+=(--apply_chat_template)

else
    echo "Invalid model type: $MODEL_TYPE" >&2
    exit 1
fi
BATCH_SIZE=4096

OUTPUT_PATH="${OUTPUT_DIR}/results_${JOB_ID}.json"

cd ${PROJECT_ROOT}/benchmarks/japanese-lm-fin-harness

uv run --isolated --locked --project "${PROJECT_ROOT}/scripts/japanese-lm-fin-harness/pyproject.toml" \
    python "${PROJECT_ROOT}/benchmarks/japanese-lm-fin-harness/main.py" \
        --model "vllm" \
        --model_args "pretrained=${MODEL_NAME},dtype=bfloat16,trust_remote_code=True${EXTRA_MODEL_ARGS}" \
        --tasks "${TASKS}" \
        --write_out \
        --log_samples \
        --device cuda \
        --batch_size "${BATCH_SIZE}" \
        --output_path "${OUTPUT_PATH}" \
        --trust_remote_code \
        "${OPTIONAL_ARGS[@]}"

cd ${PROJECT_ROOT}


# Aggregate results
uv run --isolated --locked --project "${PROJECT_ROOT}/scripts/aggregate_results/pyproject.toml" \
    python "${PROJECT_ROOT}/scripts/aggregate_results/aggregate_results.py" \
        --config_path "${PROJECT_ROOT}/scripts/aggregate_results/aggregate_config.jsonl" \
        --results_dir "${OUTPUT_DIR}/.."