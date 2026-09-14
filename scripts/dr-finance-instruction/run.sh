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
elif [ "$MODEL_TYPE" == "inst" ]; then
    echo "Using inst model type"
    OPTIONAL_ARGS+=(--apply-chat-template)
elif [ "$MODEL_TYPE" == "reasoning" ]; then
    echo "Using reasoning model type"
    OPTIONAL_ARGS+=(--apply-chat-template)
    OPTIONAL_ARGS+=(--max-tokens 32768)
    OPTIONAL_ARGS+=(--stop-tokens "")
else
    echo "Invalid model type: $MODEL_TYPE" >&2
    exit 1
fi

RESULT_PATH="${OUTPUT_DIR}/results_${JOB_ID}.json"
JOB_OUTPUT_DIR="${OUTPUT_DIR}/outputs_${JOB_ID}"
mkdir -p "${JOB_OUTPUT_DIR}"

JUDGE_MODEL="gpt-4.1-2025-04-14"
NUM_GPUS=1

uv run --isolated --locked --project "${PROJECT_ROOT}/scripts/dr-finance-instruction/pyproject.toml" \
    python "${PROJECT_ROOT}/benchmarks/dr-finance-instruction/gen_model_answers.py" \
        --model-name "${MODEL_NAME}" \
        --num-gpus ${NUM_GPUS} \
        --output-path "${JOB_OUTPUT_DIR}/model_answers.jsonl" \
        "${OPTIONAL_ARGS[@]}"

uv run --isolated --locked --project "${PROJECT_ROOT}/scripts/dr-finance-instruction/pyproject.toml" \
    python "${PROJECT_ROOT}/benchmarks/dr-finance-instruction/gen_judgment.py" \
        --answer-path "${JOB_OUTPUT_DIR}/model_answers.jsonl" \
        --judge-model "${JUDGE_MODEL}" \
        --output-path "${JOB_OUTPUT_DIR}/judgment.jsonl"

uv run --isolated --locked --project "${PROJECT_ROOT}/scripts/dr-finance-instruction/pyproject.toml" \
    python "${PROJECT_ROOT}/benchmarks/dr-finance-instruction/gen_show_results.py" \
        --model-name "${MODEL_NAME}" \
        --judge-model "${JUDGE_MODEL}" \
        --judgment-path "${JOB_OUTPUT_DIR}/judgment.jsonl" \
        --output-path "${RESULT_PATH}"


# Aggregate results
uv run --isolated --locked --project "${PROJECT_ROOT}/scripts/aggregate_results/pyproject.toml" \
    python "${PROJECT_ROOT}/scripts/aggregate_results/aggregate_results.py" \
        --config_path "${PROJECT_ROOT}/scripts/aggregate_results/aggregate_config.jsonl" \
        --results_dir "${OUTPUT_DIR}/.."