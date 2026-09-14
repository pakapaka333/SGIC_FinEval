#!/usr/bin/env bash
# Run one model against multiple benchmarks sequentially on the local machine.
#
# Usage:
#   bash scripts/run_all.sh <model_name> <model_type> [benchmark ...]
#
#   model_name : HF model ID or local path (e.g. Qwen/Qwen3-14B)
#   model_type : base | inst | reasoning
#   benchmark  : optional list; defaults to all benchmarks.
set -uo pipefail

model_name=${1:-}
model_type=${2:-}
shift 2 2>/dev/null || true

if [[ -z "${model_name}" || -z "${model_type}" ]]; then
    echo "Usage: $0 <model_name> <model_type> [benchmark ...]" >&2
    echo "  model_type: base | inst | reasoning" >&2
    exit 1
fi

if [[ $# -gt 0 ]]; then
    benchmarks=("$@")
else
    benchmarks=(PIXIU japanese-lm-fin-harness dr-finance-instruction)
fi

SCRIPT_DIR="$(dirname "$(realpath "$0")")"

failed=()
for benchmark in "${benchmarks[@]}"; do
    echo "=== ${model_name} / ${model_type} / ${benchmark} ==="
    if ! bash "${SCRIPT_DIR}/evaluate.sh" "${model_name}" "${benchmark}" "${model_type}"; then
        echo "!!! ${benchmark} failed" >&2
        failed+=("${benchmark}")
    fi
done

echo
if [[ ${#failed[@]} -gt 0 ]]; then
    echo "Finished with failures: ${failed[*]}" >&2
    exit 1
fi
echo "All benchmarks finished successfully."
