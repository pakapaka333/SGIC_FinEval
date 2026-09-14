#!/usr/bin/env bash
set -euo pipefail

model_name=${1:-}
benchmark_name=${2:-}
model_type=${3:-}

if [[ -z "$model_name" || -z "$benchmark_name" || -z "$model_type" ]]; then
    echo "Usage: $0 <model_name> <benchmark_name> <model_type>" >&2
    echo "  model_type: base | inst | reasoning" >&2
    exit 1
fi

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
if [[ ! -f "${SCRIPT_DIR}/../.env" ]]; then
    echo "Error: .env not found. Copy .env_template to .env and fill it in." >&2
    exit 1
fi
set -a; source "${SCRIPT_DIR}/../.env"; set +a;
if [[ -z "${PROJECT_ROOT:-}" ]]; then
    echo "Error: PROJECT_ROOT is not set in .env." >&2
    exit 1
fi
cd "$PROJECT_ROOT"

runner="${PROJECT_ROOT}/scripts/${benchmark_name}/run.sh"
if [[ ! -f "$runner" ]]; then
    echo "Runner script missing: $runner" >&2
    exit 1
fi

# The benchmark code must be present under benchmarks/. PIXIU and
# japanese-lm-fin-harness are fetched by scripts/build_env.sh; the
# dr-finance-instruction code ships with this repository.
benchmark_dir="${PROJECT_ROOT}/benchmarks/${benchmark_name}"
if [[ ! -d "$benchmark_dir" ]]; then
    echo "Error: benchmark code not found: $benchmark_dir" >&2
    case "$benchmark_name" in
        dr-finance-instruction)
            echo "  This directory is part of this repository, not fetched by build_env.sh." >&2
            echo "  Your checkout is incomplete - re-clone the repository (or restore" >&2
            echo "  benchmarks/${benchmark_name}/ from it)." >&2
            ;;
        *)
            echo "  Run 'bash scripts/build_env.sh' to fetch it." >&2
            ;;
    esac
    exit 1
fi

output_dir="${PROJECT_ROOT}/results/${model_name}/${model_type}/${benchmark_name}"
mkdir -p "$output_dir"

export MODEL_NAME="$model_name"
export MODEL_TYPE="$model_type"
export OUTPUT_DIR="$output_dir"
export PROJECT_ROOT="$PROJECT_ROOT"
export JOB_ID="$(date +%s)"

# flashinfer JIT-compiles CUDA kernels and caches the generated ninja build
# files, which record absolute paths into the installed flashinfer package.
# Benchmarks run via `uv run --isolated`, so that package lives in a fresh
# ephemeral venv on every invocation and a cache shared between runs points at
# directories that no longer exist ("missing and no known rule to make it").
# Give each run a private JIT cache; keep the downloaded cubins shared.
export FLASHINFER_CUBIN_DIR="${FLASHINFER_CUBIN_DIR:-${HOME}/.cache/flashinfer/cubins}"
flashinfer_workspace="$(mktemp -d "${TMPDIR:-/tmp}/flashinfer-${JOB_ID}-XXXXXX")"
trap 'rm -rf "$flashinfer_workspace"' EXIT
export FLASHINFER_WORKSPACE_BASE="$flashinfer_workspace"

bash "$runner"