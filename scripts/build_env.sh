#!/usr/bin/env bash
# Set up the evaluation environment:
#   1. Fetch the upstream benchmark repositories at the commits pinned in
#      benchmarks.tsv (they are NOT part of this repository).
#   2. Apply our patches (patches/*.patch) on top of them.
#   3. Create the project venv with uv.
#
# Safe to re-run: benchmarks that were already set up are skipped.
set -euo pipefail

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
ROOT="$(realpath "${SCRIPT_DIR}/..")"
cd "${ROOT}"

# Load .env if present (optional at build time; required to run evaluations)
if [[ -f .env ]]; then set -a; source .env; set +a; fi

MANIFEST="${ROOT}/benchmarks.tsv"
STAMP=".sgic_fineval_ok"

fetch_and_patch() {
    local dest="$1" url="$2" sha="$3" patch="$4"

    if [[ -f "${dest}/${STAMP}" ]]; then
        echo "[skip]  ${dest} (already set up)"
        return
    fi
    if [[ -e "${dest}/.git" ]]; then
        echo "Error: ${dest} exists but is not marked as set up." >&2
        echo "Remove it (rm -rf '${dest}') and re-run this script." >&2
        exit 1
    fi

    echo "[fetch] ${dest} <- ${url} @ ${sha}"
    # Remove the empty placeholder directory left by a parent checkout
    # (e.g. PIXIU records src/financial-evaluation as a submodule).
    rm -rf "${dest}"
    mkdir -p "${dest}"
    git -C "${dest}" init -q
    git -C "${dest}" remote add origin "${url}"
    git -C "${dest}" fetch -q --depth 1 origin "${sha}"
    git -C "${dest}" -c advice.detachedHead=false checkout -q FETCH_HEAD

    if [[ "${patch}" != "-" ]]; then
        echo "[patch] ${dest} <- ${patch}"
        git -C "${dest}" apply --3way "${ROOT}/${patch}"
    fi

    touch "${dest}/${STAMP}"
}

while IFS=$'\t' read -r dest url sha patch; do
    [[ -z "${dest}" || "${dest}" == \#* ]] && continue
    fetch_and_patch "${dest}" "${url}" "${sha}" "${patch}"
done < "${MANIFEST}"

# Setup uv venv
uv venv --clear ./venv
source ./venv/bin/activate
uv pip install --upgrade pip
deactivate

echo "Done."
