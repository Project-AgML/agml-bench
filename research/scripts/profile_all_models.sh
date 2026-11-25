#!/usr/bin/env bash
# Profile every registered model, then the judge. Usage: research/scripts/profile_all_models.sh <DATA_DIR> [extra eval profile args]
set -euo pipefail
DATA_DIR="$1"; shift
uv run agml-bench eval profile --data-dir "$DATA_DIR" --models all --cleanup-weights --out reports/profile "$@"
