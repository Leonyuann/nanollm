#!/usr/bin/env bash
set -euo pipefail
lrs=("5e-3" "4e-3" "3e-3" "2e-3" "1e-3" "9e-4" "8e-4" "7e-4" "6e-4" "5e-4")

for lr in "${lrs[@]}"; do
  echo "Running: $lr"
  uv run python scripts/train_llm.py --lr "$lr"
done