#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "$script_dir/../.." && pwd)"
cd "$project_root"
python_bin="${TITANIC_PYTHON:-$project_root/.venv/bin/python}"

for activation in sigmoid relu elu leaky_relu; do
  for batch_size in 32 64 128; do
    for learning_rate in 1e-2 1e-3 1e-4; do
      printf 'Training: activation=%s batch_size=%s learning_rate=%s\n' \
        "$activation" "$batch_size" "$learning_rate"
      "$python_bin" "$script_dir/train_titanic_with_argparse_wandb.py" \
        "$@" \
        --activation "$activation" \
        --batch_size "$batch_size" \
        --learning_rate "$learning_rate"
    done
  done
done
