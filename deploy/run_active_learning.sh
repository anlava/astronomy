#!/usr/bin/env bash
set -euo pipefail

# Запуск active learning пайплайна на target.
# Рекомендуется запускать внутри screen/tmux.

cd "$(dirname "$0")/.."
source .venv/bin/activate

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

# При необходимости включите зеркало или прокси:
# export HF_ENDPOINT=https://hf-mirror.com
# export HTTPS_PROXY=http://<proxy>:<port>

mkdir -p active_learning_output_target

python run_pipeline_target.py 2>&1 | tee active_learning_output_target/pipeline.log
