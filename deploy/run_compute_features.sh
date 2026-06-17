#!/usr/bin/env bash
set -euo pipefail

# Запуск извлечения признаков для target.
# Рекомендуется запускать внутри screen/tmux.

cd "$(dirname "$0")/.."
source .venv/bin/activate

mkdir -p data

# Настройки по умолчанию
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export ASTRO_DATA_DIR="${ASTRO_DATA_DIR:-$PWD/data}"

LOG_FILE="$ASTRO_DATA_DIR/compute_features_target.log"

# При необходимости включите зеркало или прокси:
# export HF_ENDPOINT=https://hf-mirror.com
# export HTTPS_PROXY=http://<proxy>:<port>

python compute_all_features.py \
    --split target \
    --cache-dir "$HF_HOME" \
    --output-dir "$ASTRO_DATA_DIR" \
    --output-file all_features.parquet \
    --n-jobs -1 \
    --chunk-size 50000 \
    2>&1 | tee "$LOG_FILE"
