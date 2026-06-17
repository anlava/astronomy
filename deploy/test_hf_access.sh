#!/usr/bin/env bash
set -euo pipefail

# Быстрая проверка доступа к HuggingFace и возможности скачать датасет.
# Пробует загрузить только первые 1000 строк сплита train.

cd "$(dirname "$0")/.."
source .venv/bin/activate

echo "=== Testing HuggingFace connectivity ==="
curl -sS -o /dev/null -w "%{http_code}\n" https://huggingface.co || true
curl -sS -o /dev/null -w "%{http_code}\n" https://hf-mirror.com || true

echo "=== Testing dataset download (train split, 1000 rows) ==="
python - <<'PY'
from datasets import load_dataset

try:
    ds = load_dataset(
        "snad-space/ztf-m-dwarf-flares-2025",
        split="train[:1000]",
        cache_dir="~/.cache/huggingface",
        trust_remote_code=True,
    )
    print(f"OK: loaded {len(ds)} rows")
    print(f"Columns: {ds.column_names}")
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")
PY
