#!/usr/bin/env python3
"""Скачать датасет snad-space/ztf-m-dwarf-flares-2025 в HF-кэш.

Используется datasets.load_dataset, который тот же механизм, что и в
compute_all_features.py / run_pipeline_target.py.
"""
import argparse
import logging
import os
from pathlib import Path

from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DATASET_NAME = "snad-space/ztf-m-dwarf-flares-2025"


def main():
    parser = argparse.ArgumentParser(description="Download ZTF M-dwarf flares dataset from HuggingFace")
    parser.add_argument("--split", default="target", choices=["train", "target", "test"])
    parser.add_argument("--cache-dir", default=str(Path.home() / ".cache" / "huggingface"))
    parser.add_argument("--num-proc", type=int, default=4, help="Parallel download processes")
    args = parser.parse_args()

    logger.info(f"Downloading {DATASET_NAME} split={args.split} to {args.cache_dir}")
    ds = load_dataset(
        DATASET_NAME,
        split=args.split,
        cache_dir=args.cache_dir,
        num_proc=args.num_proc,
        trust_remote_code=True,
    )
    logger.info(f"Loaded {len(ds)} rows")
    logger.info(f"Columns: {ds.column_names}")


if __name__ == "__main__":
    main()
