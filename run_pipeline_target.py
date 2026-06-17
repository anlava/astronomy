"""
Запуск Active Learning пайплайна на target (production).
Использует только COLS_TO_USE из FlaresBig.ipynb для экономии RAM.

Запускать:
    python run_pipeline_target.py
"""

import logging
from pathlib import Path

import polars as pl

from active_learning_pipeline import (
    run_active_learning_pipeline,
    PipelineConfig,
    ExpertMode,
)
from labels import surely_pos, surely_neg
from feature_columns import COLS_TO_USE

# ------------------------------------------------------------------
# 1. Настройка логирования
# ------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# 2. Загрузка признаков (только нужные колонки)
# ------------------------------------------------------------------
FEATURES_PATH = Path("./data/all_features.parquet")
logger.info(f"Loading features from {FEATURES_PATH}")
logger.info(f"Using {len(COLS_TO_USE)} columns from COLS_TO_USE")

# Колонки, которые обязательно нужны для пайплайна
required_cols = list(COLS_TO_USE)

features_df = pl.read_parquet(FEATURES_PATH, columns=required_cols)
logger.info(f"Loaded {len(features_df)} rows, {len(features_df.columns)} columns")
logger.info(f"RAM usage: {features_df.estimated_size() / 1024 / 1024:.1f} MB")

# ------------------------------------------------------------------
# 3. Разделение на known_flares и unlabeled pool
# ------------------------------------------------------------------
known_indices = set(surely_pos)

features_with_idx = features_df.with_row_index("row_idx")

known_flares = features_with_idx.filter(
    pl.col("row_idx").is_in(list(known_indices))
).drop("row_idx")

# Убедимся, что class = 1 для known_flares
if "class" not in known_flares.columns:
    known_flares = known_flares.with_columns(pl.lit(1).alias("class"))
else:
    known_flares = known_flares.with_columns(pl.lit(1).alias("class"))

# surely_neg как forced negatives
neg_indices = list(set(surely_neg))
forced_neg_in_parquet = features_with_idx.filter(
    pl.col("row_idx").is_in(neg_indices)
)["row_idx"].to_list()

unlabeled = features_with_idx.filter(
    ~pl.col("row_idx").is_in(list(known_indices))
).drop("row_idx")

logger.info(f"Known flares (surely_pos): {len(known_flares)}")
logger.info(f"Forced negatives (surely_neg): {len(forced_neg_in_parquet)}")
logger.info(f"Unlabeled pool: {len(unlabeled)}")

# ------------------------------------------------------------------
# 4. Конфигурация пайплайна
# ------------------------------------------------------------------
config = PipelineConfig()

# Автоподстройка требований к known flares
n_flares = len(known_flares)
if n_flares < 99:
    logger.warning(f"Only {n_flares} known flares available. Reducing split requirements.")
    config.data.n_train_flares = min(50, n_flares // 2)
    config.data.n_validation_flares = min(20, n_flares // 4)

# Режим с экспертной доразметкой
config.expert_mode = ExpertMode.EXPERT
config.plot_samples = True
config.display_sample_plots = False

# Без ограничения на итерации
config.max_iters = None
# config.max_time_hours = 6.0  # раскомментируйте, если хотите ограничить по времени

# Отключаем GPU и matplotlib plotting
config.catboost.plot = False
config.catboost.use_gpu = False

# ------------------------------------------------------------------
# 5. Загрузка сырых световых кривых для графиков
# ------------------------------------------------------------------
from datasets import load_dataset

logger.info("Loading raw dataset for plotting...")
dataset = load_dataset(
    "snad-space/ztf-m-dwarf-flares-2025",
    split="train",
    cache_dir="~/.cache/huggingface",
)
unlabeled_dataset = dataset
known_flares_dataset = dataset
logger.info(f"Raw dataset loaded: {len(dataset)} samples")

# ------------------------------------------------------------------
# 6. Запуск пайплайна
# ------------------------------------------------------------------
logger.info("Starting active learning pipeline on TARGET...")

results = run_active_learning_pipeline(
    unlabeled_samples=unlabeled,
    known_flares=known_flares,
    config=config,
    output_dir="./active_learning_output_target",
    random_state=42,
    unlabeled_dataset=unlabeled_dataset,
    known_flares_dataset=known_flares_dataset,
    forced_negative_indices=forced_neg_in_parquet,
)

logger.info("=" * 50)
logger.info(f"Pipeline finished!")
logger.info(f"Stop reason: {results['stop_reason']}")
logger.info(f"Best iteration: {results['best_iteration']}")
logger.info("=" * 50)
