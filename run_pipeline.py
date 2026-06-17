"""
Запуск Active Learning пайплайна для поиска вспышек.

Запускать тем же python, которым запускали compute_all_features.py:
    python run_pipeline.py
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

# ------------------------------------------------------------------
# 1. Настройка логирования
# ------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# 2. Загрузка признаков
# ------------------------------------------------------------------
FEATURES_PATH = Path("./data/all_features.parquet")
logger.info(f"Loading features from {FEATURES_PATH}")

features_df = pl.read_parquet(FEATURES_PATH)
logger.info(f"Loaded {len(features_df)} rows, {len(features_df.columns)} columns")

# ------------------------------------------------------------------
# 3. Разделение на known_flares и unlabeled pool
# ------------------------------------------------------------------
# surely_pos — индексы строк, которые точно являются вспышками
known_indices = set(surely_pos)

# Добавляем row_index, чтобы фильтровать
features_with_idx = features_df.with_row_index("row_idx")

known_flares = features_with_idx.filter(
    pl.col("row_idx").is_in(list(known_indices))
).drop("row_idx").with_columns(pl.lit(1).alias("class"))

# surely_neg как forced negatives — надёжные стартовые негативы
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

# Уменьшаем требования к known flares (в train сплите только 74)
config.data.n_train_flares = 50
config.data.n_validation_flares = 20

# Режим: с экспертной доразметкой
# Пайплайн будет периодически останавливаться и показывать uncertain-кандидатов
config.expert_mode = ExpertMode.EXPERT

# Визуализация: строить PNG кривых кандидатов
config.plot_samples = True

# Не показывать matplotlib-окна (мешают в консоли)
config.display_sample_plots = False

# Ограничения (уберите или увеличьте для полноценного прогона)
config.max_iters = 100
# config.max_time_hours = 2.0

# Отключаем CatBoost plotting (не нужен ipywidgets) и GPU (на Mac нет CUDA)
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
logger.info("Starting active learning pipeline...")

results = run_active_learning_pipeline(
    unlabeled_samples=unlabeled,
    known_flares=known_flares,
    config=config,
    output_dir="./active_learning_output",
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
