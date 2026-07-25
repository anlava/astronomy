"""
Запуск Active Learning пайплайна на СВОИХ световых кривых (не HF-датасет).

Сценарий: у вас есть свой набор кривых блеска ZTF (например, ~15k объектов из
звёздных скоплений), и вы хотите найти среди них вспышки, используя разметку,
уже накопленную на сплите `target` (surely_pos + expert labels).

Пайплайн работает с ТАБЛИЦАМИ ПРИЗНАКОВ, поэтому скрипт:
  1. загружает ваши кривые (id, mag, magerr, mjd);
  2. вычисляет ровно те же ~478 признаков, что и для HF-датасета
     (extract_features_from_polars с параметрами по умолчанию);
  3. подмешивает known flares с `target` (transfer learning) и/или вашу
     собственную разметку;
  4. запускает run_active_learning_pipeline.

Формат входного parquet (--input):
    Один ряд = одна кривая. Колонки:
      - id     : str/int, уникальный идентификатор объекта
      - mag    : list[f32/f64], звёздные величины
      - magerr : list[f32/f64], ошибки (та же длина, что mag)
      - mjd    : list[f64], моменты наблюдений (та же длина)
      - class  : (опционально) int, 1 для известных вспышек

    Пример создания такого файла из numpy-массивов:

        import polars as pl
        pl.DataFrame({
            "id": obj_ids,            # список str
            "mag": mags,              # список np.array / list[float]
            "magerr": magerrs,
            "mjd": mjds,
        }).write_parquet("my_curves.parquet")

Файлы с разметкой (--known-flare-indices и т.п.): текстовые файлы,
по одному целому числу (позиция ряда во входном parquet, 0-based) на строку.

Примеры запуска:
    # Только inference-подготовка признаков + AL с разметкой target:
    python run_pipeline_custom.py --input my_curves.parquet

    # Со своей разметкой и без подмеса target:
    python run_pipeline_custom.py --input my_curves.parquet \
        --known-flare-indices my_flares.txt \
        --known-negative-indices my_negatives.txt \
        --no-target-labels
"""

import argparse
import logging
from pathlib import Path

import polars as pl

from active_learning_pipeline import (
    run_active_learning_pipeline,
    PipelineConfig,
    ExpertMode,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Active learning flare search on custom light curves",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", type=Path, required=True,
                   help="Parquet с кривыми: колонки id, mag, magerr, mjd")
    p.add_argument("--features-cache", type=Path, default=Path("./data/custom_features.parquet"),
                   help="Куда сохранить/откуда читать вычисленные признаки")
    p.add_argument("--output-dir", type=Path, default=Path("./active_learning_output_custom"),
                   help="Директория результатов пайплайна")
    p.add_argument("--target-features", type=Path, default=Path("./data/all_features.parquet"),
                   help="Parquet признаков с размеченными вспышками (для transfer learning)")
    p.add_argument("--target-label-source", choices=["surely_pos", "class_one"], default="class_one",
                   help="Как брать known flares из --target-features: по индексам surely_pos "
                        "(для target-сплита) или по class==1 (для train-сплита)")
    p.add_argument("--max-target-flares", type=int, default=10000,
                   help="Максимум known flares из --target-features (случайная подвыборка, 0 = все)")
    p.add_argument("--no-target-labels", action="store_true",
                   help="Не использовать known flares из --target-features (только своя разметка)")
    p.add_argument("--known-flare-indices", type=Path, default=None,
                   help="Текстовый файл: позиции (0-based) ваших известных вспышек")
    p.add_argument("--known-negative-indices", type=Path, default=None,
                   help="Текстовый файл: позиции надёжных НЕ-вспышек (forced negatives)")
    p.add_argument("--plot", action="store_true",
                   help="Строить графики кандидатов (требует память под сырые кривые)")
    p.add_argument("--expert", action="store_true",
                   help="Включить expert mode (интерактивная разметка через flare_labeller, нужен GUI)")
    p.add_argument("--max-iters", type=int, default=None,
                   help="Ограничение числа итераций AL (по умолчанию без лимита)")
    return p.parse_args()


def load_index_file(path: Path | None) -> list[int]:
    if path is None:
        return []
    return [int(line.strip()) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------
    # 1. Признаки для своих кривых (те же ~478 колонок, что и на HF)
    # ------------------------------------------------------------------
    if args.features_cache.exists():
        logger.info(f"Loading cached features from {args.features_cache}")
        features_df = pl.read_parquet(args.features_cache)
    else:
        from astro_flares import extract_features_from_polars

        logger.info(f"Loading light curves from {args.input}")
        lc_df = pl.read_parquet(args.input)
        logger.info(f"Loaded {len(lc_df)} light curves, columns: {lc_df.columns}")

        logger.info("Computing features (defaults => compatible with HF models)...")
        features_df = extract_features_from_polars(lc_df)
        args.features_cache.parent.mkdir(parents=True, exist_ok=True)
        features_df.write_parquet(args.features_cache, compression="zstd")
        logger.info(f"Saved features to {args.features_cache}")

    logger.info(f"Features: {len(features_df)} rows x {len(features_df.columns)} cols")

    custom_known_idx = set(load_index_file(args.known_flare_indices))
    custom_neg_idx = set(load_index_file(args.known_negative_indices))

    features_with_idx = features_df.with_row_index("row_idx")

    # ------------------------------------------------------------------
    # 2. Known flares: своя разметка + (опционально) разметка с target
    # ------------------------------------------------------------------
    custom_known = (
        features_with_idx.filter(pl.col("row_idx").is_in(list(custom_known_idx)))
        .drop("row_idx")
        .with_columns(pl.lit(1).alias("class"))
    )

    known_parts = []
    if len(custom_known) > 0:
        known_parts.append(custom_known)
        logger.info(f"Custom known flares: {len(custom_known)}")

    if not args.no_target_labels and args.target_features.exists():
        logger.info(f"Loading labeled features for transfer learning from {args.target_features}")
        target_df = pl.read_parquet(args.target_features)

        if args.target_label_source == "surely_pos":
            from labels import surely_pos

            target_known = (
                target_df.with_row_index("row_idx")
                .filter(pl.col("row_idx").is_in(list(set(surely_pos))))
                .drop("row_idx")
            )
        else:  # class_one
            target_known = target_df.filter(pl.col("class") == 1)

        if args.max_target_flares and len(target_known) > args.max_target_flares:
            target_known = target_known.sample(n=args.max_target_flares, seed=42)

        target_known = target_known.with_columns(pl.lit(1).alias("class"))
        # Оставляем только колонки, общие с нашими признаками
        common_cols = [c for c in features_df.columns if c in target_known.columns]
        target_known = target_known.select(common_cols)
        known_parts.append(target_known)
        logger.info(f"Transfer known flares ({args.target_label_source}): {len(target_known)}")

    if not known_parts:
        raise ValueError(
            "Нет known flares: передайте --known-flare-indices и/или "
            "уберите --no-target-labels и проверьте --target-features"
        )

    known_flares = pl.concat(known_parts, how="diagonal_relaxed")

    # ------------------------------------------------------------------
    # 3. Unlabeled pool + forced negatives (позиции в unlabeled!)
    # ------------------------------------------------------------------
    unlabeled = features_with_idx.filter(
        ~pl.col("row_idx").is_in(list(custom_known_idx))
    ).drop("row_idx")

    # Индексы не-вспышек задаются пользователем как позиции во ВХОДНОМ файле;
    # переводим их в позиции внутри unlabeled.
    input_pos_to_unlabeled_pos = {}
    pos = 0
    for i in range(len(features_df)):
        if i not in custom_known_idx:
            input_pos_to_unlabeled_pos[i] = pos
            pos += 1
    forced_negative_indices = [
        input_pos_to_unlabeled_pos[i] for i in custom_neg_idx if i in input_pos_to_unlabeled_pos
    ]

    logger.info(f"Known flares total: {len(known_flares)}")
    logger.info(f"Unlabeled pool: {len(unlabeled)}")
    logger.info(f"Forced negatives: {len(forced_negative_indices)}")

    # ------------------------------------------------------------------
    # 4. Конфигурация пайплайна
    # ------------------------------------------------------------------
    config = PipelineConfig()

    n_flares = len(known_flares)
    if n_flares < 99:
        logger.warning(f"Only {n_flares} known flares. Reducing split requirements.")
        config.data.n_train_flares = min(50, n_flares // 2)
        config.data.n_validation_flares = min(20, n_flares // 4)
        config.data.n_held_out_flares = 0

    # Дефолтные negative splits (35k train + 75k val) рассчитаны на ~94M строк target.
    # Для маленького custom-пула масштабируем, оставляя большую часть в пуле кандидатов.
    required_negs = config.data.n_train_neg_init + config.data.n_validation_neg + config.data.n_held_out_neg
    if len(unlabeled) < required_negs:
        logger.warning(
            f"Unlabeled pool ({len(unlabeled)}) smaller than default negative splits "
            f"({required_negs}). Scaling down."
        )
        config.data.n_train_neg_init = min(5000, len(unlabeled) // 3)
        config.data.n_validation_neg = min(5000, len(unlabeled) // 3)
        config.data.n_held_out_neg = 0

    config.expert_mode = ExpertMode.EXPERT if args.expert else ExpertMode.NO_EXPERT
    config.plot_samples = args.plot
    config.display_sample_plots = False
    config.max_iters = args.max_iters
    config.catboost.plot = False
    config.catboost.use_gpu = False

    # ------------------------------------------------------------------
    # 5. Сырые кривые для графиков (опционально)
    # ------------------------------------------------------------------
    unlabeled_dataset = None
    if args.plot:
        lc_df = pl.read_parquet(args.input)
        mags = lc_df["mag"].to_list()
        magerrs = lc_df["magerr"].to_list()
        mjds = lc_df["mjd"].to_list()
        unlabeled_dataset = [
            {"mag": list(mags[i]), "magerr": list(magerrs[i]), "mjd": list(mjds[i]), "class": 0}
            for i in range(len(lc_df))
            if i not in custom_known_idx
        ]

    # ------------------------------------------------------------------
    # 6. Запуск
    # ------------------------------------------------------------------
    logger.info("Starting active learning pipeline on CUSTOM data...")

    # ВАЖНО: expert_labels.txt по умолчанию содержит ПОЗИЦИОННЫЕ метки с target.
    # Они невалидны для custom-данных, поэтому перенаправляем expert labels
    # в отдельный файл внутри output_dir (несуществующий => старые не подмешаются,
    # новые метки expert mode будут писаться туда).
    args.output_dir.mkdir(parents=True, exist_ok=True)  # _save_expert_labels не создаёт родителя
    results = run_active_learning_pipeline(
        unlabeled_samples=unlabeled,
        known_flares=known_flares,
        config=config,
        output_dir=str(args.output_dir),
        random_state=42,
        unlabeled_dataset=unlabeled_dataset,
        forced_negative_indices=forced_negative_indices,
        expert_labels_file=str(args.output_dir / "expert_labels_custom.txt"),
    )

    logger.info("=" * 50)
    logger.info("Pipeline finished!")
    logger.info(f"Stop reason: {results['stop_reason']}")
    logger.info(f"Best iteration: {results['best_iteration']}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
