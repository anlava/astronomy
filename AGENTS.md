# AGENTS.md — Astronomy / ZTF M-dwarf Flare Detection

This file contains project-specific context for AI coding agents. The reader is assumed to know nothing about the project.

---

## Project Overview

This repository implements a machine learning pipeline for detecting stellar flares in ZTF (Zwicky Transient Facility) M-dwarf light curves. It operates on the [ZTF M-dwarf Flares 2025 dataset](https://huggingface.co/datasets/snad-space/ztf-m-dwarf-flares-2025) described in [arXiv:2510.24655](https://arxiv.org/pdf/2510.24655).

The project has three main classifier approaches:
1. **Hand-crafted features + CatBoost** — the primary production pipeline (`active_learning_pipeline.py`).
2. **Recurrent neural networks** (LSTM/GRU/Transformer) on raw time series — a PyTorch Lightning drop-in replacement (`recurrent_classifier.py`).
3. **Hybrid** — combining sequence encoders with hand-crafted features.

There is also an active-learning / zero-expert self-training loop that iteratively pseudo-labels the large unlabeled dataset, a Tkinter GUI for human expert labeling (`flare_labeller.py`), and extensive feature engineering including wavelet analysis.

---

## Technology Stack

- **Language**: Python 3 (no packaging manifest — flat script layout)
- **DataFrames**: Polars (preferred), Pandas (fallback/legacy)
- **Array computing**: NumPy, SciPy
- **ML / DL**: CatBoost, scikit-learn, PyTorch, PyTorch Lightning, torchmetrics
- **Wavelets**: PyWT (reference), custom Numba JIT implementations (production)
- **Dataset I/O**: HuggingFace `datasets` library
- **Visualization**: Plotly (primary), Matplotlib (fallback), Pillow (GUI)
- **Performance**: Numba (`@njit`, `prange`), Joblib, psutil
- **Utilities**: tqdm, pyutilz (internal utilities)
- **Optional integrations**: mlframe (internal training framework)

---

## Project Structure

```
.
├── astro_flares.py                  # Core module: feature extraction, wavelets, normalization, plotting
├── active_learning_pipeline.py      # Zero-expert self-training pipeline (CatBoost-based)
├── recurrent_classifier.py          # PyTorch Lightning recurrent classifier (drop-in for CatBoost)
├── build_labeled_dataset.py         # Combines multiple label sources into a labeled feature set
├── flare_labeller.py                # Tkinter GUI for rapid manual labeling of candidates
├── compute_wavelets.py              # CLI script to compute wavelet features only
├── compute_all_features.py          # CLI script to compute all features (main + additional + wavelet + argextremum)
├── sample_probability_plotter.py    # CLI tool to sample & plot light curves from probability parquet files
├── labels.py                        # Hardcoded lists of known positive/negative row indices
├── expert_labels.txt                # JSONL file with per-iteration expert labels
├── tests/                           # pytest test suite
│   ├── conftest.py                  # Shared fixtures (sample light curves, DataFrames)
│   ├── test_astro_flares.py
│   ├── test_active_learning_pipeline.py
│   ├── test_recurrent_classifier.py
│   ├── test_build_labeled_dataset.py
│   ├── test_compute_wavelets.py
│   ├── test_flare_labeller.py
│   ├── test_feature_profiling.py
│   ├── test_wavelet_profiling.py
│   └── benchmark_wavelet_fast.py
├── debug_*.py                       # One-off debugging scripts for wavelet coefficients
├── test_numba_wavelet*.py           # Numba wavelet implementation tests / benchmarks
├── Flares.ipynb / FlaresBig.ipynb   # Jupyter exploration notebooks
├── BadSamples/                      # Example images of edge-case / misclassified samples
└── requirements.txt                 # Pip dependencies
```

### Key Module Divisions

| Module | Responsibility |
|--------|----------------|
| `astro_flares.py` | **Data layer**: load HF datasets, normalize magnitudes, extract statistical & wavelet features, Plotly/Matplotlib visualization utilities. Contains hardcoded wavelet filter coefficients for Numba. |
| `active_learning_pipeline.py` | **Training loop**: asymmetric pseudo-labeling, bootstrap consensus, three-way split (train / validation / held-out), automatic rollback, adaptive thresholds, curriculum learning. Depends on CatBoost. |
| `recurrent_classifier.py` | **Deep learning**: `RecurrentClassifierWrapper` mimics scikit-learn API (`fit`, `predict`, `predict_proba`). Supports `FEATURES_ONLY`, `SEQUENCE_ONLY`, `HYBRID` modes. |
| `build_labeled_dataset.py` | **Dataset construction**: merges `known_flares`, `freaky_held_out_indices`, `forced_positive/negative_indices`, and `expert_labels.txt` into a single Polars DataFrame. Also prepares sequence data for recurrent models. |
| `flare_labeller.py` | **Human-in-the-loop**: keyboard-driven image viewer for labeling active-learning candidates. Saves state to `~/.flare_labeller_state.json`. |

---

## Build and Test Commands

### Installation
```bash
pip install -r requirements.txt
```

### Running Tests
```bash
pytest                    # Run full suite (tests/)
pytest -ra -q -v          # Same as default (see pytest.ini)
pytest --cov              # With coverage (pytest-cov is in requirements)
```

Some tests use `pytest.importorskip` to gracefully skip when heavy dependencies (CatBoost, PyTorch) are missing.

### Feature Extraction (CLI)
```bash
# Compute only wavelet features
python compute_wavelets.py --output-dir ./data --split target

# Compute ALL features (main + additional + wavelet + argextremum)
python compute_all_features.py --output-dir ./data --split target --n-jobs -1 --chunk-size 50000
```

### Active Learning Pipeline
The pipeline is typically invoked via the `run_active_learning_pipeline()` function or from a notebook. It consumes features produced by `astro_flares.extract_all_features()` and writes probabilities, checkpoints, and candidate images to disk.

### Labeling GUI
```bash
python flare_labeller.py [folder_path]
```

---

## Code Style Guidelines

- **Docstrings**: NumPy style (`Parameters`, `Returns`, `Notes`).
- **Type hints**: Used extensively, including `from __future__ import annotations` in newer modules.
- **Constants**: Module-level constants are `UPPER_CASE` (e.g., `DEFAULT_WAVELETS`, `EPSILON`).
- **DataFrames**: Prefer **Polars** over Pandas for new code; existing code may still use Pandas.
- **Performance**: Hot numerical loops are wrapped in Numba `@njit` (often with `parallel=True` / `prange`).
- **Logging**: Use `logging.getLogger(__name__)` — never bare `print()` in library code.
- **Numerical stability**: Epsilon values are explicit (e.g., `EPSILON = 1e-10`, `STD_EPSILON = 1e-8`).
- **Imports**: Optional heavy dependencies are guarded with `try/except ImportError` and boolean flags (e.g., `MLFRAME_AVAILABLE`, `NUMBA_AVAILABLE`).
- **File paths**: Use `pathlib.Path`.
- **Language**: Code and docstrings are in **English**. Some ideation notes (`FlaresIdeas.txt`) contain mixed English and Russian.

---

## Testing Instructions

- Tests live in `tests/` and are discovered automatically by pytest.
- `pytest.ini` configures:
  - `testpaths = tests`
  - `-ra -q --strict-markers -v`
  - Deprecation warnings are ignored globally.
- `conftest.py` provides reusable fixtures (`sample_light_curve_data`, `sample_polars_df`, `feature_df`, `temp_output_dir`).
- When adding new numba-accelerated functions, add corresponding correctness tests against the PyWT reference in `test_astro_flares.py` or `test_numba_wavelet_v2.py`.
- Profiling tests (`test_feature_profiling.py`, `test_wavelet_profiling.py`) exist for performance regression checks.

---

## Runtime Architecture & Data Flow

1. **Ingest**: Load the HuggingFace dataset (`snad-space/ztf-m-dwarf-flares-2025`) via `datasets.load_dataset`.
2. **Pre-processing**: Normalize magnitudes by median magerr; optionally clean single-point outliers.
3. **Feature Engineering** (`astro_flares.py`):
   - Basic statistics (`mag_mean`, `mag_std`, `norm_skewness`, `rise_decay_idx_ratio`, etc.)
   - Additional statistics (velocity, zero-crossings, consecutive excursions)
   - Argextremum statistics (stats before/after argmax/argmin)
   - Wavelet features (DWT with `haar`, `db4`, `db6`, `coif3`, `sym4` — hardcoded filters for Numba)
   - Fraction features (outlier detection based)
4. **Active Learning** (`active_learning_pipeline.py`):
   - Seed with known flares + forced positives/negatives.
   - Iteratively train CatBoost, pseudo-label unlabeled data.
   - Bootstrap consensus validates pseudo-labels.
   - Rollback on degradation (tracked by ICE + tracked-sample sanity checks).
   - Outputs probability parquets and candidate images per iteration.
5. **Evaluation / Plotting**:
   - `sample_probability_plotter.py` samples from probability files and calls `view_series`.
   - `flare_labeller.py` allows human review and expert labeling.
6. **Recurrent Alternative** (`recurrent_classifier.py`):
   - Extracts padded variable-length sequences.
   - Trains RNN/Transformer encoders with optional attention pooling.

---

## Security Considerations

- The project reads arbitrary HuggingFace dataset names and local file paths. Do not pass unsanitized user input to dataset loading functions.
- `flare_labeller.py` writes state to `~/.flare_labeller_state.json` (configurable via `FLARE_LABELLER_STATE_FILE` env var).
- No network services are exposed; this is purely offline batch analysis.
- The `BadSamples/` directory contains images with source IDs and MJDs in filenames — treat as sensitive scientific data if redistributing.

---

## Environment Variables

| Variable | Used By | Default |
|----------|---------|---------|
| `HF_HOME` | `compute_wavelets.py`, `compute_all_features.py`, `astro_flares.py` | `~/.cache/huggingface` |
| `ASTRO_DATA_DIR` | `compute_wavelets.py`, `compute_all_features.py` | `./data` |
| `FLARE_LABELLER_STATE_FILE` | `flare_labeller.py` | `~/.flare_labeller_state.json` |

---

## Common Pitfalls for Agents

- **No `pyproject.toml` / `setup.py`**: This is a flat script project. Imports rely on `sys.path.insert(0, ...)` in test files or running from the repo root.
- **Numba compatibility**: Functions decorated with `@njit` cannot use Python lists of varying types or Polars/Pandas objects. Wavelet filter coefficients are hardcoded as NumPy arrays to avoid PyWT inside Numba.
- **Polars API changes**: The project uses `pl.Expr` extensively; Polars is pinned to `>=0.20.0`.
- **Heavy dependencies**: CatBoost and PyTorch are large. Tests guard against their absence, but running the full pipeline requires them.
- **Dataset size**: The ZTF dataset is large; feature extraction is chunked and parallelized (`n_jobs`, `chunk_size`). Do not attempt to materialize the full dataset in memory without chunking.
- **MJD epoch**: The code uses `datetime(1858, 11, 17, tzinfo=timezone.utc)` as the MJD epoch for date conversions.
