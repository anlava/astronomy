# Agent Notes

## Project overview

This is a fork of `fingoldo/astronomy` maintained at `anlava/astronomy`. The project performs active-learning-based detection of M-dwarf flares in the ZTF dataset published on HuggingFace: `snad-space/ztf-m-dwarf-flares-2025`.

Key splits:
- `train`: ~1.6M rows, ~0.8 GB raw data
- `target`: ~94M rows, ~30 GB raw data (314 parquet files)

## Local development (macOS)

- Python 3.11 venv: `/Users/anastasialavruhina/python311`
- Required local stubs: `pyutilz/`, `mlframe/` (these are not on PyPI)
- Fix package name: `PyWavelets`, not `pywt`
- Tkinter is needed only for `flare_labeller.py` GUI; on headless servers it is not required
- On macOS Python 3.11 from Homebrew install `brew install python-tk@3.11` if you want to run the labeller locally

## Feature columns

`feature_columns.py` exports `COLS_TO_USE`, a curated list of 201 columns from `FlaresBig.ipynb`. Using this list reduces RAM for `target` from ~176 GB (478 columns) to ~77 GB.

## Entry points

- `compute_all_features.py --split target` — extract features, writes `data/all_features.parquet`
- `run_pipeline_target.py` — active learning on `target`
- `run_pipeline.py` — active learning on `train` (smoke test)
- `run_pipeline_custom.py` — active learning on custom light curves (see below)
- `check_indices.py` — verify `surely_pos`/`surely_neg` overlap with a parquet file

## Custom light curves (non-HF data)

`extract_features_from_polars(df)` in `astro_flares.py` computes the exact same
~478 features as `extract_all_features` but for any polars DataFrame with
list columns `id, mag, magerr, mjd` (one row per light curve). Keep default
parameters to stay feature-compatible with models trained on the HF dataset.

`run_pipeline_custom.py` is the reference runner for custom data (e.g. ~15k
cluster light curves):

```bash
python run_pipeline_custom.py --input my_curves.parquet \
    --known-flare-indices my_flares.txt \
    --known-negative-indices my_negatives.txt
```

Key facts:
- Input parquet: one row per curve, columns `id`, `mag`, `magerr`, `mjd` (lists)
- Known flares from `target` (`surely_pos`) are mixed in as transfer learning
  by default (disable with `--no-target-labels`)
- `surely_neg` / `expert_labels.txt` are **positional indices into target** and
  are NOT valid for custom data; the script redirects expert labels to
  `<output-dir>/expert_labels_custom.txt` and translates user-supplied index
  files into unlabeled-pool positions
- Negative splits (`n_train_neg_init`, `n_validation_neg`) auto-scale when the
  unlabeled pool is smaller than the target-oriented defaults (110k)

## Deployment

See `deploy/README.md` for full instructions. Quick summary:

1. Create Yandex Cloud VM: 16 vCPU, 128 GB RAM, 500 GB SSD, Ubuntu 22.04
2. Run `deploy/setup_vm.sh` on the VM to install Python 3.11 and dependencies
3. Configure HuggingFace access (VPN/proxy/`HF_ENDPOINT`) because Russian IPs may get 403
4. Extract features: `screen -S features ./deploy/run_compute_features.sh` (данные скачиваются автоматически)
5. Run active learning: `screen -S al ./deploy/run_active_learning.sh`

## Known constraints

- `compute_all_features.py` writes the final parquet only at the end; use a non-preemptible VM for the first run
- `flare_labeller.py` requires a GUI and cannot run on a headless VM
- CatBoost runs on CPU only on the VM (`config.catboost.use_gpu = False`)
- `pyutilz` and `mlframe` are local stubs; do not try to install them from PyPI

## Git workflow

- Origin remote points to `fingoldo/astronomy`
- Fork remote points to `anlava/astronomy`
- Push local changes to `fork main`
