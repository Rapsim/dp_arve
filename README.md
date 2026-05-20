# Project overview

This repository contains notebooks, data and outputs for the hydrological analysis used in our report. The README below helps graders quickly find which notebooks perform which steps, which output files contain results, and how to run the notebooks locally.

**Purpose**: reproduce analysis and figures for the report, inspect model metrics, and view example timeseries/peak detections.

**Environment**:
- **Conda environment**: use `environment.yml` or `environment.lock.yml` to create the same environment.

Run example:

```powershell
conda env create -f environment.yml
conda activate analysis
# open notebooks with JupyterLab or VS Code
jupyter lab
```

**Notebooks (high-level)**
- `notebooks/determinist_fixed_LT_by_foen.ipynb`: deterministic lead-time analysis using FOEN forcings; creates summary metrics and plots under `outputs/OFEV_determinist/`.
- `notebooks/determinist_fixed_LT.py`: script form of the deterministic analysis (usable for batch runs).
- `notebooks/determinist_windows_LT.py`: windowed/dynamic lead-time deterministic analyses and visual checks.
- `notebooks/foen_nested_timeseries_fixed_LT.ipynb`: builds nested FOEN time-series and fixed lead-time experiments.
- `notebooks/foen_nested_timeseries_windows_LT.ipynb`: windowed variant of FOEN nested timeseries experiments.
- `notebooks/plot_foen_ensemble_events_24h.ipynb`: plotting ensemble event examples for 24h leadtime.
- `notebooks/plot_metrics_by_foen_forcing.ipynb`: creates figures summarizing metrics by FOEN forcing.
- `notebooks/probabilistic_foen.py`: script/notebook for probabilistic FOEN experiments (ensemble-based metrics).

If you are grading and want quick results, start with `notebooks/plot_metrics_by_foen_forcing.ipynb` to see aggregated figures and `plot_foen_ensemble_events_24h.ipynb` for concrete timeseries examples.

**Data**
- `data/`: raw and prepared inputs. Notable subfolders:
  - `Data Fornisseurs/ARVE/`: observational time series CSVs for station 2170 and others.
  - `Data Fornisseurs/Hydrique_model/`: model forecast archives and README files.
  - `data/moi/`: user-prepared forcings and control files (used by notebooks).

**Outputs**
- `outputs/metrics_all_models.csv`: consolidated metrics across models (good starting point for graders).
- `outputs/OFEV_determinist/`: deterministic experiment outputs and summary tables; includes subfolders per model and `metrics_all_models_summary.csv`.
- `outputs/OFEV_probabilistic/`: probabilistic experiment summaries including `OFEV_by_model_metrics.csv` and per-station quantiles like `station2170_q_quantiles_by_model.csv`.
- `outputs/LT_*h/`: lead-time specific outputs (6h, 12h, 24h, 36h, 48h).
- `outputs/peak_windows/`: peak detection results and `peak_metrics.csv` plus example timeseries in `flood_timeseries/`.

**Where to find key items for grading**
- Aggregated metrics: [outputs/metrics_all_models.csv](outputs/metrics_all_models.csv)
- Deterministic summary: [outputs/OFEV_determinist/metrics_all_models_summary.csv](outputs/OFEV_determinist/metrics_all_models_summary.csv)
- Probabilistic summary: [outputs/OFEV_probabilistic/OFEV_global_metrics.csv](outputs/OFEV_probabilistic/OFEV_global_metrics.csv)
- Example timeseries and confusion matrices: [outputs/peak_windows/](outputs/peak_windows/)

**Recommended quick checks for graders**
1. Open [notebooks/plot_metrics_by_foen_forcing.ipynb](notebooks/plot_metrics_by_foen_forcing.ipynb) to view summary plots.
2. Inspect [outputs/OFEV_determinist/metrics_all_models_summary.csv](outputs/OFEV_determinist/metrics_all_models_summary.csv) for per-model numbers used in the report.
3. Browse [outputs/peak_windows/flood_timeseries/](outputs/peak_windows/flood_timeseries/) for example event plots.

**Notes**
- Large raw data are in `data/` and some files are archived; if a notebook cannot find a file, check the matching README in the `data/*` subfolders.
- The `notebooks/*.py` script variants can be used to run parts of the analysis non-interactively.

If you want, I can: (a) add more detailed descriptions for any specific notebook, (b) generate a short grading checklist, or (c) open/preview a few example outputs to include thumbnails here.
