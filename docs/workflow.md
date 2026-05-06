# Workflow

## Setup

1. Clone the repo on a machine with the raw sensor data in `data/`
2. Run `01_segment_index_and_windows.ipynb` locally — reads `data/`, writes the window cache to `artifacts/windows/`
3. Upload `artifacts/windows/` to Google Drive at `MyDrive/cs286-project/artifacts/windows/`
4. Run notebooks 04–08 in Colab — they mount Drive and read from that path

## Notebook overview

| Notebook | Where | What |
|---|---|---|
| `01_segment_index_and_windows.ipynb` | Local | Parse raw sensor CSVs, extract segments, build resampled window cache |
| `02_supervised_baseline.ipynb` | Local | CNN supervised baseline on local data (optional — notebooks 04–05 cover the same ground in Colab) |
| `04_standalone_colab_training.ipynb` | Colab | Single-run supervised / contrastive / probe training with checkpoint resume |
| `05_standalone_colab_channel_baselines.ipynb` | Colab | Channel-mode baselines (phone / watch / fusion) |
| `06_colab_watch_ssl_pipeline.ipynb` | Colab | Full SSL pipeline: supervised watch → contrastive pretrain → probes → comparison |
| `07_colab_full_model_sweep.ipynb` | Colab | CNN / ANN / LSTM sweep across watch, fusion, phone |
| `08_loso_evaluation.ipynb` | Colab | Leave-one-subject-out evaluation (classical ML + CNN) |

## Drive layout

```
MyDrive/cs286-project/artifacts/windows/   ← upload from local artifacts/windows/
MyDrive/cs286-project-runs/                 ← training outputs (created by notebooks)
```
