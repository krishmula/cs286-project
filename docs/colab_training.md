# Colab CUDA Training

Use the Colab wrapper to run the existing training entrypoints with GPU enforcement, Drive-backed outputs, and automatic resume from `latest_checkpoint.pt`.

For a self-contained notebook that runs the `phone` and `watch` supervised baselines sequentially without cloning the repo, use `notebooks/05_standalone_colab_channel_baselines.ipynb`.

For the standalone Colab workflow that runs the watch `10%` baseline, contrastive pretraining, all six probes, and the comparison summary without cloning the repo, use `notebooks/06_colab_watch_ssl_pipeline.ipynb`.
That notebook now also includes a final "Report Figures" section that regenerates the updated watch-vs-SSL confusion matrices, per-class F1 plot, and ROC curve directly from the saved Drive artifacts.

Given the current subject-disjoint results, the project now treats the `watch` supervised model as the primary baseline for future comparisons. `phone` and `fusion` remain useful ablations, but new methods should be judged first against the watch-only benchmark.

## Typical setup in Colab

```bash
git clone https://github.com/<your-user>/<your-repo>.git /content/cs286-project
cd /content/cs286-project
```

Then switch the notebook runtime to `GPU` and launch training:

```bash
python -m src.train_colab \
  --stage supervised \
  --project-root /content/cs286-project \
  --drive-root /content/drive/MyDrive/cs286-project-runs \
  --run-name watch-full \
  --experiment-name watch_full \
  -- \
  --channel-mode watch \
  --label-fraction 1.0 \
  --epochs 40 \
  --patience 7 \
  --batch-size 256
```

## Resume behavior

- The wrapper writes artifacts into `drive-root/<stage>/<run-name>/`.
- The runners update `latest_checkpoint.pt` during training.
- Re-running the same `src.train_colab` command resumes automatically from that checkpoint.
- Pass `--no-resume` to force a fresh run.

## Probe example

```bash
python -m src.train_colab \
  --stage probe \
  --project-root /content/cs286-project \
  --drive-root /content/drive/MyDrive/cs286-project-runs \
  --run-name pair-probe-10pct \
  --experiment-name pair_probe_10pct_colab \
  -- \
  --encoder-ckpt-path /content/drive/MyDrive/cs286-project-runs/contrastive/pretrain/latest_checkpoint.pt \
  --evaluation-mode pair \
  --label-fraction 0.1 \
  --epochs 40 \
  --patience 7 \
  --batch-size 256
```

## Repo runner for phone/watch baselines

To launch the supervised `phone` and `watch` baselines sequentially from a checked-out repo, run:

```bash
python -m src.train_supervised_baselines \
  --project-root /content/cs286-project \
  --output-root /content/drive/MyDrive/cs286-project-runs/supervised_channel_baselines \
  --run-name-prefix full-baseline \
  --experiment-prefix colab_baseline \
  -- \
  --label-fraction 1.0 \
  --epochs 40 \
  --patience 7 \
  --batch-size 256
```

This creates separate output directories such as `.../full-baseline-phone/` and `.../full-baseline-watch/`, each with its own `latest_checkpoint.pt` for independent resume behavior. Use the watch run as the main supervised comparator unless you are explicitly studying device ablations or fusion behavior.

## Comparison summary with an existing watch baseline

If your full-label watch baseline came from the older channel-baselines notebook and its experiment name is `standalone_supervised_watch`, you can still use it in the watch-based report by passing:

```bash
python -m src.reporting.compare_runs \
  --baseline-dir /content/drive/MyDrive/cs286-project-runs/supervised \
  --probe-dir /content/drive/MyDrive/cs286-project-runs/probe \
  --output-dir /content/drive/MyDrive/cs286-project-runs/reports \
  --experiment-alias standalone_supervised_watch=watch_full \
  --expected-experiments watch_full watch_10pct contrastive_pair_probe_10pct contrastive_pair_probe_100pct contrastive_phone_probe_10pct contrastive_phone_probe_100pct contrastive_watch_probe_10pct contrastive_watch_probe_100pct

## Report figure generation from a checked-out repo

To regenerate the updated watch-first report figures from saved artifacts, run:

```bash
python -m src.reporting.generate_report_figures \
  --baseline-dir artifacts/baseline \
  --probe-dir artifacts/probes \
  --output-dir figures
```

If you are reusing an older full-watch artifact named `standalone_supervised_watch`, add:

```bash
  --experiment-alias standalone_supervised_watch=watch_full
```
```
