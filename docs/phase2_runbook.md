# Phase 2 Runbook

This runbook is the command path from the cached windows in `artifacts/windows` to the core Phase 2 comparison outputs.

## 1. Canonical supervised baselines

Run the full-label fused baseline:

```bash
python -m src.train_supervised \
  --project-root /Users/krishna/dev/cs286-project \
  --output-dir /Users/krishna/dev/cs286-project/artifacts/baseline \
  --experiment-name fusion_full \
  --channel-mode fusion \
  --label-fraction 1.0
```

Run the matched 10% fused baseline:

```bash
python -m src.train_supervised \
  --project-root /Users/krishna/dev/cs286-project \
  --output-dir /Users/krishna/dev/cs286-project/artifacts/baseline \
  --experiment-name fusion_10pct \
  --channel-mode fusion \
  --label-fraction 0.1
```

Expected outputs per run:

- `*_checkpoint.pt`
- `*_metrics.json`
- `*_confusion_matrix.png`
- `*_per_subject_accuracy.csv`

## 2. Contrastive pretraining

Train the phone-watch contrastive model:

```bash
python -m src.train_contrastive \
  --project-root /Users/krishna/dev/cs286-project \
  --output-dir /Users/krishna/dev/cs286-project/artifacts/contrastive
```

The default contrastive learning rate is `3e-4`.

Expected outputs:

- `*_checkpoint.pt` for the best validation-loss checkpoint
- `*_last_epoch##_checkpoint.pt` for the final epoch checkpoint
- `*_metrics.json`

The checkpoints contain full model weights plus separate encoder/projector state dicts for probe reuse.

## 3. Linear probes

Assume the contrastive checkpoint is:

```bash
CKPT=/Users/krishna/dev/cs286-project/artifacts/contrastive/<contrastive_last_epoch_checkpoint>.pt
```

For probe quality, prefer the saved `*_last_epoch##_checkpoint.pt` unless you specifically want the checkpoint with the lowest contrastive validation loss.

Run paired probes:

```bash
python -m src.train_probe \
  --project-root /Users/krishna/dev/cs286-project \
  --encoder-ckpt-path "$CKPT" \
  --output-dir /Users/krishna/dev/cs286-project/artifacts/probes \
  --experiment-name contrastive_pair_probe_10pct \
  --evaluation-mode pair \
  --label-fraction 0.1

python -m src.train_probe \
  --project-root /Users/krishna/dev/cs286-project \
  --encoder-ckpt-path "$CKPT" \
  --output-dir /Users/krishna/dev/cs286-project/artifacts/probes \
  --experiment-name contrastive_pair_probe_100pct \
  --evaluation-mode pair \
  --label-fraction 1.0
```

Run phone-only probes:

```bash
python -m src.train_probe \
  --project-root /Users/krishna/dev/cs286-project \
  --encoder-ckpt-path "$CKPT" \
  --output-dir /Users/krishna/dev/cs286-project/artifacts/probes \
  --experiment-name contrastive_phone_probe_10pct \
  --evaluation-mode phone \
  --label-fraction 0.1

python -m src.train_probe \
  --project-root /Users/krishna/dev/cs286-project \
  --encoder-ckpt-path "$CKPT" \
  --output-dir /Users/krishna/dev/cs286-project/artifacts/probes \
  --experiment-name contrastive_phone_probe_100pct \
  --evaluation-mode phone \
  --label-fraction 1.0
```

Run watch-only probes:

```bash
python -m src.train_probe \
  --project-root /Users/krishna/dev/cs286-project \
  --encoder-ckpt-path "$CKPT" \
  --output-dir /Users/krishna/dev/cs286-project/artifacts/probes \
  --experiment-name contrastive_watch_probe_10pct \
  --evaluation-mode watch \
  --label-fraction 0.1

python -m src.train_probe \
  --project-root /Users/krishna/dev/cs286-project \
  --encoder-ckpt-path "$CKPT" \
  --output-dir /Users/krishna/dev/cs286-project/artifacts/probes \
  --experiment-name contrastive_watch_probe_100pct \
  --evaluation-mode watch \
  --label-fraction 1.0
```

Expected outputs per probe run:

- `*_checkpoint.pt`
- `*_metrics.json`
- `*_confusion_matrix.png`
- `*_per_subject_accuracy.csv`

## 4. Comparison summary

Generate the standardized comparison report:

```bash
python -m src.reporting.compare_runs \
  --baseline-dir /Users/krishna/dev/cs286-project/artifacts/baseline \
  --probe-dir /Users/krishna/dev/cs286-project/artifacts/probes \
  --output-dir /Users/krishna/dev/cs286-project/artifacts/reports
```

Expected outputs:

- `comparison_summary.json`
- `comparison_summary.md`

The summary report requires these experiment names to exist:

- `fusion_full`
- `fusion_10pct`
- `contrastive_pair_probe_10pct`
- `contrastive_pair_probe_100pct`
- `contrastive_phone_probe_10pct`
- `contrastive_phone_probe_100pct`
- `contrastive_watch_probe_10pct`
- `contrastive_watch_probe_100pct`

## 5. Smoke-test command

For the current automated checks, run:

```bash
python -m unittest
```

This covers:

- cached-window validation
- metrics/checkpoint utilities
- supervised smoke training
- contrastive smoke training
- probe smoke training across pair/phone/watch
- comparison reporting
- an end-to-end mini pipeline
