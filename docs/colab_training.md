# Colab CUDA Training

Use the Colab wrapper to run the existing training entrypoints with GPU enforcement, Drive-backed outputs, and automatic resume from `latest_checkpoint.pt`.

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
  --run-name fusion-full \
  --experiment-name fusion_full_colab \
  -- \
  --channel-mode fusion \
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
