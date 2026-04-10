# Phase 2 Plan: Contrastive Extension on Top of the Existing Baseline

## Summary

- Treat the current repo as having completed the baseline and preprocessing groundwork already, not as a fresh project start.
- Build the next phase as a hybrid cleanup: keep the existing scripts, notebooks, and artifacts intact, but add a reusable `src/` training/data layer for contrastive pretraining and probing.
- Freeze the current data pipeline for the first SSL milestone: use the existing cached windows in `artifacts/windows` with the current `3.0 s` window length, `1.0 s` stride, and `20 Hz` sampling.
- Do not re-open raw-data alignment or windowing choices in this phase. Phone and watch views will be derived by slicing the existing 12-channel cached tensors.
- The first milestone is a core SSL result, not the full paper-style suite: implement contrastive pretraining, rerun the watch-only supervised baseline, and run paired plus single-device linear probes at `10%` and `100%` label fractions.

## Key Changes

- `Repo structure`: add a reusable `src/` package for data loading, model definitions, training loops, checkpointing, and evaluation. Keep current scripts as thin entrypoints or backward-compatible references. Do not do a full repo refactor in this phase.
- `Canonical sample interface`: standardize one in-memory window sample contract with `x_fusion` shape `[12, 60]`, `x_phone` shape `[6, 60]`, `x_watch` shape `[6, 60]`, plus `label_idx`, `subject_id`, `window_id`, `split`, and the metadata fields already present in `metadata.csv`.
- `View derivation`: derive `x_phone = x_fusion[0:6]` and `x_watch = x_fusion[6:12]` from cached payloads. Do not rebuild caches just to materialize separate tensors.
- `Split policy`: use existing `train` for fitting, existing `val` for early stopping/model selection, and existing `test` only for final reporting. Preserve the subject-disjoint split already encoded in `artifacts/windows/manifest.json`.
- `Normalization policy`: mirror current baseline behavior for comparability. Do not introduce a new normalization transform in phase 2 unless the cached tensors are proven unusable without it.
- `Supervised baseline`: rerun and standardize a full-label watch-only supervised baseline as the primary comparator, then keep phone-only and fusion runs as secondary ablations using the exact same subset-selection logic used for probe training.
- `Encoder architecture`: use the current 1D CNN backbone pattern as the default encoder. For SSL, instantiate separate phone and watch encoders with identical architecture and independent weights. Output embedding dimension is `256`.
- `Projection head`: use a 2-layer MLP `256 -> 128 -> 64` with ReLU between layers and L2 normalization at the output.
- `Contrastive loss`: use symmetric phone-to-watch and watch-to-phone InfoNCE with in-batch negatives. Default temperature is `0.2`.
- `Augmentations`: include a minimal, fixed v1 augmentation pipeline for pretraining only: Gaussian jitter, mild amplitude scaling, and short temporal masking. Keep them configurable by CLI flags, but ship with one default setting so the implementation is not making choices on the fly.
- `Pretraining dataset`: train on unlabeled train windows only, even though labels exist in metadata. Labels are only used later for probes and evaluation.
- `Probe setup`: make concatenated paired embeddings the headline downstream result: `h_pair = [h_phone ; h_watch]` with a linear classifier on top. Also train phone-only and watch-only linear probes on the frozen encoders as secondary results.
- `Label fractions`: for the first milestone, run `10%` and `100%` labeled training fractions only. Use subject-balanced, class-aware subset selection matching the current baseline style. Leave `1%`, `5%`, and `25%` for a later expansion.
- `Artifacts`: write contrastive checkpoints, probe checkpoints, metrics JSON, confusion matrices, per-subject accuracy CSVs, and one comparison table to a dedicated artifact area. Keep `artifacts/baseline` as the baseline source of truth and add new SSL/probe artifact folders beside it.
- `CLI/public interfaces`: add explicit training entrypoints for `train_supervised`, `train_contrastive`, `train_probe`, and one comparison/report script. Each entrypoint must accept artifact root, split root, seed, batch size, learning rate, epochs, and output directory as flags so runs are reproducible and scriptable.

## Implementation Details

- `Data layer`: create one loader that joins `metadata.csv` rows with cached chunk payloads by `window_id`, validates every payload has 12 channels and 60 timesteps, and fails fast on missing or duplicate IDs.
- `Dataset variants`: implement three dataset modes from the same cached data source: supervised fusion, contrastive pair, and frozen-embedding probe.
- `Subset selection`: centralize label-fraction sampling so supervised baselines and probes cannot silently diverge in how they choose training examples.
- `Training loops`: centralize seed handling, device detection, checkpointing, early stopping, metric computation, and confusion-matrix generation so supervised and probe results are directly comparable.
- `Probe training`: freeze encoder weights completely, precompute or on-the-fly compute embeddings, and train only a linear classifier. Do not fine-tune encoders in this phase.
- `Comparison outputs`: generate one summary table covering `watch_full`, `watch_10pct`, `contrastive_pair_probe_10pct`, `contrastive_pair_probe_100pct`, `contrastive_phone_probe_10pct`, `contrastive_phone_probe_100pct`, `contrastive_watch_probe_10pct`, and `contrastive_watch_probe_100pct`.
- `Phase boundary`: explicitly defer UMAP/t-SNE, device-classification probes, and inter-/intra-device distance analysis until after the core SSL result is working end to end.

## Test Plan

- `Data contract tests`: verify metadata row count matches unique cached `window_id`s per split, every loaded tensor has the expected shape, and phone/watch slices exactly partition the 12-channel fusion tensor.
- `Split integrity tests`: verify no subject leakage across train/val/test and confirm the loader uses the manifest-defined split membership.
- `Sampling tests`: verify the `10%` fraction selector is deterministic under seed `42`, uses at least one example per present class/subject bucket when possible, and is shared by baseline and probe pipelines.
- `Model tests`: verify encoder output shape is `[B, 256]`, projection head output shape is `[B, 64]`, and InfoNCE loss decreases on a tiny synthetic paired batch.
- `Freeze tests`: verify probe training leaves encoder weights unchanged and only updates the linear head.
- `Metric tests`: verify confusion matrix dimensions match the manifest label map and macro F1 matches the matrix-derived calculation used in the baseline code.
- `Smoke tests`: run a one-epoch supervised fusion training job, a one-epoch contrastive pretraining job, and a one-epoch paired probe on a tiny subset to prove the full pipeline executes without shape, serialization, or split errors.
- `Acceptance criteria`: the phase is complete when the repo can reproducibly produce the watch supervised baseline, one contrastive checkpoint, and all paired/phone/watch probes at `10%` and `100%`, with metrics and artifacts saved in a standardized layout.

## Assumptions and Defaults

- The current cached windows are the official dataset for phase 2 and are treated as valid four-stream paired windows.
- The current `20 Hz` / `3 s` / `1 s` preprocessing is fixed for this phase.
- The existing baseline code is functionally correct enough to use as a reference, but its logic should be absorbed into reusable modules rather than duplicated.
- Separate phone and watch encoders are the default, not shared weights.
- The main story for this milestone is "contrastive pretraining improves downstream subject-generalized activity recognition under limited labels," not "complete representation analysis."
- The canonical supervised comparator is the full-label watch-only model, with phone-only and fusion runs retained as secondary baselines for modality and fusion analysis.
- Single-device probes are included because they directly support the device-invariance/usefulness story, but they are secondary to the paired probe headline result.
- The first milestone does not require rerunning every historical notebook result, polishing the README, or doing a total repository refactor.
