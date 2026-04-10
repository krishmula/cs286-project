# Implementation Plan: Phase 2 Contrastive Pipeline

## Overview

This plan breaks Phase 2 into small, verifiable tasks that build a contrastive pretraining and linear-probe pipeline on top of the existing supervised baseline artifacts. The implementation order follows the dependency graph: first define the reusable training/data interfaces, then prove the cached windows can support the new flow, then add supervised and contrastive runners, and finally add comparison/reporting and smoke-test coverage.

## Architecture Decisions

- Keep the current cached windows in `artifacts/windows` as the phase-2 source of truth.
- Build a reusable `src/` package, but keep current `scripts/` and `notebooks/` intact.
- Derive `x_phone` and `x_watch` by slicing the cached 12-channel `x_fusion` tensor.
- Use separate phone and watch encoders with the existing 1D CNN backbone shape.
- Treat fused supervised training as the canonical comparator and paired probe results as the headline SSL result.
- Defer UMAP/t-SNE and device-invariance analyses until the core training/evaluation loop is stable.

## Dependency Graph

```text
Task 1: Runtime + config scaffolding
    |
    +--> Task 2: Window loader + dataset contracts
            |
            +--> Task 3: Shared training/evaluation utilities
                    |
                    +--> Task 4: Canonical supervised fusion runner
                    |
                    +--> Task 5: Contrastive model + loss
                            |
                            +--> Task 6: Contrastive training entrypoint
                                    |
                                    +--> Task 7: Linear probe pipeline
                                            |
                                            +--> Task 8: Comparison reporting
                                                    |
                                                    +--> Task 9: Automated smoke tests
```

## Task List

### Phase 1: Foundation

- [ ] Task 1: Create the reusable phase-2 package skeleton
- [ ] Task 2: Build the cached-window loader and dataset contracts
- [ ] Task 3: Add shared training, metrics, and checkpoint utilities

## Task 1: Create the reusable phase-2 package skeleton

**Description:** Create the initial `src/` package layout and define the common configuration/runtime surface for all phase-2 scripts. This task establishes the module boundaries so later tasks can plug into the same interfaces instead of duplicating logic across scripts.

**Acceptance criteria:**

- [ ] A `src/` package exists with clear homes for data loading, models, training utilities, and CLI entrypoints.
- [ ] One shared config/runtime module defines defaults for seed, device selection, artifact roots, and common hyperparameters used across supervised, contrastive, and probe runs.
- [ ] The package can be imported locally without modifying existing `scripts/` behavior.

**Verification:**

- [ ] Import check succeeds: `python -c "import src"`
- [ ] CLI/help check succeeds for the shared runtime module or entrypoint stubs.
- [ ] Manual check: the new package layout reflects the architecture in `docs/phase2_contrastive_plan.md`.

**Dependencies:** None

**Files likely touched:**

- `src/__init__.py`
- `src/config.py`
- `src/runtime.py`

**Estimated scope:** Medium: 3-5 files

## Task 2: Build the cached-window loader and dataset contracts

**Description:** Implement a single loader that joins split metadata with cached payload chunks and exposes consistent sample records for `x_fusion`, `x_phone`, and `x_watch`. This is the highest-risk data task and should fail fast if there are missing IDs, malformed tensors, or split inconsistencies.

**Acceptance criteria:**

- [ ] The loader can read `train`, `val`, and `test` metadata and payload chunks from `artifacts/windows`.
- [ ] Each sample exposes `window_id`, `subject_id`, `label_idx`, `split`, `x_fusion`, `x_phone`, and `x_watch` with shapes `[12, 60]`, `[6, 60]`, and `[6, 60]`.
- [ ] The loader raises explicit errors for missing payloads, duplicate IDs, or unexpected tensor shapes.

**Verification:**

- [ ] Data contract test passes against one sample from each split.
- [ ] Split summary check reproduces manifest counts from `artifacts/windows/manifest.json`.
- [ ] Manual check: slicing `x_fusion[0:6]` and `x_fusion[6:12]` yields the phone/watch views used by the dataset layer.

**Dependencies:** Task 1

**Files likely touched:**

- `src/data/window_loader.py`
- `src/data/datasets.py`
- `tests/test_window_loader.py`

**Estimated scope:** Medium: 3-5 files

## Task 3: Add shared training, metrics, and checkpoint utilities

**Description:** Centralize the training-loop primitives that every later runner will need: deterministic seeding, device detection, metric computation, confusion matrices, checkpoint save/load, and early-stopping helpers. This keeps the supervised, contrastive, and probe entrypoints aligned and prevents metric drift.

**Acceptance criteria:**

- [ ] One shared utility layer exists for seed control, device selection, metric aggregation, and checkpoint serialization.
- [ ] The metrics layer can compute accuracy, macro F1, confusion matrix, and per-subject accuracy using the current label map.
- [ ] The checkpoint format records enough metadata to reproduce a run later.

**Verification:**

- [ ] Unit tests pass for confusion matrix and macro F1 calculation.
- [ ] A dummy checkpoint save/load round trip succeeds.
- [ ] Manual check: the utilities can support both classification and contrastive runners without duplicated logic.

**Dependencies:** Task 2

**Files likely touched:**

- `src/training/utils.py`
- `src/training/metrics.py`
- `tests/test_metrics.py`

**Estimated scope:** Medium: 3-5 files

## Checkpoint: Foundation

- [ ] All foundation tests pass.
- [ ] The loader can iterate all three splits without shape or key errors.
- [ ] The repo now has one canonical place for data contracts and shared training utilities.
- [ ] Review the module boundaries before model-specific implementation begins.

### Phase 2: Core Training Paths

- [ ] Task 4: Port the canonical fused supervised runner
- [ ] Task 5: Implement the contrastive encoder and InfoNCE loss
- [ ] Task 6: Build the contrastive pretraining runner
- [ ] Task 7: Build the frozen linear-probe runner

## Task 4: Port the canonical fused supervised runner

**Description:** Rebuild the current supervised baseline as a reusable `src/` entrypoint that depends on the new loader and shared utilities instead of the ad hoc baseline script. The watch-only run becomes the primary comparator for all later SSL results, while phone-only and fusion remain comparison ablations.

**Acceptance criteria:**

- [ ] A supervised training entrypoint trains the watch-only baseline using the new package modules.
- [ ] The entrypoint supports at least `100%` and `10%` labeled training fractions using the same deterministic subset sampler that probes will use.
- [ ] The run writes checkpoint, metrics JSON, confusion matrix, and per-subject accuracy outputs to a standardized artifact location.

**Verification:**

- [ ] Smoke run succeeds for one epoch on a tiny subset.
- [ ] Full-data metric output matches the existing baseline format closely enough to compare results.
- [ ] Manual check: the watch-only run is clearly documented as the baseline source of truth for phase 2.

**Dependencies:** Task 3

**Files likely touched:**

- `src/models/encoder.py`
- `src/train_supervised.py`
- `tests/test_supervised_smoke.py`

**Estimated scope:** Medium: 3-5 files

## Task 5: Implement the contrastive encoder and InfoNCE loss

**Description:** Add the model pieces specific to self-supervised learning: phone and watch encoders, projection heads, and the symmetric InfoNCE objective. Keep this task focused on model shape and loss behavior so the training entrypoint can be built on top of something already testable.

**Acceptance criteria:**

- [ ] The contrastive model produces `256`-dimensional encoder features and normalized `64`-dimensional projection vectors for both views.
- [ ] The InfoNCE implementation is symmetric and uses in-batch negatives with a configurable temperature.
- [ ] The model code cleanly separates encoder outputs from projection-head outputs so probes can reuse frozen encoders later.

**Verification:**

- [ ] Shape tests pass for encoder and projector outputs.
- [ ] A tiny synthetic batch produces a finite contrastive loss.
- [ ] Manual check: the model interface exposes frozen encoder embeddings without needing projection heads at inference time.

**Dependencies:** Task 3

**Files likely touched:**

- `src/models/encoder.py`
- `src/models/heads.py`
- `tests/test_contrastive_model.py`

**Estimated scope:** Medium: 3-5 files

## Task 6: Build the contrastive pretraining runner

**Description:** Create the end-to-end pretraining entrypoint that consumes paired phone/watch windows from the training split, applies default augmentations, optimizes InfoNCE, and writes reusable encoder checkpoints. This is the first full SSL training slice.

**Acceptance criteria:**

- [ ] A contrastive training entrypoint runs on paired train windows and uses the validation split for checkpoint selection.
- [ ] Default augmentations include jitter, mild scaling, and short temporal masking, with CLI flags to override or disable them.
- [ ] The saved checkpoint includes both encoder weights, projector weights, label map metadata, and enough run config to drive later probes.

**Verification:**

- [ ] One-epoch smoke training succeeds without shape or serialization errors.
- [ ] Loss is finite and logged for both train and validation steps.
- [ ] Manual check: the output checkpoint is directly consumable by the probe pipeline without extra conversion.

**Dependencies:** Tasks 3 and 5

**Files likely touched:**

- `src/data/augmentations.py`
- `src/train_contrastive.py`
- `tests/test_contrastive_smoke.py`

**Estimated scope:** Medium: 3-5 files

## Task 7: Build the frozen linear-probe runner

**Description:** Implement the downstream evaluation path that loads frozen contrastive encoders, computes paired and single-device embeddings, trains only linear heads, and evaluates performance on the held-out test subjects. This task turns pretraining into the first project result.

**Acceptance criteria:**

- [ ] The probe runner supports `pair`, `phone`, and `watch` evaluation modes.
- [ ] The probe runner supports `10%` and `100%` labeled fractions using the same subset-selection logic as the supervised runner.
- [ ] Encoder weights remain frozen during probe training and the run writes metrics and checkpoints in the standardized artifact layout.

**Verification:**

- [ ] A freeze test confirms only linear-head weights change during probe training.
- [ ] A one-epoch probe smoke run succeeds in all three evaluation modes.
- [ ] Manual check: the paired probe uses concatenated `[h_phone ; h_watch]` embeddings as the default headline setup.

**Dependencies:** Tasks 4, 5, and 6

**Files likely touched:**

- `src/training/probe.py`
- `src/train_probe.py`
- `tests/test_probe_smoke.py`

**Estimated scope:** Medium: 3-5 files

## Checkpoint: Core Training Paths

- [ ] Supervised, contrastive, and probe entrypoints all run end to end on smoke settings.
- [ ] The artifact layout is consistent across baseline, pretraining, and probe runs.
- [ ] The canonical comparator and the headline paired probe are both reproducible from CLI entrypoints.
- [ ] Review early results before expanding the experiment surface.

### Phase 3: Reporting and Hardening

- [ ] Task 8: Add standardized experiment comparison outputs
- [ ] Task 9: Add automated smoke coverage and runbook updates

## Task 8: Add standardized experiment comparison outputs

**Description:** Build the reporting layer that reads baseline and probe outputs and emits one compact comparison table for the core experiments. This task prevents result drift and gives the project one standard place to answer “how did the SSL model compare?”

**Acceptance criteria:**

- [ ] A comparison script can read canonical baseline outputs and probe outputs from artifact folders.
- [ ] The script writes a single machine-readable summary and one human-readable table covering the agreed core runs.
- [ ] Missing experiment outputs fail with clear messages instead of silently omitting rows.

**Verification:**

- [ ] Comparison script runs successfully on smoke or partial outputs with explicit handling for missing artifacts.
- [ ] Summary rows include `watch_full`, `watch_10pct`, and all agreed probe variants.
- [ ] Manual check: the output is suitable for direct use in notebooks or the final report.

**Dependencies:** Tasks 4 and 7

**Files likely touched:**

- `src/reporting/compare_runs.py`
- `scripts/run_comparison_baselines.py`
- `tests/test_compare_runs.py`

**Estimated scope:** Medium: 3-5 files

## Task 9: Add automated smoke coverage and a phase-2 runbook

**Description:** Add the final test harness and operational documentation so future implementation sessions can validate the pipeline quickly and run the core experiment set without rediscovering commands or assumptions. This closes the loop from planning to reliable execution.

**Acceptance criteria:**

- [ ] Smoke tests cover the loader, supervised runner, contrastive runner, and probe runner.
- [ ] One short runbook documents the command sequence for reproducing the core milestone experiments.
- [ ] The runbook states the expected artifact outputs and checkpoint locations for each stage.

**Verification:**

- [ ] The smoke test suite passes from a clean environment with cached windows present.
- [ ] The documented command sequence works as written on smoke settings.
- [ ] Manual check: another engineer could start implementation or verification from the runbook alone.

**Dependencies:** Tasks 2, 4, 6, 7, and 8

**Files likely touched:**

- `tests/test_pipeline_smoke.py`
- `docs/phase2_contrastive_plan.md`
- `docs/phase2_runbook.md`

**Estimated scope:** Medium: 3-5 files

## Checkpoint: Complete

- [ ] All acceptance criteria across tasks are met.
- [ ] Smoke tests pass for loader, supervised training, contrastive training, probe training, and run comparison.
- [ ] The repo has one documented command path from cached windows to final comparison metrics.
- [ ] Phase 2 is ready for implementation review or agent delegation.

## Parallelization Opportunities

- Safe to parallelize after Task 3:
  - Task 4 and Task 5 can proceed in parallel because one builds the canonical supervised path and the other builds the contrastive model internals.
- Safe to parallelize after Task 7:
  - Task 8 and parts of Task 9 can proceed in parallel because reporting and final smoke/runbook work depend on completed outputs, not shared training code changes.
- Must remain sequential:
  - Tasks 1 through 3.
  - Task 6 after Task 5.
  - Task 7 after Task 6.

## Risks and Mitigations

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Cached windows do not fully match the assumed shape or metadata contract | High | Put the strict validation and fail-fast checks in Task 2 before any training code is written |
| Baseline and probe label-fraction sampling drift apart | High | Centralize subset selection in shared utilities before building either runner |
| Contrastive checkpoints omit metadata needed by probes | Medium | Define and test checkpoint contents in Task 3 and Task 6 |
| Too much logic remains split between old scripts and new modules | Medium | Treat old scripts as wrappers or references only once the `src/` entrypoints exist |
| Experiment outputs become inconsistent across stages | Medium | Standardize artifact naming and comparison reporting in Tasks 4, 7, and 8 |

## Open Questions

- No blocking product or research questions remain for the first implementation milestone.
- If the first contrastive result is weak, the next decision point should be whether to tune augmentations, temperature, or label fractions before expanding into embedding analysis.

## Verification Checklist

- [ ] Every task has explicit acceptance criteria.
- [ ] Every task has a verification section.
- [ ] Dependencies are ordered bottom-up from foundations to reporting.
- [ ] No single task requires more than a focused implementation session.
- [ ] Checkpoints exist between major phases.
- [ ] The breakdown matches `docs/phase2_contrastive_plan.md`.
