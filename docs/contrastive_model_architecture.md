# Contrastive Learning Model Architecture (PhoneWatchContrastiveModel)

The contrastive model uses two separate encoders (one per modality) trained with a symmetric InfoNCE loss to align phone and watch representations of the same activity window. The encoder backbone is identical in architecture to the supervised baseline but trained with a different objective.

## Full Training Architecture

```mermaid
flowchart TD
    PHONE_IN["Phone Input\n[B, 6, L]"]
    WATCH_IN["Watch Input\n[B, 6, L]"]

    AUG_P["Augmentations\n(jitter, scale, mask)"]
    AUG_W["Augmentations\n(jitter, scale, mask)"]

    subgraph PHONE_ENC["Phone Encoder (TimeSeriesEncoder)"]
        direction TB
        PC1["Conv1d(6→64, k=5) + BN + ReLU + MaxPool1d(2)"]
        PC2["Conv1d(64→128, k=5) + BN + ReLU + MaxPool1d(2)"]
        PC3["Conv1d(128→256, k=3) + BN + ReLU + AdaptiveAvgPool1d(1)"]
        PC1 --> PC2 --> PC3
    end

    subgraph WATCH_ENC["Watch Encoder (TimeSeriesEncoder)"]
        direction TB
        WC1["Conv1d(6→64, k=5) + BN + ReLU + MaxPool1d(2)"]
        WC2["Conv1d(64→128, k=5) + BN + ReLU + MaxPool1d(2)"]
        WC3["Conv1d(128→256, k=3) + BN + ReLU + AdaptiveAvgPool1d(1)"]
        WC1 --> WC2 --> WC3
    end

    HP["h_phone  [B, 256]"]
    HW["h_watch  [B, 256]"]

    subgraph PHONE_PROJ["Phone Projection Head"]
        direction TB
        PP1["Linear(256 → 128)"]
        PR1["ReLU"]
        PP2["Linear(128 → 64)"]
        PN["L2 Normalize (dim=1)"]
        PP1 --> PR1 --> PP2 --> PN
    end

    subgraph WATCH_PROJ["Watch Projection Head"]
        direction TB
        WP1["Linear(256 → 128)"]
        WR1["ReLU"]
        WP2["Linear(128 → 64)"]
        WN["L2 Normalize (dim=1)"]
        WP1 --> WR1 --> WP2 --> WN
    end

    ZP["z_phone  [B, 64]"]
    ZW["z_watch  [B, 64]"]

    subgraph LOSS["Symmetric InfoNCE Loss"]
        direction TB
        SIM["Similarity Matrix\nz_phone @ z_watch.T / τ\n[B, B]   τ = 0.2"]
        L1["phone→watch CE loss\ncross_entropy(S, targets)"]
        L2["watch→phone CE loss\ncross_entropy(S.T, targets)"]
        LCOMB["0.5 × (L_p→w + L_w→p)"]
        SIM --> L1
        SIM --> L2
        L1 --> LCOMB
        L2 --> LCOMB
    end

    PHONE_IN --> AUG_P --> PHONE_ENC --> HP
    WATCH_IN --> AUG_W --> WATCH_ENC --> HW
    HP --> PHONE_PROJ --> ZP
    HW --> WATCH_PROJ --> ZW
    ZP --> LOSS
    ZW --> LOSS
```

## Linear Probe Evaluation (Downstream Task)

After contrastive pretraining, encoder weights are **frozen** and a single linear layer is trained on the representations.

```mermaid
flowchart TD
    subgraph FROZEN["Frozen Contrastive Model"]
        direction LR
        FPE["Phone Encoder\n(frozen)"]
        FWE["Watch Encoder\n(frozen)"]
    end

    PROBE_IN_P["Phone Input\n[B, 6, L]"]
    PROBE_IN_W["Watch Input\n[B, 6, L]"]

    HP2["h_phone  [B, 256]"]
    HW2["h_watch  [B, 256]"]

    CONCAT["Concatenate\n[B, 512]"]

    subgraph PROBE["Linear Probe Head (trainable)"]
        LIN["Linear(512 → num_classes)\n— or —\nLinear(256 → num_classes)\n(pair / phone / watch mode)"]
    end

    CE["CrossEntropyLoss"]

    PROBE_IN_P --> FPE --> HP2
    PROBE_IN_W --> FWE --> HW2
    HP2 --> CONCAT
    HW2 --> CONCAT
    CONCAT -->|"pair mode"| PROBE
    HP2 -->|"phone mode"| PROBE
    HW2 -->|"watch mode"| PROBE
    PROBE --> CE
```

## Tensor Shape Flow

| Stage | Shape | Notes |
|---|---|---|
| Phone / watch input | `[B, 6, L]` | 6-channel IMU (accel + gyro) |
| After encoder | `[B, 256]` | `h_phone`, `h_watch` — used for probe |
| After projection | `[B, 64]` | `z_phone`, `z_watch` — L2-normalized |
| Similarity matrix | `[B, B]` | All pairwise cosine sims / τ |
| Probe input (pair) | `[B, 512]` | h_phone ‖ h_watch concatenated |
| Probe input (single) | `[B, 256]` | h_phone or h_watch only |
| Logits | `[B, num_classes]` | Linear probe output |

## Data Augmentations (Contrastive Training Only)

| Augmentation | Config |
|---|---|
| Gaussian jitter | σ = 0.01 |
| Random scaling | uniform(0.9, 1.1) per batch |
| Random masking | 10% of steps masked in segments of length 6 |

## Training Details

| Hyperparameter | Contrastive | Linear Probe |
|---|---|---|
| Optimizer | AdamW | AdamW |
| Learning rate | 3e-4 | 1e-3 |
| Weight decay | 1e-4 | 1e-4 |
| Loss | Symmetric InfoNCE | CrossEntropyLoss |
| Temperature (τ) | 0.2 | — |
| Early stopping metric | val_loss | macro F1 |
| Gradient clipping | 1.0 | 1.0 |
| Batch size | 256 | 256 |
| Encoder weights | Trained | Frozen |
