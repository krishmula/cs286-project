# Baseline CNN Architecture (SupervisedHARModel)

The supervised baseline is a 3-block 1D CNN encoder followed by a linear classification head, trained end-to-end with cross-entropy loss.

## Architecture Diagram

```mermaid
flowchart TD
    INPUT["Input\n[B, C, L]\nC = 6 (phone or watch)\nor 12 (fusion)\nL = sequence length"]

    subgraph ENCODER["TimeSeriesEncoder"]
        direction TB

        subgraph BLOCK1["Conv Block 1"]
            C1["Conv1d\n(C → 64, k=5, s=1, p=2)"]
            BN1["BatchNorm1d(64)"]
            R1["ReLU"]
            MP1["MaxPool1d(2)"]
            C1 --> BN1 --> R1 --> MP1
        end

        subgraph BLOCK2["Conv Block 2"]
            C2["Conv1d\n(64 → 128, k=5, s=1, p=2)"]
            BN2["BatchNorm1d(128)"]
            R2["ReLU"]
            MP2["MaxPool1d(2)"]
            C2 --> BN2 --> R2 --> MP2
        end

        subgraph BLOCK3["Conv Block 3"]
            C3["Conv1d\n(128 → 256, k=3, s=1, p=1)"]
            BN3["BatchNorm1d(256)"]
            R3["ReLU"]
            AAP["AdaptiveAvgPool1d(1)"]
            C3 --> BN3 --> R3 --> AAP
        end

        SQ["squeeze(-1)"]

        BLOCK1 --> BLOCK2 --> BLOCK3 --> SQ
    end

    subgraph HEAD["Classification Head"]
        direction TB
        DO["Dropout(p=0.2)"]
        FC["Linear(256 → num_classes)"]
        DO --> FC
    end

    LOSS["CrossEntropyLoss"]
    LOGITS["Logits\n[B, num_classes]"]

    INPUT --> ENCODER
    ENCODER -->|"h  [B, 256]"| HEAD
    HEAD --> LOGITS
    LOGITS --> LOSS
```

## Tensor Shape Flow

| Stage | Shape | Notes |
|---|---|---|
| Input | `[B, C, L]` | C=6 (single modality), C=12 (fusion) |
| After Block 1 | `[B, 64, L/2]` | MaxPool halves temporal dim |
| After Block 2 | `[B, 128, L/4]` | MaxPool halves again |
| After Block 3 | `[B, 256, 1]` | AdaptiveAvgPool collapses to 1 step |
| Encoder output `h` | `[B, 256]` | squeeze(-1) removes last dim |
| Logits | `[B, num_classes]` | Linear projection |

## Training Details

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 1e-3 |
| Weight decay | 1e-4 |
| Dropout | 0.2 |
| Loss | CrossEntropyLoss |
| Early stopping metric | macro F1 |
| Gradient clipping | 1.0 |
| Batch size | 256 |
