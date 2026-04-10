# Phone–Watch Contrastive Representation Learning for Wearable Activity Recognition

## 1. Project Overview

This project proposes a graduate-level final project for a Wearable AI & mHealth course that uses a smartphone–smartwatch inertial dataset with row-level activity labels, timestamps, per-subject files, and four sensor streams (phone/watch × accelerometer/gyroscope).
The core idea is to treat the phone and watch as two synchronized views of the same underlying human activity and apply contrastive self-supervised learning to learn a shared, device-invariant representation that supports label-efficient, subject-generalized activity recognition.[1][2][3]

The project is designed with two layers:

- A **supervised baseline layer** that compares watch-only, phone-only, and multimodal fusion models for subject-generalized activity recognition, with watch-only currently serving as the main supervised benchmark.
- An **ambitious contrastive extension** where synchronized phone–watch windows are used as positive pairs in an InfoNCE-style objective, enabling:
  - Device-invariant embeddings (phone vs watch).
  - Improved generalization to unseen subjects.
  - Strong label efficiency (good performance with few labels).

The dataset structure (per-subject files, timestamps, phone vs watch, accel vs gyro) is explicitly leveraged in the model design, training strategy, and evaluation protocol.

## 2. Motivation and mHealth Relevance

### 2.1 Why representation learning for wearables?

Inertial sensors from smartphones and smartwatches are widely used for activity recognition, fall detection, mobility monitoring, and early detection of health events.[4][5]
However, annotating wearable time-series data is expensive and intrusive, making **label efficiency** a central concern for real-world mHealth applications.[6][3]

Self-supervised and contrastive learning have emerged as powerful approaches to exploit large volumes of unlabeled wearable data, learning robust features that transfer well to downstream tasks with limited labeled data.[2][7][3]
Recent work demonstrates that contrastive objectives tailored to inertial data can significantly improve performance and robustness in human activity recognition (HAR), especially under cross-subject evaluation.[8][9][2]

### 2.2 Why phone vs watch cross-view learning?

Modern users often carry both a smartphone and a smartwatch, but their placement, orientation, and availability vary throughout the day.[10][5]
This leads to several practical challenges:

- **Device heterogeneity:** Phone and watch experience different motion patterns for the same activity (e.g., walking with phone in pocket vs watch on wrist).
- **Device availability:** Sometimes only one device is present (phone at desk, watch on wrist) or one sensor modality fails or is disabled.
- **Subject generalization:** Inter-subject differences in gait, posture, and device wearing style can degrade model performance when deployed to new users.[11][3]

By treating phone and watch as **two views of the same time-aligned activity**, contrastive learning can directly encourage the model to learn representations that are:

- Invariant to device location and kinematic differences.
- Rich enough to support generalization across subjects.
- Effective even when only one device is available at inference time.

These goals align closely with current trends in mHealth and wearable AI that seek user-generalizable and device-agnostic models.[3][1][11]

## 3. Dataset and Task Formulation

### 3.1 Dataset structure and signals

The project uses a smartphone–smartwatch inertial dataset where data are organized as follows (per the Kaggle description and provided examples):

- Folders:
  - `train/` and `test/` for subject-disjoint splits.
  - Within each split: `phone/accel/`, `phone/gyro/`, `watch/accel/`, `watch/gyro/`.
- Files:
  - Per-subject text files, e.g. `data_1600_accel_phone.txt`, `data_1600_gyro_watch.txt`.
- Each row (example from accelerometer):
  - `subject_id, activity_label, timestamp, x, y, z;`

Thus each sensor stream (e.g., `phone/accel`) provides a timestamped 3D time series per subject and per activity, with samples recorded from either phone or watch and from accelerometer or gyroscope.
This matches the structure of other smartphone–smartwatch HAR datasets used in research and competitions.[12][13][14]

### 3.2 Windowed activity recognition task

The project frames the downstream task as **window-level activity recognition** with subject-level generalization:

- **Inputs:** Fixed-length windows (e.g., 2–5 seconds) of multichannel inertial data from phone and watch (accel + gyro) aligned by timestamp.
- **Labels:** The majority activity label within each window; windows spanning transitions are discarded or handled separately.
- **Train/test split:** Train on `train/` subjects only; test exclusively on `test/` subjects to measure generalization.

This formulation mirrors standard practices in HAR literature, where sliding windows over raw sensor data are used to classify activities, and cross-subject splits are used to simulate deployment to new users.[15][10][3]

## 4. Baseline Supervised Multimodal Fusion Model

### 4.1 Input representation

For each subject, four synchronized time series exist:

- Phone accelerometer: \(a_p(t) = (x, y, z)\).
- Phone gyroscope: \(g_p(t) = (x, y, z)\).
- Watch accelerometer: \(a_w(t) = (x, y, z)\).
- Watch gyroscope: \(g_w(t) = (x, y, z)\).

Using timestamps, the project constructs windows of length \(T\) seconds with stride \(S\) seconds (e.g., \(T = 2\), \(S = 1\)).
Each stream is resampled to a fixed number of time steps \(L\) per window, resulting in a tensor of shape `(channels, L)` per window.

The baseline supervised model concatenates the channels from all four streams into a single multichannel input for each window, yielding an input tensor of shape `(C_total, L)` where \(C_total = 12\) (4 × 3 axes).

### 4.2 Model architecture

A straightforward yet strong baseline is a **1D convolutional neural network** operating along the temporal dimension:[15][4]

- Several Conv1D → BatchNorm → ReLU → Pooling blocks.
- Optional temporal attention or squeeze–excitation blocks.
- Global average pooling over time to produce a fixed-length feature vector.
- A final fully connected layer for activity classification.

This architecture has been shown to perform well on HAR benchmarks while being relatively easy to implement and train.[5][4][15]

### 4.3 Baseline training and evaluation

The supervised baseline is trained on labeled windows from training subjects using cross-entropy loss, with standard regularization (e.g., dropout, weight decay).
The main evaluation is done on windows from test subjects, reporting:

- Overall accuracy and macro F1 score.
- Confusion matrix across activities.
- Per-subject performance distribution (e.g., boxplots) to understand inter-subject variability.

This baseline not only provides a performance reference but also verifies the correctness of the data pipeline and windowing.

## 5. Contrastive Cross-View Representation Learning

### 5.1 Conceptual idea

The central technical contribution is a **cross-view contrastive learning framework** that uses synchronized phone and watch windows as positive pairs.
This follows recent multimodal and multiview contrastive approaches in wearable HAR, where different sensor locations or modalities are used as views to learn robust shared representations.[7][16][1][8]

For each time window:

- **View A (phone view):** Concatenation of phone accelerometer and gyroscope channels.
- **View B (watch view):** Concatenation of watch accelerometer and gyroscope channels.

The project trains two encoders (phone encoder and watch encoder) with an InfoNCE objective to bring embeddings of corresponding phone–watch windows closer while pushing apart embeddings of non-corresponding windows.

### 5.2 Encoders and projection heads

Each view has its own encoder network with the same architecture (e.g., 1D CNN) but separate parameters:

- Phone encoder \(f_p\) maps phone windows to latent vectors.
- Watch encoder \(f_w\) maps watch windows to latent vectors.

On top of each encoder, a small projection head (e.g., a 2-layer MLP) maps latent vectors to a contrastive embedding space where the InfoNCE loss is applied, following common practice in SimCLR-like frameworks.[8][7]

### 5.3 InfoNCE loss with in-batch negatives

For a batch of \(N\) synchronized phone–watch window pairs \((x^p_i, x^w_i)\), the model computes projections \(z^p_i\) and \(z^w_i\) and normalizes them.
The InfoNCE loss is applied symmetrically:

- Each \(z^p_i\) treats \(z^w_i\) as its positive and \(z^w_j\) for \(j \neq i\) as negatives.
- Each \(z^w_i\) treats \(z^p_i\) as its positive and \(z^p_j\) for \(j \neq i\) as negatives.

A temperature parameter controls the sharpness of the similarity distribution.
This setup is directly aligned with established contrastive frameworks applied to HAR and wearable signals.[9][2][8]

### 5.4 Training regime

Contrastive pretraining uses **all** windows from training subjects, ignoring labels.
Data augmentations appropriate for inertial time series may be applied to each view, such as:

- Small Gaussian jittering in acceleration/gyro values.
- Scaling (multiplying by a factor close to 1).
- Time masking or cropping.

These augmentations have been shown to be effective in self-supervised HAR to encourage robustness to noise and minor temporal distortions.[2][7][8]

The encoders and projection heads are trained to minimize the contrastive loss.
After pretraining, the projection heads are discarded; the encoders’ intermediate representations serve as general-purpose embeddings for downstream tasks.

## 6. Downstream Evaluation and Label Efficiency

### 6.1 Linear evaluation protocol

To assess the quality of the learned representation, a **linear evaluation** protocol is used, similar to those in self-supervised learning literature.[3][8][2]

- Freeze the pretrained encoders \(f_p\) and \(f_w\).
- For each window, compute embeddings (e.g., from the penultimate layer) for phone and watch views; combine them (e.g., concatenate or average) into a single feature vector.
- Train a simple linear classifier or shallow MLP on top of these embeddings using labeled windows from training subjects.

Performance on test subjects is then compared to the purely supervised baseline that was trained end-to-end from scratch.

### 6.2 Label-efficiency experiments

A key experiment is to measure how performance scales with the amount of labeled data:

- Sample different fractions of labeled windows from training subjects, e.g., 1%, 5%, 10%, 25%, 100%.
- Train separate linear classifiers for each fraction on top of the frozen embeddings.
- Compare accuracy/F1 on test subjects to supervised baselines trained directly on raw windows with the same label fractions.

Existing work shows that self-supervised pretraining can significantly improve label efficiency in HAR and other health-related wearable tasks, especially when labeled data are scarce.[17][6][2][3]
The expectation is that the contrastive phone–watch model will retain reasonable performance even when only a small subset of labeled windows is available, while the supervised model will degrade more sharply.

### 6.3 Subject-generalization evaluation

Because the train/test split is by subject, the evaluation inherently measures **generalization to unseen individuals**.
The project can further analyze:

- Per-subject accuracy distributions for the supervised baseline vs contrastive-pretrained model.
- Whether certain subjects (e.g., those with atypical movement patterns) benefit more from the learned representation.

Recent work emphasizes the importance of user-generalizable models in wearable HAR, including methods designed explicitly for subject-invariant contrastive learning.[18][11][3]
This project contributes empirical evidence on how cross-view pretraining affects subject-level robustness.

## 7. Embedding Analysis and Visualization

### 7.1 Device invariance in embedding space

One of the most compelling qualitative results is a visualization of the learned embedding space:

- Use UMAP or t-SNE to project embeddings into 2D.
- Color points by activity label.
- Use shape, border color, or separate panels to indicate device (phone vs watch).

If the cross-view contrastive learning is successful, points from phone and watch corresponding to the same activity should overlap or form tightly intertwined clusters.
In contrast, a supervised model trained only for classification might show stronger device-specific separation.

### 7.2 Quantitative domain-invariance metrics

Beyond visualization, the project can compute simple metrics to quantify how device-invariant the embeddings are:

- Train a small probe classifier to predict device (phone vs watch) from embeddings; lower accuracy indicates greater device invariance.
- Compute intra- vs inter-device distances within the same activity cluster.

These analyses connect directly to the mHealth goal of having representations that are robust to device placement and hardware differences.[1][17][3]

## 8. Relation to Prior Work

### 8.1 Self-supervised HAR and wearable SSL

Several recent papers and surveys highlight the promise of self-supervised and contrastive learning for HAR and health wearables:

- Contrastive self-supervised learning tailored to sensor-based HAR, showing improved performance under cross-subject and semi-supervised settings.[9][8][2]
- Frameworks that apply augmentations, temporal cropping, and multi-view contrastive losses to inertial sensor datasets.[7][8]
- Surveys of multimodal wearable HAR emphasizing representation learning and cross-modal fusion.[19][1]
- Applications of SSL to health-related decoding from wearables, reducing annotation demands.[6][17][3]

This project aligns with these directions but focuses specifically on **phone–watch cross-view alignment** in a single dataset with clear device separation.

### 8.2 Multimodal and multiview contrastive HAR

Recent work on multimodal HAR has applied contrastive learning across sensor locations and modalities, e.g., inertial + audio or multiple wear locations, to learn richer, more transferable features.[16][5][1]
CLEAR and related multimodal contrastive frameworks demonstrate how aligning representations from different modalities can improve robustness and cross-domain generalization.[16]

This project mirrors those ideas in a simpler setting (only phone/watch inertial) but retains the core scientific question: does cross-view contrastive alignment lead to device-invariant representations and improved label efficiency?

### 8.3 User-generalizable models

Recent studies highlight the challenge of building user-generalizable HAR models and propose approaches such as subject-invariant contrastive learning and personalized adaptation mechanisms.[11][18][3]
The subject-disjoint evaluation in this project supports a direct comparison between:

- Standard supervised fusion.
- Cross-view contrastive pretraining + linear probing.

This makes the project relevant to ongoing conversations about fairness, personalization, and robustness in mHealth.

## 9. Project Scope, Complexity, and Deliverables

### 9.1 Technical complexity

The project is technically rich but manageable within a 2–3 week total effort (with 2 days for baseline and 10–15 days for the full pipeline):

- Nontrivial data handling: aligning four sensor streams by timestamp, building windowed datasets, and ensuring subject-disjoint splits.
- A solid supervised multimodal baseline with subject-generalization evaluation.
- A self-supervised cross-view contrastive training phase.
- Label-efficiency experiments and embedding analysis.

This complexity is comparable to that of published SSL HAR prototypes and is appropriate for a graduate wearable AI project.[8][2][7][3]

### 9.2 Main deliverables

The project should produce:

- **Codebase:**
  - Data loading and windowing scripts explicitly handling `phone/accel`, `phone/gyro`, `watch/accel`, `watch/gyro` directories and per-subject files.
  - Training scripts for supervised baseline and contrastive pretraining.
  - Evaluation scripts for label efficiency and subject-wise performance.
- **Quantitative results:**
  - Baseline vs contrastive-pretrained accuracy/F1 on test subjects.
  - Label-efficiency curves.
  - Device-invariance metrics (optional but desirable).
- **Qualitative results:**
  - Embedding plots (UMAP/t-SNE) showing activity clusters and phone vs watch samples.
- **Report and presentation:**
  - Clear motivation grounded in mHealth challenges.
  - Methodology with diagrams of the cross-view contrastive setup.
  - Discussion of limitations and future directions.

### 9.3 mHealth narrative

The final narrative can emphasize:

- **Problem:** Wearable mHealth systems need models that generalize across users and devices, but labeled data are scarce.
- **Approach:** Use self-supervised contrastive learning on synchronized phone–watch signals to learn device-invariant activity representations.
- **Findings:** Contrastive pretraining improves label efficiency and subject-generalization relative to a supervised baseline; embeddings show overlapping phone/watch clusters for the same activity.
- **Implications:** Such representations could be reused across multiple downstream mHealth tasks and help reduce annotation burden in future large-scale studies.[17][1][6][3]

This connects the technical work back to the broader goals of Wearable AI and mobile health.

## 10. Extensions and Future Work

If time permits, several natural extensions can further enhance the project:

- **Modality-dropout robustness:** Combine cross-view contrastive learning with modality-dropout so the model remains strong when only phone or only watch data are available.
- **Personalized adaptation:** Explore lightweight subject-specific fine-tuning on top of the learned embeddings, examining per-subject gains.[18][11]
- **Alternative SSL objectives:** Compare cross-view contrastive learning with other self-supervised methods such as masked reconstruction or predictive coding tailored to inertial data.[2][7]
- **Deployment considerations:** Estimate computational cost and memory footprint of the encoders for on-device deployment on smartphones or smartwatches.

These extensions are not required for the core project but offer a path to additional depth if time allows.

---

This report defines the conceptual and methodological foundations of the project.
A separate implementation plan can detail the concrete steps, timelines, and engineering decisions (model hyperparameters, code structure, and day-by-day milestones) needed to execute this work within the given time frame.
