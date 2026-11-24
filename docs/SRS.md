# Software Requirements Specification (SRS)
## Project: SarcasTone — Multimodal Sarcasm Detection Using Text, Speech Prosody, and Video

---

## 1. Purpose & Scope
### 1.1 Purpose
This document specifies the functional, data, architectural, and evaluation requirements for **SarcasTone**, an end-to-end multimodal sarcasm detection repository built from scratch in PyTorch. The specification is structured to guide multi-file repository generation and execution on **Google Colab (T4 GPU)**.

### 1.2 Scope
SarcasTone detects whether a spoken utterance is sarcastic or non-sarcastic by jointly modeling:
1. **Textual semantics and literal sentiment** (what is said).
2. **Speech prosody and acoustic representations** (how it is said: pitch/F0, energy/intensity, speaking rate, pauses, and vocal affect).
3. **Visual cues** (approved video modality from utterance clips).

The repository provides modular Python packages (`src/`), verification utilities (`scripts/`), configuration files (`configs/`), and two Google Colab entry-point notebooks (`notebooks/`) to extract/cache features and execute reproducible modality ablation experiments.

---

## 2. Target Datasets & Modalities
### 2.1 Primary Datasets
* **MUStARD** and/or **MUStARD++** spoken audiovisual sarcasm corpora.
* Each sample comprises:
  * `utterance_id` (unique identifier)
  * `utterance_text` (transcript of the target utterance)
  * `context_text` (optional preceding dialogue turns)
  * `audio_path` / extracted `.wav` audio track (resampled to 16 kHz mono)
  * `video_path` / `.mp4` clip for visual frame sampling
  * `speaker` identifier (for optional speaker-aware splitting/analysis)
  * `label` (binary target: `1` = sarcastic, `0` = non-sarcastic)

### 2.2 Supported Experimental Configurations
All experiments must run on **identical train/validation/test splits** (or identical cross-validation folds) under the following modality modes:
1. **Text-Only (`text`)**
2. **Speech-Only (`speech` — prosody + acoustic features)**
3. **Video-Only (`video`)**
4. **Multimodal Core (`text_speech` — Text + Speech Prosody/Acoustics)**
5. **Full Multimodal (`text_speech_video` — Text + Speech Prosody/Acoustics + Video)**
6. **Fusion Ablation (`text_speech_concat` vs. `text_speech_pgin`)** comparing naive concatenation against explicit incongruity + gated fusion.

---

## 3. Functional Requirements
### FR-1: Data Loading & Deterministic Splitting
* **FR-1.1:** Ingest dataset metadata JSON/CSV files and resolve corresponding audio and video paths cleanly, logging any missing media files explicitly.
* **FR-1.2:** Generate and persist deterministic stratified train/validation/test split files (e.g., `splits.json`) using a fixed random seed so every modality ablation evaluates on identical samples.
* **FR-1.3:** Provide a verification script (`scripts/verify_splits.py`) confirming zero sample leakage across splits and identical sample counts across all ablation modes.

### FR-2: Offline Feature Extraction & Google Drive Caching
* **FR-2.1:** Extract and cache **Text** embeddings (`microsoft/deberta-v3-base`, configurable to `roberta-base`) per utterance.
* **FR-2.2:** Extract and cache **Explicit Speech Prosody** features (`librosa` / `scipy`):
  * Pitch ($F_0$) summary statistics (mean, std, min, max, dynamic range, slope).
  * Energy / intensity (RMS mean, std, dynamic range).
  * Speaking rate / onset density proxies.
  * Pause / silence statistics (pause ratio, silent segment count/duration).
* **FR-2.3:** Extract and cache **Acoustic** embeddings (`facebook/wav2vec2-base`, configurable to HuBERT base) from 16 kHz audio.
* **FR-2.4:** Extract and cache **Video** frame embeddings (`torchvision.models.resnet50` with `weights="DEFAULT"`, configurable to `google/vit-base-patch16-224`) by uniformly sampling $N$ frames per clip.
* **FR-2.5:** Save all extracted features to a configurable directory (targeting mounted Google Drive on Colab) in `.pt` or `.npz` format with manifest metadata so runtime disconnects do not require re-extraction.

### FR-3: Model Architecture (`PGIN-Sarcasm`)
* **FR-3.1:** Implement the **Prosody-Gated Incongruity Network (`PGIN-Sarcasm`)** in PyTorch supporting dynamic modality masking (`text`, `speech`, `video`).
* **FR-3.2:** Compute explicit **text-vs.-speech incongruity features** (signed difference, absolute difference, element-wise distance, and cosine similarity between projected text and speech representations).
* **FR-3.3:** Apply **learned modality gating** to dynamically weight active modalities before final binary sarcasm classification.

### FR-4: Training, Evaluation & Artifact Logging
* **FR-4.1:** Train models on Google Colab T4 GPU using mixed precision (`torch.amp.autocast`), AdamW optimizer, learning rate scheduling, gradient clipping, and early stopping based on validation F1-score.
* **FR-4.2:** Normalize prosodic features (`StandardScaler`) fit **strictly on the training split** and applied to validation/test splits.
* **FR-4.3:** Compute and save **Accuracy, Precision, Recall, and F1-score** (plus confusion matrices) for every ablation mode in structured JSON and CSV summaries.

### FR-5: Google Colab Execution Entry Points
* **FR-5.1:** Provide `notebooks/01_extract_and_cache_colab.ipynb` to mount Google Drive, verify T4 GPU availability, install `requirements.txt`, and run feature extraction/caching.
* **FR-5.2:** Provide `notebooks/02_train_and_evaluate_colab.ipynb` to run split verification, execute all ablation experiments on the T4 GPU, and render summary metric tables and plots.

---

## 4. Non-Functional Requirements
* **NFR-1 (Compute & Memory):** Must run cleanly within a **Google Colab T4 GPU (16 GB VRAM)** and standard Colab system RAM (~12.7 GB) without out-of-memory errors by operating primarily on cached features during training and batching feature extraction cleanly.
* **NFR-2 (Reproducibility):** Fixed random seeds across `random`, `numpy`, and `torch` (`torch.manual_seed`, deterministic flags where applicable) logged in run configs.
* **NFR-3 (Modularity & Code Quality):** Clean separation between configuration (`configs/`), data/features (`src/data/`, `src/features/`), modeling (`src/models/`), training/evaluation (`src/training/`, `src/experiments/`), and Colab notebooks (`notebooks/`).
* **NFR-4 (Repository Hygiene):** `.gitignore` must exclude raw media files, cached feature tensors, checkpoints, and temporary artifacts so the GitHub repository remains lightweight and clean.

---

## 5. Repository Structure Requirement
```text
SarcasTone/
├── docs/
│   ├── SRS.md
│   └── SRD.md
├── configs/
│   └── default.yaml
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py
│   │   └── collate.py
│   ├── features/
│   │   ├── __init__.py
│   │   ├── text_features.py
│   │   ├── prosody_features.py
│   │   ├── video_features.py
│   │   └── cache_manager.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── incongruity_gate.py
│   │   └── pgin_sarcasm.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── losses.py
│   │   ├── trainer.py
│   │   └── metrics.py
│   └── experiments/
│       └── run_ablations.py
├── notebooks/
│   ├── 01_extract_and_cache_colab.ipynb
│   └── 02_train_and_evaluate_colab.ipynb
├── scripts/
│   └── verify_splits.py
├── requirements.txt
├── .gitignore
└── README.md
```