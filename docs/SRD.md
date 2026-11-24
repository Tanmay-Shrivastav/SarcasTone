# Software Requirements & Design Specification (SRD)
## Project: SarcasTone — System Architecture & Module Contracts

---

## 1. System Architecture Overview
**SarcasTone** follows a two-stage pipeline optimized for **Google Colab (T4 GPU)** and **Google Drive** persistence:

1. **Stage 1 — Offline Feature Extraction & Caching (`01_extract_and_cache_colab.ipynb`):**
   * Raw utterances (transcript text, 16 kHz audio, sampled video frames) are passed once through frozen pretrained encoders and prosody extractors on the T4 GPU.
   * Extracted tensors are persisted to Google Drive (`cache_dir`) indexed by `utterance_id`.
2. **Stage 2 — Modular Training & Ablation Evaluation (`02_train_and_evaluate_colab.ipynb`):**
   * Cached feature tensors are loaded using a single locked split file (`splits.json`).
   * The **Prosody-Gated Incongruity Network (`PGIN-Sarcasm`)** trains and evaluates across all modality combinations (`text`, `speech`, `video`, `text_speech`, `text_speech_video`, plus `concat` baseline) and writes structured metrics/plots to `results_dir`.

---

## 2. Pretrained Models & Dependency Stack
### 2.1 Pretrained Backbones
* **Text Encoder:** `microsoft/deberta-v3-base` (hidden dim $D_T = 768$; optional fallback: `roberta-base`).
* **Speech Acoustic Encoder:** `facebook/wav2vec2-base` (hidden dim $D_W = 768$, 16 kHz input).
* **Video Frame Encoder:** `torchvision.models.resnet50(weights="DEFAULT")` with final classification head removed (pooled frame dim $D_V = 2048$; optional fallback: `google/vit-base-patch16-224` with $D_V = 768$).

### 2.2 Python Dependencies (`requirements.txt`)
* `torch>=2.1.0`, `torchvision>=0.16.0`
* `transformers>=4.38.0`, `accelerate>=0.27.0`
* `librosa>=0.10.1`, `soundfile>=0.12.1`, `scipy>=1.11.0`
* `opencv-python-headless>=4.9.0`
* `numpy>=1.24.0`, `pandas>=2.0.0`, `scikit-learn>=1.3.0`
* `pyyaml>=6.0`, `tqdm>=4.66.0`, `matplotlib>=3.8.0`, `seaborn>=0.13.0`

---

## 3. Feature Extraction Specifications
### 3.1 Text Features (`src/features/text_features.py`)
* **Input:** Utterance text (optionally concatenated with context separated by `</s>` / `[SEP]` when `use_context: true`).
* **Extraction:** Attention-masked mean pooling over last hidden states of `microsoft/deberta-v3-base`.
* **Output Tensor:** `text_emb` of shape `(768,)` (`float32`).

### 3.2 Speech Prosody & Acoustic Features (`src/features/prosody_features.py`)
* **Audio Preprocessing:** Load audio, convert to mono, resample to `16000 Hz`, trim leading/trailing silence lightly (`top_db=35`), cap max duration (e.g., `15.0s`) to prevent memory spikes.
* **Explicit Prosodic Vector (`prosody_vec`, shape `(16,)`, `float32`):**
  1. **Pitch ($F_0$) via `librosa.yin` (`fmin=65, fmax=500`):** voiced mean, voiced std, voiced min, voiced max, voiced range ($F_{0,\max} - F_{0,\min}$), linear pitch slope over voiced frames, voiced frame ratio (7 dims).
  2. **Energy / Intensity via `librosa.feature.rms`:** RMS mean, RMS std, RMS max, dynamic range ($\text{max} - \text{mean}$), energy slope (5 dims).
  3. **Speaking Rate & Rhythm:** onset rate per second via `librosa.onset.onset_detect`, zero-crossing rate mean (2 dims).
  4. **Pauses:** silence frame ratio (frames below threshold energy) and total utterance duration in seconds (2 dims).
  * *Note:* NaNs/infs (e.g., unvoiced clips) must be safely replaced with `0.0` prior to saving, and `StandardScaler` must be fit on `train` split only during training.
* **Acoustic Embedding (`audio_emb`, shape `(768,)`, `float32`):**
  * Mean pooling over temporal frames from `facebook/wav2vec2-base` last hidden state.

### 3.3 Video Features (`src/features/video_features.py`)
* **Frame Sampling:** Uniformly sample `num_frames: 8` RGB frames across the clip using OpenCV (`cv2.VideoCapture`), resize to `224x224`, and apply standard ImageNet normalization.
* **Extraction:** Extract per-frame embeddings via `ResNet50` backbone and temporally average across sampled frames.
* **Output Tensor:** `video_emb` of shape `(2048,)` (`float32`). If a clip has unreadable frames, return a zero vector and set `video_valid = 0`.

---

## 4. Model Design: `PGIN-Sarcasm`
### 4.1 Projection & Polarity Heads (`src/models/pgin_sarcasm.py`)
Let unified latent dimension be $d = 256$ (configurable in `configs/default.yaml`):
* **Text Projection:**
  $$h_T = \text{LayerNorm}(\text{GELU}(\text{Linear}(768 \rightarrow d)(x_{\text{text}})))$$
  $$\hat{s}_T = \tanh(\text{Linear}(d \rightarrow 1)(h_T)) \quad \text{(literal text polarity proxy)}$$
* **Speech Projection (Prosody + Acoustic):**
  $$h_{\text{pros}} = \text{LayerNorm}(\text{GELU}(\text{Linear}(16 \rightarrow 64)(x_{\text{prosody}})))$$
  $$h_{\text{wav}} = \text{LayerNorm}(\text{GELU}(\text{Linear}(768 \rightarrow d)(x_{\text{audio}})))$$
  $$h_A = \text{LayerNorm}(\text{GELU}(\text{Linear}(d + 64 \rightarrow d)([h_{\text{wav}} \,;\, h_{\text{pros}}])))$$
  $$\hat{s}_A = \tanh(\text{Linear}(d \rightarrow 1)(h_A)) \quad \text{(vocal affect polarity proxy)}$$
* **Video Projection (when enabled):**
  $$h_V = \text{LayerNorm}(\text{GELU}(\text{Linear}(2048 \rightarrow d)(x_{\text{video}})))$$

### 4.2 Explicit Incongruity & Prosody-Aware Gating (`src/models/incongruity_gate.py`)
When both `text` and `speech` are active:
* **Scalar Polarity Mismatch:**
  $$m_{\text{signed}} = \hat{s}_T - \hat{s}_A, \qquad m_{\text{abs}} = |\hat{s}_T - \hat{s}_A|$$
* **Vector Interaction & Cosine Similarity:**
  $$m_{\text{diff}} = |h_T - h_A|, \qquad m_{\text{prod}} = h_T \odot h_A, \qquad c_{TA} = \cos(h_T, h_A)$$
* **Incongruity Representation ($z_{\text{inc}} \in \mathbb{R}^d$):**
  $$z_{\text{inc}} = \text{GELU}(\text{Linear}(2d + 3 \rightarrow d)([m_{\text{diff}} \,;\, m_{\text{prod}} \,;\, m_{\text{signed}} \,;\, m_{\text{abs}} \,;\, c_{TA}]))$$
* **Modality Gating:**
  * Active branch vectors $\{h_T, h_A, z_{\text{inc}}, h_V\}$ (inactive branches masked to zero) are concatenated and passed through a lightweight gating MLP + Softmax over active branches to produce per-sample modality weights $g = [g_T, g_A, g_{\text{inc}}, g_V]$.
  * Weighted representation:
    $$h_{\text{fused}} = g_T h_T + g_A h_A + g_{\text{inc}} z_{\text{inc}} + g_V h_V$$
* **Fusion Baseline Mode (`fusion_mode="concat"`):**
  * Bypasses incongruity and gating; directly projects concatenated active branch vectors $[h_T \,;\, h_A \,;\, h_V]$ to $d$ for controlled ablation comparison.

### 4.3 Classification Head & Loss (`src/training/losses.py`)
* **Classifier:** `Dropout(0.3) -> Linear(d -> d // 2) -> GELU -> Dropout(0.2) -> Linear(d // 2 -> 1)` producing raw sarcasm logit $\hat{y}$.
* **Objective Function:**
  $$\mathcal{L} = \text{BCEWithLogitsLoss}(\hat{y}, y) + \lambda_{\text{inc}} \cdot \mathcal{L}_{\text{inc}}(\hat{s}_T, \hat{s}_A, y)$$
  where $\lambda_{\text{inc}} \in [0.0, 0.2]$ (default `0.1`) encourages larger polarity divergence $|\hat{s}_T - \hat{s}_A|$ on sarcastic utterances ($y=1$) than non-sarcastic utterances ($y=0$) via margin loss $\max(0, \delta - |\hat{s}_T - \hat{s}_A|)$ for $y=1$.

---

## 5. Configuration Schema (`configs/default.yaml`)
```yaml
project:
  name: SarcasTone
  seed: 42

paths:
  data_root: /content/drive/MyDrive/SarcasTone/data
  cache_dir: /content/drive/MyDrive/SarcasTone/cache
  splits_file: /content/drive/MyDrive/SarcasTone/cache/splits.json
  results_dir: /content/drive/MyDrive/SarcasTone/results

dataset:
  name: mustard_plus          # options: mustard, mustard_plus
  use_context: false
  val_ratio: 0.15
  test_ratio: 0.15
  speaker_disjoint: false

encoders:
  text_model: microsoft/deberta-v3-base
  audio_model: facebook/wav2vec2-base
  video_model: resnet50
  sample_rate: 16000
  max_audio_sec: 15.0
  num_video_frames: 8

model:
  latent_dim: 256
  prosody_dim: 16
  prosody_hidden_dim: 64
  dropout: 0.3
  fusion_mode: pgin           # options: pgin, concat

training:
  batch_size: 32
  epochs: 40
  lr: 3.0e-4
  weight_decay: 1.0e-2
  patience: 8
  lambda_inc: 0.1
  inc_margin: 0.5
  use_amp: true
```

---

## 6. Execution & Ablation Protocol (`src/experiments/run_ablations.py`)
Running `python -m src.experiments.run_ablations --config configs/default.yaml` must sequentially train and evaluate the following runs using identical splits and seeds:
1. `text_only`: `modalities=["text"]`, `fusion_mode="pgin"`
2. `speech_only`: `modalities=["speech"]`, `fusion_mode="pgin"`
3. `video_only`: `modalities=["video"]`, `fusion_mode="pgin"`
4. `text_speech_concat`: `modalities=["text", "speech"]`, `fusion_mode="concat"`
5. `text_speech_pgin`: `modalities=["text", "speech"]`, `fusion_mode="pgin"`
6. `text_speech_video_pgin`: `modalities=["text", "speech", "video"]`, `fusion_mode="pgin"`

Each run saves:
* Best checkpoint (`best_model.pt`) selected by validation **F1-score**.
* Test metrics JSON (`metrics.json`) with **Accuracy, Precision, Recall, F1-score**, confusion matrix, and mean learned gate weights (for `pgin` runs).
* Master comparison table (`ablation_summary.csv` and `ablation_summary.md`) across all 6 runs.