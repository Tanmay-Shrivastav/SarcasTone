#!/usr/bin/env python3
"""
scripts/generate_notebooks.py
=============================
Generates the two Google Colab T4 Jupyter notebooks:
  1. notebooks/01_extract_and_cache_colab.ipynb
  2. notebooks/02_train_and_evaluate_colab.ipynb
"""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def make_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {
                "gpuType": "T4",
                "provenance": []
            },
            "kernelspec": {
                "display_name": "Python 3",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10.12"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }


def markdown_cell(source):
    lines = source.strip().split("\n")
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in lines[:-1]] + [lines[-1]] if lines else []
    }


def code_cell(source):
    lines = source.strip().split("\n")
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in lines[:-1]] + [lines[-1]] if lines else []
    }


def generate_01_notebook():
    cells = [
        markdown_cell(
r"""# SarcasTone — Stage 1: Multimodal Feature Extraction & Drive Caching
### Optimized for Google Colab (T4 GPU - 16 GB VRAM)

[![Colab T4](https://img.shields.io/badge/Google_Colab-T4_GPU_Optimized-orange.svg?logo=googlecolab)](https://colab.research.google.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg?logo=pytorch)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗_Transformers-4.38+-yellow.svg)](https://huggingface.co/transformers/)

---

### Pipeline Overview
This notebook executes **Stage 1** of SarcasTone:
1. **Mounts Google Drive** for persistent storage of dataset, cache, and split manifests.
2. **Verifies T4 GPU** availability and CUDA environment (`torch.cuda.is_available()`).
3. **Installs required dependencies** from `requirements.txt`.
4. **Ingests annotations & creates deterministic stratified splits** (`splits.json`) with zero sample leakage.
5. **Extracts multimodal representations** across frozen pretrained backbones:
   * **Text:** `microsoft/deberta-v3-base` attention-masked mean pooling $\rightarrow \mathbb{R}^{768}$
   * **Speech Prosody:** 16-dimensional explicit vector ($F_0$ pitch dynamics, RMS energy, rhythm, pauses)
   * **Speech Acoustic:** `facebook/wav2vec2-base` temporal mean pooling $\rightarrow \mathbb{R}^{768}$
   * **Video:** `torchvision.models.resnet50` uniform 8-frame pooling $\rightarrow \mathbb{R}^{2048}$
6. **Persists extracted tensors** as `.pt` files to Google Drive with atomic manifest tracking for instant, idempotent resuming."""
        ),
        markdown_cell(
"""## 1. Mount Google Drive
Mount Google Drive to persist raw dataset media, offline extracted feature tensors, and deterministic split files across Colab runtimes."""
        ),
        code_cell(
"""import os
from pathlib import Path

# Mount Google Drive
try:
    from google.colab import drive
    drive.mount('/content/drive')
    print("✓ Google Drive mounted successfully at /content/drive")
except ImportError:
    print("ℹ️ Running outside Google Colab environment; skipping Drive mount.")"""
        ),
        markdown_cell(
"""## 2. Check Hardware & T4 GPU Availability
Verify that CUDA is enabled and a Google Colab T4 GPU (or equivalent) is allocated."""
        ),
        code_cell(
"""import sys
import torch

print(f"Python version : {sys.version.split()[0]}")
print(f"PyTorch version: {torch.__version__}")

cuda_available = torch.cuda.is_available()
print(f"CUDA Available : {cuda_available}")

if cuda_available:
    device_name = torch.cuda.get_device_name(0)
    vram_bytes = torch.cuda.get_device_properties(0).total_memory
    vram_gb = vram_bytes / (1024 ** 3)
    print(f"✓ GPU Allocated: {device_name}")
    print(f"✓ Total VRAM   : {vram_gb:.2f} GB")
    if "T4" in device_name:
        print("✓ Verified: NVIDIA Tesla T4 GPU ready for feature extraction!")
else:
    print("⚠️ WARNING: GPU is not enabled! To accelerate feature extraction:")
    print("   Select Runtime > Change runtime type > Hardware accelerator > T4 GPU")"""
        ),
        markdown_cell(
"""## 3. Repository Setup & Dependency Installation
Navigate to or clone the SarcasTone repository and install the verified dependency stack."""
        ),
        code_cell(
"""import os
import sys
from pathlib import Path

# Candidate repository locations
REPO_URL = "https://github.com/Tanmay-Shrivastav/SarcasTone.git"
REPO_DIR = Path("/content/SarcasTone")
DRIVE_REPO_DIR = Path("/content/drive/MyDrive/SarcasTone")

if DRIVE_REPO_DIR.exists() and (DRIVE_REPO_DIR / "configs" / "default.yaml").exists():
    os.chdir(str(DRIVE_REPO_DIR))
    print(f"✓ Working directory set to Google Drive repository: {DRIVE_REPO_DIR}")
    if (DRIVE_REPO_DIR / ".git").exists():
        print("Checking for latest repository updates on Drive...")
        !git pull origin main || git pull
elif REPO_DIR.exists() and (REPO_DIR / "configs" / "default.yaml").exists():
    os.chdir(str(REPO_DIR))
    print(f"✓ Working directory set to: {REPO_DIR}")
    if (REPO_DIR / ".git").exists():
        print("Checking for latest repository updates...")
        !git pull origin main || git pull
else:
    print(f"Cloning repository from {REPO_URL} into /content/SarcasTone...")
    !git clone https://github.com/Tanmay-Shrivastav/SarcasTone.git /content/SarcasTone
    os.chdir("/content/SarcasTone")
    print("✓ Working directory set to: /content/SarcasTone")

# Add repository root to Python module search path
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

# Install locked dependencies
!pip install -q -r requirements.txt
print("✓ Dependencies installed successfully.")"""
        ),
        markdown_cell(
"""## 4. Configure Storage Paths
Load `configs/default.yaml` and initialize persistent Google Drive directories."""
        ),
        code_cell(
"""import yaml
from pathlib import Path

CONFIG_PATH = Path("configs/default.yaml")
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

# Resolve paths
paths_cfg = cfg.get("paths", {})
data_root = Path(paths_cfg.get("data_root", "/content/drive/MyDrive/SarcasTone/data"))
cache_dir = Path(paths_cfg.get("cache_dir", "/content/drive/MyDrive/SarcasTone/cache"))
splits_file = Path(paths_cfg.get("splits_file", "/content/drive/MyDrive/SarcasTone/cache/splits.json"))
results_dir = Path(paths_cfg.get("results_dir", "/content/drive/MyDrive/SarcasTone/results"))

# Ensure directories exist
for p in [data_root, cache_dir, results_dir]:
    p.mkdir(parents=True, exist_ok=True)

print("Configured Paths:")
print(f"  Data Root   : {data_root}")
print(f"  Cache Dir   : {cache_dir}")
print(f"  Splits File : {splits_file}")
print(f"  Results Dir : {results_dir}")"""
        ),
        markdown_cell(
"""## 5. Dataset Ingestion & Deterministic Splitting
Ingest the MUStARD / MUStARD++ dataset structure:
- `data_root/annotations.json` (dialogue transcripts, speaker annotations, binary sarcasm labels)
- `data_root/utterances_final/<id>.mp4` (utterance audiovisual clips)
- `data_root/context_final/<id>_c.mp4` (preceding dialogue context clips)

`load_raw_samples` automatically resolves utterance video clips from `utterances_final/` and context clips from `context_final/`.
If dedicated `.wav` audio files are not present, `audio_path` automatically falls back to the `.mp4` video clip, and the 16 kHz audio track is extracted seamlessly in-memory during feature extraction."""
        ),
        code_cell(
"""import json
import numpy as np
from src.data.dataset import (
    MultimodalSample,
    create_or_load_splits,
    load_raw_samples,
    verify_leakage,
)
from scripts.verify_splits import verify_splits

samples = None

# Check for real dataset annotations
if any((data_root / fname).is_file() for fname in ["annotations.json", "sarcasm_data.json", "mustard_plus.csv"]):
    print(f"Loading raw annotations from {data_root}...")
    samples = load_raw_samples(
        data_root=data_root,
        annotation_file=cfg["dataset"].get("annotation_file"),
        dataset_name=cfg["dataset"].get("name", "mustard"),
        utterance_dir=cfg["dataset"].get("utterance_dir", "utterances_final"),
        context_dir=cfg["dataset"].get("context_dir", "context_final"),
    )
else:
    print("ℹ️ No raw dataset annotations found in data_root.")
    print("Creating a 60-sample synthetic corpus for zero-download pipeline validation...")
    rng = np.random.RandomState(cfg.get("project", {}).get("seed", 42))
    shows = ["friends", "tbbt", "saty"]
    speakers = ["CHANDLER", "JOEY", "MONICA", "SHELDON", "LEONARD", "PENNY"]
    samples = {}
    for i in range(1, 61):
        uid = f"sample_{i:04d}"
        samples[uid] = MultimodalSample(
            utterance_id=uid,
            utterance_text=f"Utterance {i} test transcript for multimodal sarcasm modeling.",
            context_text=f"Preceding conversational turn context for utterance {i}.",
            speaker=speakers[i % len(speakers)],
            show=shows[i % len(shows)],
            label=int(rng.choice([0, 1])),
            audio_valid=False,
            video_valid=False,
        )

# Create or load deterministic splits
splits = create_or_load_splits(
    samples=samples,
    splits_path=splits_file,
    val_ratio=cfg["dataset"].get("val_ratio", 0.15),
    test_ratio=cfg["dataset"].get("test_ratio", 0.15),
    seed=cfg["project"].get("seed", 42),
    speaker_disjoint=cfg["dataset"].get("speaker_disjoint", False),
    overwrite=False,
)

print(f"✓ Split Counts: Train={len(splits['train'])}, Val={len(splits['val'])}, Test={len(splits['test'])}")

# Run strict split verification suite
split_ok = verify_splits(splits_path=splits_file, samples=samples)
assert split_ok, "Dataset split verification failed!"
print("✓ Strict zero-leakage verified across all splits.")"""
        ),
        markdown_cell(
"""## 6. Initialize Pretrained Encoders on T4 GPU
Instantiate frozen feature extractors on the T4 GPU:
- `TextFeatureExtractor`: DeBERTa-v3 (`768,`)
- `ProsodyAcousticExtractor`: 16-dim prosodic statistics + Wav2Vec2 (`768,`)
- `VideoFeatureExtractor`: ResNet-50 (`2048,`)"""
        ),
        code_cell(
"""from src.features.text_features import TextFeatureExtractor
from src.features.prosody_features import ProsodyAcousticExtractor
from src.features.video_features import VideoFeatureExtractor

device = "cuda" if torch.cuda.is_available() else "cpu"
use_amp = bool(cfg["training"].get("use_amp", True)) and device == "cuda"
print(f"Loading pretrained models onto device: {device} (AMP: {use_amp})...")

# 1. Text Encoder (DeBERTa-v3-base)
text_extractor = TextFeatureExtractor(
    model_name=cfg["encoders"].get("text_model", "microsoft/deberta-v3-base"),
    device=device,
    use_amp=use_amp,
)

# 2. Speech Acoustic & Prosody Extractor (Wav2Vec2 + Librosa)
speech_extractor = ProsodyAcousticExtractor(
    audio_model=cfg["encoders"].get("audio_model", "facebook/wav2vec2-base"),
    sample_rate=cfg["encoders"].get("sample_rate", 16000),
    max_audio_sec=cfg["encoders"].get("max_audio_sec", 15.0),
    device=device,
    use_amp=use_amp,
)

# 3. Video Frame Extractor (ResNet-50)
video_extractor = VideoFeatureExtractor(
    model_name="resnet50",
    num_frames=cfg["encoders"].get("num_video_frames", 8),
    frame_size=cfg["encoders"].get("frame_size", 224),
    device=device,
    use_amp=use_amp,
)

print("✓ All feature extractors loaded and initialized on GPU.")"""
        ),
        markdown_cell(
"""## 7. Extract & Cache Features to Google Drive
Iterate through the samples, extract all modality tensors, and persist to `.pt` files.
Resuming is fully supported: if your runtime disconnects, re-running this cell skips already-cached samples."""
        ),
        code_cell(
"""from tqdm.auto import tqdm
from src.features.cache_manager import FeatureCacheManager

cache_mgr = FeatureCacheManager(cache_dir=cache_dir, file_format="pt", overwrite=False)

all_sample_ids = list(samples.keys())
uncached_ids = cache_mgr.get_uncached_sample_ids(all_sample_ids, show_progress=True)

print(f"Total samples in dataset : {len(all_sample_ids)}")
print(f"Already cached on Drive  : {len(all_sample_ids) - len(uncached_ids)}")
print(f"Remaining to extract     : {len(uncached_ids)}")

# Extraction loop with throttled tqdm.auto progress bar (mininterval=1.0, miniters=4)
pbar = tqdm(
    uncached_ids,
    desc="Extracting & Caching Tensors",
    unit="sample",
    mininterval=1.0,
    miniters=4,
    leave=True,
)
for uid in pbar:
    sample = samples[uid]

    # 1. Text embedding (768,)
    text_emb = text_extractor.extract_single(
        text=sample.utterance_text,
        context=sample.context_text if cfg["dataset"].get("use_context", False) else None,
    )

    # 2. Speech prosody (16,) and acoustic embedding (768,)
    if sample.audio_path and sample.audio_valid:
        prosody_vec, audio_emb, audio_valid = speech_extractor.extract_from_file(sample.audio_path)
    else:
        prosody_vec = np.zeros(16, dtype=np.float32)
        audio_emb = np.zeros(768, dtype=np.float32)
        audio_valid = False

    # 3. Video frame embedding (2048,)
    if sample.video_path and sample.video_valid:
        video_emb, video_valid = video_extractor.extract_from_file(sample.video_path)
    else:
        video_emb = np.zeros(2048, dtype=np.float32)
        video_valid = False

    # Bundle and persist sample tensor
    features = {
        "text_emb": text_emb,
        "prosody_vec": prosody_vec,
        "audio_emb": audio_emb,
        "video_emb": video_emb,
        "label": sample.label,
        "audio_valid": audio_valid,
        "video_valid": video_valid,
        "metadata": {
            "speaker": sample.speaker,
            "show": sample.show,
        },
    }
    cache_mgr.save_sample_features(uid, features)
    pbar.set_postfix({"id": uid[:12], "label": sample.label}, refresh=False)

# Save manifest atomically
cache_mgr.save_manifest()
print("✓ Feature extraction and Drive caching successfully complete!")"""
        ),
        markdown_cell(
"""## 8. Validate Cache & Inspect Extracted Tensors
Display diagnostic cache summary and load a sample tensor to verify dimensions."""
        ),
        code_cell(
"""summary = cache_mgr.get_summary()

print("=" * 60)
print("FEATURE CACHE SUMMARY (GOOGLE DRIVE)")
print("=" * 60)
for k, v in summary.items():
    print(f"  {k:<22}: {v}")
print("=" * 60)

# Inspect first sample
sample_0 = all_sample_ids[0]
loaded = cache_mgr.load_sample_features(sample_0)

print()
print(f"Tensor Inspection for sample '{sample_0}':")
print(f"  text_emb    : shape {loaded['text_emb'].shape} (dtype: {loaded['text_emb'].dtype})")
print(f"  prosody_vec : shape {loaded['prosody_vec'].shape} (dtype: {loaded['prosody_vec'].dtype})")
print(f"  audio_emb   : shape {loaded['audio_emb'].shape} (dtype: {loaded['audio_emb'].dtype})")
print(f"  video_emb   : shape {loaded['video_emb'].shape} (dtype: {loaded['video_emb'].dtype})")
print(f"  label       : {loaded['label']}")
print(f"  audio_valid : {loaded['audio_valid']}")
print(f"  video_valid : {loaded['video_valid']}")

print()
print("=" * 60)
print("STAGE 1 COMPLETE: All multimodal features are securely cached on Google Drive.")
print("Proceed to: notebooks/02_train_and_evaluate_colab.ipynb to run the 8-run ablation matrix.")
print("=" * 60)"""
        ),
    ]
    return make_notebook(cells)


def generate_02_notebook():
    cells = [
        markdown_cell(
"""# SarcasTone — Stage 2: Training & 8-Run Modality Ablation Matrix
### Optimized for Google Colab (T4 GPU - 16 GB VRAM)

[![Colab T4](https://img.shields.io/badge/Google_Colab-T4_GPU_Optimized-orange.svg?logo=googlecolab)](https://colab.research.google.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg?logo=pytorch)](https://pytorch.org/)

---

### Pipeline Overview
This notebook executes **Stage 2** of SarcasTone:
1. **Mounts Google Drive** and loads pre-extracted feature tensors from `cache_dir`.
2. **Verifies locked deterministic splits** (`splits.json`) ensuring strictly zero leakage.
3. **Runs optional lightweight smoke test** (`scripts/smoke_test.py`) for pipeline health.
4. **Executes the full 8-run modality ablation matrix** on the T4 GPU with mixed precision (`fp16`):
   * `text_only`
   * `speech_only`
   * `video_only`
   * `text_speech_concat` (Baseline naive concatenation)
   * `text_speech_pgin` (Core hypothesis: prosody-gated incongruity)
   * `text_video`
   * `speech_video`
   * `full_multimodal_pgin` (Full Text + Speech Prosody/Acoustics + Video)
5. **Displays side-by-side comparison tables** (Accuracy, Precision, Recall, Macro F1).
6. **Renders comparative visual plots** including multi-metric bar charts and learned gate weight distributions."""
        ),
        markdown_cell(
"""## 1. Mount Google Drive
Mount Google Drive to access cached feature tensors and persist training checkpoints and logs."""
        ),
        code_cell(
"""import os
from pathlib import Path

# Mount Google Drive
try:
    from google.colab import drive
    drive.mount('/content/drive')
    print("✓ Google Drive mounted successfully at /content/drive")
except ImportError:
    print("ℹ️ Running outside Google Colab environment; skipping Drive mount.")"""
        ),
        markdown_cell(
"""## 2. Verify Hardware & T4 GPU Environment
Confirm CUDA acceleration and mixed precision (`torch.amp`) support on Colab T4 GPU."""
        ),
        code_cell(
"""import sys
import torch

print(f"Python version : {sys.version.split()[0]}")
print(f"PyTorch version: {torch.__version__}")

cuda_available = torch.cuda.is_available()
print(f"CUDA Available : {cuda_available}")

if cuda_available:
    device_name = torch.cuda.get_device_name(0)
    vram_bytes = torch.cuda.get_device_properties(0).total_memory
    vram_gb = vram_bytes / (1024 ** 3)
    print(f"✓ GPU Allocated: {device_name}")
    print(f"✓ Total VRAM   : {vram_gb:.2f} GB")
    print("✓ AMP Support  : fp16 autocast enabled for Colab T4 acceleration.")
else:
    print("ℹ️ GPU is not allocated. Running in CPU mode (slower).")
    print("   To accelerate training, select Runtime > Change runtime type > T4 GPU.")"""
        ),
        markdown_cell(
"""## 3. Repository Setup & Dependencies
Set working directory and verify that `src/` modules and `configs/default.yaml` are available."""
        ),
        code_cell(
"""import os
import sys
from pathlib import Path

# Candidate repository locations
REPO_URL = "https://github.com/Tanmay-Shrivastav/SarcasTone.git"
REPO_DIR = Path("/content/SarcasTone")
DRIVE_REPO_DIR = Path("/content/drive/MyDrive/SarcasTone")

if DRIVE_REPO_DIR.exists() and (DRIVE_REPO_DIR / "configs" / "default.yaml").exists():
    os.chdir(str(DRIVE_REPO_DIR))
    print(f"✓ Working directory: {DRIVE_REPO_DIR}")
    if (DRIVE_REPO_DIR / ".git").exists():
        print("Checking for latest repository updates on Drive...")
        !git pull origin main || git pull
elif REPO_DIR.exists() and (REPO_DIR / "configs" / "default.yaml").exists():
    os.chdir(str(REPO_DIR))
    print(f"✓ Working directory: {REPO_DIR}")
    if (REPO_DIR / ".git").exists():
        print("Checking for latest repository updates...")
        !git pull origin main || git pull
else:
    print(f"Cloning repository from {REPO_URL} into /content/SarcasTone...")
    !git clone https://github.com/Tanmay-Shrivastav/SarcasTone.git /content/SarcasTone
    os.chdir("/content/SarcasTone")
    print("✓ Working directory: /content/SarcasTone")

if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

!pip install -q -r requirements.txt
print("✓ Dependencies verified.")"""
        ),
        markdown_cell(
"""## 4. Configuration & Split Verification
Confirm that `splits.json` and cached feature files exist on Drive, and run the verification suite to prove zero leakage."""
        ),
        code_cell(
"""import yaml
from pathlib import Path
from scripts.verify_splits import verify_splits, load_config

CONFIG_PATH = Path("configs/default.yaml")
cfg = load_config(CONFIG_PATH)

paths_cfg = cfg.get("paths", {})
cache_dir = Path(paths_cfg.get("cache_dir", "/content/drive/MyDrive/SarcasTone/cache"))
splits_file = Path(paths_cfg.get("splits_file", "/content/drive/MyDrive/SarcasTone/cache/splits.json"))
results_dir = Path(paths_cfg.get("results_dir", "/content/drive/MyDrive/SarcasTone/results"))

results_dir.mkdir(parents=True, exist_ok=True)

print(f"Checking splits file at: {splits_file}")
assert splits_file.is_file(), f"Splits file not found at {splits_file}! Run 01_extract_and_cache_colab.ipynb first."

passed = verify_splits(
    splits_path=splits_file,
    cache_dir=cache_dir,
    check_speaker_disjoint=cfg.get("dataset", {}).get("speaker_disjoint", False),
)
assert passed, "Dataset split verification failed!"
print("✓ Split verification passed with zero leakage across partitions.")"""
        ),
        markdown_cell(
"""## 5. Lightweight Smoke Test (Sanity Check)
Run `scripts/smoke_test.py` to verify forward passes, incongruity calculation, loss functions, and metrics end-to-end."""
        ),
        code_cell(
"""import torch

# Run smoke test dynamically selecting GPU if available, else CPU
device_arg = "cuda" if torch.cuda.is_available() else "cpu"
!python scripts/smoke_test.py --device {device_arg}"""
        ),
        markdown_cell(
"""## 6. Execute the 8-Run Modality Ablation Matrix on T4 GPU
Run `src.experiments.run_ablations` across all 8 configurations on locked splits:
1. `text_only`
2. `speech_only`
3. `video_only`
4. `text_speech_concat` (Baseline naive concatenation)
5. `text_speech_pgin` (Prosody-Gated Incongruity Network)
6. `text_video`
7. `speech_video`
8. `full_multimodal_pgin` (Full Multimodal Network)

Live `tqdm` progress bars display ETA and throughput across the 8 runs, per-run epochs, and training/validation batches."""
        ),
        code_cell(
"""import time
import torch
from src.experiments.run_ablations import main as ablations_main

print("=" * 75)
print("STARTING 8-RUN MODALITY ABLATION EXPERIMENTS")
print("=" * 75)

t0 = time.time()

# Execute all 8 runs sequentially on identical splits with live tqdm progress and ETA
device_arg = "cuda" if torch.cuda.is_available() else "cpu"
exit_code = ablations_main([
    "--config", str(CONFIG_PATH),
    "--results-dir", str(results_dir),
    "--splits-file", str(splits_file),
    "--device", device_arg,
])

elapsed = time.time() - t0
assert exit_code == 0, f"run_ablations failed with exit code: {exit_code}"
print()
print(f"✓ All 8 ablation runs completed successfully in {elapsed / 60:.2f} minutes!")"""
        ),
        markdown_cell(
"""## 7. Side-by-Side Comparison Table across All 8 Runs
Load and display the structured results table comparing Accuracy, Precision, Recall, and Macro F1-Score."""
        ),
        code_cell(
"""import pandas as pd
from IPython.display import display, HTML

csv_path = results_dir / "ablation_summary.csv"
assert csv_path.is_file(), f"Summary CSV not found at {csv_path}"

df = pd.read_csv(csv_path)

# Selected presentation columns
cols = [
    "run_name", "modalities", "fusion_mode",
    "accuracy", "precision", "recall", "f1",
    "best_val_f1", "best_epoch", "elapsed_s"
]
avail_cols = [c for c in cols if c in df.columns]
display_df = df[avail_cols].copy()

# Rename for clean presentation
display_df.rename(columns={
    "run_name": "Run Name",
    "modalities": "Modalities",
    "fusion_mode": "Fusion Mode",
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "Macro F1",
    "best_val_f1": "Best Val F1",
    "best_epoch": "Best Epoch",
    "elapsed_s": "Time (s)",
}, inplace=True)

# Format floats
for col in ["Accuracy", "Precision", "Recall", "Macro F1", "Best Val F1"]:
    if col in display_df.columns:
        display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}" if pd.notna(x) and isinstance(x, (int, float)) else str(x))

print("=" * 85)
print("SARCASTONE: 8-RUN MODALITY ABLATION STUDY COMPARISON TABLE")
print("=" * 85)
display(display_df)

# Highlight best run
if "f1" in df.columns:
    best_idx = pd.to_numeric(df["f1"], errors="coerce").idxmax()
    best_row = df.iloc[best_idx]
    print()
    print(f"🏆 Top Performing Model: '{best_row['run_name']}'")
    print(f"   Modalities: {best_row['modalities']}  |  Fusion: {best_row['fusion_mode']}")
    print(f"   Accuracy  : {best_row['accuracy']:.4f}     |  Macro F1: {best_row['f1']:.4f}")"""
        ),
        markdown_cell(
"""## 8. Comparative Visualizations & Analysis
Generate multi-metric performance bar charts and analyze learned modality gate allocations."""
        ),
        code_cell(
"""import matplotlib.pyplot as plt
import numpy as np

# Set clean styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, ax = plt.subplots(figsize=(15, 6))

plot_df = df.copy()
for col in ["accuracy", "precision", "recall", "f1"]:
    plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce").fillna(0.0)

x = np.arange(len(plot_df))
width = 0.18

ax.bar(x - 1.5 * width, plot_df["accuracy"],  width, label="Accuracy", color="#3b82f6", edgecolor="black", linewidth=0.5)
ax.bar(x - 0.5 * width, plot_df["precision"], width, label="Precision", color="#10b981", edgecolor="black", linewidth=0.5)
ax.bar(x + 0.5 * width, plot_df["recall"],    width, label="Recall", color="#f59e0b", edgecolor="black", linewidth=0.5)
ax.bar(x + 1.5 * width, plot_df["f1"],        width, label="Macro F1", color="#ef4444", edgecolor="black", linewidth=0.5)

ax.set_ylabel("Score", fontsize=12)
ax.set_title("SarcasTone: 8-Run Modality Ablation Performance on Google Colab T4", fontsize=14, fontweight="bold", pad=15)
ax.set_xticks(x)
ax.set_xticklabels(plot_df["run_name"], rotation=25, ha="right", fontsize=10)
ax.set_ylim(0.0, 1.05)
ax.legend(loc="lower right", frameon=True, fontsize=11)

plt.tight_layout()
chart_path = results_dir / "ablation_comparison_chart.png"
plt.savefig(chart_path, dpi=300)
plt.show()
print(f"✓ Chart saved to: {chart_path}")"""
        ),
        markdown_cell(
r"""### Learned Modality Gating Weights ($g_T, g_A, g_{\text{inc}}, g_V$)
Inspect how PGIN dynamically balances active modalities across the dataset."""
        ),
        code_cell(
"""gate_cols = [c for c in ["gate_text", "gate_speech", "gate_inc", "gate_video"] if c in df.columns]
pgin_subset = df[df["fusion_mode"] == "pgin"].dropna(subset=gate_cols, how="all")

if not pgin_subset.empty and gate_cols:
    fig, ax = plt.subplots(figsize=(11, 5))
    gate_data = pgin_subset.set_index("run_name")[gate_cols].fillna(0.0)
    gate_data.columns = ["Text (g_T)", "Speech (g_A)", "Incongruity (g_inc)", "Video (g_V)"]
    gate_data.plot(
        kind="bar",
        stacked=True,
        ax=ax,
        colormap="viridis",
        edgecolor="black",
        linewidth=0.5,
    )
    ax.set_ylabel("Mean Gate Weight", fontsize=12)
    ax.set_title("Learned Modality Gate Weights across PGIN Ablation Runs", fontsize=13, fontweight="bold", pad=12)
    ax.set_xticklabels(gate_data.index, rotation=25, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.legend(title="Modality Branch", loc="upper right")
    plt.tight_layout()
    gate_chart_path = results_dir / "gate_weights_chart.png"
    plt.savefig(gate_chart_path, dpi=300)
    plt.show()
    print(f"✓ Gate weight chart saved to: {gate_chart_path}")
else:
    print("ℹ️ Gate weights not available in current runs.")"""
        ),
        markdown_cell(
"""## 9. Artifact Summary & Saved Checkpoints
Review the artifacts and checkpoints saved in Google Drive."""
        ),
        code_cell(
"""print("=" * 65)
print(f"PERSISTED ARTIFACTS ON GOOGLE DRIVE: {results_dir}")
print("=" * 65)

for item in sorted(results_dir.iterdir()):
    if item.is_dir():
        files = [f.name for f in item.iterdir()]
        print(f"📁 {item.name:<26} -> {', '.join(files)}")
    else:
        print(f"📄 {item.name}")

print("=" * 65)
print("✓ All 8 ablation runs complete! Checkpoints and metrics are safely saved to Drive.")
print("=" * 65)"""
        ),
    ]
    return make_notebook(cells)


def main():
    notebooks_dir = REPO_ROOT / "notebooks"
    notebooks_dir.mkdir(parents=True, exist_ok=True)

    # 1. 01_extract_and_cache_colab.ipynb
    nb1 = generate_01_notebook()
    nb1_path = notebooks_dir / "01_extract_and_cache_colab.ipynb"
    with open(nb1_path, "w", encoding="utf-8") as f:
        json.dump(nb1, f, indent=1)
    print(f"Generated {nb1_path} ({len(nb1['cells'])} cells)")

    # 2. 02_train_and_evaluate_colab.ipynb
    nb2 = generate_02_notebook()
    nb2_path = notebooks_dir / "02_train_and_evaluate_colab.ipynb"
    with open(nb2_path, "w", encoding="utf-8") as f:
        json.dump(nb2, f, indent=1)
    print(f"Generated {nb2_path} ({len(nb2['cells'])} cells)")


if __name__ == "__main__":
    main()
