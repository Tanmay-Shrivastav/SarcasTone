"""
Dataset abstractions, raw metadata ingestion, deterministic splitting,
and cached feature dataset classes for SarcasTone.
Supports MUStARD and MUStARD++ audiovisual sarcasm corpora.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

try:
    import torch
    from torch.utils.data import Dataset as TorchDataset
    TORCH_AVAILABLE = True
except ImportError:
    TorchDataset = object  # type: ignore
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class MultimodalSample:
    """Represents a single multimodal sample in the sarcasm corpus."""
    utterance_id: str
    utterance_text: str
    context_text: Optional[str] = None
    audio_path: Optional[str] = None
    video_path: Optional[str] = None
    context_audio_path: Optional[str] = None
    context_video_path: Optional[str] = None
    speaker: Optional[str] = None
    show: Optional[str] = None
    label: int = 0  # 1 = sarcastic, 0 = non-sarcastic
    audio_valid: bool = True
    video_valid: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ProsodyScaler:
    """Standardizes prosody feature vectors strictly fit on the training split."""

    def __init__(self, feature_dim: int = 16, eps: float = 1e-7):
        self.feature_dim = feature_dim
        self.eps = eps
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None
        self.is_fit: bool = False

    def fit(self, features: np.ndarray) -> ProsodyScaler:
        """
        Fit scaler parameters on training prosody feature matrix of shape (N, feature_dim).
        Safely handles unvoiced/silent NaNs and infinite values.
        """
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        self.mean = np.mean(features, axis=0)
        self.std = np.std(features, axis=0)
        # Avoid division by zero on static features
        self.std[self.std < self.eps] = 1.0
        self.is_fit = True
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        """Standardize feature vector or matrix."""
        if not self.is_fit or self.mean is None or self.std is None:
            raise RuntimeError("ProsodyScaler must be fit on training split before transform.")
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        return (features - self.mean) / self.std

    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        return self.fit(features).transform(features)

    def save(self, file_path: Union[str, Path]) -> None:
        """Persist scaler parameters to JSON file."""
        data = {
            "feature_dim": self.feature_dim,
            "mean": self.mean.tolist() if self.mean is not None else None,
            "std": self.std.tolist() if self.std is not None else None,
            "is_fit": self.is_fit,
        }
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, file_path: Union[str, Path]) -> ProsodyScaler:
        """Load scaler parameters from JSON file."""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        scaler = cls(feature_dim=data["feature_dim"])
        if data["mean"] is not None and data["std"] is not None:
            scaler.mean = np.array(data["mean"], dtype=np.float32)
            scaler.std = np.array(data["std"], dtype=np.float32)
            scaler.is_fit = data["is_fit"]
        return scaler


def _resolve_media_path(
    base_dir: Path,
    subdirs: Sequence[str],
    sample_id: str,
    extensions: Sequence[str],
    id_suffixes: Sequence[str] = ("",),
) -> Optional[str]:
    """Search for matching media file across candidate directories, id suffixes, and extensions."""
    for subdir in subdirs:
        sub_path = base_dir / subdir
        for suffix in id_suffixes:
            file_stem = f"{sample_id}{suffix}"
            for ext in extensions:
                candidate = sub_path / f"{file_stem}{ext}"
                if candidate.is_file():
                    return str(candidate)
    # Check directly in base_dir
    for suffix in id_suffixes:
        file_stem = f"{sample_id}{suffix}"
        for ext in extensions:
            candidate = base_dir / f"{file_stem}{ext}"
            if candidate.is_file():
                return str(candidate)
    return None


def load_raw_samples(
    data_root: Union[str, Path],
    annotation_file: Optional[Union[str, Path]] = None,
    dataset_name: str = "mustard",
    utterance_dir: Optional[str] = None,
    context_dir: Optional[str] = None,
) -> Dict[str, MultimodalSample]:
    """
    Ingest MUStARD or MUStARD++ metadata files and resolve audio and video paths.
    Supports standard MUStARD audiovisual dataset conventions:
      - annotations.json
      - utterances_final/<id>.mp4 (utterance audiovisual clips)
      - context_final/<id>_c.mp4 (context video clips)
      - Dedicated audio directories (audio/, wav/, audios/) or automatic fallback to .mp4

    Args:
        data_root: Root directory of dataset containing media folders.
        annotation_file: Explicit path to annotations file (.json, .csv, or .tsv).
                         If None, searches data_root for standard annotation files.
        dataset_name: Identifier for dataset schema ('mustard' or 'mustard_plus').
        utterance_dir: Optional override directory name for utterance clips (default: utterances_final).
        context_dir: Optional override directory name for context clips (default: context_final).

    Returns:
        Dictionary mapping utterance_id -> MultimodalSample.
    """
    data_path = Path(data_root)
    ann_path: Optional[Path] = None

    if annotation_file:
        ann_path = Path(annotation_file)
    else:
        # Candidate annotation filenames
        candidates = [
            data_path / "annotations.json",
            data_path / "sarcasm_data.json",
            data_path / "mustard.json",
            data_path / "mustard_plus.csv",
            data_path / "annotations.csv",
            data_path / "data.csv",
            data_path / "mustard_plus.tsv",
        ]
        for c in candidates:
            if c.is_file():
                ann_path = c
                break

    if ann_path is None or not ann_path.is_file():
        raise FileNotFoundError(
            f"Annotation file not found in {data_root}. "
            f"Please specify an explicit annotation_file path in configs/default.yaml."
        )

    logger.info(f"Loading annotations from: {ann_path}")
    samples: Dict[str, MultimodalSample] = {}

    video_extensions = [".mp4", ".avi", ".mkv", ".mov", ".webm", ".m4v"]
    audio_extensions = [".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg"]

    # Candidate directories for utterance video clips
    video_subdirs = ["utterances_final", "utterances", "video", "videos", "mp4", "clips", "raw"]
    if utterance_dir and utterance_dir not in video_subdirs:
        video_subdirs.insert(0, utterance_dir)

    # Candidate directories for dedicated audio tracks
    audio_subdirs = ["audio", "audios", "wav", "wavs", "utterances_final", "utterances", "extracted_audio"]

    # Candidate directories for context video clips
    context_video_subdirs = ["context_final", "context", "contexts", "context_videos"]
    if context_dir and context_dir not in context_video_subdirs:
        context_video_subdirs.insert(0, context_dir)

    context_audio_subdirs = ["context_audio", "audio", "wav", "context_final", "context"]

    def _resolve_sample_media(uid_str: str):
        # 1. Resolve utterance video (.mp4)
        v_path = _resolve_media_path(data_path, video_subdirs, uid_str, video_extensions)

        # 2. Resolve utterance audio:
        # First check dedicated audio files (.wav, .mp3, etc.)
        a_path = _resolve_media_path(data_path, audio_subdirs, uid_str, audio_extensions)
        # If no separate audio file is found, fall back to the utterance video clip (.mp4)
        # since .mp4 contains the embedded audio stream
        if a_path is None and v_path is not None:
            a_path = v_path

        # 3. Resolve context video (MUStARD uses <id>_c.mp4 or <id>.mp4 in context_final/)
        c_v_path = _resolve_media_path(
            data_path, context_video_subdirs, uid_str, video_extensions, id_suffixes=["_c", ""]
        )

        # 4. Resolve context audio (dedicated audio or fallback to context video)
        c_a_path = _resolve_media_path(
            data_path, context_audio_subdirs, uid_str, audio_extensions, id_suffixes=["_c", ""]
        )
        if c_a_path is None and c_v_path is not None:
            c_a_path = c_v_path

        a_valid = a_path is not None and Path(a_path).is_file()
        v_valid = v_path is not None and Path(v_path).is_file()

        return a_path, v_path, c_a_path, c_v_path, a_valid, v_valid

    if ann_path.suffix.lower() == ".json":
        with open(ann_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        # MUStARD format: keyed by utterance ID dictionary
        if isinstance(raw_data, dict):
            for uid, info in raw_data.items():
                utterance_text = str(info.get("utterance", info.get("text", "")))
                context = info.get("context", "")
                if isinstance(context, list):
                    context_text = " </s> ".join(str(c) for c in context) if context else None
                else:
                    context_text = str(context) if context else None

                raw_label = info.get("sarcasm", info.get("label", 0))
                label = 1 if bool(raw_label) is True or raw_label in (1, "1", "true", "True") else 0
                speaker = info.get("speaker", None)
                show = info.get("show", None)

                audio_p, video_p, c_audio_p, c_video_p, a_valid, v_valid = _resolve_sample_media(str(uid))

                samples[str(uid)] = MultimodalSample(
                    utterance_id=str(uid),
                    utterance_text=utterance_text,
                    context_text=context_text,
                    audio_path=audio_p,
                    video_path=video_p,
                    context_audio_path=c_audio_p,
                    context_video_path=c_video_p,
                    speaker=str(speaker) if speaker else None,
                    show=str(show) if show else None,
                    label=label,
                    audio_valid=a_valid,
                    video_valid=v_valid,
                )
        elif isinstance(raw_data, list):
            for item in raw_data:
                uid = str(item.get("id", item.get("utterance_id", item.get("KEY", ""))))
                if not uid:
                    continue
                utterance_text = str(item.get("utterance", item.get("SENTENCE", item.get("text", ""))))
                context = item.get("context", item.get("CONTEXT", None))
                if isinstance(context, list):
                    context_text = " </s> ".join(str(c) for c in context) if context else None
                else:
                    context_text = str(context) if context else None

                raw_label = item.get("sarcasm", item.get("SARCASM", item.get("label", 0)))
                label = 1 if bool(raw_label) is True or raw_label in (1, "1", "true", "True") else 0
                speaker = item.get("speaker", item.get("SPEAKER", None))
                show = item.get("show", item.get("SHOW", None))

                audio_p, video_p, c_audio_p, c_video_p, a_valid, v_valid = _resolve_sample_media(str(uid))

                samples[uid] = MultimodalSample(
                    utterance_id=uid,
                    utterance_text=utterance_text,
                    context_text=context_text,
                    audio_path=audio_p,
                    video_path=video_p,
                    context_audio_path=c_audio_p,
                    context_video_path=c_video_p,
                    speaker=str(speaker) if speaker else None,
                    show=str(show) if show else None,
                    label=label,
                    audio_valid=a_valid,
                    video_valid=v_valid,
                )

    elif ann_path.suffix.lower() in [".csv", ".tsv"]:
        sep = "\t" if ann_path.suffix.lower() == ".tsv" else ","
        df = pd.read_csv(ann_path, sep=sep)

        # Standardize column names
        col_map = {col: col.upper() for col in df.columns}
        df_norm = df.rename(columns=col_map)

        id_col = next((c for c in ["KEY", "ID", "UTTERANCE_ID"] if c in df_norm.columns), df_norm.columns[0])
        text_col = next((c for c in ["SENTENCE", "UTTERANCE", "TEXT"] if c in df_norm.columns), None)
        context_col = next((c for c in ["CONTEXT", "CONTEXT_TEXT"] if c in df_norm.columns), None)
        label_col = next((c for c in ["SARCASM", "LABEL", "SARCASTIC"] if c in df_norm.columns), None)
        speaker_col = next((c for c in ["SPEAKER", "SPEAKER_ID"] if c in df_norm.columns), None)
        show_col = next((c for c in ["SHOW", "SHOW_ID"] if c in df_norm.columns), None)

        if text_col is None:
            raise ValueError(f"Could not locate text/utterance column in {ann_path}")

        for _, row in df_norm.iterrows():
            uid = str(row[id_col])
            utterance_text = str(row[text_col])
            context_val = row[context_col] if context_col and pd.notna(row[context_col]) else None
            context_text = str(context_val) if context_val else None

            raw_label = row[label_col] if label_col and pd.notna(row[label_col]) else 0
            label = 1 if raw_label in (1, "1", True, "true", "True") else 0
            speaker = str(row[speaker_col]) if speaker_col and pd.notna(row[speaker_col]) else None
            show = str(row[show_col]) if show_col and pd.notna(row[show_col]) else None

            audio_p, video_p, c_audio_p, c_video_p, a_valid, v_valid = _resolve_sample_media(str(uid))

            samples[uid] = MultimodalSample(
                utterance_id=uid,
                utterance_text=utterance_text,
                context_text=context_text,
                audio_path=audio_p,
                video_path=video_p,
                context_audio_path=c_audio_p,
                context_video_path=c_video_p,
                speaker=speaker,
                show=show,
                label=label,
                audio_valid=a_valid,
                video_valid=v_valid,
            )

    logger.info(f"Loaded {len(samples)} multimodal samples from {ann_path.name}")
    return samples


def create_or_load_splits(
    samples: Dict[str, MultimodalSample],
    splits_path: Union[str, Path],
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    speaker_disjoint: bool = False,
    overwrite: bool = False,
) -> Dict[str, List[str]]:
    """
    Create or load deterministic stratified train/val/test splits saved to disk.
    Guarantees zero sample leakage across splits and consistency across all ablation runs.

    Args:
        samples: Dictionary of all available samples.
        splits_path: Target JSON file path where splits are saved/loaded.
        val_ratio: Proportion of dataset for validation split.
        test_ratio: Proportion of dataset for test split.
        seed: Random seed for deterministic reproducibility.
        speaker_disjoint: If True, partitions splits so speakers do not overlap across splits.
        overwrite: If True, regenerates splits even if file already exists.

    Returns:
        Dictionary with keys 'train', 'val', 'test' mapping to lists of utterance IDs.
    """
    splits_file = Path(splits_path)

    if splits_file.is_file() and not overwrite:
        logger.info(f"Loading existing deterministic splits from: {splits_file}")
        with open(splits_file, "r", encoding="utf-8") as f:
            splits_data = json.load(f)

        train_ids = splits_data.get("train", [])
        val_ids = splits_data.get("val", [])
        test_ids = splits_data.get("test", [])

        # Strict split verification on loaded file
        verify_leakage(train_ids, val_ids, test_ids)
        return {"train": train_ids, "val": val_ids, "test": test_ids}

    logger.info(f"Generating deterministic splits (seed={seed}, val={val_ratio}, test={test_ratio})...")
    uids = sorted(list(samples.keys()))
    labels = [samples[uid].label for uid in uids]
    speakers = [samples[uid].speaker or "unknown" for uid in uids]

    if speaker_disjoint:
        # Disjoint speaker grouping
        gss_test = GroupShuffleSplit(n_splits=1, test_size=test_ratio, random_state=seed)
        train_val_idx, test_idx = next(gss_test.split(uids, labels, groups=speakers))

        train_val_uids = [uids[i] for i in train_val_idx]
        train_val_labels = [labels[i] for i in train_val_idx]
        train_val_speakers = [speakers[i] for i in train_val_idx]

        val_adjusted_ratio = val_ratio / (1.0 - test_ratio)
        gss_val = GroupShuffleSplit(n_splits=1, test_size=val_adjusted_ratio, random_state=seed)
        train_idx, val_idx = next(gss_val.split(train_val_uids, train_val_labels, groups=train_val_speakers))

        train_ids = [train_val_uids[i] for i in train_idx]
        val_ids = [train_val_uids[i] for i in val_idx]
        test_ids = [uids[i] for i in test_idx]
    else:
        # Standard stratified train/val/test split
        train_val_ids, test_ids, train_val_y, test_y = train_test_split(
            uids, labels, test_size=test_ratio, random_state=seed, stratify=labels
        )
        val_adjusted_ratio = val_ratio / (1.0 - test_ratio)
        train_ids, val_ids = train_test_split(
            train_val_ids, test_size=val_adjusted_ratio, random_state=seed, stratify=train_val_y
        )

    # Strict assertion of zero sample leakage
    verify_leakage(train_ids, val_ids, test_ids)

    # Calculate class distribution statistics
    def calc_sarcasm_ratio(id_list: Sequence[str]) -> float:
        if not id_list:
            return 0.0
        return sum(samples[uid].label for uid in id_list) / len(id_list)

    splits_data = {
        "metadata": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "seed": seed,
            "speaker_disjoint": speaker_disjoint,
            "total_samples": len(uids),
            "train_samples": len(train_ids),
            "val_samples": len(val_ids),
            "test_samples": len(test_ids),
            "train_sarcasm_ratio": round(calc_sarcasm_ratio(train_ids), 4),
            "val_sarcasm_ratio": round(calc_sarcasm_ratio(val_ids), 4),
            "test_sarcasm_ratio": round(calc_sarcasm_ratio(test_ids), 4),
        },
        "train": sorted(train_ids),
        "val": sorted(val_ids),
        "test": sorted(test_ids),
    }

    # Ensure parent directory exists
    splits_file.parent.mkdir(parents=True, exist_ok=True)
    with open(splits_file, "w", encoding="utf-8") as f:
        json.dump(splits_data, f, indent=2)

    logger.info(
        f"Deterministic splits saved to {splits_file} | "
        f"Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}"
    )
    return {"train": splits_data["train"], "val": splits_data["val"], "test": splits_data["test"]}


def verify_leakage(train_ids: Sequence[str], val_ids: Sequence[str], test_ids: Sequence[str]) -> None:
    """Verify strictly zero overlap across train, val, and test splits."""
    set_train = set(train_ids)
    set_val = set(val_ids)
    set_test = set(test_ids)

    leak_tv = set_train & set_val
    leak_tt = set_train & set_test
    leak_vt = set_val & set_test

    if leak_tv or leak_tt or leak_vt:
        err_msg = (
            f"CRITICAL: Split data leakage detected!\n"
            f"Train/Val Overlap ({len(leak_tv)}): {list(leak_tv)[:5]}\n"
            f"Train/Test Overlap ({len(leak_tt)}): {list(leak_tt)[:5]}\n"
            f"Val/Test Overlap ({len(leak_vt)}): {list(leak_vt)[:5]}"
        )
        logger.error(err_msg)
        raise ValueError(err_msg)


def get_sample_subsets(
    samples: Dict[str, MultimodalSample],
    splits: Dict[str, List[str]],
) -> Tuple[List[MultimodalSample], List[MultimodalSample], List[MultimodalSample]]:
    """Retrieve sample lists corresponding to train, val, and test partitions."""
    train_samples = [samples[uid] for uid in splits["train"] if uid in samples]
    val_samples = [samples[uid] for uid in splits["val"] if uid in samples]
    test_samples = [samples[uid] for uid in splits["test"] if uid in samples]
    return train_samples, val_samples, test_samples


class RawMultimodalDataset(TorchDataset):
    """
    Dataset wrapping raw samples for offline feature extraction (Stage 1).
    Iterates over metadata, audio paths, and video paths.
    """

    def __init__(self, samples: Sequence[MultimodalSample], use_context: bool = False):
        self.samples = list(samples)
        self.use_context = use_context

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        text = sample.utterance_text
        if self.use_context and sample.context_text:
            text = f"{sample.context_text} </s> {text}"

        return {
            "utterance_id": sample.utterance_id,
            "text": text,
            "audio_path": sample.audio_path,
            "video_path": sample.video_path,
            "context_audio_path": sample.context_audio_path,
            "context_video_path": sample.context_video_path,
            "label": sample.label,
            "speaker": sample.speaker or "",
            "show": sample.show or "",
            "audio_valid": sample.audio_valid,
            "video_valid": sample.video_valid,
        }


class MultimodalSarcasmDataset(TorchDataset):
    """
    Dataset for loading cached feature tensors during training and evaluation (Stage 2).
    Loads pre-extracted representations from Google Drive cache:
      - text_emb: (768,)
      - prosody_vec: (16,)
      - audio_emb: (768,)
      - video_emb: (2048,)
    Applies prosody normalization and modality masking based on active ablation mode.
    """

    def __init__(
        self,
        samples: Sequence[MultimodalSample],
        cache_dir: Union[str, Path],
        modalities: Sequence[str] = ("text", "speech", "video"),
        prosody_scaler: Optional[ProsodyScaler] = None,
        allow_dummy_features: bool = False,
    ):
        self.samples = list(samples)
        self.cache_dir = Path(cache_dir)
        self.modalities = set(modalities)
        self.prosody_scaler = prosody_scaler
        self.allow_dummy_features = allow_dummy_features

    def __len__(self) -> int:
        return len(self.samples)

    def _load_cached_tensor(
        self,
        utterance_id: str,
        key: str,
        expected_shape: Tuple[int, ...],
        default_val: float = 0.0,
    ) -> np.ndarray:
        """
        Load cached feature vector for an utterance.
        Supports both per-utterance .npz/.pt files and fallback synthetic generation.
        """
        sample_pt = self.cache_dir / f"{utterance_id}.pt"
        sample_npz = self.cache_dir / f"{utterance_id}.npz"

        if sample_pt.is_file() and TORCH_AVAILABLE:
            try:
                data = torch.load(sample_pt, map_location="cpu")
                if key in data:
                    t = data[key]
                    return t.cpu().numpy() if isinstance(t, torch.Tensor) else np.asarray(t)
            except Exception as e:
                logger.warning(f"Error reading {sample_pt}: {e}")

        if sample_npz.is_file():
            try:
                with np.load(sample_npz) as data:
                    if key in data:
                        return data[key].astype(np.float32)
            except Exception as e:
                logger.warning(f"Error reading {sample_npz}: {e}")

        # Check subfolder cache convention: cache_dir / {key} / {utterance_id}.npy
        modality_file = self.cache_dir / key / f"{utterance_id}.npy"
        if modality_file.is_file():
            try:
                return np.load(modality_file).astype(np.float32)
            except Exception as e:
                logger.warning(f"Error reading {modality_file}: {e}")

        if self.allow_dummy_features:
            # Deterministic mock tensor for testing pipeline without GPU / pre-extracted weights
            seed = (hash(f"{utterance_id}_{key}") % (2**31 - 1))
            rng = np.random.RandomState(seed)
            return rng.randn(*expected_shape).astype(np.float32)

        return np.full(expected_shape, fill_value=default_val, dtype=np.float32)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        uid = sample.utterance_id

        # Text embedding (microsoft/deberta-v3-base -> 768)
        text_active = "text" in self.modalities
        text_emb = (
            self._load_cached_tensor(uid, "text_emb", (768,))
            if text_active
            else np.zeros((768,), dtype=np.float32)
        )

        # Speech Prosody (16-dim explicit vector)
        speech_active = "speech" in self.modalities
        prosody_vec = (
            self._load_cached_tensor(uid, "prosody_vec", (16,))
            if speech_active
            else np.zeros((16,), dtype=np.float32)
        )
        if speech_active and self.prosody_scaler is not None:
            prosody_vec = self.prosody_scaler.transform(prosody_vec.reshape(1, -1)).squeeze(0)

        # Speech Acoustic embedding (facebook/wav2vec2-base -> 768)
        audio_emb = (
            self._load_cached_tensor(uid, "audio_emb", (768,))
            if speech_active
            else np.zeros((768,), dtype=np.float32)
        )

        # Video embedding (torchvision.models.resnet50 -> 2048)
        video_active = "video" in self.modalities
        video_emb = (
            self._load_cached_tensor(uid, "video_emb", (2048,))
            if video_active
            else np.zeros((2048,), dtype=np.float32)
        )

        return {
            "utterance_id": uid,
            "text_emb": text_emb,
            "prosody_vec": prosody_vec,
            "audio_emb": audio_emb,
            "video_emb": video_emb,
            "label": sample.label,
            "mask_text": 1.0 if text_active else 0.0,
            "mask_speech": 1.0 if speech_active else 0.0,
            "mask_video": 1.0 if video_active else 0.0,
        }
