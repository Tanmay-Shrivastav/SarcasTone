"""
Feature Cache Manager for SarcasTone.
Handles persistent storage of extracted multimodal feature tensors to Google Drive,
atomic manifest tracking, resume/skip support, and graceful fallbacks for missing media.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


# Standard expected feature shapes
TEXT_DIM = 768
PROSODY_DIM = 16
AUDIO_DIM = 768
VIDEO_DIM = 2048


class FeatureCacheManager:
    """
    Manages caching of multimodal feature representations to disk/Google Drive.
    Supports .pt (PyTorch) and .npz (NumPy compressed) serialization,
    idempotent resuming, and atomic manifest management.
    """

    def __init__(
        self,
        cache_dir: Union[str, Path],
        file_format: str = "pt",
        overwrite: bool = False,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.file_format = file_format.lower()
        if self.file_format not in ["pt", "npz"]:
            raise ValueError(f"Unsupported cache file format: {file_format}. Use 'pt' or 'npz'.")
        if self.file_format == "pt" and not TORCH_AVAILABLE:
            logger.warning("PyTorch not installed; falling back from 'pt' to 'npz' format.")
            self.file_format = "npz"

        self.overwrite = overwrite
        self.manifest_path = self.cache_dir / "manifest.json"
        self.manifest: Dict[str, Any] = self._load_manifest()

    def _load_manifest(self) -> Dict[str, Any]:
        """Load manifest from disk if it exists, or create a new one."""
        if self.manifest_path.is_file():
            try:
                with open(self.manifest_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Error loading manifest from {self.manifest_path}: {e}. Creating new manifest.")
        return {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "file_format": self.file_format,
            "feature_shapes": {
                "text_emb": [TEXT_DIM],
                "prosody_vec": [PROSODY_DIM],
                "audio_emb": [AUDIO_DIM],
                "video_emb": [VIDEO_DIM],
            },
            "samples": {},
        }

    def save_manifest(self) -> None:
        """Atomically persist manifest to disk."""
        self.manifest["last_updated"] = datetime.now(timezone.utc).isoformat()
        temp_path = self.manifest_path.with_suffix(".tmp")
        try:
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(self.manifest, f, indent=2)
            temp_path.replace(self.manifest_path)
        except Exception as e:
            logger.error(f"Failed to write manifest to {self.manifest_path}: {e}")

    def get_sample_path(self, utterance_id: str) -> Path:
        """Return expected file path for a cached sample."""
        ext = ".pt" if self.file_format == "pt" else ".npz"
        return self.cache_dir / f"{utterance_id}{ext}"

    def is_cached(self, utterance_id: str) -> bool:
        """
        Check if sample features are already cached and valid on disk.
        Returns False if overwrite=True.
        """
        if self.overwrite:
            return False

        # Check manifest first
        if utterance_id in self.manifest.get("samples", {}):
            sample_file = self.get_sample_path(utterance_id)
            if sample_file.is_file() and sample_file.stat().st_size > 0:
                return True

        # Check direct file existence as fallback
        pt_file = self.cache_dir / f"{utterance_id}.pt"
        npz_file = self.cache_dir / f"{utterance_id}.npz"
        if (pt_file.is_file() and pt_file.stat().st_size > 0) or (npz_file.is_file() and npz_file.stat().st_size > 0):
            return True

        return False

    def get_uncached_sample_ids(
        self,
        sample_ids: Sequence[str],
        show_progress: bool = False,
    ) -> List[str]:
        """
        Filter a list of sample IDs, returning only those not yet cached.
        Enables seamless resume of long-running Colab feature extraction.

        Args:
            sample_ids: Sequence of utterance IDs to evaluate.
            show_progress: If True, displays a tqdm progress bar with ETA.
        """
        if self.overwrite:
            return list(sample_ids)
        if show_progress:
            try:
                from tqdm.auto import tqdm
            except ImportError:
                from tqdm import tqdm
            import sys
            is_tty = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
            iterator = tqdm(
                sample_ids,
                desc="Checking cache status",
                unit="sample",
                leave=False,
                mininterval=1.0 if is_tty else 2.0,
                miniters=4,
            )
        else:
            iterator = sample_ids
        uncached = [uid for uid in iterator if not self.is_cached(uid)]
        logger.info(
            f"Cache status: {len(sample_ids) - len(uncached)}/{len(sample_ids)} samples already cached. "
            f"{len(uncached)} remaining to extract."
        )
        return uncached

    @staticmethod
    def create_empty_features(
        label: int = 0,
        audio_valid: bool = False,
        video_valid: bool = False,
    ) -> Dict[str, Any]:
        """
        Construct a valid dictionary of zeroed feature tensors for missing or unreadable media.
        Ensures consistent tensor shapes across the dataset.
        """
        return {
            "text_emb": np.zeros((TEXT_DIM,), dtype=np.float32),
            "prosody_vec": np.zeros((PROSODY_DIM,), dtype=np.float32),
            "audio_emb": np.zeros((AUDIO_DIM,), dtype=np.float32),
            "video_emb": np.zeros((VIDEO_DIM,), dtype=np.float32),
            "label": int(label),
            "audio_valid": bool(audio_valid),
            "video_valid": bool(video_valid),
            "metadata": {"fallback_empty": True},
        }

    def save_sample_features(
        self,
        utterance_id: str,
        features: Dict[str, Any],
        flush_manifest: bool = False,
    ) -> Path:
        """
        Save extracted features for a single utterance to disk.

        Args:
            utterance_id: Unique identifier for the utterance.
            features: Dictionary containing text_emb, prosody_vec, audio_emb, video_emb, etc.
            flush_manifest: If True, writes manifest to disk immediately.
        """
        # Ensure all array values are numpy arrays with float32 precision
        def _to_numpy(val: Any, expected_dim: int) -> np.ndarray:
            if TORCH_AVAILABLE and isinstance(val, torch.Tensor):
                arr = val.detach().cpu().numpy()
            elif isinstance(val, np.ndarray):
                arr = val
            elif isinstance(val, (list, tuple)):
                arr = np.array(val, dtype=np.float32)
            else:
                arr = np.zeros((expected_dim,), dtype=np.float32)
            return arr.astype(np.float32).reshape(-1)

        text_emb = _to_numpy(features.get("text_emb"), TEXT_DIM)
        prosody_vec = _to_numpy(features.get("prosody_vec"), PROSODY_DIM)
        audio_emb = _to_numpy(features.get("audio_emb"), AUDIO_DIM)
        video_emb = _to_numpy(features.get("video_emb"), VIDEO_DIM)

        label = int(features.get("label", 0))
        audio_valid = bool(features.get("audio_valid", True))
        video_valid = bool(features.get("video_valid", True))
        meta = features.get("metadata", {})

        target_path = self.get_sample_path(utterance_id)
        temp_path = self.cache_dir / f"{utterance_id}_tmp{target_path.suffix}"

        if self.file_format == "pt" and TORCH_AVAILABLE:
            data = {
                "utterance_id": utterance_id,
                "text_emb": torch.from_numpy(text_emb),
                "prosody_vec": torch.from_numpy(prosody_vec),
                "audio_emb": torch.from_numpy(audio_emb),
                "video_emb": torch.from_numpy(video_emb),
                "label": label,
                "audio_valid": audio_valid,
                "video_valid": video_valid,
                "metadata": meta,
            }
            torch.save(data, temp_path)
            temp_path.replace(target_path)
        else:
            np.savez_compressed(
                temp_path,
                utterance_id=utterance_id,
                text_emb=text_emb,
                prosody_vec=prosody_vec,
                audio_emb=audio_emb,
                video_emb=video_emb,
                label=label,
                audio_valid=audio_valid,
                video_valid=video_valid,
                metadata=json.dumps(meta),
            )
            temp_path.replace(target_path)

        # Update manifest record
        self.manifest.setdefault("samples", {})[utterance_id] = {
            "cached_at": datetime.now(timezone.utc).isoformat(),
            "label": label,
            "audio_valid": audio_valid,
            "video_valid": video_valid,
            "file": target_path.name,
        }

        if flush_manifest:
            self.save_manifest()

        return target_path

    def load_sample_features(self, utterance_id: str) -> Dict[str, Any]:
        """
        Load cached features for an utterance.
        Supports both .pt and .npz formats with fallback checks.
        """
        pt_path = self.cache_dir / f"{utterance_id}.pt"
        npz_path = self.cache_dir / f"{utterance_id}.npz"

        if pt_path.is_file() and TORCH_AVAILABLE:
            data = torch.load(pt_path, map_location="cpu")
            return {
                "utterance_id": utterance_id,
                "text_emb": data["text_emb"].cpu().numpy() if isinstance(data["text_emb"], torch.Tensor) else data["text_emb"],
                "prosody_vec": data["prosody_vec"].cpu().numpy() if isinstance(data["prosody_vec"], torch.Tensor) else data["prosody_vec"],
                "audio_emb": data["audio_emb"].cpu().numpy() if isinstance(data["audio_emb"], torch.Tensor) else data["audio_emb"],
                "video_emb": data["video_emb"].cpu().numpy() if isinstance(data["video_emb"], torch.Tensor) else data["video_emb"],
                "label": int(data.get("label", 0)),
                "audio_valid": bool(data.get("audio_valid", True)),
                "video_valid": bool(data.get("video_valid", True)),
                "metadata": data.get("metadata", {}),
            }

        if npz_path.is_file():
            with np.load(npz_path, allow_pickle=True) as data:
                meta = data.get("metadata")
                if isinstance(meta, np.ndarray) and meta.dtype.kind in ["U", "O"]:
                    try:
                        meta = json.loads(str(meta))
                    except Exception:
                        meta = {}
                return {
                    "utterance_id": utterance_id,
                    "text_emb": data["text_emb"].astype(np.float32),
                    "prosody_vec": data["prosody_vec"].astype(np.float32),
                    "audio_emb": data["audio_emb"].astype(np.float32),
                    "video_emb": data["video_emb"].astype(np.float32),
                    "label": int(data.get("label", 0)),
                    "audio_valid": bool(data.get("audio_valid", True)),
                    "video_valid": bool(data.get("video_valid", True)),
                    "metadata": meta or {},
                }

        raise FileNotFoundError(f"Cached feature file not found for utterance_id: '{utterance_id}' in {self.cache_dir}")

    def get_summary(self) -> Dict[str, Any]:
        """Return diagnostic overview of cached features."""
        samples_info = self.manifest.get("samples", {})
        total = len(samples_info)
        audio_ok = sum(1 for s in samples_info.values() if s.get("audio_valid", False))
        video_ok = sum(1 for s in samples_info.values() if s.get("video_valid", False))
        sarcastic = sum(1 for s in samples_info.values() if s.get("label", 0) == 1)

        return {
            "total_cached": total,
            "audio_valid_count": audio_ok,
            "video_valid_count": video_ok,
            "sarcastic_count": sarcastic,
            "non_sarcastic_count": total - sarcastic,
            "cache_dir": str(self.cache_dir),
            "file_format": self.file_format,
        }
