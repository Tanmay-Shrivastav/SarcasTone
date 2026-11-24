"""
Multimodal batch collation and DataLoader construction for SarcasTone.
Handles dynamic modality masking, stacking of heterogeneous tensors,
and device transfer for Colab T4 GPU execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

try:
    import torch
    from torch.utils.data import DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    DataLoader = object  # type: ignore
    torch = None  # type: ignore
    TORCH_AVAILABLE = False


@dataclass
class MultimodalBatch:
    """
    Collation container for a batch of multimodal samples.
    """
    utterance_ids: List[str]
    text_emb: Any        # torch.Tensor of shape (B, 768) or np.ndarray
    prosody_vec: Any     # torch.Tensor of shape (B, 16) or np.ndarray
    audio_emb: Any       # torch.Tensor of shape (B, 768) or np.ndarray
    video_emb: Any       # torch.Tensor of shape (B, 2048) or np.ndarray
    labels: Any          # torch.Tensor of shape (B, 1) or np.ndarray
    mask_text: Any       # torch.Tensor of shape (B, 1)
    mask_speech: Any     # torch.Tensor of shape (B, 1)
    mask_video: Any      # torch.Tensor of shape (B, 1)

    def __len__(self) -> int:
        return len(self.utterance_ids)

    def to(self, device: Any) -> MultimodalBatch:
        """Move all batch tensors to target device (e.g., 'cuda:0' on Colab T4)."""
        if not TORCH_AVAILABLE:
            return self

        return MultimodalBatch(
            utterance_ids=self.utterance_ids,
            text_emb=self.text_emb.to(device) if hasattr(self.text_emb, "to") else self.text_emb,
            prosody_vec=self.prosody_vec.to(device) if hasattr(self.prosody_vec, "to") else self.prosody_vec,
            audio_emb=self.audio_emb.to(device) if hasattr(self.audio_emb, "to") else self.audio_emb,
            video_emb=self.video_emb.to(device) if hasattr(self.video_emb, "to") else self.video_emb,
            labels=self.labels.to(device) if hasattr(self.labels, "to") else self.labels,
            mask_text=self.mask_text.to(device) if hasattr(self.mask_text, "to") else self.mask_text,
            mask_speech=self.mask_speech.to(device) if hasattr(self.mask_speech, "to") else self.mask_speech,
            mask_video=self.mask_video.to(device) if hasattr(self.mask_video, "to") else self.mask_video,
        )


def multimodal_collate_fn(batch_items: Sequence[Dict[str, Any]]) -> MultimodalBatch:
    """
    Collates a list of sample dictionaries into a batched MultimodalBatch object.

    Args:
        batch_items: List of dictionaries from MultimodalSarcasmDataset.

    Returns:
        MultimodalBatch instance containing batched tensors and modality masks.
    """
    utterance_ids = [item["utterance_id"] for item in batch_items]
    labels_raw = [float(item["label"]) for item in batch_items]

    text_list = [item["text_emb"] for item in batch_items]
    prosody_list = [item["prosody_vec"] for item in batch_items]
    audio_list = [item["audio_emb"] for item in batch_items]
    video_list = [item["video_emb"] for item in batch_items]

    mask_text_list = [float(item["mask_text"]) for item in batch_items]
    mask_speech_list = [float(item["mask_speech"]) for item in batch_items]
    mask_video_list = [float(item["mask_video"]) for item in batch_items]

    if TORCH_AVAILABLE:
        def _to_tensor(arrays: List[Any], dtype: Any = torch.float32) -> torch.Tensor:
            if isinstance(arrays[0], torch.Tensor):
                return torch.stack(arrays).to(dtype)
            arr = np.stack(arrays, axis=0)
            return torch.from_numpy(arr).to(dtype)

        text_emb = _to_tensor(text_list)
        prosody_vec = _to_tensor(prosody_list)
        audio_emb = _to_tensor(audio_list)
        video_emb = _to_tensor(video_list)
        labels = torch.tensor(labels_raw, dtype=torch.float32).unsqueeze(1)

        mask_text = torch.tensor(mask_text_list, dtype=torch.float32).unsqueeze(1)
        mask_speech = torch.tensor(mask_speech_list, dtype=torch.float32).unsqueeze(1)
        mask_video = torch.tensor(mask_video_list, dtype=torch.float32).unsqueeze(1)
    else:
        text_emb = np.stack(text_list, axis=0)
        prosody_vec = np.stack(prosody_list, axis=0)
        audio_emb = np.stack(audio_list, axis=0)
        video_emb = np.stack(video_list, axis=0)
        labels = np.array(labels_raw, dtype=np.float32).reshape(-1, 1)

        mask_text = np.array(mask_text_list, dtype=np.float32).reshape(-1, 1)
        mask_speech = np.array(mask_speech_list, dtype=np.float32).reshape(-1, 1)
        mask_video = np.array(mask_video_list, dtype=np.float32).reshape(-1, 1)

    return MultimodalBatch(
        utterance_ids=utterance_ids,
        text_emb=text_emb,
        prosody_vec=prosody_vec,
        audio_emb=audio_emb,
        video_emb=video_emb,
        labels=labels,
        mask_text=mask_text,
        mask_speech=mask_speech,
        mask_video=mask_video,
    )


def create_dataloader(
    dataset: Any,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 2,
    pin_memory: bool = True,
) -> Any:
    """Constructs a PyTorch DataLoader wrapping MultimodalSarcasmDataset."""
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required to create a DataLoader.")

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        collate_fn=multimodal_collate_fn,
    )
