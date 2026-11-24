"""
Data loading, splitting, and collation modules for SarcasTone.
Supports MUStARD and MUStARD++ spoken audiovisual sarcasm corpora.
"""

from src.data.dataset import (
    MultimodalSample,
    RawMultimodalDataset,
    MultimodalSarcasmDataset,
    ProsodyScaler,
    load_raw_samples,
    create_or_load_splits,
    get_sample_subsets,
)
from src.data.collate import (
    MultimodalBatch,
    multimodal_collate_fn,
    create_dataloader,
)

__all__ = [
    "MultimodalSample",
    "RawMultimodalDataset",
    "MultimodalSarcasmDataset",
    "ProsodyScaler",
    "load_raw_samples",
    "create_or_load_splits",
    "get_sample_subsets",
    "MultimodalBatch",
    "multimodal_collate_fn",
    "create_dataloader",
]
