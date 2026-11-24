"""
Feature extraction and caching package for SarcasTone.
Handles offline extraction and persistent Google Drive caching for
Text (DeBERTa-v3), Speech Prosody & Acoustics (Librosa + Wav2Vec2), and Video (ResNet-50).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Sequence

from tqdm import tqdm

from src.features.cache_manager import (
    AUDIO_DIM,
    PROSODY_DIM,
    TEXT_DIM,
    VIDEO_DIM,
    FeatureCacheManager,
)
from src.features.prosody_features import (
    ProsodyAcousticExtractor,
    extract_audio_from_media,
    extract_explicit_prosody,
    extract_wav_from_video,
)
from src.features.text_features import (
    TextFeatureExtractor,
)
from src.features.video_features import (
    VideoFeatureExtractor,
    sample_frames_from_video,
)

logger = logging.getLogger(__name__)

__all__ = [
    "FeatureCacheManager",
    "TextFeatureExtractor",
    "ProsodyAcousticExtractor",
    "VideoFeatureExtractor",
    "extract_explicit_prosody",
    "extract_audio_from_media",
    "extract_wav_from_video",
    "sample_frames_from_video",
    "extract_and_cache_dataset",
    "TEXT_DIM",
    "PROSODY_DIM",
    "AUDIO_DIM",
    "VIDEO_DIM",
]


def extract_and_cache_dataset(
    samples: Dict[str, Any],
    cache_manager: FeatureCacheManager,
    config: Dict[str, Any],
    device: Optional[str] = None,
    batch_size_text: int = 32,
) -> Dict[str, Any]:
    """
    High-level orchestrator for Stage 1 offline feature extraction on Google Colab T4.
    Extracts features for all samples in `samples`, safely skipping already-cached items,
    and persisting results to Google Drive.

    Args:
        samples: Mapping of utterance_id -> MultimodalSample or dict.
        cache_manager: Initialized FeatureCacheManager.
        config: Full configuration dict (from configs/default.yaml).
        device: 'cuda' or 'cpu'. Defaults to cuda if available.
        batch_size_text: Mini-batch size for text transformer feature extraction.

    Returns:
        Summary dict of extraction results.
    """
    enc_cfg = config.get("encoders", {})
    pros_cfg = config.get("prosody", {})
    ds_cfg = config.get("dataset", {})

    all_ids = list(samples.keys())
    to_extract_ids = cache_manager.get_uncached_sample_ids(all_ids)

    if not to_extract_ids:
        logger.info("All samples already cached! Skipping feature extraction.")
        return cache_manager.get_summary()

    logger.info(f"Beginning feature extraction for {len(to_extract_ids)} samples...")

    # Initialize extractors
    text_extractor = TextFeatureExtractor(
        model_name=enc_cfg.get("text_model", "microsoft/deberta-v3-base"),
        fallback_model=enc_cfg.get("text_fallback_model", "roberta-base"),
        max_length=enc_cfg.get("max_text_length", 128),
        use_context=ds_cfg.get("use_context", False),
        device=device,
        use_amp=config.get("training", {}).get("use_amp", True),
    )

    audio_extractor = ProsodyAcousticExtractor(
        audio_model=enc_cfg.get("audio_model", "facebook/wav2vec2-base"),
        sample_rate=enc_cfg.get("sample_rate", 16000),
        max_audio_sec=enc_cfg.get("max_audio_sec", 15.0),
        fmin=pros_cfg.get("pitch", {}).get("fmin", 65.0),
        fmax=pros_cfg.get("pitch", {}).get("fmax", 500.0),
        frame_length=pros_cfg.get("energy", {}).get("frame_length", 2048),
        hop_length=pros_cfg.get("energy", {}).get("hop_length", 512),
        silence_threshold_db=pros_cfg.get("pause", {}).get("silence_threshold_db", 35.0),
        device=device,
        use_amp=config.get("training", {}).get("use_amp", True),
    )

    video_extractor = VideoFeatureExtractor(
        model_name=enc_cfg.get("video_model", "resnet50"),
        num_frames=enc_cfg.get("num_video_frames", 8),
        frame_size=enc_cfg.get("frame_size", 224),
        device=device,
        use_amp=config.get("training", {}).get("use_amp", True),
    )

    # 1. Batch extract text features for efficiency
    sample_objs = [samples[uid] for uid in to_extract_ids]
    texts = [getattr(s, "utterance_text", s.get("utterance_text", "")) for s in sample_objs]
    contexts = [getattr(s, "context_text", s.get("context_text", None)) for s in sample_objs]

    logger.info("Extracting text representations in batches...")
    text_embeddings = text_extractor.extract_batch(texts, contexts=contexts, batch_size=batch_size_text)

    # 2. Extract audio and video representations per utterance
    logger.info("Extracting speech prosody, acoustic, and video frame representations...")
    for i, uid in enumerate(tqdm(to_extract_ids, desc="Extracting Multimodal Features")):
        sample = samples[uid]
        text_emb = text_embeddings[i]

        audio_path = getattr(sample, "audio_path", sample.get("audio_path", None))
        video_path = getattr(sample, "video_path", sample.get("video_path", None))
        label = getattr(sample, "label", sample.get("label", 0))

        # Audio / Prosody extraction
        prosody_vec, audio_emb, audio_valid = audio_extractor.extract_from_file(audio_path)

        # Video extraction
        video_emb, video_valid = video_extractor.extract_from_file(video_path)

        features = {
            "text_emb": text_emb,
            "prosody_vec": prosody_vec,
            "audio_emb": audio_emb,
            "video_emb": video_emb,
            "label": label,
            "audio_valid": audio_valid,
            "video_valid": video_valid,
            "metadata": {
                "utterance_id": uid,
                "audio_path": str(audio_path) if audio_path else None,
                "video_path": str(video_path) if video_path else None,
            },
        }

        # Flush manifest every 50 samples or on final sample
        flush_manifest = (i % 50 == 0) or (i == len(to_extract_ids) - 1)
        cache_manager.save_sample_features(uid, features, flush_manifest=flush_manifest)

    logger.info("Multimodal feature extraction complete!")
    return cache_manager.get_summary()
