"""
Tests for dataset ingestion, deterministic splitting, prosody scaler,
and multimodal batch collation.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.data.dataset import (
    MultimodalSample,
    ProsodyScaler,
    create_or_load_splits,
    get_sample_subsets,
    load_raw_samples,
    verify_leakage,
    MultimodalSarcasmDataset,
)
from src.data.collate import multimodal_collate_fn


@pytest.fixture
def dummy_samples():
    """Generates 100 deterministic dummy samples for testing."""
    samples = {}
    rng = np.random.RandomState(42)
    speakers = ["SPEAKER_A", "SPEAKER_B", "SPEAKER_C"]
    for i in range(100):
        uid = f"utt_{i:03d}"
        samples[uid] = MultimodalSample(
            utterance_id=uid,
            utterance_text=f"Utterance text for sample {i}",
            context_text=f"Preceding context {i}",
            speaker=speakers[i % len(speakers)],
            show="test_show",
            label=int(rng.choice([0, 1])),
        )
    return samples


def test_zero_leakage_and_splits(dummy_samples, tmp_path):
    """Test deterministic split creation and zero leakage verification."""
    splits_file = tmp_path / "splits.json"

    splits = create_or_load_splits(
        samples=dummy_samples,
        splits_path=splits_file,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42,
        overwrite=True,
    )

    train_set = set(splits["train"])
    val_set = set(splits["val"])
    test_set = set(splits["test"])

    # Disjointness checks
    assert len(train_set & val_set) == 0, "Train and Val overlap!"
    assert len(train_set & test_set) == 0, "Train and Test overlap!"
    assert len(val_set & test_set) == 0, "Val and Test overlap!"

    # Completeness check
    assert len(train_set | val_set | test_set) == 100
    assert abs(len(train_set) - 70) <= 2
    assert abs(len(val_set) - 15) <= 2
    assert abs(len(test_set) - 15) <= 2

    # Verification function should pass without exception
    verify_leakage(splits["train"], splits["val"], splits["test"])

    # Reloading from disk should produce identical splits
    reloaded_splits = create_or_load_splits(
        samples=dummy_samples,
        splits_path=splits_file,
        overwrite=False,
    )
    assert reloaded_splits["train"] == splits["train"]
    assert reloaded_splits["val"] == splits["val"]
    assert reloaded_splits["test"] == splits["test"]


def test_leakage_detection_raises():
    """Test that verify_leakage raises ValueError when overlap exists."""
    with pytest.raises(ValueError):
        verify_leakage(train_ids=["1", "2", "3"], val_ids=["3", "4"], test_ids=["5"])


def test_prosody_scaler(tmp_path):
    """Test ProsodyScaler fit, transform, serialization, and deserialization."""
    features = np.random.randn(50, 16).astype(np.float32)
    # Inject NaN and Inf to verify safe handling
    features[0, 0] = np.nan
    features[1, 1] = np.inf

    scaler = ProsodyScaler(feature_dim=16)
    scaled = scaler.fit_transform(features)

    assert not np.isnan(scaled).any()
    assert not np.isinf(scaled).any()
    assert scaled.shape == (50, 16)

    # Save and reload
    scaler_file = tmp_path / "scaler.json"
    scaler.save(scaler_file)

    loaded_scaler = ProsodyScaler.load(scaler_file)
    assert loaded_scaler.is_fit
    test_vec = np.ones((5, 16), dtype=np.float32)
    np.testing.assert_allclose(scaler.transform(test_vec), loaded_scaler.transform(test_vec))


def test_dataset_and_collate(dummy_samples, tmp_path):
    """Test MultimodalSarcasmDataset with dummy features and multimodal collation."""
    train_samples = list(dummy_samples.values())[:20]
    dataset = MultimodalSarcasmDataset(
        samples=train_samples,
        cache_dir=tmp_path,
        modalities=["text", "speech"],
        allow_dummy_features=True,
    )

    assert len(dataset) == 20
    item = dataset[0]
    assert "utterance_id" in item
    assert item["text_emb"].shape == (768,)
    assert item["prosody_vec"].shape == (16,)
    assert item["audio_emb"].shape == (768,)
    assert item["mask_text"] == 1.0
    assert item["mask_speech"] == 1.0
    assert item["mask_video"] == 0.0  # Video not in active modalities

    # Batch collation
    batch_items = [dataset[i] for i in range(4)]
    batch = multimodal_collate_fn(batch_items)

    assert len(batch) == 4
    assert batch.text_emb.shape == (4, 768)
    assert batch.prosody_vec.shape == (4, 16)
    assert batch.audio_emb.shape == (4, 768)
    assert batch.video_emb.shape == (4, 2048)
    assert batch.labels.shape == (4, 1)
    assert batch.mask_text.shape == (4, 1)


def test_mustard_directory_structure_resolution(tmp_path):
    """Test loading MUStARD dataset with utterances_final/<id>.mp4 and context_final/<id>_c.mp4."""
    data_root = tmp_path / "mustard_data"
    utterances_dir = data_root / "utterances_final"
    context_dir = data_root / "context_final"
    utterances_dir.mkdir(parents=True, exist_ok=True)
    context_dir.mkdir(parents=True, exist_ok=True)

    # Create dummy .mp4 files
    (utterances_dir / "1_60.mp4").write_text("dummy_utterance_video")
    (context_dir / "1_60_c.mp4").write_text("dummy_context_video")
    (utterances_dir / "2_105.mp4").write_text("dummy_utterance_video_2")

    ann_data = {
        "1_60": {
            "utterance": "It's just a joke.",
            "speaker": "CHANDLER",
            "context": ["What are you doing?", "Nothing."],
            "context_speakers": ["MONICA", "CHANDLER"],
            "show": "FRIENDS",
            "sarcasm": True,
        },
        "2_105": {
            "utterance": "Very funny.",
            "speaker": "SHELDON",
            "context": ["Look at that."],
            "context_speakers": ["LEONARD"],
            "show": "BBT",
            "sarcasm": False,
        },
    }
    with open(data_root / "annotations.json", "w", encoding="utf-8") as f:
        json.dump(ann_data, f)

    samples = load_raw_samples(data_root=data_root)

    assert len(samples) == 2
    s1 = samples["1_60"]
    assert s1.utterance_id == "1_60"
    assert s1.label == 1
    assert s1.speaker == "CHANDLER"
    assert s1.show == "FRIENDS"
    assert s1.video_path is not None and s1.video_path.endswith("1_60.mp4")
    # Automatic fallback of audio to .mp4 video clip
    assert s1.audio_path is not None and s1.audio_path.endswith("1_60.mp4")
    assert s1.context_video_path is not None and s1.context_video_path.endswith("1_60_c.mp4")
    assert s1.context_audio_path is not None and s1.context_audio_path.endswith("1_60_c.mp4")
    assert s1.audio_valid is True
    assert s1.video_valid is True

    s2 = samples["2_105"]
    assert s2.label == 0
    assert s2.video_path is not None and s2.video_path.endswith("2_105.mp4")
    assert s2.audio_path is not None and s2.audio_path.endswith("2_105.mp4")
    assert s2.audio_valid is True
    assert s2.video_valid is True
    assert s2.context_video_path is None
