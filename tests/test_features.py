"""
Unit tests for the feature extraction and caching layer.
Tests FeatureCacheManager, explicit prosody extraction, video frame sampling,
and text input formatting.
"""

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.features.cache_manager import (
    AUDIO_DIM,
    PROSODY_DIM,
    TEXT_DIM,
    VIDEO_DIM,
    FeatureCacheManager,
)
from src.features.prosody_features import (
    ProsodyAcousticExtractor,
    compute_linear_slope,
    extract_audio_from_media,
    extract_explicit_prosody,
    extract_wav_from_video,
)
from src.features.text_features import TextFeatureExtractor
from src.features.video_features import sample_frames_from_video


def test_linear_slope():
    """Test linear slope computation on synthetic sequences."""
    flat = np.array([5.0, 5.0, 5.0, 5.0], dtype=np.float32)
    assert compute_linear_slope(flat) == 0.0

    ascending = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    assert pytest.approx(compute_linear_slope(ascending), rel=1e-3) == 1.0

    descending = np.array([10.0, 8.0, 6.0, 4.0], dtype=np.float32)
    assert pytest.approx(compute_linear_slope(descending), rel=1e-3) == -2.0

    # Short sequence
    assert compute_linear_slope(np.array([1.0])) == 0.0


def test_extract_explicit_prosody_synthetic_audio():
    """Test 16-dim prosody extraction on synthetic sine wave audio with silence."""
    sr = 16000
    duration = 1.0  # 1 second
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # 220 Hz sine wave for first 0.7s, followed by 0.3s silence
    audio = np.sin(2 * np.pi * 220.0 * t).astype(np.float32)
    audio[int(sr * 0.7) :] = 0.0

    prosody_vec = extract_explicit_prosody(
        audio=audio,
        sr=sr,
        fmin=65.0,
        fmax=500.0,
    )

    assert isinstance(prosody_vec, np.ndarray)
    assert prosody_vec.shape == (PROSODY_DIM,)
    assert not np.isnan(prosody_vec).any()
    assert not np.isinf(prosody_vec).any()

    # Check duration feature (index 15) matches audio length
    assert pytest.approx(prosody_vec[15], rel=1e-2) == 1.0

    # Empty audio should yield zeros
    empty_vec = extract_explicit_prosody(np.array([], dtype=np.float32), sr=sr)
    assert empty_vec.shape == (PROSODY_DIM,)
    assert np.all(empty_vec == 0.0)


def test_sample_frames_from_synthetic_video(tmp_path):
    """Test video frame sampling using a synthetically created video clip."""
    video_path = tmp_path / "test_clip.mp4"
    frame_w, frame_h = 320, 240
    fps = 24
    num_test_frames = 48  # 2 seconds

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(video_path), fourcc, fps, (frame_w, frame_h))

    for i in range(num_test_frames):
        # Create distinct color frames
        frame = np.full((frame_h, frame_w, 3), fill_value=(i * 5) % 255, dtype=np.uint8)
        out.write(frame)
    out.release()

    frames, valid = sample_frames_from_video(
        video_path=video_path,
        num_frames=8,
        frame_size=224,
    )

    assert valid is True
    assert frames.shape == (8, 3, 224, 224)
    assert not np.isnan(frames).any()

    # Test missing video
    missing_frames, missing_valid = sample_frames_from_video(
        video_path=tmp_path / "non_existent.mp4",
        num_frames=8,
        frame_size=224,
    )
    assert missing_valid is False
    assert missing_frames.shape == (8, 3, 224, 224)
    assert np.all(missing_frames == 0.0)


def test_text_feature_formatting():
    """Test text input formatting with and without dialogue context."""
    extractor = TextFeatureExtractor(use_context=False)
    res = extractor.format_input_text("What a great day!", context="It is raining outside.")
    assert res == "What a great day!"

    extractor_ctx = TextFeatureExtractor(use_context=True)
    res_ctx = extractor_ctx.format_input_text("What a great day!", context="It is raining outside.")
    assert res_ctx == "It is raining outside. </s> What a great day!"


def test_feature_cache_manager(tmp_path):
    """Test saving, loading, manifest tracking, and resume in FeatureCacheManager."""
    cache_mgr = FeatureCacheManager(cache_dir=tmp_path, file_format="npz")

    uid_1 = "sample_001"
    features_1 = {
        "text_emb": np.random.randn(TEXT_DIM).astype(np.float32),
        "prosody_vec": np.random.randn(PROSODY_DIM).astype(np.float32),
        "audio_emb": np.random.randn(AUDIO_DIM).astype(np.float32),
        "video_emb": np.random.randn(VIDEO_DIM).astype(np.float32),
        "label": 1,
        "audio_valid": True,
        "video_valid": True,
        "metadata": {"test_key": "val1"},
    }

    assert not cache_mgr.is_cached(uid_1)
    saved_path = cache_mgr.save_sample_features(uid_1, features_1, flush_manifest=True)
    assert saved_path.is_file()
    assert cache_mgr.is_cached(uid_1)

    # Resume test
    uids = ["sample_001", "sample_002", "sample_003"]
    uncached = cache_mgr.get_uncached_sample_ids(uids)
    assert uncached == ["sample_002", "sample_003"]

    # Resume test with tqdm progress bar enabled
    uncached_pbar = cache_mgr.get_uncached_sample_ids(uids, show_progress=True)
    assert uncached_pbar == ["sample_002", "sample_003"]

    # Load and verify content
    loaded = cache_mgr.load_sample_features(uid_1)
    assert loaded["utterance_id"] == uid_1
    assert loaded["label"] == 1
    assert loaded["audio_valid"] is True
    assert loaded["video_valid"] is True
    np.testing.assert_allclose(loaded["text_emb"], features_1["text_emb"], rtol=1e-5)
    np.testing.assert_allclose(loaded["prosody_vec"], features_1["prosody_vec"], rtol=1e-5)
    np.testing.assert_allclose(loaded["audio_emb"], features_1["audio_emb"], rtol=1e-5)
    np.testing.assert_allclose(loaded["video_emb"], features_1["video_emb"], rtol=1e-5)
    assert loaded["metadata"]["test_key"] == "val1"

    # Empty feature fallback test
    uid_2 = "sample_002"
    fallback_features = cache_mgr.create_empty_features(label=0, audio_valid=False, video_valid=False)
    cache_mgr.save_sample_features(uid_2, fallback_features, flush_manifest=True)

    loaded_fallback = cache_mgr.load_sample_features(uid_2)
    assert loaded_fallback["label"] == 0
    assert loaded_fallback["audio_valid"] is False
    assert loaded_fallback["video_valid"] is False
    assert np.all(loaded_fallback["audio_emb"] == 0.0)

    # Diagnostic summary test
    summary = cache_mgr.get_summary()
    assert summary["total_cached"] == 2
    assert summary["sarcastic_count"] == 1
    assert summary["audio_valid_count"] == 1
    assert summary["video_valid_count"] == 1


def test_extract_audio_from_media_and_load_audio(tmp_path):
    """Test extract_audio_from_media and ProsodyAcousticExtractor.load_audio."""
    from scipy.io import wavfile

    # 1. Create a synthetic test .wav file (16 kHz, 0.5s)
    sr = 16000
    duration = 0.5
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    synthetic_pcm = (np.sin(2 * np.pi * 440.0 * t) * 32767.0).astype(np.int16)

    wav_path = tmp_path / "test_audio.wav"
    wavfile.write(str(wav_path), sr, synthetic_pcm)

    # 2. Extract using extract_audio_from_media
    audio_arr, valid = extract_audio_from_media(wav_path, target_sr=16000)
    assert valid is True
    assert isinstance(audio_arr, np.ndarray)
    assert audio_arr.dtype == np.float32
    assert len(audio_arr) == int(sr * duration)
    assert np.max(np.abs(audio_arr)) > 0.5

    # 3. Test extract_wav_from_video utility
    out_wav = tmp_path / "extracted_copy.wav"
    assert extract_wav_from_video(wav_path, out_wav, target_sr=16000) is True
    assert out_wav.is_file()

    # 4. Test ProsodyAcousticExtractor.load_audio
    extractor = ProsodyAcousticExtractor(sample_rate=16000, max_audio_sec=15.0)
    loaded_audio, loaded_valid = extractor.load_audio(wav_path)
    assert loaded_valid is True
    assert len(loaded_audio) == int(sr * duration)

    # 5. Non-existent file handling
    missing_audio, missing_valid = extractor.load_audio(tmp_path / "non_existent.wav")
    assert missing_valid is False
    assert len(missing_audio) == 16000
    assert np.all(missing_audio == 0.0)
