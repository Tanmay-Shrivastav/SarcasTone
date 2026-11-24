"""
tests/test_web_app.py
=====================
Integration tests for SarcasTone FastAPI backend and frontend assets.
"""

from pathlib import Path
import pytest
from pydantic import BaseModel

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_frontend_assets_exist():
    """Verify frontend HTML, static demo data, and media clips exist."""
    frontend_dir = REPO_ROOT / "frontend"
    assert (frontend_dir / "index.html").is_file(), "Missing frontend/index.html"
    assert (frontend_dir / "data" / "demo_data.json").is_file(), "Missing demo_data.json"

    clips_dir = frontend_dir / "assets" / "clips"
    for clip in ["chandler", "sheldon", "ross", "leonard"]:
        assert (clips_dir / f"{clip}.mp4").is_file(), f"Missing {clip}.mp4"
        assert (clips_dir / f"{clip}.wav").is_file(), f"Missing {clip}.wav"
        assert (clips_dir / f"{clip}_poster.jpg").is_file(), f"Missing {clip}_poster.jpg"


def test_api_endpoints_direct():
    """Verify API endpoint functions directly without needing active network ports."""
    from api.server import (
        engine,
        get_health,
        get_results,
        get_samples,
        get_sample_by_id,
        post_predict,
        PredictRequest,
    )

    assert engine.is_loaded is True

    # 1. Health
    health = get_health()
    assert health["status"] == "ok"
    assert health["model_loaded"] is True

    # 2. Results
    results = get_results()
    assert len(results["runs"]) == 8
    assert results["champion"]["run_name"] == "full_multimodal_pgin"
    assert abs(results["champion"]["f1"] - 0.7563) < 0.001
    assert abs(results["champion"]["accuracy"] - 0.7212) < 0.001

    # 3. Samples
    samples = get_samples()
    assert len(samples["keys"]) >= 4
    for key in ["chandler", "sheldon", "ross"]:
        assert key in samples["keys"]
        s = get_sample_by_id(key)
        assert s["id"] == key
        assert "caption" in s
        assert "prediction" in s

    # 4. Predict
    pred = post_predict(PredictRequest(sample_id="sheldon"))
    assert pred["prediction"] == "Sarcastic"
    assert pred["confidence"] > 0.80

    pred_literal = post_predict(PredictRequest(sample_id="ross"))
    assert pred_literal["prediction"] == "Literal"

    # 5. Custom predict
    pred_custom = post_predict(PredictRequest(
        text="Oh brilliant, that is just what we needed.",
        prosody_pitch_slope=-0.8,
        prosody_energy_delta=0.3,
    ))
    assert "gate_weights" in pred_custom
    assert "dominant_gate" in pred_custom


def test_predict_custom_regression_numpy_direct():
    """Regression test: verify POST /api/predict returns native Python types and serializes to 200 OK without TypeError."""
    import numpy as np
    from fastapi.encoders import jsonable_encoder
    from fastapi.responses import JSONResponse
    from api.server import post_predict, PredictRequest, engine

    assert engine.engine_mode == "numpy_direct" or engine.is_loaded

    req = PredictRequest(
        text="Oh yeah, that's totally going to work out brilliantly for everyone involved.",
        prosody_pitch_slope=-0.75,
        prosody_energy_delta=0.40,
        modality_mode="full_multimodal_pgin",
    )
    res = post_predict(req)

    # Verify all returned values are pure native Python types, NOT NumPy scalars
    assert isinstance(res["is_sarcastic"], bool)
    assert type(res["is_sarcastic"]) is bool
    assert not isinstance(res["is_sarcastic"], np.bool_)

    assert isinstance(res["confidence"], float)
    assert not isinstance(res["confidence"], np.floating)

    assert isinstance(res["logit"], float)
    assert not isinstance(res["logit"], np.floating)

    assert isinstance(res["incongruity_score"], float)
    assert not isinstance(res["incongruity_score"], np.floating)

    assert isinstance(res["polarity_text"], float)
    assert not isinstance(res["polarity_text"], np.floating)

    assert isinstance(res["polarity_speech"], float)
    assert not isinstance(res["polarity_speech"], np.floating)

    assert isinstance(res["gate_weights"], dict)
    for g_name, g_val in res["gate_weights"].items():
        assert isinstance(g_val, float)
        assert not isinstance(g_val, np.floating)

    # Recursive validation across the entire dictionary
    def assert_no_numpy(data, path="root"):
        if isinstance(data, dict):
            for k, v in data.items():
                assert_no_numpy(v, f"{path}.{k}")
        elif isinstance(data, (list, tuple)):
            for idx, v in enumerate(data):
                assert_no_numpy(v, f"{path}[{idx}]")
        else:
            assert not isinstance(data, (np.generic, np.ndarray)), f"Found NumPy type {type(data)} at {path}"

    assert_no_numpy(res)

    # Verify jsonable_encoder succeeds without TypeError: 'numpy.bool' object is not iterable
    encoded = jsonable_encoder(res)
    assert isinstance(encoded, dict)
    assert encoded["is_sarcastic"] in (True, False)
    assert type(encoded["is_sarcastic"]) is bool

    # Verify FastAPI JSONResponse produces 200 OK
    response = JSONResponse(content=encoded)
    assert response.status_code == 200
    assert b"is_sarcastic" in response.body

