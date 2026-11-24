#!/usr/bin/env python3
"""
run_app.py
==========
Single-command launcher for the SarcasTone web application.

Usage:
  python run_app.py             # Launches server on http://localhost:8000
  python run_app.py --port 8080 # Launches on custom port
  python run_app.py --test      # Runs quick verification of all endpoints and exits
"""

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def run_self_test():
    """Validates model loading, asset files, and API endpoints."""
    print("=" * 65)
    print("  SARCASTONE WEB APPLICATION — VERIFICATION SUITE")
    print("=" * 65)

    # 1. Check Model Checkpoint
    ckpt_path = REPO_ROOT / "results" / "full_multimodal_pgin" / "best_model.pt"
    assert ckpt_path.is_file(), f"Missing checkpoint: {ckpt_path}"
    print(f"  [PASS] Checkpoint located: {ckpt_path.name} ({ckpt_path.stat().st_size / 1e6:.1f} MB)")

    # 2. Check Ablation Results
    ablation_path = REPO_ROOT / "results" / "ablation_summary.json"
    assert ablation_path.is_file(), f"Missing results: {ablation_path}"
    print(f"  [PASS] Benchmark results located: {ablation_path.name}")

    # 3. Check Demo Data and Media Clips
    demo_data_path = REPO_ROOT / "frontend" / "data" / "demo_data.json"
    assert demo_data_path.is_file(), f"Missing fallback data: {demo_data_path}"
    print(f"  [PASS] Static fallback bundle located: {demo_data_path.name}")

    for clip in ["chandler", "sheldon", "ross", "leonard"]:
        mp4 = REPO_ROOT / "frontend" / "assets" / "clips" / f"{clip}.mp4"
        wav = REPO_ROOT / "frontend" / "assets" / "clips" / f"{clip}.wav"
        assert mp4.is_file(), f"Missing clip video: {mp4}"
        assert wav.is_file(), f"Missing clip audio: {wav}"
    print("  [PASS] All 4 MUStARD audiovisual demo clips verified.")

    # 4. Check Frontend HTML
    index_html = REPO_ROOT / "frontend" / "index.html"
    assert index_html.is_file(), f"Missing frontend index: {index_html}"
    print("  [PASS] Editorial Google Stitch frontend template verified.")

    # 5. Verify API Server & Endpoints
    from api.server import (
        engine,
        get_health,
        get_results,
        get_samples,
        get_sample_by_id,
        post_predict,
        PredictRequest,
    )

    print(f"  [PASS] PGIN Inference Engine initialized (engine_mode: {engine.engine_mode}).")

    h = get_health()
    assert h["status"] == "ok" and h["model_loaded"] is True
    print("  [PASS] GET /api/health -> OK")

    res = get_results()
    assert res["champion"]["f1"] > 0.75
    assert len(res["runs"]) == 8
    print(f"  [PASS] GET /api/results -> 8 runs loaded (Champion F1: {res['champion']['f1']:.4f})")

    samples = get_samples()
    assert len(samples["keys"]) >= 4
    print(f"  [PASS] GET /api/samples -> {len(samples['keys'])} test samples loaded")

    # Sample predict
    pred_sample = post_predict(PredictRequest(sample_id="chandler"))
    assert pred_sample["prediction"] == "Sarcastic"
    print(f"  [PASS] POST /api/predict (sample: chandler) -> {pred_sample['prediction']} ({pred_sample['confidence_pct']})")

    # Custom predict
    pred_custom = post_predict(PredictRequest(
        text="Oh brilliant idea, that will surely succeed.",
        prosody_pitch_slope=-0.85,
        prosody_energy_delta=0.40,
    ))
    assert "gate_weights" in pred_custom
    assert isinstance(pred_custom["is_sarcastic"], bool) and type(pred_custom["is_sarcastic"]) is bool

    # Verify JSON serialization with FastAPI jsonable_encoder
    from fastapi.encoders import jsonable_encoder
    encoded_custom = jsonable_encoder(pred_custom)
    assert encoded_custom["is_sarcastic"] in (True, False)
    print(f"  [PASS] POST /api/predict (custom text + prosody) -> {pred_custom['prediction']} ({pred_custom['confidence_pct']}, jsonable_encoded OK)")

    print("=" * 65)
    print("  [SUCCESS] ALL SARCASTONE ENDPOINTS & ASSETS VERIFIED!")
    print("=" * 65)
    return True


def main():
    parser = argparse.ArgumentParser(description="Launch SarcasTone Web Application")
    parser.add_argument("--host", default="0.0.0.0", help="Host interface (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on (default: 8000)")
    parser.add_argument("--test", action="store_true", help="Run verification tests and exit")
    parser.add_argument("--no-reload", action="store_true", help="Disable auto-reload")
    args = parser.parse_args()

    if args.test:
        success = run_self_test()
        sys.exit(0 if success else 1)

    # First verify assets
    run_self_test()

    import uvicorn
    print("\n" + "=" * 65)
    print(f"  Starting SarcasTone Application Server...")
    print(f"  URL: http://localhost:{args.port}")
    print(f"  API Docs: http://localhost:{args.port}/docs")
    print("=" * 65 + "\n")

    uvicorn.run(
        "api.server:app",
        host=args.host,
        port=args.port,
        reload=not args.no_reload,
    )


if __name__ == "__main__":
    main()
