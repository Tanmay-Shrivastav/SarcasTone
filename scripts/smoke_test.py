#!/usr/bin/env python3
"""
scripts/smoke_test.py
=====================
Lightweight end-to-end smoke test for SarcasTone (PGIN-Sarcasm).

Validates:
  1. Forward pass shapes and outputs for all 8 ablation modes.
  2. Modality gating constraints and polarity proxy bounds.
  3. Explicit incongruity engine computation (z_inc).
  4. SarcasmLoss (BCE + auxiliary incongruity margin loss) and backward gradient flow.
  5. Metrics computation (Accuracy, Precision, Recall, F1, confusion matrix, gate weights).
  6. End-to-end training and evaluation across all 8 ablation runs using synthetic
     cached feature tensors and splits without requiring external dataset downloads.

Usage:
  python scripts/smoke_test.py
  python scripts/smoke_test.py --device cuda
  python scripts/smoke_test.py --epochs 2 --batch-size 4
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("smoke_test")


# ---------------------------------------------------------------------------
# 8 Standard Ablation Configurations
# ---------------------------------------------------------------------------
ABLATION_MODES = [
    {"name": "text_only", "modalities": ["text"], "fusion_mode": "pgin"},
    {"name": "speech_only", "modalities": ["speech"], "fusion_mode": "pgin"},
    {"name": "video_only", "modalities": ["video"], "fusion_mode": "pgin"},
    {"name": "text_speech_concat", "modalities": ["text", "speech"], "fusion_mode": "concat"},
    {"name": "text_speech_pgin", "modalities": ["text", "speech"], "fusion_mode": "pgin"},
    {"name": "text_video", "modalities": ["text", "video"], "fusion_mode": "pgin"},
    {"name": "speech_video", "modalities": ["speech", "video"], "fusion_mode": "pgin"},
    {"name": "full_multimodal_pgin", "modalities": ["text", "speech", "video"], "fusion_mode": "pgin"},
]


# ---------------------------------------------------------------------------
# Test Stage 1: Forward Passes for all 8 Ablation Configurations
# ---------------------------------------------------------------------------
def test_forward_passes(device: Any) -> None:
    """Verify model forward pass, output shapes, gate weights, and polarity proxies."""
    import torch
    from src.models.pgin_sarcasm import PGINSarcasm

    logger.info("=" * 70)
    logger.info("STAGE 1: Model Forward Passes across All 8 Ablation Modes")
    logger.info("=" * 70)

    batch_size = 4
    d = 256
    text_dim = 768
    audio_dim = 768
    prosody_dim = 16
    video_dim = 2048

    for mode in ABLATION_MODES:
        run_name = mode["name"]
        modalities = mode["modalities"]
        fusion = mode["fusion_mode"]

        logger.info(f"  Testing run: {run_name:<24} | modalities={modalities} | fusion={fusion}")

        model = PGINSarcasm(
            latent_dim=d,
            text_dim=text_dim,
            audio_dim=audio_dim,
            prosody_dim=prosody_dim,
            video_dim=video_dim,
            fusion_mode=fusion,
            modalities=modalities,
        ).to(device)
        model.eval()

        # Generate dummy input tensors
        t_in = torch.randn(batch_size, text_dim, device=device) if "text" in modalities else None
        a_in = torch.randn(batch_size, audio_dim, device=device) if "speech" in modalities else None
        p_in = torch.randn(batch_size, prosody_dim, device=device) if "speech" in modalities else None
        v_in = torch.randn(batch_size, video_dim, device=device) if "video" in modalities else None

        with torch.no_grad():
            out = model(text_emb=t_in, audio_emb=a_in, prosody_vec=p_in, video_emb=v_in)

        # 1. Check logit
        assert "logit" in out, f"[{run_name}] 'logit' missing from output"
        assert out["logit"] is not None, f"[{run_name}] 'logit' is None"
        assert out["logit"].shape == (batch_size, 1), f"[{run_name}] Invalid logit shape: {out['logit'].shape}"
        assert not torch.isnan(out["logit"]).any(), f"[{run_name}] NaN in logit"

        # 2. Check polarity proxies
        if "text" in modalities:
            assert out["polarity_text"] is not None
            assert out["polarity_text"].shape == (batch_size, 1)
            assert (out["polarity_text"] >= -1.0001).all() and (out["polarity_text"] <= 1.0001).all()
        else:
            assert out["polarity_text"] is None

        if "speech" in modalities:
            assert out["polarity_speech"] is not None
            assert out["polarity_speech"].shape == (batch_size, 1)
            assert (out["polarity_speech"] >= -1.0001).all() and (out["polarity_speech"] <= 1.0001).all()
        else:
            assert out["polarity_speech"] is None

        # 3. Check gating behavior
        if fusion == "pgin":
            assert out["gate_weights"] is not None
            gw = out["gate_weights"]
            assert gw.shape == (batch_size, 4), f"[{run_name}] Invalid gate_weights shape: {gw.shape}"

            # Verify active mask and sum to 1.0
            gate_sums = gw.sum(dim=-1)
            assert torch.allclose(gate_sums, torch.ones_like(gate_sums), atol=1e-5), (
                f"[{run_name}] Gate weights do not sum to 1: {gate_sums}"
            )

            # Check that inactive branches receive 0 weight
            # Gate slots: 0=text, 1=speech, 2=inc, 3=video
            if "text" not in modalities:
                assert torch.all(gw[:, 0] == 0.0), f"[{run_name}] Text inactive but received gate weight > 0"
            if "speech" not in modalities:
                assert torch.all(gw[:, 1] == 0.0), f"[{run_name}] Speech inactive but received gate weight > 0"
            if not ("text" in modalities and "speech" in modalities):
                assert torch.all(gw[:, 2] == 0.0), f"[{run_name}] Inc inactive but received gate weight > 0"
            if "video" not in modalities:
                assert torch.all(gw[:, 3] == 0.0), f"[{run_name}] Video inactive but received gate weight > 0"

            # Check incongruity representation
            if "text" in modalities and "speech" in modalities:
                assert out["z_inc"] is not None
                assert out["z_inc"].shape == (batch_size, d)
            else:
                assert out["z_inc"] is None
        else:
            # Concat mode
            assert out["gate_weights"] is None
            assert out["z_inc"] is None

    logger.info("[PASS] All 8 ablation mode forward passes and gating assertions verified.")


# ---------------------------------------------------------------------------
# Test Stage 2: Incongruity Engine & Gating Logic
# ---------------------------------------------------------------------------
def test_incongruity_and_gating(device: Any) -> None:
    """Directly test IncongruityNet and IncongruityGate components."""
    import torch
    from src.models.incongruity_gate import IncongruityGate, IncongruityNet

    logger.info("=" * 70)
    logger.info("STAGE 2: Explicit Incongruity Engine & Gating Logic")
    logger.info("=" * 70)

    d = 256
    batch_size = 4
    inc_net = IncongruityNet(latent_dim=d).to(device)

    h_T = torch.randn(batch_size, d, device=device)
    h_A = torch.randn(batch_size, d, device=device)
    s_T = torch.tensor([[0.8], [-0.5], [0.9], [-0.9]], device=device)
    s_A = torch.tensor([[-0.7], [0.6], [0.8], [-0.8]], device=device)

    z_inc = inc_net(h_T, h_A, s_T, s_A)
    assert z_inc.shape == (batch_size, d)
    assert not torch.isnan(z_inc).any()

    # Full IncongruityGate
    gate = IncongruityGate(latent_dim=d).to(device)
    h_V = torch.randn(batch_size, d, device=device)

    # 1. Full multimodal
    h_fused, weights, z = gate(h_T, h_A, h_V, s_T, s_A)
    assert h_fused.shape == (batch_size, d)
    assert weights.shape == (batch_size, 4)
    assert z is not None and z.shape == (batch_size, d)
    assert torch.all(weights > 0.0)

    # 2. Text-only (unimodal fallback)
    h_fused_t, weights_t, z_t = gate(h_T, None, None, s_T, None)
    assert h_fused_t.shape == (batch_size, d)
    assert torch.allclose(weights_t[:, 0], torch.ones(batch_size, device=device))
    assert torch.all(weights_t[:, 1:] == 0.0)
    assert z_t is None

    logger.info("[PASS] IncongruityNet and IncongruityGate validated.")


# ---------------------------------------------------------------------------
# Test Stage 3: Loss Function & Backward Pass
# ---------------------------------------------------------------------------
def test_loss_and_gradients(device: Any) -> None:
    """Verify SarcasmLoss calculation, margin hinge, and gradient backpropagation."""
    import torch
    import torch.optim as optim
    from src.models.pgin_sarcasm import PGINSarcasm
    from src.training.losses import SarcasmLoss

    logger.info("=" * 70)
    logger.info("STAGE 3: Multi-Task Loss & Gradient Flow")
    logger.info("=" * 70)

    criterion = SarcasmLoss(lambda_inc=0.1, inc_margin=0.5)

    # 1. Loss with divergent polarities
    logits = torch.tensor([[2.0], [-2.0], [1.5], [-1.0]], device=device)
    labels = torch.tensor([1, 0, 1, 0], device=device)
    s_T = torch.tensor([[0.8], [-0.5], [0.1], [0.3]], device=device)
    s_A = torch.tensor([[-0.8], [0.5], [0.1], [0.4]], device=device)

    total_loss, bce, aux = criterion(logits, labels, s_T, s_A)
    assert total_loss.item() > 0.0
    assert bce.item() > 0.0
    assert aux.item() >= 0.0
    assert torch.allclose(total_loss, bce + 0.1 * aux, atol=1e-5)

    # 2. Gradient flow through PGINSarcasm
    model = PGINSarcasm(
        latent_dim=256,
        text_dim=768,
        audio_dim=768,
        prosody_dim=16,
        video_dim=2048,
        fusion_mode="pgin",
        modalities=["text", "speech", "video"],
    ).to(device)
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    t_in = torch.randn(4, 768, device=device, requires_grad=True)
    a_in = torch.randn(4, 768, device=device, requires_grad=True)
    p_in = torch.randn(4, 16, device=device, requires_grad=True)
    v_in = torch.randn(4, 2048, device=device, requires_grad=True)

    optimizer.zero_grad()
    out = model(text_emb=t_in, audio_emb=a_in, prosody_vec=p_in, video_emb=v_in)
    loss, _, _ = criterion(
        logit=out["logit"],
        labels=labels,
        polarity_text=out["polarity_text"],
        polarity_speech=out["polarity_speech"],
    )

    loss.backward()

    # Verify gradients exist and are finite for model parameters
    grad_count = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"
            assert not torch.isinf(param.grad).any(), f"Inf gradient in {name}"
            grad_count += 1

    optimizer.step()
    logger.info(f"[PASS] Gradient backpropagation verified across {grad_count} parameter tensors.")


# ---------------------------------------------------------------------------
# Test Stage 4: Metrics Computation
# ---------------------------------------------------------------------------
def test_metrics_computation() -> None:
    """Verify compute_metrics and MetricTracker."""
    import torch
    from src.training.metrics import MetricTracker, compute_metrics

    logger.info("=" * 70)
    logger.info("STAGE 4: Metrics & MetricTracker Accumulation")
    logger.info("=" * 70)

    # 1. Direct compute_metrics
    logits = torch.tensor([3.0, -3.0, 2.5, -2.5, 4.0, -1.0])
    labels = torch.tensor([1, 0, 1, 0, 1, 0])
    gates = torch.tensor([
        [0.4, 0.3, 0.2, 0.1],
        [0.2, 0.5, 0.1, 0.2],
        [0.3, 0.3, 0.3, 0.1],
        [0.1, 0.6, 0.2, 0.1],
        [0.5, 0.2, 0.2, 0.1],
        [0.2, 0.4, 0.2, 0.2],
    ])

    metrics = compute_metrics(logits, labels, gate_weights=gates)

    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["confusion_matrix"] == [[3, 0], [0, 3]]
    assert metrics["n_samples"] == 6
    assert metrics["n_sarcastic"] == 3
    assert metrics["mean_gate_weights"] is not None
    assert len(metrics["mean_gate_weights"]) == 4

    # 2. MetricTracker accumulation
    tracker = MetricTracker()
    tracker.update(logits[:3], labels[:3], gates[:3], loss=0.15)
    tracker.update(logits[3:], labels[3:], gates[3:], loss=0.25)

    accum_metrics = tracker.compute()
    assert accum_metrics["accuracy"] == 1.0
    assert pytest_approx(accum_metrics["mean_loss"], 0.20, tol=1e-4)
    tracker.reset()
    assert tracker.n_batches == 0

    logger.info("[PASS] compute_metrics and MetricTracker verified.")


# ---------------------------------------------------------------------------
# Test Stage 5: End-to-End Execution of All 8 Ablation Runs
# ---------------------------------------------------------------------------
def test_end_to_end_ablations(device: Any, epochs: int = 1, batch_size: int = 4) -> None:
    """
    Generate synthetic cached feature files and deterministic splits,
    then execute all 8 ablation runs end-to-end to verify summary table creation.
    """
    import yaml
    import torch
    from src.experiments.run_ablations import main as ablations_main
    from src.features.cache_manager import FeatureCacheManager

    logger.info("=" * 70)
    logger.info("STAGE 5: End-to-End Synthetic 8-Run Ablation Study")
    logger.info("=" * 70)

    with tempfile.TemporaryDirectory() as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        cache_dir = tmp_dir / "cache"
        results_dir = tmp_dir / "results"
        splits_file = cache_dir / "splits.json"
        config_path = tmp_dir / "smoke_config.yaml"

        cache_dir.mkdir(parents=True, exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True)

        # 1. Create synthetic cached samples
        num_samples = 24
        sample_ids = [f"smoke_utt_{i:03d}" for i in range(num_samples)]
        cache_mgr = FeatureCacheManager(cache_dir=cache_dir, file_format="pt")

        rng = torch.Generator().manual_seed(42)
        for i, uid in enumerate(sample_ids):
            label = 1 if i % 2 == 0 else 0
            features = {
                "text_emb": torch.randn(768, generator=rng),
                "audio_emb": torch.randn(768, generator=rng),
                "prosody_vec": torch.randn(16, generator=rng),
                "video_emb": torch.randn(2048, generator=rng),
                "label": label,
                "audio_valid": True,
                "video_valid": True,
                "metadata": {"smoke_test": True},
            }
            cache_mgr.save_sample_features(uid, features)
        cache_mgr.save_manifest()

        # 2. Create deterministic splits: 14 train, 5 val, 5 test
        splits = {
            "metadata": {
                "seed": 42,
                "total_samples": num_samples,
                "train_samples": 14,
                "val_samples": 5,
                "test_samples": 5,
            },
            "train": sample_ids[:14],
            "val": sample_ids[14:19],
            "test": sample_ids[19:],
        }
        with open(splits_file, "w", encoding="utf-8") as f:
            json.dump(splits, f, indent=2)

        # 3. Create mock config YAML
        config_dict = {
            "project": {"name": "SarcasTone_SmokeTest", "seed": 42},
            "paths": {
                "cache_dir": str(cache_dir),
                "splits_file": str(splits_file),
                "results_dir": str(results_dir),
            },
            "encoders": {
                "text_hidden_dim": 768,
                "audio_hidden_dim": 768,
                "video_hidden_dim": 2048,
            },
            "prosody": {"feature_dim": 16},
            "model": {
                "latent_dim": 64,  # Small latent dim for fast smoke test
                "prosody_hidden_dim": 32,
                "dropout": 0.1,
                "classifier_dropout": 0.1,
                "fusion_mode": "pgin",
            },
            "training": {
                "batch_size": batch_size,
                "num_workers": 0,
                "epochs": epochs,
                "lr": 1e-3,
                "weight_decay": 1e-2,
                "patience": 5,
                "use_amp": False,
                "lambda_inc": 0.1,
                "inc_margin": 0.5,
            },
            "ablations": {"runs": ABLATION_MODES},
        }

        with open(config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(config_dict, f)

        logger.info(f"Generated {num_samples} synthetic cached tensors in {cache_dir}")
        logger.info(f"Running all 8 ablations (epochs={epochs}, batch_size={batch_size})...")

        # 4. Run ablations via main entrypoint
        argv = [
            "--config", str(config_path),
            "--results-dir", str(results_dir),
            "--splits-file", str(splits_file),
            "--device", str(device),
        ]
        ret = ablations_main(argv)
        assert ret == 0, f"run_ablations returned non-zero exit code: {ret}"

        # 5. Verify outputs
        csv_path = results_dir / "ablation_summary.csv"
        json_path = results_dir / "ablation_summary.json"
        md_path = results_dir / "ablation_summary.md"

        assert csv_path.is_file(), f"Summary CSV missing: {csv_path}"
        assert json_path.is_file(), f"Summary JSON missing: {json_path}"
        assert md_path.is_file(), f"Summary Markdown missing: {md_path}"

        with open(json_path, "r", encoding="utf-8") as f:
            summary_data = json.load(f)

        assert len(summary_data) == 8, f"Expected 8 ablation results, got {len(summary_data)}"

        # Verify each run output
        for mode in ABLATION_MODES:
            rname = mode["name"]
            run_dir = results_dir / rname
            assert (run_dir / "best_model.pt").is_file(), f"best_model.pt missing for {rname}"
            assert (run_dir / "metrics.json").is_file(), f"metrics.json missing for {rname}"

        # Print markdown table to stdout
        with open(md_path, "r", encoding="utf-8") as f:
            md_content = f.read()

        logger.info("\n" + md_content)
        logger.info("[PASS] End-to-end ablation runner successfully executed all 8 modes.")


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def pytest_approx(a: float, b: float, tol: float = 1e-4) -> bool:
    return abs(a - b) <= tol


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="SarcasTone Smoke Test Suite")
    parser.add_argument("--device", default=None, help="Device string ('cpu', 'cuda').")
    parser.add_argument("--epochs", type=int, default=1, help="Epoch count for end-to-end runs.")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for end-to-end runs.")
    args = parser.parse_args()

    t_start = time.perf_counter()
    logger.info("=" * 70)
    logger.info("SARCASTONE LIGHTWEIGHT SMOKE TEST SUITE")
    logger.info("=" * 70)

    try:
        import torch
        has_torch = True
    except ImportError:
        has_torch = False

    if not has_torch:
        logger.warning(
            "PyTorch (torch) is not installed in the current Python environment.\n"
            "  -> On Google Colab (T4 GPU), PyTorch is pre-installed.\n"
            "  -> Verifying split logic and non-torch data abstractions instead..."
        )
        # Test split and dataset data structures
        from src.data.dataset import MultimodalSample, create_or_load_splits, verify_leakage
        samples = {
            f"utt_{i:02d}": MultimodalSample(
                utterance_id=f"utt_{i:02d}",
                utterance_text=f"Utterance text {i}",
                label=i % 2,
            )
            for i in range(20)
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            splits_p = Path(tmp_dir) / "test_splits.json"
            splits = create_or_load_splits(samples, splits_p, val_ratio=0.2, test_ratio=0.2, seed=42)
            verify_leakage(splits["train"], splits["val"], splits["test"])
            assert len(splits["train"]) == 12
            assert len(splits["val"]) == 4
            assert len(splits["test"]) == 4

        logger.info("[PASS] Non-torch dataset and split verification checks passed.")
        logger.info("Execute 'python scripts/smoke_test.py' inside Google Colab for full PyTorch validation.")
        return 0

    # Determine device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"CUDA GPU detected: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logger.info("Running on CPU.")

    logger.info(f"Target test device: {device}")

    # Run Test Stages
    test_forward_passes(device)
    test_incongruity_and_gating(device)
    test_loss_and_gradients(device)
    test_metrics_computation()
    test_end_to_end_ablations(device, epochs=args.epochs, batch_size=args.batch_size)

    elapsed = time.perf_counter() - t_start
    logger.info("=" * 70)
    logger.info(f"[SUCCESS] ALL SARCASTONE SMOKE TESTS PASSED IN {elapsed:.2f}s!")
    logger.info("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
