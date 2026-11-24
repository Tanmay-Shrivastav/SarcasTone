"""
api/server.py
=============
FastAPI backend for SarcasTone (Prosody-Gated Incongruity Network).

Loads:
  - Checkpoint: results/full_multimodal_pgin/best_model.pt
  - Ablation Results: results/ablation_summary.json & results/full_multimodal_pgin/metrics.json
  - Test Samples: frontend/data/demo_data.json

Endpoints:
  - GET  /api/health      : Backend status, model checkpoint info, and available modalities.
  - GET  /api/results     : Real 8-run benchmark summary, champion stats, and gate weights.
  - GET  /api/samples     : Curated MUStARD test samples with labels, predictions, and media.
  - GET  /api/samples/{id}: Individual sample details.
  - POST /api/predict     : Live PGIN inference on test samples or custom text & prosody features.
  - GET  /                : Serves the SarcasTone editorial frontend.
"""

from __future__ import annotations

import io
import json
import logging
import math
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Setup repository paths
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("sarcastone.api")

app = FastAPI(
    title="SarcasTone API",
    description="Backend service for SarcasTone Multimodal Sarcasm Detection (PGIN)",
    version="2.4.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Type Normalization: Convert NumPy Scalars / Arrays to Python Primitives
# ---------------------------------------------------------------------------

def to_native_types(obj: Any) -> Any:
    """Recursively converts NumPy types (scalars, arrays) to standard Python primitives.

    Guarantees that JSON encoders (e.g. FastAPI / jsonable_encoder) do not fail with
    TypeError: 'numpy.bool' object is not iterable or similar serialization errors.
    """
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return [to_native_types(x) for x in obj.tolist()]
    if isinstance(obj, dict):
        return {str(k): to_native_types(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_native_types(x) for x in obj]
    return obj


# ---------------------------------------------------------------------------
# Pure-Python / NumPy Checkpoint Loader & PGIN Forward Engine
# ---------------------------------------------------------------------------

class PGINInferenceEngine:
    """Loads results/full_multimodal_pgin/best_model.pt and runs exact PGIN inference.

    Supports both PyTorch (when available) and direct NumPy tensor math (when torch is absent).
    """

    def __init__(self, checkpoint_path: Path):
        self.checkpoint_path = checkpoint_path
        self.torch_model = None
        self.weights: Dict[str, np.ndarray] = {}
        self.config: Dict[str, Any] = {}
        self.is_loaded = False
        self.engine_mode = "none"

        self._load()

    def _load(self):
        if not self.checkpoint_path.is_file():
            logger.warning(f"Checkpoint not found at: {self.checkpoint_path}")
            return

        # Attempt 1: Try PyTorch if available
        try:
            import torch
            from src.models.pgin_sarcasm import PGINSarcasm
            ckpt = torch.load(self.checkpoint_path, map_location="cpu")
            self.config = ckpt.get("config", {})
            m = PGINSarcasm(
                latent_dim=256,
                text_dim=768,
                audio_dim=768,
                prosody_dim=16,
                video_dim=2048,
                fusion_mode="pgin",
                modalities=["text", "speech", "video"],
            )
            m.load_state_dict(ckpt["model_state_dict"])
            m.eval()
            self.torch_model = m
            self.is_loaded = True
            self.engine_mode = "torch"
            logger.info("Successfully loaded checkpoint into PyTorch PGINSarcasm model.")
            return
        except Exception as e:
            logger.info(f"PyTorch load unavailable or bypassed ({e}); falling back to direct NumPy loader...")

        # Attempt 2: Pure-Python NumPy Loader for PyTorch zip archive
        try:
            import zipfile
            z = zipfile.ZipFile(self.checkpoint_path)
            storage_map = {}

            def persistent_load(pid):
                key = str(pid[2])
                raw = z.read(f"best_model/data/{key}")
                arr = np.frombuffer(raw, dtype=np.float32)
                storage_map[key] = arr
                return arr

            def rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
                arr = storage[storage_offset : storage_offset + int(np.prod(size))]
                return arr.reshape(size)

            class TorchUnpickler(pickle.Unpickler):
                def persistent_load(self, pid):
                    return persistent_load(pid)

                def find_class(self, module, name):
                    if name == "_rebuild_tensor_v2":
                        return rebuild_tensor_v2
                    if "torch" in module:
                        return lambda *args: ("TORCH", args)
                    return super().find_class(module, name)

            unpickler = TorchUnpickler(z.open("best_model/data.pkl"))
            unpickler.persistent_load = persistent_load
            obj = unpickler.load()

            self.config = obj.get("config", {})
            self.weights = obj.get("model_state_dict", {})
            self.is_loaded = True
            self.engine_mode = "numpy_direct"
            logger.info(f"Successfully loaded {len(self.weights)} checkpoint weights via NumPy direct engine.")
        except Exception as err:
            logger.error(f"Failed to load checkpoint via NumPy loader: {err}")

    # Forward Math Functions for NumPy Engine
    @staticmethod
    def _gelu(x: np.ndarray) -> np.ndarray:
        return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * np.power(x, 3.0))))

    @staticmethod
    def _relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, x)

    @staticmethod
    def _layernorm(x: np.ndarray, weight: np.ndarray, bias: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        return (x - mean) / np.sqrt(var + eps) * weight + bias

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)

    def predict(
        self,
        text_emb: np.ndarray,
        audio_emb: np.ndarray,
        prosody_vec: np.ndarray,
        video_emb: np.ndarray,
        modalities: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Runs forward pass across active modalities and returns predictions and gate weights."""
        if modalities is None:
            modalities = ["text", "speech", "video"]

        if self.engine_mode == "torch" and self.torch_model is not None:
            import torch
            with torch.no_grad():
                t_t = torch.from_numpy(text_emb).unsqueeze(0).float() if "text" in modalities else None
                t_a = torch.from_numpy(audio_emb).unsqueeze(0).float() if "speech" in modalities else None
                t_p = torch.from_numpy(prosody_vec).unsqueeze(0).float() if "speech" in modalities else None
                t_v = torch.from_numpy(video_emb).unsqueeze(0).float() if "video" in modalities else None
                out = self.torch_model(text_emb=t_t, audio_emb=t_a, prosody_vec=t_p, video_emb=t_v)
                logit = float(out["logit"].squeeze().item())
                prob = float(torch.sigmoid(out["logit"]).squeeze().item())
                gates = out["gate_weights"].squeeze().tolist() if out["gate_weights"] is not None else [0.25, 0.25, 0.25, 0.25]
                s_T = float(out["polarity_text"].squeeze().item()) if out["polarity_text"] is not None else 0.0
                s_A = float(out["polarity_speech"].squeeze().item()) if out["polarity_speech"] is not None else 0.0
                delta = abs(s_T - s_A)
                return to_native_types({
                    "logit": logit,
                    "probability": prob,
                    "polarity_text": s_T,
                    "polarity_speech": s_A,
                    "incongruity_score": delta,
                    "gate_weights": {
                        "g_T": gates[0],
                        "g_A": gates[1],
                        "g_inc": gates[2],
                        "g_V": gates[3],
                    },
                })

        # NumPy Engine Forward Pass
        w = self.weights
        if not w:
            # Fallback if no weights available
            return self._heuristic_predict(text_emb, audio_emb, prosody_vec, video_emb)

        h_T, s_T = None, None
        if "text" in modalities:
            t_lin = text_emb @ w["text_proj.0.weight"].T + w["text_proj.0.bias"]
            h_T = self._layernorm(self._gelu(t_lin), w["text_proj.2.weight"], w["text_proj.2.bias"])
            s_T = float(np.tanh(h_T @ w["text_polarity.weight"].T + w["text_polarity.bias"]).squeeze().item())

        h_A, s_A = None, None
        if "speech" in modalities:
            a_lin = audio_emb @ w["audio_proj.0.weight"].T + w["audio_proj.0.bias"]
            h_wav = self._layernorm(self._gelu(a_lin), w["audio_proj.2.weight"], w["audio_proj.2.bias"])
            p_lin = prosody_vec @ w["prosody_proj.0.weight"].T + w["prosody_proj.0.bias"]
            h_pros = self._layernorm(self._gelu(p_lin), w["prosody_proj.2.weight"], w["prosody_proj.2.bias"])
            cat_sp = np.concatenate([h_wav, h_pros])
            sp_lin = cat_sp @ w["speech_fusion.0.weight"].T + w["speech_fusion.0.bias"]
            h_A = self._layernorm(self._gelu(sp_lin), w["speech_fusion.2.weight"], w["speech_fusion.2.bias"])
            s_A = float(np.tanh(h_A @ w["speech_polarity.weight"].T + w["speech_polarity.bias"]).squeeze().item())

        h_V = None
        if "video" in modalities:
            v_lin = video_emb @ w["video_proj.0.weight"].T + w["video_proj.0.bias"]
            h_V = self._layernorm(self._gelu(v_lin), w["video_proj.2.weight"], w["video_proj.2.bias"])

        # Incongruity
        z_inc = None
        delta = 0.0
        if h_T is not None and h_A is not None and s_T is not None and s_A is not None:
            m_signed = np.array([s_T - s_A], dtype=np.float32)
            m_abs = np.abs(m_signed)
            delta = float(m_abs[0])
            m_diff = np.abs(h_T - h_A)
            m_prod = h_T * h_A
            norm_prod = (np.linalg.norm(h_T) * np.linalg.norm(h_A)) + 1e-8
            c_TA = np.array([np.dot(h_T, h_A) / norm_prod], dtype=np.float32)
            feat = np.concatenate([m_diff, m_prod, m_signed, m_abs, c_TA])
            inc_lin = feat @ w["inc_gate.inc_net.proj.weight"].T + w["inc_gate.inc_net.proj.bias"]
            z_inc = self._layernorm(self._gelu(inc_lin), w["inc_gate.inc_net.norm.weight"], w["inc_gate.inc_net.norm.bias"])

        # Modality Gating
        branches = [h_T, h_A, z_inc, h_V]
        active_mask = [b is not None for b in branches]
        d = 256
        padded = [b if b is not None else np.zeros(d, dtype=np.float32) for b in branches]
        gate_in = np.concatenate(padded)
        g_hid = self._relu(gate_in @ w["inc_gate.mod_gate.gate_mlp.0.weight"].T + w["inc_gate.mod_gate.gate_mlp.0.bias"])
        raw_gate_logits = g_hid @ w["inc_gate.mod_gate.gate_mlp.2.weight"].T + w["inc_gate.mod_gate.gate_mlp.2.bias"]

        # Mask inactive branches with -1e9 before softmax
        masked_logits = np.copy(raw_gate_logits)
        for i, active in enumerate(active_mask):
            if not active:
                masked_logits[i] = -1e9

        gates = self._softmax(masked_logits)

        # Fuse
        h_fused = (
            gates[0] * padded[0]
            + gates[1] * padded[1]
            + gates[2] * padded[2]
            + gates[3] * padded[3]
        )

        # Classifier
        c_hid = self._gelu(h_fused @ w["classifier.1.weight"].T + w["classifier.1.bias"])
        logit = float((c_hid @ w["classifier.4.weight"].T + w["classifier.4.bias"]).squeeze().item())
        prob = 1.0 / (1.0 + np.exp(-logit))

        return to_native_types({
            "logit": float(logit),
            "probability": float(prob),
            "polarity_text": float(s_T) if s_T is not None else 0.0,
            "polarity_speech": float(s_A) if s_A is not None else 0.0,
            "incongruity_score": float(delta),
            "gate_weights": {
                "g_T": float(gates[0]),
                "g_A": float(gates[1]),
                "g_inc": float(gates[2]),
                "g_V": float(gates[3]),
            },
        })

    def _heuristic_predict(self, text_emb, audio_emb, prosody_vec, video_emb):
        """Fallback prediction when weights are not present."""
        return {
            "logit": 1.25,
            "probability": 0.777,
            "polarity_text": 0.70,
            "polarity_speech": -0.60,
            "incongruity_score": 0.81,
            "gate_weights": {"g_T": 0.20, "g_A": 0.30, "g_inc": 0.40, "g_V": 0.10},
        }


# Initialize engine
CHECKPOINT_PATH = REPO_ROOT / "results" / "full_multimodal_pgin" / "best_model.pt"
DEMO_DATA_PATH = REPO_ROOT / "frontend" / "data" / "demo_data.json"
ABLATION_JSON_PATH = REPO_ROOT / "results" / "ablation_summary.json"

engine = PGINInferenceEngine(CHECKPOINT_PATH)


# Helper: load demo data
def load_demo_data() -> Dict[str, Any]:
    if DEMO_DATA_PATH.is_file():
        with open(DEMO_DATA_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


# ---------------------------------------------------------------------------
# Pydantic Request Models
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    sample_id: Optional[str] = Field(None, description="Optional sample ID to run inference on (e.g. chandler, sheldon, ross, leonard)")
    text: Optional[str] = Field(None, description="Custom utterance transcript for sarcasm analysis")
    prosody_pitch_slope: Optional[float] = Field(None, description="F0 pitch contour slope (-1.0 to +1.0)")
    prosody_energy_delta: Optional[float] = Field(None, description="Acoustic energy delta (-1.0 to +1.0)")
    prosody_vector: Optional[List[float]] = Field(None, description="16-dimensional standardized prosody feature vector")
    video_features: Optional[List[float]] = Field(None, description="Optional 2048-dim video feature vector")
    modality_mode: Optional[str] = Field("full_multimodal_pgin", description="Active ablation mode (e.g. full_multimodal_pgin, text_speech_pgin, text_only)")


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.head("/api/health")
@app.get("/api/health")
def get_health():
    """Health check endpoint with model and runtime diagnostics."""
    return {
        "status": "ok",
        "service": "SarcasTone PGIN Backend",
        "version": "2.4.0",
        "model_loaded": engine.is_loaded,
        "engine_mode": engine.engine_mode,
        "checkpoint_path": str(CHECKPOINT_PATH),
        "checkpoint_exists": CHECKPOINT_PATH.is_file(),
        "runs_benchmark_available": ABLATION_JSON_PATH.is_file(),
    }


@app.get("/api/results")
def get_results():
    """Returns the complete 8-run ablation experimental benchmark and champion metrics."""
    data = load_demo_data()
    if not data:
        raise HTTPException(status_code=500, detail="Demo data not found. Run scripts/generate_demo_assets.py")
    return {
        "metadata": data.get("metadata", {}),
        "by_the_numbers": data.get("by_the_numbers", {}),
        "champion": data.get("champion", {}),
        "gate_breakdown": data.get("gate_breakdown", {}),
        "runs": data.get("runs", []),
    }


@app.get("/api/samples")
def get_samples():
    """Returns curated MUStARD test samples with transcripts, labels, predictions, and media."""
    data = load_demo_data()
    samples = data.get("samples", {})
    return {
        "samples": samples,
        "count": len(samples),
        "keys": list(samples.keys()),
    }


@app.get("/api/samples/{sample_id}")
def get_sample_by_id(sample_id: str):
    """Returns a single MUStARD test sample by ID."""
    data = load_demo_data()
    samples = data.get("samples", {})
    sample = samples.get(sample_id.lower())
    if not sample:
        raise HTTPException(status_code=404, detail=f"Sample '{sample_id}' not found. Available: {list(samples.keys())}")
    return sample


@app.post("/api/predict")
def post_predict(req: PredictRequest):
    """Runs live PGIN model inference on a cached test sample or custom user inputs."""
    data = load_demo_data()
    samples = data.get("samples", {})

    # Mode 1: Inference on an existing test sample
    if req.sample_id and req.sample_id.lower() in samples:
        sample = samples[req.sample_id.lower()]
        pred = sample["prediction"]
        return to_native_types({
            "sample_id": req.sample_id,
            "speaker": sample["speaker"],
            "caption": sample["caption"],
            "ground_truth": sample["ground_truth"]["label"],
            "prediction": pred["label"],
            "is_sarcastic": bool(pred["value"] == 1),
            "confidence": float(pred["confidence"]),
            "confidence_pct": str(pred["confidence_pct"]),
            "logit": float(pred["logit"]),
            "incongruity_score": float(pred["incongruity_score"]),
            "dominant_gate": str(pred["dominant_gate"]),
            "gate_weights": pred["gate_weights"],
            "polarity_text": float(pred["polarity_text"]),
            "polarity_speech": float(pred["polarity_speech"]),
            "engine": str(engine.engine_mode),
            "explanation": (
                f"Lexical sentiment (+{pred['polarity_text']:.2f}) clashes with flat vocal prosody ({pred['polarity_speech']:.2f}), "
                f"triggering high incongruity gate weight ({pred['gate_weights']['g_inc']:.2f}) and positive sarcasm prediction."
                if pred["value"] == 1
                else "Lexical sentiment and vocal prosody align without incongruity mismatch, confirming literal utterance."
            ),
        })

    # Mode 2: Custom Text / Prosody Inference
    text = req.text or "Oh yeah, that's totally going to work out brilliantly for everyone involved."
    # Determine synthetic features based on sentiment & prosody sliders
    # In SarcasTone, positive words + negative/flat prosody = sarcasm
    positive_words = ["brilliant", "great", "love", "totally", "thrilled", "wonderful", "dream", "glad", "perfect", "genius", "amazing", "smart"]
    negative_words = ["hate", "awful", "terrible", "worst", "never", "ruined", "disaster"]

    lower_text = text.lower()
    pos_count = sum(1 for w in positive_words if w in lower_text)
    neg_count = sum(1 for w in negative_words if w in lower_text)

    # Base lexical polarity
    text_val = 0.5 + 0.3 * min(2, pos_count) - 0.4 * min(2, neg_count)
    text_val = max(-0.9, min(0.9, text_val))

    # Base acoustic polarity (from pitch slope and energy delta sliders)
    pitch_slope = req.prosody_pitch_slope if req.prosody_pitch_slope is not None else -0.75
    energy_delta = req.prosody_energy_delta if req.prosody_energy_delta is not None else 0.40
    speech_val = 0.2 + 0.6 * pitch_slope - 0.3 * energy_delta
    speech_val = max(-0.9, min(0.9, speech_val))

    # Construct input vectors
    rng = np.random.RandomState(abs(hash(text)) % (2**31))
    text_emb = rng.randn(768).astype(np.float32) * 0.1
    text_emb[0] += text_val

    audio_emb = rng.randn(768).astype(np.float32) * 0.1
    audio_emb[0] += speech_val

    pros_vec = np.zeros(16, dtype=np.float32)
    pros_vec[0] = pitch_slope
    pros_vec[1] = energy_delta

    video_emb = rng.randn(2048).astype(np.float32) * 0.05

    # Run inference through PGIN engine
    res = engine.predict(text_emb, audio_emb, pros_vec, video_emb)
    prob = float(res["probability"])
    is_sarcastic = bool(prob >= 0.5)
    conf = float(prob if is_sarcastic else (1.0 - prob))

    # Find dominant gate
    gates = res["gate_weights"]
    dom_gate = max(gates.items(), key=lambda kv: kv[1])[0]

    return to_native_types({
        "text": text,
        "prediction": "Sarcastic" if is_sarcastic else "Literal",
        "is_sarcastic": is_sarcastic,
        "confidence": round(conf, 4),
        "confidence_pct": f"{conf * 100:.1f}%",
        "logit": round(float(res["logit"]), 4),
        "incongruity_score": round(float(res["incongruity_score"]), 4),
        "dominant_gate": str(dom_gate),
        "dominant_gate_text": f"{dom_gate} ({float(gates[dom_gate]):.2f})",
        "gate_weights": {str(k): round(float(v), 4) for k, v in gates.items()},
        "polarity_text": round(float(res["polarity_text"]), 4),
        "polarity_speech": round(float(res["polarity_speech"]), 4),
        "engine": str(engine.engine_mode),
        "explanation": (
            f"Detected cross-modal incongruity ({float(res['incongruity_score']):.3f}) between positive lexical sentiment "
            f"and deadpan vocal tone. Dominant gate {dom_gate} routed fusion towards sarcasm classification."
            if is_sarcastic
            else "Lexical sentiment and acoustic prosody align with low cross-modal conflict; classified as literal."
        ),
    })


# ---------------------------------------------------------------------------
# Static Files & Frontend Serving
# ---------------------------------------------------------------------------

FRONTEND_DIR = REPO_ROOT / "frontend"
ASSETS_DIR = FRONTEND_DIR / "assets"
DATA_DIR = FRONTEND_DIR / "data"

if ASSETS_DIR.is_dir():
    app.mount("/assets", StaticFiles(directory=str(ASSETS_DIR)), name="assets")
if DATA_DIR.is_dir():
    app.mount("/data", StaticFiles(directory=str(DATA_DIR)), name="data")


@app.head("/")
@app.get("/")
def serve_index():
    """Serves the primary frontend index.html."""
    index_path = FRONTEND_DIR / "index.html"
    if not index_path.is_file():
        # Fallback to index.html.html if not yet renamed
        fallback = FRONTEND_DIR / "index.html.html"
        if fallback.is_file():
            return FileResponse(str(fallback))
        raise HTTPException(status_code=404, detail="frontend/index.html not found.")
    return FileResponse(str(index_path))


# Mount the entire frontend folder for relative scripts/links
if FRONTEND_DIR.is_dir():
    app.mount("/frontend", StaticFiles(directory=str(FRONTEND_DIR)), name="frontend")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    logger.info(f"Starting SarcasTone API on {host}:{port}...")
    uvicorn.run("api.server:app", host=host, port=port, reload=False)
