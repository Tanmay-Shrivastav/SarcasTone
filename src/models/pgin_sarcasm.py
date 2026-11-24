"""
pgin_sarcasm.py
===============
Prosody-Gated Incongruity Network for Sarcasm Detection (PGIN-Sarcasm).

Architecture (see SRD §4.1 – §4.3)
------------------------------------
1. Modality-specific projection heads:
     h_T  = LayerNorm(GELU(Linear(768  → d)(x_text)))
     ŝ_T  = tanh(Linear(d → 1)(h_T))               ← literal text polarity proxy
     h_A  = LayerNorm(GELU(Linear(d+64 → d)([h_wav; h_pros])))
     ŝ_A  = tanh(Linear(d → 1)(h_A))               ← vocal-affect polarity proxy
     h_V  = LayerNorm(GELU(Linear(2048 → d)(x_vid)))

2. Fusion modes (selected by `fusion_mode` kwarg):
     "pgin"   → IncongruityGate  (z_inc + learned gating)
     "concat" → naive projection of [h_T ; h_A ; h_V] → d  (baseline)

3. Classification head:
     Dropout(0.3) → Linear(d → d//2) → GELU → Dropout(0.2) → Linear(d//2 → 1)
     Output: raw logit ŷ (apply torch.sigmoid externally if probability needed).

Supported modality combinations (all 8 ablation runs):
  - text_only              modalities=["text"]
  - speech_only            modalities=["speech"]
  - video_only             modalities=["video"]
  - text_speech_concat     modalities=["text","speech"],  fusion_mode="concat"
  - text_speech_pgin       modalities=["text","speech"],  fusion_mode="pgin"
  - text_video             modalities=["text","video"],   fusion_mode="pgin"
  - speech_video           modalities=["speech","video"], fusion_mode="pgin"
  - full_multimodal_pgin   modalities=["text","speech","video"], fusion_mode="pgin"

Usage
-----
    cfg = yaml.safe_load(open("configs/default.yaml"))
    model = build_model(cfg, modalities=["text", "speech"], fusion_mode="pgin")
    out   = model(text_emb=text_emb, prosody_vec=prosody_vec, audio_emb=audio_emb)
    logit = out["logit"]          # (B, 1)
    s_T   = out["polarity_text"]  # (B, 1) or None
    s_A   = out["polarity_speech"]# (B, 1) or None
    gates = out["gate_weights"]   # (B, 4) or None
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.incongruity_gate import IncongruityGate


# ---------------------------------------------------------------------------
# Helper: single projection block
# ---------------------------------------------------------------------------

def _proj_block(in_dim: int, out_dim: int) -> nn.Sequential:
    """Linear → GELU → LayerNorm  (SRD §4.1 convention)."""
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.GELU(),
        nn.LayerNorm(out_dim),
    )


# ---------------------------------------------------------------------------
# PGINSarcasm
# ---------------------------------------------------------------------------

class PGINSarcasm(nn.Module):
    """Prosody-Gated Incongruity Network for Sarcasm Detection.

    Parameters
    ----------
    latent_dim : int
        Shared latent dimension d (default 256).
    text_dim : int
        Input text embedding dimension (DeBERTa / RoBERTa hidden dim, default 768).
    audio_dim : int
        Input acoustic embedding dimension (wav2vec2 hidden dim, default 768).
    prosody_dim : int
        Input prosody vector dimension (default 16).
    prosody_hidden_dim : int
        Intermediate prosody MLP dimension (default 64).
    video_dim : int
        Input video frame embedding dimension (ResNet50 pool dim, default 2048).
    dropout : float
        Primary dropout rate (default 0.3).
    classifier_dropout : float
        Secondary dropout rate inside classifier (default 0.2).
    fusion_mode : str
        "pgin" (incongruity + gating) | "concat" (naive baseline).
    modalities : list[str]
        Active modalities: any subset of ["text", "speech", "video"].
    """

    def __init__(
        self,
        latent_dim: int = 256,
        text_dim: int = 768,
        audio_dim: int = 768,
        prosody_dim: int = 16,
        prosody_hidden_dim: int = 64,
        video_dim: int = 2048,
        dropout: float = 0.3,
        classifier_dropout: float = 0.2,
        fusion_mode: str = "pgin",
        modalities: Optional[List[str]] = None,
    ) -> None:
        super().__init__()

        if modalities is None:
            modalities = ["text", "speech", "video"]
        self.modalities = modalities
        self.fusion_mode = fusion_mode
        self.latent_dim = latent_dim

        # ---- Text projection ------------------------------------------------
        if "text" in modalities:
            self.text_proj = _proj_block(text_dim, latent_dim)
            self.text_polarity = nn.Linear(latent_dim, 1)   # ŝ_T

        # ---- Speech projection (prosody + acoustic) -------------------------
        if "speech" in modalities:
            # Prosody sub-projection: 16 → prosody_hidden_dim (= 64)
            self.prosody_proj = _proj_block(prosody_dim, prosody_hidden_dim)
            # Acoustic sub-projection: audio_dim → latent_dim
            self.audio_proj = _proj_block(audio_dim, latent_dim)
            # Combined: [h_wav ; h_pros] → latent_dim
            self.speech_fusion = _proj_block(latent_dim + prosody_hidden_dim, latent_dim)
            self.speech_polarity = nn.Linear(latent_dim, 1)  # ŝ_A

        # ---- Video projection -----------------------------------------------
        if "video" in modalities:
            self.video_proj = _proj_block(video_dim, latent_dim)

        # ---- Fusion layers --------------------------------------------------
        if fusion_mode == "pgin":
            self.inc_gate = IncongruityGate(latent_dim, gate_hidden_dim=128)
        else:
            # concat baseline: project concatenated active branches → latent_dim
            n_active = len(modalities)
            # concat mode has no incongruity; just the raw branch vectors
            self.concat_proj = nn.Linear(n_active * latent_dim, latent_dim)

        # ---- Dropout wrapper ------------------------------------------------
        self.dropout = nn.Dropout(dropout)

        # ---- Classification head --------------------------------------------
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(latent_dim, latent_dim // 2),
            nn.GELU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(latent_dim // 2, 1),
        )

    # -----------------------------------------------------------------------
    # Internal: project individual modalities
    # -----------------------------------------------------------------------

    def _encode_text(self, text_emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (h_T, ŝ_T)."""
        h_T = self.text_proj(text_emb)                     # (B, d)
        s_T = torch.tanh(self.text_polarity(h_T))          # (B, 1)
        return h_T, s_T

    def _encode_speech(
        self,
        audio_emb: torch.Tensor,
        prosody_vec: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (h_A, ŝ_A)."""
        h_wav  = self.audio_proj(audio_emb)                # (B, d)
        h_pros = self.prosody_proj(prosody_vec)            # (B, prosody_hidden_dim)
        h_A    = self.speech_fusion(
            torch.cat([h_wav, h_pros], dim=-1)
        )                                                   # (B, d)
        s_A    = torch.tanh(self.speech_polarity(h_A))     # (B, 1)
        return h_A, s_A

    def _encode_video(self, video_emb: torch.Tensor) -> torch.Tensor:
        """Return h_V."""
        return self.video_proj(video_emb)                  # (B, d)

    # -----------------------------------------------------------------------
    # forward
    # -----------------------------------------------------------------------

    def forward(
        self,
        text_emb:    Optional[torch.Tensor] = None,
        audio_emb:   Optional[torch.Tensor] = None,
        prosody_vec: Optional[torch.Tensor] = None,
        video_emb:   Optional[torch.Tensor] = None,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Parameters
        ----------
        text_emb    : (B, text_dim)    or None if text not active
        audio_emb   : (B, audio_dim)   or None if speech not active
        prosody_vec : (B, prosody_dim) or None if speech not active
        video_emb   : (B, video_dim)   or None if video not active

        Returns
        -------
        dict with keys:
          "logit"          : (B, 1)  — raw sarcasm logit (pre-sigmoid)
          "polarity_text"  : (B, 1) or None
          "polarity_speech": (B, 1) or None
          "gate_weights"   : (B, 4) or None  [g_T, g_A, g_inc, g_V]
          "z_inc"          : (B, d) or None  — incongruity repr
        """
        h_T, h_A, h_V = None, None, None
        s_T, s_A       = None, None

        # ---- Encode active modalities ---------------------------------------
        if "text" in self.modalities and text_emb is not None:
            h_T, s_T = self._encode_text(text_emb)

        if "speech" in self.modalities and audio_emb is not None and prosody_vec is not None:
            h_A, s_A = self._encode_speech(audio_emb, prosody_vec)

        if "video" in self.modalities and video_emb is not None:
            h_V = self._encode_video(video_emb)

        # ---- Fuse ----------------------------------------------------------
        gate_weights: Optional[torch.Tensor] = None
        z_inc:        Optional[torch.Tensor] = None

        if self.fusion_mode == "pgin":
            h_fused, gate_weights, z_inc = self.inc_gate(h_T, h_A, h_V, s_T, s_A)

        else:  # "concat" baseline
            active_branches = [b for b in [h_T, h_A, h_V] if b is not None]
            if len(active_branches) == 0:
                raise ValueError("No active modality branches available for forward pass.")
            cat = torch.cat(active_branches, dim=-1)   # (B, n_active * d)
            h_fused = F.gelu(self.concat_proj(cat))    # (B, d)

        # ---- Classify ------------------------------------------------------
        h_fused = self.dropout(h_fused)
        logit = self.classifier(h_fused)              # (B, 1)

        return {
            "logit":           logit,
            "polarity_text":   s_T,
            "polarity_speech": s_A,
            "gate_weights":    gate_weights,
            "z_inc":           z_inc,
        }


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------

def build_model(
    cfg: dict,
    modalities: Optional[List[str]] = None,
    fusion_mode: Optional[str] = None,
) -> PGINSarcasm:
    """Instantiate PGINSarcasm from a parsed YAML config dict.

    Parameters
    ----------
    cfg : dict
        Full parsed config (output of ``yaml.safe_load``).
    modalities : list[str] or None
        Override for the active modalities (default: all three).
    fusion_mode : str or None
        Override for fusion mode; falls back to ``cfg["model"]["fusion_mode"]``.

    Returns
    -------
    PGINSarcasm (not moved to device — caller's responsibility).
    """
    mcfg  = cfg.get("model", {})
    ecfg  = cfg.get("encoders", {})
    pcfg  = cfg.get("prosody", {})

    if modalities is None:
        modalities = ["text", "speech", "video"]
    if fusion_mode is None:
        fusion_mode = mcfg.get("fusion_mode", "pgin")

    return PGINSarcasm(
        latent_dim           = mcfg.get("latent_dim", 256),
        text_dim             = ecfg.get("text_hidden_dim", 768),
        audio_dim            = ecfg.get("audio_hidden_dim", 768),
        prosody_dim          = pcfg.get("feature_dim", 16),
        prosody_hidden_dim   = mcfg.get("prosody_hidden_dim", 64),
        video_dim            = ecfg.get("video_hidden_dim", 2048),
        dropout              = mcfg.get("dropout", 0.3),
        classifier_dropout   = mcfg.get("classifier_dropout", 0.2),
        fusion_mode          = fusion_mode,
        modalities           = modalities,
    )
