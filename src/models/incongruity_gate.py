"""
incongruity_gate.py
===================
Prosody-Aware Incongruity Gate for SarcasTone (PGIN-Sarcasm).

This module implements:
  1. The explicit text-vs.-speech incongruity feature computation
     (signed/absolute polarity mismatch, vector diff/product, cosine sim).
  2. The incongruity representation network z_inc.
  3. The learned modality gating MLP that dynamically weights active branch
     vectors {h_T, h_A, z_inc, h_V} before final classification.

It is used exclusively by pgin_sarcasm.py and operates on **already-projected**
latent representations of shape (B, d).

Design notes
------------
* When only one modality is active (unimodal forward passes), incongruity
  is undefined and gating degenerates gracefully to a single-branch pass.
* The gating MLP produces a Softmax distribution over *active* branches only;
  inactive branches receive gate weight 0 and are masked to zero before the
  weighted sum.
* fusion_mode="concat" bypasses this module entirely (handled in pgin_sarcasm).

Math (see SRD §4.2):
  m_signed  = ŝ_T  - ŝ_A                                    (B, 1)
  m_abs     = |ŝ_T - ŝ_A|                                    (B, 1)
  m_diff    = |h_T  - h_A|                                    (B, d)
  m_prod    = h_T ⊙ h_A                                      (B, d)
  c_TA      = cos(h_T, h_A)                                  (B, 1)
  z_inc     = GELU(Linear(2d+3 → d)([m_diff; m_prod; m_signed; m_abs; c_TA]))
  gate_in   = concat of active branch vectors                 (B, n*d)
  g         = Softmax(Linear(n*d → n), masked to active slots) (B, n)
  h_fused   = Σ_i g_i * branch_i                             (B, d)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Incongruity Representation Network
# ---------------------------------------------------------------------------

class IncongruityNet(nn.Module):
    """Maps vector-level mismatch features to a d-dimensional incongruity repr.

    Input concatenation:  [m_diff ; m_prod ; m_signed ; m_abs ; c_TA]
                          shape = (B, 2*d + 3)
    """

    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        in_dim = 2 * latent_dim + 3  # m_diff(d) + m_prod(d) + 3 scalars
        self.proj = nn.Linear(in_dim, latent_dim)
        self.norm = nn.LayerNorm(latent_dim)

    def forward(
        self,
        h_T: torch.Tensor,
        h_A: torch.Tensor,
        s_T: torch.Tensor,
        s_A: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        h_T : (B, d)   projected text representation
        h_A : (B, d)   projected speech representation
        s_T : (B, 1)   text polarity proxy  ∈ (-1, 1)
        s_A : (B, 1)   speech polarity proxy ∈ (-1, 1)

        Returns
        -------
        z_inc : (B, d)
        """
        m_signed = s_T - s_A                           # (B, 1)
        m_abs    = torch.abs(m_signed)                 # (B, 1)
        m_diff   = torch.abs(h_T - h_A)               # (B, d)
        m_prod   = h_T * h_A                           # (B, d)

        # Cosine similarity → (B, 1)
        c_TA = F.cosine_similarity(h_T, h_A, dim=-1, eps=1e-8).unsqueeze(-1)

        feat = torch.cat([m_diff, m_prod, m_signed, m_abs, c_TA], dim=-1)
        z_inc = self.norm(F.gelu(self.proj(feat)))
        return z_inc


# ---------------------------------------------------------------------------
# Modality Gating MLP
# ---------------------------------------------------------------------------

class ModalityGate(nn.Module):
    """Learned Softmax gating over active modality branches.

    Branches (in fixed order): [text, speech, incongruity, video]
    Inactive branches are masked *before* Softmax so they receive weight 0.

    Parameters
    ----------
    latent_dim : int
        Shared latent dimension d for all branch vectors.
    n_branches : int
        Maximum number of branches (default 4: T, A, inc, V).
    hidden_dim : int
        Hidden dimension of the gating MLP.
    """

    N_BRANCHES = 4  # text | speech | incongruity | video

    def __init__(self, latent_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        # Gate input = concatenation of ALL branch vectors (zeros for inactive)
        gate_in_dim = self.N_BRANCHES * latent_dim
        self.gate_mlp = nn.Sequential(
            nn.Linear(gate_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.N_BRANCHES),
        )

    def forward(
        self,
        branches: dict[str, torch.Tensor | None],
        active_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        branches : dict with keys 'text', 'speech', 'inc', 'video'
                   Each value is (B, d) or None.
        active_mask : (N_BRANCHES,) bool tensor indicating which branches exist.

        Returns
        -------
        h_fused : (B, d)
        gate_weights : (B, N_BRANCHES)  — Softmax weights (zeros for inactive).
        """
        d = self.latent_dim
        B = next(v for v in branches.values() if v is not None).shape[0]
        device = next(v for v in branches.values() if v is not None).device

        # Build padded branch tensor (B, 4, d)
        ordered = ["text", "speech", "inc", "video"]
        stack = []
        for key in ordered:
            v = branches.get(key)
            if v is not None:
                stack.append(v)
            else:
                stack.append(torch.zeros(B, d, device=device, dtype=torch.float32))
        branch_tensor = torch.stack(stack, dim=1)       # (B, 4, d)
        gate_in = branch_tensor.view(B, -1)             # (B, 4*d)

        # Raw gate logits → mask inactive → Softmax
        raw_logits = self.gate_mlp(gate_in)             # (B, 4)

        # active_mask: (4,) bool  →  set inactive to -inf before softmax
        mask = active_mask.to(device=device, dtype=torch.bool)
        raw_logits = raw_logits.masked_fill(~mask.unsqueeze(0), float("-inf"))
        gate_weights = torch.softmax(raw_logits, dim=-1)  # (B, 4)

        # Weighted sum over branches
        # branch_tensor: (B, 4, d), gate_weights: (B, 4)
        h_fused = (gate_weights.unsqueeze(-1) * branch_tensor).sum(dim=1)  # (B, d)
        return h_fused, gate_weights


# ---------------------------------------------------------------------------
# Public IncongruityGate: wraps IncongruityNet + ModalityGate
# ---------------------------------------------------------------------------

class IncongruityGate(nn.Module):
    """Prosody-Aware Incongruity Gate (Section 4.2 of SRD).

    Computes z_inc from (h_T, h_A, ŝ_T, ŝ_A) and applies learned gating over
    all active branch vectors to produce h_fused.

    Parameters
    ----------
    latent_dim : int
        Latent dimension d (must match PGINSarcasm projections).
    gate_hidden_dim : int
        Hidden size of the gating MLP (default 128).
    """

    def __init__(self, latent_dim: int, gate_hidden_dim: int = 128) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.inc_net = IncongruityNet(latent_dim)
        self.mod_gate = ModalityGate(latent_dim, hidden_dim=gate_hidden_dim)

    def forward(
        self,
        h_T: torch.Tensor | None,
        h_A: torch.Tensor | None,
        h_V: torch.Tensor | None,
        s_T: torch.Tensor | None,
        s_A: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """
        Parameters
        ----------
        h_T : (B, d) or None   — text latent
        h_A : (B, d) or None   — speech latent
        h_V : (B, d) or None   — video latent
        s_T : (B, 1) or None   — text polarity proxy
        s_A : (B, 1) or None   — speech polarity proxy

        Returns
        -------
        h_fused      : (B, d)
        gate_weights : (B, 4)   [g_T, g_A, g_inc, g_V]
        z_inc        : (B, d) or None  — incongruity repr if both T & A active
        """
        # Determine batch size from first active branch
        candidates = [h_T, h_A, h_V]
        B = next(x for x in candidates if x is not None).shape[0]
        d = self.latent_dim
        device = next(x for x in candidates if x is not None).device

        # Compute incongruity repr only when BOTH text AND speech are available
        z_inc: torch.Tensor | None = None
        if h_T is not None and h_A is not None and s_T is not None and s_A is not None:
            z_inc = self.inc_net(h_T, h_A, s_T, s_A)   # (B, d)

        # Branch dict for gating  (None → zero-padded inside ModalityGate)
        branches = {
            "text":   h_T,
            "speech": h_A,
            "inc":    z_inc,
            "video":  h_V,
        }
        # Active mask: shape (4,) bool
        active_mask = torch.tensor(
            [h_T is not None, h_A is not None, z_inc is not None, h_V is not None],
            dtype=torch.bool,
        )

        h_fused, gate_weights = self.mod_gate(branches, active_mask)
        return h_fused, gate_weights, z_inc
