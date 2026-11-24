"""
losses.py
=========
Loss functions for SarcasTone PGIN-Sarcasm training.

Objective (SRD §4.3):
    L = BCEWithLogitsLoss(ŷ, y)
      + λ_inc · L_inc(ŝ_T, ŝ_A, y)

Auxiliary incongruity loss L_inc:
  For *sarcastic* utterances (y == 1) the network should produce a large
  polarity mismatch |ŝ_T − ŝ_A|.  A margin loss encodes this:
      L_inc = max(0, δ − |ŝ_T − ŝ_A|)   for y == 1   (push divergence ≥ δ)
              max(0, |ŝ_T − ŝ_A| − δ_neg) for y == 0  (optional push congruence)
  Only the y==1 margin is active by default (non-sarcastic utterances carry
  no auxiliary penalty, since non-sarcastic speech can still have expressive
  prosody that happens to differ from text sentiment).

  When ŝ_T or ŝ_A is not available (unimodal runs), L_inc is 0 and the total
  loss reduces to plain BCEWithLogitsLoss.

Usage
-----
    criterion = SarcasmLoss(lambda_inc=0.1, inc_margin=0.5)
    out = model(...)
    total, bce, aux = criterion(
        logit          = out["logit"],
        labels         = labels,
        polarity_text  = out["polarity_text"],
        polarity_speech= out["polarity_speech"],
    )
    total.backward()
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SarcasmLoss(nn.Module):
    """Combined BCEWithLogitsLoss + optional incongruity margin auxiliary loss.

    Parameters
    ----------
    lambda_inc : float
        Weight λ for the auxiliary incongruity loss (0.0 → pure BCE, default 0.1).
    inc_margin : float
        Margin δ: for sarcastic samples, the network is penalised if
        |ŝ_T − ŝ_A| < δ (default 0.5).
    pos_weight : float or None
        Optional class-imbalance weight passed to BCEWithLogitsLoss.
    reduction : str
        'mean' (default) | 'sum' | 'none'
    """

    def __init__(
        self,
        lambda_inc: float = 0.1,
        inc_margin: float = 0.5,
        pos_weight: Optional[float] = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.lambda_inc = lambda_inc
        self.inc_margin = inc_margin
        self.reduction  = reduction

        pw = torch.tensor([pos_weight]) if pos_weight is not None else None
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pw, reduction=reduction)

    def _incongruity_loss(
        self,
        s_T: torch.Tensor,
        s_A: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the auxiliary incongruity margin loss.

        For sarcastic samples (label == 1):
            penalty = max(0, δ − |ŝ_T − ŝ_A|)
        For non-sarcastic samples: no penalty.

        Parameters
        ----------
        s_T    : (B, 1) text polarity proxy ∈ (-1,1)
        s_A    : (B, 1) speech polarity proxy ∈ (-1,1)
        labels : (B,) or (B,1) binary ground truth {0, 1}

        Returns
        -------
        Scalar loss tensor.
        """
        labels_f = labels.view(-1, 1).float()   # (B, 1)
        divergence = torch.abs(s_T - s_A)       # (B, 1)

        # Margin hinge for sarcastic utterances
        hinge = F.relu(self.inc_margin - divergence)  # (B, 1)
        per_sample = labels_f * hinge                 # zero out non-sarcastic

        if self.reduction == "mean":
            # Normalise over sarcastic samples (avoid div-by-zero)
            n_sarc = labels_f.sum().clamp(min=1.0)
            return per_sample.sum() / n_sarc
        elif self.reduction == "sum":
            return per_sample.sum()
        else:  # 'none'
            return per_sample.squeeze(-1)

    def forward(
        self,
        logit:           torch.Tensor,
        labels:          torch.Tensor,
        polarity_text:   Optional[torch.Tensor] = None,
        polarity_speech: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        logit           : (B, 1) raw model output (pre-sigmoid)
        labels          : (B,) or (B, 1) binary {0.0, 1.0}
        polarity_text   : (B, 1) ŝ_T or None
        polarity_speech : (B, 1) ŝ_A or None

        Returns
        -------
        (total_loss, bce_loss, aux_loss) — all scalar tensors.
        """
        labels_f = labels.view(-1, 1).float()

        bce_loss = self.bce(logit, labels_f)

        # Auxiliary incongruity loss (only when BOTH polarities are available)
        if (
            self.lambda_inc > 0.0
            and polarity_text   is not None
            and polarity_speech is not None
        ):
            aux_loss = self._incongruity_loss(polarity_text, polarity_speech, labels)
        else:
            aux_loss = torch.zeros_like(bce_loss)

        total_loss = bce_loss + self.lambda_inc * aux_loss
        return total_loss, bce_loss, aux_loss
