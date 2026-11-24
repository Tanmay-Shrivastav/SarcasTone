"""
metrics.py
==========
Evaluation metrics for SarcasTone sarcasm detection experiments.

Implements consistent computation and logging of:
  - Accuracy
  - Precision  (macro, binary positive class = 1)
  - Recall     (macro, binary positive class = 1)
  - F1-score   (macro)
  - Confusion matrix
  - Mean learned gate weights (for pgin runs)

Also provides MetricTracker — a lightweight stateful accumulator that collects
per-batch predictions during an epoch and flushes them at epoch end.

Usage
-----
    # One-shot computation
    result = compute_metrics(logits, labels)

    # Stateful accumulation over batches
    tracker = MetricTracker()
    for logits, labels in dataloader:
        tracker.update(logits, labels, gate_weights)
    epoch_metrics = tracker.compute()
    tracker.reset()
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


# ---------------------------------------------------------------------------
# Public one-shot function
# ---------------------------------------------------------------------------

def compute_metrics(
    logits:       torch.Tensor,
    labels:       torch.Tensor,
    gate_weights: Optional[torch.Tensor] = None,
    threshold:    float = 0.5,
) -> Dict[str, object]:
    """Compute classification metrics from raw model logits.

    Parameters
    ----------
    logits       : (N,) or (N, 1) raw pre-sigmoid model outputs
    labels       : (N,) or (N, 1) binary ground truth {0, 1}
    gate_weights : (N, 4) or None — per-sample gate weights for pgin runs.
                   Columns correspond to [text, speech, inc, video].
    threshold    : decision threshold on sigmoid probability (default 0.5)

    Returns
    -------
    dict with keys:
      "accuracy"   : float ∈ [0, 1]
      "precision"  : float ∈ [0, 1]  (zero_division=0)
      "recall"     : float ∈ [0, 1]  (zero_division=0)
      "f1"         : float ∈ [0, 1]  (macro)
      "confusion_matrix": list[list[int]]  [[TN, FP], [FN, TP]]
      "mean_gate_weights": list[float] or None  — [g_T, g_A, g_inc, g_V]
      "n_samples"  : int
      "n_sarcastic": int
    """
    logits_np = logits.detach().cpu().float().view(-1).numpy()
    labels_np = labels.detach().cpu().long().view(-1).numpy()

    probs = 1.0 / (1.0 + np.exp(-logits_np))  # sigmoid
    preds = (probs >= threshold).astype(int)

    acc  = float(accuracy_score(labels_np, preds))
    prec = float(precision_score(labels_np, preds, zero_division=0))
    rec  = float(recall_score(labels_np, preds, zero_division=0))
    f1   = float(f1_score(labels_np, preds, zero_division=0))
    cm   = confusion_matrix(labels_np, preds).tolist()

    mean_gates: Optional[List[float]] = None
    if gate_weights is not None:
        gw = gate_weights.detach().cpu().float().numpy()
        mean_gates = gw.mean(axis=0).tolist()  # [g_T, g_A, g_inc, g_V]

    return {
        "accuracy":          acc,
        "precision":         prec,
        "recall":            rec,
        "f1":                f1,
        "confusion_matrix":  cm,
        "mean_gate_weights": mean_gates,
        "n_samples":         int(len(labels_np)),
        "n_sarcastic":       int(labels_np.sum()),
    }


# ---------------------------------------------------------------------------
# MetricTracker
# ---------------------------------------------------------------------------

class MetricTracker:
    """Stateful accumulator for per-batch predictions across an epoch.

    Accumulates logits, labels, and optional gate weights in CPU buffers, then
    computes full metrics at epoch end without redundant forward passes.

    Example
    -------
        tracker = MetricTracker()
        for batch in loader:
            out = model(**batch)
            tracker.update(out["logit"], batch["label"], out["gate_weights"])
        metrics = tracker.compute()
        tracker.reset()
    """

    def __init__(self) -> None:
        self._logits:  List[torch.Tensor] = []
        self._labels:  List[torch.Tensor] = []
        self._gates:   List[torch.Tensor] = []
        self._losses:  List[float]        = []

    def update(
        self,
        logits:       torch.Tensor,
        labels:       torch.Tensor,
        gate_weights: Optional[torch.Tensor] = None,
        loss:         Optional[float]        = None,
    ) -> None:
        """Append a mini-batch's worth of predictions.

        Parameters
        ----------
        logits       : (B,) or (B, 1)
        labels       : (B,) or (B, 1)
        gate_weights : (B, 4) or None
        loss         : scalar float or None — step loss for running average
        """
        self._logits.append(logits.detach().cpu())
        self._labels.append(labels.detach().cpu())
        if gate_weights is not None:
            self._gates.append(gate_weights.detach().cpu())
        if loss is not None:
            self._losses.append(float(loss))

    def compute(self, threshold: float = 0.5) -> Dict[str, object]:
        """Flush accumulated predictions and return epoch metrics.

        Returns
        -------
        dict — same schema as ``compute_metrics`` plus "mean_loss".
        """
        if not self._logits:
            return {}

        all_logits = torch.cat(self._logits, dim=0)
        all_labels = torch.cat(self._labels, dim=0)
        all_gates  = (
            torch.cat(self._gates, dim=0) if self._gates else None
        )

        result = compute_metrics(all_logits, all_labels, all_gates, threshold)
        result["mean_loss"] = (
            float(np.mean(self._losses)) if self._losses else None
        )
        return result

    def reset(self) -> None:
        """Clear all accumulated state."""
        self._logits.clear()
        self._labels.clear()
        self._gates.clear()
        self._losses.clear()

    @property
    def n_batches(self) -> int:
        return len(self._logits)
