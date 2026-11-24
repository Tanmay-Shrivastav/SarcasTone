"""
SarcasTone — Training Package
==============================
Exports training utilities so the rest of the codebase can import them from
``src.training`` directly.

Usage
-----
    from src.training import SarcasmLoss, Trainer, compute_metrics
    from src.training.metrics import MetricTracker
"""

from src.training.losses import SarcasmLoss  # noqa: F401
from src.training.metrics import compute_metrics, MetricTracker  # noqa: F401
from src.training.trainer import Trainer  # noqa: F401

__all__ = [
    "SarcasmLoss",
    "compute_metrics",
    "MetricTracker",
    "Trainer",
]
