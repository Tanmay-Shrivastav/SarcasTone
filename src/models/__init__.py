"""
SarcasTone — Model Package
==========================
Exports the top-level PGIN-Sarcasm model and the IncongruityGate module so
that the rest of the codebase can import them from `src.models` directly.

Usage
-----
    from src.models import PGINSarcasm, build_model
    from src.models.incongruity_gate import IncongruityGate
"""

from src.models.incongruity_gate import IncongruityGate  # noqa: F401
from src.models.pgin_sarcasm import PGINSarcasm, build_model  # noqa: F401

__all__ = ["IncongruityGate", "PGINSarcasm", "build_model"]
