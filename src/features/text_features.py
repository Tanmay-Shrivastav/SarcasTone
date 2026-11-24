"""
Text feature extraction using pretrained Transformer backbones (DeBERTa-v3 / RoBERTa).
Extracts attention-masked mean-pooled representations of shape (768,).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

try:
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore
    TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)

TEXT_EMB_DIM = 768


class TextFeatureExtractor:
    """
    Extracts fixed-dimensional semantic representations from utterance text
    (and optional preceding dialogue context) using DeBERTa-v3-base.
    """

    def __init__(
        self,
        model_name: str = "microsoft/deberta-v3-base",
        fallback_model: str = "roberta-base",
        max_length: int = 128,
        use_context: bool = False,
        device: Optional[str] = None,
        use_amp: bool = True,
    ):
        self.model_name = model_name
        self.fallback_model = fallback_model
        self.max_length = max_length
        self.use_context = use_context
        self.use_amp = use_amp

        if device is None:
            if torch is not None and torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device

        self.tokenizer = None
        self.model = None
        self._is_initialized = False

    def _initialize_model(self) -> None:
        """Lazily load Hugging Face tokenizer and model on target device."""
        if self._is_initialized:
            return

        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError(
                "transformers and torch are required to run TextFeatureExtractor. "
                "Please run on Google Colab or install requirements.txt."
            )

        logger.info(f"Loading text encoder: {self.model_name} on {self.device}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
        except Exception as e:
            logger.warning(
                f"Failed to load primary model '{self.model_name}': {e}. "
                f"Attempting fallback model: '{self.fallback_model}'..."
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.fallback_model)
            self.model = AutoModel.from_pretrained(self.fallback_model)

        self.model.to(self.device)
        self.model.eval()
        self._is_initialized = True
        logger.info(f"Text encoder loaded successfully on {self.device}.")

    def format_input_text(self, text: str, context: Optional[str] = None) -> str:
        """
        Format target utterance with optional preceding dialogue context.
        Uses standard separator token '</s>' between context and target utterance.
        """
        clean_text = str(text).strip() if text else ""
        if self.use_context and context:
            clean_context = str(context).strip()
            if clean_context:
                return f"{clean_context} </s> {clean_text}"
        return clean_text

    @staticmethod
    def _mean_pooling(last_hidden_state: Any, attention_mask: Any) -> Any:
        """
        Compute attention-weighted mean pooling across token representations.
        Zeros out padded positions before averaging.
        """
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def extract_single(
        self,
        text: str,
        context: Optional[str] = None,
    ) -> np.ndarray:
        """
        Extract attention-masked mean-pooled text embedding of shape (768,).
        """
        embeddings = self.extract_batch([text], contexts=[context] if context else None)
        return embeddings[0]

    def extract_batch(
        self,
        texts: Sequence[str],
        contexts: Optional[Sequence[Optional[str]]] = None,
        batch_size: int = 32,
    ) -> np.ndarray:
        """
        Extract embeddings for a sequence of texts in batched mini-batches.
        Returns NumPy array of shape (N, 768) with float32 precision.
        """
        if not texts:
            return np.empty((0, TEXT_EMB_DIM), dtype=np.float32)

        self._initialize_model()

        # Format inputs with optional context
        formatted_texts: List[str] = []
        for i, text in enumerate(texts):
            ctx = contexts[i] if contexts and i < len(contexts) else None
            formatted_texts.append(self.format_input_text(text, ctx))

        all_embeddings: List[np.ndarray] = []

        for i in range(0, len(formatted_texts), batch_size):
            batch_slice = formatted_texts[i : i + batch_size]
            encoded = self.tokenizer(
                batch_slice,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)

            with torch.no_grad():
                if self.use_amp and self.device == "cuda":
                    with torch.amp.autocast("cuda"):
                        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                else:
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

                pooled = self._mean_pooling(outputs.last_hidden_state, attention_mask)

                # Project or ensure dimension matches 768
                if pooled.shape[-1] != TEXT_EMB_DIM:
                    # Defensive slice or pad if architecture diverges
                    if pooled.shape[-1] > TEXT_EMB_DIM:
                        pooled = pooled[:, :TEXT_EMB_DIM]
                    else:
                        pad_dim = TEXT_EMB_DIM - pooled.shape[-1]
                        pooled = F.pad(pooled, (0, pad_dim))

                all_embeddings.append(pooled.cpu().float().numpy())

        return np.concatenate(all_embeddings, axis=0).astype(np.float32)
