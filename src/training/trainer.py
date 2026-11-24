"""
trainer.py
==========
Training and evaluation loop for SarcasTone PGIN-Sarcasm experiments.

Design goals
------------
* Mixed-precision training via ``torch.amp.autocast('cuda')`` + ``GradScaler``
  tuned for the Colab T4 (16 GB VRAM).  Gracefully falls back to FP32 when
  CUDA is unavailable (CPU / MPS).
* AdamW optimizer with cosine-annealing learning-rate schedule + linear warm-up.
* Gradient clipping (default norm 1.0).
* Early stopping on validation **macro F1-score** with configurable patience.
* Checkpoint saving: best model by val-F1 (``best_model.pt``) and optionally
  the last epoch (``last_model.pt``).
* Consistent metric logging via MetricTracker.

Usage
-----
    trainer = Trainer(
        model      = model,
        criterion  = SarcasmLoss(lambda_inc=0.1, inc_margin=0.5),
        cfg        = cfg,
        run_dir    = Path("results/text_speech_pgin"),
        device     = torch.device("cuda"),
    )
    test_metrics = trainer.fit(train_loader, val_loader, test_loader)
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
try:
    from tqdm.auto import tqdm
except ImportError:
    from tqdm import tqdm

from src.training.losses import SarcasmLoss
from src.training.metrics import MetricTracker

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """Full training / evaluation lifecycle for one ablation run.

    Parameters
    ----------
    model     : nn.Module          — PGINSarcasm instance (on CPU; Trainer moves it).
    criterion : SarcasmLoss        — combined BCE + incongruity loss.
    cfg       : dict               — parsed YAML config (``configs/default.yaml``).
    run_dir   : Path               — output directory for this run's artifacts.
    device    : torch.device       — target device (default: cuda if available).
    """

    def __init__(
        self,
        model:     nn.Module,
        criterion: SarcasmLoss,
        cfg:       dict,
        run_dir:   Path,
        device:    Optional[torch.device] = None,
        show_progress: bool = True,
    ) -> None:
        self.cfg      = cfg
        self.run_dir  = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.model     = model.to(self.device)
        self.criterion = criterion

        # ---- Hyper-parameters from config -----------------------------------
        tcfg = cfg.get("training", {})
        self.epochs         = int(tcfg.get("epochs", 40))
        self.lr             = float(tcfg.get("lr", 3e-4))
        self.weight_decay   = float(tcfg.get("weight_decay", 1e-2))
        self.patience       = int(tcfg.get("patience", 8))
        self.grad_clip_norm = float(tcfg.get("grad_clip_norm", 1.0))
        self.use_amp        = bool(tcfg.get("use_amp", True)) and self.device.type == "cuda"
        self.show_progress  = show_progress and bool(tcfg.get("show_progress", True))
        self.is_tty         = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()

        # ---- Optimizer & LR schedule ----------------------------------------
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=self.epochs, eta_min=self.lr * 0.01
        )

        # ---- Mixed precision -------------------------------------------------
        # torch.amp.GradScaler is available from PyTorch ≥ 2.0
        self.scaler = (
            torch.amp.GradScaler("cuda") if self.use_amp else None
        )

        # ---- Internal state -------------------------------------------------
        self.best_val_f1   = -1.0
        self.best_epoch    = 0
        self._no_improve   = 0
        self._history: Dict[str, list] = {
            "train_loss": [], "val_loss": [], "val_f1": [], "val_acc": []
        }

    # -----------------------------------------------------------------------
    # Internal: single batch → model output
    # -----------------------------------------------------------------------

    def _batch_to_device(self, batch: dict) -> dict:
        """Move all tensor values in a batch dict to self.device."""
        return {
            k: (v.to(self.device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }

    def _forward_batch(self, batch: dict) -> tuple:
        """Run model on batch, return (output_dict, labels)."""
        batch  = self._batch_to_device(batch)
        labels = batch["label"]                     # (B,)

        output = self.model(
            text_emb    = batch.get("text_emb"),
            audio_emb   = batch.get("audio_emb"),
            prosody_vec = batch.get("prosody_vec"),
            video_emb   = batch.get("video_emb"),
        )
        return output, labels

    # -----------------------------------------------------------------------
    # Training step
    # -----------------------------------------------------------------------

    def _train_epoch(self, loader: DataLoader, epoch: int = 1) -> float:
        """One full training epoch; returns mean total loss."""
        self.model.train()
        total_loss = 0.0
        n_batches  = 0

        # In non-TTY streams (piped logs, background jobs), inner batch bars cause
        # line spam because carriage returns cannot overwrite lines. Disable batch bars
        # when not attached to a TTY, letting the outer epoch progress / logger handle progress.
        show_batch_bar = self.show_progress and self.is_tty

        batch_pbar = tqdm(
            loader,
            desc=f"Epoch {epoch:2d}/{self.epochs} [Train]",
            unit="batch",
            leave=False,
            mininterval=1.0,
            miniters=4,
            dynamic_ncols=True,
            disable=not show_batch_bar,
        )

        for batch in batch_pbar:
            self.optimizer.zero_grad(set_to_none=True)

            if self.use_amp:
                with torch.amp.autocast("cuda"):
                    output, labels = self._forward_batch(batch)
                    loss, bce, aux = self.criterion(
                        logit           = output["logit"],
                        labels          = labels,
                        polarity_text   = output.get("polarity_text"),
                        polarity_speech = output.get("polarity_speech"),
                    )
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                output, labels = self._forward_batch(batch)
                loss, bce, aux = self.criterion(
                    logit           = output["logit"],
                    labels          = labels,
                    polarity_text   = output.get("polarity_text"),
                    polarity_speech = output.get("polarity_speech"),
                )
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                self.optimizer.step()

            loss_val = float(loss.item())
            total_loss += loss_val
            n_batches  += 1
            if show_batch_bar:
                batch_pbar.set_postfix({"loss": f"{loss_val:.4f}"}, refresh=False)

        return total_loss / max(n_batches, 1)

    # -----------------------------------------------------------------------
    # Evaluation step
    # -----------------------------------------------------------------------

    @torch.no_grad()
    def _eval_epoch(
        self,
        loader: DataLoader,
        desc: str = "Evaluating",
    ) -> Dict[str, object]:
        """Evaluate on a DataLoader; returns metrics dict."""
        self.model.eval()
        tracker = MetricTracker()

        show_batch_bar = self.show_progress and self.is_tty

        eval_pbar = tqdm(
            loader,
            desc=desc,
            unit="batch",
            leave=False,
            mininterval=1.0,
            miniters=4,
            dynamic_ncols=True,
            disable=not show_batch_bar,
        )

        for batch in eval_pbar:
            if self.use_amp:
                with torch.amp.autocast("cuda"):
                    output, labels = self._forward_batch(batch)
                    loss, _, _ = self.criterion(
                        logit           = output["logit"],
                        labels          = labels,
                        polarity_text   = output.get("polarity_text"),
                        polarity_speech = output.get("polarity_speech"),
                    )
            else:
                output, labels = self._forward_batch(batch)
                loss, _, _ = self.criterion(
                    logit           = output["logit"],
                    labels          = labels,
                    polarity_text   = output.get("polarity_text"),
                    polarity_speech = output.get("polarity_speech"),
                )

            loss_val = float(loss.item())
            tracker.update(
                logits       = output["logit"].float(),
                labels       = labels,
                gate_weights = output.get("gate_weights"),
                loss         = loss_val,
            )
            if show_batch_bar:
                eval_pbar.set_postfix({"loss": f"{loss_val:.4f}"}, refresh=False)

        return tracker.compute()

    # -----------------------------------------------------------------------
    # Checkpoint helpers
    # -----------------------------------------------------------------------

    def _save_checkpoint(self, filename: str = "best_model.pt", epoch: int = 0) -> None:
        ckpt_path = self.run_dir / filename
        torch.save(
            {
                "epoch":      epoch,
                "model_state_dict":     self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "scaler_state_dict":    self.scaler.state_dict() if self.scaler else None,
                "best_val_f1":          self.best_val_f1,
                "config":               self.cfg,
            },
            ckpt_path,
        )
        logger.info(f"Checkpoint saved → {ckpt_path}")

    # -----------------------------------------------------------------------
    # Public: fit
    # -----------------------------------------------------------------------

    def fit(
        self,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        test_loader:  Optional[DataLoader] = None,
    ) -> Dict[str, object]:
        """Train the model and return test metrics at the best checkpoint.

        Parameters
        ----------
        train_loader : DataLoader  — training split
        val_loader   : DataLoader  — validation split (early stopping / ckpt)
        test_loader  : DataLoader or None  — if given, evaluate at the end

        Returns
        -------
        dict — test metrics (or val metrics if test_loader is None).
        """
        logger.info(
            f"Starting training | epochs={self.epochs} | "
            f"device={self.device} | amp={self.use_amp}"
        )

        run_label = self.run_dir.name
        epoch_pbar = tqdm(
            range(1, self.epochs + 1),
            desc=f"Training [{run_label}]",
            unit="epoch",
            leave=True,
            mininterval=1.0 if self.is_tty else 2.0,
            miniters=1,
            dynamic_ncols=True,
            disable=not self.show_progress,
        )

        for epoch in epoch_pbar:
            train_loss = self._train_epoch(train_loader, epoch=epoch)
            val_metrics = self._eval_epoch(
                val_loader,
                desc=f"Epoch {epoch:2d}/{self.epochs} [Val]",
            )
            val_f1   = float(val_metrics.get("f1", 0.0))
            val_loss = float(val_metrics.get("mean_loss", 0.0))
            val_acc  = float(val_metrics.get("accuracy", 0.0))

            self._history["train_loss"].append(train_loss)
            self._history["val_loss"].append(val_loss)
            self._history["val_f1"].append(val_f1)
            self._history["val_acc"].append(val_acc)

            self.scheduler.step()

            best_display = self.best_val_f1 if self.best_val_f1 >= 0 else val_f1
            epoch_pbar.set_postfix({
                "loss": f"{train_loss:.4f}",
                "val_loss": f"{val_loss:.4f}",
                "val_f1": f"{val_f1:.4f}",
                "best_f1": f"{best_display:.4f}",
            }, refresh=False)

            logger.info(
                f"Epoch {epoch:3d}/{self.epochs} | "
                f"train_loss={train_loss:.4f} | "
                f"val_loss={val_loss:.4f} | "
                f"val_acc={val_acc:.4f} | "
                f"val_f1={val_f1:.4f}"
            )

            # ---- Early stopping / best checkpoint --------------------------
            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                self.best_epoch  = epoch
                self._no_improve = 0
                self._save_checkpoint("best_model.pt", epoch=epoch)
                logger.info(f"  ↑ New best val F1={val_f1:.4f} at epoch {epoch}")
            else:
                self._no_improve += 1
                if self._no_improve >= self.patience:
                    logger.info(
                        f"Early stopping at epoch {epoch} "
                        f"(no improvement for {self.patience} epochs)"
                    )
                    break

        # Save last checkpoint
        self._save_checkpoint("last_model.pt", epoch=epoch)

        # ---- Reload best model and evaluate test set -----------------------
        best_ckpt = self.run_dir / "best_model.pt"
        if best_ckpt.exists():
            try:
                state = torch.load(best_ckpt, map_location=self.device, weights_only=True)
            except Exception:
                state = torch.load(best_ckpt, map_location=self.device, weights_only=False)
            self.model.load_state_dict(state["model_state_dict"])
            logger.info(
                f"Best model loaded from epoch {state.get('epoch', '?')} "
                f"(val F1={self.best_val_f1:.4f})"
            )

        if test_loader is not None:
            logger.info("Evaluating on test split …")
            test_metrics = self._eval_epoch(
                test_loader,
                desc=f"Evaluating Test [{run_label}]",
            )
        else:
            test_metrics = self._eval_epoch(
                val_loader,
                desc=f"Evaluating Val [{run_label}]",
            )

        # Enrich with run metadata
        test_metrics["best_epoch"]  = self.best_epoch
        test_metrics["best_val_f1"] = self.best_val_f1
        test_metrics["history"]     = self._history

        # Persist metrics JSON
        metrics_path = self.run_dir / "metrics.json"
        _safe_json = {
            k: (v if not isinstance(v, float) or (v == v and v not in (float("inf"), float("-inf"))) else str(v))
            for k, v in test_metrics.items()
        }
        with open(metrics_path, "w") as f:
            json.dump(_safe_json, f, indent=2)
        logger.info(f"Test metrics saved → {metrics_path}")

        return test_metrics

    # -----------------------------------------------------------------------
    # Public: standalone evaluate (no training)
    # -----------------------------------------------------------------------

    def evaluate(
        self,
        loader: DataLoader,
        desc: str = "Evaluating",
    ) -> Dict[str, object]:
        """Run evaluation only (no weight updates)."""
        return self._eval_epoch(loader, desc=desc)
