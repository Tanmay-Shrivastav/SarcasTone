"""
run_ablations.py
================
8-Run Modality Ablation Runner for SarcasTone (PGIN-Sarcasm).

Runs all ablation configurations defined in ``configs/default.yaml``
(``ablations.runs``) sequentially on the **same** train/val/test split,
then writes:
  • Per-run directory (under ``results_dir/<run_name>/``):
      - ``best_model.pt``  — best checkpoint by val F1
      - ``last_model.pt``  — checkpoint at the end of training
      - ``metrics.json``   — Accuracy, Precision, Recall, F1, confusion matrix,
                              mean gate weights, history
  • Master comparison table:
      - ``ablation_summary.csv``  — one row per run, all metrics
      - ``ablation_summary.json`` — structured dict of all runs

Usage (from repository root)
-----------------------------
    python -m src.experiments.run_ablations --config configs/default.yaml

Optional flags
--------------
    --results-dir   Override results_dir from config (useful on Colab).
    --splits-file   Override splits_file path.
    --epochs        Override training epoch count (e.g. quick smoke-test: 2).
    --device        Force device string, e.g. "cpu" or "cuda".
    --smoke-test    Train for 2 epochs, 1 batch each — for CI / quick checks.

Environment
-----------
  On Google Colab the results_dir should be inside /content/drive/MyDrive/…
  to persist across runtimes.  The script does NOT mount Google Drive itself;
  use the companion notebook (02_train_and_evaluate_colab.ipynb) for that.

Split contract
--------------
  The ``splits.json`` file (written by src/data/dataset.py) stores
  ``{"train": [...ids...], "val": [...ids...], "test": [...ids...]}``
  with utterance IDs drawn from the full dataset.  This file is loaded once
  and reused for every ablation run to guarantee identical splits.

Feature loading contract
------------------------
  Cached features are stored as ``{utterance_id}.pt`` tensors under
  ``cache_dir/{modality}/`` with the following keys per tensor dict:
      text_emb    : (768,)   float32
      audio_emb   : (768,)   float32
      prosody_vec : (16,)    float32   (StandardScaler already applied)
      video_emb   : (2048,)  float32
  Each ``.pt`` file also carries a ``label`` int key.
  Missing modality tensors for a given sample are represented as zero vectors
  (set during extraction by src/features/).
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import yaml
try:
    from tqdm.auto import tqdm
except ImportError:
    from tqdm import tqdm

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("run_ablations")


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Fix all random seeds for reproducibility (NFR-2)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # NOTE: fully deterministic conv ops can be ~10 % slower on T4;
        # enable only when strict reproducibility is required.
        torch.backends.cudnn.benchmark     = False
        torch.backends.cudnn.deterministic = True
    logger.info(f"Global seed set to {seed}")


# ---------------------------------------------------------------------------
# Feature-loading dataset (runs from cached .pt files)
# ---------------------------------------------------------------------------

class CachedFeatureDataset(torch.utils.data.Dataset):
    """Minimal dataset that reads pre-extracted feature tensors from disk.

    Expects ``cache_dir`` to contain one ``.pt`` file per utterance named
    ``{utterance_id}.pt`` with keys:
        label       : int   (0 or 1)
        text_emb    : Tensor (768,)    or absent / zeros
        audio_emb   : Tensor (768,)    or absent / zeros
        prosody_vec : Tensor (16,)     or absent / zeros
        video_emb   : Tensor (2048,)   or absent / zeros

    Parameters
    ----------
    utt_ids   : list[str]   — utterance IDs for this split.
    cache_dir : Path        — root directory of cached ``.pt`` files.
    modalities: list[str]   — which modality tensors to return.
    text_dim, audio_dim, prosody_dim, video_dim
                            — fallback dims for missing/zero tensors.
    """

    def __init__(
        self,
        utt_ids:    List[str],
        cache_dir:  Path,
        modalities: List[str],
        text_dim:   int = 768,
        audio_dim:  int = 768,
        prosody_dim:int = 16,
        video_dim:  int = 2048,
    ) -> None:
        self.utt_ids    = utt_ids
        self.cache_dir  = Path(cache_dir)
        self.modalities = modalities
        self.dims = {
            "text":   text_dim,
            "audio":  audio_dim,
            "prosody":prosody_dim,
            "video":  video_dim,
        }
        # Verify at least one file is accessible (optional warning)
        if utt_ids:
            probe = self.cache_dir / f"{utt_ids[0]}.pt"
            if not probe.exists():
                logger.warning(
                    f"Cache file not found for first utterance: {probe}\n"
                    "  → Will return zero tensors for all missing samples."
                )

    def __len__(self) -> int:
        return len(self.utt_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        utt_id = self.utt_ids[idx]
        path   = self.cache_dir / f"{utt_id}.pt"

        if path.exists():
            try:
                data = torch.load(path, map_location="cpu", weights_only=True)
            except Exception:
                data = torch.load(path, map_location="cpu", weights_only=False)
        else:
            data = {}

        label = int(data.get("label", 0))
        item  = {"label": torch.tensor(label, dtype=torch.long)}

        if "text" in self.modalities:
            item["text_emb"] = data.get(
                "text_emb",
                torch.zeros(self.dims["text"], dtype=torch.float32),
            ).float()

        if "speech" in self.modalities:
            item["audio_emb"] = data.get(
                "audio_emb",
                torch.zeros(self.dims["audio"], dtype=torch.float32),
            ).float()
            item["prosody_vec"] = data.get(
                "prosody_vec",
                torch.zeros(self.dims["prosody"], dtype=torch.float32),
            ).float()

        if "video" in self.modalities:
            item["video_emb"] = data.get(
                "video_emb",
                torch.zeros(self.dims["video"], dtype=torch.float32),
            ).float()

        return item


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def _make_loader(
    utt_ids:    List[str],
    cache_dir:  Path,
    modalities: List[str],
    batch_size: int,
    shuffle:    bool,
    num_workers:int,
    cfg:        dict,
    device:     Optional[torch.device] = None,
) -> torch.utils.data.DataLoader:
    ecfg = cfg.get("encoders", {})
    pcfg = cfg.get("prosody", {})
    dataset = CachedFeatureDataset(
        utt_ids     = utt_ids,
        cache_dir   = cache_dir,
        modalities  = modalities,
        text_dim    = ecfg.get("text_hidden_dim", 768),
        audio_dim   = ecfg.get("audio_hidden_dim", 768),
        prosody_dim = pcfg.get("feature_dim", 16),
        video_dim   = ecfg.get("video_hidden_dim", 2048),
    )
    is_cuda = (device.type == "cuda") if device is not None else torch.cuda.is_available()
    return torch.utils.data.DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = shuffle,
        num_workers = num_workers,
        pin_memory  = is_cuda,
        drop_last   = False,
    )


# ---------------------------------------------------------------------------
# Smoke-test dataset  (purely random tensors — no disk I/O)
# ---------------------------------------------------------------------------

class _SmokeDataset(torch.utils.data.Dataset):
    """Generates random feature tensors for quick CI validation."""

    def __init__(self, n: int, modalities: List[str], cfg: dict) -> None:
        self.n = n
        self.modalities = modalities
        ecfg = cfg.get("encoders", {})
        pcfg = cfg.get("prosody", {})
        self.text_dim    = ecfg.get("text_hidden_dim", 768)
        self.audio_dim   = ecfg.get("audio_hidden_dim", 768)
        self.prosody_dim = pcfg.get("feature_dim", 16)
        self.video_dim   = ecfg.get("video_hidden_dim", 2048)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {"label": torch.tensor(idx % 2, dtype=torch.long)}
        if "text"   in self.modalities:
            item["text_emb"]    = torch.randn(self.text_dim)
        if "speech" in self.modalities:
            item["audio_emb"]   = torch.randn(self.audio_dim)
            item["prosody_vec"] = torch.randn(self.prosody_dim)
        if "video"  in self.modalities:
            item["video_emb"]   = torch.randn(self.video_dim)
        return item


def _make_smoke_loader(
    n: int, modalities: List[str], batch_size: int, cfg: dict
) -> torch.utils.data.DataLoader:
    return torch.utils.data.DataLoader(
        _SmokeDataset(n, modalities, cfg),
        batch_size=batch_size,
        shuffle=True,
    )


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------

def run_single_ablation(
    run_cfg:     dict,
    cfg:         dict,
    splits:      dict,
    cache_dir:   Path,
    results_dir: Path,
    device:      torch.device,
    smoke_test:  bool = False,
) -> Dict[str, object]:
    """Train and evaluate one ablation configuration.

    Parameters
    ----------
    run_cfg     : dict  — single entry from ``cfg["ablations"]["runs"]``
    cfg         : dict  — full parsed YAML config
    splits      : dict  — {"train": [...], "val": [...], "test": [...]}
    cache_dir   : Path  — root of extracted feature cache
    results_dir : Path  — base results directory; run sub-dir is created here
    device      : torch.device
    smoke_test  : bool  — use tiny synthetic data for speed

    Returns
    -------
    dict — test metrics with run metadata included.
    """
    from src.models.pgin_sarcasm import build_model
    from src.training.losses import SarcasmLoss
    from src.training.trainer import Trainer

    run_name   = run_cfg["name"]
    modalities = run_cfg["modalities"]
    fusion     = run_cfg.get("fusion_mode", "pgin")

    logger.info("=" * 70)
    logger.info(f"RUN: {run_name}  |  modalities={modalities}  |  fusion={fusion}")
    logger.info("=" * 70)

    run_dir = results_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    tcfg       = cfg.get("training", {})
    batch_size = int(tcfg.get("batch_size", 32))
    n_workers  = int(tcfg.get("num_workers", 2))

    # ---- DataLoaders -------------------------------------------------------
    if smoke_test:
        logger.info("  [smoke-test] Using synthetic random data (8 samples per split).")
        train_loader = _make_smoke_loader(16, modalities, batch_size=4, cfg=cfg)
        val_loader   = _make_smoke_loader(8,  modalities, batch_size=4, cfg=cfg)
        test_loader  = _make_smoke_loader(8,  modalities, batch_size=4, cfg=cfg)
    else:
        train_loader = _make_loader(
            splits["train"], cache_dir, modalities, batch_size, True,  n_workers, cfg, device=device
        )
        val_loader   = _make_loader(
            splits["val"],   cache_dir, modalities, batch_size, False, n_workers, cfg, device=device
        )
        test_loader  = _make_loader(
            splits["test"],  cache_dir, modalities, batch_size, False, n_workers, cfg, device=device
        )

    # ---- Model & loss ------------------------------------------------------
    model     = build_model(cfg, modalities=modalities, fusion_mode=fusion)
    criterion = SarcasmLoss(
        lambda_inc = float(tcfg.get("lambda_inc", 0.1)),
        inc_margin = float(tcfg.get("inc_margin", 0.5)),
    )

    # ---- Trainer -----------------------------------------------------------
    trainer = Trainer(
        model     = model,
        criterion = criterion,
        cfg       = cfg,
        run_dir   = run_dir,
        device    = device,
    )

    t0           = time.perf_counter()
    test_metrics = trainer.fit(train_loader, val_loader, test_loader)
    elapsed      = time.perf_counter() - t0

    # Enrich with run identity and wall-clock time
    test_metrics["run_name"]   = run_name
    test_metrics["modalities"] = modalities
    test_metrics["fusion_mode"]= fusion
    test_metrics["elapsed_s"]  = round(elapsed, 1)

    logger.info(
        f"  DONE {run_name} | "
        f"acc={test_metrics.get('accuracy', 0):.4f} | "
        f"prec={test_metrics.get('precision', 0):.4f} | "
        f"rec={test_metrics.get('recall', 0):.4f} | "
        f"f1={test_metrics.get('f1', 0):.4f} | "
        f"elapsed={elapsed:.0f}s"
    )

    return test_metrics


# ---------------------------------------------------------------------------
# Summary table helpers
# ---------------------------------------------------------------------------

_SUMMARY_COLS = [
    "run_name",
    "modalities",
    "fusion_mode",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "best_val_f1",
    "best_epoch",
    "elapsed_s",
]


def _build_summary_rows(all_metrics: List[Dict]) -> List[Dict]:
    rows = []
    for m in all_metrics:
        row: dict = {}
        for col in _SUMMARY_COLS:
            val = m.get(col, "")
            if isinstance(val, list):
                val = "|".join(str(v) for v in val)
            if isinstance(val, float):
                val = round(val, 6)
            row[col] = val
        # Append mean gate weights as separate columns
        gw = m.get("mean_gate_weights")
        if gw is not None and isinstance(gw, (list, tuple)):
            names = ["gate_text", "gate_speech", "gate_inc", "gate_video"]
            for i, name in enumerate(names):
                row[name] = round(gw[i], 6) if i < len(gw) else ""
        rows.append(row)
    return rows


def _write_csv(rows: List[Dict], path: Path) -> None:
    if not rows:
        return
    # Union of all keys preserving order
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for k in row:
            if k not in seen:
                fieldnames.append(k)
                seen.add(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"Summary CSV written → {path}")


def _write_json(all_metrics: List[Dict], path: Path) -> None:
    # Strip un-serialisable tensors / numpy arrays
    def _clean(obj):
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_clean(v) for v in obj]
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        return obj

    cleaned = _clean(all_metrics)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, indent=2)
    logger.info(f"Summary JSON written → {path}")


def _print_summary_table(rows: List[Dict]) -> None:
    """Pretty-print a fixed-column ASCII comparison table."""
    cols = [
        ("run_name",   30),
        ("modalities", 24),
        ("fusion_mode", 8),
        ("accuracy",    8),
        ("precision",   9),
        ("recall",      7),
        ("f1",          7),
    ]
    sep = "+" + "+".join("-" * (w + 2) for _, w in cols) + "+"
    header = "|" + "|".join(
        f" {name:<{w}} " for name, w in cols
    ) + "|"

    print("\n" + sep)
    print(header)
    print(sep)
    for row in rows:
        line = "|"
        for col, w in cols:
            val = row.get(col, "")
            if isinstance(val, float):
                val = f"{val:.4f}"
            line += f" {str(val):<{w}} |"
        print(line)
    print(sep + "\n")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run all SarcasTone modality ablation experiments."
    )
    parser.add_argument(
        "--config", default="configs/default.yaml",
        help="Path to YAML configuration file (default: configs/default.yaml)."
    )
    parser.add_argument(
        "--results-dir", default=None,
        help="Override results_dir from config."
    )
    parser.add_argument(
        "--splits-file", default=None,
        help="Override splits_file path from config."
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override number of training epochs."
    )
    parser.add_argument(
        "--device", default=None,
        help='Force device string, e.g. "cpu" or "cuda".'
    )
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Run with tiny synthetic data (2 epochs) for quick validation."
    )
    args = parser.parse_args(argv)

    # ---- Load config -------------------------------------------------------
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        return 1
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # ---- Apply CLI overrides -----------------------------------------------
    if args.results_dir:
        cfg.setdefault("paths", {})["results_dir"] = args.results_dir
    if args.splits_file:
        cfg.setdefault("paths", {})["splits_file"] = args.splits_file
    if args.epochs:
        cfg.setdefault("training", {})["epochs"] = args.epochs
    if args.smoke_test:
        cfg.setdefault("training", {})["epochs"] = 2
        logger.info("[smoke-test] Overriding epochs=2")

    # ---- Reproducibility seed ----------------------------------------------
    seed = int(cfg.get("project", {}).get("seed", 42))
    set_seed(seed)

    # ---- Device selection --------------------------------------------------
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"CUDA device: {gpu_name}")
    else:
        device = torch.device("cpu")
        logger.warning("CUDA not available — running on CPU (slow).")
    logger.info(f"Using device: {device}")

    # ---- Paths -------------------------------------------------------------
    pcfg        = cfg.get("paths", {})
    results_dir = Path(pcfg.get("results_dir", "results"))
    cache_dir   = Path(pcfg.get("cache_dir",   "cache"))
    splits_file = Path(pcfg.get("splits_file", "cache/splits.json"))

    results_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Results directory: {results_dir}")

    # ---- Load splits -------------------------------------------------------
    if args.smoke_test:
        # Synthetic placeholder splits
        splits = {"train": [], "val": [], "test": []}
        logger.info("[smoke-test] Using empty splits (synthetic data only).")
    elif splits_file.exists():
        with open(splits_file, "r") as f:
            splits = json.load(f)
        logger.info(
            f"Splits loaded from {splits_file} | "
            f"train={len(splits.get('train', []))} | "
            f"val={len(splits.get('val', []))} | "
            f"test={len(splits.get('test', []))}"
        )
    else:
        logger.error(
            f"Splits file not found: {splits_file}\n"
            "  → Run feature extraction notebook first, or use --smoke-test."
        )
        return 1

    # ---- Ablation run list -------------------------------------------------
    ablation_runs: List[dict] = cfg.get("ablations", {}).get("runs", [])
    if not ablation_runs:
        logger.error("No ablation runs defined in config (ablations.runs is empty).")
        return 1

    logger.info(f"Total ablation runs: {len(ablation_runs)}")
    for i, r in enumerate(ablation_runs, 1):
        logger.info(f"  {i}. {r['name']:<30} modalities={r['modalities']}  fusion={r.get('fusion_mode','pgin')}")

    # ---- Execute all runs --------------------------------------------------
    all_metrics: List[Dict] = []
    is_tty = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()

    ablation_pbar = tqdm(
        ablation_runs,
        desc="Ablation Matrix (8 runs)",
        unit="run",
        leave=True,
        mininterval=1.0 if is_tty else 2.0,
        miniters=1,
        dynamic_ncols=True,
    )

    for run_cfg in ablation_pbar:
        set_seed(seed)  # re-seed before every run for identical init
        ablation_pbar.set_description(f"Ablation Matrix [{run_cfg['name']}]", refresh=False)
        try:
            metrics = run_single_ablation(
                run_cfg     = run_cfg,
                cfg         = cfg,
                splits      = splits,
                cache_dir   = cache_dir,
                results_dir = results_dir,
                device      = device,
                smoke_test  = args.smoke_test,
            )
            all_metrics.append(metrics)
            f1_val = metrics.get("f1")
            f1_str = f"{f1_val:.4f}" if isinstance(f1_val, (int, float)) else str(f1_val)
            ablation_pbar.set_postfix({"run": run_cfg["name"], "f1": f1_str}, refresh=False)
        except Exception as exc:
            logger.exception(f"Run '{run_cfg['name']}' FAILED: {exc}")
            # Record failure entry so the summary still has all rows
            all_metrics.append({
                "run_name":    run_cfg["name"],
                "modalities":  run_cfg.get("modalities", []),
                "fusion_mode": run_cfg.get("fusion_mode", "pgin"),
                "accuracy":    None,
                "precision":   None,
                "recall":      None,
                "f1":          None,
                "error":       str(exc),
            })
            ablation_pbar.set_postfix({"run": run_cfg["name"], "status": "FAILED"}, refresh=False)

    # ---- Write consolidated outputs ----------------------------------------
    summary_rows = _build_summary_rows(all_metrics)
    _write_csv(summary_rows,  results_dir / "ablation_summary.csv")
    _write_json(all_metrics,  results_dir / "ablation_summary.json")

    # ---- Markdown comparison table -----------------------------------------
    md_path = results_dir / "ablation_summary.md"
    _write_markdown_table(summary_rows, md_path)

    # ---- Print to stdout ----------------------------------------------------
    _print_summary_table(summary_rows)
    logger.info("All ablation runs complete.")
    return 0


def _write_markdown_table(rows: List[Dict], path: Path) -> None:
    """Write a GitHub-flavoured markdown comparison table."""
    if not rows:
        return
    headers  = ["Run", "Modalities", "Fusion", "Accuracy", "Precision", "Recall", "F1",
                 "Best-Val-F1", "Best-Epoch", "Time (s)"]
    row_keys = ["run_name", "modalities", "fusion_mode", "accuracy", "precision",
                "recall", "f1", "best_val_f1", "best_epoch", "elapsed_s"]

    def _fmt(val, key):
        if val is None or val == "":
            return "—"
        if key in {"accuracy", "precision", "recall", "f1", "best_val_f1"}:
            try:
                return f"{float(val):.4f}"
            except (TypeError, ValueError):
                return str(val)
        if isinstance(val, list):
            return ", ".join(str(v) for v in val)
        return str(val)

    with open(path, "w", encoding="utf-8") as f:
        f.write("# SarcasTone — Ablation Study Results\n\n")
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("|" + "|".join([":---"] + ["---:"] * (len(headers) - 1)) + "|\n")
        for row in rows:
            cells = [_fmt(row.get(k), k) for k in row_keys]
            f.write("| " + " | ".join(cells) + " |\n")
        f.write("\n")
        f.write("> Gate weight columns: `gate_text`, `gate_speech`, `gate_inc`, `gate_video`\n")
        f.write("> are written to `ablation_summary.csv` and `ablation_summary.json`.\n")
    logger.info(f"Markdown table written → {path}")


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sys.exit(main())
