"""
tests/test_notebooks_and_smoke.py
=================================
Validates Google Colab notebooks (metadata, schema, execution cells)
and smoke test functionality.
"""

import json
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_colab_notebooks_exist():
    """Verify both Colab notebooks exist."""
    nb1_path = REPO_ROOT / "notebooks" / "01_extract_and_cache_colab.ipynb"
    nb2_path = REPO_ROOT / "notebooks" / "02_train_and_evaluate_colab.ipynb"

    assert nb1_path.is_file(), f"Missing {nb1_path}"
    assert nb2_path.is_file(), f"Missing {nb2_path}"


def test_notebook_01_schema_and_contents():
    """Verify notebook 01 structure, cells, and Colab T4 GPU metadata."""
    nb1_path = REPO_ROOT / "notebooks" / "01_extract_and_cache_colab.ipynb"
    with open(nb1_path, "r", encoding="utf-8") as f:
        nb1 = json.load(f)

    # Check top-level notebook metadata
    assert nb1.get("nbformat") == 4
    metadata = nb1.get("metadata", {})
    assert metadata.get("accelerator") == "GPU"
    assert metadata.get("colab", {}).get("gpuType") == "T4"

    # Check cells
    cells = nb1.get("cells", [])
    assert len(cells) >= 10

    all_source = "".join("".join(c.get("source", [])) for c in cells)

    # Key requirements checks
    assert "drive.mount" in all_source
    assert "torch.cuda.is_available" in all_source
    assert "requirements.txt" in all_source
    assert "create_or_load_splits" in all_source
    assert "TextFeatureExtractor" in all_source
    assert "ProsodyAcousticExtractor" in all_source
    assert "VideoFeatureExtractor" in all_source
    assert "FeatureCacheManager" in all_source
    assert "from tqdm.auto import tqdm" in all_source
    assert "unit=\"sample\"" in all_source
    assert "Tanmay-Shrivastav/SarcasTone.git" in all_source
    assert "git pull" in all_source


def test_notebook_02_schema_and_contents():
    """Verify notebook 02 structure, cells, all 8 ablation runs, and comparison table."""
    nb2_path = REPO_ROOT / "notebooks" / "02_train_and_evaluate_colab.ipynb"
    with open(nb2_path, "r", encoding="utf-8") as f:
        nb2 = json.load(f)

    # Check top-level notebook metadata
    assert nb2.get("nbformat") == 4
    metadata = nb2.get("metadata", {})
    assert metadata.get("accelerator") == "GPU"
    assert metadata.get("colab", {}).get("gpuType") == "T4"

    # Check cells
    cells = nb2.get("cells", [])
    assert len(cells) >= 10

    all_source = "".join("".join(c.get("source", [])) for c in cells)

    # Key requirements checks
    assert "drive.mount" in all_source
    assert "torch.cuda.is_available" in all_source
    assert "verify_splits" in all_source
    assert "run_ablations" in all_source
    assert "ablation_summary.csv" in all_source
    assert "Tanmay-Shrivastav/SarcasTone.git" in all_source
    assert "git pull" in all_source

    # Check that all 8 ablation runs are documented
    expected_runs = [
        "text_only",
        "speech_only",
        "video_only",
        "text_speech_concat",
        "text_speech_pgin",
        "text_video",
        "speech_video",
        "full_multimodal_pgin",
    ]
    for r in expected_runs:
        assert r in all_source, f"Run '{r}' not documented in notebook 02"


def test_smoke_test_script_execution():
    """Test executing scripts/smoke_test.py as a script."""
    import subprocess
    import sys

    script_path = REPO_ROOT / "scripts" / "smoke_test.py"
    assert script_path.is_file(), f"Missing {script_path}"

    result = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    assert result.returncode == 0, f"smoke_test failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    combined_output = result.stdout + result.stderr
    assert "SARCASTONE LIGHTWEIGHT SMOKE TEST SUITE" in combined_output


def test_tqdm_progress_bars_configuration():
    """Verify that Trainer, run_ablations, and cache_manager configure clean tqdm progress bars with throttling."""
    trainer_file = REPO_ROOT / "src" / "training" / "trainer.py"
    ablations_file = REPO_ROOT / "src" / "experiments" / "run_ablations.py"
    cache_mgr_file = REPO_ROOT / "src" / "features" / "cache_manager.py"

    trainer_src = trainer_file.read_text(encoding="utf-8")
    ablations_src = ablations_file.read_text(encoding="utf-8")
    cache_mgr_src = cache_mgr_file.read_text(encoding="utf-8")

    # Trainer progress bars (epochs and batches) with throttling and refresh=False
    assert "from tqdm import tqdm" in trainer_src
    assert "from tqdm.auto import tqdm" in trainer_src
    assert "batch_pbar" in trainer_src
    assert "epoch_pbar" in trainer_src
    assert "eval_pbar" in trainer_src
    assert "unit=\"epoch\"" in trainer_src
    assert "unit=\"batch\"" in trainer_src
    assert "mininterval=" in trainer_src
    assert "refresh=False" in trainer_src
    assert "is_tty" in trainer_src

    # run_ablations progress bar (8-run matrix) with throttling and refresh=False
    assert "from tqdm import tqdm" in ablations_src
    assert "from tqdm.auto import tqdm" in ablations_src
    assert "ablation_pbar" in ablations_src
    assert "unit=\"run\"" in ablations_src
    assert "mininterval=" in ablations_src
    assert "refresh=False" in ablations_src
    assert "is_tty" in ablations_src

    # cache_manager progress bar option
    assert "show_progress" in cache_mgr_src
    assert "Checking cache status" in cache_mgr_src
