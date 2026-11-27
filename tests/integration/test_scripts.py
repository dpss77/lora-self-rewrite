import json
import subprocess
import sys
from pathlib import Path

import pytest

from lora_self_rewrite.constants import MODEL_REVISION

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parents[2]


def _run(*args):
    return subprocess.run(
        [sys.executable, *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )


def test_run_automation_end_to_end(tmp_path):
    runs = tmp_path / "runs"
    out = _run(
        "scripts/new_run.py",
        "--config",
        "configs/experiments/S02-DS-SEED00.toml",
        "--runs-dir",
        str(runs),
        "--code-sha",
        "fe71f74",
        "--file-sha256",
        "fe71f74b0c",
    )
    run_id = out.stdout.splitlines()[0]
    run_dir = runs / run_id
    assert (run_dir / "00_RUN_MANIFEST.json").exists()

    (run_dir / "20_METRICS").mkdir(exist_ok=True)
    (run_dir / "20_METRICS" / "METRICS.json").write_text(json.dumps({"turns": 1}))

    # Summarize while the run is still non-terminal (writes SUMMARY/TABLES/ANALYSIS
    # and seals checksums); then finalize to a terminal status.
    _run("scripts/summarize_run.py", "--run-dir", str(run_dir))
    _run(
        "scripts/finalize_run.py",
        "--run-dir",
        str(run_dir),
        "--status",
        "withdrawn",
        "--decision",
        "WITHDRAW",
    )
    _run("scripts/index_runs.py", "--runs-dir", str(runs), "--out", str(tmp_path / "INDEX.md"))
    _run("scripts/check_artifacts.py", "--runs-dir", str(runs))

    manifest = json.loads((run_dir / "00_RUN_MANIFEST.json").read_text())
    assert manifest["model"]["revision"] == MODEL_REVISION
    assert manifest["dataset"]["source_revisions"]
    assert manifest["config"]["protocol"]["passes_budget"] == 20
    assert manifest["status"] == "withdrawn"
    assert manifest["results"]["decision"] == "WITHDRAW"
    assert (run_dir / "80_FINAL" / "CHECKSUMS.sha256").exists()
    assert (tmp_path / "INDEX.md").exists()


def test_summarize_uses_real_heldout_schema_and_stored_stats(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "00_RUN_MANIFEST.json").write_text(
        json.dumps({"run_id": "run_heldout", "status": "running"})
    )
    (run / "30_VALIDATION").mkdir()
    (run / "40_HELDOUT").mkdir()

    base_rows = [
        {"id": 1, "sample_idx": 0, "correct": False, "cut": 0, "loop": 0},
        {"id": 2, "sample_idx": 0, "correct": False, "cut": 0, "loop": 0},
    ]
    edited_rows = [
        {"id": 1, "sample_idx": 0, "correct": False, "cut": 0, "loop": 0},
        {"id": 2, "sample_idx": 0, "correct": False, "cut": 0, "loop": 0},
    ]
    stored_stats = {
        "n_problems": 49,
        "n_answers": 392,
        "base_acc": 0.25,
        "edited_acc": 0.75,
        "delta": 0.5,
        "se": 0.0123,
        "ci_low": 0.4,
        "ci_high": 0.6,
        "signflip_p_one_sided": 0.0001,
        "wrong_to_right": 10,
        "right_to_wrong": 2,
    }
    adapter_sha = "ab" * 32
    heldout = {
        "schema_version": 1,
        "cache_tag": "a370202bd599",
        "base_cache_signature": "sig0123456789",
        "base": {"rows": base_rows},
        "edited": {"rows": edited_rows, "adapter_sha256": adapter_sha},
        "stats": stored_stats,
    }
    (run / "40_HELDOUT" / "HELDOUT.json").write_text(json.dumps(heldout))

    validation = {
        "base_rows": [
            {"id": 1, "sample_idx": 0, "correct": False},
            {"id": 2, "sample_idx": 0, "correct": True},
        ],
        "rows": [
            {"id": 1, "sample_idx": 0, "correct": True},
            {"id": 2, "sample_idx": 0, "correct": False},
        ],
    }
    (run / "30_VALIDATION" / "VALIDATION.json").write_text(json.dumps(validation))

    _run("scripts/summarize_run.py", "--run-dir", str(run))
    text = (run / "50_ANALYSIS" / "TABLES.md").read_text()

    held = text.split("### Held-out evaluation", 1)[1]
    # Authoritative stats used unchanged (rows would give n_answers=2, delta=0).
    assert "| n answers | 392 |" in held
    assert "| delta | +0.5000 |" in held
    assert "| base acc | 0.2500 |" in held
    assert "```json" not in held
    # Provenance reaches the table.
    assert "a370202bd599" in held
    assert "sig0123456789" in held
    assert adapter_sha in held

    reserved = text.split("### Reserved validation", 1)[1].split("### Held-out", 1)[0]
    assert "```json" not in reserved  # legacy VALIDATION schema still renders
    assert "| base acc | 0.5000 |" in reserved


def _run_fail(*args):
    return subprocess.run(
        [sys.executable, *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )


def _prepare_terminal_run(tmp_path: Path, code_sha: str) -> Path:
    runs = tmp_path / "runs"
    out = _run(
        "scripts/new_run.py",
        "--config",
        "configs/experiments/S02-DS-SEED00.toml",
        "--runs-dir",
        str(runs),
        "--code-sha",
        code_sha,
        "--file-sha256",
        "fe71f74b0c",
    )
    run_dir = runs / out.stdout.splitlines()[0]
    (run_dir / "20_METRICS").mkdir(exist_ok=True)
    (run_dir / "20_METRICS" / "METRICS.json").write_text(json.dumps({"turns": 1}))
    _run("scripts/summarize_run.py", "--run-dir", str(run_dir))
    _run(
        "scripts/finalize_run.py",
        "--run-dir",
        str(run_dir),
        "--status",
        "withdrawn",
        "--decision",
        "WITHDRAW",
    )
    return run_dir


def test_summarize_refuses_checksum_reseal_for_terminal_run(tmp_path):
    run_dir = _prepare_terminal_run(tmp_path, "aaaaaaa1")
    sums = run_dir / "80_FINAL" / "CHECKSUMS.sha256"
    before = sums.read_text()

    # Post-finalization mutation.
    (run_dir / "20_METRICS" / "METRICS.json").write_text(json.dumps({"turns": 2}))

    refused = _run_fail("scripts/summarize_run.py", "--run-dir", str(run_dir))
    assert refused.returncode != 0
    assert "refusing to re-seal" in refused.stderr
    assert sums.read_text() == before  # not silently re-sealed
    assert (run_dir / "50_ANALYSIS" / "SUMMARY.md").exists()

    forced = _run_fail("scripts/summarize_run.py", "--run-dir", str(run_dir), "--force")
    assert forced.returncode == 0
    assert sums.read_text() != before


def test_finalize_terminal_missing_required_files_fails(tmp_path):
    runs = tmp_path / "runs"
    out = _run(
        "scripts/new_run.py",
        "--config",
        "configs/experiments/S02-DS-SEED00.toml",
        "--runs-dir",
        str(runs),
        "--code-sha",
        "aaaaaaa2",
        "--file-sha256",
        "fe71f74b0c",
    )
    run_dir = runs / out.stdout.splitlines()[0]

    result = _run_fail(
        "scripts/finalize_run.py",
        "--run-dir",
        str(run_dir),
        "--status",
        "withdrawn",
        "--decision",
        "WITHDRAW",
    )
    assert result.returncode != 0
    assert "missing required files" in result.stderr


def test_check_artifacts_detects_unlisted_file(tmp_path):
    run_dir = _prepare_terminal_run(tmp_path, "aaaaaaa3")
    logs = run_dir / "60_LOGS"
    logs.mkdir(exist_ok=True)
    (logs / "EXTRA.log").write_text("not in checksums")

    result = _run_fail("scripts/check_artifacts.py", "--run-dir", str(run_dir))
    assert result.returncode != 0
    assert "not listed" in result.stderr
    assert "EXTRA.log" in result.stderr
