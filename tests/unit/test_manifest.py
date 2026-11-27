from datetime import UTC, datetime

import pytest

from lora_self_rewrite.manifest import (
    make_experiment_id,
    make_run_id,
    validate_manifest,
    write_checksums,
)

pytestmark = pytest.mark.unit


def _valid_manifest():
    return {
        "schema_version": 1,
        "run_id": "20261126T061343Z_S02-DS-SEED00_H200_fe71f74",
        "experiment_id": "S02-DS-SEED00",
        "status": "withdrawn",
        "created_at_utc": "2026-11-26T06:13:43Z",
        "finished_at_utc": "2026-11-26T09:14:12Z",
        "code": {"repo_sha": "24200a1", "file_sha256": "abc"},
        "model": {"id": "Qwen/Qwen3.5-27B", "revision": "rev", "dtype": "bfloat16"},
        "dataset": {
            "name": "deepscaler",
            "splits": {"practice": 3000, "reserved": 300, "heldout": 1000},
        },
        "protocol": {
            "max_tokens": 4096,
            "passes_budget": 20,
            "code_budget": 200,
            "turns_budget": 160,
            "wall_limit_hours": 3.0,
        },
        "hardware": {"gpu_requested": "H200"},
        "results": {"decision": "WITHDRAW", "submitted_lora": False},
    }


def test_make_ids():
    assert make_experiment_id(2, "ds", "seed00") == "S2-DS-SEED00"
    dt = datetime(2026, 11, 26, 6, 13, 43, tzinfo=UTC)
    assert (
        make_run_id("S02-DS-SEED00", "H200", "fe71f74b0c", dt)
        == "20261126T061343Z_S02-DS-SEED00_H200_fe71f74"
    )


def test_valid_manifest_passes():
    assert validate_manifest(_valid_manifest()) == []


def test_unknown_status_rejected():
    m = _valid_manifest()
    m["status"] = "done"
    assert any("unknown status" in e for e in validate_manifest(m))


def test_terminal_requires_finished_at():
    m = _valid_manifest()
    m.pop("finished_at_utc")
    assert any("finished_at_utc" in e for e in validate_manifest(m))


def test_bad_run_id_rejected():
    m = _valid_manifest()
    m["run_id"] = "not-a-run"
    assert any("run_id" in e for e in validate_manifest(m))


def test_write_checksums_omits_self(tmp_path):
    (tmp_path / "a.txt").write_text("hello")
    out = write_checksums(str(tmp_path))
    body = open(out).read()
    assert "a.txt" in body
    assert "CHECKSUMS.sha256" not in body
    # deterministic
    assert body == open(write_checksums(str(tmp_path))).read()


def test_unknown_model_revision_rejected():
    m = _valid_manifest()
    m["model"]["revision"] = "unknown"
    assert any("model.revision" in e for e in validate_manifest(m))
