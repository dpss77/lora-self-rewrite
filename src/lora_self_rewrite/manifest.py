"""Run manifest schema, naming, and atomic finalization helpers.

The manifest is the single source of truth for a run (docs/05-standards/
STD-0001-output-contract.md). Keep this module pure and dependency-free so scripts and
tests can validate run directories without Modal/HF/torch.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import UTC, datetime

from .constants import STATUSES, TERMINAL_STATUSES

__all__ = [
    "RUN_ID_RE",
    "REQUIRED_RUN_FILES",
    "make_experiment_id",
    "make_run_id",
    "validate_manifest",
    "atomic_write_json",
    "sha256_file",
    "write_checksums",
    "required_run_files",
]

RUN_ID_RE = re.compile(
    r"^(?P<ts>\d{8}T\d{6}Z)_(?P<exp>[A-Z0-9-]+)_(?P<gpu>[A-Za-z0-9]+)_(?P<sha>[0-9a-f]{7,40})$"
)

REQUIRED_RUN_FILES = (
    "00_RUN_MANIFEST.json",
    "01_CONFIG.toml",
    "02_PROMPT.md",
    "20_METRICS/METRICS.json",
    "50_ANALYSIS/SUMMARY.md",
    "50_ANALYSIS/ANALYSIS.md",
    "80_FINAL/STATUS.json",
    "80_FINAL/CHECKSUMS.sha256",
)


def make_experiment_id(stage: int | str, dataset: str, arm: str) -> str:
    """``S<stage>-<DATASET>-<ARM>`` in upper-kebab form."""
    return f"S{stage}-{dataset.upper()}-{arm.upper()}"


def make_run_id(
    experiment_id: str,
    gpu: str,
    code_sha7: str,
    when: datetime | None = None,
) -> str:
    """``<UTC timestamp>_<experiment-id>_<gpu>_<code-sha7>``."""
    when = when or datetime.now(UTC)
    ts = when.strftime("%Y%m%dT%H%M%SZ")
    sha = code_sha7[:7]
    run_id = f"{ts}_{experiment_id}_{gpu}_{sha}"
    if not RUN_ID_RE.match(run_id):
        raise ValueError(f"Invalid run id: {run_id!r}")
    return run_id


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def validate_manifest(m: dict) -> list[str]:
    """Return a list of human-readable schema violations (empty = valid)."""
    errors: list[str] = []
    if not isinstance(m, dict):
        return ["manifest must be a JSON object"]

    required_top = {
        "schema_version",
        "run_id",
        "experiment_id",
        "status",
        "created_at_utc",
        "code",
        "model",
        "dataset",
        "protocol",
        "hardware",
        "results",
    }
    missing = required_top - set(m)
    if missing:
        errors.append(f"missing top-level fields: {sorted(missing)}")
    if not isinstance(m.get("schema_version"), int):
        errors.append("schema_version must be an integer")

    run_id = m.get("run_id")
    if not isinstance(run_id, str) or not RUN_ID_RE.match(run_id):
        errors.append(f"run_id does not match naming contract: {run_id!r}")

    if m.get("status") not in STATUSES:
        errors.append(f"unknown status {m.get('status')!r}; allowed={STATUSES}")

    code = m.get("code")
    if not isinstance(code, dict) or not {"repo_sha", "file_sha256"} <= set(code):
        errors.append("code must contain repo_sha and file_sha256")

    model = m.get("model")
    if not isinstance(model, dict) or not {"id", "revision", "dtype"} <= set(model):
        errors.append("model must contain id, revision, dtype")
    else:
        revision = str(model.get("revision") or "").strip().lower()
        if revision in {"", "unknown", "none", "null"}:
            errors.append("model.revision must be pinned to a non-unknown revision")

    dataset = m.get("dataset")
    if not isinstance(dataset, dict) or not {"name", "splits"} <= set(dataset):
        errors.append("dataset must contain name and splits")
    else:
        splits = dataset.get("splits")
        if not isinstance(splits, dict) or not all(isinstance(v, int) for v in splits.values()):
            errors.append("dataset.splits must be an object of integer counts")

    protocol = m.get("protocol")
    if not isinstance(protocol, dict):
        errors.append("protocol must be an object")
    else:
        for key in ("max_tokens", "passes_budget", "code_budget", "turns_budget"):
            if not isinstance(protocol.get(key), int):
                errors.append(f"protocol.{key} must be an integer")
        if not _is_number(protocol.get("wall_limit_hours")):
            errors.append("protocol.wall_limit_hours must be a number")

    hardware = m.get("hardware")
    if not isinstance(hardware, dict) or "gpu_requested" not in hardware:
        errors.append("hardware must contain gpu_requested")

    results = m.get("results")
    if not isinstance(results, dict):
        errors.append("results must be an object")
    else:
        for key in ("decision", "submitted_lora"):
            if key not in results:
                errors.append(f"results.{key} is required")

    if m.get("status") in TERMINAL_STATUSES and not m.get("finished_at_utc"):
        errors.append("terminal status requires finished_at_utc")

    return errors


def atomic_write_json(path: str, obj: dict) -> None:
    """Write JSON atomically (tmp + os.replace)."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def required_run_files(run_dir: str) -> list[str]:
    """Return required run files that are missing."""
    return [rel for rel in REQUIRED_RUN_FILES if not os.path.exists(os.path.join(run_dir, rel))]


def write_checksums(run_dir: str, output: str | None = None) -> str:
    """Write a sorted ``sha256  relpath`` manifest for all files under ``run_dir``.

    ``80_FINAL/CHECKSUMS.sha256`` is excluded from its own listing.
    """
    output = output or os.path.join(run_dir, "80_FINAL", "CHECKSUMS.sha256")
    rows = []
    for root, _dirs, files in os.walk(run_dir):
        for name in files:
            path = os.path.join(root, name)
            rel = os.path.relpath(path, run_dir)
            if os.path.abspath(path) == os.path.abspath(output):
                continue
            rows.append(f"{sha256_file(path)}  {rel}")
    rows.sort()
    os.makedirs(os.path.dirname(output), exist_ok=True)
    tmp = f"{output}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        f.write("\n".join(rows) + ("\n" if rows else ""))
    os.replace(tmp, output)
    return output
