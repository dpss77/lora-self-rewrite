"""Shared helpers for the run-automation scripts (repo-local, no install needed)."""

from __future__ import annotations

import os
import sys
import tomllib
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

RUN_TREE = (
    "03_DATASET",
    "10_AGENT/PASSES",
    "20_METRICS",
    "30_VALIDATION",
    "40_HELDOUT",
    "50_ANALYSIS",
    "60_LOGS",
    "70_ARTIFACTS/ADAPTERS",
    "80_FINAL",
)


def utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_config(path: str | Path) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def init_run_tree(run_dir: str | Path) -> Path:
    run_dir = Path(run_dir)
    for rel in RUN_TREE:
        (run_dir / rel).mkdir(parents=True, exist_ok=True)
    return run_dir


def write_text(path: str | Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    tmp.write_text(text)
    os.replace(tmp, path)
