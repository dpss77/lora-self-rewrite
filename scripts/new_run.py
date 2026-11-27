#!/usr/bin/env python3
"""Allocate a run id and write the pre-run manifest + directory tree."""

from __future__ import annotations

import argparse
import json
import shutil
import sys

from _common import ROOT, init_run_tree, load_config, utc_now, write_text

from lora_self_rewrite.constants import MODEL_REVISION
from lora_self_rewrite.datasets import DATASET_SOURCE_REVISIONS
from lora_self_rewrite.manifest import make_run_id, validate_manifest
from lora_self_rewrite.run_config import validate_run_config


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, help="experiment TOML under configs/experiments/")
    p.add_argument("--runs-dir", default=str(ROOT / "runs"))
    p.add_argument("--gpu", default="H200")
    p.add_argument("--code-sha", required=True, help="git commit sha of the code being run")
    p.add_argument("--file-sha256", default="unknown")
    p.add_argument("--model-revision", default=None)
    p.add_argument("--force", action="store_true", help="overwrite an existing run dir")
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    validate_run_config(cfg)
    experiment_id = cfg["experiment_id"]
    run_id = make_run_id(experiment_id, args.gpu, args.code_sha)
    run_dir = init_run_tree(f"{args.runs_dir}/{run_id}")
    manifest_path = run_dir / "00_RUN_MANIFEST.json"
    if manifest_path.exists() and not args.force:
        print(f"refusing to overwrite {manifest_path} (use --force)", file=sys.stderr)
        return 2

    protocol = cfg.get("protocol", {})
    splits = cfg.get("splits", {})
    model_revision = args.model_revision or cfg.get("model_revision") or MODEL_REVISION
    if str(model_revision).strip().lower() in {"unknown", "none", "null", ""}:
        print("manifest requires a pinned model revision", file=sys.stderr)
        return 1
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "experiment_id": experiment_id,
        "parent_run_id": None,
        "resumed_from": None,
        "status": "planned",
        "created_at_utc": utc_now(),
        "finished_at_utc": None,
        "code": {"repo_sha": args.code_sha, "file_sha256": args.file_sha256},
        "model": {
            "id": cfg.get("model_id", "Qwen/Qwen3.5-27B"),
            "revision": model_revision,
            "dtype": "bfloat16",
        },
        "dataset": {
            "name": cfg.get("dataset_name", "unknown"),
            "splits": splits,
            "source_revisions": DATASET_SOURCE_REVISIONS,
        },
        "config": cfg,
        "protocol": {
            "max_tokens": protocol.get("max_tokens"),
            "passes_budget": protocol.get("passes_budget"),
            "code_budget": protocol.get("code_budget"),
            "turns_budget": protocol.get("turns_budget"),
            "wall_limit_hours": protocol.get("wall_limit_hours"),
        },
        "hardware": {"gpu_requested": args.gpu, "gpu_actual": None, "cache_tag": None},
        "results": {
            "passes_used": 0,
            "code_runs_used": 0,
            "decision": None,
            "submitted_lora": False,
            "validation": None,
            "heldout": None,
        },
        "artifacts": [],
        "analysis": {
            "summary": "50_ANALYSIS/SUMMARY.md",
            "detail": "50_ANALYSIS/ANALYSIS.md",
        },
        "checksums": "80_FINAL/CHECKSUMS.sha256",
    }
    errors = validate_manifest(manifest)
    if errors:
        print("invalid manifest: " + "; ".join(errors), file=sys.stderr)
        return 1

    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    shutil.copy2(args.config, run_dir / "01_CONFIG.toml")
    write_text(
        run_dir / "02_PROMPT.md",
        "# Task prompt\n\n"
        "This run uses the Table-5 prompt suffix: 'Please reason step by step, and put "
        "your final answer within \\\\boxed{}.'\n",
    )
    print(run_id)
    print(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
