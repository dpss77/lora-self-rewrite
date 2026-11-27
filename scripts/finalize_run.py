#!/usr/bin/env python3
"""Atomically finalize a run: status, decision, checksums, and 80_FINAL/STATUS.json."""

from __future__ import annotations

import argparse
import json
import sys

from _common import utc_now

from lora_self_rewrite.constants import STATUSES, TERMINAL_STATUSES
from lora_self_rewrite.manifest import (
    atomic_write_json,
    required_run_files,
    validate_manifest,
    write_checksums,
)

TRANSITIONS = {
    "planned": {"running", "preempted", "failed", "withdrawn"},
    "running": {"preempted", "failed", "withdrawn", "submitted"},
    "preempted": {"running", "failed", "withdrawn", "submitted"},
    "submitted": {"validated-null", "validated-positive", "archived"},
    "withdrawn": {"archived"},
    "validated-null": {"archived"},
    "validated-positive": {"archived"},
    "failed": {"archived"},
    "archived": set(),
}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--status", required=True, choices=STATUSES)
    p.add_argument("--decision", default=None)
    p.add_argument("--submitted-lora", action="store_true")
    p.add_argument("--validation", default=None, help="path to VALIDATION.json")
    p.add_argument("--heldout", default=None, help="path to HELDOUT.json")
    args = p.parse_args(argv)

    manifest_path = f"{args.run_dir}/00_RUN_MANIFEST.json"
    try:
        with open(manifest_path) as f:
            manifest = json.load(f)
    except OSError as e:
        print(f"cannot read manifest: {e}", file=sys.stderr)
        return 2

    current = manifest.get("status")
    if current != args.status and args.status not in TRANSITIONS.get(current, set()):
        print(
            f"illegal transition {current!r} -> {args.status!r}",
            file=sys.stderr,
        )
        return 2

    manifest["status"] = args.status
    manifest.setdefault("results", {})
    if args.decision is not None:
        manifest["results"]["decision"] = args.decision
    manifest["results"]["submitted_lora"] = bool(args.submitted_lora)
    if args.validation:
        manifest["results"]["validation"] = args.validation
    if args.heldout:
        manifest["results"]["heldout"] = args.heldout
    if args.status in {
        "preempted",
        "failed",
        "withdrawn",
        "submitted",
        "validated-null",
        "validated-positive",
        "archived",
    }:
        manifest["finished_at_utc"] = manifest.get("finished_at_utc") or utc_now()

    errors = validate_manifest(manifest)
    if errors:
        print("invalid manifest: " + "; ".join(errors), file=sys.stderr)
        return 1

    atomic_write_json(manifest_path, manifest)
    status = {
        "run_id": manifest["run_id"],
        "status": manifest["status"],
        "decision": manifest["results"].get("decision"),
        "submitted_lora": manifest["results"].get("submitted_lora"),
        "finished_at_utc": manifest.get("finished_at_utc"),
        "checksums": manifest.get("checksums"),
    }
    atomic_write_json(f"{args.run_dir}/80_FINAL/STATUS.json", status)
    write_checksums(args.run_dir)

    missing = required_run_files(args.run_dir)
    if missing:
        if manifest["status"] in TERMINAL_STATUSES:
            print(f"error: missing required files: {missing}", file=sys.stderr)
            return 1
        print(f"warning: missing required files: {missing}", file=sys.stderr)
    print(json.dumps(status, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
