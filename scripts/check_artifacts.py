#!/usr/bin/env python3
"""Validate run manifests, required files, and checksums. Exits non-zero on failure."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from _common import ROOT

from lora_self_rewrite.manifest import required_run_files, sha256_file, validate_manifest


def check_run(run_dir: Path) -> list[str]:
    errors: list[str] = []
    manifest_path = run_dir / "00_RUN_MANIFEST.json"
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        return [f"{run_dir}: unreadable manifest ({e})"]

    errors.extend(f"{run_dir}: {e}" for e in validate_manifest(manifest))
    errors.extend(f"{run_dir}: missing {rel}" for rel in required_run_files(str(run_dir)))

    checksums_path = run_dir / "80_FINAL" / "CHECKSUMS.sha256"
    if checksums_path.exists():
        listed: set[str] = set()
        for line in checksums_path.read_text().splitlines():
            if not line.strip():
                continue
            digest, _, rel = line.partition("  ")
            listed.add(rel)
            target = run_dir / rel
            if not target.exists():
                errors.append(f"{run_dir}: checksum target missing: {rel}")
            elif sha256_file(str(target)) != digest:
                errors.append(f"{run_dir}: checksum mismatch: {rel}")
        # An incomplete manifest must not pass: every file on disk (except the
        # checksum file itself) must be listed.
        for path in sorted(run_dir.rglob("*")):
            if not path.is_file():
                continue
            rel = str(path.relative_to(run_dir))
            if rel == "80_FINAL/CHECKSUMS.sha256" or rel in listed:
                continue
            errors.append(f"{run_dir}: file not listed in checksums: {rel}")
    else:
        errors.append(f"{run_dir}: missing 80_FINAL/CHECKSUMS.sha256")
    return errors


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", default=None)
    p.add_argument("--runs-dir", default=str(ROOT / "runs"))
    args = p.parse_args(argv)

    if args.run_dir:
        run_dirs = [Path(args.run_dir)]
    else:
        runs = Path(args.runs_dir)
        run_dirs = sorted(p for p in runs.glob("*") if p.is_dir()) if runs.exists() else []

    if not run_dirs:
        print("no runs to check")
        return 0

    all_errors: list[str] = []
    for run_dir in run_dirs:
        errors = check_run(run_dir)
        all_errors.extend(errors)
        print(f"[{'FAIL' if errors else 'OK'}] {run_dir}")

    for err in all_errors:
        print(f"  - {err}", file=sys.stderr)
    return 1 if all_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
