#!/usr/bin/env python3
"""Generate 50_ANALYSIS/SUMMARY.md and TABLES.md from a run's JSON artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from _common import write_text

from lora_self_rewrite.constants import TERMINAL_STATUSES
from lora_self_rewrite.manifest import write_checksums
from lora_self_rewrite.stats import compute_paired_stats


def _read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _fmt(value, spec: str) -> str:
    try:
        return format(value, spec)
    except (TypeError, ValueError):
        return "n/a"


def _resolve_stats(payload: dict) -> dict | None:
    """Prefer the authoritative ``stats`` block; recompute only if it is absent."""
    stats = payload.get("stats")
    if isinstance(stats, dict) and stats:
        return stats
    edited = payload.get("edited")
    base = payload.get("base")
    if isinstance(edited, dict) and isinstance(base, dict):
        rows = edited.get("rows")
        base_rows = base.get("rows")
    else:
        rows = payload.get("rows")
        base_rows = payload.get("base_rows")
    if rows and base_rows:
        return compute_paired_stats(base_rows, rows, signflip_resamples=100_000)
    return None


def _stats_block(label: str, payload: dict | None) -> str:
    if not payload:
        return f"### {label}\n\n_not available_\n"
    stats = _resolve_stats(payload)
    if not stats:
        return f"### {label}\n\n```json\n{json.dumps(payload, indent=2)[:2000]}\n```\n"
    lines = [
        f"### {label}",
        "",
        "| metric | value |",
        "| :--- | ---: |",
        f"| n problems | {_fmt(stats.get('n_problems'), '')} |",
        f"| n answers | {_fmt(stats.get('n_answers'), '')} |",
        f"| base acc | {_fmt(stats.get('base_acc'), '.4f')} |",
        f"| edited acc | {_fmt(stats.get('edited_acc'), '.4f')} |",
        f"| delta | {_fmt(stats.get('delta'), '+.4f')} |",
        f"| SE | {_fmt(stats.get('se'), '.4f')} |",
        f"| 95% CI | [{_fmt(stats.get('ci_low'), '+.4f')}, {_fmt(stats.get('ci_high'), '+.4f')}] |",
        f"| one-sided sign-flip p | {_fmt(stats.get('signflip_p_one_sided'), '.4f')} |",
        "| wrong->right / right->wrong | "
        f"{_fmt(stats.get('wrong_to_right'), '')} / "
        f"{_fmt(stats.get('right_to_wrong'), '')} |",
    ]
    cache_tag = payload.get("cache_tag")
    if cache_tag:
        lines.append(f"| cache tag | `{cache_tag}` |")
    signature = payload.get("base_cache_signature")
    if signature:
        lines.append(f"| base cache signature | `{signature}` |")
    edited = payload.get("edited")
    adapter = edited.get("adapter_sha256") if isinstance(edited, dict) else None
    if adapter is None:
        adapter = payload.get("adapter_sha256")
    if adapter:
        lines.append(f"| adapter sha256 | `{adapter}` |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True)
    p.add_argument(
        "--force",
        action="store_true",
        help="re-seal CHECKSUMS.sha256 even for a terminal run",
    )
    args = p.parse_args(argv)
    run = Path(args.run_dir)

    manifest = _read_json(run / "00_RUN_MANIFEST.json") or {}
    metrics = _read_json(run / "20_METRICS" / "METRICS.json") or {}
    validation = _read_json(run / "30_VALIDATION" / "VALIDATION.json")
    heldout = _read_json(run / "40_HELDOUT" / "HELDOUT.json")

    summary = [
        f"# Run summary — {manifest.get('run_id', '?')}",
        "",
        f"- Status: **{manifest.get('status', '?')}**",
        f"- Decision: {manifest.get('results', {}).get('decision')}",
        f"- Submitted LoRA: {manifest.get('results', {}).get('submitted_lora')}",
        f"- Created (UTC): {manifest.get('created_at_utc')}",
        f"- Finished (UTC): {manifest.get('finished_at_utc')}",
        "",
        "## TL;DR",
        "",
        "_Fill the one-paragraph interpretation here; numbers below are authoritative._",
        "",
        "## Key metrics",
        "",
        "```json",
        json.dumps(metrics, indent=2, sort_keys=True)[:4000],
        "```",
        "",
        _stats_block("Reserved validation", validation),
        _stats_block("Held-out evaluation", heldout),
        "",
        "## Links",
        "",
        "- Manifest: `00_RUN_MANIFEST.json`",
        "- Validation: `30_VALIDATION/VALIDATION.json`",
        "- Held-out: `40_HELDOUT/HELDOUT.json`",
        "- Checksums: `80_FINAL/CHECKSUMS.sha256`",
        "",
    ]
    write_text(run / "50_ANALYSIS" / "SUMMARY.md", "\n".join(summary))

    tables = [
        f"# Tables — {manifest.get('run_id', '?')}",
        "",
        "Generated from JSON; do not hand-edit.",
        "",
        _stats_block("Reserved validation", validation),
        _stats_block("Held-out evaluation", heldout),
    ]
    write_text(run / "50_ANALYSIS" / "TABLES.md", "\n".join(tables))

    analysis_path = run / "50_ANALYSIS" / "ANALYSIS.md"
    if not analysis_path.exists():
        analysis = [
            f"# Analysis — {manifest.get('run_id', '?')}",
            "",
            "## Context and protocol",
            "",
            "_Describe the exact protocol/budgets used; link to `01_CONFIG.toml`._",
            "",
            "## Evidence table",
            "",
            "_Passes/batches/cache provenance; see `20_METRICS/METRICS.json`._",
            "",
            "## Statistical interpretation and limitations",
            "",
            "_Use the problem-clustered sign-flip result; state limitations explicitly._",
            "",
            "## Operational issues",
            "",
            "_Preemption/resume/cache incidents, or 'none'._",
            "",
            "## What it means for the next experiment",
            "",
            "_One paragraph._",
            "",
            "## Provenance",
            "",
            "- Manifest: `00_RUN_MANIFEST.json`",
            "- Checksums: `80_FINAL/CHECKSUMS.sha256`",
            "",
        ]
        write_text(analysis_path, "\n".join(analysis))

    if manifest.get("status") in TERMINAL_STATUSES and not args.force:
        print(
            "refusing to re-seal CHECKSUMS for terminal status "
            f"{manifest.get('status')!r}; pass --force to override",
            file=sys.stderr,
        )
        return 1

    write_checksums(str(run))
    print(f"wrote {run / '50_ANALYSIS' / 'SUMMARY.md'}, TABLES.md, ANALYSIS.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
