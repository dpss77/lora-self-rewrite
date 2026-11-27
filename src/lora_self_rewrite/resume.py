"""Preemption/resume continuity helpers.

Hard-won lesson (`S02-DS-SEED00`): a manual resume kept the active LoRA and
candidates but reset the pass budget and transcript to zero. That is an
operational defect, not a scientific result. These helpers make the resume
decision explicit and restore the counters from durable artifacts when asked to
resume.
"""

from __future__ import annotations

import glob
import json
import os
import re

__all__ = ["scan_work_dir", "resume_plan"]

_PASS_FILE_RE = re.compile(r"^pass_(\d+)\.json$")
_PASS_INFLIGHT_RE = re.compile(r"^pass_(\d+)\.inflight$")
_PASS_OBS_RE = re.compile(r"\[pass (\d+)/")
_CODE_OBS_RE = re.compile(r"\[code (\d+)/")


def _read_json(path: str):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _read_jsonl(path: str) -> list[dict]:
    rows = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        pass
    return rows


def scan_work_dir(work_dir: str) -> dict:
    """Inspect a ``WORK`` directory and return a resumable summary.

    ``passes_used`` counts not just completed ``pass_<n>.json`` files but also
    in-flight markers written before a pass spends GPU time. A preemption during
    pass N leaves no ``pass_N.json``; without the marker the resume would refund
    that pass and the run could exceed its budget. Transcript pass counters are a
    fallback for read passes (which write no ``pass_<n>.json``) and for older
    transcripts. ``turns`` come from ``transcript.jsonl`` when it is longer than
    the ``transcript.json`` snapshot (the jsonl is committed every turn); the
    counters are then recovered from ``[pass N/`` / ``[code N/M]`` obs rows.
    """
    pass_files = sorted(glob.glob(os.path.join(work_dir, "pass_*.json")))
    inflight_files = sorted(glob.glob(os.path.join(work_dir, "pass_*.inflight")))
    passes_used = 0
    for path in pass_files + inflight_files:
        match = _PASS_FILE_RE.match(os.path.basename(path)) or _PASS_INFLIGHT_RE.match(
            os.path.basename(path)
        )
        if match:
            passes_used = max(passes_used, int(match.group(1)))

    transcript_json = _read_json(os.path.join(work_dir, "transcript.json"))
    summary: dict = {}
    turns: list[dict] = []
    if isinstance(transcript_json, dict):
        summary = transcript_json.get("summary") or {}
        turns = transcript_json.get("turns") or []
    elif isinstance(transcript_json, list):
        turns = transcript_json
    # The jsonl is appended/committed every turn, while transcript.json is a
    # per-attempt snapshot written only on clean exit. Prefer the longer live log.
    jsonl_turns = _read_jsonl(os.path.join(work_dir, "transcript.jsonl"))
    if len(jsonl_turns) > len(turns):
        turns = jsonl_turns

    # Durable per-turn resume state: messages, cumulative turn count, accumulated
    # wall time, counters, and lineage. Written every turn so a hard preemption does
    # not reset the turn/wall budgets to zero on the next attempt.
    resume_state = _read_json(os.path.join(work_dir, "resume_state.json"))
    if not isinstance(resume_state, dict):
        resume_state = {}
    resume_messages = resume_state.get("messages")
    if not isinstance(resume_messages, list):
        resume_messages = None
    resume_lineage = resume_state.get("resume_lineage")
    if not isinstance(resume_lineage, list):
        resume_lineage = []

    # Read passes leave no pass_<n>.json, and older summaries may predate the
    # in-flight marker. Recover the counter from any durable record we have.
    summary_passes = summary.get("passes_used")
    if isinstance(summary_passes, int):
        passes_used = max(passes_used, summary_passes)
    for entry in turns:
        if not isinstance(entry, dict):
            continue
        entry_passes = entry.get("passes_used")
        if isinstance(entry_passes, int):
            passes_used = max(passes_used, entry_passes)
            continue
        match = _PASS_OBS_RE.search(str(entry.get("obs") or ""))
        if match:
            passes_used = max(passes_used, int(match.group(1)))

    code_runs_used = int(summary.get("code_runs_used") or 0)
    for entry in turns:
        if not isinstance(entry, dict):
            continue
        match = _CODE_OBS_RE.search(str(entry.get("obs") or ""))
        if match:
            code_runs_used = max(code_runs_used, int(match.group(1)))

    state_passes = resume_state.get("passes_used")
    if isinstance(state_passes, int):
        passes_used = max(passes_used, state_passes)
    state_code_runs = resume_state.get("code_runs_used")
    if isinstance(state_code_runs, int):
        code_runs_used = max(code_runs_used, state_code_runs)

    turns_used_prior = resume_state.get("turns_used_prior")
    if not isinstance(turns_used_prior, int):
        turns_used_prior = summary.get("turns_used_prior")
    if not isinstance(turns_used_prior, int):
        turns_used_prior = len(turns)
    # Never let a stale state file under-count durable completed turns.
    turns_used_prior = max(turns_used_prior, len(turns))

    elapsed_wall_prior_s = resume_state.get("elapsed_wall_prior_s")
    if not isinstance(elapsed_wall_prior_s, (int, float)):
        elapsed_wall_prior_s = summary.get("elapsed_wall_prior_s")
    if not isinstance(elapsed_wall_prior_s, (int, float)):
        elapsed_wall_prior_s = summary.get("elapsed_sec")
    if not isinstance(elapsed_wall_prior_s, (int, float)):
        elapsed_wall_prior_s = 0.0
    elapsed_wall_prior_s = max(0.0, float(elapsed_wall_prior_s))

    candidates_dir = os.path.join(work_dir, "candidates")
    reads_dir = os.path.join(work_dir, "reads")
    submitted_dir = os.path.join(work_dir, "submitted_lora")
    dirty_reasons: list[str] = []
    if os.path.isdir(candidates_dir) and os.listdir(candidates_dir):
        dirty_reasons.append("candidates/")
    if os.path.isdir(reads_dir) and os.listdir(reads_dir):
        dirty_reasons.append("reads/")
    if os.path.isdir(submitted_dir) and os.listdir(submitted_dir):
        dirty_reasons.append("submitted_lora/")
    if pass_files:
        dirty_reasons.append("pass_*.json")
    if inflight_files:
        dirty_reasons.append("pass_*.inflight")
    for name in (
        "notes.md",
        "transcript.jsonl",
        "transcript.json",
        "metrics.json",
        "resume_state.json",
        "census.jsonl",
        "census.jsonl.meta.json",
    ):
        if os.path.exists(os.path.join(work_dir, name)):
            dirty_reasons.append(name)

    return {
        "work_dir": work_dir,
        "has_work": bool(dirty_reasons),
        "dirty_reasons": dirty_reasons,
        "passes_used": passes_used,
        "code_runs_used": code_runs_used,
        "turns": turns,
        "turns_completed": len(turns),
        "turns_used_prior": turns_used_prior,
        "elapsed_wall_prior_s": elapsed_wall_prior_s,
        "messages": resume_messages,
        "resume_lineage": resume_lineage,
        "review_summary": summary,
    }


def resume_plan(
    work_dir: str,
    *,
    overwrite_session: bool = False,
    reset_active_lora: bool = True,
) -> dict:
    """Decide how to handle an existing ``WORK`` directory.

    Actions:
      * ``fresh``     — no work; start normally.
      * ``overwrite`` — caller asked to wipe; start normally.
      * ``refuse``    — dirty work but neither overwrite nor deliberate resume.
      * ``resume``    — deliberate resume; restore counters/transcript.
    """
    scan = scan_work_dir(work_dir)
    if not scan["has_work"]:
        return {"action": "fresh", "resumed": False, "scan": scan}
    if overwrite_session:
        return {"action": "overwrite", "resumed": False, "scan": scan}
    if reset_active_lora:
        return {
            "action": "refuse",
            "resumed": False,
            "scan": scan,
            "reason": (
                f"Session WORK state already exists {scan['dirty_reasons']}. "
                "Use a fresh --session-id, pass --overwrite-session to wipe it, or "
                "use --no-reset-active-lora to resume deliberately."
            ),
        }
    return {"action": "resume", "resumed": True, "scan": scan}
