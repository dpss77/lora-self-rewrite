"""Dataset construction and session splits (PAPER: Table 5).

The pure split/shuffle logic is importable without ``datasets``; the network
loader is lazy so pytest never needs to reach the Hub.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import random
import time

from .caches import norm_question
from .constants import (
    DEEPSCALER_SCRAPER_VERSION,
    DEEPSCALER_SHUFFLE_SEED,
    DEEPSCALER_SPLITS,
    HELDOUT_N_PROBLEMS,
    HELDOUT_SLICE,
    WORKSPACE_DIR,
)

__all__ = [
    "build_deepscaler_payload",
    "dataset_identity",
    "dataset_source_revisions",
    "heldout_dataset_identity",
    "heldout_problems",
    "payload_content_sha256",
    "require_pinned_sources",
    "split_payload",
    "validate_heldout_pool",
    "load_session_datasets",
    "load_deepscaler_payload",
]

# Pinned revision for every dataset source used by the scrubbed pool. Resolved
# from the HF Hub dataset repos; a loader must request these exact revisions.
MATH_SOURCES = [
    (
        "HuggingFaceH4/MATH-500",
        None,
        "test",
        "problem",
        "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be",
    ),
    (
        "Maxwell-Jia/AIME_2024",
        None,
        "train",
        "Problem",
        "8d88b2876a82a080e2f172cc9b25d0d9d2cb4792",
    ),
    (
        "math-ai/aime25",
        None,
        "test",
        "problem",
        "563bb8404243c5f09de6ec262f2db674fe5bce9b",
    ),
    (
        "math-ai/amc23",
        None,
        "test",
        "question",
        "80815d37005feb82cd7f8fbc6901d5d3eff43057",
    ),
    (
        "MathArena/hmmt_feb_2025",
        None,
        "train",
        "problem",
        "6fdc4277120810ff75aa22d2d5489b91f7a262a1",
    ),
    (
        "Hothan/OlympiadBench",
        "OE_TO_maths_en_COMP",
        "train",
        "question",
        "91184b52131e7fc9455fef848035173aea8cc01a",
    ),
    (
        "Hothan/OlympiadBench",
        "TP_TO_maths_en_COMP",
        "train",
        "question",
        "91184b52131e7fc9455fef848035173aea8cc01a",
    ),
]
GPQA_SOURCE = (
    "Idavidrein/gpqa",
    "gpqa_diamond",
    "train",
    "Question",
    "83022cefff930aea54f654c0b282e74b9eeda5c6",
)
DEEPSCALER_SOURCE = (
    "agentica-org/DeepScaleR-Preview-Dataset",
    None,
    "train",
    "b6ae8c60f5c1f2b594e2140b91c49c9ad0949e29",
)

DATASET_SOURCE_REVISIONS = {repo: revision for repo, _cfg, _split, _col, revision in MATH_SOURCES}
DATASET_SOURCE_REVISIONS[GPQA_SOURCE[0]] = GPQA_SOURCE[4]
DATASET_SOURCE_REVISIONS[DEEPSCALER_SOURCE[0]] = DEEPSCALER_SOURCE[3]


def _call_loader(loader, repo, cfg, split, revision):
    """Call a loader with a pinned revision when its signature supports one."""
    try:
        parameters = inspect.signature(loader).parameters
    except (TypeError, ValueError):
        parameters = {}
    if "revision" in parameters:
        return loader(repo, cfg, split=split, revision=revision)
    return loader(repo, cfg, split)


def _text_hash(value: object) -> str:
    return hashlib.sha256(norm_question(str(value)).encode()).hexdigest()


def payload_content_sha256(payload: dict) -> str:
    """Content hash of the ordered practice/reserved/heldout problem subsets."""
    content = {key: payload.get(key, []) for key in ("practice", "reserved", "heldout")}
    return hashlib.sha256(
        json.dumps(content, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def dataset_source_revisions(dataset_mode: str) -> dict[str, str]:
    """Pinned HF revisions for every source that defines a session split."""
    if dataset_mode == "math500":
        return {"HuggingFaceH4/MATH-500": DATASET_SOURCE_REVISIONS["HuggingFaceH4/MATH-500"]}
    if dataset_mode == "deepscaler":
        return {
            repo: revision
            for repo, revision in DATASET_SOURCE_REVISIONS.items()
            if repo not in {"HuggingFaceH4/MATH-500"}
        }
    raise ValueError(f"Unknown dataset_mode '{dataset_mode}'.")


def require_pinned_sources(dataset_mode: str) -> dict[str, str]:
    """Reject an unknown dataset revision on a cacheable/scientific path."""
    revisions = dataset_source_revisions(dataset_mode)
    missing = [
        repo
        for repo, revision in revisions.items()
        if not revision or str(revision).strip().lower() in {"unknown", "none", "null"}
    ]
    if missing:
        raise ValueError(
            f"Dataset sources for '{dataset_mode}' have unpinned revisions: {missing}."
        )
    return revisions


def dataset_identity(dataset_mode: str, practice: list[dict], reserved: list[dict]) -> dict:
    """Content + source-revision identity recorded in the run manifest/metrics."""
    rows = [
        (_text_hash(item["problem"]), _text_hash(item["answer"])) for item in [*practice, *reserved]
    ]
    content_sha = hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()
    revisions = dataset_source_revisions(dataset_mode)
    identity = hashlib.sha256(
        json.dumps(
            {
                "dataset_mode": dataset_mode,
                "source_revisions": revisions,
                "content_sha256": content_sha,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()[:24]
    return {
        "dataset_mode": dataset_mode,
        "source_revisions": revisions,
        "content_sha256": content_sha,
        "identity": identity,
    }


def split_payload(clean: list[dict], *, allow_short: bool = False) -> dict:
    """Apply the fixed practice/reserved/heldout split to a cleaned problem list.

    The fixed held-out gate needs ``DEEPSCALER_SPLITS["heldout"][1]`` (4300) clean
    problems. A shorter pool would silently produce a short held-out slice, so it
    raises unless ``allow_short`` is explicitly set for tests/smoke.
    """
    required = DEEPSCALER_SPLITS["heldout"][1]
    if len(clean) < required and not allow_short:
        raise ValueError(
            f"cleaned pool has {len(clean)} problems; the fixed practice/reserved/"
            f"heldout split needs at least {required}. Pass allow_short=True only "
            "for a deliberate test/smoke run."
        )
    return {
        "practice": clean[DEEPSCALER_SPLITS["practice"][0] : DEEPSCALER_SPLITS["practice"][1]],
        "reserved": clean[DEEPSCALER_SPLITS["reserved"][0] : DEEPSCALER_SPLITS["reserved"][1]],
        "heldout": clean[DEEPSCALER_SPLITS["heldout"][0] : DEEPSCALER_SPLITS["heldout"][1]],
    }


def heldout_problems(payload: dict) -> list[dict]:
    """Return the fixed Table-5 held-out slice (``heldout[500:990]``)."""
    pool = payload.get("heldout", [])
    return pool[HELDOUT_SLICE[0] : HELDOUT_SLICE[1]]


def validate_heldout_pool(
    problems: list[dict],
    *,
    expected: int = HELDOUT_N_PROBLEMS,
    smoke_problems: int | None = None,
) -> list[dict]:
    """Enforce the fixed held-out gate size, allowing an explicit smoke truncation."""
    if not problems:
        raise ValueError("heldout pool is empty; cannot evaluate")
    if smoke_problems is None and len(problems) != expected:
        raise ValueError(
            f"held-out pool has {len(problems)} problems; expected {expected} "
            f"(heldout[{HELDOUT_SLICE[0]}:{HELDOUT_SLICE[1]}]). Pass smoke_problems "
            "for a deliberate small run."
        )
    if smoke_problems is not None:
        return problems[:smoke_problems]
    return problems


def heldout_dataset_identity(payload: dict, problems: list[dict]) -> dict:
    """Identity for the fixed held-out slice, including source revisions."""
    meta = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}
    revisions = meta.get("source_revisions") or dataset_source_revisions("deepscaler")
    rows = [(_text_hash(problem["problem"]), _text_hash(problem["answer"])) for problem in problems]
    content_sha = hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()
    identity = hashlib.sha256(
        json.dumps(
            {
                "slice": list(HELDOUT_SLICE),
                "source_revisions": revisions,
                "content_sha256": content_sha,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()[:24]
    return {
        "heldout_slice": list(HELDOUT_SLICE),
        "source_revisions": revisions,
        "content_sha256": content_sha,
        "identity": identity,
    }


def build_deepscaler_payload(loader) -> dict:
    """Scrub DeepScaleR against all eval sources and return the dataset payload.

    ``loader(repo, cfg, split)`` must return an iterable of row dicts. All seven
    math sources must load; GPQA is optional because it is gated.
    """
    forbidden: set[str] = set()
    dedup_counts: dict[str, dict] = {}
    n_math_ok = 0
    gpqa_ok = False

    for repo, cfg, split, col, revision in [*MATH_SOURCES, GPQA_SOURCE]:
        key = f"{repo}:{cfg or 'default'}:{split}"
        optional = repo == GPQA_SOURCE[0]
        try:
            rows = _call_loader(loader, repo, cfg, split, revision)
            added = 0
            for row in rows:
                value = row.get(col)
                if isinstance(value, str):
                    norm = norm_question(value)
                    if norm not in forbidden:
                        forbidden.add(norm)
                        added += 1
            dedup_counts[key] = {
                "ok": True,
                "questions": added,
                "optional": optional,
                "revision": revision,
            }
            if optional:
                gpqa_ok = True
            else:
                n_math_ok += 1
        except Exception as e:  # noqa: BLE001 - recorded, then enforced below
            dedup_counts[key] = {
                "ok": False,
                "error": f"{type(e).__name__}: {e}",
                "optional": optional,
                "revision": revision,
            }
            print(f"[dedup] WARNING skipping {key}: {e}")

    if n_math_ok < len(MATH_SOURCES):
        raise RuntimeError(
            f"DeepScaleR dedup requires all {len(MATH_SOURCES)} math sources; "
            f"only {n_math_ok} loaded. Details: {dedup_counts}"
        )

    ds_repo, ds_cfg, ds_split, ds_revision = DEEPSCALER_SOURCE
    ds = _call_loader(loader, ds_repo, ds_cfg, ds_split, ds_revision)
    seen_clean: set[str] = set()
    clean: list[dict] = []
    internal_duplicates = 0
    for i, row in enumerate(ds):
        prob, ans = row.get("problem", ""), row.get("answer", "")
        if not prob or not ans:
            continue
        norm = norm_question(prob)
        if norm in forbidden:
            continue
        if norm in seen_clean:
            internal_duplicates += 1
            continue
        seen_clean.add(norm)
        clean.append({"id": i, "problem": prob, "answer": str(ans)})
    random.Random(DEEPSCALER_SHUFFLE_SEED).shuffle(clean)

    payload = split_payload(clean)
    payload["meta"] = {
        "dedup_sources": dedup_counts,
        "scraper_version": DEEPSCALER_SCRAPER_VERSION,
        "internal_dedup": True,
        "n_math_ok": n_math_ok,
        "gpqa_ok": gpqa_ok,
        "n_ok": n_math_ok + int(gpqa_ok),
        "n_expected_math_sources": len(MATH_SOURCES),
        "n_internal_duplicates_removed": internal_duplicates,
        "n_forbidden_questions": len(forbidden),
        "n_clean": len(clean),
        "n_heldout": len(payload["heldout"]),
        "source_revisions": {
            **{
                key.split(":", 1)[0]: value["revision"]
                for key, value in dedup_counts.items()
                if value.get("ok")
            },
            ds_repo: ds_revision,
        },
        "dataset_sha256": payload_content_sha256(payload),
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    return payload


def load_deepscaler_payload(
    path: str,
    loader,
    *,
    force_rebuild: bool = False,
) -> dict:
    """Load a cached scrubbed DeepScaleR payload, rebuilding if stale/missing."""
    payload = None
    if not force_rebuild and os.path.exists(path):
        try:
            with open(path) as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError):
            payload = None
        meta = payload.get("meta") if isinstance(payload, dict) else None
        expected_revisions = dataset_source_revisions("deepscaler")
        outdated = (
            not isinstance(payload, dict)
            or not isinstance(meta, dict)
            or meta.get("scraper_version") != DEEPSCALER_SCRAPER_VERSION
            or "heldout" not in payload
            or int(meta.get("n_math_ok", 0)) < len(MATH_SOURCES)
            or any(
                (meta.get("source_revisions") or {}).get(repo) != revision
                for repo, revision in expected_revisions.items()
                if repo != GPQA_SOURCE[0]
            )
            or meta.get("dataset_sha256") != payload_content_sha256(payload)
        )
        if outdated:
            payload = None

    if payload is None:
        payload = build_deepscaler_payload(loader)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = f"{path}.tmp.{os.getpid()}"
        with open(tmp, "w") as f:
            json.dump(payload, f)
        os.replace(tmp, path)
    return payload


def load_session_datasets(
    dataset_mode: str,
    *,
    workspace_dir: str = WORKSPACE_DIR,
    loader=None,
) -> tuple[list[dict], list[dict]]:
    """Return ``(practice, reserved)`` for a session.

    ``loader`` is a ``datasets.load_dataset``-compatible callable; when ``None``
    the real ``datasets`` library is imported lazily.
    """
    if loader is None:
        from datasets import load_dataset as _load

        def loader(repo, cfg, split, revision=None):  # noqa: ANN001
            kwargs = {"split": split}
            if revision:
                kwargs["revision"] = revision
            return _load(repo, cfg, **kwargs) if cfg else _load(repo, **kwargs)

    if dataset_mode == "math500":
        revision = DATASET_SOURCE_REVISIONS["HuggingFaceH4/MATH-500"]
        ds = _call_loader(loader, "HuggingFaceH4/MATH-500", None, "test", revision)
        all_probs = [
            {"id": i, "problem": row["problem"], "answer": row["answer"]}
            for i, row in enumerate(ds)
        ]
        return all_probs[:400], all_probs[400:500]

    if dataset_mode == "deepscaler":
        path = f"{workspace_dir}/datasets/deepscaler_scrubbed.json"
        payload = load_deepscaler_payload(path, loader)
        return payload["practice"], payload["reserved"]

    raise ValueError(f"Unknown dataset_mode '{dataset_mode}'.")
