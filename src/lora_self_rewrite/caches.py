"""Exact-batch frozen-base cache and content-addressed LoRA ids.

vLLM bf16 kernels are not batch-size invariant, so a base-result cache is only
valid for a *whole identical ordered batch*: same dataset content, ordered pids,
sample count, token budget, hardware/kernel tag, and generation config. Never
mix rows across batches or shards.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Callable

from .constants import BASE_CACHE_SCHEMA, GENERATION_CONFIG_HASH, LOOP_METRIC_MODE

__all__ = [
    "norm_question",
    "label_schema",
    "CUT_DEFINITION",
    "GRADER_IDENTITY",
    "DIAGNOSIS_CACHE_SCHEMA",
    "base_batch_signature",
    "diagnosis_identity",
    "load_base_batch_cache",
    "load_diagnosis_cache",
    "load_validation_base_cache",
    "save_base_batch_cache",
    "save_diagnosis_cache",
    "save_validation_base_cache",
    "content_addressed_lora_key",
    "LoraIdRegistry",
]

# Stable identity of the engine/donor ``cut`` label definition. Bump this string
# (and BASE_CACHE_SCHEMA) whenever the label definition changes, so cached labels
# can never be silently mixed across definitions.
CUT_DEFINITION = "tokens>=max_tokens or finish_reason==length"

# Identity of the grading implementation. Bump when the math-verify version or the
# ``</think>``-closed / final-``\\boxed{}`` rule changes, so diagnosis labels are
# re-derived rather than silently reused across grader versions.
GRADER_IDENTITY = "grade_answer:math_verify==0.9.0:closed-think+final-boxed"

# Diagnosis caches store generated text, so they can be reused across label/grader
# changes (labels are re-derived from text) but never across a different generation
# identity (``max_tokens``/content/order).
DIAGNOSIS_CACHE_SCHEMA = 1
VALIDATION_CACHE_SCHEMA = 1


def label_schema() -> dict[str, str]:
    """Identity of the persisted label definitions (``cut`` + ``loop``)."""
    return {
        "loop_metric_mode": LOOP_METRIC_MODE,
        "cut_definition": CUT_DEFINITION,
    }


def norm_question(s: str) -> str:
    """Normalize a problem for dedup/cache hashing."""
    return re.sub(r"\s+", " ", s.strip().lower())


def base_batch_signature(
    dataset_mode: str,
    pids: list[int],
    problems: list[str],
    n_samples: int,
    max_tokens: int,
    cache_tag: str,
    extra: dict | None = None,
) -> tuple[str, dict]:
    """Identity of an exact ordered base-generation batch.

    ``extra`` extends the identity with caller-specific dimensions (e.g. held-out
    ``k_shards`` and model revision) without changing session callers.
    """
    problem_hashes = [
        hashlib.sha256(norm_question(problem).encode()).hexdigest() for problem in problems
    ]
    payload = {
        "cache_schema": BASE_CACHE_SCHEMA,
        "dataset_mode": dataset_mode,
        "pids": list(pids),
        "problem_hashes": problem_hashes,
        "n_samples": int(n_samples),
        "max_tokens": int(max_tokens),
        "cache_tag": cache_tag,
        "generation_config_hash": GENERATION_CONFIG_HASH,
        "label_schema": label_schema(),
    }
    if extra:
        payload.update(extra)
    signature = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:24]
    return signature, {**payload, "signature": signature}


def _text_hash(value: object) -> str:
    return hashlib.sha256(norm_question(str(value)).encode()).hexdigest()


# Fields that describe *what text* a diagnosis cache contains. A difference in any
# of these means the stored generations were produced under a different budget,
# dataset, or ordering and must not be relabelled.
_DIAGNOSIS_GENERATION_FIELDS = (
    "cache_schema",
    "dataset_mode",
    "dataset_identity",
    "pids",
    "problem_hashes",
    "answer_hashes",
    "n_items",
    "max_tokens",
    "generation_config_hash",
)


def diagnosis_identity(
    dataset_mode: str,
    items: list[dict],
    max_tokens: int,
) -> tuple[str, dict]:
    """Identity of an ordered diagnosis generation set.

    The signature keys on dataset mode/content, ordered problem *and* answer hashes,
    token budget, generation config, label schema, grader identity, and cache schema.
    """
    pids = [item["id"] for item in items]
    problem_hashes = [_text_hash(item["problem"]) for item in items]
    answer_hashes = [_text_hash(item["answer"]) for item in items]
    dataset_identity = hashlib.sha256(
        json.dumps(
            {
                "pids": pids,
                "problem_hashes": problem_hashes,
                "answer_hashes": answer_hashes,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    payload = {
        "cache_schema": DIAGNOSIS_CACHE_SCHEMA,
        "dataset_mode": dataset_mode,
        "dataset_identity": dataset_identity,
        "pids": pids,
        "problem_hashes": problem_hashes,
        "answer_hashes": answer_hashes,
        "n_items": len(items),
        "max_tokens": int(max_tokens),
        "generation_config_hash": GENERATION_CONFIG_HASH,
        "label_schema": label_schema(),
        "grader_identity": GRADER_IDENTITY,
    }
    signature = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:24]
    return signature, {**payload, "signature": signature}


_DIAG_REQUIRED_ROW_KEYS = {"id", "problem", "answer", "text"}
_DIAG_EXPECTED_IDENTITY_KEYS = ("id", "problem", "answer")


def _diagnosis_generation_compatible(stored: dict, expected: dict) -> bool:
    """True when a stored cache was generated under the same content/budget.

    Meta written by the pre-fix session lacks the hash fields; those old caches are
    still usable through the content-match compatibility path, but a disagreement on
    any field that *is* present (notably ``max_tokens``) rejects the cache.
    """
    for field in _DIAGNOSIS_GENERATION_FIELDS:
        if field in stored and stored[field] != expected.get(field):
            return False
    if "max_tokens" in stored:
        try:
            if int(stored["max_tokens"]) != int(expected["max_tokens"]):
                return False
        except (TypeError, ValueError):
            return False
    return True


def load_diagnosis_cache(
    path: str,
    meta: dict,
    expected: list[dict],
    grade: Callable[[str, str], bool],
    *,
    allow_legacy_text: bool = True,
) -> tuple[list[dict], str] | None:
    """Load and re-grade a diagnosis census, or return ``None`` on identity mismatch.

    Returns ``(rows, status)`` where status is ``"hit"`` (fully keyed), ``"regraded"``
    (label/grader identity moved but generation identity and content match), or
    ``"legacy"`` (no usable sidecar; content-match compatibility path for already-paid
    text). Cached ``correct``/``cut``/``loop`` labels are always re-derived from
    ``text`` + ``answer`` with the current definitions.
    """
    from .protocol_shared import is_looping_answer

    try:
        with open(path) as f:
            rows = [json.loads(line) for line in f if line.strip()]
    except (OSError, json.JSONDecodeError):
        return None
    if len(rows) != len(expected):
        return None
    for row, exp in zip(rows, expected, strict=True):
        if not isinstance(row, dict) or not _DIAG_REQUIRED_ROW_KEYS <= set(row):
            return None
        if any(row.get(k) != exp.get(k) for k in _DIAG_EXPECTED_IDENTITY_KEYS):
            return None

    stored = None
    meta_path = f"{path}.meta.json"
    if os.path.exists(meta_path):
        try:
            with open(meta_path) as f:
                stored = json.load(f)
        except (OSError, json.JSONDecodeError):
            stored = None
    if stored is not None and not isinstance(stored, dict):
        stored = None

    max_tokens = int(meta["max_tokens"])
    if stored is None:
        if not allow_legacy_text:
            return None
        # Without a keyed budget, refuse text that visibly exceeded this budget:
        # relabelling a larger-budget generation as a current cut/non-cut would be
        # scientifically wrong.
        if any(int(row.get("tokens") or 0) > max_tokens for row in rows):
            return None
        status = "legacy"
    else:
        if not _diagnosis_generation_compatible(stored, meta):
            return None
        if "max_tokens" not in stored and any(
            int(row.get("tokens") or 0) > max_tokens for row in rows
        ):
            return None
        label_changed = stored.get("label_schema") != meta.get("label_schema") or (
            stored.get("grader_identity") != meta.get("grader_identity")
        )
        status = "regraded" if label_changed else "hit"

    for row in rows:
        row.setdefault("sample_idx", 0)
        try:
            tokens = int(row.get("tokens") or 0)
        except (TypeError, ValueError):
            tokens = 0
        row["tokens"] = tokens
        row.setdefault("finish_reason", None)
        cut = tokens >= max_tokens or row["finish_reason"] == "length"
        row["cut"] = cut
        row["correct"] = grade(row["text"], row["answer"])
        row["loop"] = is_looping_answer(row["text"], cut)
    return rows, status


def save_diagnosis_cache(path: str, meta: dict, rows: list[dict]) -> None:
    """Atomically write a JSONL census plus its identity sidecar."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    os.replace(tmp, path)
    meta_path = f"{path}.meta.json"
    tmp_meta = f"{meta_path}.tmp.{os.getpid()}"
    with open(tmp_meta, "w") as f:
        json.dump(meta, f)
    os.replace(tmp_meta, meta_path)


_REQUIRED_ROW_KEYS = {
    "id",
    "sample_idx",
    "problem",
    "answer",
    "text",
    "tokens",
    "cut",
    "loop",
}


def load_base_batch_cache(
    path: str,
    meta: dict,
    expected: list[dict],
    grade: Callable[[str, str], bool],
) -> list[dict] | None:
    """Load a cached base batch only if its exact identity and rows match.

    Cached correctness and loop labels are always re-derived with the current
    grader/loop definition.
    """
    from .protocol_shared import is_looping_answer

    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict) or data.get("meta") != meta:
        return None
    rows = data.get("rows")
    if not isinstance(rows, list) or len(rows) != len(expected):
        return None
    for row, exp in zip(rows, expected, strict=True):
        if (
            not isinstance(row, dict)
            or not _REQUIRED_ROW_KEYS <= set(row)
            or any(row.get(k) != v for k, v in exp.items())
        ):
            return None
        row.setdefault("finish_reason", None)
        row["correct"] = grade(row["text"], row["answer"])
        row["loop"] = is_looping_answer(row["text"], bool(row["cut"]))
    return rows


def save_base_batch_cache(path: str, meta: dict, rows: list[dict]) -> None:
    """Atomically write a base-batch cache file."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump({"meta": meta, "rows": rows}, f)
    os.replace(tmp, path)


_VALIDATION_REQUIRED_ROW_KEYS = {
    "id",
    "sample_idx",
    "problem",
    "answer",
    "text",
    "tokens",
    "cut",
    "loop",
}


def load_validation_base_cache(
    path: str,
    expected: list[dict],
    *,
    engine_config_hash: str,
    max_tokens: int,
    grade: Callable[[str, str], bool],
) -> list[dict] | None:
    """Load the reserved-validation base arm only on full content identity.

    The answer is part of the identity, so a changed ground truth cannot be graded
    against stale rows. Labels are always re-derived from stored text.
    """
    from .protocol_shared import is_looping_answer

    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    if data.get("label_schema") != label_schema():
        return None
    if data.get("engine_config_hash") != engine_config_hash:
        return None
    if data.get("cache_schema") != VALIDATION_CACHE_SCHEMA:
        return None
    rows = data.get("rows")
    if not isinstance(rows, list) or len(rows) != len(expected):
        return None
    for row, exp in zip(rows, expected, strict=True):
        if (
            not isinstance(row, dict)
            or not _VALIDATION_REQUIRED_ROW_KEYS <= set(row)
            or any(row.get(key) != value for key, value in exp.items())
        ):
            return None
        row.setdefault("finish_reason", None)
        try:
            tokens = int(row.get("tokens") or 0)
        except (TypeError, ValueError):
            tokens = 0
        row["tokens"] = tokens
        row["cut"] = tokens >= int(max_tokens) or row["finish_reason"] == "length"
        row["correct"] = grade(row["text"], row["answer"])
        row["loop"] = is_looping_answer(row["text"], bool(row["cut"]))
    return rows


def save_validation_base_cache(
    path: str,
    rows: list[dict],
    *,
    engine_config_hash: str,
    max_tokens: int,
) -> None:
    """Atomically write a validation base cache with its full identity."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    payload = {
        "cache_schema": VALIDATION_CACHE_SCHEMA,
        "label_schema": label_schema(),
        "engine_config_hash": engine_config_hash,
        "max_tokens": int(max_tokens),
        "rows": rows,
    }
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(payload, f)
    os.replace(tmp, path)


def content_addressed_lora_key(lora_dir: str) -> str:
    """SHA-256 over ``adapter_model.safetensors`` + ``adapter_config.json`` bytes."""
    digest = hashlib.sha256()
    for name in ("adapter_model.safetensors", "adapter_config.json"):
        path = os.path.join(lora_dir, name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing LoRA file: {path}")
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


class LoraIdRegistry:
    """Map a content digest to a monotonic positive vLLM LoRA id.

    vLLM ignores/rejects id 0 and can reuse stale GPU adapter caches when an
    in-place edit keeps the same id. Content addressing prevents both.
    """

    def __init__(self) -> None:
        self._next_id = 1
        self._by_key: dict[str, int] = {}

    def allocate(self) -> int:
        lora_id = self._next_id
        if lora_id >= 2**31:
            raise RuntimeError("vLLM LoRA id space exhausted in this container.")
        self._next_id += 1
        return lora_id

    def id_for_dir(self, lora_dir: str) -> int:
        key = content_addressed_lora_key(lora_dir)
        lora_id = self._by_key.get(key)
        if lora_id is None:
            lora_id = self.allocate()
            self._by_key[key] = lora_id
        return lora_id


# ---------------------------------------------------------------------------
# Donor-compatible aliases used by the mechanically ported engine/session.
# ---------------------------------------------------------------------------
_base_batch_signature_alias = base_batch_signature
_save_base_batch_cache_alias = save_base_batch_cache
_norm_q_alias = norm_question


def _load_base_batch_cache_alias(path: str, meta: dict, expected: list[dict]):
    from .protocol_shared import grade_answer

    return load_base_batch_cache(path, meta, expected, grade_answer)
