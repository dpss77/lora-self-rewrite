"""Pure helpers shared by held-out evaluation (no Modal/vLLM imports)."""

from __future__ import annotations

import gc
import hashlib

from .caches import LoraIdRegistry, base_batch_signature
from .constants import GENERATION_CONFIG_HASH, HELDOUT_SLICE


def shard_problems(problems: list[dict], k: int) -> list[list[dict]]:
    """Split whole problems into ``k`` contiguous shards (stable order)."""
    if k < 1:
        raise ValueError("k must be >= 1")
    n = len(problems)
    if n == 0:
        return []
    k = min(k, n)
    base, rem = divmod(n, k)
    out = []
    start = 0
    for i in range(k):
        size = base + (1 if i < rem else 0)
        out.append(problems[start : start + size])
        start += size
    return out


def cache_tag_for(gpu_name: str, torch_version: str, vllm_version: str) -> str:
    """Stable hardware/kernel tag for cache separation."""
    payload = f"{gpu_name}|{torch_version}|{vllm_version}|{GENERATION_CONFIG_HASH}"
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def heldout_base_signature(
    problems: list[dict],
    n_samples: int,
    max_tokens: int,
    cache_tag: str,
    k_shards: int,
    model_revision: str | None,
    *,
    engine_config_hash: str | None = None,
    dataset_identity: str | None = None,
) -> tuple[str, dict]:
    """Exact-batch identity for the held-out base arm.

    Extends ``caches.base_batch_signature`` with the held-out slice, shard count
    (batch composition), model revision, engine config, and dataset identity so a
    stale dataset rebuild, a different ``k_shards``, or a different engine cannot
    reuse rows produced for another batch layout.
    """
    extra = {
        "heldout_slice": list(HELDOUT_SLICE),
        "k_shards": int(k_shards),
        "model_revision": model_revision or "unknown",
        "engine_config_hash": engine_config_hash,
        "dataset_identity": dataset_identity,
    }
    return base_batch_signature(
        "heldout",
        [p["id"] for p in problems],
        [p["problem"] for p in problems],
        n_samples,
        max_tokens,
        cache_tag,
        extra=extra,
    )


def heldout_lora_int_id(registry: LoraIdRegistry, adapter_dir: str) -> int:
    """Content-addressed, container-unique vLLM LoRA id for an adapter directory.

    Distinct adapter bytes get distinct ids within one warm container; identical
    bytes reuse one id. This prevents a warm vLLM adapter cache from returning a
    previous edit's numbers under a fixed ``lora_int_id``.
    """
    return registry.id_for_dir(adapter_dir)


def assert_shard_identity(
    result: dict,
    *,
    expected_cache_tag: str,
    expected_adapter_sha256: str | None = None,
    expected_eval_engine_config_hash: str | None = None,
) -> None:
    """Fail when a shard ran under a different GPU/kernel, adapter, or engine config."""
    observed_tag = result.get("cache_tag")
    if observed_tag != expected_cache_tag:
        raise RuntimeError(
            "Held-out cache-tag drift: expected "
            f"{expected_cache_tag!r}, shard reported {observed_tag!r}."
        )
    if expected_adapter_sha256 is not None:
        observed_sha = result.get("adapter_sha256")
        if observed_sha != expected_adapter_sha256:
            raise RuntimeError(
                "Held-out adapter mismatch: expected "
                f"{expected_adapter_sha256!r}, shard reported {observed_sha!r}."
            )
    if expected_eval_engine_config_hash is not None:
        observed_engine = result.get("eval_engine_config_hash")
        if observed_engine != expected_eval_engine_config_hash:
            raise RuntimeError(
                "Held-out eval engine config drift: expected "
                f"{expected_eval_engine_config_hash!r}, shard reported "
                f"{observed_engine!r}."
            )


def cut_at_token_limit(token_count: int, max_tokens: int, finish_reason: str | None) -> bool:
    """Engine/donor ``cut`` definition, shared so both arms label identically.

    An output is cut when it reached the token budget OR vLLM reported ``length``;
    the held-out path previously used only ``finish_reason == "length"``.
    """
    return token_count >= max_tokens or finish_reason == "length"


def reload_with_retry(volume, context: str, attempts: int = 2) -> None:
    """Reload a Modal Volume with a GC retry, so transient errors are not fatal."""
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            volume.reload()
            return
        except Exception as error:  # noqa: BLE001 - retry then surface context
            last_error = error
            print(
                f"[heldout] volume reload failed ({context}) attempt "
                f"{attempt}/{attempts}: {type(error).__name__}: {error}"
            )
            gc.collect()
    raise RuntimeError(
        f"Volume reload failed after {attempts} attempts ({context}): "
        f"{type(last_error).__name__}: {last_error}"
    ) from last_error
