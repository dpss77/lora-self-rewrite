from __future__ import annotations

# Mechanically ported from the donor `modal_app.py` (SHA-256
# fe71f74b0cebe933c155e03f59941c62ce402d92ca31ac2ce4e038c9b23dcbc2).
# Ported for structural parity; GPU behaviour must be re-validated before use.
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import statistics

from .constants import (
    GENERATION_CONFIG_HASH,
    GENERATION_SAMPLING,
    METRICS_VERSION,
    MODEL_ID,
    MODEL_REVISION,
)


class _LimitedBuffer(io.StringIO):
    """StringIO that stops growing past ``limit`` characters.

    A model-written ``while True: print(...)`` would otherwise materialise unbounded
    output in the orchestrator's 16 GiB heap before the post-hoc slice.
    """

    def __init__(self, limit: int):
        super().__init__()
        self.limit = limit
        self.truncated = False

    def write(self, s: str) -> int:
        remaining = self.limit - self.tell()
        if remaining <= 0:
            self.truncated = True
            return len(s)
        if len(s) > remaining:
            self.truncated = True
            super().write(s[:remaining])
        else:
            super().write(s)
        return len(s)


# ---------------------------------------------------------------------------
# Stage-0 instrumentation: environment fingerprint + phase metrics
# ---------------------------------------------------------------------------
def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _sha256_file(path: str, chunk_size: int = 1 << 20) -> str | None:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def get_model_revision() -> str | None:
    """Return the protocol-pinned HF snapshot revision.

    The cache previously read the local snapshot basename, which could drift with a
    fresh download. The pinned revision is the single source of truth for caches.
    """
    return MODEL_REVISION


def _fingerprint(torch_mod, include_gpu: bool) -> dict:
    """Version/device/file identity for reproducibility and cache invalidation."""
    fp = {
        "metrics_version": METRICS_VERSION,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "torch": getattr(torch_mod, "__version__", None),
        "torch_cuda": getattr(getattr(torch_mod, "version", None), "cuda", None),
        "triton": _package_version("triton"),
        "vllm": _package_version("vllm"),
        "transformers": _package_version("transformers"),
        "peft": _package_version("peft"),
        "accelerate": _package_version("accelerate"),
        "model_id": MODEL_ID,
        "model_revision": get_model_revision(),
        "model_dtype": "bfloat16",
        "generation_config_hash": GENERATION_CONFIG_HASH,
        "generation_sampling": GENERATION_SAMPLING,
        "file_sha256": _sha256_file(os.path.abspath(__file__)),
        "jit_cache_dir": os.environ.get("TRITON_CACHE_DIR"),
        "vllm_cache_dir": os.environ.get("VLLM_CACHE_ROOT"),
    }
    if include_gpu:
        try:
            if torch_mod.cuda.is_available():
                props = torch_mod.cuda.get_device_properties(0)
                fp.update(
                    {
                        "gpu": torch_mod.cuda.get_device_name(0),
                        "gpu_capability": list(torch_mod.cuda.get_device_capability(0)),
                        "gpu_total_mib": props.total_memory // (1024**2),
                    }
                )
            else:
                fp["gpu"] = None
        except Exception as e:  # noqa: BLE001
            fp["gpu"] = f"unavailable: {type(e).__name__}: {e}"
    return fp


# Fingerprint keys that scope the hardware/kernel cache namespace. Keep this list
# and ``cache_schema`` stable: changing either invalidates every Volume cache.
CACHE_FINGERPRINT_FIELDS = (
    "gpu",
    "gpu_capability",
    "torch",
    "torch_cuda",
    "triton",
    "vllm",
    "transformers",
    "peft",
    "model_id",
    "model_revision",
    "model_dtype",
    "generation_config_hash",
)


def cache_fingerprint_from_env(gpu_fp: dict, cache_schema: int) -> dict:
    """Select the cache-scoping keys from a full environment fingerprint."""
    cache_fp = {key: gpu_fp.get(key) for key in CACHE_FINGERPRINT_FIELDS}
    cache_fp["cache_schema"] = cache_schema
    return cache_fp


def cache_tag_from_fingerprint(cache_fp: dict) -> str:
    """Stable 12-char cache tag for an already-selected fingerprint dict."""
    return hashlib.sha256(json.dumps(cache_fp, sort_keys=True, default=str).encode()).hexdigest()[
        :12
    ]


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round((pct / 100.0) * (len(ordered) - 1))))
    return ordered[idx]


def _summarize_metrics(metrics: dict) -> dict:
    """Aggregate Stage-0 phase/cost metrics; all values are derived from measured events."""
    turns = metrics.get("turns", [])
    code_runs = list(metrics.get("code_run_s", []))
    commits = list(metrics.get("commit_s", []))
    reads = list(metrics.get("read_fixed_overhead_s", []))
    total_gpu = sum(float(t.get("gpu_s") or 0.0) for t in turns)
    total_cpu = sum(float(t.get("cpu_s") or 0.0) for t in turns)
    loop_wall = sum(float(t.get("wall_s") or 0.0) for t in turns)
    gpu_fraction = (total_gpu / loop_wall) if loop_wall > 0 else None
    phases = metrics.get("phases", {})
    session_gpu = (
        float(phases.get("engine_ready_s") or 0.0)
        + float(phases.get("preflight_s") or 0.0)
        + float(phases.get("diagnosis_s") or 0.0)
        + total_gpu
    )
    session_wall = float(metrics.get("session_wall_s") or 0.0)
    session_gpu_fraction = (session_gpu / session_wall) if session_wall > 0 else None
    read_count = int(metrics.get("counters", {}).get("reads", 0))
    generate_calls = metrics.get("generate_calls", [])
    return {
        "metrics_version": METRICS_VERSION,
        "session_wall_s": round(session_wall, 3),
        "phases": phases,
        "turns": len(turns),
        "loop_wall_s": round(loop_wall, 3),
        "loop_gpu_s": round(total_gpu, 3),
        "loop_cpu_s": round(total_cpu, 3),
        "loop_gpu_fraction": round(gpu_fraction, 4) if gpu_fraction is not None else None,
        "session_gpu_call_s": round(session_gpu, 3),
        "session_gpu_fraction": (
            round(session_gpu_fraction, 4) if session_gpu_fraction is not None else None
        ),
        "code_runs": len(code_runs),
        "code_run_s_p50": round(_percentile(code_runs, 50) or 0.0, 3),
        "code_run_s_p95": round(_percentile(code_runs, 95) or 0.0, 3),
        "commit_s_mean": round(statistics.mean(commits), 4) if commits else None,
        "commit_s_p95": round(_percentile(commits, 95) or 0.0, 4) if commits else None,
        "read_count": read_count,
        "read_fixed_overhead_s_mean": (round(statistics.mean(reads), 3) if reads else None),
        "read_fixed_overhead_s_total": round(sum(reads), 3) if reads else 0.0,
        "generate_calls": len(generate_calls),
        "generate_answers": int(metrics.get("counters", {}).get("generated_answers", 0)),
        "generated_tokens": int(metrics.get("counters", {}).get("generated_tokens", 0)),
        "counters": metrics.get("counters", {}),
        "gpu_container_billed_s_est": metrics.get("gpu_container_billed_s_est"),
        "gpu_container_billed_s_est_is_lower_bound": metrics.get(
            "gpu_container_billed_s_est_is_lower_bound"
        ),
    }
