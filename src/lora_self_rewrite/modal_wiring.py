"""Thin Modal wiring: app, image, volume, secrets, and durability helpers.

This is the only module that imports ``modal`` at module scope. Everything that
can be tested on CPU lives elsewhere.
"""

from __future__ import annotations

import gc
import os
import time

import modal

from .constants import (
    JIT_CACHE_DIR,
    MODAL_APP_NAME,
    MODAL_VOLUME_NAME,
    VLLM_CACHE_DIR,
    WORKSPACE_DIR,
)

app = modal.App(MODAL_APP_NAME)

vol = modal.Volume.from_name(MODAL_VOLUME_NAME, create_if_missing=True)

# Optional HF auth: keep it opt-in. An unnamed/missing Secret at decoration time
# raises NotFoundError and breaks an otherwise working deploy.
_SECRETS = (
    [modal.Secret.from_name(os.environ["HF_SECRET_NAME"])]
    if os.environ.get("HF_SECRET_NAME")
    else []
)

# Pin the empirically verified stack. Pins are the load-bearing part; see
# docs/03-protocol/INVARIANTS.md. JIT caches stay on container-local disk.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "vllm==0.34.0",
        "transformers==5.22.2",
        "peft==0.22.0",
        "accelerate>=1.0.0",
        "safetensors>=0.4.5",
        # Grading is INVARIANT 11; pin the resolved version so a resolver change
        # between runs cannot silently alter every accuracy label.
        "math-verify==0.9.0",
        "datasets",
        "huggingface_hub",
    )
    .env(
        {
            "HF_HOME": f"{WORKSPACE_DIR}/hf_cache",
            "PYTHONUNBUFFERED": "1",
            "TRITON_CACHE_DIR": JIT_CACHE_DIR,
            "TORCHINDUCTOR_CACHE_DIR": JIT_CACHE_DIR,
            "VLLM_CACHE_ROOT": VLLM_CACHE_DIR,
        }
    )
    .add_local_python_source("lora_self_rewrite")
)


def safe_commit() -> None:
    """Commit the Volume, downgrading transport hiccups to a warning."""
    try:
        vol.commit()
    except Exception as e:  # noqa: BLE001 - best-effort durability, never fatal
        print(f"[volume] commit failed (non-fatal): {type(e).__name__}: {e}")


def safe_reload(context: str) -> None:
    """Reload the Volume with one GC/retry, then fail with actionable context."""
    try:
        vol.reload()
        return
    except Exception as first:  # noqa: BLE001
        print(
            f"[volume] reload failed ({context}): "
            f"{type(first).__name__}: {first}; retrying after gc"
        )
    gc.collect()
    time.sleep(0.5)
    try:
        vol.reload()
    except Exception as second:  # noqa: BLE001
        raise RuntimeError(
            f"Volume reload failed after gc retry ({context}): {type(second).__name__}: {second}"
        ) from second
