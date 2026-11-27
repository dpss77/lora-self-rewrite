#!/usr/bin/env python3
"""Top-level Modal entrypoint: ``modal run modal_app.py -- --session-id ...``.

Only the Modal wiring lives here; protocol logic is in ``src/lora_self_rewrite``.
No GPU job may be launched without explicit operator approval (see README).
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "src"))

from lora_self_rewrite.constants import (  # noqa: E402
    MAX_GEN_ANSWERS,
    MAX_READ_LENGTH,
    TURN_BUDGET,
    WORKER_TIMEOUT_S,
)
from lora_self_rewrite.engine import QwenSelfRewriteEngine  # noqa: E402,F401
from lora_self_rewrite.modal_wiring import app  # noqa: E402
from lora_self_rewrite.run_config import (  # noqa: E402
    load_run_config,
    resolve_session_config,
    validate_protocol_session,
)
from lora_self_rewrite.session import run_self_rewrite_session  # noqa: E402


@app.local_entrypoint()
def main(
    session_id: str = "sanity_01",
    dataset: str = "math500",
    config: str | None = None,
    max_tokens: int = 4096,
    diag_n: int = 64,
    val_n: int = 64,
    val_samples: int = 2,
    max_turns: int = TURN_BUDGET,
    max_passes: int = 20,
    max_code_runs: int = 200,
    max_gen_answers: int = MAX_GEN_ANSWERS,
    wall_limit_hours: float = 3.0,
    read_max_length: int = MAX_READ_LENGTH,
    deviation: str | None = None,
    preflight: bool = True,
    preflight_sleep: bool = False,
    reuse_diagnosis: bool = True,
    reset_active_lora: bool = True,
    overwrite_session: bool = False,
    worker_timeout_s: int = WORKER_TIMEOUT_S,
) -> None:
    # Fail-fast before spending money.
    if config:
        resolved = resolve_session_config(load_run_config(config))
        session_id = resolved["experiment_id"]
        dataset = resolved["dataset_mode"]
        max_tokens = resolved["max_tokens"]
        diag_n = resolved["diag_n"] if resolved["diag_n"] is not None else diag_n
        val_n = resolved["val_n"] if resolved["val_n"] is not None else val_n
        val_samples = (
            resolved["val_samples"] if resolved["val_samples"] is not None else val_samples
        )
        max_turns = resolved["max_turns"]
        max_passes = resolved["max_passes"]
        max_code_runs = resolved["max_code_runs"]
        max_gen_answers = resolved["max_gen_answers"]
        wall_limit_hours = resolved["wall_limit_hours"]
        read_max_length = resolved["read_max_length"]
    validate_protocol_session(
        max_tokens=max_tokens,
        max_passes=max_passes,
        max_code_runs=max_code_runs,
        max_turns=max_turns,
        wall_limit_hours=wall_limit_hours,
        deviation=deviation,
    )
    if dataset not in ("math500", "deepscaler"):
        raise ValueError(f"--dataset must be 'math500' or 'deepscaler', got {dataset!r}")
    if not (16 <= max_tokens <= 16384):
        raise ValueError(f"--max-tokens must be in [16, 16384], got {max_tokens}")
    if not (1 <= max_gen_answers <= MAX_GEN_ANSWERS):
        raise ValueError(
            f"--max-gen-answers must be in [1, {MAX_GEN_ANSWERS}], got {max_gen_answers}"
        )
    if not (1 <= read_max_length <= MAX_READ_LENGTH):
        raise ValueError(
            f"--read-max-length must be in [1, {MAX_READ_LENGTH}], got {read_max_length}"
        )
    if diag_n < 1 or val_n < 1 or val_samples < 1 or max_turns < 1:
        raise ValueError("diag_n, val_n, val_samples, max_turns must all be >= 1")
    if worker_timeout_s < 60:
        raise ValueError(f"--worker-timeout-s must be >= 60, got {worker_timeout_s}")

    run_self_rewrite_session.remote(
        session_id=session_id,
        dataset_mode=dataset,
        max_tokens=max_tokens,
        diag_n=diag_n,
        val_n=val_n,
        val_samples=val_samples,
        max_turns=max_turns,
        max_passes=max_passes,
        max_code_runs=max_code_runs,
        max_gen_answers=max_gen_answers,
        wall_limit_hours=wall_limit_hours,
        read_max_length=read_max_length,
        config_path=config,
        deviation=deviation,
        run_preflight=preflight,
        preflight_sleep=preflight_sleep,
        reuse_diagnosis=reuse_diagnosis,
        reset_active_lora=reset_active_lora,
        overwrite_session=overwrite_session,
        worker_timeout_s=worker_timeout_s,
    )
