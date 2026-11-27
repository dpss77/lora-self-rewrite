from __future__ import annotations

# Mechanically ported from the donor `modal_app.py` (SHA-256
# fe71f74b0cebe933c155e03f59941c62ce402d92ca31ac2ce4e038c9b23dcbc2).
# Ported for structural parity; GPU behaviour must be re-validated before use.
import contextlib
import gc
import json
import math
import os
import re
import shutil
import signal
import time
import traceback

from .adapter import load_and_validate_adapter, write_submit_provenance
from .caches import (
    _base_batch_signature_alias as _base_batch_signature,
)
from .caches import (
    _load_base_batch_cache_alias as _load_base_batch_cache,
)
from .caches import (
    _save_base_batch_cache_alias as _save_base_batch_cache,
)
from .caches import (
    content_addressed_lora_key,
    diagnosis_identity,
    load_diagnosis_cache,
    load_validation_base_cache,
    save_diagnosis_cache,
    save_validation_base_cache,
)
from .constants import (
    CODE_STDOUT_LIMIT,
    CODE_TIMEOUT_S,
    COMPACT_TOKEN_LIMIT,
    EDITOR_MAX_TOKENS,
    MAX_GEN_ANSWERS,
    MAX_MODEL_LEN,
    MAX_READ_BYTES,
    MAX_READ_LAYERS,
    MAX_READ_LENGTH,
    MAX_READ_TEXTS,
    METRICS_VERSION,
    MODEL_ID,
    MODEL_REVISION,
    SCALEDOWN_WINDOW_S,
    WORKER_TIMEOUT_S,
    validate_compaction_budget,
)
from .datasets import dataset_identity, load_session_datasets, require_pinned_sources
from .engine import QwenSelfRewriteEngine
from .engine_identity import engine_config_hash, require_pinned_revision
from .instrumentation import (
    _fingerprint,
    _LimitedBuffer,
    _summarize_metrics,
    cache_fingerprint_from_env,
    cache_tag_from_fingerprint,
)
from .modal_wiring import _SECRETS, WORKSPACE_DIR, app, image, vol
from .modal_wiring import safe_commit as _safe_commit
from .modal_wiring import safe_reload as _safe_reload
from .model_layout import init_empty_lora
from .paths import session_dir as session_dir_for
from .paths import validate_session_id
from .protocol_shared import (
    apply_chat as _apply_chat,
)
from .protocol_shared import (
    grade_answer,
    is_looping_answer,
    strip_thinking,
)
from .resume import resume_plan
from .run_config import (
    cap_read_length,
    load_run_config,
    resolve_session_config,
    validate_protocol_session,
)
from .sandbox import SAFE_BUILTINS, validate_sandbox_code
from .sandbox_runtime import bind_sandbox_helpers
from .stats import compute_paired_stats


@app.function(
    image=image,
    cpu=4.0,
    memory=16384,
    volumes={WORKSPACE_DIR: vol},
    secrets=_SECRETS,
    timeout=21600,  # orchestrator must outlive a late validation gate
    retries=0,
)
def run_self_rewrite_session(
    session_id: str = "run_01",
    dataset_mode: str = "math500",
    max_tokens: int = 4096,
    diag_n: int = 64,
    val_n: int = 64,
    val_samples: int = 2,
    max_turns: int = 160,
    max_passes: int = 20,
    max_code_runs: int = 200,
    max_gen_answers: int = 128,
    wall_limit_hours: float = 3.0,
    run_preflight: bool = True,
    preflight_sleep: bool = False,
    reuse_diagnosis: bool = True,
    reset_active_lora: bool = True,
    overwrite_session: bool = False,
    worker_timeout_s: int = WORKER_TIMEOUT_S,
    config_path: str | None = None,
    deviation: str | None = None,
    read_max_length: int = MAX_READ_LENGTH,
):
    import numpy as np
    import torch
    from transformers import AutoTokenizer

    # Config is the explicit source of truth when supplied. Every key consumed by
    # the run is resolved and recorded below; malformed/unknown keys are rejected.
    loaded_config = None
    if config_path:
        loaded_config = load_run_config(config_path)
        resolved_config = resolve_session_config(loaded_config)
        if resolved_config["model_id"] != MODEL_ID:
            raise ValueError(
                f"Config model_id {resolved_config['model_id']!r} does not match "
                f"the protocol model {MODEL_ID!r}."
            )
        if resolved_config["model_revision"] != MODEL_REVISION:
            raise ValueError(
                "Config model_revision must equal the pinned MODEL_REVISION "
                f"{MODEL_REVISION!r}, got {resolved_config['model_revision']!r}."
            )
        session_id = resolved_config["experiment_id"]
        dataset_mode = resolved_config["dataset_mode"]
        max_tokens = resolved_config["max_tokens"]
        max_passes = resolved_config["max_passes"]
        max_code_runs = resolved_config["max_code_runs"]
        max_turns = resolved_config["max_turns"]
        wall_limit_hours = resolved_config["wall_limit_hours"]
        max_gen_answers = resolved_config["max_gen_answers"]
        read_max_length = resolved_config["read_max_length"]
        if resolved_config.get("diag_n") is not None:
            diag_n = int(resolved_config["diag_n"])
        if resolved_config.get("val_n") is not None:
            val_n = int(resolved_config["val_n"])
        if resolved_config.get("val_samples") is not None:
            val_samples = int(resolved_config["val_samples"])
    else:
        resolved_config = {
            "session_id": session_id,
            "dataset_mode": dataset_mode,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "max_tokens": max_tokens,
            "diag_n": diag_n,
            "val_n": val_n,
            "val_samples": val_samples,
            "max_turns": max_turns,
            "max_passes": max_passes,
            "max_code_runs": max_code_runs,
            "max_gen_answers": max_gen_answers,
            "wall_limit_hours": wall_limit_hours,
            "read_max_length": read_max_length,
        }
    protocol_deviations = validate_protocol_session(
        max_tokens=max_tokens,
        max_passes=max_passes,
        max_code_runs=max_code_runs,
        max_turns=max_turns,
        wall_limit_hours=wall_limit_hours,
        deviation=deviation,
    )
    effective_read_max_length = cap_read_length(read_max_length)

    # ---- Stage-0 measurement (no model-behavior change) -------------------------
    session_t0 = time.perf_counter()
    metrics: dict = {
        "metrics_version": METRICS_VERSION,
        "session_id": session_id,
        "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": {
            "session_id": session_id,
            "dataset_mode": dataset_mode,
            "max_tokens": max_tokens,
            "diag_n": diag_n,
            "val_n": val_n,
            "val_samples": val_samples,
            "max_turns": max_turns,
            "max_passes": max_passes,
            "max_code_runs": max_code_runs,
            "max_gen_answers": max_gen_answers,
            "wall_limit_hours": wall_limit_hours,
            "read_max_length": effective_read_max_length,
            "run_preflight": run_preflight,
            "preflight_sleep": preflight_sleep,
            "reuse_diagnosis": reuse_diagnosis,
            "reset_active_lora": reset_active_lora,
            "overwrite_session": overwrite_session,
            "worker_timeout_s": worker_timeout_s,
            "config_path": config_path,
            "deviation": deviation,
            "protocol_deviations": protocol_deviations,
        },
        "resolved_config": resolved_config,
        "loaded_config": loaded_config,
        "orchestrator_env": None,
        "gpu_env": None,
        "phases": {},
        "turns": [],
        "code_run_s": [],
        "commit_s": [],
        "read_fixed_overhead_s": [],
        "generate_calls": [],
        "counters": {
            "reads": 0,
            "generated_answers": 0,
            "generated_tokens": 0,
            "diagnosis_cache_hits": 0,
            "validation_cache_hits": 0,
            "validation_cache_misses": 0,
            "base_cache_hits": 0,
            "base_cache_misses": 0,
        },
        "diagnosis": {},
        "validation": {},
        "preflight": {},
        "gpu_container_billed_s_est": None,
    }

    def _gpu_call(fn, *args, **kwargs):
        """Call a Modal GPU method, recording wall time and last-call timestamp."""
        t0 = time.perf_counter()
        if metrics.get("first_gpu_call_start_t") is None:
            metrics["first_gpu_call_start_t"] = t0
        try:
            result = fn(*args, **kwargs)
        finally:
            metrics["last_gpu_call_end_t"] = time.perf_counter()
        return result, round(time.perf_counter() - t0, 3)

    # ---- FAIL-FAST LAYER 2: CPU checks before allocating any GPU container ----
    if not (16 <= max_tokens <= 16384):
        raise ValueError(f"max_tokens must be in [16, 16384], got {max_tokens}.")
    if min(diag_n, val_n, val_samples, max_turns, max_passes, max_code_runs) < 1:
        raise ValueError(
            "diag_n/val_n/val_samples/max_turns/max_passes/max_code_runs must be >= 1."
        )
    if not (1 <= max_gen_answers <= MAX_GEN_ANSWERS):
        raise ValueError(
            f"max_gen_answers must be in [1, {MAX_GEN_ANSWERS}], got {max_gen_answers}."
        )
    if not (1 <= effective_read_max_length <= MAX_READ_LENGTH):
        raise ValueError(
            f"read_max_length must be in [1, {MAX_READ_LENGTH}], got {effective_read_max_length}."
        )
    if wall_limit_hours <= 0 or worker_timeout_s < 60:
        raise ValueError("wall_limit_hours>0 and worker_timeout_s>=60 required.")
    # Pure constant check before any GPU container exists: a bad compaction/context
    # pairing must fail at $0, not as a billed engine boot-loop.
    validate_compaction_budget(COMPACT_TOKEN_LIMIT, MAX_MODEL_LEN)

    # Grader invariants: a closed </think> with a matching \boxed{} must pass, and an
    # unclosed <think> must fail. If either breaks, every accuracy number is meaningless.
    assert grade_answer("<think>x</think>\\boxed{42}", "42") is True, "Grader self-test failed!"
    assert grade_answer("<think>x\\boxed{42}", "42") is False, "Grader </think> gate failed!"

    validate_session_id(session_id)
    session_dir = session_dir_for(session_id, workspace_dir=f"{WORKSPACE_DIR}")
    work_dir = f"{session_dir}/WORK"
    reads_dir = f"{work_dir}/reads"
    candidates_dir = f"{work_dir}/candidates"
    submitted_lora_dir = f"{work_dir}/submitted_lora"
    lora_dir = f"{session_dir}/active_lora"
    os.makedirs(reads_dir, exist_ok=True)
    os.makedirs(candidates_dir, exist_ok=True)

    # Session hygiene: reusing a session_id must not silently mix old candidates,
    # pass files, notes, or transcript into a fresh run. Keep the diagnosis census and
    # the shared /workspace/cache (hardware-tagged) intact.
    if overwrite_session:
        reset_active_lora = True
    top_level = set(os.listdir(work_dir))
    plan = resume_plan(
        work_dir,
        overwrite_session=overwrite_session,
        reset_active_lora=reset_active_lora,
    )
    resume_scan = plan["scan"] if plan["action"] == "resume" else None
    if plan["action"] == "overwrite":
        # Overwrite means a truly fresh session: the active adapter and census must
        # not survive, even if the caller forgot reset_active_lora.
        reset_active_lora = True
        print(f"[session] overwriting stale WORK state: {plan['scan']['dirty_reasons']}")
        shutil.rmtree(lora_dir, ignore_errors=True)
        shutil.rmtree(candidates_dir, ignore_errors=True)
        shutil.rmtree(reads_dir, ignore_errors=True)
        shutil.rmtree(submitted_lora_dir, ignore_errors=True)
        os.makedirs(candidates_dir, exist_ok=True)
        os.makedirs(reads_dir, exist_ok=True)
        for stale_name in top_level:
            if (
                stale_name.startswith("pass_") and stale_name.endswith((".json", ".inflight"))
            ) or stale_name in {
                "notes.md",
                "transcript.jsonl",
                "transcript.json",
                "metrics.json",
                "resume_state.json",
                "census.jsonl",
                "census.jsonl.meta.json",
            }:
                try:
                    os.remove(os.path.join(work_dir, stale_name))
                except OSError:
                    pass
    elif plan["action"] == "refuse":
        raise ValueError(
            f"Session '{session_id}' already has WORK state "
            f"{plan['scan']['dirty_reasons']}. "
            "Use a fresh --session-id, pass --overwrite-session to wipe it, or use "
            "--no-reset-active-lora to resume deliberately."
        )
    elif resume_scan is not None:
        adapter_model = os.path.join(lora_dir, "adapter_model.safetensors")
        adapter_config = os.path.join(lora_dir, "adapter_config.json")
        if not (os.path.exists(adapter_model) and os.path.exists(adapter_config)):
            raise ValueError(
                f"Cannot resume session '{session_id}': active_lora is missing "
                f"({lora_dir}). Refusing to silently restart from a fresh zero "
                "adapter; pass --overwrite-session for a deliberate new run."
            )
        print(f"[session] resuming with existing WORK state: {plan['scan']['dirty_reasons']}")

    module_registry = init_empty_lora(
        lora_dir, rank=4, reset=reset_active_lora, model_revision=MODEL_REVISION
    )
    helpers = bind_sandbox_helpers(lora_dir, work_dir, module_registry)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)

    # Fail-fast on the chat template BEFORE booting the A100. Qwen3.5's Jinja template
    # raises `TemplateError: No user query found in messages.` on a system-only history,
    # which previously only surfaced on the GPU at Turn 1 after a 7-minute diagnosis.
    # Validating the exact (system + user) shape here turns a $0.40 crash into a $0 one.
    tokenizer.apply_chat_template(
        [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}],
        tokenize=False,
        add_generation_prompt=True,
    )

    practice_set, reserved_check_set = load_session_datasets(dataset_mode)
    # `datasets` can hold Arrow mmaps under HF_HOME on the Volume; release them before
    # any Volume reload/probe so they cannot masquerade as a sandbox-tensor leak.
    gc.collect()
    if len(practice_set) < diag_n:
        raise ValueError(f"practice_set ({len(practice_set)}) < diag_n ({diag_n}).")
    if len(reserved_check_set) < val_n:
        raise ValueError(f"reserved_check_set ({len(reserved_check_set)}) < val_n ({val_n}).")
    dataset_meta = dataset_identity(dataset_mode, practice_set, reserved_check_set)
    if not deviation:
        require_pinned_revision(MODEL_REVISION)
        require_pinned_sources(dataset_mode)
    metrics["dataset"] = dataset_meta
    dataset_identity_hash = dataset_meta["identity"]

    n_layers = max(info["layer"] for info in module_registry.values()) + 1
    hidden_dim = max(info["shape"][0] for info in module_registry.values())
    mlp_dim = max(info["shape"][1] for info in module_registry.values())

    # Fingerprint the CPU/orchestrator side after the model cache exists. This must never
    # be a fast-fail path: a metadata/__file__ anomaly should not abort before metrics exist.
    try:
        metrics["orchestrator_env"] = _fingerprint(torch, include_gpu=False)
    except Exception as fp_err:  # noqa: BLE001
        print(
            f"[env] orchestrator fingerprint failed (non-fatal): {type(fp_err).__name__}: {fp_err}"
        )
        metrics["orchestrator_env"] = {}

    engine = QwenSelfRewriteEngine()
    env_res, engine_ready_s = _gpu_call(engine.get_environment.remote)
    metrics["gpu_env"] = env_res
    metrics["phases"]["engine_ready_s"] = engine_ready_s
    print(
        f"[env] engine ready in {engine_ready_s:.1f}s; "
        f"gpu={env_res.get('fingerprint', {}).get('gpu')} "
        f"startup={env_res.get('startup')}"
    )

    # Hardware/kernel-scoped cache namespace. This is a hard scientific requirement:
    # an A100 base row paired with an H200 edited row would confound hardware with the
    # edit. Legacy caches without provenance are accepted only on A100.
    gpu_fp = env_res.get("fingerprint", {})
    current_gpu = gpu_fp.get("gpu") or ""
    cache_fp = cache_fingerprint_from_env(gpu_fp, METRICS_VERSION)
    cache_tag = cache_tag_from_fingerprint(cache_fp)
    metrics["cache_tag"] = cache_tag
    metrics["cache_fingerprint"] = cache_fp
    print(f"[cache] hardware/kernel cache tag: {cache_tag} (gpu={current_gpu})")

    engine_cfg = env_res.get("engine_config") or {}
    engine_cfg_hash = env_res.get("engine_config_hash")
    if not engine_cfg_hash and engine_cfg:
        engine_cfg_hash = engine_config_hash(engine_cfg)
    if not engine_cfg_hash:
        if not deviation:
            raise RuntimeError("Engine config identity is missing; refusing a cacheable run.")
        engine_cfg_hash = "unknown-deviation"
    metrics["engine_config"] = engine_cfg
    metrics["engine_config_hash"] = engine_cfg_hash
    model_revision = gpu_fp.get("model_revision") or MODEL_REVISION
    if not deviation:
        require_pinned_revision(model_revision)
    metrics["model_revision"] = model_revision

    base_cache_dir = f"{WORKSPACE_DIR}/cache/base/{dataset_mode}"

    def _base_cache_lookup(pids: list[int], n_samples: int, max_tokens: int):
        """Return (path, meta, cached_rows) for an exact ordered base batch."""
        problems = [practice_set[i]["problem"] for i in pids]
        signature, meta = _base_batch_signature(
            dataset_mode,
            pids,
            problems,
            n_samples,
            max_tokens,
            cache_tag,
            extra={
                "engine_config_hash": engine_cfg_hash,
                "model_revision": model_revision,
                "dataset_identity": dataset_identity_hash,
            },
        )
        path = os.path.join(base_cache_dir, f"base_{signature}.json")
        expected = [
            {
                "id": practice_set[i]["id"],
                "sample_idx": s_idx,
                "problem": practice_set[i]["problem"],
                "answer": practice_set[i]["answer"],
            }
            for i in pids
            for s_idx in range(n_samples)
        ]
        return path, meta, _load_base_batch_cache(path, meta, expected)

    # ---- FAIL-FAST LAYER 4: 1s GPU LoRA (and optional sleep/wake) preflight ----
    if run_preflight:
        probe_mod = module_registry["layers.8.mlp.down_proj"]
        pf, preflight_s = _gpu_call(
            engine.preflight_check.remote,
            probe_mod["peft_prefix"],
            probe_mod["shape"],
            test_sleep=preflight_sleep,
        )
        metrics["preflight"] = pf
        metrics["phases"]["preflight_s"] = preflight_s
        print(f"GPU preflight passed: {pf}")

    # 1. Starting Diagnosis (not counted toward the 3.0h session clock per Sec. E.9).
    # Reusing a cached census avoids re-running ~7 minutes of GPU generation on every
    # restart. The cache key binds content, ordered problem+answer hashes, token
    # budget, label schema, grader identity, and cache schema; every load re-derives
    # correct/cut/loop from the stored text instead of trusting persisted labels.
    diag_items = practice_set[:diag_n]
    census_path = os.path.join(work_dir, "census.jsonl")
    shared_cache_path = (
        f"{WORKSPACE_DIR}/cache/diag_{dataset_mode}_{diag_n}_{max_tokens}_{cache_tag}.jsonl"
    )
    legacy_shared_cache_path = (
        f"{WORKSPACE_DIR}/cache/diag_{dataset_mode}_{diag_n}_{max_tokens}.jsonl"
    )
    _, diag_identity = diagnosis_identity(dataset_mode, diag_items, max_tokens)
    census_meta = {
        **diag_identity,
        "cache_tag": cache_tag,
        "cache_fingerprint": cache_fp,
        "diag_n": diag_n,
        "metrics_version": METRICS_VERSION,
    }
    expected_diag = [
        {"id": item["id"], "problem": item["problem"], "answer": item["answer"]}
        for item in diag_items
    ]

    def _load_census(path: str):
        # Hardware provenance check: a tagged cache must match this container. An
        # untagged legacy cache is still usable through the content-match migration
        # path (re-grade only; generations are never recomputed), but it is never
        # rewritten in place.
        meta_path = f"{path}.meta.json"
        stored_meta = None
        if os.path.exists(meta_path):
            try:
                with open(meta_path) as f:
                    stored_meta = json.load(f)
            except (OSError, json.JSONDecodeError):
                stored_meta = None
        if isinstance(stored_meta, dict) and stored_meta.get("cache_tag") not in (
            None,
            cache_tag,
        ):
            print(f"[cache] {path} hardware/kernel tag mismatch; ignoring.")
            return None
        if not isinstance(stored_meta, dict) or not stored_meta.get("cache_tag"):
            print(
                f"[cache] {path} has no hardware provenance; "
                "using content-match compatibility path (re-grade only)."
            )
        result = load_diagnosis_cache(
            path,
            census_meta,
            expected_diag,
            grade_answer,
            allow_legacy_text=True,
        )
        if result is None:
            print(f"[cache] {path} diagnosis identity/content mismatch; ignoring.")
            return None
        rows, status = result
        print(f"[cache] {path} diagnosis {status} ({len(rows)} rows).")
        return rows

    diag_graded = None
    diagnosis_fresh = False
    metrics["diagnosis"].update({"diag_n": diag_n, "max_tokens": max_tokens, "cache_hit": False})
    if reuse_diagnosis:
        for cand_census in (
            census_path,
            shared_cache_path,
            legacy_shared_cache_path,
        ):
            if os.path.exists(cand_census):
                rows = _load_census(cand_census)
                if rows is not None:
                    diag_graded = rows
                    metrics["diagnosis"]["cache_hit"] = True
                    metrics["diagnosis"]["cache_path"] = cand_census
                    metrics["counters"]["diagnosis_cache_hits"] += 1
                    metrics["phases"]["diagnosis_s"] = 0.0
                    print(f"Reused pre-computed diagnosis ({len(rows)} items) from {cand_census}.")
                    break

    if diag_graded is None:
        diagnosis_fresh = True
        print(f"Running starting diagnosis on {diag_n} practice problems ({dataset_mode})...")
        diag_result, diagnosis_s = _gpu_call(
            engine.generate_batch.remote,
            practice_set[:diag_n],
            lora_specs={},
            include_base=True,
            n_samples=1,
            max_tokens=max_tokens,
        )
        metrics["phases"]["diagnosis_s"] = diagnosis_s
        metrics["diagnosis"]["gpu_s"] = diagnosis_s
        metrics["diagnosis"]["timings"] = diag_result.get("timings", {})
        diag_raw = diag_result["base"]
        if not diag_raw or all(r["tokens"] == 0 for r in diag_raw):
            raise RuntimeError("FAIL-FAST: Starting diagnosis produced 0 generated tokens.")
        diag_graded = []
        for item in diag_raw:
            cut = int(item.get("tokens") or 0) >= max_tokens or (
                item.get("finish_reason") == "length"
            )
            diag_graded.append(
                {
                    **item,
                    "cut": cut,
                    "loop": is_looping_answer(item["text"], cut),
                    "correct": grade_answer(item["text"], item["answer"]),
                }
            )
        diag_timings = diag_result.get("timings", {})
        metrics["counters"]["generated_answers"] += int(diag_timings.get("n_answers", 0))
        metrics["counters"]["generated_tokens"] += int(diag_timings.get("generated_tokens", 0))

    # Always refresh the per-session census (it is derived state for resume). The
    # shared cross-session cache is written only after fresh generation: a reused
    # per-session row set must never overwrite the shared cache, which would let one
    # session's stale/relabelled rows poison every future session.
    save_diagnosis_cache(census_path, census_meta, diag_graded)
    if diagnosis_fresh:
        os.makedirs(os.path.dirname(shared_cache_path), exist_ok=True)
        save_diagnosis_cache(shared_cache_path, census_meta, diag_graded)
    _safe_commit()

    acc = sum(x["correct"] for x in diag_graded) / len(diag_graded)
    cuts = sum(x["cut"] for x in diag_graded)
    loops = sum(x["loop"] for x in diag_graded)
    finished = [x for x in diag_graded if not x["cut"]]
    acc_finished = sum(x["correct"] for x in finished) / len(finished) if finished else 0.0
    print(
        f"Diagnosis complete: acc={acc:.1%}, cut={cuts}/{diag_n}, "
        f"looping_cut={loops}/{cuts}, acc_given_finished={acc_finished:.1%}"
    )

    # The wall-clock budget is cumulative across preemptions. Offset the start time
    # by the wall time already spent so a resume cannot reset (or extend) the limit.
    elapsed_wall_prior_s = (
        float(resume_scan.get("elapsed_wall_prior_s") or 0.0) if resume_scan is not None else 0.0
    )
    turns_used_prior = (
        int(resume_scan.get("turns_used_prior") or len(resume_scan.get("turns") or []))
        if resume_scan is not None
        else 0
    )
    start_time = time.time() - elapsed_wall_prior_s

    short_modules = sorted([k for k in module_registry if k.startswith("layers.")])
    shapes_map = {k: module_registry[k]["shape"] for k in short_modules}

    # Sandboxed namespace: restricted builtins + a guarded __import__. State persists
    # across code runs on purpose (the model builds on prior cells), so it is a *single*
    # dict reused for every exec.
    sandbox_namespace = {
        "torch": torch,
        "np": np,
        "math": math,
        "json": json,
        "tokenizer": tokenizer,
        "available_modules": short_modules,
        "module_shapes": shapes_map,
        "diagnosis": diag_graded,
        "practice_set": practice_set,
        "last_read": {},
        **helpers,
        "__builtins__": SAFE_BUILTINS,
        "__name__": "sandbox",
    }

    # Regression probe for the v03 Turn-20 failure: deliberately retain BOTH
    # volume-backed helper outputs across a reload. With the clone fixes this passes;
    # without them it fails in seconds instead of after ~20 turns.
    t_probe0 = time.perf_counter()
    probe_name = (
        "layers.8.mlp.down_proj"
        if "layers.8.mlp.down_proj" in module_registry
        else short_modules[0]
    )
    sandbox_namespace["__probe_base__"] = helpers["load_base_weight"](probe_name)
    sandbox_namespace["__probe_lora__"] = helpers["load_lora_tensors"]()
    gc.collect()
    _safe_reload("orchestrator:volume-handle-probe")
    sandbox_namespace.pop("__probe_base__", None)
    sandbox_namespace.pop("__probe_lora__", None)
    gc.collect()
    _safe_reload("orchestrator:volume-handle-probe-cleanup")
    metrics["phases"]["volume_probe_s"] = round(time.perf_counter() - t_probe0, 3)

    # Scrub model-visible secrets before any model-authored code runs. HF auth was only
    # needed on the CPU for snapshot/tokenizer/dataset setup above and on the GPU via its
    # own Secret; leaving tokens in os.environ here would let sandbox code print them.
    for _secret_key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "GITHUB_TOKEN"):
        os.environ.pop(_secret_key, None)

    def _count_prompt_tokens(msgs: list[dict]) -> int:
        return len(tokenizer.encode(_apply_chat(tokenizer, msgs)))

    system_prompt = f"""You are Qwen3.5-27B (64-layer hybrid Gated DeltaNet + full attention) editing your own weights via a zero-initialized rank-4 LoRA adapter on output projections (self_attn.o_proj, linear_attn.out_proj, mlp.down_proj).
The session itself runs on your unedited base weights; your goal is to improve competition mathematics accuracy by diagnosing why you fail and writing targeted edits to lora_A and lora_B. Nothing will train you.
Architecture schedule: self_attn.o_proj exists only on layers 3, 7, 11, ..., 63 (every 4th layer); the other 48 layers have linear_attn.out_proj; mlp.down_proj exists on all 64 layers.

HOW YOU WILL BE JUDGED:
Unedited vs. edited weights are evaluated on unseen problems with a {max_tokens}-token answer budget. Only the final \\boxed{{}} answer after </think> is graded; an answer whose thinking is still open when the token budget runs out earns no credit.
Out-of-scope edits are disallowed (embeddings, final norm, LM head, whole-tensor scalar scaling, or forcing yourself to stop thinking early regardless of the problem).

STARTING DIAGNOSIS ({diag_n} practice problems, {max_tokens}-token limit):
- Accuracy: {acc:.1%}
- Ran out of tokens: {cuts}/{diag_n} ({cuts / diag_n:.1%}), of which {loops}/{cuts} end in repetitive looping text (zlib <15% on last 6,000 chars)
- Accuracy when finished before token limit: {acc_finished:.1%} ({len(finished)} finished)
Saved in WORK/census.jsonl (access via `diagnosis` or `load_census()`).

TOOLS (every turn: brief reasoning, then EXACTLY ONE ```python``` block OR ONE ```json``` block):
1. ```python``` block (stateful CPU Python, gradients disabled, max {max_code_runs} runs, {CODE_TIMEOUT_S}s limit, {CODE_STDOUT_LIMIT} char stdout):
   - `available_modules`, `module_shapes`, `tokenizer`, `practice_set`, `diagnosis`, `load_census()`, `load_pass(k)`
   - `diagnosis[i]` / `load_census()[i]` keys: `id, sample_idx, problem, answer, text, tokens, finish_reason, cut, loop, correct` (`finish_reason` may be None in legacy caches)
   - `load_pass(k)` -> `{{"base": [rows] or None, "candidates": {{name: [rows]}}}}`; rows have the same keys above. `base` is None when the pass used `include_base: false`.
   - Pairing rule: compare edited vs base ONLY from the same `generate` pass (e.g., `load_pass(k)`); never use `census.jsonl` as the base for a new pass. vLLM bf16 kernels are not batch-size invariant (batch=64 census vs batch=5 pass can flip samples).
   - `layer_outputs(layer_idx)` -> output-projection module names for that layer
   - `load_base_weight(name)` -> detached copy of frozen base tensor [out_features, in_features] (~tens of MB); avoid retaining many weights (16 GiB CPU orchestrator)
   - `load_lora_tensors()` / `save_lora_tensors(dict)` -> read/write active LoRA tensors
   - `unit(v)`, `mean_diff(group_a, group_b)` -> unit vector & unit difference of two group means
   - `as_float(t)` -> cast bf16 features to float32 (required before CPU `torch.linalg.svd`/PCA)
   - `set_rank1_ablation(name, u, alpha, slot=0)` -> encodes W_base - alpha * u u^T W_base into active LoRA slot (rank 4 -> slots 0..3; overwriting a non-zero slot prints a warning)
   - `drop_edit(name=None, slot=None)` / `list_edits()` -> clear or inspect active LoRA edits
   - `save_candidate(name)` / `load_candidate(name)` / `list_candidates()` / `drop_candidate(name)` -> manage up to 4 candidate edits for parallel testing (a candidate snapshots the WHOLE active adapter; call `drop_edit()` before building an unrelated candidate)
   - `read_notes()` / `write_notes(str)` -> persistent WORK/notes.md, max 36k chars (restored if context compacts)
   - `last_read` -> dict after a `"read"`: `hidden_states[layer]` is Tensor [n, {hidden_dim}] for `answer_mean`, or list[Tensor [len_i, {hidden_dim}]] for `last_k`/`every_k`/`all`; `mlp_activations[layer]` has dim {mlp_dim} (the INPUT to `mlp.down_proj`), while `set_rank1_ablation` expects an OUTPUT direction `u.shape == (module_shapes[name][0],)` ({hidden_dim} for output projections). Only the layers listed in the read OBS are present; other layer keys raise KeyError. Features are bf16.
   - Installed sandbox libs: `torch`, `numpy` (as `np`), `math`, `json`, plus stdlib `collections`, `itertools`, `functools`, `statistics`, `random`, `re`. No `scipy`/`sklearn`.

2. ```json``` block (budget: {max_passes} forward passes total; failed requests cost 0):
   - Generation / parallel candidate test (total generated answers <= {max_gen_answers} per pass):
     * Active edit vs. base: {{"action": "generate", "problem_ids": [0, 1, ...], "weights": "both", "n_samples": 1, "max_tokens": {max_tokens}}}
     * Up to 4 saved candidates vs. base side-by-side: {{"action": "generate", "problem_ids": [0, 1, ...], "candidates": ["c1", "c2"], "include_base": true, "n_samples": 1, "max_tokens": {max_tokens}}}
   - Hidden-state / MLP / logprob read (up to {MAX_READ_TEXTS} texts, <= {effective_read_max_length} answer tokens):
     {{"action": "read", "diag_indices": [0, 1, ...], "layers": [8, 16, 32, 56], "extract": ["hidden_states"], "token_mode": "answer_mean", "k": 32, "use_edited": false}}
     * Sources: `"diag_indices": [...]`, `"pass_ref": {{"pass_idx": 1, "indices": [...], "which": "base"}}`, or `"texts": [...]`
     * Optional `"candidate": "c1"` to read a saved candidate instead of `use_edited`
     * `extract`: subset of `["hidden_states", "mlp_activations", "logprobs"]`
     * `token_mode`: `"answer_mean"` | `"every_k"` | `"last_k"` | `"all"`

HOW TO WORK:
Inspect wrong and unfinished answers first; test components separately and keep only parts your tests support; compare base vs. edited on identical seeds; validate on practice problems not used to choose the edit. Keep concise notes via `write_notes(...)`. To stretch the 20-pass budget, use `save_candidate(name)` and test up to 4 edits side-by-side in one `generate` pass.
When ready (or when passes run out), ensure your winning edit is loaded into active LoRA (`load_candidate(...)` if using candidates), summarize your evidence, and output `FINAL_SUBMIT` (triggers the reserved validation check before final `SUBMIT`/`WITHDRAW`) or `FINAL_WITHDRAW`."""  # noqa: E501

    # Qwen3.5's Jinja chat template rejects a history with no `"user"` turn
    # (`TemplateError: No user query found in messages.`), so seed an explicit
    # kickoff. This also guarantees compaction never produces system-only state.
    resumed = resume_scan is not None
    if resumed:
        kickoff = (
            "Session resumed from durable state. Inspect `diagnosis` in Python to "
            "examine wrong/unfinished answers, review WORK/notes.md, or issue a "
            "`read` / `generate` forward request."
        )
    else:
        kickoff = (
            "Session started. Inspect `diagnosis` in Python to examine "
            "wrong/unfinished answers, or issue a `read` / `generate` forward request."
        )
    default_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": kickoff},
    ]
    restored_messages = resume_scan.get("messages") if resume_scan is not None else None
    messages_valid = (
        isinstance(restored_messages, list)
        and bool(restored_messages)
        and all(
            isinstance(message, dict)
            and message.get("role") in {"system", "user", "assistant"}
            and isinstance(message.get("content"), str)
            for message in restored_messages
        )
    )
    messages = list(restored_messages) if messages_valid else default_messages

    passes_used = 0
    code_runs_used = 0
    previous_passes_used = 0
    if resumed:
        previous_passes_used = int(resume_scan["passes_used"])
        passes_used = previous_passes_used
        code_runs_used = int(resume_scan["code_runs_used"])
    pass_log = []
    # Restore prior turns so the resumed summary is cumulative, not reset.
    transcript = list(resume_scan["turns"]) if resumed else []
    resume_lineage = list(resume_scan.get("resume_lineage") or []) if resumed else []
    try:
        active_lora_sha256 = content_addressed_lora_key(lora_dir)
    except (OSError, FileNotFoundError):
        active_lora_sha256 = None
    resume_lineage.append(
        {
            "attempt": len(resume_lineage) + 1,
            "resumed": resumed,
            "turns_used_prior": turns_used_prior,
            "elapsed_wall_prior_s": round(elapsed_wall_prior_s, 3),
            "active_lora_present": active_lora_sha256 is not None,
            "active_lora_sha256": active_lora_sha256,
            "config_path": config_path,
            "deviation": deviation,
        }
    )
    # Default to ABORTED_ERROR so an exception mid-loop is never mislabeled MAX_TURNS.
    final_status = "ABORTED_ERROR"
    last_obs = "Session started."
    consecutive_format_errors = 0
    consecutive_empty_replies = 0

    transcript_jsonl = os.path.join(work_dir, "transcript.jsonl")
    # Per-turn append log. Truncate only on a fresh run so re-runs do not
    # concatenate stale turns; a deliberate resume appends to the prior log.
    with open(transcript_jsonl, "a" if resumed else "w"):
        pass

    def _mark_pass_inflight(pass_no: int) -> None:
        """Durably record a pass before it spends GPU time.

        A preemption mid-pass never writes ``pass_<n>.json``; without this marker
        the resume scan would refund the pass and the run could exceed its budget.
        """
        with open(os.path.join(work_dir, f"pass_{pass_no}.inflight"), "w") as f:
            f.write("1")
        vol.commit()

    def _clear_pass_inflight(pass_no: int) -> None:
        try:
            os.remove(os.path.join(work_dir, f"pass_{pass_no}.inflight"))
        except OSError:
            pass

    turn_metrics: dict | None = None
    turn_t0: float | None = None

    def _write_resume_state() -> None:
        """Durably persist everything needed to continue after a hard preemption."""
        state = {
            "schema": 1,
            "turns_used_prior": len(transcript),
            "elapsed_wall_prior_s": round(time.time() - start_time, 3),
            "messages": messages,
            "passes_used": passes_used,
            "code_runs_used": code_runs_used,
            "resume_lineage": resume_lineage,
        }
        state_path = os.path.join(work_dir, "resume_state.json")
        tmp_state = f"{state_path}.tmp.{os.getpid()}"
        with open(tmp_state, "w") as f:
            json.dump(state, f, default=str)
        os.replace(tmp_state, state_path)

    def _record_turn(entry: dict) -> None:
        """Append one turn to the in-memory + on-disk transcript and commit.

        Committing per turn makes the transcript (and any LoRA edit from that turn)
        durable before the next expensive GPU call, so a hard Modal timeout still
        leaves forensics. This replaces several per-helper commits with one.
        """
        if turn_metrics is not None:
            if turn_t0 is not None:
                wall_s = time.perf_counter() - turn_t0
                gpu_s = float(turn_metrics.get("gpu_s") or 0.0)
                turn_metrics["wall_s"] = round(wall_s, 3)
                turn_metrics["cpu_s"] = round(max(0.0, wall_s - gpu_s), 3)
                turn_metrics["gpu_fraction"] = round(gpu_s / wall_s, 4) if wall_s > 0 else None
            entry.setdefault("metrics", {}).update(turn_metrics)
        transcript.append(entry)
        with open(transcript_jsonl, "a") as f:
            f.write(json.dumps(entry) + "\n")
        _write_resume_state()
        # Strict: if this fails, the GPU would test stale adapter weights next turn,
        # silently corrupting the science. Fail loudly instead.
        t_commit0 = time.perf_counter()
        vol.commit()
        commit_s = round(time.perf_counter() - t_commit0, 4)
        metrics["commit_s"].append(commit_s)
        if turn_metrics is not None:
            turn_metrics["commit_s"] = commit_s
            entry.setdefault("metrics", {})["commit_s"] = commit_s
            metrics["turns"].append(dict(turn_metrics))

    def _load_val_base_cache(path: str, expected: list[dict]):
        """Load a fully-keyed validation base cache and re-grade its labels."""
        rows = load_validation_base_cache(
            path,
            expected,
            engine_config_hash=engine_cfg_hash,
            max_tokens=max_tokens,
            grade=grade_answer,
        )
        if rows is None:
            print(f"[val-cache] miss: identity/content mismatch at {path}")
            return None
        print(f"[val-cache] hit ({len(rows)} rows) from {path}")
        return rows

    def _save_val_base_cache(path: str, rows: list[dict]) -> None:
        save_validation_base_cache(
            path,
            rows,
            engine_config_hash=engine_cfg_hash,
            max_tokens=max_tokens,
        )
        vol.commit()
        print(f"[val-cache] saved {len(rows)} base rows to {path}")

    def _clear_submitted_lora() -> None:
        metrics["submitted_lora_snapshot"] = False
        if os.path.isdir(submitted_lora_dir):
            shutil.rmtree(submitted_lora_dir, ignore_errors=True)
            print("[submit] cleared WORK/submitted_lora")

    def _snapshot_submitted_lora() -> None:
        _clear_submitted_lora()
        os.makedirs(submitted_lora_dir, exist_ok=True)
        for name in ("adapter_model.safetensors", "adapter_config.json"):
            src = os.path.join(lora_dir, name)
            if not os.path.exists(src):
                raise FileNotFoundError(f"Active LoRA missing {name}; cannot snapshot.")
            shutil.copy2(src, os.path.join(submitted_lora_dir, name))
        write_submit_provenance(
            submitted_lora_dir,
            decision="PENDING",
            session_id=session_id,
        )
        metrics["submitted_lora_snapshot"] = True
        _safe_commit()
        print(f"[submit] snapshotted active LoRA to {submitted_lora_dir}")

    # Persist the starting point before the first expensive turn so a preemption in
    # Turn 1 still leaves the restored lineage and cumulative budget on disk.
    _write_resume_state()
    vol.commit()

    try:
        for turn in range(turns_used_prior + 1, max_turns + 1):
            turn_t0 = time.perf_counter()
            turn_metrics = {
                "turn": turn,
                "gpu_s": 0.0,
                "editor_s": None,
                "action": None,
                "action_s": None,
                "code_s": None,
                "compaction_s": None,
                "prompt_tokens": None,
                "gen_tokens": None,
                "finish_reason": None,
            }
            elapsed_min = (time.time() - start_time) / 60.0
            if elapsed_min >= wall_limit_hours * 60.0:
                print(f"Wall-clock limit ({wall_limit_hours}h) reached at turn {turn}.")
                final_status = "WALL_LIMIT"
                break

            # Count locally with the orchestrator's tokenizer: same result as the GPU
            # tokenizer, without a per-turn Modal round-trip.
            prompt_toks = _count_prompt_tokens(messages)
            if prompt_toks > COMPACT_TOKEN_LIMIT:
                t_compact0 = time.perf_counter()
                print(
                    f"[Turn {turn}] Compacting history "
                    f"({prompt_toks} tokens > {COMPACT_TOKEN_LIMIT})..."
                )
                # Bound every compaction input: otherwise a large notes.md or pass log
                # can keep the compacted prompt above COMPACT_TOKEN_LIMIT forever
                # (infinite compaction loop) and overflow max_model_len.
                notes_excerpt = helpers["read_notes"]()
                if len(notes_excerpt) > 16_000:
                    notes_excerpt = notes_excerpt[-16_000:]
                pass_log_tail = [entry[:2_000] for entry in pass_log[-20:]]
                compact_summary = (
                    f"[CONTEXT COMPACTED AT {prompt_toks} TOKENS]\n"
                    f"Budgets: passes={passes_used}/{max_passes}, "
                    f"code_runs={code_runs_used}/{max_code_runs}, "
                    f"elapsed={elapsed_min:.1f}m / {wall_limit_hours * 60:.0f}m.\n"
                    f"Active edits: {json.dumps(helpers['list_edits']())[:8_000]}\n"
                    f"Saved candidates: {helpers['list_candidates']()[:32]}\n"
                    f"Pass log (last {len(pass_log_tail)}):\n"
                    f"{json.dumps(pass_log_tail, indent=2)}\n\n"
                    f"Your WORK/notes.md (tail):\n{notes_excerpt}\n\n"
                    f"Last observation:\n{last_obs[:6_000]}"
                )
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": compact_summary},
                ]
                if _count_prompt_tokens(messages) > COMPACT_TOKEN_LIMIT:
                    # Last-resort fallback: drop pass log + notes so the next turn is
                    # guaranteed to be under the threshold.
                    compact_summary = (
                        f"[CONTEXT COMPACTED AT {prompt_toks} TOKENS - HARD TRUNCATED]\n"
                        f"Budgets: passes={passes_used}/{max_passes}, "
                        f"code_runs={code_runs_used}/{max_code_runs}, "
                        f"elapsed={elapsed_min:.1f}m / {wall_limit_hours * 60:.0f}m.\n"
                        f"Active edits: {json.dumps(helpers['list_edits']())[:4_000]}\n"
                        f"Saved candidates: {helpers['list_candidates']()[:32]}\n"
                        f"Last observation:\n{last_obs[:2_000]}"
                    )
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": compact_summary},
                    ]
                    print(
                        f"[Turn {turn}] Compaction hard-truncated to fit "
                        f"{COMPACT_TOKEN_LIMIT} tokens."
                    )
                turn_metrics["compaction_s"] = round(time.perf_counter() - t_compact0, 3)

            step_res, editor_s = _gpu_call(
                engine.editor_step.remote, messages, max_tokens=EDITOR_MAX_TOKENS
            )
            turn_metrics["editor_s"] = editor_s
            turn_metrics["gpu_s"] += editor_s
            turn_metrics["prompt_tokens"] = step_res.get("prompt_tokens")
            turn_metrics["gen_tokens"] = step_res.get("tokens")
            turn_metrics["finish_reason"] = step_res.get("finish_reason")
            turn_metrics["editor_timings"] = step_res.get("timings")
            turn_metrics["gpu_memory_editor"] = step_res.get("gpu_memory")
            reply = step_res["text"]
            if not reply.strip():
                consecutive_empty_replies += 1
                if consecutive_empty_replies >= 3:
                    final_status = "ABORTED_EMPTY_REPLIES"
                    raise RuntimeError("FAIL-FAST: 3 consecutive empty replies from editor.")
                # Do not append an empty assistant turn (some chat templates reject it).
                obs = (
                    "Empty reply. Emit exactly ONE ```python``` or ```json``` block after </think>."
                )
                last_obs = obs
                messages.append({"role": "user", "content": obs})
                _record_turn({"turn": turn, "reply": reply, "obs": obs})
                continue
            consecutive_empty_replies = 0

            messages.append({"role": "assistant", "content": reply})
            print(
                f"\n--- Turn {turn} ({elapsed_min:.1f}m, prompt={step_res['prompt_tokens']}t, "
                f"gen={step_res['tokens']}t, finish={step_res['finish_reason']}) ---"
            )
            print(reply[:500] + ("..." if len(reply) > 500 else ""))

            if "</think>" not in reply and step_res["finish_reason"] == "length":
                consecutive_format_errors += 1
                obs = (
                    f"Turn truncated at {EDITOR_MAX_TOKENS} tokens before closing </think>. "
                    "Be much more concise and emit exactly ONE ```python``` or ```json``` block."
                )
                _record_turn({"turn": turn, "reply": reply, "obs": obs, "finish": "length"})
                if consecutive_format_errors >= 5:
                    final_status = "ABORTED_TRUNCATION_LOOP"
                    raise RuntimeError("FAIL-FAST: 5 consecutive format/truncation failures.")
                last_obs = obs
                messages.append({"role": "user", "content": obs})
                continue

            reply_visible = strip_thinking(reply)
            py_blocks = re.findall(
                r"```python[ \t]*\r?\n?(.*?)```", reply_visible, re.DOTALL | re.IGNORECASE
            )
            json_blocks = re.findall(
                r"```json[ \t]*\r?\n?(.*?)```", reply_visible, re.DOTALL | re.IGNORECASE
            )

            # Submission / withdrawal gate (Sec. E.9)
            if not py_blocks and not json_blocks:
                if "FINAL_WITHDRAW" in reply_visible:
                    print(f"[Turn {turn}] Model withdrew its edit.")
                    helpers["drop_edit"]()
                    _clear_submitted_lora()
                    final_status = "WITHDRAWN"
                    _record_turn({"turn": turn, "reply": reply, "obs": "FINAL_WITHDRAW"})
                    break
                if "FINAL_SUBMIT" in reply_visible:
                    # Enforce the protocol key/shape/rank/finiteness whitelist and the
                    # immutable canonical config before any adapter can be submitted.
                    # This must run before list_edits(), which itself validates the
                    # active adapter and would otherwise raise out of the turn loop.
                    try:
                        load_and_validate_adapter(lora_dir, module_registry)
                    except Exception as adapter_err:  # noqa: BLE001 - feed back to model
                        obs = (
                            "FINAL_SUBMIT rejected: the active adapter failed protocol "
                            f"validation ({type(adapter_err).__name__}: {adapter_err}). "
                            "Use only the registered rank-4 lora_A/lora_B keys with "
                            "finite values, or output FINAL_WITHDRAW."
                        )
                        print(obs)
                        _record_turn({"turn": turn, "reply": reply, "obs": obs})
                        last_obs = obs
                        messages.append({"role": "user", "content": obs})
                        continue

                    active_edits = helpers["list_edits"]()
                    saved_cands = helpers["list_candidates"]()
                    if not active_edits and not saved_cands:
                        obs = (
                            "Active LoRA has no non-zero edits and there are no saved "
                            "candidates. Apply an edit or output FINAL_WITHDRAW."
                        )
                        _record_turn({"turn": turn, "reply": reply, "obs": obs})
                        last_obs = obs
                        messages.append({"role": "user", "content": obs})
                        continue
                    if not active_edits and saved_cands:
                        obs = (
                            "Your active LoRA is currently empty (all zeros), while "
                            f"saved candidates {saved_cands} exist. Run "
                            "`load_candidate(name)` in ```python``` first, "
                            "or output `FINAL_WITHDRAW` if you intend to submit no edit."
                        )
                        _record_turn({"turn": turn, "reply": reply, "obs": obs})
                        last_obs = obs
                        messages.append({"role": "user", "content": obs})
                        continue

                    # Durable audit copy before spending the reserved validation gate.
                    _snapshot_submitted_lora()
                    metrics["submitted_lora_snapshot"] = True

                    n_val = min(val_n, len(reserved_check_set))
                    val_problems = reserved_check_set[:n_val]
                    model_tag = MODEL_ID.replace("/", "_").replace(" ", "_")
                    val_cache_path = (
                        f"{WORKSPACE_DIR}/cache/val_base_{model_tag}_{dataset_mode}_"
                        f"{n_val}_{val_samples}_{max_tokens}_{cache_tag}_"
                        f"{engine_cfg_hash}_{dataset_identity_hash}.json"
                    )
                    expected_meta = [
                        {
                            "id": p["id"],
                            "sample_idx": s_idx,
                            "problem": p["problem"],
                            "answer": p["answer"],
                        }
                        for p in val_problems
                        for s_idx in range(val_samples)
                    ]
                    print(
                        f"[Turn {turn}] Reserved validation gate ({n_val} problems x "
                        f"{val_samples} samples)..."
                    )
                    b_rows = _load_val_base_cache(val_cache_path, expected_meta)
                    metrics["validation"].update(
                        {
                            "n_val": n_val,
                            "val_samples": val_samples,
                            "base_cache_hit": b_rows is not None,
                        }
                    )
                    if b_rows is None:
                        metrics["counters"]["validation_cache_misses"] += 1
                        print("[val-cache] computing frozen base + edited together.")
                        val_res, validation_s = _gpu_call(
                            engine.generate_batch.remote,
                            val_problems,
                            lora_specs={"active": lora_dir},
                            include_base=True,
                            n_samples=val_samples,
                            max_tokens=max_tokens,
                        )
                        b_rows = [
                            {**r, "correct": grade_answer(r["text"], r["answer"])}
                            for r in val_res["base"]
                        ]
                        e_rows = [
                            {**r, "correct": grade_answer(r["text"], r["answer"])}
                            for r in val_res["candidates"]["active"]
                        ]
                        try:
                            _save_val_base_cache(val_cache_path, b_rows)
                        except Exception as cache_err:  # noqa: BLE001
                            print(
                                f"[val-cache] save failed (non-fatal): "
                                f"{type(cache_err).__name__}: {cache_err}"
                            )
                    else:
                        metrics["counters"]["validation_cache_hits"] += 1
                        print("[val-cache] generating edited only (base reused).")
                        val_res, validation_s = _gpu_call(
                            engine.generate_batch.remote,
                            val_problems,
                            lora_specs={"active": lora_dir},
                            include_base=False,
                            n_samples=val_samples,
                            max_tokens=max_tokens,
                        )
                        e_rows = [
                            {**r, "correct": grade_answer(r["text"], r["answer"])}
                            for r in val_res["candidates"]["active"]
                        ]
                    metrics["validation"]["gpu_s"] = validation_s
                    metrics["validation"]["timings"] = val_res.get("timings", {})
                    metrics["phases"]["validation_s"] = validation_s
                    turn_metrics["validation_s"] = validation_s
                    turn_metrics["gpu_s"] += validation_s
                    val_timings = val_res.get("timings", {})
                    metrics["counters"]["generated_answers"] += int(val_timings.get("n_answers", 0))
                    metrics["counters"]["generated_tokens"] += int(
                        val_timings.get("generated_tokens", 0)
                    )
                    st = compute_paired_stats(b_rows, e_rows, signflip_resamples=100_000)
                    p_txt = (
                        f"one-sided sign-flip p={st['signflip_p_one_sided']:.4f} "
                        f"(two-sided p={st['signflip_p_two_sided']:.4f}"
                    )
                    if st.get("mcnemar_p") is not None:
                        p_txt += f", McNemar p={st['mcnemar_p']:.4f}"
                    p_txt += ")"
                    check_msg = (
                        f"Reserved validation check ({st['n_problems']} problems x "
                        f"{val_samples} samples = {st['n_answers']} answers, "
                        f"{max_tokens} tokens; this gate is exempt from the per-pass "
                        "answer cap): "
                        f"Base={st['base_acc']:.1%} "
                        f"(cut={st['base_cut']}, loop={st['base_loop']}), "
                        f"Edited={st['edited_acc']:.1%} "
                        f"(cut={st['edited_cut']}, loop={st['edited_loop']}), "
                        f"diff={st['delta']:+.1%} ± {st['se']:.1%} "
                        f"[95% CI {st['ci_low']:+.1%}, {st['ci_high']:+.1%}], "
                        f"{p_txt} "
                        f"(w2r={st['wrong_to_right']}, r2w={st['right_to_wrong']}). "
                        "Reply simply `SUBMIT` or `WITHDRAW`."
                    )
                    print(check_msg)
                    messages.append({"role": "user", "content": check_msg})
                    confirm_res, confirm_s = _gpu_call(
                        engine.editor_step.remote,
                        messages,
                        max_tokens=EDITOR_MAX_TOKENS,
                    )
                    turn_metrics["gpu_s"] += confirm_s
                    turn_metrics["confirm_s"] = confirm_s
                    turn_metrics["confirm_timings"] = confirm_res.get("timings")
                    confirm = confirm_res["text"]
                    confirm_visible = strip_thinking(confirm).upper()
                    print(f"Final confirmation reply: {confirm_visible[-300:]}")
                    # Parse the last standalone decision keyword; a phrase like
                    # "I will not SUBMIT; I WITHDRAW" must resolve to WITHDRAW.
                    decisions = re.findall(r"\b(SUBMIT|WITHDRAW)\b", confirm_visible)
                    withdrawn = (not decisions) or decisions[-1] == "WITHDRAW"
                    if withdrawn:
                        helpers["drop_edit"]()
                        _clear_submitted_lora()
                        final_status = "WITHDRAWN_AFTER_VAL"
                        print("Final decision: WITHDRAWN (active_lora cleared).")
                    else:
                        final_status = "SUBMITTED"
                        write_submit_provenance(
                            submitted_lora_dir,
                            decision="SUBMIT",
                            session_id=session_id,
                            extra={"validation_stats": st},
                        )
                        _safe_commit()
                        print("Final decision: SUBMITTED (active_lora kept).")
                    _record_turn(
                        {
                            "turn": turn,
                            "reply": reply,
                            "obs": check_msg,
                            "confirm": confirm,
                            "val_stats": st,
                        }
                    )
                    break

            obs = ""
            if len(py_blocks) + len(json_blocks) != 1:
                consecutive_format_errors += 1
                obs = (
                    "Format error: reply with exactly ONE ```python``` block or ONE "
                    "```json``` block after </think> (or FINAL_SUBMIT / "
                    "FINAL_WITHDRAW without tool blocks). Cost: 0."
                )
                _record_turn({"turn": turn, "reply": reply, "obs": obs})
                if consecutive_format_errors >= 5:
                    final_status = "ABORTED_FORMAT_LOOP"
                    raise RuntimeError("FAIL-FAST: 5 consecutive format/tool-block errors.")
            elif py_blocks:
                consecutive_format_errors = 0
                if code_runs_used >= max_code_runs:
                    obs = (
                        f"Code-execution budget exhausted "
                        f"({max_code_runs}/{max_code_runs}). Use remaining passes or "
                        "FINAL_SUBMIT / FINAL_WITHDRAW."
                    )
                else:
                    code = py_blocks[0]
                    buf = _LimitedBuffer(CODE_STDOUT_LIMIT)
                    prev_handler = signal.getsignal(signal.SIGALRM)

                    def _timeout_handler(_signum, _frame):
                        raise TimeoutError(
                            f"CPU sandbox execution exceeded {CODE_TIMEOUT_S}s limit."
                        )

                    try:
                        t_code0 = time.perf_counter()
                        validate_sandbox_code(code)
                        signal.signal(signal.SIGALRM, _timeout_handler)
                        signal.alarm(CODE_TIMEOUT_S)
                        with (
                            torch.no_grad(),
                            contextlib.redirect_stdout(buf),
                            contextlib.redirect_stderr(buf),
                        ):
                            exec(code, sandbox_namespace)
                        code_runs_used += 1
                        out_txt = (
                            buf.getvalue()
                            or "Code executed successfully (no stdout; no forward pass was run)."
                        )
                        if buf.truncated:
                            out_txt += f"\n[output truncated at {CODE_STDOUT_LIMIT} chars]"
                        obs = (
                            f"[code {code_runs_used}/{max_code_runs}, "
                            f"{elapsed_min:.1f}m / {wall_limit_hours * 60:.0f}m]\n{out_txt}"
                        )
                    except Exception as e:  # noqa: BLE001 - errors are the model's feedback
                        code_runs_used += 1
                        tb = traceback.format_exc(limit=4)
                        obs = (
                            f"[code {code_runs_used}/{max_code_runs}, "
                            f"{elapsed_min:.1f}m / {wall_limit_hours * 60:.0f}m] "
                            f"Python Error: {type(e).__name__}: {e}\n"
                            f"Traceback (tail):\n{tb[-1200:]}"
                        )
                    finally:
                        signal.alarm(0)
                        signal.signal(signal.SIGALRM, prev_handler)
                        gc.collect()
                        code_s = round(time.perf_counter() - t_code0, 3)
                        metrics["code_run_s"].append(code_s)
                        if turn_metrics is not None:
                            turn_metrics["code_s"] = code_s
                _record_turn({"turn": turn, "reply": reply, "obs": obs})
            else:
                consecutive_format_errors = 0
                if passes_used >= max_passes:
                    obs = (
                        f"Forward-pass budget exhausted "
                        f"({max_passes}/{max_passes}). Output FINAL_SUBMIT or "
                        "FINAL_WITHDRAW."
                    )
                    _record_turn({"turn": turn, "reply": reply, "obs": obs})
                else:
                    # Strictly separate LLM JSON/parameter validation errors (cost: 0)
                    # from fatal GPU/Modal RuntimeErrors (fail fast!). Everything the model
                    # controls is validated here, so a typo returns cheap feedback instead
                    # of crashing a paid container.
                    valid_token_modes = {
                        "answer_mean",
                        "mean",
                        "every_k",
                        "last_k",
                        "all",
                    }
                    valid_extracts = {
                        "hidden_states",
                        "mlp_activations",
                        "logprobs",
                    }
                    try:
                        req = json.loads(json_blocks[0])
                        if not isinstance(req, dict):
                            raise ValueError("JSON block must decode to an object.")
                        action = req.get("action")
                        # The protocol read budget is capped at 8,192 tokens (Table 5);
                        # model-authored requests can only lower it, never raise it.
                        read_max_length = effective_read_max_length

                        if action == "generate":
                            pids = [
                                i
                                for i in req.get("problem_ids", list(range(16)))
                                if type(i) is int and 0 <= i < len(practice_set)
                            ]
                            pids = list(dict.fromkeys(pids))
                            if not pids:
                                raise ValueError("Empty or invalid problem_ids.")
                            n_samples = max(1, int(req.get("n_samples", 1)))
                            req_max_tok = int(req.get("max_tokens", max_tokens))
                            if req_max_tok > max_tokens:
                                raise ValueError(
                                    f"Requested max_tokens={req_max_tok} exceeds the "
                                    f"session answer budget of {max_tokens}; refusing a "
                                    "model-authored budget bypass."
                                )
                            if req_max_tok < 16:
                                raise ValueError(f"Requested max_tokens={req_max_tok} is below 16.")

                            if "candidates" in req:
                                raw_cnames = req["candidates"]
                                if not isinstance(raw_cnames, list) or not raw_cnames:
                                    raise ValueError("'candidates' must be a non-empty list.")
                                cnames = [str(c) for c in raw_cnames[:4]]
                                known = set(helpers["list_candidates"]())
                                unknown = [c for c in cnames if c not in known]
                                if unknown:
                                    raise ValueError(
                                        f"Unknown candidates {unknown}. Saved: {sorted(known)}"
                                    )
                                lora_specs = {c: os.path.join(candidates_dir, c) for c in cnames}
                                include_base = bool(req.get("include_base", True))
                            else:
                                w_flag = req.get("weights", "both")
                                if w_flag not in ("base", "edited", "both"):
                                    raise ValueError("weights must be 'base', 'edited', or 'both'.")
                                include_base = w_flag in ("base", "both")
                                lora_specs = (
                                    {"active": lora_dir} if w_flag in ("edited", "both") else {}
                                )

                            total_answers = (
                                len(pids) * n_samples * (int(include_base) + len(lora_specs))
                            )
                            if total_answers > max_gen_answers:
                                raise ValueError(
                                    f"Requested {total_answers} total answers exceeds "
                                    f"cap of {max_gen_answers}. Reduce problem_ids, "
                                    "n_samples, or number of candidates."
                                )

                            sub_probs = [practice_set[i] for i in pids]

                            base_cache_path = base_cache_meta = None
                            cached_base_rows = None
                            if include_base:
                                (
                                    base_cache_path,
                                    base_cache_meta,
                                    cached_base_rows,
                                ) = _base_cache_lookup(pids, n_samples, req_max_tok)
                        elif action == "read":
                            raw_layers = req.get("layers", [8, 16, 32, 56])
                            if not isinstance(raw_layers, list):
                                raise ValueError("'layers' must be a list.")
                            layers = sorted(
                                {
                                    int(layer_idx)
                                    for layer_idx in raw_layers
                                    if type(layer_idx) is int and 0 <= int(layer_idx) < n_layers
                                }
                            )[:MAX_READ_LAYERS]
                            if not layers:
                                raise ValueError(f"No valid layer indices in [0, {n_layers}).")
                            extract_kinds = req.get("extract", ["hidden_states"])
                            if isinstance(extract_kinds, str):
                                extract_kinds = [extract_kinds]
                            if not isinstance(extract_kinds, list) or not extract_kinds:
                                raise ValueError("'extract' must be a non-empty list.")
                            extract_kinds = [str(e) for e in extract_kinds]
                            bad_extracts = [e for e in extract_kinds if e not in valid_extracts]
                            if bad_extracts:
                                raise ValueError(
                                    f"Unknown extract kinds {bad_extracts}; "
                                    f"allowed {sorted(valid_extracts)}."
                                )
                            token_mode = str(req.get("token_mode", "answer_mean"))
                            if token_mode not in valid_token_modes:
                                raise ValueError(
                                    f"token_mode must be one of {sorted(valid_token_modes)}."
                                )
                            k_val = max(1, int(req.get("k", 32)))
                            use_edited = bool(req.get("use_edited", False))
                            cand_name = req.get("candidate")
                            if cand_name is not None:
                                cand_name = str(cand_name)
                                if cand_name not in helpers["list_candidates"]():
                                    raise ValueError(f"Unknown candidate '{cand_name}'.")
                                read_lora_dir = os.path.join(candidates_dir, cand_name)
                            else:
                                read_lora_dir = lora_dir if use_edited else None

                            if "diag_indices" in req:
                                idxs = [
                                    i
                                    for i in req["diag_indices"]
                                    if type(i) is int and 0 <= i < len(diag_graded)
                                ][:MAX_READ_TEXTS]
                                if not idxs:
                                    raise ValueError("No valid diag_indices.")
                                prompt_strs = [diag_graded[i]["problem"] for i in idxs]
                                answer_strs = [diag_graded[i]["text"] for i in idxs]
                            elif "pass_ref" in req:
                                pref = req["pass_ref"]
                                if not isinstance(pref, dict):
                                    raise ValueError("'pass_ref' must be an object.")
                                pdata = helpers["load_pass"](int(pref["pass_idx"]))
                                which = str(pref.get("which", "base"))
                                if which == "base":
                                    prows = pdata.get("base")
                                    if prows is None:
                                        raise ValueError(
                                            "pass_ref requested base rows but this pass "
                                            "has no base outputs."
                                        )
                                else:
                                    if which not in pdata.get("candidates", {}):
                                        raise ValueError(
                                            f"pass has no candidate '{which}'. "
                                            f"Available: {sorted(pdata.get('candidates', {}))}"
                                        )
                                    prows = pdata["candidates"][which]
                                idxs = [
                                    i
                                    for i in pref.get("indices", list(range(len(prows))))
                                    if type(i) is int and 0 <= i < len(prows)
                                ][:MAX_READ_TEXTS]
                                if not idxs:
                                    raise ValueError("No valid pass_ref indices.")
                                prompt_strs = [prows[i]["problem"] for i in idxs]
                                answer_strs = [prows[i]["text"] for i in idxs]
                            elif "texts" in req and isinstance(req["texts"], list) and req["texts"]:
                                answer_strs = [str(t) for t in req["texts"][:MAX_READ_TEXTS]]
                                if any(not t.strip() for t in answer_strs):
                                    raise ValueError("'texts' entries must be non-empty strings.")
                                prompt_strs = [""] * len(answer_strs)
                            else:
                                raise ValueError(
                                    "Provide 'diag_indices', 'pass_ref', or non-empty 'texts'."
                                )

                            # Reject oversized reads on CPU before sleeping vLLM/spawning.
                            n_texts = len(answer_strs)
                            if token_mode in ("answer_mean", "mean"):
                                tokens_per_text = 1
                            elif token_mode == "last_k":
                                tokens_per_text = max(1, k_val)
                            elif token_mode == "every_k":
                                tokens_per_text = max(1, (read_max_length + k_val - 1) // k_val)
                            else:  # "all"
                                tokens_per_text = max(1, read_max_length)
                            n_hs = len(layers) if "hidden_states" in extract_kinds else 0
                            n_mlp = len(layers) if "mlp_activations" in extract_kinds else 0
                            est_bytes = (
                                n_texts
                                * tokens_per_text
                                * (n_hs * hidden_dim + n_mlp * mlp_dim)
                                * 2
                            )
                            if est_bytes > MAX_READ_BYTES:
                                raise ValueError(
                                    f"Estimated read payload ~{est_bytes / 2**30:.1f} GiB "
                                    f"exceeds {MAX_READ_BYTES / 2**30:.0f} GiB budget; "
                                    "reduce texts/layers or use token_mode='answer_mean'."
                                )
                        else:
                            raise ValueError(
                                f"Unknown action '{action}'. Use 'generate' or 'read'."
                            )
                    except (
                        ValueError,
                        KeyError,
                        TypeError,
                        FileNotFoundError,
                        json.JSONDecodeError,
                    ) as e:
                        obs = f"Forward request failed (cost: 0 passes): {type(e).__name__}: {e}"
                        _record_turn({"turn": turn, "reply": reply, "obs": obs})
                        last_obs = obs
                        print(f"OBS: {obs[:400]}")
                        messages.append({"role": "user", "content": obs})
                        continue

                    if action == "generate":
                        _mark_pass_inflight(passes_used + 1)
                        base_cache_hit = bool(include_base and cached_base_rows is not None)
                        if base_cache_hit:
                            metrics["counters"]["base_cache_hits"] += 1
                            if lora_specs:
                                raw, generate_s = _gpu_call(
                                    engine.generate_batch.remote,
                                    sub_probs,
                                    lora_specs=lora_specs,
                                    include_base=False,
                                    n_samples=n_samples,
                                    max_tokens=req_max_tok,
                                )
                                raw["base"] = cached_base_rows
                            else:
                                # Base-only request: no GPU call is needed.
                                raw = {
                                    "base": cached_base_rows,
                                    "candidates": {},
                                    "timings": {
                                        "n_answers": 0,
                                        "generated_tokens": 0,
                                        "base_cache_hit": True,
                                    },
                                    "gpu_memory": {},
                                }
                                generate_s = 0.0
                        else:
                            if include_base:
                                metrics["counters"]["base_cache_misses"] += 1
                            raw, generate_s = _gpu_call(
                                engine.generate_batch.remote,
                                sub_probs,
                                lora_specs=lora_specs,
                                include_base=include_base,
                                n_samples=n_samples,
                                max_tokens=req_max_tok,
                            )
                            if include_base and raw.get("base") is not None:
                                try:
                                    _save_base_batch_cache(
                                        base_cache_path, base_cache_meta, raw["base"]
                                    )
                                except Exception as cache_err:  # noqa: BLE001
                                    print(
                                        "[base-cache] save failed (non-fatal): "
                                        f"{type(cache_err).__name__}: {cache_err}"
                                    )
                        passes_used += 1
                        turn_metrics["action"] = "generate"
                        turn_metrics["action_s"] = generate_s
                        turn_metrics["gpu_s"] += generate_s
                        turn_metrics["base_cache_hit"] = base_cache_hit
                        gen_timings = raw.get("timings", {})
                        turn_metrics["action_timings"] = gen_timings
                        turn_metrics["gpu_memory_generate"] = raw.get("gpu_memory")
                        metrics["generate_calls"].append(
                            {
                                "turn": turn,
                                "passes_used": passes_used,
                                "wall_s": generate_s,
                                "base_cache_hit": base_cache_hit,
                                "timings": gen_timings,
                            }
                        )
                        metrics["counters"]["generated_answers"] += int(
                            gen_timings.get("n_answers", 0)
                        )
                        metrics["counters"]["generated_tokens"] += int(
                            gen_timings.get("generated_tokens", 0)
                        )

                        graded_payload = {"base": None, "candidates": {}}
                        if raw["base"] is not None:
                            graded_payload["base"] = [
                                {**r, "correct": grade_answer(r["text"], r["answer"])}
                                for r in raw["base"]
                            ]
                        for cname, rows in raw["candidates"].items():
                            graded_payload["candidates"][cname] = [
                                {**r, "correct": grade_answer(r["text"], r["answer"])} for r in rows
                            ]

                        pass_path = os.path.join(work_dir, f"pass_{passes_used}.json")
                        tmp = f"{pass_path}.tmp.{os.getpid()}"
                        with open(tmp, "w") as f:
                            json.dump(graded_payload, f)
                        os.replace(tmp, pass_path)
                        _clear_pass_inflight(passes_used)
                        # _record_turn() below commits this pass file + any pending edits.

                        base_cache_label = (
                            "hit" if base_cache_hit else ("miss" if include_base else "n/a")
                        )
                        lines = [
                            f"[pass {passes_used}/{max_passes}, "
                            f"{elapsed_min:.1f}m / "
                            f"{wall_limit_hours * 60:.0f}m] "
                            f"Generate complete (problems={len(pids)}, "
                            f"n_samples={n_samples}, "
                            f"total_answers={total_answers}, "
                            f"base_cache={base_cache_label}, "
                            f"saved to load_pass({passes_used})):"
                        ]
                        b_rows = graded_payload["base"]
                        if b_rows is not None and not graded_payload["candidates"]:
                            st = compute_paired_stats(b_rows)
                            lines.append(
                                f"  base: acc={st['base_acc']:.1%}, "
                                f"cut={st['base_cut']}, loop={st['base_loop']}"
                            )
                        for cname, e_rows in graded_payload["candidates"].items():
                            if b_rows is not None:
                                st = compute_paired_stats(b_rows, e_rows)
                                p_txt = (
                                    f"one-sided p={st['signflip_p_one_sided']:.4f} "
                                    f"(two-sided p={st['signflip_p_two_sided']:.4f}"
                                )
                                if st.get("mcnemar_p") is not None:
                                    p_txt += f", McNemar={st['mcnemar_p']:.4f}"
                                p_txt += ")"
                                lines.append(
                                    f"  {cname} vs base: "
                                    f"base={st['base_acc']:.1%} "
                                    f"(cut={st['base_cut']}, loop={st['base_loop']}), "
                                    f"{cname}={st['edited_acc']:.1%} "
                                    f"(cut={st['edited_cut']}, "
                                    f"loop={st['edited_loop']}), "
                                    f"diff={st['delta']:+.1%} ± {st['se']:.1%} "
                                    f"[95% CI {st['ci_low']:+.1%}, {st['ci_high']:+.1%}], "
                                    f"{p_txt}, "
                                    f"w2r={st['wrong_to_right']}, r2w={st['right_to_wrong']}"
                                )
                            else:
                                st = compute_paired_stats(e_rows)
                                lines.append(
                                    f"  {cname}: acc={st['base_acc']:.1%}, "
                                    f"cut={st['base_cut']}, loop={st['base_loop']}"
                                )

                        obs = "\n".join(lines)
                        pass_log.append(obs)
                        _record_turn({"turn": turn, "reply": reply, "obs": obs})
                    else:
                        next_pass_num = passes_used + 1
                        _mark_pass_inflight(next_pass_num)
                        out_pt_path = os.path.join(reads_dir, f"read_{next_pass_num}.pt")
                        read_res, read_s = _gpu_call(
                            engine.read_hidden_states.remote,
                            prompt_strs=prompt_strs,
                            answer_strs=answer_strs,
                            layers=layers,
                            extract_kinds=extract_kinds,
                            token_mode=token_mode,
                            k=k_val,
                            lora_dir=read_lora_dir,
                            out_path=out_pt_path,
                            max_length=read_max_length,
                            worker_timeout_s=worker_timeout_s,
                        )
                        passes_used += 1
                        turn_metrics["action"] = "read"
                        turn_metrics["action_s"] = read_s
                        turn_metrics["gpu_s"] += read_s
                        read_timings = read_res.get("timings", {})
                        turn_metrics["action_timings"] = read_timings
                        metrics["counters"]["reads"] += 1
                        metrics["read_fixed_overhead_s"].append(
                            float(read_timings.get("fixed_overhead_s", 0.0))
                        )
                        _safe_reload("orchestrator:read")
                        # Payload is tensors + primitives only, so weights_only=True is
                        # both sufficient and safer than unpickling arbitrary objects.
                        loaded = torch.load(
                            out_pt_path,
                            map_location="cpu",
                            weights_only=True,
                            mmap=False,
                        )
                        sandbox_namespace["last_read"] = loaded

                        hs_keys = list(loaded["hidden_states"].keys())
                        mlp_keys = list(loaded["mlp_activations"].keys())
                        obs = (
                            f"[pass {passes_used}/{max_passes}, "
                            f"{elapsed_min:.1f}m / "
                            f"{wall_limit_hours * 60:.0f}m] "
                            f"Read complete (n={len(answer_strs)}, "
                            f"token_mode={token_mode}, k={k_val}). "
                            f"Stored in `last_read`: "
                            f"hidden_states layers={hs_keys}, "
                            f"mlp_activations layers={mlp_keys}, "
                            f"logprobs={loaded['logprobs'] is not None}. "
                            "Features are bf16: call `as_float(t)` before "
                            "torch.linalg.* (CPU SVD/PCA)."
                        )
                        pass_log.append(obs)
                        _clear_pass_inflight(next_pass_num)
                        _record_turn({"turn": turn, "reply": reply, "obs": obs})

            last_obs = obs
            print(f"OBS: {obs[:400]}")
            messages.append({"role": "user", "content": obs})
        else:
            # `for ... else` runs only when the loop exhausts max_turns without a
            # `break`, i.e. the model never submitted/withdrew within the budget.
            final_status = "MAX_TURNS"
    finally:
        # Never let persistence errors mask the exception that ended the session.
        try:
            metrics["finished_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            metrics["session_wall_s"] = round(time.perf_counter() - session_t0, 3)
            if (
                metrics.get("first_gpu_call_start_t") is not None
                and metrics.get("last_gpu_call_end_t") is not None
            ):
                metrics["gpu_container_billed_s_est"] = round(
                    (metrics["last_gpu_call_end_t"] - metrics["first_gpu_call_start_t"])
                    + SCALEDOWN_WINDOW_S,
                    1,
                )
                # Lower bound when the keep-alive is shorter than a max code block: a
                # rare full-length block may force a cold start not counted here.
                metrics["gpu_container_billed_s_est_is_lower_bound"] = (
                    SCALEDOWN_WINDOW_S < CODE_TIMEOUT_S
                )
            metrics_summary: dict = {}
            try:
                metrics["phases"]["loop_wall_s"] = round(
                    sum(float(t.get("wall_s") or 0.0) for t in metrics["turns"]), 3
                )
                metrics_summary = _summarize_metrics(metrics)
                metrics["summary"] = metrics_summary
            except Exception as metrics_err:  # noqa: BLE001 - metrics are best-effort
                print(
                    "[session] failed to summarize metrics: "
                    f"{type(metrics_err).__name__}: {metrics_err}"
                )
                metrics_summary = {"error": f"{type(metrics_err).__name__}: {metrics_err}"}
                metrics["summary"] = metrics_summary

            # Persist metrics before the transcript summary. If listing edits/candidates
            # or dumping the transcript raises, Stage-0 data still reaches the Volume.
            try:
                metrics_path = os.path.join(work_dir, "metrics.json")
                tmp_metrics = f"{metrics_path}.tmp.{os.getpid()}"
                with open(tmp_metrics, "w") as f:
                    json.dump(metrics, f, indent=2, default=str)
                os.replace(tmp_metrics, metrics_path)
            except Exception as metrics_err:  # noqa: BLE001
                print(
                    "[session] failed to write metrics.json: "
                    f"{type(metrics_err).__name__}: {metrics_err}"
                )

            def _safe_helper(name: str, default):
                try:
                    return helpers[name]()
                except Exception as helper_err:  # noqa: BLE001
                    print(
                        f"[session] helper {name} failed during summary: "
                        f"{type(helper_err).__name__}: {helper_err}"
                    )
                    return default

            gpu_fp = (metrics.get("gpu_env") or {}).get("fingerprint", {})
            orch_fp = metrics.get("orchestrator_env") or {}
            summary = {
                "session_id": session_id,
                "dataset_mode": dataset_mode,
                "model_id": MODEL_ID,
                "max_tokens": max_tokens,
                "diag_n": diag_n,
                "val_n": val_n,
                "val_samples": val_samples,
                "wall_limit_hours": wall_limit_hours,
                "max_gen_answers": max_gen_answers,
                "read_max_length": effective_read_max_length,
                "config_path": config_path,
                "deviation": deviation,
                "protocol_deviations": protocol_deviations,
                "resolved_config": resolved_config,
                "reset_active_lora": reset_active_lora,
                "overwrite_session": overwrite_session,
                "resumed": resumed,
                "previous_passes_used": previous_passes_used,
                "turns_used_prior": turns_used_prior,
                "elapsed_wall_prior_s": round(elapsed_wall_prior_s, 3),
                "resume_lineage": resume_lineage,
                "worker_timeout_s": worker_timeout_s,
                "final_status": final_status,
                "turns_completed": min(len(transcript), max_turns),
                "turns_recorded": len(transcript),
                "passes_used": passes_used,
                "code_runs_used": code_runs_used,
                "active_edits": _safe_helper("list_edits", []),
                "saved_candidates": _safe_helper("list_candidates", []),
                "submitted_lora": os.path.isdir(submitted_lora_dir),
                "elapsed_sec": round(time.time() - start_time, 1),
                "session_wall_s": metrics["session_wall_s"],
                "gpu": gpu_fp.get("gpu"),
                "gpu_capability": gpu_fp.get("gpu_capability"),
                "model_revision": metrics.get("model_revision")
                or gpu_fp.get("model_revision")
                or orch_fp.get("model_revision"),
                "file_sha256": gpu_fp.get("file_sha256") or orch_fp.get("file_sha256"),
                "dataset": dataset_meta,
                "engine_config": metrics.get("engine_config"),
                "engine_config_hash": metrics.get("engine_config_hash"),
                "cache_tag": metrics.get("cache_tag"),
                "metrics": metrics_summary,
            }
            with open(os.path.join(work_dir, "transcript.json"), "w") as f:
                json.dump(
                    {"summary": summary, "turns": transcript},
                    f,
                    indent=2,
                    default=str,
                )
            _safe_commit()
            print(f"Session saved ({final_status}): {json.dumps(summary, default=str)}")
        except Exception as e:  # noqa: BLE001 - persistence is best-effort here
            print(f"[session] failed to write final summary: {type(e).__name__}: {e}")
