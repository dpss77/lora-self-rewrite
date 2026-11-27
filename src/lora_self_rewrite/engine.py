from __future__ import annotations

# Mechanically ported from the donor `modal_app.py` (SHA-256
# fe71f74b0cebe933c155e03f59941c62ce402d92ca31ac2ce4e038c9b23dcbc2).
# Ported for structural parity; GPU behaviour must be re-validated before use.
import gc
import hashlib
import json
import multiprocessing as mp
import os
import shutil
import time
import traceback

import modal

from .constants import (
    COMPACT_TOKEN_LIMIT,
    EDITOR_MAX_TOKENS,
    GENERATION_CONFIG_HASH,
    GENERATION_SAMPLING,
    GPU_SPEC,
    JIT_CACHE_DIR,
    MAX_MODEL_LEN,
    MAX_READ_TEXTS,
    MODEL_ID,
    MODEL_REVISION,
    SCALEDOWN_WINDOW_S,
    VLLM_CACHE_DIR,
    WORKER_TIMEOUT_S,
    validate_compaction_budget,
)
from .engine_identity import (
    engine_config,
    engine_config_hash,
    require_deterministic_repeat,
)
from .hf_worker import _hf_extract_worker
from .instrumentation import (
    _fingerprint,
)
from .modal_wiring import _SECRETS, WORKSPACE_DIR, app, image, vol
from .modal_wiring import safe_reload as _safe_reload
from .protocol_shared import (
    apply_chat as _apply_chat,
)
from .protocol_shared import (
    format_math_prompt,
    is_looping_answer,
)
from .seeds import derive_sample_seed


@app.cls(
    image=image,
    # H200-only: cache tags are hardware-specific and H100 fallback is disallowed
    # (ADR-0001). The summary records the actual GPU.
    gpu=GPU_SPEC,
    memory=131072,  # 128 GiB host RAM for vLLM sleep(level=1) + HF spawn worker
    volumes={WORKSPACE_DIR: vol},
    secrets=_SECRETS,
    timeout=21600,  # 6h: wall_limit is 3h, but a late FINAL_SUBMIT gate can run long
    # 240s, based on measured code p95 (0.43s); avoids a long idle-billed GPU tail.
    scaledown_window=SCALEDOWN_WINDOW_S,
    retries=0,
)
class QwenSelfRewriteEngine:
    @modal.enter()
    def startup(self):
        import torch
        from transformers import AutoTokenizer
        from vllm import LLM

        self.llm = None
        self.tokenizer = None
        self._startup_error = None
        self._startup_timings = {}
        self._env_fingerprint = None
        self._lora_ids: dict[str, int] = {}
        self._next_lora_id = 1

        t_enter = time.perf_counter()
        try:
            # Inside the try: an unhandled failure in @modal.enter() boot-loops
            # containers; capturing it surfaces once via _check_ready instead.
            validate_compaction_budget(COMPACT_TOKEN_LIMIT, MAX_MODEL_LEN)
            os.makedirs(JIT_CACHE_DIR, exist_ok=True)
            os.makedirs(VLLM_CACHE_DIR, exist_ok=True)
            t0 = time.perf_counter()
            _safe_reload("gpu:startup")
            self._startup_timings["volume_reload_s"] = round(time.perf_counter() - t0, 3)

            t0 = time.perf_counter()
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
            self._startup_timings["tokenizer_s"] = round(time.perf_counter() - t0, 3)

            # Hybrid Mamba/DeltaNet note: each concurrent sequence reserves a state block.
            # The H100 fallback is 80GB like the A100, so keep max_num_seqs=128; raising
            # to 256 can crash on fallback. This also matches the 128-answer pass cap.
            t0 = time.perf_counter()
            self.llm = LLM(
                model=MODEL_ID,
                revision=MODEL_REVISION,
                dtype="bfloat16",
                enable_lora=True,
                max_loras=5,  # cache base + up to 4 candidates across calls (LRU headroom)
                max_lora_rank=16,
                max_model_len=MAX_MODEL_LEN,
                max_num_seqs=128,
                gpu_memory_utilization=0.88,
                enable_sleep_mode=True,
                # Prefix caching is NOT adapter-aware on this stack: a base-only call
                # between identical LoRA calls changes the LoRA output (stage1_02
                # preflight). That would confound base-vs-edited pairs and invalidate the
                # frozen-base cache. Correctness over editor-prefill speed.
                enable_prefix_caching=False,
            )
            self._startup_timings["llm_init_s"] = round(time.perf_counter() - t0, 3)
            self.engine_config = engine_config(
                model_id=MODEL_ID,
                model_revision=MODEL_REVISION,
                dtype="bfloat16",
                max_model_len=MAX_MODEL_LEN,
                max_num_seqs=128,
                max_loras=5,
                max_lora_rank=16,
                gpu_memory_utilization=0.88,
                enable_sleep_mode=True,
                enable_prefix_caching=False,
                trust_remote_code=True,
                generation_config_hash=GENERATION_CONFIG_HASH,
            )
            self.engine_config_hash = engine_config_hash(self.engine_config)
        except Exception:
            # Capture into self._startup_error instead of crashing @modal.enter():
            # an unhandled crash inside @modal.enter() causes Modal to boot retry
            # containers in a loop, whereas raising from @modal.method() stops immediately.
            self._startup_error = traceback.format_exc()
            print(f"[FAIL-FAST STARTUP ERROR]\n{self._startup_error}")
            if self.llm is not None:
                del self.llm
                self.llm = None
            gc.collect()
        finally:
            self._startup_timings["enter_total_s"] = round(time.perf_counter() - t_enter, 3)
            try:
                self._env_fingerprint = _fingerprint(torch, include_gpu=True)
                print("[env] " + json.dumps(self._env_fingerprint, default=str))
                print("[startup] " + json.dumps(self._startup_timings))
            except Exception as e:  # noqa: BLE001 - fingerprint is best-effort
                print(f"[fingerprint] failed: {type(e).__name__}: {e}")

    def _gpu_memory(self) -> dict:
        try:
            import torch

            free, total = torch.cuda.mem_get_info()
            return {
                "free_gib": round(free / 2**30, 2),
                "total_gib": round(total / 2**30, 2),
            }
        except Exception:  # noqa: BLE001
            return {}

    def _allocate_lora_id(self) -> int:
        """Monotonic positive vLLM LoRA id; zero is rejected/ignored by vLLM."""
        lora_id = self._next_lora_id
        if lora_id >= 2**31:
            raise RuntimeError("vLLM LoRA id space exhausted in this container.")
        self._next_lora_id += 1
        return lora_id

    def _lora_int_id(self, lora_dir: str) -> int:
        """Content-addressed vLLM LoRA id from adapter weights *and* config bytes.

        Identical candidates reuse one adapter-cache entry; an in-place active edit
        changes the digest and therefore gets a new id (never stale cached weights).
        """
        digest = hashlib.sha256()
        for name in ("adapter_model.safetensors", "adapter_config.json"):
            path = os.path.join(lora_dir, name)
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing LoRA file: {path}")
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    digest.update(chunk)
            digest.update(b"\0")
        key = digest.hexdigest()
        lora_id = self._lora_ids.get(key)
        if lora_id is None:
            lora_id = self._allocate_lora_id()
            self._lora_ids[key] = lora_id
        return lora_id

    def _check_ready(self):
        if self._startup_error is not None or self.llm is None:
            raise RuntimeError(
                f"Engine startup failed (fail-fast, no retries):\n{self._startup_error}"
            )

    @modal.exit()
    def shutdown(self):
        if getattr(self, "llm", None) is not None:
            del self.llm
            self.llm = None
            gc.collect()

    @modal.method()
    def get_environment(self) -> dict:
        """Return the measured GPU/container fingerprint for Stage-0 artifacts."""
        self._check_ready()
        return {
            "fingerprint": self._env_fingerprint or {},
            "startup": self._startup_timings,
            "gpu_memory": self._gpu_memory(),
            "engine_config": getattr(self, "engine_config", {}),
            "engine_config_hash": getattr(self, "engine_config_hash", None),
        }

    def _apply_chat(self, messages: list[dict]) -> str:
        return _apply_chat(self.tokenizer, messages)

    def _read_hidden_states_impl(
        self,
        prompt_strs: list[str],
        answer_strs: list[str],
        layers: list[int],
        extract_kinds: list[str],
        token_mode: str,
        k: int,
        lora_dir: str | None,
        out_path: str,
        max_length: int = 8192,
        worker_timeout_s: int = WORKER_TIMEOUT_S,
    ) -> dict:
        self._check_ready()
        if len(prompt_strs) != len(answer_strs):
            raise ValueError(
                f"prompt_strs ({len(prompt_strs)}) and answer_strs "
                f"({len(answer_strs)}) must have equal length."
            )
        _safe_reload("gpu:read")
        formatted_prompts = [
            format_math_prompt(self.tokenizer, p) if p else "" for p in prompt_strs[:MAX_READ_TEXTS]
        ]
        answers = answer_strs[:MAX_READ_TEXTS]

        if os.path.exists(out_path):
            os.remove(out_path)

        timings: dict = {"n_texts": len(answers)}
        result = None
        t_total0 = time.perf_counter()
        t0 = time.perf_counter()
        self.llm.sleep(level=1)
        timings["sleep_s"] = round(time.perf_counter() - t0, 3)
        proc = None
        try:
            ctx = mp.get_context("spawn")
            proc = ctx.Process(
                target=_hf_extract_worker,
                args=(
                    MODEL_ID,
                    lora_dir,
                    formatted_prompts,
                    answers,
                    layers,
                    extract_kinds,
                    token_mode,
                    k,
                    max_length,
                    out_path,
                ),
            )
            proc.start()
            t0 = time.perf_counter()
            proc.join(timeout=worker_timeout_s)
            timings["worker_wall_s"] = round(time.perf_counter() - t0, 3)
            # join(timeout) returning does NOT mean the child exited. If it is still
            # alive it still owns ~54 GB of weights, so wake_up() must not proceed.
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=30)
                if proc.is_alive():
                    proc.kill()
                    proc.join(timeout=30)
                raise RuntimeError(
                    f"HF extraction worker exceeded {worker_timeout_s}s and was killed."
                )
            if proc.exitcode != 0:
                raise RuntimeError(f"HF extraction worker failed (exitcode={proc.exitcode})")
            if not os.path.exists(out_path):
                raise RuntimeError(
                    f"HF extraction worker exited 0 but wrote no payload at {out_path}."
                )
            timings["payload_bytes"] = os.path.getsize(out_path)
            sidecar = f"{out_path}.timings.json"
            if os.path.exists(sidecar):
                with open(sidecar) as f:
                    timings["worker"] = json.load(f)
            # Strict commit: the orchestrator runs in a different container and can
            # only see the payload after commit + reload.
            t0 = time.perf_counter()
            vol.commit()
            timings["commit_s"] = round(time.perf_counter() - t0, 3)
            result = {
                "status": "ok",
                "out_path": out_path,
                "n_texts": len(answers),
            }
        finally:
            # Defence in depth: never wake vLLM under a child that still holds VRAM.
            if proc is not None and proc.is_alive():
                proc.kill()
                proc.join(timeout=30)
            t0 = time.perf_counter()
            self.llm.wake_up()
            timings["wake_s"] = round(time.perf_counter() - t0, 3)

        timings["total_s"] = round(time.perf_counter() - t_total0, 3)
        timings["fixed_overhead_s"] = round(
            timings.get("sleep_s", 0.0)
            + timings.get("wake_s", 0.0)
            + float(timings.get("worker", {}).get("load_s", 0.0)),
            3,
        )
        assert result is not None
        result["timings"] = timings
        return result

    @modal.method()
    def preflight_check(
        self,
        sample_peft_prefix: str,
        sample_shape: tuple[int, int],
        test_sleep: bool = False,
    ) -> dict:
        """
        Fast GPU preflight before starting diagnosis:
        1. Confirms EngineCore started cleanly.
        2. Writes a deliberately large 1-module LoRA and asserts the adapter changes
           output tokens *or* per-position logprobs vs. base weights.
        3. If `test_sleep=True`, verifies one `sleep -> spawn HF -> wake_up` cycle.
        """
        import torch
        from safetensors.torch import save_file
        from vllm import SamplingParams
        from vllm.lora.request import LoRARequest

        self._check_ready()
        tmp_lora_dir = "/tmp/preflight_lora"
        out_pt = "/tmp/preflight_hs.pt"
        shutil.rmtree(tmp_lora_dir, ignore_errors=True)
        os.makedirs(tmp_lora_dir, exist_ok=True)
        if os.path.exists(out_pt):
            os.remove(out_pt)
        # A tiny perturbation can leave the argmax and first-token logprobs unchanged,
        # making the guard flaky. Rank-4 N(0,1)*1.0 is already large relative to model
        # weights (LoRA scale alpha/r = 1), while aggregating differences across all
        # generated positions makes "adapter ignored" unambiguous without risking NaN.
        noise_scale = 1.0
        t_preflight0 = time.perf_counter()
        timings: dict = {}
        try:
            out_f, in_f = sample_shape
            torch.manual_seed(42)
            tensors = {
                f"{sample_peft_prefix}.lora_A.weight": (
                    torch.randn((4, in_f), dtype=torch.float32) * noise_scale
                ).to(torch.bfloat16),
                f"{sample_peft_prefix}.lora_B.weight": (
                    torch.randn((out_f, 4), dtype=torch.float32) * noise_scale
                ).to(torch.bfloat16),
            }
            cfg = {
                "peft_type": "LORA",
                "base_model_name_or_path": MODEL_ID,
                "r": 4,
                "lora_alpha": 4,
                "lora_dropout": 0.0,
                "target_modules": ["down_proj"],
                "layers_to_transform": [8],
                "bias": "none",
                "task_type": "CAUSAL_LM",
            }
            with open(os.path.join(tmp_lora_dir, "adapter_config.json"), "w") as f:
                json.dump(cfg, f)
            save_file(tensors, os.path.join(tmp_lora_dir, "adapter_model.safetensors"))

            p = format_math_prompt(self.tokenizer, "Compute 17 * 23.")
            gp = SamplingParams(temperature=0.0, max_tokens=8, logprobs=20)
            t0 = time.perf_counter()
            base_out = self.llm.generate([p], gp, use_tqdm=False)[0].outputs[0]

            lora_id = self._allocate_lora_id()
            lora_req = LoRARequest(f"preflight_{lora_id}", lora_id, tmp_lora_dir)
            lora_out = self.llm.generate(
                [p],
                gp,
                lora_request=lora_req,
                use_tqdm=False,
            )[0].outputs[0]
            timings["probe_generate_s"] = round(time.perf_counter() - t0, 3)

            if base_out.logprobs is None or lora_out.logprobs is None:
                raise RuntimeError("FAIL-FAST: preflight requested logprobs but got None.")

            max_lp_diff = 0.0
            sum_lp_diff = 0.0
            for b_pos, l_pos in zip(base_out.logprobs, lora_out.logprobs, strict=False):
                b_lp = {t: lp.logprob for t, lp in b_pos.items()}
                l_lp = {t: lp.logprob for t, lp in l_pos.items()}
                for t in set(b_lp) & set(l_lp):
                    d = abs(b_lp[t] - l_lp[t])
                    max_lp_diff = max(max_lp_diff, d)
                    sum_lp_diff += d

            tokens_differ = list(base_out.token_ids) != list(lora_out.token_ids)
            if not (tokens_differ or max_lp_diff > 1e-2 or sum_lp_diff > 1e-1):
                raise RuntimeError(
                    f"FAIL-FAST: vLLM ignored LoRA '{sample_peft_prefix}' "
                    f"(tokens_differ={tokens_differ}, max_lp_diff={max_lp_diff:.6f}, "
                    f"sum_lp_diff={sum_lp_diff:.6f})."
                )

            # Frozen-base cache safety: does a base-only call between two identical LoRA
            # calls change the LoRA output? If vLLM prefix caching leaks base KV into an
            # adapter generation, the whole cache strategy is invalid. Fail fast.
            t0 = time.perf_counter()
            gp_ci = SamplingParams(temperature=0.0, max_tokens=32, seed=1234)
            lora_warm = self.llm.generate([p], gp_ci, lora_request=lora_req, use_tqdm=False)[
                0
            ].outputs[0]
            lora_ref = self.llm.generate([p], gp_ci, lora_request=lora_req, use_tqdm=False)[
                0
            ].outputs[0]
            base_mid = self.llm.generate([p], gp_ci, use_tqdm=False)[0].outputs[0]  # base between
            lora_after = self.llm.generate([p], gp_ci, lora_request=lora_req, use_tqdm=False)[
                0
            ].outputs[0]
            deterministic_repeat = list(lora_warm.token_ids) == list(lora_ref.token_ids)
            base_between_stable = list(lora_ref.token_ids) == list(lora_after.token_ids)
            timings["cache_isolation_s"] = round(time.perf_counter() - t0, 3)
            require_deterministic_repeat(deterministic_repeat)
            if not base_between_stable:
                raise RuntimeError(
                    "FAIL-FAST: a base-only call between identical LoRA calls changed "
                    "the LoRA output (vLLM prefix-cache / adapter interaction). "
                    "The frozen-base cache cannot be trusted. "
                    f"ref_ids={list(lora_ref.token_ids)[:16]}, "
                    f"after_ids={list(lora_after.token_ids)[:16]}, "
                    f"base_ids={list(base_mid.token_ids)[:16]}"
                )

            if test_sleep:
                t0 = time.perf_counter()
                read_res = self._read_hidden_states_impl(
                    prompt_strs=["Compute 17 * 23."],
                    answer_strs=["<think>391</think>\\boxed{391}"],
                    layers=[8],
                    extract_kinds=["hidden_states"],
                    token_mode="answer_mean",
                    k=1,
                    lora_dir=tmp_lora_dir,
                    out_path=out_pt,
                    max_length=512,
                )
                timings["sleep_test_s"] = round(time.perf_counter() - t0, 3)
                timings["read_timings"] = read_res.get("timings", {})
                post_wake = self.llm.generate(
                    [p], SamplingParams(temperature=0.0, max_tokens=4), use_tqdm=False
                )[0].outputs[0]
                if len(post_wake.token_ids) == 0:
                    raise RuntimeError("FAIL-FAST: Post-wake generation returned 0 tokens.")

            timings["total_s"] = round(time.perf_counter() - t_preflight0, 3)
            return {
                "lora_applied": True,
                "tokens_differ": tokens_differ,
                "max_lp_diff": round(max_lp_diff, 4),
                "sum_lp_diff": round(sum_lp_diff, 4),
                "noise_scale": noise_scale,
                "sleep_tested": test_sleep,
                "cache_isolation": {
                    "deterministic_repeat": deterministic_repeat,
                    "base_between_stable": base_between_stable,
                },
                "gpu_memory": self._gpu_memory(),
                "timings": timings,
            }
        finally:
            shutil.rmtree(tmp_lora_dir, ignore_errors=True)
            if os.path.exists(out_pt):
                os.remove(out_pt)

    @modal.method()
    def editor_step(self, messages: list[dict], max_tokens: int = EDITOR_MAX_TOKENS) -> dict:
        from vllm import SamplingParams

        self._check_ready()
        prompt = self._apply_chat(messages)
        prompt_tokens = len(self.tokenizer.encode(prompt))
        params = SamplingParams(
            temperature=GENERATION_SAMPLING["temperature"],
            top_p=GENERATION_SAMPLING["top_p"],
            top_k=GENERATION_SAMPLING["top_k"],
            max_tokens=max_tokens,
        )
        t0 = time.perf_counter()
        out = self.llm.generate([prompt], params, use_tqdm=False)[0].outputs[0]
        wall_s = time.perf_counter() - t0
        gen_tokens = len(out.token_ids)
        return {
            "text": out.text,
            "tokens": gen_tokens,
            "finish_reason": out.finish_reason,
            "prompt_tokens": prompt_tokens,
            "gpu_memory": self._gpu_memory(),
            "timings": {
                "wall_s": round(wall_s, 3),
                "prompt_tokens": prompt_tokens,
                "gen_tokens": gen_tokens,
                "gen_tokens_per_s": round(gen_tokens / wall_s, 2) if wall_s > 0 else None,
            },
        }

    @modal.method()
    def generate_batch(
        self,
        problems: list[dict],
        lora_specs: dict[str, str],
        include_base: bool = True,
        n_samples: int = 1,
        max_tokens: int = 4096,
    ) -> dict:
        """
        Runs unedited base + up to 4 LoRA candidates on identical deterministic
        per-(problem_id, sample_idx) seeds.

        Base and each candidate are separate `llm.generate` calls. They are *not* fused
        into one multi-adapter batch (not all vLLM builds accept a per-prompt adapter
        list, and mixing base with adapters in one call is version-dependent).
        `max_loras=5` only lets the adapter cache hold base+candidates without LRU
        reloads across calls/turns.

        Every candidate gets a fresh `LoRARequest` id so vLLM never serves stale
        weights after an in-place disk edit.
        """
        from vllm import SamplingParams
        from vllm.lora.request import LoRARequest

        self._check_ready()
        t_total0 = time.perf_counter()
        timings: dict = {}
        t0 = time.perf_counter()
        _safe_reload("gpu:generate")
        timings["reload_s"] = round(time.perf_counter() - t0, 3)
        mem_before = self._gpu_memory()

        if not problems:
            raise ValueError("generate_batch requires at least one problem.")
        if not include_base and not lora_specs:
            raise ValueError("generate_batch has nothing to run (no base, no Lora specs).")

        t0 = time.perf_counter()
        prompts = []
        params_list = []
        meta = []
        for p in problems:
            prompt_str = format_math_prompt(self.tokenizer, p["problem"])
            for s_idx in range(n_samples):
                seed = derive_sample_seed(p["id"], s_idx)
                prompts.append(prompt_str)
                params_list.append(
                    SamplingParams(
                        temperature=GENERATION_SAMPLING["temperature"],
                        top_p=GENERATION_SAMPLING["top_p"],
                        top_k=GENERATION_SAMPLING["top_k"],
                        max_tokens=max_tokens,
                        seed=seed,
                    )
                )
                meta.append(
                    {
                        "id": p["id"],
                        "sample_idx": s_idx,
                        "problem": p["problem"],
                        "answer": p["answer"],
                    }
                )
        timings["setup_s"] = round(time.perf_counter() - t0, 3)

        def _extract_outputs(outs):
            rows = []
            for m, out_req in zip(meta, outs, strict=True):
                out = out_req.outputs[0]
                t_ids = list(out.token_ids)
                cut = len(t_ids) >= max_tokens or out.finish_reason == "length"
                rows.append(
                    {
                        **m,
                        "text": out.text,
                        "tokens": len(t_ids),
                        "finish_reason": out.finish_reason,
                        "cut": cut,
                        "loop": is_looping_answer(out.text, cut),
                    }
                )
            return rows

        results = {"base": None, "candidates": {}}
        if include_base:
            t0 = time.perf_counter()
            base_outs = self.llm.generate(prompts, params_list, use_tqdm=False)
            results["base"] = _extract_outputs(base_outs)
            timings["base_s"] = round(time.perf_counter() - t0, 3)

        timings["candidates_s"] = {}
        for cname, cdir in lora_specs.items():
            if not (cdir and os.path.exists(os.path.join(cdir, "adapter_model.safetensors"))):
                raise FileNotFoundError(f"Missing LoRA adapter for '{cname}' at {cdir}")
            lora_id = self._lora_int_id(cdir)
            lora_req = LoRARequest(f"lora_{lora_id}", lora_id, cdir)
            t0 = time.perf_counter()
            c_outs = self.llm.generate(prompts, params_list, lora_request=lora_req, use_tqdm=False)
            results["candidates"][cname] = _extract_outputs(c_outs)
            timings["candidates_s"][cname] = round(time.perf_counter() - t0, 3)

        all_rows = list(results["base"] or [])
        for rows in results["candidates"].values():
            all_rows.extend(rows)
        timings["total_s"] = round(time.perf_counter() - t_total0, 3)
        timings["n_problems"] = len(problems)
        timings["n_samples"] = n_samples
        timings["n_prompts"] = len(prompts)
        timings["n_answers"] = len(prompts) * (int(include_base) + len(lora_specs))
        timings["generated_tokens"] = int(sum(r.get("tokens", 0) for r in all_rows))
        results["timings"] = timings
        results["gpu_memory"] = {"before": mem_before, "after": self._gpu_memory()}
        return results

    @modal.method()
    def read_hidden_states(
        self,
        prompt_strs: list[str],
        answer_strs: list[str],
        layers: list[int],
        extract_kinds: list[str],
        token_mode: str,
        k: int,
        lora_dir: str | None,
        out_path: str,
        max_length: int = 8192,
        worker_timeout_s: int = WORKER_TIMEOUT_S,
    ) -> dict:
        return self._read_hidden_states_impl(
            prompt_strs=prompt_strs,
            answer_strs=answer_strs,
            layers=layers,
            extract_kinds=extract_kinds,
            token_mode=token_mode,
            k=k,
            lora_dir=lora_dir,
            out_path=out_path,
            max_length=max_length,
            worker_timeout_s=worker_timeout_s,
        )
