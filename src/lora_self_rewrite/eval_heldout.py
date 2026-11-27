"""Held-out evaluation (PAPER: Table 5) — H200-only, prefix caching OFF.

Design constraints (docs/03-protocol/INVARIANTS.md):
  * strictly ``gpu="H200"`` (no H100 fallback; cache tags must not mix)
  * ``enable_prefix_caching=False``
  * exact ordered problem/sample batch per shard; whole problems per shard
  * a single CPU orchestrator writes results (GPU shards return rows in memory)
  * 490 problems x 8 samples at 16,384 tokens
  * problem-clustered one-sided sign-flip test (100k resamples)

This module is deliberately separate from ``modal_app.py``: importing the
session app from a second ``modal.App`` risks image/app hydration conflicts.
"""

from __future__ import annotations

import json
import os
import pathlib
import sys

# Bootstrap package imports for direct execution:
#   modal run src/lora_self_rewrite/eval_heldout.py -- ...
if __package__ in (None, ""):
    _SRC_ROOT = pathlib.Path(__file__).resolve().parents[1]
    if str(_SRC_ROOT) not in sys.path:
        sys.path.insert(0, str(_SRC_ROOT))
    __package__ = "lora_self_rewrite"

import modal

from .caches import (
    LoraIdRegistry,
    content_addressed_lora_key,
    load_base_batch_cache,
    save_base_batch_cache,
)
from .constants import (
    GENERATION_CONFIG_HASH,
    GENERATION_SAMPLING,
    HELDOUT_GPU,
    HELDOUT_K_SHARDS,
    HELDOUT_MAX_TOKENS,
    HELDOUT_N_SAMPLES,
    HELDOUT_PREFIX_CACHING,
    HELDOUT_SIGNFLIP_RESAMPLES,
    HELDOUT_SLICE,
    MODEL_ID,
    MODEL_REVISION,
    WORKSPACE_DIR,
)
from .datasets import (
    heldout_dataset_identity,
    heldout_problems,
    require_pinned_sources,
    validate_heldout_pool,
)
from .eval_common import (
    assert_shard_identity,
    cache_tag_for,
    cut_at_token_limit,
    heldout_base_signature,
    heldout_lora_int_id,
    reload_with_retry,
    shard_problems,
)
from .heldout_gate import (
    available_package_versions,
    eval_engine_config,
    eval_engine_config_hash,
    validate_heldout_adapter,
    validate_heldout_run_args,
)
from .instrumentation import get_model_revision
from .paths import resolve_under
from .protocol_shared import format_math_prompt, grade_answer, is_looping_answer
from .seeds import derive_sample_seed
from .stats import compute_paired_stats

APP_NAME = "qwen35-27b-heldout-eval"
VOLUME_NAME = "self-rewrite-workspace"

app = modal.App(APP_NAME)
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

# Optional HF auth: keep it opt-in, mirroring the session wiring.
_SECRETS = (
    [modal.Secret.from_name(os.environ["HF_SECRET_NAME"])]
    if os.environ.get("HF_SECRET_NAME")
    else []
)

EVAL_DTYPE = "bfloat16"
EVAL_MAX_MODEL_LEN = 32768
EVAL_MAX_NUM_SEQS = 64
EVAL_MAX_LORAS = 2
EVAL_MAX_LORA_RANK = 16
EVAL_GPU_MEMORY_UTILIZATION = 0.88
EVAL_PREFIX_CACHING = HELDOUT_PREFIX_CACHING
EVAL_TRUST_REMOTE_CODE = True

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "vllm==0.34.0",
        "transformers==5.22.2",
        "peft==0.22.0",
        "accelerate>=1.0.0",
        "safetensors>=0.4.5",
        # Grading is INVARIANT 11; pin the resolved version so the held-out arm
        # cannot drift from the adaptation arm's grader.
        "math-verify==0.9.0",
        "datasets",
        "huggingface_hub",
    )
    .env({"HF_HOME": f"{WORKSPACE_DIR}/hf_cache", "PYTHONUNBUFFERED": "1"})
    .add_local_python_source("lora_self_rewrite")
)


@app.cls(
    image=image,
    gpu=HELDOUT_GPU,  # strictly H200
    memory=131072,
    volumes={WORKSPACE_DIR: vol},
    secrets=_SECRETS,
    timeout=21600,
    scaledown_window=240,
    retries=0,
)
class HeldoutEngine:
    @modal.enter()
    def startup(self):
        import torch
        from transformers import AutoTokenizer
        from vllm import LLM

        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION, trust_remote_code=True
        )
        self.llm = LLM(
            model=MODEL_ID,
            revision=MODEL_REVISION,
            dtype=EVAL_DTYPE,
            enable_lora=True,
            max_loras=EVAL_MAX_LORAS,
            max_lora_rank=EVAL_MAX_LORA_RANK,
            max_model_len=EVAL_MAX_MODEL_LEN,
            max_num_seqs=EVAL_MAX_NUM_SEQS,
            gpu_memory_utilization=EVAL_GPU_MEMORY_UTILIZATION,
            enable_prefix_caching=EVAL_PREFIX_CACHING,
            trust_remote_code=EVAL_TRUST_REMOTE_CODE,
        )
        self.torch = torch
        self.vllm_version = __import__("vllm").__version__
        self.gpu_name = torch.cuda.get_device_name(0)
        self.model_revision = get_model_revision() or MODEL_REVISION
        self.eval_engine_config = eval_engine_config(
            model_id=MODEL_ID,
            model_revision=self.model_revision,
            dtype=EVAL_DTYPE,
            max_model_len=EVAL_MAX_MODEL_LEN,
            max_num_seqs=EVAL_MAX_NUM_SEQS,
            max_loras=EVAL_MAX_LORAS,
            max_lora_rank=EVAL_MAX_LORA_RANK,
            gpu_memory_utilization=EVAL_GPU_MEMORY_UTILIZATION,
            enable_prefix_caching=EVAL_PREFIX_CACHING,
            trust_remote_code=EVAL_TRUST_REMOTE_CODE,
            generation_config_hash=GENERATION_CONFIG_HASH,
            sleep_mode=False,
        )
        self.eval_engine_config_hash = eval_engine_config_hash(self.eval_engine_config)
        self.lora_ids = LoraIdRegistry()

    @modal.method()
    def generate_shard(
        self,
        problems: list[dict],
        n_samples: int,
        max_tokens: int,
        lora_dir: str | None = None,
    ) -> dict:
        from vllm import SamplingParams
        from vllm.lora.request import LoRARequest

        prompts: list[str] = []
        sampling: list = []
        index: list[tuple[int, int]] = []
        for problem in problems:
            for sample_idx in range(n_samples):
                prompts.append(format_math_prompt(self.tokenizer, problem["problem"]))
                sampling.append(
                    SamplingParams(
                        temperature=GENERATION_SAMPLING["temperature"],
                        top_p=GENERATION_SAMPLING["top_p"],
                        top_k=GENERATION_SAMPLING["top_k"],
                        max_tokens=max_tokens,
                        seed=derive_sample_seed(problem["id"], sample_idx),
                    )
                )
                index.append((problem["id"], sample_idx))

        lora_request = None
        measured_adapter_sha256 = None
        if lora_dir:
            # Reload (retry-safe) so we measure/load the committed adapter, not a
            # stale mount.
            reload_with_retry(vol, "gpu:generate_shard")
            measured_adapter_sha256 = content_addressed_lora_key(lora_dir)
            lora_request = LoRARequest(
                lora_name="edited",
                lora_int_id=heldout_lora_int_id(self.lora_ids, lora_dir),
                lora_path=lora_dir,
            )

        outputs = self.llm.generate(
            prompts,
            sampling,
            lora_request=lora_request,
            use_tqdm=False,
        )
        by_id = {p["id"]: p for p in problems}
        rows = []
        for (pid, sample_idx), request_output in zip(index, outputs, strict=True):
            out = request_output.outputs[0]
            text = out.text
            tokens = len(out.token_ids)
            cut = cut_at_token_limit(tokens, max_tokens, out.finish_reason)
            rows.append(
                {
                    "id": pid,
                    "sample_idx": sample_idx,
                    "problem": by_id[pid]["problem"],
                    "answer": by_id[pid]["answer"],
                    "text": text,
                    "tokens": tokens,
                    "cut": cut,
                    "loop": is_looping_answer(text, cut),
                    "correct": grade_answer(text, by_id[pid]["answer"]),
                }
            )
        return {
            "rows": rows,
            "gpu_name": self.gpu_name,
            "vllm_version": self.vllm_version,
            "cache_tag": cache_tag_for(self.gpu_name, self.torch.__version__, self.vllm_version),
            "eval_engine_config_hash": self.eval_engine_config_hash,
            "adapter_sha256": measured_adapter_sha256,
        }

    @modal.method()
    def get_environment(self) -> dict:
        """Cheap identity probe: no generation, used to key the base cache."""
        return {
            "gpu_name": self.gpu_name,
            "torch_version": self.torch.__version__,
            "vllm_version": self.vllm_version,
            "model_id": MODEL_ID,
            "model_revision": self.model_revision,
            "cache_tag": cache_tag_for(self.gpu_name, self.torch.__version__, self.vllm_version),
            "eval_engine_config": self.eval_engine_config,
            "eval_engine_config_hash": self.eval_engine_config_hash,
            "package_versions": available_package_versions(),
        }


def _load_payload(dataset_path: str) -> dict:
    with open(dataset_path) as f:
        return json.load(f)


def _base_rows_path(volume_root: str, cache_tag: str) -> str:
    return os.path.join(volume_root, "eval", f"heldout_base_{cache_tag}.json")


@app.function(
    image=image,
    cpu=4.0,
    memory=16384,
    volumes={WORKSPACE_DIR: vol},
    secrets=_SECRETS,
    timeout=21600,
    retries=0,
)
def run_heldout(
    dataset_path: str,
    adapter_dir: str | None = None,
    k_shards: int = HELDOUT_K_SHARDS,
    max_tokens: int = HELDOUT_MAX_TOKENS,
    n_samples: int = HELDOUT_N_SAMPLES,
    smoke_problems: int | None = None,
    base_only: bool = False,
    dev_override: bool = False,
    out_path: str = f"{WORKSPACE_DIR}/eval/heldout_results.json",
) -> dict:
    """Single-writer orchestrator: compute base (cached) + edited, then stats."""
    dataset_path = resolve_under(WORKSPACE_DIR, dataset_path)
    out_path = resolve_under(WORKSPACE_DIR, out_path)
    if adapter_dir is not None:
        adapter_dir = resolve_under(os.path.join(WORKSPACE_DIR, "sessions"), adapter_dir)
    payload = _load_payload(dataset_path)
    # Hard-fails unless the pool is exactly HELDOUT_N_PROBLEMS, except for an
    # explicit smoke run.
    problems = validate_heldout_pool(heldout_problems(payload), smoke_problems=smoke_problems)
    validate_heldout_run_args(
        n_problems=len(problems),
        n_samples=n_samples,
        max_tokens=max_tokens,
        k_shards=k_shards,
        smoke_problems=smoke_problems,
        dev_override=dev_override,
    )
    dataset_meta = heldout_dataset_identity(payload, problems)
    if not dev_override:
        expected_revisions = require_pinned_sources("deepscaler")
        payload_revisions = (payload.get("meta") or {}).get("source_revisions") or {}
        required_repos = [repo for repo in expected_revisions if repo != "Idavidrein/gpqa"]
        missing = [
            repo
            for repo in required_repos
            if not payload_revisions.get(repo)
            or str(payload_revisions[repo]).strip().lower() in {"unknown", "none", "null"}
        ]
        if missing:
            raise ValueError(
                "Held-out dataset payload has missing/unpinned source revisions: "
                f"{missing}. Rebuild the dataset with the pinned loader."
            )
        if not dataset_meta.get("identity"):
            raise ValueError("Held-out dataset identity is missing or unpinned.")
        if not MODEL_REVISION or MODEL_REVISION == "unknown":
            raise ValueError("MODEL_REVISION must be pinned for held-out evaluation.")

    # Gate the adapter before allocating any GPU time. ``--base-only`` precompute
    # is allowed without an adapter; an edited run needs a SUBMIT provenance and
    # a non-zero adapter from the session's WORK/submitted_lora.
    if adapter_dir:
        reload_with_retry(vol, "orchestrator:adapter-validation")
    adapter_provenance = validate_heldout_adapter(adapter_dir, base_only=base_only)

    shards = shard_problems(problems, k_shards)
    engine = HeldoutEngine()

    # --- base arm: keyed by exact ordered batch + batch composition + revision ---
    env = engine.get_environment.remote()
    cache_tag = env["cache_tag"]
    engine_config = env.get("eval_engine_config") or {}
    engine_config_hash = env.get("eval_engine_config_hash")
    recomputed_engine_hash = eval_engine_config_hash(engine_config)
    if engine_config_hash != recomputed_engine_hash:
        raise RuntimeError(
            "Held-out eval engine config hash mismatch: "
            f"{engine_config_hash!r} != {recomputed_engine_hash!r}."
        )
    if not dev_override:
        if "H200" not in str(env.get("gpu_name", "")):
            raise RuntimeError(f"Held-out evaluation requires H200, got {env.get('gpu_name')!r}.")
        if str(env.get("model_revision")) != MODEL_REVISION:
            raise RuntimeError(
                "Held-out model revision is not the pinned protocol revision: "
                f"{env.get('model_revision')!r} != {MODEL_REVISION!r}."
            )
        if engine_config.get("enable_prefix_caching") is not False:
            raise RuntimeError("Held-out evaluation requires prefix caching OFF.")
        if int(engine_config.get("max_model_len") or 0) < int(max_tokens):
            raise RuntimeError("Held-out eval engine max_model_len is smaller than max_tokens.")

    signature, meta = heldout_base_signature(
        problems,
        n_samples,
        max_tokens,
        cache_tag,
        k_shards,
        env.get("model_revision"),
        engine_config_hash=engine_config_hash,
        dataset_identity=dataset_meta["identity"],
    )
    base_path = _base_rows_path(WORKSPACE_DIR, signature)
    expected = [
        {
            "id": p["id"],
            "sample_idx": s_idx,
            "problem": p["problem"],
            "answer": p["answer"],
        }
        for p in problems
        for s_idx in range(n_samples)
    ]
    # Re-grades stored rows with the current grader; misses on any identity drift.
    base_rows = load_base_batch_cache(base_path, meta, expected, grade_answer)
    if base_rows is None:
        base_rows = []
        for shard in shards:
            shard_result = engine.generate_shard.remote(
                shard, n_samples=n_samples, max_tokens=max_tokens
            )
            assert_shard_identity(
                shard_result,
                expected_cache_tag=cache_tag,
                expected_eval_engine_config_hash=engine_config_hash,
            )
            base_rows.extend(shard_result["rows"])
        save_base_batch_cache(base_path, meta, base_rows)
        vol.commit()
        print(f"[heldout] base computed and cached: {base_path}")
    else:
        print(f"[heldout] base cache hit: {base_path}")

    result = {
        "schema_version": 1,
        "experiment_id": "EVAL-HELDOUT-490",
        "slice": list(HELDOUT_SLICE),
        "heldout_slice": list(HELDOUT_SLICE),
        "n_problems": len(problems),
        "n_samples": n_samples,
        "max_tokens": max_tokens,
        "k_shards": k_shards,
        "cache_tag": cache_tag,
        "model_id": env.get("model_id", MODEL_ID),
        "model_revision": env.get("model_revision"),
        "eval_engine_config": engine_config,
        "eval_engine_config_hash": engine_config_hash,
        "package_versions": env.get("package_versions") or available_package_versions(),
        "environment": {
            "gpu_name": env.get("gpu_name"),
            "torch_version": env.get("torch_version"),
            "vllm_version": env.get("vllm_version"),
            "model_id": env.get("model_id", MODEL_ID),
            "model_revision": env.get("model_revision"),
            "eval_engine_config": engine_config,
            "eval_engine_config_hash": engine_config_hash,
            "package_versions": env.get("package_versions") or available_package_versions(),
        },
        "dataset": dataset_meta,
        "base_cache_signature": signature,
        "dataset_path": dataset_path,
        "adapter_dir": adapter_dir,
        "adapter_provenance": adapter_provenance,
        "base_only": base_only,
        "dev_override": dev_override,
        "base": {"rows": base_rows},
    }

    if adapter_dir and not base_only:
        # Reload (retry-safe) so the digest covers the committed adapter bytes.
        reload_with_retry(vol, "orchestrator:adapter-hash")
        adapter_sha = content_addressed_lora_key(adapter_dir)
        edited_rows = []
        verified_adapter_sha = None
        for shard in shards:
            shard_result = engine.generate_shard.remote(
                shard,
                n_samples=n_samples,
                max_tokens=max_tokens,
                lora_dir=adapter_dir,
            )
            # The shard measured its own mounted bytes; compare against our digest.
            verified_adapter_sha = shard_result.get("adapter_sha256")
            assert_shard_identity(
                shard_result,
                expected_cache_tag=cache_tag,
                expected_adapter_sha256=adapter_sha,
                expected_eval_engine_config_hash=engine_config_hash,
            )
            edited_rows.extend(shard_result["rows"])
        result["edited"] = {
            "rows": edited_rows,
            "adapter_sha256": verified_adapter_sha,
        }
        result["adapter_sha256"] = verified_adapter_sha
        result["stats"] = compute_paired_stats(
            base_rows,
            edited_rows,
            signflip_resamples=HELDOUT_SIGNFLIP_RESAMPLES,
        )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp = f"{out_path}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(result, f, indent=2)
    os.replace(tmp, out_path)
    vol.commit()
    print(f"[heldout] wrote {out_path}")
    return {"out_path": out_path, "stats": result.get("stats")}


@app.local_entrypoint()
def main(
    dataset_path: str = f"{WORKSPACE_DIR}/datasets/deepscaler_scrubbed.json",
    adapter_dir: str | None = None,
    k_shards: int = HELDOUT_K_SHARDS,
    max_tokens: int = HELDOUT_MAX_TOKENS,
    n_samples: int = HELDOUT_N_SAMPLES,
    smoke_problems: int | None = None,
    base_only: bool = False,
    dev_override: bool = False,
    out_path: str = f"{WORKSPACE_DIR}/eval/heldout_results.json",
) -> None:
    """Run held-out evaluation. H200-only; requires explicit operator approval."""
    result = run_heldout.remote(
        dataset_path=dataset_path,
        adapter_dir=adapter_dir,
        k_shards=k_shards,
        max_tokens=max_tokens,
        n_samples=n_samples,
        smoke_problems=smoke_problems,
        base_only=base_only,
        dev_override=dev_override,
        out_path=out_path,
    )
    print(json.dumps(result, indent=2))
