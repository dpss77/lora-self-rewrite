import importlib
import json
import os
import sys
import types
from pathlib import Path

import pytest
import torch

from lora_self_rewrite.adapter import (
    canonical_adapter_config,
    write_submit_provenance,
)
from lora_self_rewrite.caches import LoraIdRegistry, content_addressed_lora_key
from lora_self_rewrite.constants import GENERATION_CONFIG_HASH, MODEL_ID, MODEL_REVISION
from lora_self_rewrite.eval_common import cache_tag_for
from lora_self_rewrite.heldout_gate import eval_engine_config, eval_engine_config_hash

pytestmark = pytest.mark.integration

_MODULE = None


class _App:
    def __init__(self, *_args, **_kwargs):
        pass

    def function(self, *_args, **_kwargs):
        return lambda fn: fn

    def cls(self, *_args, **_kwargs):
        return lambda cls: cls

    def local_entrypoint(self, *_args, **_kwargs):
        return lambda fn: fn


class _Image:
    @staticmethod
    def debian_slim(**_kwargs):
        return _Image()

    def pip_install(self, *_args, **_kwargs):
        return self

    def env(self, *_args, **_kwargs):
        return self

    def add_local_python_source(self, *_args, **_kwargs):
        return self


class _Volume:
    @classmethod
    def from_name(cls, *_args, **_kwargs):
        return cls()

    def commit(self):
        pass

    def reload(self):
        pass


class _Secret:
    @classmethod
    def from_name(cls, *_args, **_kwargs):
        return object()


def _load_module():
    global _MODULE
    if _MODULE is not None:
        return _MODULE
    stub = types.ModuleType("modal")
    stub.App = _App
    stub.Image = _Image
    stub.Volume = _Volume
    stub.Secret = _Secret
    stub.method = lambda *_a, **_k: lambda fn: fn
    stub.enter = lambda *_a, **_k: lambda fn: fn
    stub.exit = lambda *_a, **_k: lambda fn: fn
    saved = sys.modules.get("modal")
    sys.modules["modal"] = stub
    try:
        _MODULE = importlib.import_module("lora_self_rewrite.eval_heldout")
    finally:
        if saved is None:
            sys.modules.pop("modal", None)
        else:
            sys.modules["modal"] = saved
    return _MODULE


ENGINE_CONFIG = eval_engine_config(
    model_id=MODEL_ID,
    model_revision=MODEL_REVISION,
    dtype="bfloat16",
    max_model_len=32768,
    max_num_seqs=64,
    max_loras=2,
    max_lora_rank=16,
    gpu_memory_utilization=0.88,
    enable_prefix_caching=False,
    trust_remote_code=True,
    generation_config_hash=GENERATION_CONFIG_HASH,
)
ENGINE_HASH = eval_engine_config_hash(ENGINE_CONFIG)
CACHE_TAG = cache_tag_for("NVIDIA H200", "2.14.0", "0.34.0")


class _Remote:
    def __init__(self, fn):
        self.remote = fn


class _RecordingVolume:
    def __init__(self):
        self.commits = 0
        self.reloads = 0
        self.events: list[str] = []

    def commit(self):
        self.commits += 1
        self.events.append("commit")

    def reload(self):
        self.reloads += 1
        self.events.append("reload")


class _NoCommitVolume:
    def __init__(self):
        self.reloads = 0

    def reload(self):
        self.reloads += 1

    def commit(self):
        raise AssertionError("GPU shard must not commit results")


def _write_payload(tmp_path: Path) -> Path:
    datasets = tmp_path / "datasets"
    datasets.mkdir()
    heldout = [
        {"id": index, "problem": f"heldout problem {index}", "answer": str(index)}
        for index in range(990)
    ]
    payload = {
        "heldout": heldout,
        "meta": {"source_revisions": {"fake/source": "pinned-revision"}},
    }
    path = datasets / "deepscaler_scrubbed.json"
    path.write_text(json.dumps(payload))
    return path


def _write_adapter(tmp_path: Path, *, nonzero: bool, provenance: bool) -> Path:
    adapter = tmp_path / "sessions" / "s1" / "WORK" / "submitted_lora"
    adapter.mkdir(parents=True)
    tensor = torch.ones((2, 2), dtype=torch.bfloat16)
    if not nonzero:
        tensor = torch.zeros((2, 2), dtype=torch.bfloat16)
    from safetensors.torch import save_file

    save_file(
        {"layers.8.mlp.down_proj.lora_A.weight": tensor}, adapter / "adapter_model.safetensors"
    )
    (adapter / "adapter_config.json").write_text(json.dumps(canonical_adapter_config(rank=4)))
    if provenance:
        write_submit_provenance(adapter, decision="SUBMIT", session_id="s1")
    return adapter


def _rows(problems, n_samples, *, edited: bool) -> list[dict]:
    rows = []
    for problem in problems:
        for sample_idx in range(n_samples):
            correct = (int(problem["id"]) % 2 == 0) if edited else True
            text = (
                "</think>\\boxed{" + problem["answer"] + "}"
                if correct
                else "</think>\\boxed{definitely-wrong}"
            )
            rows.append(
                {
                    "id": problem["id"],
                    "sample_idx": sample_idx,
                    "problem": problem["problem"],
                    "answer": problem["answer"],
                    "text": text,
                    "tokens": 1,
                    "finish_reason": "stop",
                    "cut": False,
                    "loop": False,
                    "correct": correct,
                }
            )
    return rows


def _install_engine(monkeypatch, module, volume, out_path: Path):
    class _FakeHeldoutEngine:
        instances: list = []

        def __init__(self):
            self.shard_calls: list[dict] = []
            self.environment_calls = 0
            self.get_environment = _Remote(self._get_environment)
            self.generate_shard = _Remote(self._generate_shard)
            _FakeHeldoutEngine.instances.append(self)

        def _get_environment(self):
            self.environment_calls += 1
            return {
                "gpu_name": "NVIDIA H200",
                "torch_version": "2.14.0",
                "vllm_version": "0.34.0",
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "cache_tag": CACHE_TAG,
                "eval_engine_config": ENGINE_CONFIG,
                "eval_engine_config_hash": ENGINE_HASH,
                "package_versions": {"fake": "1"},
            }

        def _generate_shard(self, problems, n_samples, max_tokens, lora_dir=None):
            self.shard_calls.append(
                {
                    "n_problems": len(problems),
                    "n_samples": n_samples,
                    "max_tokens": max_tokens,
                    "lora_dir": lora_dir,
                    "commits_at_call": volume.commits,
                    "out_exists_at_call": os.path.exists(out_path),
                }
            )
            rows = _rows(problems, n_samples, edited=lora_dir is not None)
            return {
                "rows": rows,
                "gpu_name": "NVIDIA H200",
                "vllm_version": "0.34.0",
                "cache_tag": CACHE_TAG,
                "eval_engine_config_hash": ENGINE_HASH,
                "adapter_sha256": (
                    content_addressed_lora_key(lora_dir) if lora_dir is not None else None
                ),
            }

    monkeypatch.setattr(module, "HeldoutEngine", _FakeHeldoutEngine)
    return _FakeHeldoutEngine


def _run_base_only(tmp_path, monkeypatch):
    module = _load_module()
    volume = _RecordingVolume()
    out_path = tmp_path / "eval" / "heldout_results.json"
    engine_cls = _install_engine(monkeypatch, module, volume, out_path)
    monkeypatch.setattr(module, "WORKSPACE_DIR", str(tmp_path))
    monkeypatch.setattr(module, "vol", volume)
    result = module.run_heldout(
        dataset_path=str(_write_payload(tmp_path)),
        adapter_dir=None,
        k_shards=2,
        max_tokens=64,
        n_samples=1,
        smoke_problems=4,
        base_only=True,
        dev_override=True,
        out_path=str(out_path),
    )
    return result, json.loads(out_path.read_text()), volume, engine_cls.instances[-1]


def test_run_heldout_base_only_allows_missing_adapter_and_writes_once(tmp_path, monkeypatch):
    result, artifact, volume, engine = _run_base_only(tmp_path, monkeypatch)

    assert result["out_path"] == str(tmp_path / "eval" / "heldout_results.json")
    assert artifact["base_only"] is True
    assert artifact["adapter_dir"] is None
    assert artifact["adapter_provenance"] is None
    assert len(artifact["base"]["rows"]) == 4
    assert "edited" not in artifact
    assert artifact["eval_engine_config_hash"] == ENGINE_HASH

    # Shards return rows in memory; the CPU orchestrator alone writes the cache and
    # final artifact, and only after every shard for that arm has returned.
    assert engine.environment_calls == 1
    assert len(engine.shard_calls) == 2
    assert all(call["commits_at_call"] == 0 for call in engine.shard_calls)
    assert all(call["out_exists_at_call"] is False for call in engine.shard_calls)
    assert all(call["lora_dir"] is None for call in engine.shard_calls)
    assert volume.commits == 2
    assert volume.events == ["commit", "commit"]


@pytest.mark.parametrize(
    ("nonzero", "provenance", "match"),
    [
        (True, False, "SUBMIT provenance"),
        (False, True, "all zeros"),
    ],
)
def test_run_heldout_submitted_gate_rejects_before_any_shard(
    tmp_path, monkeypatch, nonzero, provenance, match
):
    module = _load_module()
    volume = _RecordingVolume()
    out_path = tmp_path / "eval" / "heldout_results.json"
    engine_cls = _install_engine(monkeypatch, module, volume, out_path)
    monkeypatch.setattr(module, "WORKSPACE_DIR", str(tmp_path))
    monkeypatch.setattr(module, "vol", volume)
    adapter = _write_adapter(tmp_path, nonzero=nonzero, provenance=provenance)

    with pytest.raises(ValueError, match=match):
        module.run_heldout(
            dataset_path=str(_write_payload(tmp_path)),
            adapter_dir=str(adapter),
            k_shards=2,
            max_tokens=64,
            n_samples=1,
            smoke_problems=4,
            base_only=False,
            dev_override=True,
            out_path=str(out_path),
        )

    assert engine_cls.instances == []
    assert volume.commits == 0
    assert not out_path.exists()


def test_run_heldout_submitted_uses_same_engine_and_orchestrator_writes(tmp_path, monkeypatch):
    module = _load_module()
    volume = _RecordingVolume()
    out_path = tmp_path / "eval" / "heldout_results.json"
    engine_cls = _install_engine(monkeypatch, module, volume, out_path)
    monkeypatch.setattr(module, "WORKSPACE_DIR", str(tmp_path))
    monkeypatch.setattr(module, "vol", volume)
    adapter = _write_adapter(tmp_path, nonzero=True, provenance=True)
    adapter_sha = content_addressed_lora_key(adapter)

    result = module.run_heldout(
        dataset_path=str(_write_payload(tmp_path)),
        adapter_dir=str(adapter),
        k_shards=2,
        max_tokens=64,
        n_samples=1,
        smoke_problems=4,
        base_only=False,
        dev_override=True,
        out_path=str(out_path),
    )

    artifact = json.loads(out_path.read_text())
    assert result["out_path"] == str(out_path)
    assert artifact["base_only"] is False
    assert artifact["adapter_provenance"]["decision"] == "SUBMIT"
    assert artifact["adapter_sha256"] == adapter_sha
    assert artifact["edited"]["adapter_sha256"] == adapter_sha
    assert len(artifact["base"]["rows"]) == 4
    assert len(artifact["edited"]["rows"]) == 4
    assert artifact["stats"]["n_problems"] == 4
    # Both arms ran under one eval-engine identity; the orchestrator rejects drift.
    assert artifact["eval_engine_config_hash"] == ENGINE_HASH
    assert artifact["environment"]["eval_engine_config_hash"] == ENGINE_HASH

    engine = engine_cls.instances[-1]
    base_calls = [call for call in engine.shard_calls if call["lora_dir"] is None]
    edited_calls = [call for call in engine.shard_calls if call["lora_dir"] is not None]
    assert len(base_calls) == 2
    assert len(edited_calls) == 2
    assert all(call["commits_at_call"] == 0 for call in base_calls)
    # The base cache is committed by the orchestrator after the base shards return
    # and before the edited shards start.
    assert all(call["commits_at_call"] == 1 for call in edited_calls)
    assert all(call["out_exists_at_call"] is False for call in engine.shard_calls)
    assert volume.commits == 2
    # One reload before adapter validation, one before the edited adapter digest.
    assert volume.reloads == 2


class _FakeTokenizer:
    def apply_chat_template(self, messages, **_kwargs):
        return messages[-1]["content"]


class _FakeOutput:
    def __init__(self, text, token_ids, finish_reason):
        self.text = text
        self.token_ids = token_ids
        self.finish_reason = finish_reason


class _FakeRequestOutput:
    def __init__(self, output):
        self.outputs = [output]


class _FakeLLM:
    def __init__(self):
        self.calls = []

    def generate(self, prompts, sampling, lora_request=None, use_tqdm=False):
        self.calls.append(
            {
                "prompts": prompts,
                "sampling": sampling,
                "lora_request": lora_request,
                "use_tqdm": use_tqdm,
            }
        )
        output = _FakeOutput("</think>\\boxed{500}", [1, 2, 3, 4], "stop")
        return [_FakeRequestOutput(output) for _ in prompts]


def _install_fake_vllm(monkeypatch):
    class _SamplingParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class _LoRARequest:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    vllm = types.ModuleType("vllm")
    vllm.SamplingParams = _SamplingParams
    vllm.__version__ = "0.34.0"
    lora_pkg = types.ModuleType("vllm.lora")
    lora_request = types.ModuleType("vllm.lora.request")
    lora_request.LoRARequest = _LoRARequest
    lora_pkg.request = lora_request
    vllm.lora = lora_pkg
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.lora", lora_pkg)
    monkeypatch.setitem(sys.modules, "vllm.lora.request", lora_request)


def test_real_gpu_shard_is_read_only_and_uses_token_limit_cut(tmp_path, monkeypatch):
    module = _load_module()
    _install_fake_vllm(monkeypatch)
    volume = _NoCommitVolume()
    monkeypatch.setattr(module, "vol", volume)
    adapter = _write_adapter(tmp_path, nonzero=True, provenance=True)
    before = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))
    fake_self = types.SimpleNamespace(
        tokenizer=_FakeTokenizer(),
        llm=_FakeLLM(),
        gpu_name="NVIDIA H200",
        vllm_version="0.34.0",
        torch=types.SimpleNamespace(__version__="2.14.0"),
        eval_engine_config_hash=ENGINE_HASH,
        lora_ids=LoraIdRegistry(),
    )
    problems = [{"id": 500, "problem": "p", "answer": "500"}]

    result = module.HeldoutEngine.generate_shard(
        fake_self,
        problems,
        n_samples=1,
        max_tokens=4,  # equals the generated token count, finish_reason stays "stop"
        lora_dir=str(adapter),
    )

    after = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))
    assert after == before
    assert volume.reloads == 1
    assert result["adapter_sha256"] == content_addressed_lora_key(adapter)
    row = result["rows"][0]
    assert row["tokens"] == 4
    # The shard uses the shared token-limit cut definition, not finish_reason alone.
    assert row["cut"] is True
