import copy
import importlib
import json
import subprocess
import sys
import types
from pathlib import Path

import pytest
import torch
from jsonschema import Draft202012Validator
from safetensors.torch import save_file

import lora_self_rewrite as _pkg
from lora_self_rewrite.adapter import canonical_adapter_config, write_submit_provenance
from lora_self_rewrite.caches import content_addressed_lora_key
from lora_self_rewrite.constants import GENERATION_CONFIG_HASH, MODEL_ID, MODEL_REVISION
from lora_self_rewrite.eval_common import cache_tag_for
from lora_self_rewrite.heldout_gate import eval_engine_config, eval_engine_config_hash

pytestmark = pytest.mark.contract

ROOT = Path(__file__).resolve().parents[2]
SCHEMA_DIR = ROOT / "schemas"


def _validator(name: str) -> Draft202012Validator:
    schema = json.loads((SCHEMA_DIR / name).read_text())
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def _assert_conforms(validator: Draft202012Validator, payload: object, label: str) -> None:
    errors = sorted(validator.iter_errors(payload), key=lambda error: list(error.path))
    if errors:
        details = "\n".join(f"  - {list(error.path)}: {error.message}" for error in errors)
        pytest.fail(f"{label} does not conform to its schema:\n{details}")


def _assert_rejected(validator: Draft202012Validator, payload: object, label: str) -> None:
    assert list(validator.iter_errors(payload)), f"{label} unexpectedly conformed"


def _run_script(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )


def test_manifest_and_status_from_real_scripts(tmp_path):
    runs = tmp_path / "runs"
    out = _run_script(
        "scripts/new_run.py",
        "--config",
        "configs/experiments/S02-DS-SEED00.toml",
        "--runs-dir",
        str(runs),
        "--code-sha",
        "fe71f74",
        "--file-sha256",
        "fe71f74b0c",
    )
    run_dir = runs / out.stdout.splitlines()[0]
    manifest = json.loads((run_dir / "00_RUN_MANIFEST.json").read_text())

    manifest_validator = _validator("run_manifest.schema.json")
    _assert_conforms(manifest_validator, manifest, "new_run manifest")

    missing_status = copy.deepcopy(manifest)
    missing_status.pop("status")
    _assert_rejected(manifest_validator, missing_status, "manifest without status")

    terminal_without_finish = copy.deepcopy(manifest)
    terminal_without_finish["status"] = "withdrawn"
    terminal_without_finish["finished_at_utc"] = None
    _assert_rejected(
        manifest_validator,
        terminal_without_finish,
        "terminal manifest without finished_at_utc",
    )

    _run_script("scripts/finalize_run.py", "--run-dir", str(run_dir), "--status", "running")
    status = json.loads((run_dir / "80_FINAL" / "STATUS.json").read_text())

    status_validator = _validator("status.schema.json")
    _assert_conforms(status_validator, status, "finalize STATUS.json")

    missing_run_id = copy.deepcopy(status)
    missing_run_id.pop("run_id")
    _assert_rejected(status_validator, missing_run_id, "status without run_id")


_SESSION = None
_ROOT = {"path": ""}


class _DynamicRoot:
    def __format__(self, _spec):
        return _ROOT["path"]

    def __str__(self):
        return _ROOT["path"]


class _App:
    def function(self, *_args, **_kwargs):
        return lambda fn: fn


class _Vol:
    def commit(self):
        pass

    def reload(self):
        pass


class _Remote:
    def __init__(self, fn):
        self.remote = fn


class _FakeTokenizer:
    def apply_chat_template(self, *_args, **_kwargs):
        return "x"

    def encode(self, _text):
        return [1]


def _fake_helpers(_lora_dir, _work_dir, _registry):
    return {
        "load_base_weight": lambda _name: None,
        "load_lora_tensors": lambda: {},
        "read_notes": lambda: "",
        "list_edits": lambda: [],
        "list_candidates": lambda: [],
        "drop_edit": lambda *_a, **_k: None,
    }


def _helpers_with_edits(*_args, **_kwargs):
    helpers = _fake_helpers(None, None, None)
    helpers["list_edits"] = lambda: [
        {"module": "layers.8.mlp.down_proj", "lora_A_norm": 1.0, "lora_B_norm": 1.0}
    ]
    return helpers


def _load_session():
    global _SESSION
    if _SESSION is not None:
        return _SESSION
    pkg = _pkg.__name__
    saved: dict[str, object] = {}

    def install(name, **attrs):
        module = types.ModuleType(name)
        for key, value in attrs.items():
            setattr(module, key, value)
        saved[name] = sys.modules.get(name)
        sys.modules[name] = module

    install(
        f"{pkg}.modal_wiring",
        app=_App(),
        image=object(),
        vol=_Vol(),
        WORKSPACE_DIR=_DynamicRoot(),
        _SECRETS=[],
        safe_commit=lambda: None,
        safe_reload=lambda _ctx: None,
    )
    install(f"{pkg}.engine", QwenSelfRewriteEngine=object)
    install(
        f"{pkg}.instrumentation",
        _LimitedBuffer=object,
        _fingerprint=lambda *_a, **_k: {},
        _package_version=lambda *_a, **_k: "0",
        _percentile=lambda *_a, **_k: 0.0,
        _sha256_file=lambda *_a, **_k: "",
        _summarize_metrics=lambda *_a, **_k: {},
        get_model_revision=lambda *_a, **_k: None,
        cache_fingerprint_from_env=lambda gpu_fp, schema: {
            "gpu": gpu_fp.get("gpu"),
            "cache_schema": schema,
        },
        cache_tag_from_fingerprint=lambda _fp: "cachetag0000",
    )
    install(
        f"{pkg}.model_layout",
        init_empty_lora=lambda *_a, **_k: {
            "layers.8.mlp.down_proj": {
                "layer": 8,
                "module_path": "layers.8.mlp.down_proj",
                "peft_prefix": "x",
                "full_key": "x",
                "shape": (16, 8),
                "shard_path": "x",
            }
        },
        get_model_index=lambda *_a, **_k: {},
    )
    install(f"{pkg}.sandbox_runtime", bind_sandbox_helpers=_fake_helpers)
    install(
        f"{pkg}.datasets",
        load_session_datasets=lambda _mode: (
            [{"id": i, "problem": f"p{i}", "answer": str(i)} for i in range(6)],
            [{"id": 100 + i, "problem": f"v{i}", "answer": str(i)} for i in range(4)],
        ),
        dataset_identity=lambda mode, _practice, _reserved: {
            "dataset_mode": mode,
            "source_revisions": {"fake": "rev"},
            "content_sha256": "fake-content",
            "identity": "fakedatasetidentity",
        },
        require_pinned_sources=lambda _mode: {"fake": "rev"},
    )
    try:
        _SESSION = importlib.import_module(f"{pkg}.session")
    finally:
        for name, original in saved.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original
    return _SESSION


class _ArtifactEngine:
    def __init__(self):
        self.editor_calls = 0
        self.editor_messages: list[list[dict]] = []
        self.get_environment = _Remote(
            lambda *_a, **_k: {
                "fingerprint": {"gpu": "A100", "model_revision": "rev"},
                "startup": {},
                "engine_config": {"fake": "engine"},
                "engine_config_hash": "fakeenginehash",
            }
        )
        self.generate_batch = _Remote(self._generate)
        self.read_hidden_states = _Remote(self._unexpected_read)
        self.editor_step = _Remote(self._editor)

    @staticmethod
    def _unexpected_read(*_args, **_kwargs):
        raise AssertionError("unexpected read")

    def _editor(self, messages, *_args, **_kwargs):
        self.editor_messages.append(messages)
        self.editor_calls += 1
        if self.editor_calls == 1:
            text = (
                "</think>\n```json\n"
                '{"action": "generate", "problem_ids": [0], '
                '"include_base": true, "n_samples": 1}\n```'
            )
        elif self.editor_calls == 2:
            text = "</think>\nFINAL_SUBMIT"
        else:
            text = "</think>\nSUBMIT"
        return {
            "text": text,
            "prompt_tokens": 1,
            "tokens": 1,
            "finish_reason": "stop",
            "timings": {},
            "gpu_memory": {},
        }

    def _generate(self, problems, *_args, **kwargs):
        n_samples = int(kwargs.get("n_samples", 1))

        def rows_for(text_for):
            return [
                {
                    "id": problem["id"],
                    "sample_idx": sample_idx,
                    "problem": problem["problem"],
                    "answer": problem["answer"],
                    "text": text_for(problem),
                    "tokens": 1,
                    "finish_reason": "stop",
                    "cut": False,
                    "loop": False,
                }
                for problem in problems
                for sample_idx in range(n_samples)
            ]

        base = rows_for(lambda problem: "</think>\\boxed{" + problem["answer"] + "}")
        candidates = {"active": [dict(row) for row in base]} if kwargs.get("lora_specs") else {}
        return {
            "base": base,
            "candidates": candidates,
            "timings": {
                "n_answers": len(base) + sum(len(rows) for rows in candidates.values()),
                "generated_tokens": len(base) + sum(len(rows) for rows in candidates.values()),
            },
            "gpu_memory": {},
        }


def _drop_session_module():
    global _SESSION
    sys.modules.pop(f"{_pkg.__name__}.session", None)
    _SESSION = None


@pytest.fixture
def isolated_session_module():
    yield
    _drop_session_module()


def _install_session_fakes(monkeypatch, tmp_path, engine):
    session = _load_session()
    monkeypatch.setattr(session, "WORKSPACE_DIR", str(tmp_path))
    monkeypatch.setattr(session, "QwenSelfRewriteEngine", lambda: engine)
    monkeypatch.setattr(session, "bind_sandbox_helpers", _helpers_with_edits)
    monkeypatch.setattr(session, "load_and_validate_adapter", lambda *_a, **_k: ({}, {}))

    transformers = types.ModuleType("transformers")
    transformers.AutoTokenizer = type(
        "AutoTokenizer", (), {"from_pretrained": staticmethod(lambda _m, **_k: _FakeTokenizer())}
    )
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    math_verify = types.ModuleType("math_verify")

    class _Cfg:
        def __init__(self, *_args, **_kwargs):
            pass

    def _parse(text, extraction_config=None):
        import re

        match = re.search(r"\\boxed\{([^{}]*)\}", text)
        if match:
            return match.group(1).strip()
        stripped = text.strip().strip("$").strip()
        return stripped or None

    math_verify.ExprExtractionConfig = _Cfg
    math_verify.LatexExtractionConfig = _Cfg
    math_verify.parse = _parse
    math_verify.verify = lambda a, b: str(a) == str(b)
    monkeypatch.setitem(sys.modules, "math_verify", math_verify)
    return session


def test_session_metrics_pass_and_validation_match_schemas(
    tmp_path, monkeypatch, isolated_session_module
):
    _ROOT["path"] = str(tmp_path)
    engine = _ArtifactEngine()
    session = _install_session_fakes(monkeypatch, tmp_path, engine)

    lora = tmp_path / "sessions" / "s_artifact" / "active_lora"
    lora.mkdir(parents=True)
    (lora / "adapter_model.safetensors").write_bytes(b"weights")
    (lora / "adapter_config.json").write_text(json.dumps(canonical_adapter_config(rank=4)))

    session.run_self_rewrite_session(
        session_id="s_artifact",
        dataset_mode="math500",
        max_tokens=64,
        diag_n=2,
        val_n=1,
        val_samples=1,
        max_turns=5,
        max_passes=20,
        max_code_runs=10,
        max_gen_answers=64,
        wall_limit_hours=1.0,
        run_preflight=False,
        preflight_sleep=False,
        reuse_diagnosis=False,
        reset_active_lora=True,
        overwrite_session=False,
        worker_timeout_s=60,
        deviation="unit-test: schema fake engine",
    )

    work = tmp_path / "sessions" / "s_artifact" / "WORK"
    metrics = json.loads((work / "metrics.json").read_text())
    pass_payload = json.loads((work / "pass_1.json").read_text())
    transcript = json.loads((work / "transcript.json").read_text())
    provenance = json.loads((work / "submitted_lora" / "submit_provenance.json").read_text())

    metrics_validator = _validator("metrics.schema.json")
    _assert_conforms(metrics_validator, metrics, "session metrics.json")
    missing_counters = copy.deepcopy(metrics)
    missing_counters.pop("counters")
    _assert_rejected(metrics_validator, missing_counters, "metrics without counters")

    pass_validator = _validator("pass.schema.json")
    _assert_conforms(pass_validator, pass_payload, "pass_1.json")
    missing_candidates = copy.deepcopy(pass_payload)
    missing_candidates.pop("candidates")
    _assert_rejected(pass_validator, missing_candidates, "pass without candidates")

    validation_validator = _validator("validation.schema.json")
    val_stats = transcript["turns"][-1]["val_stats"]
    _assert_conforms(validation_validator, val_stats, "transcript val_stats")
    _assert_conforms(
        validation_validator,
        provenance["validation_stats"],
        "SUBMIT provenance validation_stats",
    )
    missing_p_value = copy.deepcopy(val_stats)
    missing_p_value.pop("signflip_p_one_sided")
    _assert_rejected(validation_validator, missing_p_value, "validation without p-value")

    assert pass_payload["base"]
    assert pass_payload["candidates"]["active"]


_EVAL = None

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


class _ModalApp:
    def __init__(self, *_args, **_kwargs):
        pass

    def function(self, *_args, **_kwargs):
        return lambda fn: fn

    def cls(self, *_args, **_kwargs):
        return lambda cls: cls

    def local_entrypoint(self, *_args, **_kwargs):
        return lambda fn: fn


class _ModalImage:
    @staticmethod
    def debian_slim(**_kwargs):
        return _ModalImage()

    def pip_install(self, *_args, **_kwargs):
        return self

    def env(self, *_args, **_kwargs):
        return self

    def add_local_python_source(self, *_args, **_kwargs):
        return self


class _ModalVolume:
    @classmethod
    def from_name(cls, *_args, **_kwargs):
        return cls()

    def commit(self):
        pass

    def reload(self):
        pass


class _ModalSecret:
    @classmethod
    def from_name(cls, *_args, **_kwargs):
        return object()


def _load_eval_module():
    global _EVAL
    if _EVAL is not None:
        return _EVAL
    stub = types.ModuleType("modal")
    stub.App = _ModalApp
    stub.Image = _ModalImage
    stub.Volume = _ModalVolume
    stub.Secret = _ModalSecret
    stub.method = lambda *_a, **_k: lambda fn: fn
    stub.enter = lambda *_a, **_k: lambda fn: fn
    stub.exit = lambda *_a, **_k: lambda fn: fn
    saved = sys.modules.get("modal")
    sys.modules["modal"] = stub
    try:
        _EVAL = importlib.import_module("lora_self_rewrite.eval_heldout")
    finally:
        if saved is None:
            sys.modules.pop("modal", None)
        else:
            sys.modules["modal"] = saved
    return _EVAL


class _FakeVolumeState:
    def commit(self):
        pass

    def reload(self):
        pass


class _FakeHeldoutEngine:
    def __init__(self):
        self.get_environment = _Remote(self._environment)
        self.generate_shard = _Remote(self._generate_shard)

    @staticmethod
    def _environment():
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

    @staticmethod
    def _generate_shard(problems, n_samples, max_tokens, lora_dir=None):
        rows = []
        for problem in problems:
            for sample_idx in range(n_samples):
                correct = (int(problem["id"]) % 2 == 0) if lora_dir else True
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


def _write_heldout_payload(tmp_path: Path) -> Path:
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


def _write_heldout_adapter(tmp_path: Path) -> Path:
    adapter = tmp_path / "sessions" / "s1" / "WORK" / "submitted_lora"
    adapter.mkdir(parents=True)
    tensor = torch.ones((2, 2), dtype=torch.bfloat16)
    save_file(
        {"layers.8.mlp.down_proj.lora_A.weight": tensor},
        adapter / "adapter_model.safetensors",
    )
    (adapter / "adapter_config.json").write_text(json.dumps(canonical_adapter_config(rank=4)))
    write_submit_provenance(adapter, decision="SUBMIT", session_id="s1")
    return adapter


def test_heldout_artifact_from_fake_engine_matches_schema(tmp_path, monkeypatch):
    module = _load_eval_module()
    monkeypatch.setattr(module, "HeldoutEngine", _FakeHeldoutEngine)
    monkeypatch.setattr(module, "WORKSPACE_DIR", str(tmp_path))
    monkeypatch.setattr(module, "vol", _FakeVolumeState())
    dataset_path = _write_heldout_payload(tmp_path)
    heldout_validator = _validator("heldout.schema.json")

    base_out = tmp_path / "eval" / "heldout_base.json"
    module.run_heldout(
        dataset_path=str(dataset_path),
        adapter_dir=None,
        k_shards=2,
        max_tokens=64,
        n_samples=1,
        smoke_problems=4,
        base_only=True,
        dev_override=True,
        out_path=str(base_out),
    )
    base_artifact = json.loads(base_out.read_text())
    _assert_conforms(heldout_validator, base_artifact, "base-only heldout artifact")
    assert "edited" not in base_artifact

    out_path = tmp_path / "eval" / "heldout_results.json"
    adapter = _write_heldout_adapter(tmp_path)
    module.run_heldout(
        dataset_path=str(dataset_path),
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
    _assert_conforms(heldout_validator, artifact, "submitted heldout artifact")

    missing_cache_tag = copy.deepcopy(artifact)
    missing_cache_tag.pop("cache_tag")
    _assert_rejected(heldout_validator, missing_cache_tag, "heldout without cache_tag")

    no_edited = copy.deepcopy(artifact)
    no_edited["base_only"] = False
    no_edited.pop("edited")
    _assert_rejected(heldout_validator, no_edited, "submitted heldout without edited rows")
