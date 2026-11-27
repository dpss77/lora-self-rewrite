import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parents[2]

_MODAL_STUB = """
import sys
import types

class _App:
    def __init__(self, *args, **kwargs):
        pass
    def function(self, *args, **kwargs):
        return lambda fn: fn
    def cls(self, *args, **kwargs):
        return lambda cls: cls
    def local_entrypoint(self, *args, **kwargs):
        return lambda fn: fn

class _Image:
    @staticmethod
    def debian_slim(**kwargs):
        return _Image()
    def pip_install(self, *args, **kwargs):
        return self
    def env(self, *args, **kwargs):
        return self
    def add_local_python_source(self, *args, **kwargs):
        return self

class _Volume:
    @classmethod
    def from_name(cls, *args, **kwargs):
        return cls()
    def commit(self):
        pass
    def reload(self):
        pass

class _Secret:
    @classmethod
    def from_name(cls, *args, **kwargs):
        return object()

_modal = types.ModuleType("modal")
_modal.App = _App
_modal.Image = _Image
_modal.Volume = _Volume
_modal.Secret = _Secret
_modal.method = lambda *args, **kwargs: (lambda fn: fn)
_modal.enter = lambda *args, **kwargs: (lambda fn: fn)
_modal.exit = lambda *args, **kwargs: (lambda fn: fn)
sys.modules["modal"] = _modal
"""

HEAVY = '{"torch", "transformers", "peft", "vllm"}'


def _run_python(source: str, *, env_extra: dict[str, str] | None = None) -> str:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT / "src"), env.get("PYTHONPATH", "")]).rstrip(
        os.pathsep
    )
    if env_extra:
        env.update(env_extra)
    completed = subprocess.run(
        [sys.executable, "-c", source],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    return completed.stdout


def test_hf_worker_and_engine_modules_do_not_import_hf_at_import_time():
    source = f"""
import sys
{_MODAL_STUB}
import lora_self_rewrite.hf_worker as hf_worker
assert not ({HEAVY} & set(sys.modules)), sorted({HEAVY} & set(sys.modules))
import lora_self_rewrite.engine as engine
assert engine._hf_extract_worker is hf_worker._hf_extract_worker
assert engine._hf_extract_worker.__module__ == "lora_self_rewrite.hf_worker"
assert not ({HEAVY} & set(sys.modules)), sorted({HEAVY} & set(sys.modules))
print("ok")
"""
    assert _run_python(source).strip() == "ok"


def test_read_hidden_states_routes_through_spawned_worker(tmp_path):
    out_path = tmp_path / "read.pt"
    source = f"""
import json
import os
import sys
import types
{_MODAL_STUB}
import lora_self_rewrite.engine as engine

captured = {{}}

class _Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return messages[-1]["content"]

class _LLM:
    def sleep(self, level=1):
        captured["sleep_level"] = level
    def wake_up(self):
        captured["woke"] = True

class _FakeProc:
    def __init__(self):
        self.exitcode = 0
    def start(self):
        with open(os.environ["READ_OUT"], "wb") as f:
            f.write(b"{{}}")
    def join(self, timeout=None):
        pass
    def is_alive(self):
        return False
    def terminate(self):
        pass
    def kill(self):
        pass

class _Ctx:
    def Process(self, target, args):
        captured["target"] = target
        captured["target_module"] = target.__module__
        captured["args"] = args
        return _FakeProc()

class _Mp:
    def get_context(self, method):
        captured["method"] = method
        return _Ctx()

class _Volume:
    def __init__(self):
        self.commits = 0
    def commit(self):
        self.commits += 1
    def reload(self):
        pass

volume = _Volume()
engine.vol = volume
engine.mp = _Mp()
engine._safe_reload = lambda context: None
fake_self = types.SimpleNamespace(
    _check_ready=lambda: None,
    tokenizer=_Tokenizer(),
    llm=_LLM(),
)
result = engine.QwenSelfRewriteEngine._read_hidden_states_impl(
    fake_self,
    ["prompt"],
    ["answer"],
    [8],
    ["hidden_states"],
    "answer_mean",
    1,
    None,
    os.environ["READ_OUT"],
    max_length=16,
    worker_timeout_s=5,
)
assert captured["method"] == "spawn"
assert captured["target"] is engine._hf_extract_worker
assert captured["target_module"] == "lora_self_rewrite.hf_worker"
assert captured["args"][0] == engine.MODEL_ID
assert captured["args"][1] is None
assert result["status"] == "ok"
assert volume.commits == 1
assert captured["sleep_level"] == 1
assert captured["woke"] is True
assert not ({HEAVY} & set(sys.modules)), sorted({HEAVY} & set(sys.modules))
print(json.dumps({{"out_path": result["out_path"], "commits": volume.commits}}))
"""
    output = _run_python(source, env_extra={"READ_OUT": str(out_path)})
    payload = __import__("json").loads(output)
    assert payload["out_path"] == str(out_path)
    assert payload["commits"] == 1
    assert out_path.read_bytes() == b"{}"
