import importlib
import json
import sys
import types
from pathlib import Path

import pytest

import lora_self_rewrite as _pkg
from lora_self_rewrite.caches import diagnosis_identity, save_diagnosis_cache
from lora_self_rewrite.constants import MODEL_REVISION
from lora_self_rewrite.resume import resume_plan, scan_work_dir

pytestmark = pytest.mark.integration


def _work(tmp_path: Path) -> Path:
    work = tmp_path / "WORK"
    work.mkdir()
    return work


def test_empty_work_is_fresh(tmp_path):
    work = _work(tmp_path)
    assert scan_work_dir(str(work))["has_work"] is False
    assert resume_plan(str(work))["action"] == "fresh"


def test_pass_count_and_transcript_restore(tmp_path):
    work = _work(tmp_path)
    (work / "pass_19.json").write_text("{}")
    (work / "pass_3.json").write_text("{}")
    transcript = {
        "summary": {"passes_used": 19, "code_runs_used": 37, "turns_completed": 2},
        "turns": [{"turn": 1}, {"turn": 2}],
    }
    (work / "transcript.json").write_text(json.dumps(transcript))
    (work / "candidates").mkdir()
    (work / "candidates" / "c26").mkdir()

    scan = scan_work_dir(str(work))
    assert scan["has_work"] is True
    assert scan["passes_used"] == 19
    assert scan["code_runs_used"] == 37
    assert scan["turns_completed"] == 2
    assert "candidates/" in scan["dirty_reasons"]

    plan = resume_plan(str(work), reset_active_lora=False)
    assert plan["action"] == "resume"
    assert plan["resumed"] is True
    assert plan["scan"]["passes_used"] == 19


def test_refuse_when_reset_requested_and_overwrite_when_asked(tmp_path):
    work = _work(tmp_path)
    (work / "pass_1.json").write_text("{}")
    assert resume_plan(str(work), reset_active_lora=True)["action"] == "refuse"
    assert resume_plan(str(work), overwrite_session=True)["action"] == "overwrite"


def test_jsonl_fallback_and_code_runs(tmp_path):
    work = _work(tmp_path)
    # Rows in the exact shape _record_turn writes: no code_runs_used key, only obs.
    lines = [
        json.dumps({"turn": 1, "reply": "", "obs": "[code 2/200, 1.0m / 60m] ok"}),
        json.dumps({"turn": 2, "reply": "", "obs": "[code 5/200, 2.0m / 60m] ok"}),
        "not json",
    ]
    (work / "transcript.jsonl").write_text("\n".join(lines))
    scan = scan_work_dir(str(work))
    assert scan["turns_completed"] == 2
    assert scan["code_runs_used"] == 5


# ---------------------------------------------------------------------------
# Call-site tests: drive the real ``run_self_rewrite_session`` on CPU.
#
# ``session.py`` imports Modal and GPU workers at module scope. We install
# lightweight stand-ins for those modules only for the duration of the import so
# the ported function body itself is exercised, then restore ``sys.modules``.
# The engine is monkeypatched per test; no GPU/network is touched.
# ---------------------------------------------------------------------------

_SESSION = None
_ROOT = {"path": ""}


class _DynamicRoot:
    """Mutable WORKSPACE_DIR bound into the imported session module."""

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


class _FakeEngine:
    def __init__(self):
        self.generate_calls = 0
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
        self.read_hidden_states = _Remote(
            lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("unexpected read"))
        )
        self.editor_step = _Remote(self._editor)

    def _editor(self, messages, *_args, **_kwargs):
        self.editor_messages.append(messages)
        return {
            "text": "</think>\nFINAL_WITHDRAW",
            "prompt_tokens": 1,
            "tokens": 1,
            "finish_reason": "stop",
            "timings": {},
            "gpu_memory": {},
        }

    def _generate(self, problems, *_args, **kwargs):
        self.generate_calls += 1
        n_samples = int(kwargs.get("n_samples", 1))
        base = [
            {
                "id": p["id"],
                "sample_idx": sample_idx,
                "problem": p["problem"],
                "answer": p["answer"],
                "text": "</think>\\boxed{" + p["answer"] + "}",
                "tokens": 1,
                "finish_reason": "stop",
                "cut": False,
                "loop": False,
            }
            for p in problems
            for sample_idx in range(n_samples)
        ]
        return {
            "base": base,
            "candidates": {},
            "timings": {
                "n_answers": len(base),
                "generated_tokens": len(base),
            },
            "gpu_memory": {},
        }


@pytest.fixture
def session_env(tmp_path, monkeypatch):
    _ROOT["path"] = str(tmp_path)
    engine = _FakeEngine()
    session = _load_session()
    monkeypatch.setattr(session, "QwenSelfRewriteEngine", lambda: engine)
    transformers = types.ModuleType("transformers")
    transformers.AutoTokenizer = type(
        "AutoTokenizer", (), {"from_pretrained": staticmethod(lambda _m, **_k: _FakeTokenizer())}
    )
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    # Minimal math_verify so the grader self-test and diagnosis grading work.
    math_verify = types.ModuleType("math_verify")

    class _Cfg:
        def __init__(self, *_a, **_k):
            pass

    def _parse(text, extraction_config=None):  # noqa: ARG001
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
    yield session, engine, tmp_path


def _work_dir(tmp_path: Path, session_id: str) -> Path:
    return tmp_path / "sessions" / session_id / "WORK"


def _run(session, tmp_path, session_id="s_resume", **overrides):
    kwargs = dict(
        session_id=session_id,
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
        reset_active_lora=False,
        overwrite_session=False,
        worker_timeout_s=60,
        deviation="unit-test: CPU fake engine",
    )
    kwargs.update(overrides)
    session.run_self_rewrite_session(**kwargs)
    return json.loads((_work_dir(tmp_path, session_id) / "transcript.json").read_text())


def _write_active_lora(tmp_path: Path, session_id: str = "s_resume") -> Path:
    lora = tmp_path / "sessions" / session_id / "active_lora"
    lora.mkdir(parents=True, exist_ok=True)
    (lora / "adapter_model.safetensors").write_bytes(b"active-weights")
    (lora / "adapter_config.json").write_text("{}")
    return lora


def _seed_dirty_work(tmp_path: Path, session_id: str = "s_resume") -> list[str]:
    work = _work_dir(tmp_path, session_id)
    work.mkdir(parents=True)
    _write_active_lora(tmp_path, session_id)
    (work / "pass_19.json").write_text("{}")
    prior = [
        json.dumps({"turn": 1, "reply": "", "obs": "[code 5/200, 1.0m / 60m] ok"}),
        json.dumps({"turn": 2, "reply": "", "obs": "[pass 19/20, 2.0m / 60m] Generate complete"}),
    ]
    (work / "transcript.jsonl").write_text("\n".join(prior) + "\n")
    return prior


def test_resume_restores_counters_and_appends_transcript(session_env):
    session, _engine, tmp_path = session_env
    prior = _seed_dirty_work(tmp_path)

    summary = _run(session, tmp_path)

    assert summary["summary"]["resumed"] is True
    assert summary["summary"]["previous_passes_used"] == 19
    assert summary["summary"]["passes_used"] == 19
    assert summary["summary"]["code_runs_used"] == 5
    # Prior turns restored (2) plus this attempt's WITHDRAW turn (1), and the new
    # turn continues at 3 without ever exceeding the cumulative max_turns budget.
    assert summary["summary"]["turns_completed"] == 3
    assert summary["summary"]["turns_completed"] <= summary["summary"]["turns_recorded"]
    assert summary["turns"][-1]["turn"] == 3
    assert len(summary["turns"]) == 3

    disk = (_work_dir(tmp_path, "s_resume") / "transcript.jsonl").read_text()
    lines = disk.splitlines()
    assert lines[:2] == prior  # appended, never truncated
    assert len(lines) == 3


def test_fresh_run_truncates_transcript_and_is_not_resumed(session_env):
    session, _engine, tmp_path = session_env
    summary = _run(session, tmp_path, session_id="s_fresh", reset_active_lora=True)
    assert summary["summary"]["resumed"] is False
    assert summary["summary"]["previous_passes_used"] == 0
    assert summary["summary"]["passes_used"] == 0
    # Only this run's single turn is present (no stale lines).
    assert len(summary["turns"]) == 1


def test_overwrite_wipes_stale_state(session_env):
    session, _engine, tmp_path = session_env
    _seed_dirty_work(tmp_path, session_id="s_over")
    summary = _run(
        session,
        tmp_path,
        session_id="s_over",
        reset_active_lora=True,
        overwrite_session=True,
    )
    assert summary["summary"]["resumed"] is False
    work = _work_dir(tmp_path, "s_over")
    assert not (work / "pass_19.json").exists()
    assert len((work / "transcript.jsonl").read_text().splitlines()) == 1


def test_refuse_raises_before_engine(session_env):
    session, engine, tmp_path = session_env
    _seed_dirty_work(tmp_path, session_id="s_refuse")
    with pytest.raises(ValueError, match="already has WORK state"):
        _run(
            session,
            tmp_path,
            session_id="s_refuse",
            reset_active_lora=True,
            overwrite_session=False,
        )


def test_inflight_marker_prevents_free_pass(tmp_path):
    work = _work(tmp_path)
    (work / "pass_19.json").write_text("{}")
    (work / "pass_20.inflight").write_text("1")
    scan = scan_work_dir(str(work))
    assert scan["passes_used"] == 20
    assert "pass_*.inflight" in scan["dirty_reasons"]
    assert resume_plan(str(work), reset_active_lora=False)["scan"]["passes_used"] == 20


def test_passes_derived_from_transcript_for_read_passes(tmp_path):
    work = _work(tmp_path)
    # Read passes write no pass_<n>.json; the obs counter is the only record.
    (work / "transcript.jsonl").write_text(
        json.dumps({"turn": 1, "obs": "[pass 7/20] Read complete"}) + "\n"
    )
    assert scan_work_dir(str(work))["passes_used"] == 7


def test_code_runs_restored_from_obs_on_preemption_path(tmp_path):
    # F1: a hard preemption leaves no transcript.json; code_runs_used must be
    # recovered from the [code N/M, ...] obs rows _record_turn actually writes.
    work = _work(tmp_path)
    (work / "transcript.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"turn": 1, "reply": "", "obs": "[code 1/200, 0.5m / 60m] ok"}),
                json.dumps({"turn": 2, "reply": "", "obs": "[code 3/200, 1.0m / 60m] ok"}),
            ]
        )
        + "\n"
    )
    assert not (work / "transcript.json").exists()
    scan = scan_work_dir(str(work))
    assert scan["code_runs_used"] == 3
    assert scan["turns_completed"] == 2


def _row(turn: int, obs: str) -> str:
    return json.dumps({"turn": turn, "reply": "", "obs": obs})


def test_longer_jsonl_beats_stale_transcript_snapshot(tmp_path):
    # F2: attempt A exited cleanly (snapshot written), attempt B was hard-killed.
    # The live jsonl is fresher and must win for both counters.
    work = _work(tmp_path)
    (work / "transcript.json").write_text(
        json.dumps(
            {
                "summary": {"passes_used": 1, "code_runs_used": 5, "turns_completed": 2},
                "turns": [
                    {"turn": 1, "reply": "", "obs": "[code 5/200, 1.0m / 60m] ok"},
                    {"turn": 2, "reply": "", "obs": "[pass 1/20, 2.0m / 60m] gen"},
                ],
            }
        )
    )
    (work / "transcript.jsonl").write_text(
        "\n".join(
            [
                _row(1, "[code 5/200, 1.0m / 60m] ok"),
                _row(2, "[pass 1/20, 2.0m / 60m] gen"),
                _row(3, "[code 6/200, 3.0m / 60m] ok"),
                _row(4, "[code 7/200, 4.0m / 60m] ok"),
                _row(5, "[code 8/200, 5.0m / 60m] ok"),
                _row(6, "[pass 3/20, 6.0m / 60m] read"),
            ]
        )
        + "\n"
    )
    scan = scan_work_dir(str(work))
    assert scan["turns_completed"] == 6
    assert scan["code_runs_used"] == 8
    assert scan["passes_used"] == 3


def test_read_only_chain_keeps_passes_from_jsonl(tmp_path):
    # F2: attempt A ends after pass 1 (pass_1.json exists); resume B adds two
    # read passes (no pass_<n>.json) and is hard-killed. passes_used must be 3.
    work = _work(tmp_path)
    (work / "pass_1.json").write_text("{}")
    (work / "transcript.json").write_text(
        json.dumps(
            {
                "summary": {"passes_used": 1, "code_runs_used": 0},
                "turns": [
                    {"turn": 1, "reply": "", "obs": "[code 1/200, 1.0m / 60m] ok"},
                    {"turn": 2, "reply": "", "obs": "[pass 1/20, 2.0m / 60m] gen"},
                ],
            }
        )
    )
    (work / "transcript.jsonl").write_text(
        "\n".join(
            [
                _row(1, "[code 1/200, 1.0m / 60m] ok"),
                _row(2, "[pass 1/20, 2.0m / 60m] gen"),
                _row(3, "[pass 2/20, 3.0m / 60m] read"),
                _row(4, "[pass 3/20, 4.0m / 60m] read"),
            ]
        )
        + "\n"
    )
    scan = scan_work_dir(str(work))
    assert scan["turns_completed"] == 4
    assert scan["passes_used"] == 3


REQUIRED_METRICS_KEYS = {"cache_tag", "phases", "turns", "counters", "session_wall_s"}


def _read_metrics(tmp_path: Path, session_id: str) -> dict:
    path = _work_dir(tmp_path, session_id) / "metrics.json"
    assert path.exists(), f"missing metrics.json for {session_id}"
    return json.loads(path.read_text())


def test_withdraw_writes_complete_metrics_json(session_env):
    # D3: the scripted editor replies FINAL_WITHDRAW on Turn 1.
    session, _engine, tmp_path = session_env
    transcript = _run(session, tmp_path, session_id="s_withdraw", reset_active_lora=True)

    metrics = _read_metrics(tmp_path, "s_withdraw")
    assert REQUIRED_METRICS_KEYS <= set(metrics)
    assert metrics["cache_tag"]
    assert isinstance(metrics["phases"], dict)
    assert isinstance(metrics["turns"], list)
    assert isinstance(metrics["counters"], dict)
    assert metrics["session_wall_s"] >= 0

    summary = transcript["summary"]
    assert summary["final_status"] == "WITHDRAWN"
    assert {"passes_used", "code_runs_used", "resumed"} <= set(summary)


def test_resume_withdraw_writes_complete_metrics_json(session_env):
    # The resumed finalization must also write a full metrics.json on disk.
    session, _engine, tmp_path = session_env
    _seed_dirty_work(tmp_path, session_id="s_resume_metrics")
    transcript = _run(session, tmp_path, session_id="s_resume_metrics")

    metrics = _read_metrics(tmp_path, "s_resume_metrics")
    assert REQUIRED_METRICS_KEYS <= set(metrics)
    assert metrics["cache_tag"]
    assert metrics["session_wall_s"] >= 0

    summary = transcript["summary"]
    assert summary["final_status"] == "WITHDRAWN"
    assert summary["resumed"] is True
    assert summary["previous_passes_used"] == 19
    assert summary["passes_used"] == 19
    assert summary["code_runs_used"] == 5


def test_bad_compaction_budget_fails_before_engine(session_env, monkeypatch):
    # N1: a bad compaction/context pairing must fail on the orchestrator at $0,
    # before any GPU engine is constructed.
    session, _engine, tmp_path = session_env
    constructed = []
    monkeypatch.setattr(session, "QwenSelfRewriteEngine", lambda: constructed.append("engine"))
    monkeypatch.setattr(session, "COMPACT_TOKEN_LIMIT", 90_000)
    monkeypatch.setattr(session, "MAX_MODEL_LEN", 65_536)
    with pytest.raises(ValueError, match="114_688"):
        _run(session, tmp_path, session_id="s_guard", reset_active_lora=True)
    assert constructed == []


def _seed_census(
    tmp_path: Path,
    session_id: str,
    *,
    max_tokens: int = 64,
    cache_tag: str = "cachetag0000",
) -> Path:
    work = _work_dir(tmp_path, session_id)
    work.mkdir(parents=True, exist_ok=True)
    _write_active_lora(tmp_path, session_id)
    items = [{"id": i, "problem": f"p{i}", "answer": str(i)} for i in range(2)]
    _, meta = diagnosis_identity("math500", items, max_tokens)
    meta = {
        **meta,
        "cache_tag": cache_tag,
        "cache_fingerprint": {},
        "diag_n": len(items),
        "metrics_version": 1,
    }
    rows = [
        {
            "id": item["id"],
            "sample_idx": 0,
            "problem": item["problem"],
            "answer": item["answer"],
            "text": "</think>\\boxed{" + item["answer"] + "}",
            "tokens": 1,
            "finish_reason": "stop",
            "cut": False,
            "loop": False,
            "correct": False,
        }
        for item in items
    ]
    save_diagnosis_cache(str(work / "census.jsonl"), meta, rows)
    return work


def test_reused_census_never_rewrites_shared_cache(session_env):
    session, engine, tmp_path = session_env
    _seed_census(tmp_path, "s_shared_census")
    shared = tmp_path / "cache" / "diag_math500_2_64_cachetag0000.jsonl"
    shared.parent.mkdir(parents=True)
    shared.write_text('{"sentinel": true}\n')
    before = shared.read_bytes()

    summary = _run(
        session,
        tmp_path,
        session_id="s_shared_census",
        reuse_diagnosis=True,
        reset_active_lora=False,
    )

    assert summary["summary"]["resumed"] is True
    assert engine.generate_calls == 0  # census re-graded, not regenerated
    assert shared.read_bytes() == before


def test_overwrite_wipes_census_and_regenerates(session_env):
    session, engine, tmp_path = session_env
    _seed_census(tmp_path, "s_census_over")

    summary = _run(
        session,
        tmp_path,
        session_id="s_census_over",
        reuse_diagnosis=True,
        overwrite_session=True,
        reset_active_lora=True,
    )

    assert summary["summary"]["resumed"] is False
    # The stale census was wiped, so diagnosis had to be generated again.
    assert engine.generate_calls == 1
    assert (_work_dir(tmp_path, "s_census_over") / "census.jsonl").exists()


def test_census_participates_in_dirty_state(tmp_path):
    work = _work(tmp_path)
    (work / "census.jsonl").write_text("{}\n")
    scan = scan_work_dir(str(work))
    assert scan["has_work"] is True
    assert "census.jsonl" in scan["dirty_reasons"]


def test_resume_at_turn_budget_does_not_exceed(session_env):
    session, engine, tmp_path = session_env
    _seed_dirty_work(tmp_path)

    summary = _run(session, tmp_path, max_turns=2)

    assert summary["summary"]["final_status"] == "MAX_TURNS"
    assert summary["summary"]["turns_completed"] == 2
    assert summary["summary"]["turns_recorded"] == 2
    assert engine.editor_messages == []


def test_resume_wall_budget_is_cumulative_across_preemption(session_env):
    session, engine, tmp_path = session_env
    _seed_dirty_work(tmp_path)
    state = {
        "schema": 1,
        "turns_used_prior": 2,
        "elapsed_wall_prior_s": 3600.0,
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "RESUME-MARKER"},
        ],
        "passes_used": 19,
        "code_runs_used": 5,
        "resume_lineage": [{"attempt": 1}],
    }
    (_work_dir(tmp_path, "s_resume") / "resume_state.json").write_text(json.dumps(state))

    summary = _run(session, tmp_path, max_turns=10, wall_limit_hours=1.0)

    assert summary["summary"]["final_status"] == "WALL_LIMIT"
    assert summary["summary"]["turns_completed"] == 2
    assert summary["summary"]["turns_used_prior"] == 2
    assert summary["summary"]["elapsed_sec"] >= 3600
    assert summary["summary"]["resume_lineage"][-1]["attempt"] == 2
    assert engine.editor_messages == []


def test_resume_restores_messages_and_does_not_disclose_turn(session_env):
    session, engine, tmp_path = session_env
    work = _work_dir(tmp_path, "s_messages")
    work.mkdir(parents=True)
    _write_active_lora(tmp_path, "s_messages")
    (work / "pass_1.json").write_text("{}")
    (work / "transcript.jsonl").write_text(json.dumps({"turn": 1, "reply": "", "obs": "ok"}) + "\n")
    state = {
        "schema": 1,
        "turns_used_prior": 1,
        "elapsed_wall_prior_s": 0.0,
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "RESUME-MARKER"},
        ],
        "passes_used": 1,
        "code_runs_used": 0,
        "resume_lineage": [],
    }
    (work / "resume_state.json").write_text(json.dumps(state))

    summary = _run(
        session,
        tmp_path,
        session_id="s_messages",
        max_turns=5,
        reset_active_lora=False,
    )

    assert summary["summary"]["resumed"] is True
    assert summary["turns"][-1]["turn"] == 2
    first_call = engine.editor_messages[0]
    assert first_call[1]["content"] == "RESUME-MARKER"
    assert all("Begin Turn 1" not in message["content"] for message in first_call)


def test_resume_fallback_kickoff_has_no_turn_number(session_env):
    session, engine, tmp_path = session_env
    _seed_dirty_work(tmp_path)

    _run(session, tmp_path, max_turns=5)

    first_call = engine.editor_messages[0]
    assert first_call[1]["content"].startswith("Session resumed")
    assert all("Begin Turn 1" not in message["content"] for message in first_call)


# ---------------------------------------------------------------------------
# P0-3/P0-4 submission boundary: adapter whitelist + SUBMIT provenance.
# ---------------------------------------------------------------------------


def _helpers_with_active_edits():
    helpers = _fake_helpers(None, None, None)
    helpers["list_edits"] = lambda: [
        {
            "module": "layers.8.mlp.down_proj",
            "lora_A_norm": 1.0,
            "lora_B_norm": 1.0,
        }
    ]
    return helpers


def test_final_submit_rejects_out_of_scope_adapter_key(session_env, monkeypatch):
    session, engine, tmp_path = session_env

    def editor(messages, *_args, **_kwargs):
        engine.editor_messages.append(messages)
        return {
            "text": "</think>\nFINAL_SUBMIT",
            "prompt_tokens": 1,
            "tokens": 1,
            "finish_reason": "stop",
            "timings": {},
            "gpu_memory": {},
        }

    engine.editor_step = _Remote(editor)
    monkeypatch.setattr(
        session, "bind_sandbox_helpers", lambda *_a, **_k: _helpers_with_active_edits()
    )

    def _reject(_lora_dir, _registry, **_kwargs):
        raise ValueError("LoRA tensor key whitelist violation: unknown=['lm_head.lora_A.weight']")

    monkeypatch.setattr(session, "load_and_validate_adapter", _reject)

    summary = _run(
        session,
        tmp_path,
        session_id="s_bad_adapter",
        max_turns=1,
        reset_active_lora=True,
    )

    assert summary["summary"]["final_status"] == "MAX_TURNS"
    assert summary["summary"]["submitted_lora"] is False
    assert any("whitelist" in str(turn.get("obs") or "") for turn in summary["turns"])


def test_final_submit_writes_submit_provenance(session_env, monkeypatch):
    from lora_self_rewrite.adapter import canonical_adapter_config

    session, engine, tmp_path = session_env
    calls = {"n": 0}

    def editor(messages, *_args, **_kwargs):
        engine.editor_messages.append(messages)
        calls["n"] += 1
        text = "</think>\nFINAL_SUBMIT" if calls["n"] == 1 else "</think>\nSUBMIT"
        return {
            "text": text,
            "prompt_tokens": 1,
            "tokens": 1,
            "finish_reason": "stop",
            "timings": {},
            "gpu_memory": {},
        }

    def generate(problems, *_args, **kwargs):
        n_samples = int(kwargs.get("n_samples", 1))
        rows = [
            {
                "id": problem["id"],
                "sample_idx": sample_idx,
                "problem": problem["problem"],
                "answer": problem["answer"],
                "text": "</think>\\boxed{" + problem["answer"] + "}",
                "tokens": 1,
                "finish_reason": "stop",
                "cut": False,
                "loop": False,
            }
            for problem in problems
            for sample_idx in range(n_samples)
        ]
        return {
            "base": [dict(row) for row in rows],
            "candidates": {"active": [dict(row) for row in rows]},
            "timings": {"n_answers": len(rows), "generated_tokens": len(rows)},
            "gpu_memory": {},
        }

    engine.editor_step = _Remote(editor)
    engine.generate_batch = _Remote(generate)
    monkeypatch.setattr(
        session, "bind_sandbox_helpers", lambda *_a, **_k: _helpers_with_active_edits()
    )
    monkeypatch.setattr(session, "load_and_validate_adapter", lambda *_a, **_k: ({}, {}))

    lora = tmp_path / "sessions" / "s_submit" / "active_lora"
    lora.mkdir(parents=True)
    (lora / "adapter_model.safetensors").write_bytes(b"weights")
    (lora / "adapter_config.json").write_text(json.dumps(canonical_adapter_config(rank=4)))

    summary = _run(
        session,
        tmp_path,
        session_id="s_submit",
        reset_active_lora=True,
        max_turns=2,
    )

    assert summary["summary"]["final_status"] == "SUBMITTED"
    assert summary["summary"]["submitted_lora"] is True
    provenance_path = _work_dir(tmp_path, "s_submit") / "submitted_lora" / "submit_provenance.json"
    provenance = json.loads(provenance_path.read_text())
    assert provenance["decision"] == "SUBMIT"
    assert provenance["session_id"] == "s_submit"


def _install_submit_path(
    session,
    engine,
    tmp_path: Path,
    session_id: str,
    monkeypatch,
    *,
    edited_correct: bool,
) -> dict:
    from lora_self_rewrite.adapter import canonical_adapter_config

    calls = {"editor": 0, "generate": 0, "validation": 0}

    def editor(messages, *_args, **_kwargs):
        engine.editor_messages.append(messages)
        calls["editor"] += 1
        text = "</think>\nFINAL_SUBMIT" if calls["editor"] == 1 else "</think>\nSUBMIT"
        return {
            "text": text,
            "prompt_tokens": 1,
            "tokens": 1,
            "finish_reason": "stop",
            "timings": {},
            "gpu_memory": {},
        }

    def generate(problems, *_args, **kwargs):
        calls["generate"] += 1
        n_samples = int(kwargs.get("n_samples", 1))
        # The reserved-validation gate is the only generate call carrying an active
        # LoRA spec; diagnosis calls pass an empty lora_specs mapping.
        is_validation = bool(kwargs.get("lora_specs"))
        if is_validation:
            calls["validation"] += 1

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

        def correct_text(problem):
            return "</think>\\boxed{" + problem["answer"] + "}"

        def edited_text(problem):
            if edited_correct:
                return correct_text(problem)
            return "</think>\\boxed{definitely-wrong}"

        base = rows_for(correct_text)
        candidates = {"active": rows_for(edited_text)} if is_validation else {}
        return {
            "base": base,
            "candidates": candidates,
            "timings": {
                "n_answers": len(base) + sum(len(rows) for rows in candidates.values()),
                "generated_tokens": len(base) + sum(len(rows) for rows in candidates.values()),
            },
            "gpu_memory": {},
        }

    engine.editor_step = _Remote(editor)
    engine.generate_batch = _Remote(generate)
    monkeypatch.setattr(
        session, "bind_sandbox_helpers", lambda *_a, **_k: _helpers_with_active_edits()
    )
    monkeypatch.setattr(session, "load_and_validate_adapter", lambda *_a, **_k: ({}, {}))

    lora = tmp_path / "sessions" / session_id / "active_lora"
    lora.mkdir(parents=True)
    (lora / "adapter_model.safetensors").write_bytes(b"weights")
    (lora / "adapter_config.json").write_text(json.dumps(canonical_adapter_config(rank=4)))
    return calls


def test_final_submit_validation_is_off_budget_and_recorded(session_env, monkeypatch):
    session, engine, tmp_path = session_env
    calls = _install_submit_path(
        session, engine, tmp_path, "s_val_budget", monkeypatch, edited_correct=True
    )
    summary = _run(
        session,
        tmp_path,
        session_id="s_val_budget",
        reset_active_lora=True,
        max_turns=2,
    )

    assert calls == {"editor": 2, "generate": 2, "validation": 1}
    assert summary["summary"]["final_status"] == "SUBMITTED"
    # FINAL_SUBMIT validation is reserved and therefore off the adaptation budget.
    assert summary["summary"]["passes_used"] == 0
    work = _work_dir(tmp_path, "s_val_budget")
    assert not list(work.glob("pass_*.json"))
    assert not list(work.glob("pass_*.inflight"))

    metrics = _read_metrics(tmp_path, "s_val_budget")
    assert metrics["validation"]["n_val"] == 1
    assert metrics["validation"]["val_samples"] == 1
    assert metrics["validation"]["base_cache_hit"] is False
    assert metrics["validation"]["gpu_s"] >= 0
    assert metrics["counters"]["validation_cache_misses"] == 1
    assert metrics["counters"]["validation_cache_hits"] == 0
    assert metrics["phases"]["validation_s"] >= 0
    assert metrics["generate_calls"] == []

    # The validation run is recorded as its own transcript entry, separate from any
    # adaptation pass.
    last = summary["turns"][-1]
    assert last["val_stats"]["n_problems"] == 1
    assert last["val_stats"]["n_answers"] == 1
    assert "Reserved validation check" in last["obs"]
    assert last["metrics"]["validation_s"] >= 0
    assert last["metrics"]["action"] is None


def test_final_submit_without_pvalue_gate_still_submits(session_env, monkeypatch):
    session, engine, tmp_path = session_env
    calls = _install_submit_path(
        session, engine, tmp_path, "s_no_p_gate", monkeypatch, edited_correct=False
    )
    summary = _run(
        session,
        tmp_path,
        session_id="s_no_p_gate",
        reset_active_lora=True,
        max_turns=2,
    )

    assert calls["validation"] == 1
    last = summary["turns"][-1]
    stats = last["val_stats"]
    # The edited arm is worse than base and the one-sided p-value is uninformative;
    # the protocol never hard-blocks SUBMIT on that p-value.
    assert stats["base_acc"] == 1.0
    assert stats["edited_acc"] == 0.0
    assert stats["signflip_p_one_sided"] == 1.0
    assert summary["summary"]["final_status"] == "SUBMITTED"
    assert summary["summary"]["submitted_lora"] is True
    assert summary["summary"]["passes_used"] == 0

    submitted = _work_dir(tmp_path, "s_no_p_gate") / "submitted_lora"
    assert (submitted / "adapter_model.safetensors").read_bytes() == b"weights"
    assert (submitted / "adapter_config.json").exists()
    provenance = json.loads((submitted / "submit_provenance.json").read_text())
    assert provenance["decision"] == "SUBMIT"
    assert provenance["validation_stats"]["edited_acc"] == 0.0
    assert provenance["validation_stats"]["signflip_p_one_sided"] == 1.0
    # Snapshot provenance was rewritten from PENDING to the final SUBMIT decision.
    assert provenance["session_id"] == "s_no_p_gate"


def test_resume_fails_when_active_lora_is_missing(session_env):
    session, _engine, tmp_path = session_env
    work = _work_dir(tmp_path, "s_missing_adapter")
    work.mkdir(parents=True)
    (work / "pass_1.json").write_text("{}")
    (work / "transcript.jsonl").write_text(json.dumps({"turn": 1, "reply": "", "obs": "ok"}) + "\n")
    with pytest.raises(ValueError, match="active_lora is missing"):
        _run(session, tmp_path, session_id="s_missing_adapter")


def test_overwrite_forces_adapter_reset_even_if_caller_disabled_it(session_env, monkeypatch):
    session, _engine, tmp_path = session_env
    _seed_dirty_work(tmp_path, session_id="s_force_reset")
    captured = []

    def fake_init(_lora_dir, **kwargs):
        captured.append(kwargs)
        return {
            "layers.8.mlp.down_proj": {
                "layer": 8,
                "module_path": "layers.8.mlp.down_proj",
                "peft_prefix": "x",
                "full_key": "x",
                "shape": (16, 8),
                "shard_path": "x",
            }
        }

    monkeypatch.setattr(session, "init_empty_lora", fake_init)
    summary = _run(
        session,
        tmp_path,
        session_id="s_force_reset",
        overwrite_session=True,
        reset_active_lora=False,
    )
    assert captured and captured[0]["reset"] is True
    assert summary["summary"]["resumed"] is False
    assert summary["summary"]["reset_active_lora"] is True


def test_generate_max_tokens_above_session_budget_is_rejected(session_env):
    session, engine, tmp_path = session_env

    def editor(messages, *_args, **_kwargs):
        engine.editor_messages.append(messages)
        return {
            "text": (
                "</think>\n```json\n"
                '{"action": "generate", "problem_ids": [0], "max_tokens": 8192}'
                "\n```"
            ),
            "prompt_tokens": 1,
            "tokens": 1,
            "finish_reason": "stop",
            "timings": {},
            "gpu_memory": {},
        }

    engine.editor_step = _Remote(editor)
    summary = _run(
        session,
        tmp_path,
        session_id="s_budget_bypass",
        max_turns=1,
        reset_active_lora=True,
    )
    assert summary["summary"]["passes_used"] == 0
    assert any("session answer budget" in str(turn.get("obs") or "") for turn in summary["turns"])


def test_config_path_is_the_resolved_source_of_truth(session_env, tmp_path):
    session, _engine, _ = session_env
    config = tmp_path / "s_cfg.toml"
    config.write_text(
        f'''experiment_id = "s_config"
dataset_name = "math500"
model_id = "Qwen/Qwen3.5-27B"
model_revision = "{MODEL_REVISION}"

[protocol]
max_tokens = 64
passes_budget = 20
code_budget = 10
turns_budget = 1
wall_limit_hours = 1.0
max_gen_answers = 64
read_max_length = 8192
diag_n = 2
val_n = 1
val_samples = 1

[splits]
practice = 6
reserved = 4
heldout = 0
'''
    )
    session.run_self_rewrite_session(
        session_id="ignored",
        dataset_mode="math500",
        max_tokens=4096,
        diag_n=2,
        val_n=1,
        val_samples=1,
        max_turns=160,
        max_passes=20,
        max_code_runs=200,
        max_gen_answers=128,
        wall_limit_hours=3.0,
        run_preflight=False,
        preflight_sleep=False,
        reuse_diagnosis=False,
        reset_active_lora=True,
        overwrite_session=False,
        worker_timeout_s=60,
        config_path=str(config),
        deviation="unit-test: config resolution",
    )
    summary = json.loads((_work_dir(tmp_path, "s_config") / "transcript.json").read_text())[
        "summary"
    ]
    assert summary["session_id"] == "s_config"
    assert summary["max_tokens"] == 64
    assert summary["wall_limit_hours"] == 1.0
    assert summary["read_max_length"] == 8192
    assert summary["deviation"] == "unit-test: config resolution"
    assert summary["resolved_config"]["max_turns"] == 1
    assert summary["protocol_deviations"]
