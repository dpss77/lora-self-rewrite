import json

import pytest

from lora_self_rewrite import caches
from lora_self_rewrite.caches import (
    CUT_DEFINITION,
    LoraIdRegistry,
    base_batch_signature,
    content_addressed_lora_key,
    diagnosis_identity,
    label_schema,
    load_base_batch_cache,
    load_diagnosis_cache,
    load_validation_base_cache,
    norm_question,
    save_base_batch_cache,
    save_diagnosis_cache,
    save_validation_base_cache,
)
from lora_self_rewrite.constants import BASE_CACHE_SCHEMA

pytestmark = pytest.mark.contract


def test_norm_question_collapses_whitespace():
    assert norm_question("  A   B\nC ") == "a b c"


def test_base_batch_signature_is_exact_batch_identity():
    args = ("deepscaler", [1, 2], ["a", "b"], 2, 4096, "tag")
    sig1, meta1 = base_batch_signature(*args)
    sig2, _ = base_batch_signature(*args)
    assert sig1 == sig2
    assert meta1["signature"] == sig1
    assert meta1["generation_config_hash"]
    # Any component change must change the signature.
    assert base_batch_signature("deepscaler", [1, 2], ["a", "b"], 3, 4096, "tag")[0] != sig1
    assert base_batch_signature("deepscaler", [2, 1], ["a", "b"], 2, 4096, "tag")[0] != sig1
    assert base_batch_signature("deepscaler", [1, 2], ["a", "b"], 2, 8192, "tag")[0] != sig1
    assert base_batch_signature("deepscaler", [1, 2], ["a", "b"], 2, 4096, "x")[0] != sig1


def test_base_cache_roundtrip_and_regrade(tmp_path):
    path = str(tmp_path / "cache.json")
    expected = [
        {"id": 1, "sample_idx": 0, "problem": "p1", "answer": "4"},
    ]
    rows = [
        {
            "id": 1,
            "sample_idx": 0,
            "problem": "p1",
            "answer": "4",
            "text": "<think>x</think>\\boxed{4}",
            "tokens": 1,
            "cut": 0,
            "loop": 0,
            "correct": False,
        }
    ]
    meta = {"signature": "abc"}
    save_base_batch_cache(path, meta, rows)

    calls = []

    def grade(text, answer):
        calls.append((text, answer))
        return True

    loaded = load_base_batch_cache(path, meta, expected, grade)
    assert loaded is not None
    assert loaded[0]["correct"] is True
    assert calls == [(rows[0]["text"], "4")]

    assert load_base_batch_cache(path, {"signature": "other"}, expected, grade) is None
    assert load_base_batch_cache(path, meta, [], grade) is None


def test_lora_id_registry_is_content_addressed(tmp_path):
    d1 = tmp_path / "a"
    d2 = tmp_path / "b"
    for d, blob in ((d1, b"one"), (d2, b"two")):
        d.mkdir()
        (d / "adapter_model.safetensors").write_bytes(blob)
        (d / "adapter_config.json").write_text("{}")

    reg = LoraIdRegistry()
    a1 = reg.id_for_dir(str(d1))
    a2 = reg.id_for_dir(str(d1))
    b1 = reg.id_for_dir(str(d2))
    assert a1 == a2
    assert b1 == a1 + 1
    assert a1 >= 1
    assert content_addressed_lora_key(str(d1)) != content_addressed_lora_key(str(d2))


def test_base_cache_handles_missing_file(tmp_path):
    assert load_base_batch_cache(str(tmp_path / "nope.json"), {}, [], lambda *_: True) is None


def _label_rows(text="ok", cut=False, loop=False):
    return [
        {
            "id": 1,
            "sample_idx": 0,
            "problem": "p",
            "answer": "4",
            "text": text,
            "tokens": 1,
            "cut": cut,
            "loop": loop,
        }
    ]


def _label_expected():
    return [{"id": 1, "sample_idx": 0, "problem": "p", "answer": "4"}]


def test_label_schema_is_pinned_and_schema_bumped():
    assert BASE_CACHE_SCHEMA == 3
    assert CUT_DEFINITION == "tokens>=max_tokens or finish_reason==length"
    assert label_schema() == {
        "loop_metric_mode": caches.LOOP_METRIC_MODE,
        "cut_definition": CUT_DEFINITION,
    }
    _, meta = base_batch_signature("m", [1], ["p"], 1, 4096, "tag")
    assert meta["label_schema"] == label_schema()


def test_loop_mode_change_invalidates_signature_and_cache(tmp_path, monkeypatch):
    signature, meta = base_batch_signature("m", [1], ["p"], 1, 4096, "tag")
    path = str(tmp_path / f"{signature}.json")
    save_base_batch_cache(path, meta, _label_rows())

    monkeypatch.setattr(caches, "LOOP_METRIC_MODE", "bytes")
    new_signature, new_meta = base_batch_signature("m", [1], ["p"], 1, 4096, "tag")
    assert new_signature != signature
    assert load_base_batch_cache(path, new_meta, _label_expected(), lambda *_: True) is None


def test_cut_definition_change_invalidates_signature_and_cache(tmp_path, monkeypatch):
    signature, meta = base_batch_signature("m", [1], ["p"], 1, 4096, "tag")
    path = str(tmp_path / f"{signature}.json")
    save_base_batch_cache(path, meta, _label_rows())

    monkeypatch.setattr(caches, "CUT_DEFINITION", "different-cut")
    new_signature, new_meta = base_batch_signature("m", [1], ["p"], 1, 4096, "tag")
    assert new_signature != signature
    assert load_base_batch_cache(path, new_meta, _label_expected(), lambda *_: True) is None


def test_load_rederives_loop_label(tmp_path):
    _, meta = base_batch_signature("m", [1], ["p"], 1, 4096, "tag")
    path = str(tmp_path / "cache.json")
    # Stored loop label is stale: a repetitive, cut answer must re-derive to True.
    save_base_batch_cache(path, meta, _label_rows(text="abc" * 3000, cut=True, loop=False))
    loaded = load_base_batch_cache(path, meta, _label_expected(), lambda *_: True)
    assert loaded is not None
    assert loaded[0]["loop"] is True


def _diag_items():
    return [{"id": 1, "problem": "p", "answer": "4"}]


def _diag_rows(text="</think>\\boxed{4}", tokens=1, finish_reason="stop"):
    return [
        {
            "id": 1,
            "sample_idx": 0,
            "problem": "p",
            "answer": "4",
            "text": text,
            "tokens": tokens,
            "finish_reason": finish_reason,
            "cut": False,
            "loop": False,
            "correct": False,
        }
    ]


def test_diagnosis_identity_binds_budget_labels_and_grader():
    items = _diag_items()
    sig, meta = diagnosis_identity("math500", items, 4096)
    assert meta["signature"] == sig
    assert meta["max_tokens"] == 4096
    assert meta["answer_hashes"]
    assert meta["dataset_identity"]
    assert meta["grader_identity"]
    assert diagnosis_identity("math500", items, 2048)[0] != sig
    changed = [{"id": 1, "problem": "p", "answer": "5"}]
    assert diagnosis_identity("math500", changed, 4096)[0] != sig


def test_diagnosis_cache_rejects_stale_max_tokens(tmp_path):
    items = _diag_items()
    _, meta = diagnosis_identity("math500", items, 4096)
    path = str(tmp_path / "census.jsonl")
    save_diagnosis_cache(path, meta, _diag_rows())
    _, stale_meta = diagnosis_identity("math500", items, 2048)
    assert load_diagnosis_cache(path, stale_meta, items, lambda *_: True) is None


def test_diagnosis_cache_regrades_and_reports_label_schema_change(tmp_path):
    items = _diag_items()
    _, meta = diagnosis_identity("math500", items, 4096)
    path = str(tmp_path / "census.jsonl")
    stale_meta = {
        **meta,
        "label_schema": {"loop_metric_mode": "old", "cut_definition": "old"},
    }
    save_diagnosis_cache(path, stale_meta, _diag_rows())

    result = load_diagnosis_cache(path, meta, items, lambda *_: True)
    assert result is not None
    rows, status = result
    assert status == "regraded"
    assert rows[0]["correct"] is True


def test_diagnosis_cache_rederives_cut_loop_and_correct_from_text(tmp_path):
    items = _diag_items()
    _, meta = diagnosis_identity("math500", items, 4096)
    path = str(tmp_path / "census.jsonl")
    save_diagnosis_cache(
        path,
        meta,
        _diag_rows(text="abc" * 3000, tokens=4096, finish_reason="length"),
    )

    result = load_diagnosis_cache(path, meta, items, lambda *_: False)
    assert result is not None
    rows, status = result
    assert status == "hit"
    assert rows[0]["cut"] is True
    assert rows[0]["loop"] is True
    assert rows[0]["correct"] is False


def test_diagnosis_legacy_text_compatibility_regrades_without_generation(tmp_path):
    items = _diag_items()
    _, meta = diagnosis_identity("math500", items, 4096)
    path = str(tmp_path / "census.jsonl")
    (tmp_path / "census.jsonl").write_text("\n".join(json.dumps(r) for r in _diag_rows()) + "\n")
    # No sidecar: content-match compatibility path re-grades the paid text.
    result = load_diagnosis_cache(path, meta, items, lambda *_: True)
    assert result is not None
    rows, status = result
    assert status == "legacy"
    assert rows[0]["correct"] is True


def _validation_rows(answer="4"):
    return [
        {
            "id": 1,
            "sample_idx": 0,
            "problem": "p",
            "answer": answer,
            "text": "</think>\\boxed{4}",
            "tokens": 1,
            "cut": False,
            "loop": False,
            "correct": False,
        }
    ]


def test_validation_cache_identity_includes_answer_and_regrades(tmp_path):
    path = str(tmp_path / "val.json")
    expected = [
        {"id": 1, "sample_idx": 0, "problem": "p", "answer": "4"},
    ]
    save_validation_base_cache(
        path,
        _validation_rows("4"),
        engine_config_hash="engine1",
        max_tokens=4096,
    )

    calls = []

    def grade(text, answer):
        calls.append((text, answer))
        return answer == "4"

    loaded = load_validation_base_cache(
        path,
        expected,
        engine_config_hash="engine1",
        max_tokens=4096,
        grade=grade,
    )
    assert loaded is not None
    assert loaded[0]["correct"] is True
    assert calls == [(loaded[0]["text"], "4")]

    # A changed answer must miss even when id/problem/sample_idx are unchanged.
    changed = [{"id": 1, "sample_idx": 0, "problem": "p", "answer": "5"}]
    assert (
        load_validation_base_cache(
            path,
            changed,
            engine_config_hash="engine1",
            max_tokens=4096,
            grade=grade,
        )
        is None
    )
    # Engine-config drift must also miss.
    assert (
        load_validation_base_cache(
            path,
            expected,
            engine_config_hash="engine2",
            max_tokens=4096,
            grade=grade,
        )
        is None
    )
