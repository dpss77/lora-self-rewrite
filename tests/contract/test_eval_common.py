import pytest

from lora_self_rewrite.caches import (
    LoraIdRegistry,
    content_addressed_lora_key,
    load_base_batch_cache,
    save_base_batch_cache,
)
from lora_self_rewrite.eval_common import (
    assert_shard_identity,
    cache_tag_for,
    cut_at_token_limit,
    heldout_base_signature,
    heldout_lora_int_id,
    reload_with_retry,
    shard_problems,
)

pytestmark = pytest.mark.contract


def test_shard_problems_keeps_whole_problems_and_covers_all():
    problems = [{"id": i} for i in range(490)]
    shards = shard_problems(problems, 5)
    assert len(shards) == 5
    assert sum(len(s) for s in shards) == 490
    assert [p["id"] for s in shards for p in s] == list(range(490))
    assert max(len(s) for s in shards) - min(len(s) for s in shards) <= 1


def test_shard_problems_edge_cases():
    assert shard_problems([], 3) == []
    assert len(shard_problems([{"id": i} for i in range(2)], 10)) == 2
    with pytest.raises(ValueError):
        shard_problems([{"id": 0}], 0)


def test_cache_tag_is_stable_and_sensitive():
    a = cache_tag_for("NVIDIA H200", "2.14.0", "0.34.0")
    b = cache_tag_for("NVIDIA H200", "2.14.0", "0.34.0")
    c = cache_tag_for("NVIDIA H100", "2.14.0", "0.34.0")
    assert a == b
    assert a != c
    assert len(a) == 12


PROBLEMS = [{"id": i, "problem": f"p{i}", "answer": str(i)} for i in range(10)]


def _expected(problems, n_samples):
    return [
        {"id": p["id"], "sample_idx": s, "problem": p["problem"], "answer": p["answer"]}
        for p in problems
        for s in range(n_samples)
    ]


def _rows(problems, n_samples, correct=False):
    return [
        {
            "id": p["id"],
            "sample_idx": s,
            "problem": p["problem"],
            "answer": p["answer"],
            "text": "</think>\\boxed{" + p["answer"] + "}",
            "tokens": 1,
            "cut": False,
            "loop": False,
            "correct": correct,
        }
        for p in problems
        for s in range(n_samples)
    ]


def test_heldout_base_signature_tracks_composition_and_revision():
    base = heldout_base_signature(PROBLEMS, 2, 64, "tag", 5, "rev1")
    assert heldout_base_signature(PROBLEMS, 2, 64, "tag", 5, "rev1") == base
    # k_shards determines shard/batch composition.
    assert heldout_base_signature(PROBLEMS, 2, 64, "tag", 10, "rev1")[0] != base[0]
    # Model revision and dataset content/order are part of the identity.
    assert heldout_base_signature(PROBLEMS, 2, 64, "tag", 5, "rev2")[0] != base[0]
    assert heldout_base_signature(list(reversed(PROBLEMS)), 2, 64, "tag", 5, "rev1")[0] != base[0]
    altered = [dict(p) for p in PROBLEMS]
    altered[0]["problem"] = "changed"
    assert heldout_base_signature(altered, 2, 64, "tag", 5, "rev1")[0] != base[0]


def test_heldout_cache_hit_regrades_and_k_shards_miss(tmp_path):
    problems = PROBLEMS[:4]
    signature, meta = heldout_base_signature(problems, 2, 64, "tag", 5, "rev1")
    path = str(tmp_path / f"{signature}.json")
    # Stored labels are deliberately wrong; load must re-grade from text.
    save_base_batch_cache(path, meta, _rows(problems, 2, correct=False))
    expected = _expected(problems, 2)

    graded = []
    loaded = load_base_batch_cache(
        path, meta, expected, lambda text, answer: graded.append((text, answer)) or True
    )
    assert loaded is not None
    assert all(row["correct"] is True for row in loaded)
    assert len(graded) == len(expected)  # re-graded on load, not trusted

    # Same ordered batch at a different k_shards must miss.
    other_signature, other_meta = heldout_base_signature(problems, 2, 64, "tag", 10, "rev1")
    assert other_signature != signature
    assert load_base_batch_cache(path, other_meta, expected, lambda *_: True) is None


def _adapter(tmp_path, name, blob):
    adapter = tmp_path / name
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(blob)
    (adapter / "adapter_config.json").write_text("{}")
    return str(adapter)


def test_heldout_lora_id_is_content_addressed(tmp_path):
    registry = LoraIdRegistry()
    first = _adapter(tmp_path, "a", b"one")
    second = _adapter(tmp_path, "b", b"two")
    first_id = heldout_lora_int_id(registry, first)
    assert first_id == heldout_lora_int_id(registry, first)
    second_id = heldout_lora_int_id(registry, second)
    # Distinct adapter bytes must never share an id in one warm container.
    assert second_id != first_id
    assert first_id >= 1 and second_id >= 1


def test_shard_identity_mismatch_fails():
    ok = {"cache_tag": "tag", "adapter_sha256": "sha"}
    # Matching identity passes for both base (no adapter) and edited shards.
    assert_shard_identity(ok, expected_cache_tag="tag", expected_adapter_sha256="sha")
    assert_shard_identity({"cache_tag": "tag"}, expected_cache_tag="tag")
    with pytest.raises(RuntimeError, match="cache-tag drift"):
        assert_shard_identity(ok, expected_cache_tag="other")
    with pytest.raises(RuntimeError, match="adapter mismatch"):
        assert_shard_identity(ok, expected_cache_tag="tag", expected_adapter_sha256="other")


def test_shard_measures_its_own_adapter_bytes(tmp_path):
    # Orchestrator hashes its mount; the shard measures the bytes it loaded.
    # A stale shard mount with different bytes must be caught.
    orchestrator_dir = _adapter(tmp_path, "orchestrator", b"bytes-A")
    stale_shard_dir = _adapter(tmp_path, "stale-shard", b"bytes-B")
    expected = content_addressed_lora_key(orchestrator_dir)
    measured_by_shard = content_addressed_lora_key(stale_shard_dir)
    assert measured_by_shard != expected
    with pytest.raises(RuntimeError, match="adapter mismatch"):
        assert_shard_identity(
            {"cache_tag": "tag", "adapter_sha256": measured_by_shard},
            expected_cache_tag="tag",
            expected_adapter_sha256=expected,
        )
    # A shard that loaded the same bytes passes.
    same_shard_dir = _adapter(tmp_path, "same-shard", b"bytes-A")
    assert_shard_identity(
        {
            "cache_tag": "tag",
            "adapter_sha256": content_addressed_lora_key(same_shard_dir),
        },
        expected_cache_tag="tag",
        expected_adapter_sha256=expected,
    )


def test_cut_matches_engine_definition_for_truncated_output():
    # Max-token-truncated but vLLM finished with a non-"length" reason: the old
    # held-out definition (finish_reason only) would have labelled it uncut.
    max_tokens = 16_384
    assert ("stop" == "length") is False  # old held-out definition
    assert (max_tokens >= max_tokens or "stop" == "length") is True  # engine/donor
    assert cut_at_token_limit(max_tokens, max_tokens, "stop") is True
    # Both definitions agree for the ordinary cases.
    assert cut_at_token_limit(100, max_tokens, "length") is True
    assert cut_at_token_limit(100, max_tokens, "stop") is False


def test_reload_with_retry_recovers_from_transient_failure():
    calls = []

    class _Volume:
        def reload(self):
            calls.append(1)
            if len(calls) == 1:
                raise OSError("transient")

    reload_with_retry(_Volume(), "test")
    assert len(calls) == 2


def test_reload_with_retry_raises_after_attempts():
    calls = []

    class _Volume:
        def reload(self):
            calls.append(1)
            raise OSError("down")

    with pytest.raises(RuntimeError, match="after 2 attempts"):
        reload_with_retry(_Volume(), "test")
    assert len(calls) == 2


def test_heldout_base_signature_tracks_engine_config_and_dataset_identity():
    base = heldout_base_signature(
        PROBLEMS,
        2,
        64,
        "tag",
        5,
        "rev1",
        engine_config_hash="engine1",
        dataset_identity="dataset1",
    )
    assert (
        heldout_base_signature(
            PROBLEMS,
            2,
            64,
            "tag",
            5,
            "rev1",
            engine_config_hash="engine1",
            dataset_identity="dataset1",
        )[0]
        == base[0]
    )
    assert (
        heldout_base_signature(
            PROBLEMS,
            2,
            64,
            "tag",
            5,
            "rev1",
            engine_config_hash="engine2",
            dataset_identity="dataset1",
        )[0]
        != base[0]
    )
    assert (
        heldout_base_signature(
            PROBLEMS,
            2,
            64,
            "tag",
            5,
            "rev1",
            engine_config_hash="engine1",
            dataset_identity="dataset2",
        )[0]
        != base[0]
    )
