from pathlib import Path

import pytest

from lora_self_rewrite.datasets import (
    DATASET_SOURCE_REVISIONS,
    build_deepscaler_payload,
    dataset_identity,
    dataset_source_revisions,
    heldout_problems,
    require_pinned_sources,
    split_payload,
    validate_heldout_pool,
)

pytestmark = pytest.mark.unit

SRC = Path(__file__).resolve().parents[2] / "src" / "lora_self_rewrite"


def test_split_payload_uses_fixed_counts():
    clean = [{"id": i, "problem": f"p{i}", "answer": str(i)} for i in range(5000)]
    payload = split_payload(clean)
    assert len(payload["practice"]) == 3000
    assert len(payload["reserved"]) == 300
    assert len(payload["heldout"]) == 1000


def test_heldout_problems_slice_is_500_990():
    pool = [{"id": i} for i in range(1000)]
    out = heldout_problems({"heldout": pool})
    assert len(out) == 490
    assert out[0]["id"] == 500
    assert out[-1]["id"] == 989


def _fake_loader(repo, cfg, split):
    if repo == "agentica-org/DeepScaleR-Preview-Dataset":
        return [{"problem": f"deep problem {i}", "answer": str(i)} for i in range(4400)]
    if repo == "Idavidrein/gpqa":
        raise RuntimeError("gated")
    col = {
        "HuggingFaceH4/MATH-500": "problem",
        "Maxwell-Jia/AIME_2024": "Problem",
        "math-ai/aime25": "problem",
        "math-ai/amc23": "question",
        "MathArena/hmmt_feb_2025": "problem",
        "Hothan/OlympiadBench": "question",
    }[repo]
    return [{col: f"eval {repo} {cfg}"}]


def test_build_payload_scrubs_and_splits():
    payload = build_deepscaler_payload(_fake_loader)
    assert payload["meta"]["n_math_ok"] == 7
    assert payload["meta"]["gpqa_ok"] is False
    assert len(payload["practice"]) == 3000
    assert len(payload["reserved"]) == 300
    assert len(payload["heldout"]) == 1000
    assert payload["meta"]["n_clean"] == 4400
    assert len(heldout_problems(payload)) == 490


def test_build_payload_requires_all_math_sources():
    def loader(repo, cfg, split):
        if repo == "HuggingFaceH4/MATH-500":
            return [{"problem": "x"}]
        raise RuntimeError("missing")

    with pytest.raises(RuntimeError):
        build_deepscaler_payload(loader)


def test_split_payload_rejects_short_pool():
    short = [{"id": i, "problem": f"p{i}", "answer": str(i)} for i in range(4299)]
    with pytest.raises(ValueError, match="at least 4300"):
        split_payload(short)
    allowed = split_payload(short, allow_short=True)
    assert len(allowed["heldout"]) == 999


def test_validate_heldout_pool_enforces_fixed_size():
    full = [{"id": i} for i in range(490)]
    assert len(validate_heldout_pool(full)) == 490

    short = [{"id": i} for i in range(100)]
    with pytest.raises(ValueError, match="expected 490"):
        validate_heldout_pool(short)
    assert len(validate_heldout_pool(short, smoke_problems=10)) == 10

    with pytest.raises(ValueError, match="empty"):
        validate_heldout_pool([])


def test_eval_heldout_calls_hard_pool_check():
    src = (SRC / "eval_heldout.py").read_text()
    assert "validate_heldout_pool(" in src


def test_payload_records_pinned_source_revisions_and_content_hash():
    payload = build_deepscaler_payload(_fake_loader)
    revisions = payload["meta"]["source_revisions"]
    assert revisions
    assert all(value and value != "unknown" for value in revisions.values())
    assert (
        revisions["agentica-org/DeepScaleR-Preview-Dataset"]
        == (DATASET_SOURCE_REVISIONS["agentica-org/DeepScaleR-Preview-Dataset"])
    )
    assert payload["meta"]["dataset_sha256"]


def test_loader_receives_pinned_revision():
    seen = []

    def loader(repo, cfg, split, revision=None):
        seen.append((repo, revision))
        return _fake_loader(repo, cfg, split)

    build_deepscaler_payload(loader)
    assert seen
    assert all(revision for _repo, revision in seen)
    assert (
        "HuggingFaceH4/MATH-500",
        DATASET_SOURCE_REVISIONS["HuggingFaceH4/MATH-500"],
    ) in seen


def test_dataset_identity_binds_answers_and_pinned_sources():
    practice = [{"id": 1, "problem": "p", "answer": "4"}]
    reserved = [{"id": 2, "problem": "r", "answer": "5"}]
    first = dataset_identity("math500", practice, reserved)
    assert first["identity"]
    assert require_pinned_sources("math500")
    changed = [{"id": 1, "problem": "p", "answer": "6"}]
    assert dataset_identity("math500", changed, reserved)["identity"] != first["identity"]
    assert dataset_source_revisions("deepscaler") == {
        repo: revision
        for repo, revision in DATASET_SOURCE_REVISIONS.items()
        if repo != "HuggingFaceH4/MATH-500"
    }
