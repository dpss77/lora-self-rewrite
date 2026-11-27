import pytest

from lora_self_rewrite.stats import (
    bootstrap_ci,
    compute_paired_stats,
    mcnemar_exact,
    signflip_p,
)

pytestmark = pytest.mark.unit


def test_mcnemar_symmetric_is_one():
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(5, 5) == pytest.approx(1.0)


def test_mcnemar_known_asymmetry_is_smaller():
    assert mcnemar_exact(0, 5) == mcnemar_exact(5, 0)
    assert mcnemar_exact(0, 5) < 0.1


def test_bootstrap_ci_deterministic_and_ordered():
    diffs = [1.0, -1.0, 2.0, 0.0, 1.0, -2.0]
    a = bootstrap_ci(diffs, resamples=200, seed=7)
    b = bootstrap_ci(diffs, resamples=200, seed=7)
    assert a == b
    assert a[0] <= a[1]
    assert bootstrap_ci([], resamples=10) == (0.0, 0.0)


def test_signflip_deterministic_and_bounded():
    diffs = [1, 1, 0, 1, -1, 1]
    a = signflip_p(diffs, resamples=500, seed=3, exact_int=True)
    b = signflip_p(diffs, resamples=500, seed=3, exact_int=True)
    assert a == b
    assert 0.0 < a[0] <= 1.0 and 0.0 < a[1] <= 1.0


def _rows(pid, base_values, edited_values):
    base = [
        {"id": pid, "sample_idx": s, "correct": b, "cut": 1, "loop": 0}
        for s, b in enumerate(base_values)
    ]
    edited = [
        {"id": pid, "sample_idx": s, "correct": e, "cut": 1, "loop": 0}
        for s, e in enumerate(edited_values)
    ]
    return base, edited


def test_compute_paired_stats_clusters_by_problem():
    rows = [
        _rows(1, [True, True], [True, True]),
        _rows(2, [False, False], [True, True]),
        _rows(3, [False, False], [False, False]),
    ]
    base = [r for pair in rows for r in pair[0]]
    edited = [r for pair in rows for r in pair[1]]
    stats = compute_paired_stats(base, edited, bootstrap_resamples=100, signflip_resamples=200)
    assert stats["n_problems"] == 3
    assert stats["n_answers"] == 6
    assert stats["wrong_to_right"] == 2
    assert stats["right_to_wrong"] == 0
    assert stats["samples_per_problem"] == 2
    assert stats["delta"] == pytest.approx(1 / 3)


def test_paired_stats_rejects_unequal_lengths():
    with pytest.raises(ValueError):
        compute_paired_stats(
            [{"id": 1, "correct": True, "cut": 0, "loop": 0}],
            [],
        )


def test_paired_stats_rejects_duplicate_sample_idx():
    rows = [
        {"id": 1, "sample_idx": 0, "correct": True, "cut": 0, "loop": 0},
        {"id": 1, "sample_idx": 0, "correct": True, "cut": 0, "loop": 0},
    ]
    with pytest.raises(ValueError):
        compute_paired_stats(rows, rows)
