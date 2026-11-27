"""Problem-clustered paired statistics (PAPER: Sec. E.9 / Table 5).

Answers within a problem are correlated, so the standard error, confidence
interval, and primary p-value are computed over **one value per problem**. The
donor's exact algorithms are preserved so results remain comparable.
"""

from __future__ import annotations

import math
import random

__all__ = [
    "mcnemar_exact",
    "bootstrap_ci",
    "signflip_p",
    "compute_paired_stats",
]

DEFAULT_SEED = 20260101


def mcnemar_exact(wrong_to_right: int, right_to_wrong: int) -> float:
    """Two-sided exact McNemar p-value via the binomial distribution."""
    n = wrong_to_right + right_to_wrong
    if n == 0:
        return 1.0
    k = min(wrong_to_right, right_to_wrong)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


def bootstrap_ci(
    diffs: list[float], resamples: int = 1000, seed: int = DEFAULT_SEED
) -> tuple[float, float]:
    """Percentile bootstrap CI over **problem-level** differences."""
    if not diffs:
        return 0.0, 0.0
    rng = random.Random(seed)
    n = len(diffs)
    means = []
    for _ in range(resamples):
        total = 0
        for _ in range(n):
            total += diffs[rng.randrange(n)]
        means.append(total / n)
    means.sort()
    lo = means[int(0.025 * (resamples - 1))]
    hi = means[int(0.975 * (resamples - 1))]
    return lo, hi


def signflip_p(
    diffs: list[float],
    resamples: int = 10_000,
    seed: int = DEFAULT_SEED,
    exact_int: bool = False,
) -> tuple[float, float]:
    """One-sided and two-sided paired sign-flip p-values.

    H1 for the primary test is ``mean(d) > 0`` (edited > base). When the diffs are
    integer numerators we use exact integer arithmetic so half-answer ties cannot
    be dropped asymmetrically. The usual ``+1`` correction avoids p=0.
    """
    if not diffs:
        return 1.0, 1.0
    rng = random.Random(seed)
    observed = sum(diffs)
    count_one = 0
    count_two = 0
    if exact_int:
        int_diffs = [int(d) for d in diffs]
        observed = sum(int_diffs)
        for _ in range(resamples):
            total = 0
            for d in int_diffs:
                total += d if rng.getrandbits(1) else -d
            if total >= observed:
                count_one += 1
            if abs(total) >= abs(observed):
                count_two += 1
    else:
        tol = 1e-12
        for _ in range(resamples):
            total = 0.0
            for d in diffs:
                total += d if rng.getrandbits(1) else -d
            if total >= observed - tol:
                count_one += 1
            if abs(total) >= abs(observed) - tol:
                count_two += 1
    denom = resamples + 1
    return (count_one + 1) / denom, (count_two + 1) / denom


def compute_paired_stats(
    base_rows: list[dict],
    edited_rows: list[dict] | None = None,
    bootstrap_resamples: int = 1000,
    signflip_resamples: int = 10_000,
) -> dict:
    """Problem-clustered paired statistics for self-rewrite decisions.

    Required row fields: ``id``, ``correct``; optional ``sample_idx`` (default 0),
    ``cut``, ``loop``. ``edited_rows=None`` returns descriptive base stats only.
    """
    n_answers = len(base_rows)
    if n_answers == 0:
        return {}
    b_ok_n = sum(int(r["correct"]) for r in base_rows)
    b_cut_n = sum(int(r.get("cut", 0)) for r in base_rows)
    b_loop_n = sum(int(r.get("loop", 0)) for r in base_rows)
    base_stats = {
        "n": n_answers,
        "n_answers": n_answers,
        "base_acc": b_ok_n / n_answers,
        "base_cut": b_cut_n,
        "base_loop": b_loop_n,
    }
    if edited_rows is None:
        return base_stats
    if len(edited_rows) != n_answers:
        raise ValueError(
            f"Paired stats require equal lengths, got base={n_answers}, edited={len(edited_rows)}."
        )

    def _by_problem(rows: list[dict], label: str) -> dict:
        grouped: dict = {}
        for row in rows:
            pid = row["id"]
            s_idx = int(row.get("sample_idx", 0))
            samples = grouped.setdefault(pid, {})
            if s_idx in samples:
                raise ValueError(f"Duplicate {label} row for problem {pid}, sample_idx {s_idx}.")
            samples[s_idx] = row
        return grouped

    base_by = _by_problem(base_rows, "base")
    edited_by = _by_problem(edited_rows, "edited")
    if set(base_by) != set(edited_by):
        raise ValueError(
            f"Paired stats problem sets differ: {sorted(set(base_by) ^ set(edited_by))}."
        )

    problem_ids = list(base_by.keys())
    e_ok_n = sum(int(r["correct"]) for r in edited_rows)
    e_cut_n = sum(int(r.get("cut", 0)) for r in edited_rows)
    e_loop_n = sum(int(r.get("loop", 0)) for r in edited_rows)
    w2r = 0
    r2w = 0
    d_numerators: list[int] = []
    d_means: list[float] = []
    samples_per_problem: list[int] = []

    for pid in problem_ids:
        b_samples = base_by[pid]
        e_samples = edited_by[pid]
        if set(b_samples) != set(e_samples):
            raise ValueError(
                f"Paired stats sample sets differ for problem {pid}: "
                f"base={sorted(b_samples)}, edited={sorted(e_samples)}."
            )
        count = len(b_samples)
        samples_per_problem.append(count)
        numerator = 0
        for s_idx in sorted(b_samples):
            b_ok = bool(b_samples[s_idx]["correct"])
            e_ok = bool(e_samples[s_idx]["correct"])
            if not b_ok and e_ok:
                w2r += 1
            if b_ok and not e_ok:
                r2w += 1
            numerator += int(e_ok) - int(b_ok)
        d_numerators.append(numerator)
        d_means.append(numerator / count)

    n_problems = len(problem_ids)
    equal_samples = len(set(samples_per_problem)) == 1
    mean_d = sum(d_means) / n_problems
    if n_problems > 1:
        var_d = sum((d - mean_d) ** 2 for d in d_means) / (n_problems - 1)
        se = math.sqrt(var_d / n_problems)
    else:
        se = 0.0
    ci_lo, ci_hi = bootstrap_ci(d_means, resamples=bootstrap_resamples)
    p_one, p_two = signflip_p(
        d_numerators if equal_samples else d_means,
        resamples=signflip_resamples,
        exact_int=equal_samples,
    )
    mcnemar_p = mcnemar_exact(w2r, r2w) if equal_samples and samples_per_problem[0] == 1 else None

    stats = dict(base_stats)
    stats.update(
        {
            "n_problems": n_problems,
            "n_answers": n_answers,
            "samples_per_problem": (samples_per_problem[0] if equal_samples else "mixed"),
            "edited_acc": e_ok_n / n_answers,
            "delta": mean_d,
            "se": se,
            "ci_low": ci_lo,
            "ci_high": ci_hi,
            "signflip_p_one_sided": p_one,
            "signflip_p_two_sided": p_two,
            "mcnemar_p": mcnemar_p,
            "wrong_to_right": w2r,
            "right_to_wrong": r2w,
            "edited_cut": e_cut_n,
            "edited_loop": e_loop_n,
        }
    )
    return stats
