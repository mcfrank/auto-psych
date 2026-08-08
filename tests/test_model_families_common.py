"""Regression tests for the model_families/common.py clean-once + lru_cache refactor.

Every public str -> scalar helper in ``model_families/common.py`` was refactored to
clean its input exactly once (via a private already-cleaned helper) and to cache on
the raw input, instead of re-cleaning internally (e.g. ``max_run_norm`` calling
``max_run_length``, which cleaned again). This must be a pure speedup: identical
outputs, identical exceptions, for every H/T string. The reference implementations
below are written independently of the refactored code (not copy-pasted) so this is
a genuine regression check, not a tautology.
"""

from __future__ import annotations

import itertools
import math

import pytest

from src.subjective_randomness.model_families import (
    bayesian_diagnosticity,
    common,
    encoding_compressibility,
    prototype_similarity,
    window_typicality,
)

FAMILIES = (window_typicality, encoding_compressibility, prototype_similarity, bayesian_diagnosticity)


def _all_sequences(max_length: int) -> list[str]:
    seqs = []
    for length in range(1, max_length + 1):
        seqs.extend("".join(bits) for bits in itertools.product("HT", repeat=length))
    return seqs


# --- Independent reference formulas (not sharing code with common.py) -------------


def ref_prop_heads(seq: str) -> float:
    return seq.count("H") / len(seq)


def ref_imbalance(seq: str) -> float:
    return 2.0 * abs(ref_prop_heads(seq) - 0.5)


def ref_n_switches(seq: str) -> int:
    return sum(1 for i in range(len(seq) - 1) if seq[i] != seq[i + 1])


def ref_alternation_rate(seq: str) -> float:
    if len(seq) <= 1:
        return 0.0
    return ref_n_switches(seq) / (len(seq) - 1)


def ref_max_run_length(seq: str) -> int:
    best = cur = 1
    for i in range(1, len(seq)):
        if seq[i] == seq[i - 1]:
            cur += 1
            best = max(best, cur)
        else:
            cur = 1
    return best


def ref_max_run_norm(seq: str) -> float:
    if len(seq) <= 1:
        return 0.0
    return (ref_max_run_length(seq) - 1) / (len(seq) - 1)


def ref_parse_motifs(seq: str) -> tuple[int, int]:
    """Minimal-DP parse by exhaustive partition search: over every way of cutting
    ``seq``, keep the lexicographically smallest (DP cost, chunk count) whose
    chunks are all constant runs (cost 1) or strictly alternating runs of length
    >= 2 (cost 2). Exponential in ``len(seq)``, and shares no code with either
    dynamic-programming implementation."""
    n = len(seq)
    best = None
    for mask in range(1 << (n - 1)):
        cuts = [0] + [i + 1 for i in range(n - 1) if mask >> i & 1] + [n]
        dp = chunks = 0
        for start, end in zip(cuts, cuts[1:]):
            chunk = seq[start:end]
            if all(c == chunk[0] for c in chunk):
                dp += 1
            elif all(a != b for a, b in zip(chunk, chunk[1:])):
                dp += 2
            else:
                break
            chunks += 1
        else:
            best = min(best, (dp, chunks)) if best is not None else (dp, chunks)
    dp, chunks = best
    return 2 * chunks - dp, dp - chunks


def ref_periodicity_score(seq: str) -> float:
    n = len(seq)
    if n <= 2:
        return 0.0
    best_match = 0.5
    for period in range(1, n // 2 + 1):
        template = seq[:period]
        matches = sum(1 for i in range(n) if seq[i] == template[i % period])
        best_match = max(best_match, matches / n)
    return max(0.0, min(1.0, 2.0 * (best_match - 0.5)))


@pytest.mark.parametrize(
    "fn,ref",
    [
        (common.prop_heads, ref_prop_heads),
        (common.imbalance, ref_imbalance),
        (common.n_switches, ref_n_switches),
        (common.alternation_rate, ref_alternation_rate),
        (common.max_run_length, ref_max_run_length),
        (common.max_run_norm, ref_max_run_norm),
        (common.periodicity_score, ref_periodicity_score),
    ],
)
def test_refactored_stats_match_independent_reference(fn, ref):
    """Every stat, refactored to clean once + cache, matches a fresh re-derivation."""
    for seq in _all_sequences(max_length=12):
        assert fn(seq) == ref(seq), f"{fn.__name__}({seq!r})"


def test_parse_motifs_matches_a_brute_force_minimal_partition_reference():
    """Falk & Konold's parse is the DP-minimising partition, ties broken toward the
    fewest chunks -- checked against exhaustive partition search rather than against
    a second copy of the same dynamic program. The shorter length bound than the
    other stats is the price of the reference being exponential."""
    for seq in _all_sequences(max_length=10):
        assert common.parse_motifs(seq) == ref_parse_motifs(seq), seq


def test_clean_sequence_exception_behavior_preserved():
    assert common.clean_sequence("hth ") == "HTH"
    assert common.clean_sequence("h t t") == "HTT"  # internal whitespace is dropped, not rejected
    for bad in ("", "   ", "HTX", "123", "H_T"):
        with pytest.raises(ValueError):
            common.clean_sequence(bad)


def test_lru_cache_does_not_change_scores():
    """Cache-cleared vs warm calls return identical scores for every family."""
    seqs = _all_sequences(max_length=8)
    for module in FAMILIES:
        for name in (
            "clean_sequence",
            "prop_heads",
            "imbalance",
            "n_switches",
            "alternation_rate",
            "max_run_length",
            "max_run_norm",
            "parse_motifs",
            "periodicity_score",
        ):
            getattr(common, name).cache_clear()
        cold = [module.score_sequence(s, module.DEFAULT_PARAMS) for s in seqs]
        warm = [module.score_sequence(s, module.DEFAULT_PARAMS) for s in seqs]
        assert cold == warm, module.__name__


def test_double_clean_paths_are_consistent():
    """Functions whose old implementation double-cleaned (imbalance->prop_heads,
    alternation_rate->n_switches, max_run_norm->max_run_length) still agree with a
    single clean_sequence call on the same input."""
    for seq in _all_sequences(max_length=10):
        cleaned = common.clean_sequence(seq)
        assert common.imbalance(seq) == common.imbalance(cleaned)
        assert common.alternation_rate(seq) == common.alternation_rate(cleaned)
        assert common.max_run_norm(seq) == common.max_run_norm(cleaned)


def test_logsumexp_and_bernoulli_untouched():
    # Sanity: functions not part of the refactor still behave as documented.
    assert math.isclose(common.logsumexp([0.0, 0.0]), math.log(2.0))
    assert common.bernoulli_log_prob(0, 0, 0.5) == 0.0
