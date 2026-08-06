"""Tests for src/subjective_randomness/exhaustive_search.py.

Covers deriving a safe quotient from models' own SUFFICIENT_STATS /
COMPLEMENT_INVARIANT declarations, and auditing that quotient against each
model's actual score_sequence rather than trusting the declaration.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.subjective_randomness import exhaustive_search as es
from src.subjective_randomness import sequence_stats as ss
from src.subjective_randomness.model_families import (
    bayesian_diagnosticity,
    encoding_compressibility,
    prototype_similarity,
    window_typicality,
)

FAMILIES = [window_typicality, encoding_compressibility, prototype_similarity, bayesian_diagnosticity]


class _Undeclared:
    """A stand-in for a brand-new inner-loop model with no declarations yet."""


class _ExplicitlyNotComplementInvariant:
    COMPLEMENT_INVARIANT = False


class _LyingPositionReader:
    """Declares a quotient it does not actually respect -- reads seq[0]."""

    MODEL_NAME = "lying_position_reader"
    DEFAULT_PARAMS: dict = {}
    PARAM_BOUNDS: dict = {}
    SUFFICIENT_STATS = ("n", "max_run")
    COMPLEMENT_INVARIANT = False

    @staticmethod
    def score_sequence(seq, params=None):
        return 1.0 if seq[0] == "H" else 0.0


def test_quotient_stat_names_is_the_union_of_declarations():
    names = es.quotient_stat_names(FAMILIES)
    expected = set(window_typicality.SUFFICIENT_STATS)
    expected |= set(encoding_compressibility.SUFFICIENT_STATS)
    expected |= set(prototype_similarity.SUFFICIENT_STATS)
    expected |= set(bayesian_diagnosticity.SUFFICIENT_STATS)
    assert set(names) == expected


def test_quotient_stat_names_falls_back_to_superset_when_any_module_undeclared():
    assert es.quotient_stat_names(FAMILIES + [_Undeclared]) == ss.CANONICAL_STAT_NAMES


def test_quotient_stat_names_falls_back_to_superset_when_empty():
    assert es.quotient_stat_names([]) == ss.CANONICAL_STAT_NAMES


def test_complement_invariant_requires_unanimous_true():
    assert es.complement_invariant(FAMILIES) is True
    assert es.complement_invariant(FAMILIES + [_ExplicitlyNotComplementInvariant]) is False
    assert es.complement_invariant(FAMILIES + [_Undeclared]) is False
    assert es.complement_invariant([]) is False


@pytest.mark.parametrize("complement_canonical", [False, True])
def test_audit_quotient_passes_for_the_real_declared_quotient(complement_canonical):
    stat_names = es.quotient_stat_names(FAMILIES)
    if complement_canonical:
        assert es.complement_invariant(FAMILIES)
    classes = ss.build_sequence_classes(
        range(2, 11), stat_names=stat_names, complement_canonical=complement_canonical
    )
    # Must not raise.
    es.audit_quotient(classes, FAMILIES, n_probe_classes=500, seed=1)


def test_audit_quotient_catches_a_lying_declaration():
    stat_names = ("n", "max_run")
    classes = ss.build_sequence_classes(range(2, 9), stat_names=stat_names)
    with pytest.raises(es.QuotientViolation, match="lying_position_reader"):
        es.audit_quotient(classes, [_LyingPositionReader], n_probe_classes=2000, seed=3)


def test_audit_quotient_is_a_noop_with_no_modules_or_no_multi_member_classes():
    classes = ss.build_sequence_classes(range(2, 9))
    es.audit_quotient(classes, [], n_probe_classes=500)  # no modules -> nothing to check

    # An identity quotient (every canonical stat retained, one sequence per class
    # at these short lengths would still have some multi-member classes in
    # practice, so force the no-multi-member case directly).
    singleton_classes = ss.SequenceClasses(
        representatives=("HT",),
        sizes=np.array([1], dtype=np.int64),
        stats={name: np.array([0]) for name in ss.CANONICAL_STAT_NAMES},
        stat_names=ss.CANONICAL_STAT_NAMES,
        complement_canonical=False,
        n_sequences=1,
    )
    es.audit_quotient(singleton_classes, FAMILIES)  # nothing to sample -> no-op


def test_build_score_table_and_pair_probabilities_match_predict_left_default_params():
    classes = ss.build_sequence_classes(range(2, 9))
    reps = classes.representatives
    table = es.build_score_table(FAMILIES, reps)
    assert table.scores.shape == (len(reps), len(FAMILIES), 1)

    rng = np.random.default_rng(0)
    n = len(reps)
    ii = rng.integers(0, n, size=200)
    jj = rng.integers(0, n, size=200)
    keep = ii != jj
    ii, jj = ii[keep], jj[keep]
    got = es.pair_probabilities(table, ii, jj)

    for row in range(len(ii)):
        stim = {"sequence_a": reps[ii[row]], "sequence_b": reps[jj[row]]}
        for k, module in enumerate(FAMILIES):
            direct = min(max(module.predict_left(stim), 1e-6), 1.0 - 1e-6)
            assert np.isclose(got[row, k], direct, atol=1e-9), (module.MODEL_NAME, stim)


@pytest.mark.parametrize("n_draws", [1, 8, 50])
def test_pair_probabilities_matches_parameter_averaged_predict_left(n_draws):
    classes = ss.build_sequence_classes(range(2, 9))
    reps = classes.representatives
    table = es.build_score_table(FAMILIES, reps, param_samples=n_draws, seed=42)
    assert table.scores.shape[-1] == n_draws

    rng = np.random.default_rng(1)
    n = len(reps)
    ii = rng.integers(0, n, size=100)
    jj = rng.integers(0, n, size=100)
    keep = ii != jj
    ii, jj = ii[keep], jj[keep]
    got = es.pair_probabilities(table, ii, jj)

    for row in range(len(ii)):
        stim = {"sequence_a": reps[ii[row]], "sequence_b": reps[jj[row]]}
        for k, module in enumerate(FAMILIES):
            draws = table.param_draws[module.MODEL_NAME]
            direct = float(np.mean([module.predict_left(stim, d) for d in draws]))
            direct = min(max(direct, 1e-6), 1.0 - 1e-6)
            assert np.isclose(got[row, k], direct, atol=1e-9), (module.MODEL_NAME, stim)


def test_build_score_table_rejects_ragged_draw_counts():
    with pytest.raises(ValueError, match="same number of parameter draws"):
        es.build_score_table(
            FAMILIES,
            ["HT", "TH"],
            param_sets_by_model={
                window_typicality.MODEL_NAME: [dict(window_typicality.DEFAULT_PARAMS)],
                encoding_compressibility.MODEL_NAME: [dict(encoding_compressibility.DEFAULT_PARAMS)] * 2,
                prototype_similarity.MODEL_NAME: [dict(prototype_similarity.DEFAULT_PARAMS)],
                bayesian_diagnosticity.MODEL_NAME: [dict(bayesian_diagnosticity.DEFAULT_PARAMS)],
            },
        )


def test_build_score_table_requires_at_least_one_model():
    with pytest.raises(ValueError, match="at least one model"):
        es.build_score_table([], ["HT"])


@pytest.mark.parametrize("draw_block", [1, 7, 37, 1000])
def test_pair_probabilities_is_draw_block_invariant(draw_block):
    classes = ss.build_sequence_classes(range(2, 9))
    reps = classes.representatives
    table = es.build_score_table(FAMILIES, reps, param_samples=37, seed=9)
    rng = np.random.default_rng(2)
    n = len(reps)
    ii = rng.integers(0, n, size=150)
    jj = rng.integers(0, n, size=150)
    keep = ii != jj
    ii, jj = ii[keep], jj[keep]
    ref = es.pair_probabilities(table, ii, jj, draw_block=1)
    got = es.pair_probabilities(table, ii, jj, draw_block=draw_block)
    assert np.allclose(got, ref, atol=1e-12)


def test_audit_decomposition_passes_for_the_real_families():
    classes = ss.build_sequence_classes(range(2, 9))
    table = es.build_score_table(FAMILIES, classes.representatives, param_samples=20, seed=5)
    es.audit_decomposition(table, FAMILIES, classes.representatives, n_probe_pairs=64, seed=5)


def test_audit_decomposition_catches_a_broken_pair_probabilities():
    classes = ss.build_sequence_classes(range(2, 9))
    table = es.build_score_table(FAMILIES, classes.representatives, param_samples=5, seed=5)
    # Corrupt the beta row for one model to simulate a param-ordering bug.
    broken = es.ScoreTable(
        model_names=table.model_names,
        scores=table.scores,
        beta=table.beta * 0.0,  # wrong: drops all model discrimination
        side_bias=table.side_bias,
        param_draws=table.param_draws,
    )
    with pytest.raises(es.ExhaustiveSearchError):
        es.audit_decomposition(broken, FAMILIES, classes.representatives, n_probe_pairs=64, seed=5)


def test_audit_quotient_samples_new_param_draws_not_just_defaults():
    """A declaration that only happens to be right at DEFAULT_PARAMS but wrong
    elsewhere in PARAM_BOUNDS should still be caught."""

    class _RightAtDefaultsOnly:
        MODEL_NAME = "right_at_defaults_only"
        DEFAULT_PARAMS = {"gain": 0.0}
        PARAM_BOUNDS = {"gain": (0.0, 1.0)}
        SUFFICIENT_STATS = ("n", "max_run")
        COMPLEMENT_INVARIANT = False

        @staticmethod
        def score_sequence(seq, params=None):
            gain = (params or {}).get("gain", 0.0)
            return gain * (1.0 if seq[0] == "H" else 0.0)

    classes = ss.build_sequence_classes(range(2, 9), stat_names=("n", "max_run"))
    with pytest.raises(es.QuotientViolation):
        es.audit_quotient(
            classes, [_RightAtDefaultsOnly], n_probe_classes=2000, n_param_draws=5, seed=3
        )
