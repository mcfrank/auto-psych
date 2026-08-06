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
