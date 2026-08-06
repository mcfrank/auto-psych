"""Correctness tests for src/subjective_randomness/sequence_stats.py.

sequence_stats.py recomputes model_families/common.py's 9 sequence statistics
(plus 2 featurizer-only redundant ones) with numpy over an entire length's
sequence space at once, then groups sequences into feature-equivalence classes.
The bar for every test here is exactness against the existing scalar path, not
approximation: any two sequences merged into one class must be provably
indistinguishable to every model that reads only the declared statistics.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from src.subjective_randomness import sequence_stats as ss
from src.subjective_randomness.model_families import (
    bayesian_diagnosticity,
    common,
    encoding_compressibility,
    prototype_similarity,
    window_typicality,
)

# The four live model families' declared sufficient statistics (mirrors the table
# landed with SUFFICIENT_STATS constants in a later change; hardcoded here so this
# test does not depend on that change landing first).
FAMILY_DECLARATIONS = {
    window_typicality: ("n", "max_run"),
    encoding_compressibility: ("max_run_norm", "periodicity", "imbalance"),
    prototype_similarity: ("imbalance", "p_alts"),
    bayesian_diagnosticity: ("n", "h", "rep_motifs", "alt_motifs"),
}


def _scalar_reference(seq: str) -> dict:
    rep_motifs, alt_motifs = common.parse_motifs(seq)
    return dict(
        n=len(seq),
        h=seq.count("H"),
        p=seq.count("H") / len(seq),
        alts=common.n_switches(seq),
        p_alts=common.alternation_rate(seq),
        max_run=common.max_run_length(seq),
        max_run_norm=common.max_run_norm(seq),
        imbalance=common.imbalance(seq),
        rep_motifs=rep_motifs,
        alt_motifs=alt_motifs,
        periodicity=common.periodicity_score(seq),
    )


@pytest.mark.parametrize("length", range(1, 13))
def test_enumerate_sequences_matches_itertools_product_order(length):
    assert ss.enumerate_sequences(length) == [
        "".join(bits) for bits in itertools.product("HT", repeat=length)
    ]


@pytest.mark.parametrize("length", range(1, 13))
def test_vectorized_stats_are_bit_exact_vs_scalar_common(length):
    columns, _bits = ss.stats_for_length(length)
    seqs = ss.enumerate_sequences(length)
    for i, seq in enumerate(seqs):
        ref = _scalar_reference(seq)
        for name, want in ref.items():
            got = columns[name][i]
            if isinstance(want, float):
                assert np.isclose(got, want, atol=1e-12), f"{name}({seq!r}): {got} != {want}"
            else:
                assert got == want, f"{name}({seq!r}): {got} != {want}"


@pytest.mark.slow
@pytest.mark.parametrize("length", (13, 14))
def test_vectorized_stats_are_bit_exact_vs_scalar_common_slow(length):
    columns, _bits = ss.stats_for_length(length)
    seqs = ss.enumerate_sequences(length)
    for i, seq in enumerate(seqs):
        ref = _scalar_reference(seq)
        for name, want in ref.items():
            got = columns[name][i]
            if isinstance(want, float):
                assert np.isclose(got, want, atol=1e-12)
            else:
                assert got == want


@pytest.mark.parametrize(
    "max_length,expected_classes",
    [(8, 291), (10, 934), (12, 2696), (14, 7221), (16, 17545)],
)
def test_full_superset_class_counts_are_golden(max_length, expected_classes):
    classes = ss.build_sequence_classes(range(2, max_length + 1))
    assert classes.n_classes == expected_classes
    assert classes.sizes.sum() == classes.n_sequences
    assert len(classes.representatives) == classes.n_classes == len(classes.sizes)


def test_representatives_are_real_sequences_of_the_right_length():
    classes = ss.build_sequence_classes((3, 4, 5))
    for rep, n in zip(classes.representatives, classes.stats["n"]):
        assert len(rep) == n
        assert set(rep) <= {"H", "T"}


@pytest.mark.parametrize("module,declared_stats", FAMILY_DECLARATIONS.items())
def test_declared_quotient_never_merges_distinguishable_sequences(module, declared_stats):
    """Every pair of sequences sharing a class key (built from one family's
    declared SUFFICIENT_STATS) must score identically under that family — the
    correctness bar for narrowing the quotient below the full 11-stat superset."""
    max_length = 10
    for length in range(2, max_length + 1):
        columns, bits = ss.stats_for_length(length)
        seqs = ss.enumerate_sequences(length)
        key = ss._pack_class_key(columns, bits, declared_stats, complement_canonical=False)
        groups: dict[int, list[str]] = {}
        for k, seq in zip(key, seqs):
            groups.setdefault(int(k), []).append(seq)
        checked_a_multi_member_group = False
        for members in groups.values():
            if len(members) < 2:
                continue
            checked_a_multi_member_group = True
            scores = [module.score_sequence(s, module.DEFAULT_PARAMS) for s in members]
            assert np.allclose(scores, scores[0], atol=1e-9), (module.__name__, members)
        if length >= 6:
            # Redundancy should show up well before length 6 for every declared family.
            assert checked_a_multi_member_group, (module.__name__, length)


@pytest.mark.parametrize("module,declared_stats", FAMILY_DECLARATIONS.items())
def test_complement_canonical_never_merges_distinguishable_sequences(module, declared_stats):
    """All four live families are complement-invariant; complement_canonical=True
    additionally folds h/p into min(h, n-h). Every merged group must still score
    identically -- this is the correctness bar the h_canon substitution exists to
    satisfy (a naive 'just drop h' would over-merge for families like
    bayesian_diagnosticity that read raw h)."""
    max_length = 10
    for length in range(2, max_length + 1):
        columns, bits = ss.stats_for_length(length)
        seqs = ss.enumerate_sequences(length)
        key = ss._pack_class_key(columns, bits, declared_stats, complement_canonical=True)
        groups: dict[int, list[str]] = {}
        for k, seq in zip(key, seqs):
            groups.setdefault(int(k), []).append(seq)
        for members in groups.values():
            if len(members) < 2:
                continue
            scores = [module.score_sequence(s, module.DEFAULT_PARAMS) for s in members]
            assert np.allclose(scores, scores[0], atol=1e-9), (module.__name__, members)


def test_complement_canonical_merges_exact_complement_pairs():
    comp = str.maketrans("HT", "TH")
    classes_plain = ss.build_sequence_classes((8,), stat_names=("n", "h", "rep_motifs", "alt_motifs"))
    classes_comp = ss.build_sequence_classes((8,), stat_names=("n", "h", "rep_motifs", "alt_motifs"), complement_canonical=True)
    assert classes_comp.n_classes < classes_plain.n_classes
    # "TH"-vs-"HT"-style complement pairs collapse: e.g. all-heads and all-tails.
    columns, bits = ss.stats_for_length(8)
    seqs = ss.enumerate_sequences(8)
    key = ss._pack_class_key(columns, bits, ("n", "h", "rep_motifs", "alt_motifs"), complement_canonical=True)
    key_of = dict(zip(seqs, key))
    assert key_of["H" * 8] == key_of["T" * 8]


def test_quotienting_never_merges_across_lengths():
    """Even a narrow stat_names request must not merge sequences of different
    lengths -- classes are always computed per-length (see build_sequence_classes
    docstring)."""
    classes = ss.build_sequence_classes((4, 8), stat_names=("imbalance",))
    lengths_seen = set(int(n) for n in classes.stats["n"])
    assert lengths_seen == {4, 8}
    # A length-4 rep and a length-8 rep never share a class even if they'd tie on
    # imbalance alone.
    assert len(classes.representatives) == len(set(classes.representatives))


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"lengths": ()}, "non-empty"),
        ({"lengths": (0,)}, ">= 1"),
        ({"lengths": (25,)}, "max_length"),
        ({"lengths": (4,), "stat_names": ("not_a_stat",)}, "Unknown stat name"),
    ],
)
def test_build_sequence_classes_rejects_bad_input(kwargs, match):
    with pytest.raises(ValueError, match=match):
        ss.build_sequence_classes(**kwargs)


def test_build_sequence_classes_is_deterministic_given_seed():
    a = ss.build_sequence_classes((4, 5, 6), seed=7)
    b = ss.build_sequence_classes((4, 5, 6), seed=7)
    assert a.representatives == b.representatives
    assert np.array_equal(a.sizes, b.sizes)


def test_representative_choice_is_not_always_lexicographically_first():
    """Guards against silently reverting to 'first member wins', which would
    systematically bias every representative toward starting with H."""
    classes = ss.build_sequence_classes((8,), seed=0)
    multi_member = [
        rep for rep, size in zip(classes.representatives, classes.sizes) if size > 3
    ]
    assert multi_member, "expected at least one class with >3 members at length 8"
    assert not all(rep.startswith("H") for rep in multi_member)
