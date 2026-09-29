"""Fast tests for the candidate pools of H/T sequence pairs."""

from __future__ import annotations

import pytest

from src.subjective_randomness.stimulus_design import (
    enumerate_all_pairs,
    generate_candidate_pool,
)


def test_generate_candidate_pool_is_diverse_valid_and_deterministic():
    pool = generate_candidate_pool(n_pairs=40, lengths=(6, 8), seed=1)
    assert len(pool) == 40
    # Every item is a pair of distinct H/T strings of an allowed length.
    for item in pool:
        a, b = item["sequence_a"], item["sequence_b"]
        assert a != b
        assert set(a) <= {"H", "T"} and set(b) <= {"H", "T"}
        assert len(a) == len(b) and len(a) in (6, 8)
    # Pairs are unique, and the same seed reproduces the same pool.
    keys = {(d["sequence_a"], d["sequence_b"]) for d in pool}
    assert len(keys) == 40
    again = generate_candidate_pool(n_pairs=40, lengths=(6, 8), seed=1)
    assert [(d["sequence_a"], d["sequence_b"]) for d in pool] == [
        (d["sequence_a"], d["sequence_b"]) for d in again
    ]


def test_generate_candidate_pool_rejects_oversized_request():
    # Only 2^4 = 16 sequences of length 4 -> C(16,2)=120 distinct pairs.
    with pytest.raises(ValueError, match="distinct pairs"):
        generate_candidate_pool(n_pairs=1000, lengths=(4,), seed=0)


def test_generate_candidate_pool_survives_exhausting_one_length():
    # Length 2 has only C(4, 2) = 6 distinct pairs, but the round-robin over
    # lengths hands it n_pairs/2 = 10 slots. Once its 6 pairs are all taken the
    # sampler must reassign the shortfall to the other lengths instead of
    # rejection-sampling already-seen length-2 pairs forever (the infinite loop
    # that hung the EIG scaling benchmark for over an hour). The alarm turns a
    # regression back into a loud failure instead of a hung test suite.
    import signal

    def _bail(signum, frame):
        raise TimeoutError(
            "generate_candidate_pool did not finish; per-length exhaustion loop?"
        )

    old_handler = signal.signal(signal.SIGALRM, _bail)
    signal.alarm(30)
    try:
        pool = generate_candidate_pool(n_pairs=20, lengths=(2, 8), seed=0)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

    assert len(pool) == 20
    keys = {(d["sequence_a"], d["sequence_b"]) for d in pool}
    assert len(keys) == 20
    for item in pool:
        assert len(item["sequence_a"]) == len(item["sequence_b"])
        assert len(item["sequence_a"]) in (2, 8)
    # The small length is fully used (all 6 pairs) before falling back.
    n_len2 = sum(1 for d in pool if len(d["sequence_a"]) == 2)
    assert n_len2 == 6


def test_enumerate_all_pairs_covers_every_pair_including_cross_length():
    # Lengths 2 and 3 pool to 4 + 8 = 12 sequences, so every unordered pair is
    # C(12, 2) = 66: 6 same-length-2, 28 same-length-3, and 32 cross-length (4*8).
    pool = enumerate_all_pairs([2, 3])
    assert len(pool) == 66
    cross = [p for p in pool if len(p["sequence_a"]) != len(p["sequence_b"])]
    assert len(cross) == 4 * 8
    for item in pool:
        a, b = item["sequence_a"], item["sequence_b"]
        assert a != b
        assert len(a) in (2, 3) and len(b) in (2, 3)
        assert set(a) <= {"H", "T"} and set(b) <= {"H", "T"}
    # Every pair is distinct (unordered), and enumeration is deterministic.
    keys = {frozenset((d["sequence_a"], d["sequence_b"])) for d in pool}
    assert len(keys) == 66
    assert enumerate_all_pairs([2, 3]) == pool


def test_enumerate_all_pairs_caps_length_and_rejects_empty():
    with pytest.raises(ValueError, match="capped at 12"):
        enumerate_all_pairs([13])
    with pytest.raises(ValueError, match="non-empty"):
        enumerate_all_pairs([])
