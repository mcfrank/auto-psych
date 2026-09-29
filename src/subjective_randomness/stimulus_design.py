"""Candidate pools of H/T sequence pairs.

``generate_candidate_pool`` samples a diverse pool of distinct same-length
pairs; ``enumerate_all_pairs`` returns the whole pair space over a set of
lengths. The outer loop's design, the inner loop's novelty pool and the
holdout evaluation pool are all drawn from these.
"""

from __future__ import annotations

import itertools
from typing import Dict, List, Sequence, Tuple

import numpy as np


def generate_candidate_pool(
    n_pairs: int = 200,
    *,
    lengths: Tuple[int, ...] = (6, 8),
    seed: int = 0,
) -> List[Dict[str, str]]:
    """Sample a diverse pool of candidate stimulus pairs to mine for high EIG.

    For each length in ``lengths`` the full sequence space (``2**length`` H/T
    strings) is enumerated; ``n_pairs`` distinct unordered same-length pairs are
    then sampled across lengths. Full enumeration makes the pool maximally
    varied (every run/alternation/imbalance structure is represented), and
    sampling is deterministic given ``seed``. Lengths are capped at 12 to bound
    enumeration.
    """
    if n_pairs < 1:
        raise ValueError(f"n_pairs must be >= 1, got {n_pairs}.")
    if any(length > 12 for length in lengths):
        raise ValueError("Sequence lengths are capped at 12 to bound enumeration.")

    sequences_by_length = {
        length: ["".join(bits) for bits in itertools.product("HT", repeat=length)]
        for length in lengths
    }
    total_pairs = sum(
        len(seqs) * (len(seqs) - 1) // 2 for seqs in sequences_by_length.values()
    )
    if n_pairs > total_pairs:
        raise ValueError(
            f"Requested {n_pairs} pairs but only {total_pairs} distinct pairs "
            f"exist for lengths {lengths}."
        )

    rng = np.random.default_rng(seed)
    seen: set = set()
    pool: List[Dict[str, str]] = []
    max_pairs_by_length = {
        length: len(seqs) * (len(seqs) - 1) // 2
        for length, seqs in sequences_by_length.items()
    }
    taken_by_length = {length: 0 for length in sequences_by_length}
    lengths_cycle = list(lengths)
    while len(pool) < n_pairs:
        if not lengths_cycle:
            raise RuntimeError(
                "All lengths exhausted before reaching n_pairs; the total-pairs "
                "feasibility check above should have caught this."
            )
        length = lengths_cycle[len(pool) % len(lengths_cycle)]
        # A short length can run out of distinct pairs before its round-robin
        # share is met (length 4 has only 120); once exhausted, every further
        # draw for it would be rejected forever, so hand its remaining slots to
        # the other lengths.
        if taken_by_length[length] == max_pairs_by_length[length]:
            lengths_cycle.remove(length)
            continue
        seqs = sequences_by_length[length]
        i, j = rng.integers(0, len(seqs), size=2)
        if i == j:
            continue
        key = (seqs[i], seqs[j]) if i < j else (seqs[j], seqs[i])
        if key in seen:
            continue
        seen.add(key)
        taken_by_length[length] += 1
        pool.append({"sequence_a": key[0], "sequence_b": key[1]})
    return pool


def enumerate_all_pairs(
    lengths: Sequence[int], *, same_length_only: bool = False
) -> List[Dict[str, str]]:
    """Every distinct unordered H/T pair over all sequences of the given lengths.

    The full ``2**L`` sequence space is enumerated for each length ``L`` in
    ``lengths`` and pooled into one sequence set; every unordered pair of two
    distinct sequences from that pool is emitted in deterministic order. By
    default this includes cross-length pairs; ``same_length_only=True`` filters
    them for paper-anchored models whose scores are only comparable within a
    common length.
    This is the exhaustive counterpart to :func:`generate_candidate_pool`:
    instead of sampling ``n_pairs``, it returns the *whole* pair space over the
    union of the lengths. For lengths ``1..8`` the unfiltered pool is 510
    sequences, so
    ``C(510, 2) = 129,795`` pairs. Duplicate lengths are ignored; lengths are
    capped at 12 to bound enumeration.
    """
    lengths = tuple(sorted(set(lengths)))
    if not lengths:
        raise ValueError("lengths must be non-empty.")
    if any(length < 1 for length in lengths):
        raise ValueError(f"Sequence lengths must be >= 1, got {lengths}.")
    if any(length > 12 for length in lengths):
        raise ValueError("Sequence lengths are capped at 12 to bound enumeration.")

    sequences: List[str] = []
    for length in lengths:
        sequences.extend(
            "".join(bits) for bits in itertools.product("HT", repeat=length)
        )
    return [
        {"sequence_a": seq_a, "sequence_b": seq_b}
        for seq_a, seq_b in itertools.combinations(sequences, 2)
        if not same_length_only or len(seq_a) == len(seq_b)
    ]
