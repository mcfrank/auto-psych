"""Vectorized H/T sequence statistics and an exact feature-equivalence quotient.

``model_families/common.py`` computes the 9-11 sequence statistics one string at a
time; exhaustive stimulus search needs them for every sequence of a given length at
once. This module recomputes the same statistics with numpy over the full ``2**L``
sequence space per length, and — since every model in the repo reads a sequence only
through these statistics (see ``model_families/common.py`` docstrings) — groups
sequences that agree on every statistic into one equivalence class. Design-time
search only needs to score one representative per class.

Nothing here changes what any model computes; it is purely a faster way to compute
exactly the same 11 statistics the featurizer (``features.py``) and the pure-Python
model families already use, at every sequence at once instead of one string at a
time, plus the bookkeeping to detect which sequences are truly indistinguishable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

# The 11 per-sequence statistics the featurizer (features.py) and every model family
# (model_families/common.py) read, stem names (no _a/_b suffix).
CANONICAL_STAT_NAMES: Tuple[str, ...] = (
    "n",
    "h",
    "p",
    "alts",
    "p_alts",
    "max_run",
    "max_run_norm",
    "imbalance",
    "rep_motifs",
    "alt_motifs",
    "periodicity",
)

# Bit widths for the integer determinants packed into the int64 class key. Each
# field's value is bounded by the sequence length; 6 bits (0-63) covers every length
# this module supports (max_length is capped well below 63 — see
# build_sequence_classes). Order/widths are an implementation detail, not part of
# the public contract.
_KEY_FIELDS = ("h", "alts", "max_run", "rep_motifs", "alt_motifs", "bm")
_KEY_BITS = 6
_KEY_SHIFT = {name: i * _KEY_BITS for i, name in enumerate(_KEY_FIELDS)}


def enumerate_sequences(length: int) -> list[str]:
    """Every H/T string of ``length``, in the same order as
    ``itertools.product("HT", repeat=length)`` (bit 0 -> 'H', bit 1 -> 'T', most
    significant bit first) — verified in tests so cross-checks against
    ``itertools.product`` and ``enumerate_all_pairs`` are exact.
    """
    if length < 1:
        raise ValueError(f"length must be >= 1, got {length}.")
    bits = _bits_for_length(length)
    return _bits_to_strings(bits)


def _bits_for_length(length: int) -> np.ndarray:
    """``(2**length, length)`` uint8 array; 0 -> 'H', 1 -> 'T'. Row ``i`` is the
    binary expansion of ``i`` with the most significant bit in column 0, which is
    exactly ``itertools.product("HT", repeat=length)``'s iteration order."""
    idx = np.arange(2**length, dtype=np.int64)[:, None]
    shifts = np.arange(length - 1, -1, -1, dtype=np.int64)
    return ((idx >> shifts) & 1).astype(np.uint8)


def _bits_to_strings(bits: np.ndarray) -> list[str]:
    chars = np.where(bits == 0, "H", "T")
    return ["".join(row) for row in chars]


def _run_and_motif_stats(
    bits: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized ``(alts, max_run, rep_motifs, alt_motifs)`` for every row of
    ``bits`` (an ``(N, L)`` array). Verified bit-exact against
    ``common.n_switches`` / ``max_run_length`` / ``parse_motifs`` for every
    sequence of length 1-14."""
    n_rows, length = bits.shape
    if length == 1:
        alts = np.zeros(n_rows, dtype=np.int64)
        max_run = np.ones(n_rows, dtype=np.int64)
        rep_motifs = np.ones(n_rows, dtype=np.int64)
        alt_motifs = np.zeros(n_rows, dtype=np.int64)
        return alts, max_run, rep_motifs, alt_motifs

    # b[:, i] is True iff there is a run boundary between position i and i+1.
    b = bits[:, 1:] != bits[:, :-1]
    alts = b.sum(axis=1).astype(np.int64)

    # Longest constant run: L-1 sequential updates (max_run_length's scan, vectorized
    # over rows). `same[:, j]` is True iff positions j, j+1 are equal.
    same = ~b
    cur = np.ones(n_rows, dtype=np.int64)
    best = np.ones(n_rows, dtype=np.int64)
    for j in range(length - 1):
        cur = np.where(same[:, j], cur + 1, 1)
        best = np.maximum(best, cur)
    max_run = best

    # Falk & Konold motifs via a boundary mask. e[:, i] is True iff there is a run
    # boundary immediately before position i (e[:, 0] and e[:, length] are the
    # sequence's own edges, always boundaries). x[:, i] is True iff position i is an
    # isolated (length-1) run: boundaries on both sides. `starts` marks the first
    # position of a maximal run of consecutive singleton runs; `ge2` marks those
    # stretches with >= 2 singleton runs (one alternation motif each, per Falk &
    # Konold's parse — see common.parse_motifs).
    e = np.ones((n_rows, length + 1), dtype=bool)
    e[:, 1:length] = b
    x = e[:, :length] & e[:, 1:]
    xl = np.zeros_like(x)
    xl[:, 1:] = x[:, :-1]
    xr = np.zeros_like(x)
    xr[:, :-1] = x[:, 1:]
    starts = x & ~xl
    ge2 = starts & xr

    n_runs = 1 + alts
    alt_motifs = ge2.sum(axis=1).astype(np.int64)
    rep_motifs = (
        (n_runs - x.sum(axis=1)) + (starts.sum(axis=1) - alt_motifs)
    ).astype(np.int64)
    return alts, max_run, rep_motifs, alt_motifs


def _periodicity_stats(bits: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized ``(periodicity, bm_clipped)``.

    ``bm_clipped`` is the integer determinant used in the class key: the best
    prefix-template match count, clipped below at ``length // 2``. Every match
    count at or below that floor produces ``periodicity == 0.0`` (matching
    ``common.periodicity_score``'s explicit ``n <= 2`` early return and the
    ``best_match = max(0.5, ...)`` floor), so clipping merges every non-periodic
    sequence into one class without losing any distinction ``periodicity_score``
    itself can make. Verified bit-exact against ``common.periodicity_score`` for
    every sequence of length 1-14.
    """
    n_rows, length = bits.shape
    if length <= 2:
        return np.zeros(n_rows), np.zeros(n_rows, dtype=np.int64)

    half = length // 2
    idx = np.arange(length)
    best_match_count = np.zeros(n_rows, dtype=np.int64)
    for period in range(1, half + 1):
        template_idx = idx % period
        matches = (bits == bits[:, template_idx]).sum(axis=1)
        best_match_count = np.maximum(best_match_count, matches)

    bm_clipped = np.maximum(best_match_count, half)
    frac = np.maximum(0.5, best_match_count / length)
    periodicity = np.clip(2.0 * frac - 1.0, 0.0, 1.0)
    return periodicity, bm_clipped.astype(np.int64)


def stats_for_length(length: int) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """All 11 canonical statistics for every sequence of ``length``, vectorized.

    Returns ``(columns, bits)``: ``columns`` maps each name in
    ``CANONICAL_STAT_NAMES`` to a ``(2**length,)`` array (same row order as
    :func:`enumerate_sequences`); ``bits`` is the underlying ``(2**length,
    length)`` array (0='H', 1='T'), returned so callers can derive additional
    determinants (e.g. the complement-canonical head-count fold) without
    re-enumerating.
    """
    if length < 1:
        raise ValueError(f"length must be >= 1, got {length}.")
    bits = _bits_for_length(length)
    n_rows = bits.shape[0]

    h = (length - bits.sum(axis=1)).astype(np.int64)  # bit 0 -> 'H'
    alts, max_run, rep_motifs, alt_motifs = _run_and_motif_stats(bits)
    periodicity, _bm_clipped = _periodicity_stats(bits)

    p = h / length
    p_alts = alts / (length - 1) if length > 1 else np.zeros(n_rows)
    max_run_norm = (
        (max_run - 1) / (length - 1) if length > 1 else np.zeros(n_rows)
    )
    imbalance = 2.0 * np.abs(p - 0.5)

    columns = {
        "n": np.full(n_rows, length, dtype=np.int64),
        "h": h,
        "p": p,
        "alts": alts,
        "p_alts": p_alts,
        "max_run": max_run,
        "max_run_norm": max_run_norm,
        "imbalance": imbalance,
        "rep_motifs": rep_motifs,
        "alt_motifs": alt_motifs,
        "periodicity": periodicity,
    }
    return columns, bits


def _pack_class_key(
    columns: Dict[str, np.ndarray],
    bits: np.ndarray,
    stat_names: Sequence[str],
    *,
    complement_canonical: bool,
) -> np.ndarray:
    """The int64 equivalence-class key for the requested statistics.

    Each of the 11 canonical stats maps to one of six packed integer fields
    (``_KEY_FIELDS``); several stats share a field because one determines the
    other exactly given a fixed length (``p`` <-> ``h``, ``p_alts`` <-> ``alts``,
    ``max_run_norm`` <-> ``max_run``, ``periodicity`` <-> the clipped best-match
    count). ``n`` is not packed: this function is always called on the sequences
    of one fixed length at a time, so it is already constant within one key array.

    ``h``/``p``/``imbalance`` are the one case that is not a straightforward
    1-1 rename: ``imbalance = 2*|h/n - 0.5|`` is an exact function of
    ``min(h, n-h)`` (proof: substituting ``h`` or ``n-h`` gives the identical
    value), so a request for ``imbalance`` alone only ever needs that folded
    value, not raw ``h`` — using raw ``h`` would only over-split classes
    ``imbalance`` truly cannot distinguish. A request for raw ``h``/``p``
    needs the *finer* raw value UNLESS ``complement_canonical`` is set, in
    which case the caller has already established (via
    ``exhaustive_search.complement_invariant`` — every declared model
    unanimously invariant under H<->T complementation) that folding is safe:
    every one of these models' scores is provably identical for ``h`` and
    ``n - h``, so merging on the folded value drops no real distinction.
    """
    stat_names = set(stat_names)
    key = np.zeros(bits.shape[0], dtype=np.int64)

    wants_h_raw = "h" in stat_names or "p" in stat_names
    wants_imbalance = "imbalance" in stat_names
    if wants_h_raw or wants_imbalance:
        if complement_canonical or (wants_imbalance and not wants_h_raw):
            length = bits.shape[1]
            h_field = np.minimum(columns["h"], length - columns["h"])
        else:
            h_field = columns["h"]
        key |= h_field << _KEY_SHIFT["h"]

    if "alts" in stat_names or "p_alts" in stat_names:
        key |= columns["alts"] << _KEY_SHIFT["alts"]

    if "max_run" in stat_names or "max_run_norm" in stat_names:
        key |= columns["max_run"] << _KEY_SHIFT["max_run"]

    if "rep_motifs" in stat_names:
        key |= columns["rep_motifs"] << _KEY_SHIFT["rep_motifs"]

    if "alt_motifs" in stat_names:
        key |= columns["alt_motifs"] << _KEY_SHIFT["alt_motifs"]

    if "periodicity" in stat_names:
        _, bm_clipped = _periodicity_stats(bits)
        key |= bm_clipped << _KEY_SHIFT["bm"]

    return key


@dataclass(frozen=True)
class SequenceClasses:
    """One row per feature-equivalence class, pooled across ``lengths``."""

    representatives: Tuple[str, ...]
    sizes: np.ndarray  # (C,) int64 — number of sequences merged into this class
    stats: Dict[str, np.ndarray]  # each (C,) — the representative's full 11 stats
    stat_names: Tuple[str, ...]  # the subset that actually determined the quotient
    complement_canonical: bool
    n_sequences: int  # total sequences enumerated across all lengths, pre-quotient

    @property
    def n_classes(self) -> int:
        return len(self.representatives)


def build_sequence_classes(
    lengths: Sequence[int],
    *,
    stat_names: Optional[Sequence[str]] = None,
    complement_canonical: bool = False,
    max_length: int = 20,
    seed: int = 0,
) -> SequenceClasses:
    """Enumerate every sequence of each length in ``lengths`` and quotient by the
    requested statistics, one representative per equivalence class.

    ``stat_names=None`` uses the full 11-name superset — the finest, always-safe
    quotient: it merges two sequences only when every statistic every model family
    in this repo can read is identical between them (see
    ``model_families/common.py``, which established that no model reads anything
    beyond ``CANONICAL_STAT_NAMES``). A narrower ``stat_names`` (e.g. only the
    statistics one specific model set declares reading) merges more aggressively;
    callers are responsible for establishing that narrowing is safe for the models
    they intend to score (see ``exhaustive_search.quotient_stat_names``, which
    derives a safe subset from models' own declarations rather than hardcoding one
    here).

    Quotienting happens independently *within* each length — sequences of
    different lengths are never merged, even if some narrow ``stat_names`` request
    would technically permit it (e.g. two same-imbalance sequences of different
    lengths). This is deliberately conservative: the space this module targets is
    dominated by intra-length redundancy (e.g. 508 length-2..8 sequences collapse
    to 291 classes), and merging across lengths risks conflating "same feature
    value" with "same feature value that happens to coincide across lengths" for
    no real efficiency gain (`n` is always part of what the class key would need to
    distinguish anyway, once you consider the whole model set).

    The representative for each class is chosen uniformly at random (seeded, so
    deterministic given ``seed``) rather than lexicographically first, so the
    surface form of the generated design isn't systematically biased toward
    "starts with H".
    """
    lengths = tuple(sorted(set(lengths)))
    if not lengths:
        raise ValueError("lengths must be non-empty.")
    if any(length < 1 for length in lengths):
        raise ValueError(f"Sequence lengths must be >= 1, got {lengths}.")
    if any(length > max_length for length in lengths):
        raise ValueError(
            f"Sequence lengths are capped at max_length={max_length} "
            f"(got {lengths}); raise max_length to enumerate longer sequences "
            "(2**length grows fast — see stimulus_design's projected cost table)."
        )
    names = tuple(stat_names) if stat_names is not None else CANONICAL_STAT_NAMES
    unknown = set(names) - set(CANONICAL_STAT_NAMES)
    if unknown:
        raise ValueError(f"Unknown stat name(s) {sorted(unknown)}; expected one of {CANONICAL_STAT_NAMES}.")

    rng = np.random.default_rng(seed)
    reps: list[str] = []
    sizes: list[np.ndarray] = []
    stat_cols: Dict[str, list[np.ndarray]] = {name: [] for name in CANONICAL_STAT_NAMES}
    n_sequences = 0

    for length in lengths:
        columns, bits = stats_for_length(length)
        n_sequences += bits.shape[0]
        key = _pack_class_key(
            columns, bits, names, complement_canonical=complement_canonical
        )

        r = rng.random(bits.shape[0])
        order = np.lexsort((r, key))  # sort by key asc, then by r asc (random tiebreak)
        sorted_key = key[order]
        group_start = np.concatenate(([True], sorted_key[1:] != sorted_key[:-1]))
        rep_positions = order[group_start]

        uniq_keys, counts = np.unique(key, return_counts=True)
        # np.unique's ascending key order matches sorted_key[group_start]'s order by
        # construction (both are the sorted distinct keys), so rep_positions aligns
        # 1:1 with counts.
        assert np.array_equal(sorted_key[group_start], uniq_keys)

        strings = np.where(bits[rep_positions] == 0, "H", "T")
        reps.extend("".join(row) for row in strings)
        sizes.append(counts.astype(np.int64))
        for name in CANONICAL_STAT_NAMES:
            stat_cols[name].append(columns[name][rep_positions])

    return SequenceClasses(
        representatives=tuple(reps),
        sizes=np.concatenate(sizes),
        stats={name: np.concatenate(cols) for name, cols in stat_cols.items()},
        stat_names=names,
        complement_canonical=complement_canonical,
        n_sequences=n_sequences,
    )
