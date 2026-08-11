"""Vectorized H/T sequence statistics and an exact feature-equivalence quotient.

``model_families/common.py`` computes the 9-11 sequence statistics one string at a
time; exhaustive stimulus search needs them for every sequence of a given length at
once. This module recomputes the same statistics with numpy over the full ``2**L``
sequence space per length and groups sequences that agree on requested statistics
into one equivalence class. Design-time search may score one representative per
class only when every active model explicitly declares that those statistics are
sufficient; exact-order models instead use :func:`identity_classes`.

Nothing here changes what any model computes; it is purely a faster way to compute
exactly the same 11 statistics the featurizer (``features.py``) and the pure-Python
model families already use, at every sequence at once instead of one string at a
time, plus the bookkeeping to detect which sequences are truly indistinguishable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

# The original 11 per-sequence statistics shared by the featurizer and summary-
# statistic model families, as stem names (no _a/_b suffix). This is not a safe
# quotient for models that inspect exact sequence order.
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


def _motif_stats(bits: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized ``(rep_motifs, alt_motifs)`` of the Falk & Konold minimal-DP
    parse for every row of ``bits`` — the same dynamic program as
    ``common.parse_motifs`` (partition into constant-run chunks costing 1 and
    strictly alternating chunks of length >= 2 costing 2, minimising total cost
    and then chunk count), run over every row at once.
    """
    n_rows, length = bits.shape

    # const_len[:, j] / alt_len[:, j]: length of the longest constant / strictly
    # alternating chunk starting at position j, so "is seq[j:i] a legal chunk?"
    # is one comparison against the span rather than a re-scan of the chunk.
    const_len = np.ones((n_rows, length), dtype=np.int64)
    alt_len = np.ones((n_rows, length), dtype=np.int64)
    for j in range(length - 2, -1, -1):
        same_next = bits[:, j] == bits[:, j + 1]
        const_len[:, j] = np.where(same_next, const_len[:, j + 1] + 1, 1)
        alt_len[:, j] = np.where(same_next, 1, alt_len[:, j + 1] + 1)

    # best[:, i] is the minimal (DP cost, chunk count) over all partitions of the
    # first i symbols, packed as ``cost * scale + chunks`` so that plain integer
    # order is the lexicographic (fewest DP, then fewest chunks) order the parse
    # is defined by. A partition never has more than ``length`` chunks, which is
    # below ``scale``, so the packing is exact.
    scale = length + 1
    unreachable = (2 * length + 1) * scale + length + 1
    best = np.full((n_rows, length + 1), unreachable, dtype=np.int64)
    best[:, 0] = 0
    for i in range(1, length + 1):
        for j in range(i):
            span = i - j
            best[:, i] = np.minimum(
                best[:, i],
                np.where(const_len[:, j] >= span, best[:, j] + scale + 1, unreachable),
            )
            if span >= 2:
                best[:, i] = np.minimum(
                    best[:, i],
                    np.where(
                        alt_len[:, j] >= span, best[:, j] + 2 * scale + 1, unreachable
                    ),
                )

    dp, chunks = np.divmod(best[:, length], scale)
    return 2 * chunks - dp, dp - chunks


def _run_and_motif_stats(
    bits: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized ``(alts, max_run, rep_motifs, alt_motifs)`` for every row of
    ``bits`` (an ``(N, L)`` array). Verified bit-exact against
    ``common.n_switches`` / ``max_run_length`` / ``parse_motifs`` (the minimal-DP
    parse) for every sequence of length 1-14."""
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

    rep_motifs, alt_motifs = _motif_stats(bits)
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

    ``stat_names=None`` uses the full 11-name summary-statistic superset. This is
    safe only for model sets that explicitly declare those statistics sufficient;
    it is not safe for exact-order models. A narrower ``stat_names`` merges more
    aggressively. Callers are responsible for establishing safety (see
    ``exhaustive_search.quotient_stat_names``); use :func:`identity_classes` when
    any model lacks a sufficient-statistics declaration.

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


def identity_classes(lengths: Sequence[int], *, max_length: int = 20) -> SequenceClasses:
    """Every sequence of ``lengths`` is its own singleton class -- the "no
    quotienting" fallback for when a model set's quotient can't be trusted (see
    ``exhaustive_search.audit_quotient``: a declaration that turns out to be
    stale or wrong, or a model that reads something outside
    ``CANONICAL_STAT_NAMES`` entirely, e.g. raw sequence position). Safe by
    construction -- nothing is ever merged -- at the cost of getting none of
    :func:`build_sequence_classes`'s collapse (508 sequences stay 508 classes at
    ``lengths=(2..8)``, not 291). Affordable at the default exhaustive-design
    lengths; ``build_exhaustive_design`` is expected to keep ``lengths`` modest
    when it falls back to this.
    """
    lengths = tuple(sorted(set(lengths)))
    if not lengths:
        raise ValueError("lengths must be non-empty.")
    if any(length < 1 for length in lengths):
        raise ValueError(f"Sequence lengths must be >= 1, got {lengths}.")
    if any(length > max_length for length in lengths):
        raise ValueError(
            f"Sequence lengths are capped at max_length={max_length} (got {lengths})."
        )

    reps: list[str] = []
    stat_cols: Dict[str, list[np.ndarray]] = {name: [] for name in CANONICAL_STAT_NAMES}
    for length in lengths:
        columns, bits = stats_for_length(length)
        reps.extend(_bits_to_strings(bits))
        for name in CANONICAL_STAT_NAMES:
            stat_cols[name].append(columns[name])

    n_total = len(reps)
    return SequenceClasses(
        representatives=tuple(reps),
        sizes=np.ones(n_total, dtype=np.int64),
        stats={name: np.concatenate(cols) for name, cols in stat_cols.items()},
        stat_names=CANONICAL_STAT_NAMES,
        complement_canonical=False,
        n_sequences=n_total,
    )
