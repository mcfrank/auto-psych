"""Fast exhaustive stimulus search over the pure-Python model families.

Builds on ``sequence_stats.py``'s feature-equivalence quotient: this module
decides, for a given model set, *how much* quotienting is safe (derived from each
model's own declarations, never hardcoded), and audits that decision against the
model's actual ``score_sequence`` rather than trusting the declaration blindly.
``build_exhaustive_design`` (in ``stimulus_design.py``) wires this together with the
streamed pair scan and the greedy selection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, Mapping, Optional, Sequence, Tuple

import numpy as np

from . import sequence_stats
from .model_families.common import merge_params


class ExhaustiveSearchError(RuntimeError):
    """Base class for exhaustive-search-specific failures."""


class QuotientViolation(ExhaustiveSearchError):
    """A model in the set assigns different scores to two sequences the
    feature-equivalence quotient (or the complement canonicalization) merged into
    one class."""


def quotient_stat_names(modules: Sequence[Any]) -> Tuple[str, ...]:
    """The statistics safe to quotient on for this exact model set.

    Returns the **union** of every module's declared ``SUFFICIENT_STATS``. If
    *any* module lacks a declaration (e.g. a model the inner loop invented this
    round, with no time to have been audited), falls back to the full
    ``sequence_stats.CANONICAL_STAT_NAMES`` superset — the finest quotient the
    featurizer and every PyMC seed model can express, so an undeclared model
    never gets over-merged by omission. An empty model set also falls back to the
    superset (there is nothing to derive a narrower quotient from).
    """
    if not modules:
        return sequence_stats.CANONICAL_STAT_NAMES
    names: set[str] = set()
    for module in modules:
        declared = getattr(module, "SUFFICIENT_STATS", None)
        if declared is None:
            return sequence_stats.CANONICAL_STAT_NAMES
        names.update(declared)
    return tuple(sorted(names)) if names else sequence_stats.CANONICAL_STAT_NAMES


def complement_invariant(modules: Sequence[Any]) -> bool:
    """Whether it is safe to additionally fold H<->T complement pairs.

    Requires **unanimous** ``COMPLEMENT_INVARIANT = True`` across every module —
    an undeclared module, an explicit ``False``, or an empty model set all default
    to ``False``. This must never be hardcoded True: the archived hero-run seed
    ``minkowski_accumulated_typicality`` has a free ``ideal_p != 0.5`` and is
    genuinely complement-*sensitive*, so a model set that includes such a model
    must not fold complements.
    """
    if not modules:
        return False
    return all(getattr(module, "COMPLEMENT_INVARIANT", False) is True for module in modules)


def _sample_param_draws(
    module: Any, *, n_draws: int, rng: np.random.Generator
) -> list[Dict[str, float]]:
    draws = [dict(module.DEFAULT_PARAMS)]
    bounds: Mapping[str, tuple] = getattr(module, "PARAM_BOUNDS", {})
    for _ in range(n_draws):
        draws.append({name: float(rng.uniform(lo, hi)) for name, (lo, hi) in bounds.items()})
    return draws


def audit_quotient(
    classes: sequence_stats.SequenceClasses,
    modules: Sequence[Any],
    *,
    n_probe_classes: int = 256,
    n_param_draws: int = 3,
    seed: int = 0,
    atol: float = 1e-9,
) -> None:
    """Raise :class:`QuotientViolation` if any module actually distinguishes two
    sequences the quotient merged.

    A ``SUFFICIENT_STATS``/``COMPLEMENT_INVARIANT`` declaration can be stale (the
    model was edited after being declared) or simply wrong, and
    :func:`quotient_stat_names` cannot see that from the declaration alone — this
    audit is the independent check that catches it, by actually calling
    ``score_sequence`` on sampled class members instead of trusting the claim.
    Sampling (rather than exhaustively checking every class) keeps this cheap
    enough to run on every production design at the default ``lengths=(2..8)``;
    it recomputes a length's full statistics at most once per distinct length any
    sampled class belongs to (cached), so cost scales with how many *distinct*
    lengths are touched, not with pool size.
    """
    multi_idx = np.flatnonzero(classes.sizes > 1)
    if multi_idx.size == 0 or not modules:
        return

    rng = np.random.default_rng(seed)
    n_probe = min(n_probe_classes, multi_idx.size)
    probe_idx = rng.choice(multi_idx, size=n_probe, replace=False)

    param_draws_by_module = {
        module: _sample_param_draws(module, n_draws=n_param_draws, rng=rng)
        for module in modules
    }

    # (by_key, seq_to_key) per length, built lazily and cached across probes that
    # land on the same length.
    length_cache: Dict[int, tuple[Dict[int, list], Dict[str, int]]] = {}

    def _lookup(length: int) -> tuple[Dict[int, list], Dict[str, int]]:
        if length not in length_cache:
            columns, bits = sequence_stats.stats_for_length(length)
            seqs = sequence_stats.enumerate_sequences(length)
            key = sequence_stats._pack_class_key(
                columns,
                bits,
                classes.stat_names,
                complement_canonical=classes.complement_canonical,
            )
            by_key: Dict[int, list] = {}
            seq_to_key: Dict[str, int] = {}
            for k, s in zip(key, seqs):
                ik = int(k)
                by_key.setdefault(ik, []).append(s)
                seq_to_key[s] = ik
            length_cache[length] = (by_key, seq_to_key)
        return length_cache[length]

    for idx in probe_idx:
        length = int(classes.stats["n"][idx])
        rep = classes.representatives[idx]
        by_key, seq_to_key = _lookup(length)
        members = by_key[seq_to_key[rep]]
        probe_members = (
            members
            if len(members) <= 8
            else [members[i] for i in rng.choice(len(members), size=8, replace=False)]
        )

        for module, draws in param_draws_by_module.items():
            for params in draws:
                scores = [module.score_sequence(s, params) for s in probe_members]
                if not np.allclose(scores, scores[0], atol=atol):
                    bad = next(
                        s
                        for s in probe_members[1:]
                        if not np.isclose(scores[0], module.score_sequence(s, params), atol=atol)
                    )
                    name = getattr(module, "MODEL_NAME", getattr(module, "__name__", repr(module)))
                    raise QuotientViolation(
                        f"{name!r} assigns different scores to {probe_members[0]!r} and "
                        f"{bad!r}, which the quotient (stat_names={classes.stat_names}, "
                        f"complement_canonical={classes.complement_canonical}) treats as "
                        "equivalent. Its SUFFICIENT_STATS/COMPLEMENT_INVARIANT declaration "
                        "is stale or wrong for this model."
                    )


def _model_name(module: Any) -> str:
    return getattr(module, "MODEL_NAME", getattr(module, "__name__", repr(module)))


@dataclass(frozen=True)
class ScoreTable:
    """Every model's per-sequence score, decomposed from any pair's probability.

    Every model in this repo computes ``p_left = sigmoid(beta*(score(seq_a) -
    score(seq_b)) + side_bias)`` (``model_families/common.py:choice_probability``).
    Because ``score_sequence`` never reads ``beta``/``side_bias`` -- only
    ``choice_probability`` does -- those two can be pulled out of the score and
    applied at pair-query time instead of at scoring time, which is what makes
    :func:`pair_probabilities` a numpy gather instead of a `predict_left` call per
    pair.
    """

    model_names: Tuple[str, ...]
    scores: np.ndarray  # (C, K, D) float64 -- score_sequence per class x model x draw
    beta: np.ndarray  # (K, D) float64
    side_bias: np.ndarray  # (K, D) float64
    param_draws: Dict[str, Tuple[Dict[str, float], ...]]  # per-model raw draws, length D each; kept for audit_decomposition


def build_score_table(
    modules: Sequence[Any],
    sequences: Sequence[str],
    *,
    param_sets_by_model: Optional[Mapping[str, Sequence[Mapping[str, float]]]] = None,
    param_samples: Optional[int] = None,
    seed: int = 0,
) -> ScoreTable:
    """Score every sequence in ``sequences`` under every model in ``modules``, once
    per parameter draw.

    Draw source, in priority order (mirrors ``stimulus_design.family_predict_fns``):

    - ``param_sets_by_model``: explicit per-model parameter draws (e.g. posterior
      draws from a previous experiment's fit).
    - ``param_samples=N``: ``N`` prior draws per model from its own
      ``PARAM_BOUNDS``, generated by the *existing*
      ``stimulus_design._prior_param_sets`` -- reused rather than reimplemented so
      this stays bit-identical to today's prior-predictive path, its known
      shared-RNG-stream quirk included (fixed separately, later, as its own
      attributable change).
    - neither: one draw, each model's own ``DEFAULT_PARAMS`` (a point score).

    Every model must end up with the same number of draws ``D`` -- there is no
    ragged-array support here, by design; every existing caller already requests
    the same ``n_draws`` for every model.
    """
    if not modules:
        raise ValueError("Need at least one model to build a score table.")
    model_names = tuple(_model_name(m) for m in modules)

    draws_by_name: Dict[str, list] = {}
    for module, name in zip(modules, model_names):
        if param_sets_by_model is not None:
            draws_by_name[name] = [dict(d) for d in param_sets_by_model[name]]
        elif param_samples is not None:
            from .stimulus_design import _prior_param_sets  # local import: avoids a module cycle

            draws_by_name[name] = _prior_param_sets(module, param_samples, seed)
        else:
            draws_by_name[name] = [dict(module.DEFAULT_PARAMS)]

    d_counts = {len(v) for v in draws_by_name.values()}
    if len(d_counts) != 1:
        raise ValueError(
            "All models must have the same number of parameter draws; got "
            f"{ {n: len(v) for n, v in draws_by_name.items()} }"
        )
    n_draws = d_counts.pop()
    n_seqs = len(sequences)
    n_models = len(modules)

    scores = np.empty((n_seqs, n_models, n_draws), dtype=np.float64)
    beta = np.empty((n_models, n_draws), dtype=np.float64)
    side_bias = np.empty((n_models, n_draws), dtype=np.float64)

    for k, (module, name) in enumerate(zip(modules, model_names)):
        draws = draws_by_name[name]
        for d, raw_params in enumerate(draws):
            # Mirrors choice_probability's own defaulting -- see predict_left,
            # which merges DEFAULT_PARAMS the same way before extracting beta/bias.
            merged = merge_params(module.DEFAULT_PARAMS, raw_params)
            beta[k, d] = float(merged.get("beta", 1.0))
            side_bias[k, d] = float(merged.get("side_bias", 0.0))
            for c, seq in enumerate(sequences):
                scores[c, k, d] = module.score_sequence(seq, raw_params)

    return ScoreTable(
        model_names=model_names,
        scores=scores,
        beta=beta,
        side_bias=side_bias,
        param_draws={name: tuple(draws_by_name[name]) for name in model_names},
    )


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Vectorized twin of common.sigmoid's numerically stable branch."""
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    neg = ~pos
    ez = np.exp(z[neg])
    out[neg] = ez / (1.0 + ez)
    return out


_TARGET_ELEMENTS = 4_000_000  # ~32MB of float64 per temporary at (B, K, draw_block)


def pair_probabilities(
    table: ScoreTable,
    i: np.ndarray,
    j: np.ndarray,
    *,
    draw_block: Optional[int] = None,
) -> np.ndarray:
    """``(B, K)`` parameter-averaged ``P(choose left)`` for pairs
    ``(i[b], j[b])`` -- the exact twin of averaging
    ``module.predict_left({"sequence_a": seq[i], "sequence_b": seq[j]}, draw)``
    over every draw, for every model, without ever calling ``predict_left``.
    Clipped to ``[1e-6, 1-1e-6]``, matching ``stimulus_design._predict_matrix``.

    Blocked over the draw axis so peak memory is bounded regardless of how many
    draws ``table`` holds: the ``(B, K, draw_block)`` temporary is capped near
    ``_TARGET_ELEMENTS`` elements by default.
    """
    i = np.asarray(i)
    j = np.asarray(j)
    n_models, n_draws = table.beta.shape
    b = i.shape[0]
    if draw_block is None:
        draw_block = max(1, min(n_draws, _TARGET_ELEMENTS // max(1, b * n_models)))

    acc = np.zeros((b, n_models), dtype=np.float64)
    for d0 in range(0, n_draws, draw_block):
        d1 = min(d0 + draw_block, n_draws)
        diff = table.scores[i, :, d0:d1] - table.scores[j, :, d0:d1]  # (B, K, Db)
        z = table.beta[None, :, d0:d1] * diff + table.side_bias[None, :, d0:d1]
        acc += _sigmoid(z).sum(axis=2)
    p = acc / n_draws
    return np.clip(p, 1e-6, 1.0 - 1e-6)


def audit_decomposition(
    table: ScoreTable,
    modules: Sequence[Any],
    sequences: Sequence[str],
    *,
    n_probe_pairs: int = 64,
    seed: int = 0,
    atol: float = 1e-9,
) -> None:
    """Cross-check a handful of pairs' :func:`pair_probabilities` output against
    directly calling each module's ``predict_left`` with the same draws stored on
    ``table``. Cheap (a few dozen ``predict_left`` calls) and meant to run on every
    production design, not just in tests -- it catches a param-ordering or
    averaging bug the unit tests happened not to exercise.
    """
    n_seqs = len(sequences)
    if n_seqs < 2:
        return
    rng = np.random.default_rng(seed)
    max_pairs = n_seqs * (n_seqs - 1) // 2
    n_probe = min(n_probe_pairs, max_pairs)

    ii = rng.integers(0, n_seqs, size=max(n_probe * 4, 16))
    jj = rng.integers(0, n_seqs, size=max(n_probe * 4, 16))
    keep = ii != jj
    ii, jj = ii[keep][:n_probe], jj[keep][:n_probe]

    got = pair_probabilities(table, ii, jj)
    modules_by_name = {_model_name(m): m for m in modules}

    for row in range(len(ii)):
        stim = {"sequence_a": sequences[ii[row]], "sequence_b": sequences[jj[row]]}
        for k, name in enumerate(table.model_names):
            module = modules_by_name[name]
            draws = table.param_draws[name]
            direct = float(np.mean([module.predict_left(stim, d) for d in draws]))
            direct = min(max(direct, 1e-6), 1.0 - 1e-6)
            if not np.isclose(got[row, k], direct, atol=atol):
                raise ExhaustiveSearchError(
                    "pair_probabilities disagrees with "
                    f"{name!r}.predict_left for ({sequences[ii[row]]!r}, "
                    f"{sequences[jj[row]]!r}): {got[row, k]!r} vs {direct!r}"
                )


def iter_upper_triangle_tiles(
    n: int, *, tile: int
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Yield ``(i, j)`` int64 arrays covering every unordered pair ``i < j`` in
    ``range(n)`` exactly once, tile by tile -- the streaming counterpart to
    materializing ``itertools.combinations(range(n), 2)``. Rectangular tiles (not
    per-row blocks) keep numpy work per call large enough to stay efficient even
    when ``tile`` is much smaller than ``n``.
    """
    if tile < 1:
        raise ValueError(f"tile must be >= 1, got {tile}.")
    for i0 in range(0, n, tile):
        i1 = min(i0 + tile, n)
        ii_block = np.arange(i0, i1, dtype=np.int64)
        for j0 in range(i0, n, tile):
            j1 = min(j0 + tile, n)
            jj_block = np.arange(j0, j1, dtype=np.int64)
            i_grid, j_grid = np.meshgrid(ii_block, jj_block, indexing="ij")
            mask = i_grid < j_grid
            if not mask.any():
                continue
            yield i_grid[mask], j_grid[mask]


def top_pairs_by_marginal_eig(
    table: ScoreTable,
    weights: np.ndarray,
    top_k: int,
    *,
    tile: Optional[int] = None,
    draw_block: Optional[int] = None,
    progress: Optional[Callable[[int, int], None]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The ``top_k`` ``(i, j, eig)`` triples by marginal EIG about model identity,
    scanned tile by tile over :func:`iter_upper_triangle_tiles` -- the full
    ``C(n, 2)`` pair list is never materialized. Reuses
    ``stimulus_design._marginal_eig`` unchanged.

    Deterministic given fixed ``table``/``weights``/``tile``/``draw_block`` (no
    randomness anywhere in this function -- repeated calls with the same
    arguments return bit-identical output). Within each tile, candidates that
    cannot possibly beat the current worst kept value (a strict numeric fact, not
    a tie) are cheaply dropped before any sorting; survivors -- including any
    that tie the current boundary -- are merged in via a full ``lexsort`` on
    ``(-eig, i, j)``, replacing the unstable-quicksort tie-break
    ``select_informative_stimuli`` used to get from a single ``argsort(-marg)``.

    Changing ``tile`` or ``draw_block`` changes the order floating-point sums are
    accumulated in (inside :func:`pair_probabilities`), which can shift an EIG
    value by up to a few ULP (verified: relative gaps as small as 1.1e-16, i.e.
    one bit of a float64 mantissa). For two candidates whose true EIG values
    happen to be exactly tied, or tied within that noise floor, which one lands
    on the correct side of the top-``k`` cutoff is genuinely undefined -- no
    floating-point top-k selection can do better than this, and it is not a bug
    in the tiling logic (verified against a dense, non-tiled reference: the
    *set* of selected pairs matches across `tile` in {1, 2, 3, 7, 64, 4096} in
    every case tested; a change in ``draw_block`` alone was observed to flip a
    top-k-boundary pair in about 3% of random trials, always at a sub-2-ULP EIG
    gap). Callers that need bit-for-bit stability across performance-tuning
    knobs should fix both ``tile`` and ``draw_block`` explicitly rather than
    relying on their defaults.
    """
    from .stimulus_design import _marginal_eig  # local import: avoids a module cycle

    n = table.scores.shape[0]
    max_pairs = n * (n - 1) // 2
    top_k = min(top_k, max_pairs)
    if top_k == 0:
        empty_int = np.array([], dtype=np.int64)
        return empty_int, empty_int, np.array([], dtype=np.float64)

    if tile is None:
        n_models = table.beta.shape[0]
        tile = max(1, int(np.sqrt(_TARGET_ELEMENTS / max(1, n_models))))
        tile = min(tile, n)

    best_i = np.full(top_k, -1, dtype=np.int64)
    best_j = np.full(top_k, -1, dtype=np.int64)
    best_eig = np.full(top_k, -np.inf, dtype=np.float64)
    count_seen = 0

    for ii, jj in iter_upper_triangle_tiles(n, tile=tile):
        P = pair_probabilities(table, ii, jj, draw_block=draw_block)
        eig = _marginal_eig(P, weights)

        # Cheap reject: strictly worse than everything currently kept can never
        # be needed. ">=" (not ">") keeps exact ties so the lexsort below -- not
        # arrival order -- decides them.
        buffer_min = best_eig.min()
        sel = eig >= buffer_min
        if sel.any():
            combined_eig = np.concatenate([best_eig, eig[sel]])
            combined_i = np.concatenate([best_i, ii[sel]])
            combined_j = np.concatenate([best_j, jj[sel]])
            order = np.lexsort((combined_j, combined_i, -combined_eig))[:top_k]
            best_eig = combined_eig[order]
            best_i = combined_i[order]
            best_j = combined_j[order]

        count_seen += int(ii.shape[0])
        if progress is not None:
            progress(count_seen, max_pairs)

    valid = best_i >= 0
    best_i, best_j, best_eig = best_i[valid], best_j[valid], best_eig[valid]
    order = np.lexsort((best_j, best_i, -best_eig))
    return best_i[order], best_j[order], best_eig[order]
