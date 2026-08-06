"""Fast exhaustive stimulus search over the pure-Python model families.

Builds on ``sequence_stats.py``'s feature-equivalence quotient: this module
decides, for a given model set, *how much* quotienting is safe (derived from each
model's own declarations, never hardcoded), and audits that decision against the
model's actual ``score_sequence`` rather than trusting the declaration blindly.
``build_exhaustive_design`` (in ``stimulus_design.py``) wires this together with the
streamed pair scan and the greedy selection.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np

from . import sequence_stats


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
    to ``False``. This must never be hardcoded True: the live PyMC seed
    ``minkowski_accumulated_typicality`` has a free ``ideal_p != 0.5`` and is
    genuinely complement-*sensitive*, so a model set that includes it (or its
    pure-Python twin, if one is ever added) must not fold complements.
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
