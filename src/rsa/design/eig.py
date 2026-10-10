"""Joint expected information gain for reference-game designs, and the power it implies.

The RSA counterpart of `src.models.eig_selection` (subjective randomness, on
main), with the same estimator: per-draw prior-predictive probabilities keep
the correlation that shared parameters induce across displays, Monte Carlo
scenarios (a model from the prior, one of its draws, responses simulated from
that draw) give a posterior over models, the scenario's own draw is left out
of its model's likelihood average, and selection is greedy with a noise-floor
stop, the remaining slots filled by single-response EIG conditioned on the
picks so far.

Two things differ from the binary task:

* **Multinomial outcomes.** A display has 2-4 choice classes (identical objects
  are one class, `Context.choice_classes`), and with every participant seeing
  every display a display yields counts ~ Multinomial(N, p). The outcome space
  is too large to enumerate, so each scenario draws its counts for every pool
  display once, up front, with the scenario's draw (common random numbers:
  every candidate is scored on the same simulated worlds). The multinomial
  coefficient is the same for every model and cancels.
* **Power.** The same scenarios, scored on a finished design at several N,
  say how often the model that generated the data ends with the highest
  posterior: the power of one experiment to tell the promoted models apart.
* **Quotas** (PI 2026-10-10). Optional minimum counts of kinds of display
  (`Quota`, e.g. at least 6 with two objects). Selection stays greedy on EIG,
  among the displays that leave every quota reachable with the picks still to
  make: free while the quotas can wait, and confined to the kinds still short
  once they cannot. Unconstrained EIG put 31 of 40 displays on 4 x 4 games in
  the first rehearsal, and 37 of 40 on mumble trials in another experiment.

Inputs are arrays: ``probs[m]`` is model m's (draws, displays, width) class
probabilities, each class's probability at its first object's index and zero
at the other members (`src.rsa.fit.class_probs`), so every model shares one
layout; ``valid`` (displays, width) marks the class slots.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

LOG_FLOOR = 1e-12
NEGLIGIBLE_GAIN_BITS = 1e-6  # as main's src.models.eig_selection


def _entropy_bits(w: np.ndarray) -> np.ndarray:
    w = np.clip(w, 0.0, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(w > 0, w * np.log2(w), 0.0)
    return -terms.sum(-1)


def _validate(probs: Sequence[np.ndarray], valid: np.ndarray) -> None:
    shapes = {p.shape[1:] for p in probs}
    if len(shapes) != 1 or next(iter(shapes)) != valid.shape:
        raise ValueError(f"models' probabilities must all be (draws, {valid.shape}), not {[p.shape for p in probs]}")
    if any(p.shape[0] < 2 for p in probs):
        raise ValueError("leave-one-out needs at least two draws per model")
    for m, p in enumerate(probs):
        if not np.all(np.isfinite(p)) or np.any(p < 0):
            raise ValueError(f"model {m}'s probabilities are not finite and non-negative")
        if np.any(p[:, ~valid] > 1e-9):
            raise ValueError(f"model {m} puts probability outside the class slots")
        if not np.allclose(p.sum(-1), 1.0, atol=1e-4):
            raise ValueError(f"model {m}'s class probabilities do not sum to 1")


@dataclass
class Scenarios:
    """Simulated worlds: each a model, one of its draws and counts for every display."""

    model: np.ndarray  # (S,)
    draw: np.ndarray  # (S,)
    counts: np.ndarray  # (S, displays, width) int


def simulate(probs: Sequence[np.ndarray], prior: np.ndarray, n_responses: int, n_scenarios: int,
             seed: int) -> Scenarios:
    rng = np.random.default_rng(seed)
    model = rng.choice(len(probs), size=n_scenarios, p=prior)
    draw = np.array([rng.integers(probs[m].shape[0]) for m in model])
    p = np.stack([probs[m][d] for m, d in zip(model, draw)]).astype(np.float64)  # (S, C, W)
    p /= p.sum(-1, keepdims=True)
    counts = rng.multinomial(n_responses, p).astype(np.int32)
    return Scenarios(model, draw, counts)


class State:
    """Per-scenario, per-model, per-draw log-likelihood of the displays chosen so far."""

    def __init__(self, probs: Sequence[np.ndarray], valid: np.ndarray, prior: np.ndarray, scen: Scenarios,
                 leave_one_out: bool = True):
        self.logp = [np.where(valid, np.log(np.maximum(p, LOG_FLOOR)), 0.0) for p in probs]  # (D_m, C, W)
        self.prior = prior
        self.scen = scen
        self.ll = [np.zeros((len(scen.model), lp.shape[0])) for lp in self.logp]
        # A scenario's own draw is left out of its own model's average (main's C5).
        self.mask = [np.ones((len(scen.model), lp.shape[0]), dtype=bool) for lp in self.logp]
        if leave_one_out:
            for m in range(len(probs)):
                rows = np.where(scen.model == m)[0]
                self.mask[m][rows, scen.draw[rows]] = False

    def add(self, c: int) -> None:
        for m, lp in enumerate(self.logp):
            self.ll[m] += self.scen.counts[:, c, :] @ lp[:, c, :].T

    def _posterior(self, ll: List[np.ndarray]) -> np.ndarray:
        """(S, M) model posterior of each scenario."""
        logm = []
        for m, l in enumerate(ll):
            l = np.where(self.mask[m], l, -np.inf)
            top = l.max(1, keepdims=True)
            logm.append(top[:, 0] + np.log(np.exp(l - top).sum(1) / self.mask[m].sum(1)))
        logpost = np.stack(logm, 1) + np.log(self.prior)
        logpost -= logpost.max(1, keepdims=True)
        post = np.exp(logpost)
        return post / post.sum(1, keepdims=True)

    def posterior(self) -> np.ndarray:
        return self._posterior(self.ll)

    def entropies_with(self, c: int) -> np.ndarray:
        """Each scenario's posterior entropy (bits) if display c were added."""
        ll = [l + self.scen.counts[:, c, :] @ lp[:, c, :].T for l, lp in zip(self.ll, self.logp)]
        return _entropy_bits(self._posterior(ll))


@dataclass(frozen=True)
class Quota:
    """At least ``minimum`` of a design's displays from ``members`` (a bool mask
    over the displays). The quotas of one ``dimension`` (e.g. "objects") are
    disjoint kinds of display, so each dimension's shortfall is at most the
    picks left; kinds of different dimensions (objects, query) cross."""

    name: str
    dimension: str
    members: np.ndarray  # (displays,) bool
    minimum: int


def _check_quotas(quotas: Sequence[Quota], pool: Sequence[int], n_select: int) -> None:
    by_dim: Dict[str, List[Quota]] = {}
    for q in quotas:
        by_dim.setdefault(q.dimension, []).append(q)
        available = int(q.members[list(pool)].sum())
        if q.minimum > available:
            raise ValueError(f"quota {q.name}: at least {q.minimum} displays, but the pool has {available}")
    for dim, qs in by_dim.items():
        if sum(q.minimum for q in qs) > n_select:
            raise ValueError(f"the {dim} quotas need {sum(q.minimum for q in qs)} displays of {n_select}")
        overlap = np.sum([q.members for q in qs], axis=0)
        if np.any(overlap > 1):
            raise ValueError(f"the {dim} quotas overlap: a display may count toward one kind per dimension")


def _allowed(quotas: Sequence[Quota], picks: Sequence[int], left: Sequence[int], n_select: int) -> List[int]:
    """The displays in ``left`` whose pick leaves every quota reachable."""
    if not quotas:
        return list(left)
    remaining = n_select - len(picks) - 1
    short: Dict[str, List[Quota]] = {}
    for q in quotas:
        if q.minimum > int(q.members[list(picks)].sum()):
            short.setdefault(q.dimension, []).append(q)
    gap = {dim: sum(q.minimum - int(q.members[list(picks)].sum()) for q in qs) for dim, qs in short.items()}
    out = []
    for c in left:
        if all(gap[dim] - any(q.members[c] for q in qs) <= remaining for dim, qs in short.items()):
            out.append(c)
    return out


def composition(quotas: Sequence[Quota], picks: Sequence[int]) -> Dict[str, int]:
    """How many of ``picks`` each quota's kind has."""
    return {q.name: int(q.members[list(picks)].sum()) for q in quotas}


@dataclass
class Selection:
    indices: List[int]
    joint_eig_bits: List[float]  # after each pick
    sources: List[str]  # "eig" or "eig_single_response_fill"


def select(probs: Sequence[np.ndarray], valid: np.ndarray, n_select: int, *, n_responses: int,
           prior: Optional[np.ndarray] = None, n_scenarios: int = 2000, seed: int = 0,
           candidates: Optional[Sequence[int]] = None, leave_one_out: bool = True,
           quotas: Sequence[Quota] = ()) -> Selection:
    """Greedy joint-EIG selection of ``n_select`` displays, each answered ``n_responses`` times.

    Stops at the noise floor (the best gain within two Monte Carlo SE of zero)
    and fills the rest by single-response EIG conditioned on the picks, as
    main's design does (user decision 2026-09-26). With ``quotas``, each pick
    is the best among the displays that keep every quota reachable.
    """
    _validate(probs, valid)
    prior = np.full(len(probs), 1 / len(probs)) if prior is None else np.asarray(prior, float) / np.sum(prior)
    pool = list(range(valid.shape[0])) if candidates is None else list(candidates)
    if n_select > len(pool):
        raise ValueError(f"{n_select} picks from a pool of {len(pool)}")
    _check_quotas(quotas, pool, n_select)
    h0 = float(_entropy_bits(prior))
    out = Selection([], [], [])
    for responses, source in ((n_responses, "eig"), (1, "eig_single_response_fill")):
        state = State(probs, valid, prior, simulate(probs, prior, responses, n_scenarios, seed), leave_one_out)
        for c in out.indices:
            state.add(c)
        current = _entropy_bits(state.posterior())
        while len(out.indices) < n_select:
            left = _allowed(quotas, out.indices, [c for c in pool if c not in out.indices], n_select)
            if not left:
                raise ValueError(f"no display keeps the quotas reachable after {len(out.indices)} picks")
            ents = {c: state.entropies_with(c) for c in left}
            best = min(left, key=lambda c: ents[c].mean())
            gain = current - ents[best]
            se = gain.std(ddof=1) / np.sqrt(len(gain))
            if source == "eig" and (gain.mean() <= 2 * se or gain.mean() <= NEGLIGIBLE_GAIN_BITS):
                break
            state.add(best)
            current = ents[best]
            out.indices.append(best)
            out.joint_eig_bits.append(h0 - float(current.mean()))
            out.sources.append(source)
        seed += 1
    return out


def power(probs: Sequence[np.ndarray], valid: np.ndarray, design: Sequence[int], n_responses: Sequence[int], *,
          prior: Optional[np.ndarray] = None, n_scenarios: int = 4000, seed: int = 1,
          names: Optional[Sequence[str]] = None, leave_one_out: bool = True) -> List[Dict]:
    """For each N: the design's joint EIG and how often the generating model
    ends with the highest posterior (overall and per generating model), on
    scenarios the design was not selected on."""
    _validate(probs, valid)
    prior = np.full(len(probs), 1 / len(probs)) if prior is None else np.asarray(prior, float) / np.sum(prior)
    names = list(names) if names is not None else [str(m) for m in range(len(probs))]
    rows = []
    for n in n_responses:
        scen = simulate(probs, prior, n, n_scenarios, seed)
        state = State(probs, valid, prior, scen, leave_one_out)
        for c in design:
            state.add(c)
        post = state.posterior()
        hit = post.argmax(1) == scen.model
        truth = post[np.arange(len(scen.model)), scen.model]
        rows.append(dict(
            n_responses=int(n),
            joint_eig_bits=float(_entropy_bits(prior) - _entropy_bits(post).mean()),
            max_bits=float(_entropy_bits(prior)),
            p_correct=float(hit.mean()),
            p_correct_se=float(hit.std(ddof=1) / np.sqrt(len(hit))),
            mean_posterior_on_truth=float(truth.mean()),
            by_model={names[m]: dict(p_correct=float(hit[scen.model == m].mean()),
                                     mean_posterior_on_truth=float(truth[scen.model == m].mean()),
                                     n_scenarios=int((scen.model == m).sum()))
                      for m in range(len(probs)) if (scen.model == m).any()},
        ))
    return rows
