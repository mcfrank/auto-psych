"""Links: from a theory's latent quantities to the likelihood of one response type
(docs/auto_open/PLAN.md §3).

A theory is paradigm-free: it defines ``PARAMS`` and ``listener(params, ctx)``, the listener's
posterior over the objects of one scene. A link is a response process with its own parameters,
shared by every theory on a paradigm and frozen with the paradigm before any data. Theorists
never write links.

Each link has a **likelihood** (`likelihood`): its parameters' priors, a check of each
response, the log-likelihood of a group of trials given the theory's latent posteriors, and a
simulator. `src.rsa_open.fit` fits any theory under any link with it.

Every link shares one response process (PI 2026-10-21; `src.rsa_open.link_kit`): the posterior
is **tempered** (``temperature``, a decision temperature: p ** temperature, renormalized; prior
LogNormal(0, 0.5), median 1) and then mixed with a uniform guess with probability ``lapse``
(prior Beta(1, 9)). Then, per link:

- **choice**: pick one referent; summed over classes of identical objects (the closed
  campaign's likelihood, which is this link at temperature 1).
- **rating**: rate how likely a probed referent is, on a ``scale_points`` scale. An ordered
  probit on the logit of the tempered probability, with equally spaced cutpoints (``center``,
  ``spread``); the lapse is a uniform rating.
- **allocation**: split ``total_points`` points over the referents (a bet). A
  Dirichlet-multinomial with concentration ``kappa`` times the tempered, lapse-mixed
  posterior: the discrete counterpart of a Dirichlet, exact for integer bets, with zeros as
  ordinary outcomes. This is the betting measure of Frank & Goodman (2012); the paradigm
  response type is ``bet``.
- **ordered_beta**: one slider (or bet) from 0 to ``total_points`` on a probed referent
  (Kubinec, 2022): with ``eta`` the logit of its tempered probability q, P(0) = 1 - σ(eta - k1)
  and P(max) = σ(eta - k2), with cutpoints k1 = ``cut_low`` and k2 = k1 + ``cut_gap``; the
  interior values 1..max-1 share P(interior) as a **discrete Beta**: proportional to the
  Beta(q φ, (1 - q) φ) density at v / max (``phi``, the precision), normalized over them, so the
  slider's max + 1 positions get an exact distribution (a density times 1 / max would not be
  one: near 0 or 1 a Beta's mass sits between the slider's steps). The lapse is uniform over the
  max + 1 positions. The paradigm response type is ``slider``. For the design tool the same
  distribution is binned (`SLIDER_BINS`): coarsening can only lose information, so the tool's
  score is a lower bound.

The speaker-side links score the theory's **speaker** (`src.rsa_open.fit.TheoryModel`): its own
``speaker(params, ctx)`` if it defines one, else the speaker derived from its listener
(PI 2026-10-21). Their responses name a ``target`` object and an ``utterance``, an index into
the scene's real messages (never the sink).

- **production**: choose a message for a target. The tempered speaker distribution over the
  real messages, mixed with a uniform choice among them (``lapse``).
- **binary_judgment**: "would the speaker say this message for this one?" The rating link at
  two points on the tempered speaker probability: P(yes) = Φ(logit(q) - ``center``), with a
  random yes/no lapse.

**Two measures** (PI 2026-10-26): a paradigm may ask a second question about the same screen
(`PairLikelihood`, the ``pair`` link). Both parts read the same side of the theory (its listener,
or its speaker), their responses are independent given it, so the pair's log-likelihood is the
sum of the parts' and its outcomes, for the design tool, are their product (at most
``MAX_PAIR_OUTCOMES``). The second part's parameters carry the prefix ``q2_`` (its own temperature,
lapse, ...), its trial fields the prefix ``second_`` (``second_probe``, ``second_target``,
``second_utterance``), and its response sits in the first's as ``"second"``. A second measure is
small: a choice or a rating of at most ``MAX_SECOND_POINTS`` points after a listener measure, a
yes/no judgment after a speaker measure (`SECOND_TYPES`).

The choice link also has a **model-file form**: `compose` puts a theory and the link together
into an ordinary RSA model file (``PARAMS`` and ``choice_probs``), so the closed campaign's
fitting, cross-validation and loop code runs it unchanged. The link's parameters are appended
to the theory's, so a theory may not use their names (`LinkClash`).
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist

from src.rsa_open.link_kit import temper, with_uniform
from src.rsa_open.port import LISTENER

RATING_CLIP = 1e-6
SLIDER_BINS = 10  # interior bins of a slider for the design tool; with 0 and max, 12 outcomes  # logit(p) is taken at p clipped to [RATING_CLIP, 1 - RATING_CLIP]
CONCENTRATION_FLOOR = 1e-8


class LinkClash(ValueError):
    """A theory defines a parameter, or a function, that the link defines."""


@dataclass(frozen=True)
class ModelFileForm:
    params: Dict[str, str]  # name -> numpyro prior, as source over ``_link_dist``
    imports: str
    body: str  # defines choice_probs(params, ctx) in terms of listener(params, ctx)


@dataclass(frozen=True)
class Link:
    name: str
    response_type: str  # the paradigm response type it scores (src.rsa_open.paradigm)
    description: str
    model_file: Optional[ModelFileForm] = None


LINKS: Dict[str, Link] = {
    "choice": Link(
        name="choice", response_type="choice",
        description=("Pick one referent: the tempered listener posterior, mixed with a uniform "
                     "guess with probability lapse."),
        model_file=ModelFileForm(
            params={"lapse": "_link_dist.Beta(1.0, 9.0)", "temperature": "_link_dist.LogNormal(0.0, 0.5)"},
            imports=("import numpyro.distributions as _link_dist\n"
                     "from src.rsa.memo_kit import with_lapse as _link_with_lapse\n"
                     "from src.rsa_open.link_kit import temper as _link_temper\n"),
            body=("def choice_probs(params, ctx):\n"
                  "    return _link_with_lapse(_link_temper(listener(params, ctx), params[\"temperature\"]),\n"
                  "                            params[\"lapse\"])\n"),
        ),
    ),
    "rating": Link(
        name="rating", response_type="rating",
        description=("Rate how likely a probed referent is: an ordered probit on the logit of its "
                     "tempered probability, equally spaced cutpoints, and a uniform-rating lapse."),
    ),
    "allocation": Link(
        name="allocation", response_type="bet",
        description=("Split points over the referents (a bet): a Dirichlet-multinomial around the "
                     "tempered posterior mixed with a uniform guess."),
    ),
    "ordered_beta": Link(
        name="ordered_beta", response_type="slider",
        description=("A 0-max slider on a probed referent: ordered beta, with point masses at 0 and "
                     "max and a Beta in between around its tempered probability."),
    ),
    "production": Link(
        name="production", response_type="production",
        description=("Choose a message for a target: the tempered speaker distribution over the "
                     "scene's real messages, with a uniform lapse among them."),
    ),
    "binary_judgment": Link(
        name="binary_judgment", response_type="binary_judgment",
        description=("Would the speaker say this message for this target? A probit on the logit "
                     "of the tempered speaker probability, with a bias and a random yes/no lapse."),
    ),
}
SPEAKER_LINKS = ("production", "binary_judgment")


# --------------------------------------------------------------------------- the likelihoods

def _safe_log(p: jnp.ndarray) -> jnp.ndarray:
    positive = p > 0
    return jnp.where(positive, jnp.log(jnp.where(positive, p, 1.0)), -jnp.inf)


def rating_probs(p, center, spread, lapse, points: int) -> jnp.ndarray:
    """(..., points) probability of each rating 1..points given the probed referent's
    (tempered) probability p."""
    q = jnp.clip(jnp.asarray(p), RATING_CLIP, 1.0 - RATING_CLIP)
    eta = jnp.log(q) - jnp.log1p(-q)
    cuts = center + spread * (jnp.arange(1, points) - points / 2.0)
    cdf = jax.scipy.stats.norm.cdf(cuts - eta[..., None])
    zeros, ones = jnp.zeros_like(cdf[..., :1]), jnp.ones_like(cdf[..., :1])
    probs = jnp.diff(jnp.concatenate([zeros, cdf, ones], axis=-1), axis=-1)
    return (1.0 - lapse) * probs + lapse / points


# The design tool scores a bet on a sub-bet of at most BET_DESIGN_POINTS points with at most
# MAX_BET_OUTCOMES splits (`design_bet_total`): n points drawn at random from a Dirichlet-
# multinomial bet are themselves Dirichlet-multinomial with n points (a Polya urn), so the sub-bet
# is a garbling of the bet and its EIG a lower bound on the bet's (the data-processing inequality).
BET_DESIGN_POINTS = 10
MAX_BET_OUTCOMES = 300


def bet_compositions(n_objects: int, total: int) -> np.ndarray:
    """(splits, n_objects): every way to put ``total`` points on ``n_objects`` objects, in a fixed order."""
    return np.asarray(_compositions(n_objects, total), dtype=np.int32)


def _compositions(n: int, total: int) -> list:
    if n == 1:
        return [(total,)]
    return [(k,) + rest for k in range(total + 1) for rest in _compositions(n - 1, total - k)]


def design_bet_total(n_objects: int, total: int) -> int:
    """The sub-bet's points for a scene of ``n_objects``: the most, up to the bet's own and
    ``BET_DESIGN_POINTS``, with at most ``MAX_BET_OUTCOMES`` splits."""
    from math import comb

    t = min(total, BET_DESIGN_POINTS)
    while t > 1 and comb(t + n_objects - 1, n_objects - 1) > MAX_BET_OUTCOMES:
        t -= 1
    return t


def allocation_log_prob(latent, points, total: int, kappa, lapse, temperature=1.0) -> jnp.ndarray:
    """log P(points | latent): Dirichlet-multinomial around the tempered, lapse-mixed posterior."""
    conc = jnp.maximum(kappa * with_uniform(temper(jnp.asarray(latent), temperature), lapse),
                       CONCENTRATION_FLOOR)
    return dist.DirichletMultinomial(conc, total_count=total).log_prob(points)


def _ordered_beta_parts(q, p):
    """(P(0), P(max), P(interior), Beta a, Beta b) for the probed referent's tempered q."""
    q = jnp.clip(q, RATING_CLIP, 1.0 - RATING_CLIP)
    eta = jnp.log(q) - jnp.log1p(-q)
    k1, k2 = p["cut_low"], p["cut_low"] + p["cut_gap"]
    p0 = 1.0 - jax.nn.sigmoid(eta - k1)
    p1 = jax.nn.sigmoid(eta - k2)
    return p0, p1, 1.0 - p0 - p1, q * p["phi"], (1.0 - q) * p["phi"]


def ordered_beta_pmf(q, top: int, p) -> jnp.ndarray:
    """(..., top + 1): P(each slider position 0..top) for the probed referent's tempered q."""
    p0, p1, mid, a, b = _ordered_beta_parts(q, p)
    y = jnp.arange(1, top, dtype=jnp.float32) / top
    log_density = dist.Beta(a[..., None], b[..., None]).log_prob(y)
    interior = mid[..., None] * jax.nn.softmax(log_density, axis=-1)
    core = jnp.concatenate([p0[..., None], interior, p1[..., None]], axis=-1)
    return (1.0 - p["lapse"]) * core + p["lapse"] / (top + 1)


def ordered_beta_log_prob(q, value, top: int, p) -> jnp.ndarray:
    """log P(slider at ``value`` of 0..top) for the probed referent's tempered probability q."""
    pmf = ordered_beta_pmf(q, top, p)
    return jnp.log(jnp.take_along_axis(pmf, jnp.asarray(value)[..., None], axis=-1)[..., 0])


def slider_bin_edges(top: int) -> np.ndarray:
    """Interior bin edges over 1..top-1 (values, inclusive lower, exclusive upper)."""
    return np.unique(np.linspace(1, top, SLIDER_BINS + 1).round().astype(int))


def ordered_beta_bin_probs(q, top: int, p) -> jnp.ndarray:
    """(..., n_bins + 2): P(0), each interior bin, P(top), from the same distribution as
    `ordered_beta_pmf`."""
    pmf = ordered_beta_pmf(q, top, p)
    edges = slider_bin_edges(top)
    bins = [pmf[..., lo:hi].sum(axis=-1) for lo, hi in zip(edges[:-1], edges[1:])]
    return jnp.stack([pmf[..., 0], *bins, pmf[..., top]], axis=-1)


@dataclass(frozen=True)
class LinkLikelihood:
    """One link's likelihood, with the options its paradigm fixes."""

    name: str
    response_type: str
    params: Dict[str, Any]  # name -> numpyro prior
    reference_params: Dict[str, float]  # typical values, for simulations and recovery tests
    options: Dict[str, int] = field(default_factory=dict)

    @property
    def side(self) -> str:
        """'speaker' when the link scores the theory's speaker, else 'listener'."""
        return "speaker" if self.name in SPEAKER_LINKS else "listener"

    def named_params(self, by_link: Mapping[str, Mapping[str, float]]) -> Dict[str, float]:
        """This link's entries of a by-link-name parameter table (a ground truth's)."""
        return dict(by_link.get(self.name, {}))

    # -- responses
    def encode(self, ctx, response: Mapping[str, Any], i: int) -> Dict[str, Any]:
        """Check one trial's response against its scene; its numeric fields."""
        n = ctx.shape[0]
        if self.name == "choice":
            c = response.get("choice")
            if not isinstance(c, (int, np.integer)) or not 0 <= c < n:
                raise ValueError(f"trial {i}: a choice response needs 'choice', an object index "
                                 f"0..{n - 1} of its scene; got {dict(response)}")
            return {"cls": ctx.choice_classes()[int(c)]}
        if self.name == "rating":
            k, r = self.options["scale_points"], response.get("probe")
            if not isinstance(r, (int, np.integer)) or not 0 <= r < n:
                raise ValueError(f"trial {i}: a rating response needs 'probe', an object index "
                                 f"0..{n - 1}; got {dict(response)}")
            v = response.get("rating")
            if not isinstance(v, (int, np.integer)) or not 1 <= v <= k:
                raise ValueError(f"trial {i}: rating must be an integer between 1 and {k}; got {v!r}")
            return {"probe": int(r), "rating": int(v) - 1}
        if self.name == "allocation":
            total, pts = self.options["total_points"], response.get("points")
            if not isinstance(pts, (list, tuple)) or len(pts) != n:
                raise ValueError(f"trial {i}: an allocation response needs 'points', one entry per "
                                 f"object ({n}); got {dict(response)}")
            if any(not isinstance(x, (int, np.integer)) for x in pts) or any(x < 0 for x in pts):
                raise ValueError(f"trial {i}: points must be non-negative integers, none negative; got {pts}")
            if sum(pts) != total:
                raise ValueError(f"trial {i}: points must sum to {total}; got {sum(pts)}")
            return {"points": [int(x) for x in pts]}
        if self.name == "ordered_beta":
            top, r = self.options["total_points"], response.get("probe")
            if not isinstance(r, (int, np.integer)) or not 0 <= r < n:
                raise ValueError(f"trial {i}: a slider response needs 'probe', an object index "
                                 f"0..{n - 1}; got {dict(response)}")
            v = response.get("value")
            if not isinstance(v, (int, np.integer)) or not 0 <= v <= top:
                raise ValueError(f"trial {i}: a slider value must be an integer between 0 and {top}; got {v!r}")
            return {"probe": int(r), "value": int(v)}
        if self.name in SPEAKER_LINKS:
            n_utt = ctx.shape[1]
            n_real = n_utt - (1 if ctx.needs_sink else 0)
            t, u = response.get("target"), response.get("utterance")
            if not isinstance(t, (int, np.integer)) or not 0 <= t < n:
                raise ValueError(f"trial {i}: a {self.name} response needs 'target', an object index "
                                 f"0..{n - 1}; got {dict(response)}")
            if not isinstance(u, (int, np.integer)) or not 0 <= u < n_utt:
                raise ValueError(f"trial {i}: a {self.name} response needs 'utterance', a message index "
                                 f"0..{n_real - 1}; got {dict(response)}")
            if u >= n_real:
                raise ValueError(f"trial {i}: utterance {u} is the scene's sink ('no word applies'), "
                                 f"which no speaker says; the real messages are 0..{n_real - 1}")
            out = {"target": int(t), "utterance": int(u)}
            if self.name == "binary_judgment":
                a = response.get("answer")
                if not isinstance(a, bool):
                    raise ValueError(f"trial {i}: a judgment needs 'answer', true or false; got {a!r}")
                out["answer"] = int(a)
            return out
        raise KeyError(self.name)

    def stack(self, encoded: List[Dict[str, Any]]) -> Dict[str, jnp.ndarray]:
        keys = encoded[0].keys()
        return {k: jnp.asarray(np.asarray([e[k] for e in encoded])) for k in keys}

    # -- the likelihood
    def log_lik(self, latent: jnp.ndarray, p: Mapping[str, Any], enc: Mapping[str, jnp.ndarray],
                arrays, classes) -> jnp.ndarray:
        """(trials,) log-likelihood of each response. ``latent`` is (trials, N_OBJ) listener
        posteriors, or for a speaker link (trials, N_UTT, N_OBJ) speaker distributions;
        ``arrays`` are the trials' scene arrays and ``classes`` their choice classes."""
        if self.name == "choice":
            from src.rsa.fit import class_probs

            cp = class_probs(with_uniform(temper(latent, p["temperature"]), p["lapse"]), classes)
            return _safe_log(jnp.take_along_axis(cp, enc["cls"][:, None], axis=1)[:, 0])
        if self.name == "rating":
            tempered = temper(latent, p["temperature"])
            pr = jnp.take_along_axis(tempered, enc["probe"][:, None], axis=1)[:, 0]
            probs = rating_probs(pr, p["center"], p["spread"], p["lapse"], self.options["scale_points"])
            return jnp.log(jnp.take_along_axis(probs, enc["rating"][:, None], axis=1)[:, 0])
        if self.name == "allocation":
            return allocation_log_prob(latent, enc["points"].astype(latent.dtype),
                                       self.options["total_points"], p["kappa"], p["lapse"], p["temperature"])
        if self.name == "ordered_beta":
            q = jnp.take_along_axis(temper(latent, p["temperature"]), enc["probe"][:, None], axis=1)[:, 0]
            return ordered_beta_log_prob(q, enc["value"], self.options["total_points"], p)
        if self.name in SPEAKER_LINKS:
            if self.name == "production":
                dist_u = _speaker_choice(latent, enc["target"], arrays.is_sink, p)
                return _safe_log(jnp.take_along_axis(dist_u, enc["utterance"][:, None], axis=1)[:, 0])
            q = jnp.take_along_axis(_tempered_speaker(latent, enc["target"], arrays.is_sink, p),
                                    enc["utterance"][:, None], axis=1)[:, 0]
            probs = rating_probs(q, p["center"], 1.0, p["lapse"], 2)
            return jnp.log(jnp.take_along_axis(probs, enc["answer"][:, None], axis=1)[:, 0])
        raise KeyError(self.name)

    # -- outcomes, for the design tool (src.rsa_open.design)
    # ordered_beta: binned; allocation: a sub-bet (`design_bet_total`)
    CATEGORICAL = ("choice", "rating", "production", "binary_judgment", "ordered_beta", "allocation")

    def outcome_probs(self, latent: jnp.ndarray, p: Mapping[str, Any], unit: Mapping[str, jnp.ndarray],
                      arrays, classes) -> jnp.ndarray:
        """(units, W): the probability of each outcome of each trial unit; ``unit`` holds a
        unit's response fields other than the outcome itself (a probe, a target, a message).
        log of entry k is `log_lik` of the response `outcome_response(k)`."""
        if self.name == "choice":
            from src.rsa.fit import class_probs

            return class_probs(with_uniform(temper(latent, p["temperature"]), p["lapse"]), classes)
        if self.name == "rating":
            tempered = temper(latent, p["temperature"])
            pr = jnp.take_along_axis(tempered, unit["probe"][:, None], axis=1)[:, 0]
            return rating_probs(pr, p["center"], p["spread"], p["lapse"], self.options["scale_points"])
        if self.name == "ordered_beta":
            q = jnp.take_along_axis(temper(latent, p["temperature"]), unit["probe"][:, None], axis=1)[:, 0]
            return ordered_beta_bin_probs(q, self.options["total_points"], p)
        if self.name == "production":
            return _speaker_choice(latent, unit["target"], arrays.is_sink, p)
        if self.name == "binary_judgment":
            q = jnp.take_along_axis(_tempered_speaker(latent, unit["target"], arrays.is_sink, p),
                                    unit["utterance"][:, None], axis=1)[:, 0]
            return rating_probs(q, p["center"], 1.0, p["lapse"], 2)
        if self.name == "allocation":
            n = latent.shape[1]
            t = design_bet_total(n, self.options["total_points"])
            comps = jnp.asarray(bet_compositions(n, t), dtype=latent.dtype)  # (W, n)
            ll = allocation_log_prob(latent[:, None, :], comps[None], t, p["kappa"], p["lapse"], p["temperature"])
            return jnp.exp(ll)
        raise NotImplementedError(
            f"the {self.name!r} link's outcomes are not categorical; the design tool cannot score it yet")

    def outcome_valid(self, arrays, classes, unit: Mapping[str, jnp.ndarray]) -> jnp.ndarray:
        """(units, W) bool: the outcome slots that exist for each unit."""
        if self.name == "choice":
            c = jnp.asarray(classes)
            return c == jnp.arange(c.shape[1])[None, :]  # a class sits at its first object
        if self.name == "rating":
            return jnp.ones((unit["probe"].shape[0], self.options["scale_points"]), dtype=bool)
        if self.name == "ordered_beta":
            return jnp.ones((unit["probe"].shape[0], len(slider_bin_edges(self.options["total_points"])) + 1),
                            dtype=bool)
        if self.name == "production":
            return arrays.is_sink < 0.5
        if self.name == "binary_judgment":
            return jnp.ones((unit["target"].shape[0], 2), dtype=bool)
        if self.name == "allocation":
            n = jnp.asarray(classes).shape[1]
            w = len(bet_compositions(n, design_bet_total(n, self.options["total_points"])))
            return jnp.ones((jnp.asarray(classes).shape[0], w), dtype=bool)
        raise NotImplementedError(self.name)

    def outcome_response(self, k: int) -> Dict[str, Any]:
        """The encoded response fields that name outcome slot ``k``."""
        field_name = {"choice": "cls", "rating": "rating", "production": "utterance",
                      "binary_judgment": "answer"}[self.name]
        return {field_name: jnp.asarray([k])}

    # -- simulation
    def sample(self, key, latent: np.ndarray, p: Mapping[str, Any], probes: np.ndarray,
               arrays=None, utterances: Optional[np.ndarray] = None) -> List[dict]:
        """One response per trial. ``probes`` are the probed (rating) or target (speaker) objects;
        ``utterances`` the messages judged (binary_judgment)."""
        latent = jnp.asarray(latent)
        if self.name == "choice":
            picks = jax.random.categorical(
                key, _safe_log(with_uniform(temper(latent, p["temperature"]), p["lapse"])), axis=-1)
            return [{"choice": int(c)} for c in np.asarray(picks)]
        if self.name == "rating":
            tempered = temper(latent, p["temperature"])
            pr = tempered[jnp.arange(latent.shape[0]), jnp.asarray(probes)]
            probs = rating_probs(pr, p["center"], p["spread"], p["lapse"], self.options["scale_points"])
            ratings = jax.random.categorical(key, jnp.log(probs), axis=-1)
            return [{"probe": int(r), "rating": int(v) + 1} for r, v in zip(probes, np.asarray(ratings))]
        if self.name == "allocation":
            conc = jnp.maximum(p["kappa"] * with_uniform(temper(latent, p["temperature"]), p["lapse"]),
                               CONCENTRATION_FLOOR)
            pts = dist.DirichletMultinomial(conc, total_count=self.options["total_points"]).sample(key)
            return [{"points": [int(x) for x in row]} for row in np.asarray(pts)]
        if self.name == "ordered_beta":
            top = self.options["total_points"]
            q = temper(latent, p["temperature"])[jnp.arange(latent.shape[0]), jnp.asarray(probes)]
            values = np.asarray(jax.random.categorical(key, jnp.log(ordered_beta_pmf(q, top, p)), axis=-1))
            return [{"probe": int(r), "value": int(v)} for r, v in zip(probes, values)]
        if self.name == "production":
            dist_u = _speaker_choice(latent, jnp.asarray(probes), arrays.is_sink, p)
            said = jax.random.categorical(key, _safe_log(dist_u), axis=-1)
            return [{"target": int(t), "utterance": int(u)} for t, u in zip(probes, np.asarray(said))]
        if self.name == "binary_judgment":
            q = _tempered_speaker(latent, jnp.asarray(probes), arrays.is_sink, p)[
                jnp.arange(latent.shape[0]), jnp.asarray(utterances)]
            yes = jax.random.uniform(key, q.shape) < rating_probs(q, p["center"], 1.0, p["lapse"], 2)[:, 1]
            return [{"target": int(t), "utterance": int(u), "answer": bool(a)}
                    for t, u, a in zip(probes, utterances, np.asarray(yes))]
        raise KeyError(self.name)


SECOND_PREFIX = "q2_"  # the second measure's parameters
SECOND_ENC = "q2."  # its encoded response fields
SECOND_FIELDS = {"probe": "second_probe", "target": "second_target", "utterance": "second_utterance"}
SECOND_TYPES = {"listener": ("choice", "rating"), "speaker": ("binary_judgment",)}
MAX_SECOND_POINTS = 7
MAX_PAIR_OUTCOMES = 300


@dataclass(frozen=True)
class PairLikelihood:
    """Two measures of one screen (see the module's docstring): the parts' likelihoods summed."""

    first: LinkLikelihood
    second: LinkLikelihood
    name: str = "pair"

    def __post_init__(self) -> None:
        if self.first.side != self.second.side:
            raise ValueError(f"a pair's measures read one side of the theory: {self.first.name} reads its "
                             f"{self.first.side}, {self.second.name} its {self.second.side}")
        if self.second.name not in SECOND_TYPES[self.first.side]:
            raise ValueError(f"after a {self.first.side} measure the second is one of "
                             f"{SECOND_TYPES[self.first.side]}, not {self.second.name}")

    @property
    def side(self) -> str:
        return self.first.side

    @property
    def response_type(self) -> str:
        return f"{self.first.response_type}+{self.second.response_type}"

    @property
    def params(self) -> Dict[str, Any]:
        return {**self.first.params, **{SECOND_PREFIX + k: v for k, v in self.second.params.items()}}

    @property
    def reference_params(self) -> Dict[str, float]:
        return {**self.first.reference_params,
                **{SECOND_PREFIX + k: v for k, v in self.second.reference_params.items()}}

    @property
    def options(self) -> Dict[str, Any]:
        return {"first": {"name": self.first.name, **self.first.options},
                "second": {"name": self.second.name, **self.second.options}}

    def named_params(self, by_link: Mapping[str, Mapping[str, float]]) -> Dict[str, float]:
        return {**self.first.named_params(by_link),
                **{SECOND_PREFIX + k: v for k, v in self.second.named_params(by_link).items()}}

    def _p(self, p: Mapping[str, Any]):
        return ({k: p[k] for k in self.first.params},
                {k: p[SECOND_PREFIX + k] for k in self.second.params})

    @staticmethod
    def _enc(enc: Mapping[str, Any]):
        return ({k: v for k, v in enc.items() if not k.startswith(SECOND_ENC)},
                {k[len(SECOND_ENC):]: v for k, v in enc.items() if k.startswith(SECOND_ENC)})

    @staticmethod
    def _unit(unit: Mapping[str, Any]):
        second = {v: k for k, v in SECOND_FIELDS.items()}
        return ({k: v for k, v in unit.items() if k not in second},
                {second[k]: v for k, v in unit.items() if k in second})

    def encode(self, ctx, response: Mapping[str, Any], i: int) -> Dict[str, Any]:
        r2 = response.get("second")
        if not isinstance(r2, Mapping):
            raise ValueError(f"trial {i}: a two-measure response needs 'second', the second measure's "
                             f"response; got {dict(response)}")
        e1 = self.first.encode(ctx, {k: v for k, v in response.items() if k != "second"}, i)
        e2 = self.second.encode(ctx, r2, i)
        return {**e1, **{SECOND_ENC + k: v for k, v in e2.items()}}

    def stack(self, encoded: List[Dict[str, Any]]) -> Dict[str, jnp.ndarray]:
        keys = encoded[0].keys()
        return {k: jnp.asarray(np.asarray([e[k] for e in encoded])) for k in keys}

    def log_lik(self, latent, p, enc, arrays, classes) -> jnp.ndarray:
        (p1, p2), (e1, e2) = self._p(p), self._enc(enc)
        return (self.first.log_lik(latent, p1, e1, arrays, classes)
                + self.second.log_lik(latent, p2, e2, arrays, classes))

    def outcome_probs(self, latent, p, unit, arrays, classes) -> jnp.ndarray:
        (p1, p2), (u1, u2) = self._p(p), self._unit(unit)
        a = self.first.outcome_probs(latent, p1, u1, arrays, classes)
        b = self.second.outcome_probs(latent, p2, u2, arrays, classes)
        return (a[:, :, None] * b[:, None, :]).reshape(a.shape[0], -1)

    def outcome_valid(self, arrays, classes, unit) -> jnp.ndarray:
        u1, u2 = self._unit(unit)
        a = self.first.outcome_valid(arrays, classes, u1)
        b = self.second.outcome_valid(arrays, classes, u2)
        return (a[:, :, None] & b[:, None, :]).reshape(a.shape[0], -1)

    def sample(self, key, latent, p, probes, arrays=None, utterances=None, second_probes=None,
               second_utterances=None) -> List[dict]:
        """One response per trial, the second measure's under ``"second"``: its probed or target
        objects ``second_probes`` and judged messages ``second_utterances`` (default: the first's)."""
        (p1, p2), (k1, k2) = self._p(p), jax.random.split(key)
        first = self.first.sample(k1, latent, p1, probes, arrays, utterances)
        second = self.second.sample(k2, latent, p2, probes if second_probes is None else second_probes, arrays,
                                    utterances if second_utterances is None else second_utterances)
        return [{**a, "second": b} for a, b in zip(first, second)]


def _tempered_speaker(speaker, targets, is_sink, p) -> jnp.ndarray:
    """(trials, N_UTT): the tempered speaker distribution over messages for each trial's target."""
    column = jnp.take_along_axis(speaker, targets[:, None, None], axis=2)[:, :, 0]
    return temper(column * (1.0 - is_sink), p["temperature"])


def _speaker_choice(speaker, targets, is_sink, p) -> jnp.ndarray:
    """(trials, N_UTT): what is said, with the lapse uniform over the real messages."""
    real = 1.0 - is_sink
    return ((1.0 - p["lapse"]) * _tempered_speaker(speaker, targets, is_sink, p)
            + p["lapse"] * real / real.sum(axis=-1, keepdims=True))


SHARED_REFERENCE = {"lapse": 0.05, "temperature": 1.0}


def _shared_params() -> Dict[str, Any]:
    """The response process every link shares; the choice link's model-file form says the same."""
    return {"lapse": dist.Beta(1.0, 9.0), "temperature": dist.LogNormal(0.0, 0.5)}


def likelihood(name: str, *, scale_points: Optional[int] = None,
               total_points: Optional[int] = None, first: Optional[Mapping[str, Any]] = None,
               second: Optional[Mapping[str, Any]] = None):
    """The named link's likelihood, with the options its paradigm fixes; ``pair`` takes its parts
    (``first``, ``second``: each a link name and its options)."""
    if name == "pair":
        if first is None or second is None:
            raise ValueError("the pair link needs first and second")
        parts = [likelihood(**dict(part)) for part in (first, second)]
        return PairLikelihood(*parts)
    if name not in LINKS:
        raise KeyError(f"no link {name!r}; the library has {sorted(LINKS)}")
    link = LINKS[name]
    if name == "choice":
        return LinkLikelihood(name, link.response_type, {**_shared_params()}, {**SHARED_REFERENCE})
    if name == "rating":
        if scale_points is None:
            raise ValueError("the rating link needs scale_points")
        return LinkLikelihood(
            name, link.response_type,
            {"center": dist.Normal(0.0, 2.0), "spread": dist.LogNormal(0.0, 1.0), **_shared_params()},
            {"center": 0.0, "spread": 0.8, **SHARED_REFERENCE},
            {"scale_points": int(scale_points)})
    if name == "allocation":
        if total_points is None:
            raise ValueError("the allocation link needs total_points")
        return LinkLikelihood(
            name, link.response_type,
            {"kappa": dist.LogNormal(float(np.log(10.0)), 1.0), **_shared_params()},
            {"kappa": 10.0, **SHARED_REFERENCE},
            {"total_points": int(total_points)})
    if name == "ordered_beta":
        if total_points is None:
            raise ValueError("the ordered_beta link needs total_points (the slider's maximum)")
        return LinkLikelihood(
            name, link.response_type,
            {"cut_low": dist.Normal(-3.0, 1.5), "cut_gap": dist.LogNormal(float(np.log(6.0)), 0.5),
             "phi": dist.LogNormal(float(np.log(10.0)), 1.0), **_shared_params()},
            {"cut_low": -3.0, "cut_gap": 6.0, "phi": 10.0, **SHARED_REFERENCE},
            {"total_points": int(total_points)})
    if name == "production":
        return LinkLikelihood(name, link.response_type, {**_shared_params()}, {**SHARED_REFERENCE})
    if name == "binary_judgment":
        return LinkLikelihood(name, link.response_type, {"center": dist.Normal(0.0, 2.0), **_shared_params()},
                              {"center": 0.0, **SHARED_REFERENCE})
    raise KeyError(name)


# --------------------------------------------------------------------------- the model-file form

def _theory_params(tree: ast.Module) -> List[str]:
    for node in tree.body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "PARAMS"):
            if not isinstance(node.value, ast.Dict) or not all(
                    isinstance(k, ast.Constant) and isinstance(k.value, str) for k in node.value.keys):
                raise ValueError("a theory's PARAMS must be a dict literal with string keys")
            return [k.value for k in node.value.keys]
    raise ValueError("a theory must define PARAMS")


def compose(theory_source: str, link: str) -> str:
    """The model file for ``theory_source`` under the named link (links with a model-file form)."""
    if link not in LINKS:
        raise KeyError(f"no link {link!r}; the library has {sorted(LINKS)}")
    spec = LINKS[link].model_file
    if spec is None:
        raise ValueError(f"the {link!r} link has no model-file form; fit it with src.rsa_open.fit")
    tree = ast.parse(theory_source)
    functions = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    if LISTENER not in functions:
        raise ValueError(f"a theory must define {LISTENER}(params, ctx)")
    if "choice_probs" in functions:
        raise LinkClash("a theory may not define choice_probs: the link does")
    clash = sorted(set(_theory_params(tree)) & set(spec.params))
    if clash:
        raise LinkClash(f"theory parameter(s) {', '.join(clash)} belong to the {link!r} link")
    entries = ", ".join(f'"{k}": {v}' for k, v in spec.params.items())
    footer = (f"\n\n# --- link {link!r}, composed by src.rsa_open.links.compose "
              f"(docs/auto_open/PLAN.md §3) ---\n"
              f"{spec.imports}\n"
              f"PARAMS = {{**PARAMS, {entries}}}\n\n\n"
              f"{spec.body}")
    return theory_source.rstrip("\n") + footer
