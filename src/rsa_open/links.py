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
- **allocation**: split ``total_points`` points (or sliders) over the referents. A
  Dirichlet-multinomial with concentration ``kappa`` times the tempered, lapse-mixed
  posterior. Exact zeros are ordinary outcomes. This is the betting measure of Frank & Goodman
  (2012); the paradigm response type is ``slider``.

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

RATING_CLIP = 1e-6  # logit(p) is taken at p clipped to [RATING_CLIP, 1 - RATING_CLIP]
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
        name="allocation", response_type="slider",
        description=("Split points (or sliders) over the referents: a Dirichlet-multinomial "
                     "around the posterior mixed with a uniform guess."),
    ),
}


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


def allocation_log_prob(latent, points, total: int, kappa, lapse, temperature=1.0) -> jnp.ndarray:
    """log P(points | latent): Dirichlet-multinomial around the tempered, lapse-mixed posterior."""
    conc = jnp.maximum(kappa * with_uniform(temper(jnp.asarray(latent), temperature), lapse),
                       CONCENTRATION_FLOOR)
    return dist.DirichletMultinomial(conc, total_count=total).log_prob(points)


@dataclass(frozen=True)
class LinkLikelihood:
    """One link's likelihood, with the options its paradigm fixes."""

    name: str
    response_type: str
    params: Dict[str, Any]  # name -> numpyro prior
    reference_params: Dict[str, float]  # typical values, for simulations and recovery tests
    options: Dict[str, int] = field(default_factory=dict)

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
        raise KeyError(self.name)

    def stack(self, encoded: List[Dict[str, Any]]) -> Dict[str, jnp.ndarray]:
        keys = encoded[0].keys()
        return {k: jnp.asarray(np.asarray([e[k] for e in encoded])) for k in keys}

    # -- the likelihood
    def log_lik(self, latent: jnp.ndarray, p: Mapping[str, Any], enc: Mapping[str, jnp.ndarray],
                classes: jnp.ndarray) -> jnp.ndarray:
        """(trials,) log-likelihood of each response, from (trials, N_OBJ) latent posteriors."""
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
        raise KeyError(self.name)

    # -- simulation
    def sample(self, key, latent: np.ndarray, p: Mapping[str, Any], probes: np.ndarray) -> List[dict]:
        """One response per trial from (trials, N_OBJ) latent posteriors."""
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
        raise KeyError(self.name)


SHARED_REFERENCE = {"lapse": 0.05, "temperature": 1.0}


def _shared_params() -> Dict[str, Any]:
    """The response process every link shares; the choice link's model-file form says the same."""
    return {"lapse": dist.Beta(1.0, 9.0), "temperature": dist.LogNormal(0.0, 0.5)}


def likelihood(name: str, *, scale_points: Optional[int] = None,
               total_points: Optional[int] = None) -> LinkLikelihood:
    """The named link's likelihood, with the options its paradigm fixes."""
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
