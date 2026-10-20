"""Fitting a theory under a link to any paradigm's responses (docs/auto_open/PLAN.md §3).

The closed campaign's fit (`src.rsa.fit`) knows one likelihood, a choice among objects. Here
the likelihood is the link's (`src.rsa_open.links.LinkLikelihood`): the theory gives each
trial's latent posterior over the objects of its scene, and the link scores the response. The
closed fit is left untouched, so the closed campaign's frozen results cannot move; under the
choice link the two give the same per-trial log-likelihood (tests/test_rsa_open_fit.py).

- `TheoryModel` loads a theory file (``PARAMS``, ``listener``) with the closed loader.
- `OpenData` checks every response against its scene and its link, and groups the trials by
  display shape.
- `fit` samples theory and link parameters jointly with NUTS (the closed fit's settings) and
  stores each trial's pointwise log-likelihood.
- `simulate` draws responses from a theory and link; `recovery_check` simulates from known
  parameters, refits and reports whether they come back: the test every link must pass,
  library and designer-proposed alike.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import arviz as az
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
from numpyro.infer import MCMC, NUTS

from src.models.pymc_inference import convergence_problems
from src.rsa.context import Context, ShapeGroup, group_by_shape
from src.rsa.fit import FitSettings, draw_batches
from src.rsa.model_file import ModelContractViolation, RSAModel
from src.rsa_open.links import LinkLikelihood
from src.rsa_open.port import LISTENER

OBSERVED_SITE = "response"


class TheoryModel(RSAModel):
    """A theory file, loaded like a model file but evaluated through ``listener``."""

    def _choice_probs(self, shape: Tuple[int, int]):
        fn = getattr(self.module(shape), LISTENER, None)
        if not callable(fn):
            raise ModelContractViolation(
                f"{self.path.name} must define {LISTENER}(params, ctx) returning the listener's "
                f"posterior over the objects, shape (N_OBJ,)")
        return fn


@dataclass(frozen=True)
class OpenData:
    """Trials grouped by display shape, each response checked and encoded for its link."""

    groups: List[ShapeGroup]
    encoded: List[Dict[str, jnp.ndarray]]  # per group, the link's response arrays
    n_trials: int

    @classmethod
    def build(cls, contexts: Sequence[Context], responses: Sequence[Mapping[str, Any]],
              link: LinkLikelihood) -> "OpenData":
        if len(contexts) != len(responses):
            raise ValueError(f"{len(contexts)} scenes but {len(responses)} responses")
        per_trial = [link.encode(ctx, r, i) for i, (ctx, r) in enumerate(zip(contexts, responses))]
        groups = list(group_by_shape(contexts).values())
        encoded = [link.stack([per_trial[i] for i in g.indices]) for g in groups]
        return cls(groups, encoded, len(contexts))


def _split(theory: RSAModel, link: LinkLikelihood, params: Mapping[str, Any]):
    theory_params = {k: params[k] for k in theory.params}
    link_params = {k: params[k] for k in link.params}
    return theory_params, link_params


def trial_log_lik(theory: RSAModel, link: LinkLikelihood, data: OpenData,
                  params: Mapping[str, Any]) -> jnp.ndarray:
    """(trials,) log-likelihood of each response at one parameter setting, in input order."""
    tp, lp = _split(theory, link, params)
    out = jnp.zeros(data.n_trials)
    for group, enc in zip(data.groups, data.encoded):
        latent = theory.group_probs(tp, group)
        out = out.at[jnp.asarray(group.indices)].set(link.log_lik(latent, lp, enc, jnp.asarray(group.classes)))
    return out


def _numpyro_model(theory: RSAModel, link: LinkLikelihood, data: OpenData):
    params = {name: numpyro.sample(name, prior) for name, prior in theory.params.items()}
    for name, prior in link.params.items():
        if name in params:
            raise ValueError(f"theory parameter {name!r} belongs to the {link.name!r} link")
        params[name] = numpyro.sample(name, prior)
    tp, lp = _split(theory, link, params)
    total = 0.0
    for group, enc in zip(data.groups, data.encoded):
        total = total + jnp.sum(link.log_lik(theory.group_probs(tp, group), lp, enc, jnp.asarray(group.classes)))
    numpyro.factor(OBSERVED_SITE, total)


def pointwise_log_lik(theory: RSAModel, link: LinkLikelihood, data: OpenData,
                      flat: Mapping[str, np.ndarray]) -> np.ndarray:
    """(draws, trials) log-likelihood of each response at each posterior draw."""
    n_draws = len(next(iter(flat.values())))
    out = np.empty((n_draws, data.n_trials), dtype=np.float32)
    per_draw = jax.vmap(lambda latent, lp, enc, classes: link.log_lik(latent, lp, enc, classes),
                        in_axes=(0, 0, None, None))
    for group, enc in zip(data.groups, data.encoded):
        start = 0
        for n, params in draw_batches(flat, np.arange(n_draws)):
            tp, lp = _split(theory, link, params)
            latent = theory.draws_probs(tp, group)
            out[start:start + n, group.indices] = np.asarray(per_draw(latent, lp, enc, jnp.asarray(group.classes)))[:n]
            start += n
    return out


@dataclass
class OpenFit:
    theory_name: str
    link: str
    idata: Any
    param_names: List[str]
    convergence_problems: List[str]

    @property
    def converged(self) -> bool:
        return not self.convergence_problems


def fit(theory: RSAModel, link: LinkLikelihood, data: OpenData,
        settings: FitSettings = FitSettings(dense_mass=True)) -> OpenFit:
    kernel = NUTS(_numpyro_model, target_accept_prob=settings.target_accept, dense_mass=settings.dense_mass)
    mcmc = MCMC(kernel, num_warmup=settings.num_warmup, num_samples=settings.num_samples,
                num_chains=settings.num_chains, chain_method="vectorized", progress_bar=False)
    mcmc.run(jax.random.PRNGKey(settings.seed), theory, link, data, extra_fields=("diverging",))
    samples = {k: np.asarray(v) for k, v in mcmc.get_samples(group_by_chain=True).items()}
    chains, draws = next(iter(samples.values())).shape[:2]
    ll = pointwise_log_lik(theory, link, data, {k: v.reshape(chains * draws) for k, v in samples.items()})
    idata = az.from_dict(
        posterior=samples,
        log_likelihood={OBSERVED_SITE: ll.reshape(chains, draws, -1)},
        sample_stats={"diverging": np.asarray(mcmc.get_extra_fields(group_by_chain=True)["diverging"])},
    )
    names = [*theory.params, *link.params]
    return OpenFit(theory.name, link.name, idata, names, convergence_problems(idata, names))


def simulate(theory: RSAModel, link: LinkLikelihood, contexts: Sequence[Context],
             params: Mapping[str, Any], key, probes: Optional[Sequence[int]] = None) -> List[dict]:
    """One simulated response per scene, in input order. ``probes`` (for ratings) default to a
    random object per trial."""
    tp, lp = _split(theory, link, {k: jnp.asarray(np.asarray(v).reshape(())) for k, v in params.items()})
    rng = np.random.default_rng(np.asarray(jax.random.randint(key, (), 0, 2**31 - 1)).item())
    if probes is None:
        probes = [int(rng.integers(ctx.shape[0])) for ctx in contexts]
    out: List[Optional[dict]] = [None] * len(contexts)
    for k, group in zip(jax.random.split(key, len(contexts) or 1), group_by_shape(contexts).values()):
        latent = theory.group_probs(tp, group)
        for i, response in zip(group.indices, link.sample(k, latent, lp, np.asarray(probes)[group.indices])):
            out[i] = response
    return out  # type: ignore[return-value]


RECOVERY_REPLICATES = 3


def _recovery_run(theory: RSAModel, link: LinkLikelihood, truth: Mapping[str, float], pool: Sequence[Context],
                  n_trials: int, seed: int, settings: Optional[FitSettings], interval: float,
                  min_shrinkage: float) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    scenes = [pool[i] for i in rng.choice(len(pool), n_trials, replace=True)]
    responses = simulate(theory, link, scenes, truth, jax.random.PRNGKey(seed))
    data = OpenData.build(scenes, responses, link)
    settings = settings or FitSettings(num_warmup=500, num_samples=500, num_chains=2, seed=seed,
                                       dense_mass=True)
    fitted = fit(theory, link, data, settings)
    lo, hi = (1 - interval) / 2, 1 - (1 - interval) / 2
    params = {}
    for name in fitted.param_names:
        draws = np.asarray(fitted.idata.posterior[name]).reshape(-1)
        prior = (link.params.get(name) or theory.params[name]).sample(jax.random.PRNGKey(1), (4000,))
        q = np.quantile(draws, [lo, hi])
        row = {"true": float(truth[name]), "mean": float(draws.mean()), "sd": float(draws.std()),
               "interval": [float(q[0]), float(q[1])], "prior_sd": float(np.std(np.asarray(prior)))}
        row["inside"] = bool(q[0] <= row["true"] <= q[1])
        row["learned"] = name not in link.params or row["sd"] <= min_shrinkage * row["prior_sd"]
        params[name] = row
    return {"seed": seed, "converged": fitted.converged, "convergence_problems": fitted.convergence_problems,
            "params": params}


def recovery_check(theory_path: Path, link: LinkLikelihood, truth: Mapping[str, float], *,
                   n_trials: int = 1500, seed: int = 0, replicates: int = RECOVERY_REPLICATES,
                   contexts: Optional[Sequence[Context]] = None, settings: Optional[FitSettings] = None,
                   interval: float = 0.98, min_shrinkage: float = 0.5) -> Dict[str, Any]:
    """Whether a link's parameters (and the theory's) come back from simulated data.

    ``replicates`` independent datasets of ``n_trials`` responses are simulated at ``truth``
    (seeds ``seed``, ``seed + 1``, ...) and refitted. In one replicate a parameter comes back
    when its true value is inside its posterior's central ``interval`` and, for a link
    parameter, its posterior sd is at most ``min_shrinkage`` times its prior's (it is learned,
    not left to the prior). The link passes when every replicate converges and every parameter
    comes back in all replicates but at most one: a single dataset misses a 98% interval by
    chance often enough, over several parameters, to fail a sound link (one replicate of the
    allocation link put lapse 3 posterior sds off; the likelihood's profile peaks at the truth).
    """
    from src.rsa.loop.novelty import plain_pool

    theory = TheoryModel(Path(theory_path))
    missing = sorted((set(theory.params) | set(link.params)) - set(truth))
    if missing:
        raise ValueError(f"truth lacks {missing}")
    pool = list(contexts) if contexts is not None else plain_pool()
    runs = [_recovery_run(theory, link, truth, pool, n_trials, seed + r, settings, interval, min_shrinkage)
            for r in range(replicates)]
    names = list(runs[0]["params"])
    misses = {n: sum(not (run["params"][n]["inside"] and run["params"][n]["learned"]) for run in runs)
              for n in names}
    return {"link": link.name, "theory": theory.name, "n_trials": n_trials, "replicates": replicates,
            "converged": all(run["converged"] for run in runs),
            "convergence_problems": [p for run in runs for p in run["convergence_problems"]],
            "misses": misses, "recovered": all(m <= 1 for m in misses.values()), "runs": runs}
