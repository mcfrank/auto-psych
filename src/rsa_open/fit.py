"""Fitting a theory under a link to any paradigm's responses (docs/auto_open/PLAN.md §3).

The closed campaign's fit (`src.rsa.fit`) knows one likelihood, a choice among objects. Here
the likelihood is the link's (`src.rsa_open.links.LinkLikelihood`): the theory gives each
trial's latent posterior over the objects of its scene, and the link scores the response. The
closed fit is left untouched, so the closed campaign's frozen results cannot move; under the
choice link the two give the same per-trial log-likelihood (tests/test_rsa_open_fit.py).

- `TheoryModel` loads a theory file (``PARAMS``, ``listener``, optionally ``speaker``) with the
  closed loader. Every theory has a speaker (PI 2026-10-21): its own, or the one derived from
  its listener (`TheoryModel.group_speaker`).
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
    """A theory file, loaded like a model file but evaluated through ``listener``.

    Its speaker (``group_speaker``) is the file's own ``speaker(params, ctx)``, an
    (N_UTT, N_OBJ) array whose column r is P(message | target r), when it defines one; else the
    speaker derived from its listener:

        speaker(u | r) ∝ listener(r | u)   over the scene's real messages (never the sink),

    the speaker who expects this theory's listener. It has no parameter of its own: a speaker
    link's temperature is its rationality. When no real message is true of r for this listener
    (every listener(r | u) is 0), the speaker is indifferent: uniform over the real messages.
    """

    def _choice_probs(self, shape: Tuple[int, int]):
        fn = getattr(self.module(shape), LISTENER, None)
        if not callable(fn):
            raise ModelContractViolation(
                f"{self.path.name} must define {LISTENER}(params, ctx) returning the listener's "
                f"posterior over the objects, shape (N_OBJ,)")
        return fn

    @property
    def params(self) -> Dict[str, Any]:
        """The theory's priors. Unlike a model file's, they may be empty: a theory with no free
        parameters (the literal listener, once its lapse went to the link) is a fixed prediction."""
        from collections.abc import Mapping as _Mapping

        from numpyro.distributions import Distribution

        params = getattr(self._any_module(), "PARAMS", None)
        if not isinstance(params, _Mapping):
            raise ModelContractViolation(f"{self.path.name} must define PARAMS, a dict of parameter "
                                         f"name -> numpyro distribution (it may be empty)")
        bad = [k for k, v in params.items() if not isinstance(v, Distribution)]
        if bad:
            raise ModelContractViolation(f"PARAMS entries {bad} in {self.path.name} are not numpyro distributions")
        return dict(params)

    def latent_draws(self, params: Mapping[str, Any], group: ShapeGroup, speaker: bool, n_draws: int) -> jnp.ndarray:
        """(n_draws, trials, ...) listener posteriors or speakers at a batch of parameter draws; a
        theory without parameters gives the same prediction at every draw."""
        if not self.params:
            one = (self.group_speaker if speaker else self.group_probs)({}, group)
            return jnp.broadcast_to(one[None], (n_draws, *one.shape))
        return (self.draws_speaker if speaker else self.draws_probs)(params, group)

    def has_own_speaker(self) -> bool:
        return callable(getattr(self._any_module(), "speaker", None))

    def _speaker_fn(self, shape: Tuple[int, int]):
        own = getattr(self.module(shape), "speaker", None)
        if callable(own):
            return own
        listener = self._choice_probs(shape)

        def derived(params, ctx):
            heard = lambda u: listener(params, ctx._replace(utterance=u, is_prior=jnp.zeros_like(ctx.is_prior)))  # noqa: E731
            table = jax.vmap(heard)(jnp.arange(ctx.lex.shape[0], dtype=ctx.utterance.dtype))  # (N_UTT, N_OBJ)
            real = (1.0 - ctx.is_sink)[:, None]
            weights = table * real
            total = weights.sum(axis=0, keepdims=True)
            indifferent = jnp.broadcast_to(real / real.sum(), weights.shape)
            return jnp.where(total > 0, weights / jnp.where(total > 0, total, 1.0), indifferent)

        return derived

    def group_speaker(self, params: Mapping[str, Any], group: ShapeGroup) -> jnp.ndarray:
        """(trials, N_UTT, N_OBJ) speaker distributions for one shape group."""
        key = ("speaker", group.shape)
        if key not in self._jitted:
            fn = self._speaker_fn(group.shape)
            self._jitted[key] = jax.jit(lambda p, arrays: jax.vmap(lambda ctx: fn(p, ctx))(arrays))
        return self._jitted[key](dict(params), group.arrays)

    def draws_speaker(self, params: Mapping[str, Any], group: ShapeGroup) -> jnp.ndarray:
        """(draws, trials, N_UTT, N_OBJ) for a batch of parameter draws."""
        key = ("speaker draws", group.shape)
        if key not in self._jitted:
            fn = self._speaker_fn(group.shape)
            per_draw = lambda p, arrays: jax.vmap(lambda ctx: fn(p, ctx))(arrays)  # noqa: E731
            self._jitted[key] = jax.jit(jax.vmap(per_draw, in_axes=(0, None)))
        return self._jitted[key](dict(params), group.arrays)


def _latent(theory: TheoryModel, link: LinkLikelihood, params, group: ShapeGroup, draws: bool = False,
            n_draws: int = 0):
    if draws:
        return theory.latent_draws(params, group, link.side == "speaker", n_draws)
    return (theory.group_speaker if link.side == "speaker" else theory.group_probs)(params, group)


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
        ll = link.log_lik(_latent(theory, link, tp, group), lp, enc, group.arrays, jnp.asarray(group.classes))
        out = out.at[jnp.asarray(group.indices)].set(ll)
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
        total = total + jnp.sum(link.log_lik(_latent(theory, link, tp, group), lp, enc, group.arrays,
                                             jnp.asarray(group.classes)))
    numpyro.factor(OBSERVED_SITE, total)


def pointwise_log_lik(theory: RSAModel, link: LinkLikelihood, data: OpenData,
                      flat: Mapping[str, np.ndarray]) -> np.ndarray:
    """(draws, trials) log-likelihood of each response at each posterior draw."""
    n_draws = len(next(iter(flat.values())))
    out = np.empty((n_draws, data.n_trials), dtype=np.float32)
    for group, enc in zip(data.groups, data.encoded):
        classes = jnp.asarray(group.classes)
        per_draw = jax.vmap(lambda latent, lp: link.log_lik(latent, lp, enc, group.arrays, classes))
        start = 0
        for n, params in draw_batches(flat, np.arange(n_draws)):
            tp, lp = _split(theory, link, params)
            batch = len(next(iter(params.values())))
            latent = _latent(theory, link, tp, group, True, batch)
            out[start:start + n, group.indices] = np.asarray(per_draw(latent, lp))[:n]
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
             params: Mapping[str, Any], key, probes: Optional[Sequence[int]] = None,
             utterances: Optional[Sequence[int]] = None) -> List[dict]:
    """One simulated response per scene, in input order. ``probes`` (the probed object of a
    rating, the target of a speaker link) default to a random object per trial; a judgment's
    message (``utterances``) to a random real message of the scene."""
    tp, lp = _split(theory, link, {k: jnp.asarray(np.asarray(v).reshape(())) for k, v in params.items()})
    rng = np.random.default_rng(np.asarray(jax.random.randint(key, (), 0, 2**31 - 1)).item())
    if probes is None:
        probes = [int(rng.integers(ctx.shape[0])) for ctx in contexts]
    if utterances is None:
        utterances = [int(rng.integers(ctx.shape[1] - (1 if ctx.needs_sink else 0))) for ctx in contexts]
    utterances = np.asarray(utterances)
    out: List[Optional[dict]] = [None] * len(contexts)
    for k, group in zip(jax.random.split(key, len(contexts) or 1), group_by_shape(contexts).values()):
        latent = _latent(theory, link, tp, group)
        responses = link.sample(k, latent, lp, np.asarray(probes)[group.indices], group.arrays,
                                utterances[group.indices])
        for i, response in zip(group.indices, responses):
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
