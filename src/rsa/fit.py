"""Fitting memo models with numpyro NUTS.

The likelihood is the harness's, not the model's: each trial's choice is
Categorical(choice_probs(params, ctx)). Trials are evaluated per shape group
(`src.rsa.context.group_by_shape`), the groups' probabilities are padded with
zeros to the widest context (a padded object is never chosen) and observed as
one site, so the pointwise log-likelihood comes back in input trial order.

The result is an arviz ``InferenceData`` with a ``log_likelihood`` group, so
the repo's PSIS-LOO reliability verdict (`src.models.loo_reliability`), the
convergence gate (`src.models.pymc_inference.convergence_problems`) and
``az.compare`` apply unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import arviz as az
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, log_likelihood

from src.models.loo_reliability import LooDiagnostics, loo_diagnostics
from src.models.pymc_inference import convergence_problems
from src.rsa.context import Context, ShapeGroup, group_by_shape
from src.rsa.model_file import RSAModel, check_contract

OBSERVED_SITE = "choice"


@dataclass(frozen=True)
class FitSettings:
    num_warmup: int = 1000
    num_samples: int = 1000
    num_chains: int = 4
    target_accept: float = 0.9
    seed: int = 0


class ZeroProbabilityChoice(ValueError):
    """A model gives an observed choice probability zero at its prior draws."""


@dataclass
class RSAFit:
    model_name: str
    idata: Any
    param_names: List[str]
    convergence_problems: List[str] = field(default_factory=list)

    @property
    def converged(self) -> bool:
        return not self.convergence_problems

    def loo(self) -> LooDiagnostics:
        return loo_diagnostics(self.idata)


def _padded_probs(
    model: RSAModel, params: Mapping[str, Any], groups: Sequence[ShapeGroup], width: int
) -> jnp.ndarray:
    blocks = []
    for group in groups:
        p = model.group_probs(params, group)
        blocks.append(jnp.pad(p, ((0, 0), (0, width - p.shape[1]))))
    return jnp.concatenate(blocks, axis=0)


def _numpyro_model(model: RSAModel, groups: Sequence[ShapeGroup], width: int, choices=None):
    params = {name: numpyro.sample(name, prior) for name, prior in model.params.items()}
    probs = _padded_probs(model, params, groups, width)
    numpyro.sample(OBSERVED_SITE, dist.Categorical(probs=probs), obs=choices)


def prepare(
    contexts: Sequence[Context], choices: Sequence[int]
) -> Tuple[List[ShapeGroup], np.ndarray, np.ndarray, int]:
    """Shape groups, choices in group order, the permutation back to input order, width."""
    if len(contexts) != len(choices):
        raise ValueError(f"{len(contexts)} contexts but {len(choices)} choices")
    choices = np.asarray(choices, dtype=np.int32)
    for i, (ctx, c) in enumerate(zip(contexts, choices)):
        if not 0 <= c < ctx.shape[0]:
            raise ValueError(f"trial {i}: choice {c} is not an object index of its context")
    groups = list(group_by_shape(contexts).values())
    order = np.concatenate([g.indices for g in groups])
    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))
    width = max(g.shape[0] for g in groups)
    return groups, choices[order], inverse, width


def _check_possible(model, groups, grouped_choices, width, n_draws: int = 4) -> None:
    from src.rsa.model_file import prior_draws

    for draw in prior_draws(model.params, n_draws, seed=1):
        probs = np.asarray(_padded_probs(model, draw, groups, width))
        p_chosen = probs[np.arange(len(grouped_choices)), grouped_choices]
        if np.any(p_chosen <= 0):
            raise ZeroProbabilityChoice(
                f"{model.name} gives probability 0 to {(p_chosen <= 0).sum()} observed "
                f"choices at prior draw {draw}; a model must leave every object possible "
                f"(e.g. a lapse or noise process)"
            )


def fit(
    model: RSAModel,
    contexts: Sequence[Context],
    choices: Sequence[int],
    settings: FitSettings = FitSettings(),
) -> RSAFit:
    groups, grouped_choices, inverse, width = prepare(contexts, choices)
    check_contract(model, {g.shape: g for g in groups})
    _check_possible(model, groups, grouped_choices, width)
    kernel = NUTS(_numpyro_model, target_accept_prob=settings.target_accept)
    mcmc = MCMC(
        kernel,
        num_warmup=settings.num_warmup,
        num_samples=settings.num_samples,
        num_chains=settings.num_chains,
        chain_method="vectorized",
        progress_bar=False,
    )
    obs = jnp.asarray(grouped_choices)
    mcmc.run(
        jax.random.PRNGKey(settings.seed), model, groups, width, obs,
        extra_fields=("diverging",),
    )
    samples = mcmc.get_samples(group_by_chain=True)
    ll = log_likelihood(_numpyro_model, samples, model, groups, width, obs, batch_ndims=2)
    pointwise = np.asarray(ll[OBSERVED_SITE])[..., inverse]
    diverging = np.asarray(mcmc.get_extra_fields(group_by_chain=True)["diverging"])
    idata = az.from_dict(
        posterior={k: np.asarray(v) for k, v in samples.items()},
        log_likelihood={OBSERVED_SITE: pointwise},
        sample_stats={"diverging": diverging},
        observed_data={OBSERVED_SITE: np.asarray(choices, dtype=np.int32)},
    )
    names = list(model.params)
    return RSAFit(
        model_name=model.name,
        idata=idata,
        param_names=names,
        convergence_problems=convergence_problems(idata, names),
    )


def posterior_mean_probs(
    model: RSAModel, fitted: RSAFit, contexts: Sequence[Context], max_draws: int = 200
) -> List[np.ndarray]:
    """Posterior-mean choice probabilities per context (ragged: one array per trial)."""
    post = fitted.idata.posterior
    flat = {k: np.asarray(post[k]).reshape(-1) for k in fitted.param_names}
    n = len(next(iter(flat.values())))
    take = np.linspace(0, n - 1, min(max_draws, n)).astype(int)
    out: List[np.ndarray] = [None] * len(contexts)  # type: ignore[list-item]
    for group in group_by_shape(contexts).values():
        acc = 0.0
        for i in take:
            acc = acc + np.asarray(model.group_probs({k: v[i] for k, v in flat.items()}, group))
        mean = acc / len(take)
        for row, idx in enumerate(group.indices):
            out[idx] = mean[row]
    return out


def compare(fits: Mapping[str, RSAFit]) -> Any:
    """``az.compare`` (PSIS-LOO) over fits keyed by model name."""
    return az.compare({name: f.idata for name, f in fits.items()}, ic="loo")
