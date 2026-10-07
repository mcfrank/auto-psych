"""Fitting memo models with numpyro NUTS.

The likelihood is the harness's, not the model's: each trial's choice is
Categorical(choice_probs(params, ctx)). A choice is observed as its *class*
of identical objects (`src.rsa.context.Context.choice_classes`): no model can
tell identical objects apart, and the pragmods data often cannot say which
copy was chosen, so the class's summed probability is the likelihood.

Many trials show the same display, and many give the same answer to it.
The model is evaluated once per distinct display (`unique_contexts`, per
shape group), and the likelihood is a sum over *patterns* (display, chosen
class) weighted by how many trials have each: on the combined data, 40k
trials are ~3k displays. The pointwise log-likelihood stored in the
``InferenceData`` is per pattern too (``constant_data.trial_pattern`` maps
each trial, in input order, to its pattern), and `RSAFit.loo` expands
PSIS-LOO to trials: trials with one pattern have identical log-likelihood
draws, so their PSIS-LOO terms and Pareto k are identical — the expansion is
exact, not an approximation. Storing every trial's draws (4000 x 40k) made a
fit ~1 GB in memory and on disk, and az.loo several times that.

Posterior predictions over many draws are computed in batches of draws
(`DRAW_CHUNK`), vectorised over the batch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Sequence

import arviz as az
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

from src.models.loo_reliability import LooDiagnostics
from src.models.pymc_inference import convergence_problems
from src.rsa.context import Context, ShapeGroup, group_by_shape, unique_contexts
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

    _loo: Any = field(default=None, repr=False, compare=False)

    def loo(self) -> LooDiagnostics:
        """Trial-level PSIS-LOO and its reliability verdict (computed once)."""
        if self._loo is None:
            from src.rsa.loo import trial_loo

            self._loo = trial_loo(self.idata)
        return self._loo


def class_probs(probs, classes) -> jnp.ndarray:
    """Collapse (..., trials, N_OBJ) probabilities onto choice classes.

    Each class's summed probability sits at its first object's index; the
    other members of a class get zero.
    """
    onehot = jax.nn.one_hot(jnp.asarray(classes), probs.shape[-1], dtype=probs.dtype)
    return jnp.einsum("...tr,trs->...ts", probs, onehot)


def _padded_probs(
    model: RSAModel, params: Mapping[str, Any], groups: Sequence[ShapeGroup], width: int
) -> jnp.ndarray:
    blocks = []
    for group in groups:
        p = class_probs(model.group_probs(params, group), group.classes)
        blocks.append(jnp.pad(p, ((0, 0), (0, width - p.shape[1]))))
    return jnp.concatenate(blocks, axis=0)


@dataclass(frozen=True)
class Prepared:
    """Trials reduced to distinct displays and (display, chosen class) patterns."""

    groups: List[ShapeGroup]  # shape groups over the distinct displays
    width: int  # the widest display's object count
    pattern_row: np.ndarray  # (P,) the pattern's display: its row in the groups' concatenation
    pattern_class: np.ndarray  # (P,) the chosen class
    pattern_count: np.ndarray  # (P,) trials with this pattern
    trial_pattern: np.ndarray  # (T,) each trial's pattern, in input order

    @property
    def n_trials(self) -> int:
        return len(self.trial_pattern)


def _numpyro_model(model: RSAModel, prep: Prepared):
    params = {name: numpyro.sample(name, prior) for name, prior in model.params.items()}
    logp = _safe_logits(_padded_probs(model, params, prep.groups, prep.width))
    ll = logp[prep.pattern_row, prep.pattern_class]
    numpyro.factor(OBSERVED_SITE, jnp.sum(jnp.asarray(prep.pattern_count, dtype=ll.dtype) * ll))


def _safe_logits(probs: jnp.ndarray) -> jnp.ndarray:
    """log(probs), -inf at zeros, with a finite gradient.

    Zeros (padding, non-first members of a choice class) depend on the
    parameters through the class sum; log(0)'s infinite derivative times a
    zero cotangent would be NaN, so log is never evaluated at zero.
    """
    positive = probs > 0
    return jnp.where(positive, jnp.log(jnp.where(positive, probs, 1.0)), -jnp.inf)


def prepare(contexts: Sequence[Context], choices: Sequence[int]) -> Prepared:
    if len(contexts) != len(choices):
        raise ValueError(f"{len(contexts)} contexts but {len(choices)} choices")
    choices = np.asarray(choices, dtype=np.int64)
    for i, (ctx, c) in enumerate(zip(contexts, choices)):
        if not 0 <= c < ctx.shape[0]:
            raise ValueError(f"trial {i}: choice {c} is not an object index of its context")
    uniques, inverse = unique_contexts(contexts)
    groups = list(group_by_shape(uniques).values())
    row_of_unique = np.empty(len(uniques), dtype=np.int64)
    offset = 0
    for g in groups:
        row_of_unique[g.indices] = offset + np.arange(len(g.indices))
        offset += len(g.indices)
    classes = [u.choice_classes() for u in uniques]
    trial_row = row_of_unique[inverse]
    trial_class = np.asarray([classes[u][c] for u, c in zip(inverse, choices)], dtype=np.int64)
    keys, trial_pattern, counts = np.unique(
        np.stack([trial_row, trial_class], axis=1), axis=0, return_inverse=True, return_counts=True
    )
    return Prepared(
        groups=groups,
        width=max(g.shape[0] for g in groups),
        pattern_row=keys[:, 0],
        pattern_class=keys[:, 1],
        pattern_count=counts,
        trial_pattern=np.asarray(trial_pattern).reshape(-1),
    )


def _check_possible(model, prep: Prepared, n_draws: int = 4) -> None:
    from src.rsa.model_file import prior_draws

    for draw in prior_draws(model.params, n_draws, seed=1):
        probs = np.asarray(_padded_probs(model, draw, prep.groups, prep.width))
        p_chosen = probs[prep.pattern_row, prep.pattern_class]
        if np.any(p_chosen <= 0):
            n_bad = int(prep.pattern_count[p_chosen <= 0].sum())
            raise ZeroProbabilityChoice(
                f"{model.name} gives probability 0 to {n_bad} observed "
                f"choices at prior draw {draw}; a model must leave every object possible "
                f"(e.g. a lapse or noise process)"
            )


DRAW_CHUNK = 64  # posterior draws evaluated together (memory ~ chunk x displays x the model's arrays)


def draw_batches(flat: Mapping[str, np.ndarray], take: np.ndarray, chunk: int = DRAW_CHUNK):
    """(n_real, params) batches of the draws ``take``; the last is padded to a
    fixed size so each shape compiles once."""
    chunk = max(1, min(chunk, len(take)))
    for start in range(0, len(take), chunk):
        idx = take[start:start + chunk]
        n = len(idx)
        if n < chunk:
            idx = np.concatenate([idx, np.repeat(idx[-1], chunk - n)])
        yield n, {k: jnp.asarray(v[idx]) for k, v in flat.items()}


def posterior_flat(fitted: "RSAFit") -> Dict[str, np.ndarray]:
    post = fitted.idata.posterior
    return {k: np.asarray(post[k]).reshape(-1) for k in fitted.param_names}


def pattern_log_lik(model: RSAModel, prep: Prepared, flat: Mapping[str, np.ndarray],
                    chunk: int = DRAW_CHUNK) -> np.ndarray:
    """(draws, patterns) log-likelihood of each pattern's choice at each draw."""
    n_draws = len(next(iter(flat.values())))
    out = np.empty((n_draws, len(prep.pattern_row)), dtype=np.float32)
    offset = 0
    for g in prep.groups:
        rows = np.arange(offset, offset + len(g.indices))
        offset += len(g.indices)
        mine = np.where((prep.pattern_row >= rows[0]) & (prep.pattern_row <= rows[-1]))[0]
        if not len(mine):
            continue
        local, cls = prep.pattern_row[mine] - rows[0], prep.pattern_class[mine]
        start = 0
        for n, params in draw_batches(flat, np.arange(n_draws), chunk):
            p = class_probs(model.draws_probs(params, g), g.classes)[:n]
            out[start:start + n, mine] = np.asarray(_safe_logits(p[:, local, cls]))
            start += n
    return out


def fit(
    model: RSAModel,
    contexts: Sequence[Context],
    choices: Sequence[int],
    settings: FitSettings = FitSettings(),
) -> RSAFit:
    prep = prepare(contexts, choices)
    check_contract(model, {g.shape: g for g in prep.groups})
    _check_possible(model, prep)
    kernel = NUTS(_numpyro_model, target_accept_prob=settings.target_accept)
    mcmc = MCMC(
        kernel,
        num_warmup=settings.num_warmup,
        num_samples=settings.num_samples,
        num_chains=settings.num_chains,
        chain_method="vectorized",
        progress_bar=False,
    )
    mcmc.run(jax.random.PRNGKey(settings.seed), model, prep, extra_fields=("diverging",))
    samples = {k: np.asarray(v) for k, v in mcmc.get_samples(group_by_chain=True).items()}
    chains, draws = next(iter(samples.values())).shape[:2]
    ll = pattern_log_lik(model, prep, {k: v.reshape(chains * draws) for k, v in samples.items()})
    diverging = np.asarray(mcmc.get_extra_fields(group_by_chain=True)["diverging"])
    idata = az.from_dict(
        posterior=samples,
        log_likelihood={OBSERVED_SITE: ll.reshape(chains, draws, -1)},
        sample_stats={"diverging": diverging},
        constant_data={"trial_pattern": prep.trial_pattern, "pattern_count": prep.pattern_count},
    )
    names = list(model.params)
    return RSAFit(
        model_name=model.name,
        idata=idata,
        param_names=names,
        convergence_problems=convergence_problems(idata, names),
    )


def mean_probs(model: RSAModel, flat: Mapping[str, np.ndarray], contexts: Sequence[Context],
               max_draws: int, *, by_class: bool, chunk: int = DRAW_CHUNK) -> List[np.ndarray]:
    """Posterior-mean probabilities per context (objects, or classes with
    ``by_class``), over ``max_draws`` evenly spaced draws; one array per input."""
    n = len(next(iter(flat.values())))
    take = np.linspace(0, n - 1, min(max_draws, n)).astype(int)
    uniques, inverse = unique_contexts(contexts)
    means: List[np.ndarray] = [None] * len(uniques)  # type: ignore[list-item]
    for group in group_by_shape(uniques).values():
        acc = 0.0
        for k, params in draw_batches(flat, take, chunk):
            p = model.draws_probs(params, group)
            if by_class:
                p = class_probs(p, group.classes)
            acc = acc + np.asarray(p[:k], dtype=np.float64).sum(axis=0)
        mean = acc / len(take)
        for row, idx in enumerate(group.indices):
            means[idx] = mean[row]
    return [means[u] for u in inverse]


def posterior_mean_probs(
    model: RSAModel, fitted: RSAFit, contexts: Sequence[Context], max_draws: int = 200
) -> List[np.ndarray]:
    """Posterior-mean choice probabilities per context (ragged: one array per trial)."""
    return mean_probs(model, posterior_flat(fitted), contexts, max_draws, by_class=False)


def compare(fits: Mapping[str, RSAFit]) -> Any:
    """``az.compare`` (PSIS-LOO, trial-level) over fits keyed by model name."""
    return az.compare({name: f.loo().loo for name, f in fits.items()}, ic="loo")
