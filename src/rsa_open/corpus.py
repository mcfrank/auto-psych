"""A theory fitted to many paradigms at once (docs/auto_open/PLAN.md §5, Phase 2b).

A chain's data are **blocks**: the literature data the closed campaign started from (E, forced
choice) and each experiment's paradigm. Every block keeps its frozen link (`src.rsa_open.links`).
A theory has one set of parameters for every block; each block has its own copy of its link's
parameters (``temperature@exp1``), since a response process belongs to a paradigm, not to a
theory. So:

- `fit_corpus` samples the theory's parameters once and each block's link parameters, with
  the closed campaign's NUTS settings, and keeps every trial's pointwise log-likelihood;
- `heldout_lpd` scores trials a fit did not see: the log of the posterior-mean probability of
  each response, as the closed loop's `src.rsa.evaluate_heldout.heldout_lpd`;
- `assign_folds` and `cv_corpus` are the closed loop's grouped cross-validation made per block:
  whole units (E's display units, `src.rsa.split.unit_keys`; a paradigm's design units) go to
  ``k`` folds balanced within each block, and fold f's fit trains on every block without its
  fold-f trials, so a candidate costs ``k`` fits whatever the number of blocks;
- `standing` ranks theories on the newest block and applies the accountability rule (PLAN §5):
  a theory is **eligible** only when, on every earlier block, it is within ``guard_k`` clustered
  standard errors of that block's best theory. The closed loop's guard pooled every earlier
  trial into one comparison; here each paradigm, E included, is guarded on its own.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import arviz as az
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
from numpyro.infer import MCMC, NUTS

from src.models.pymc_inference import convergence_problems
from src.rsa.context import Context
from src.rsa.fit import FitSettings
from src.rsa_open.fit import OBSERVED_SITE, OpenData, TheoryModel, _latent, _split, pointwise_log_lik
from src.rsa_open.links import LinkLikelihood

GUARD_K = 4.0  # PI 2026-10-21: no more than 4 clustered SEs worse on any earlier paradigm
DEFAULT_FOLDS = 5
MAX_HELDOUT_DRAWS = 400  # as the closed loop's held-out score


@dataclass(frozen=True)
class Block:
    """One paradigm's trials under its frozen link. ``units`` are the cross-validation and
    standard-error units (one per trial), ``sources`` the data sources within the block, ``fixed``
    the link parameters held at a value instead of fitted (the literature's temperature,
    `literature_block`)."""

    name: str
    link: LinkLikelihood
    contexts: Tuple[Context, ...]
    responses: Tuple[Mapping[str, Any], ...]
    units: Tuple[str, ...]
    sources: Tuple[str, ...] = ()
    fixed: Tuple[Tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        n = len(self.contexts)
        if not n:
            raise ValueError(f"block {self.name}: no trials")
        if "@" in self.name:
            raise ValueError(f"block {self.name}: '@' names a block's link parameters")
        if len(self.responses) != n or len(self.units) != n:
            raise ValueError(f"block {self.name}: {n} scenes, {len(self.responses)} responses, {len(self.units)} units")
        if self.sources and len(self.sources) != n:
            raise ValueError(f"block {self.name}: {len(self.sources)} sources for {n} trials")
        for k, _ in self.fixed:
            if k not in self.link.params:
                raise ValueError(f"block {self.name}: fixed {k!r} is not a parameter of the {self.link.name} link")

    @property
    def fixed_params(self) -> Dict[str, float]:
        return {k: float(v) for k, v in self.fixed}

    @property
    def free_link_params(self) -> Dict[str, Any]:
        """The link parameters fitted for this block (name -> prior)."""
        return {k: d for k, d in self.link.params.items() if k not in self.fixed_params}

    @property
    def n(self) -> int:
        return len(self.contexts)

    @cached_property
    def data(self) -> OpenData:
        return OpenData.build(self.contexts, self.responses, self.link)

    def param(self, name: str) -> str:
        """This block's copy of a link parameter."""
        return f"{name}@{self.name}"

    def subset(self, rows: Sequence[int]) -> "Block":
        rows = list(rows)
        return Block(self.name, self.link, tuple(self.contexts[i] for i in rows),
                     tuple(self.responses[i] for i in rows), tuple(self.units[i] for i in rows),
                     tuple(self.sources[i] for i in rows) if self.sources else (), self.fixed)


def check_blocks(blocks: Sequence[Block]) -> None:
    names = [b.name for b in blocks]
    if len(set(names)) != len(names):
        raise ValueError(f"block names repeat: {names}")
    if not blocks:
        raise ValueError("no blocks")


# E is the reference block (PI 2026-10-22): its link's temperature is 1, so on E every theory is its
# closed original (the port equals it at temperature 1) and each new paradigm's temperature is
# relative to E. Fitted on E alone, a free temperature traded off against a theory's own sharpness
# (rsa_l1_shared_prior: alpha 0.15 at temperature 2.5, against 1.4 at 1) and one fit stalled for
# hours; a theory in which people sharpen or soften their beliefs into a choice says so with a
# parameter of its own, which every paradigm then tests.
LITERATURE_FIXED = (("temperature", 1.0),)


def literature_block(path=None, name: str = "E") -> Block:
    """The literature's forced-choice trials (the closed loader) as a choice-link block, its
    temperature fixed (`LITERATURE_FIXED`)."""
    from src.rsa.dataset import DEFAULT_TRIALS_CSV, load_forced_choice
    from src.rsa.split import unit_keys
    from src.rsa_open.links import likelihood

    trials = load_forced_choice(path or DEFAULT_TRIALS_CSV)
    frame = trials.frame
    sources = frame["source"].astype(str) if "source" in frame.columns else ["pragmods"] * len(frame)
    return Block(name, likelihood("choice"), tuple(trials.contexts), tuple({"choice": int(c)} for c in trials.choices),
                 tuple(unit_keys(frame).astype(str)), tuple(str(s) for s in sources), LITERATURE_FIXED)


def paradigm_block(name: str, paradigm, rows: Sequence[Mapping[str, Any]]) -> Block:
    """An experiment's test trials (`src.rsa_open.screens.page_rows`) under its paradigm's link;
    a trial's unit is the design unit it showed."""
    from src.rsa_open.paradigm import paradigm_likelihood
    from src.rsa_open.screens import scenes_and_responses

    tests = [r for r in rows if r.get("phase") == "test"]
    contexts, responses = scenes_and_responses(tests)
    return Block(name, paradigm_likelihood(paradigm), tuple(contexts), tuple(responses),
                 tuple(f"unit{r['unit']}" for r in tests), tuple(name for _ in tests))


# --------------------------------------------------------------------------- the joint fit

def _corpus_model(theory: TheoryModel, blocks: Sequence[Block]):
    params = {name: numpyro.sample(name, prior) for name, prior in theory.params.items()}
    total = 0.0
    for block in blocks:
        link = block.link
        fixed = block.fixed_params
        lp = {k: jnp.asarray(fixed[k]) if k in fixed else numpyro.sample(block.param(k), prior)
              for k, prior in link.params.items()}
        tp = {k: params[k] for k in theory.params}
        for group, enc in zip(block.data.groups, block.data.encoded):
            total = total + jnp.sum(link.log_lik(_latent(theory, link, tp, group), lp, enc, group.arrays,
                                                 jnp.asarray(group.classes)))
    numpyro.factor(OBSERVED_SITE, total)


def block_draws(flat: Mapping[str, np.ndarray], theory: TheoryModel, block: Block) -> Dict[str, np.ndarray]:
    """One block's view of flat posterior draws: the theory's parameters and its link's, unsuffixed."""
    out = {k: flat[k] for k in theory.params}
    n = len(next(iter(flat.values())))
    fixed = block.fixed_params
    out.update({k: np.full(n, fixed[k]) if k in fixed else flat[block.param(k)] for k in block.link.params})
    return out


@dataclass
class CorpusFit:
    theory_name: str
    blocks: List[str]
    idata: Any
    param_names: List[str]
    convergence_problems: List[str]
    offsets: Dict[str, Tuple[int, int]] = field(default_factory=dict)  # block -> (start, stop) in log_likelihood
    timing: Dict[str, float] = field(default_factory=dict)
    """warmup_sec (compilation included), sampling_sec, and the leapfrog steps of each phase
    (per iteration, the most any chain took); empty for a fit recorded before they were kept."""

    @property
    def converged(self) -> bool:
        return not self.convergence_problems

    @property
    def diagnostics(self):
        """Convergence numbers (`src.models.pymc_inference.ConvergenceDiagnostics`) over the free parameters."""
        from src.models.pymc_inference import convergence_diagnostics

        return convergence_diagnostics(self.idata, self.param_names)

    @property
    def near_miss(self) -> bool:
        """A failed fit close enough to the gate that a refit at smaller steps may pass (main's rule)."""
        from src.models.pymc_inference import is_near_miss

        return not self.converged and is_near_miss(self.diagnostics)

    def sampler_report(self):
        """Steps per draw, step size and correlated parameters (`src.rsa_open.sampler`), or None."""
        from src.rsa_open.sampler import report_of

        return report_of(self.idata, self.param_names)

    def flat(self) -> Dict[str, np.ndarray]:
        post = self.idata.posterior
        return {k: np.asarray(post[k]).reshape(-1) for k in post.data_vars}

    def pointwise(self, block: str) -> np.ndarray:
        """(draws, trials) log-likelihood of one block's trials."""
        start, stop = self.offsets[block]
        ll = np.asarray(self.idata.log_likelihood[OBSERVED_SITE])
        return ll.reshape(-1, ll.shape[-1])[:, start:stop]


def fit_corpus(theory: TheoryModel, blocks: Sequence[Block],
               settings: FitSettings = FitSettings(dense_mass=True),
               init: Optional[Mapping[str, float]] = None,
               on_warmup: Optional[Callable[[Dict[str, Any]], None]] = None) -> CorpusFit:
    """NUTS on the corpus; with ``init`` (constrained values of sampled sites) every chain starts
    there instead of at random. Warmup and sampling run as two calls (sampling reuses warmup's
    compiled kernel), each timed with its leapfrog steps; ``on_warmup`` receives the warmup's
    `src.rsa_open.sampler.warmup_report` (as JSON) and timing once warmup ends, so a fit killed
    while sampling still says what its geometry was."""
    check_blocks(blocks)
    clash = sorted(set(theory.params) & {b.param(k) for b in blocks for k in b.link.params})
    if clash:
        raise ValueError(f"theory parameters {clash} collide with block link parameters")
    from numpyro.infer import init_to_uniform, init_to_value

    strategy = init_to_value(values={k: float(v) for k, v in init.items()}) if init else init_to_uniform
    kernel = NUTS(_corpus_model, target_accept_prob=settings.target_accept, dense_mass=settings.dense_mass,
                  init_strategy=strategy)
    mcmc = MCMC(kernel, num_warmup=settings.num_warmup, num_samples=settings.num_samples,
                num_chains=settings.num_chains, chain_method="vectorized", progress_bar=False)
    names = [*theory.params, *(b.param(k) for b in blocks for k in b.free_link_params)]
    t0 = time.perf_counter()
    mcmc.warmup(jax.random.PRNGKey(settings.seed), theory, list(blocks), collect_warmup=True,
                extra_fields=("num_steps", "adapt_state.step_size"))
    warm = mcmc.get_extra_fields(group_by_chain=True)
    warm_steps = np.asarray(warm["num_steps"])
    timing = dict(warmup_sec=time.perf_counter() - t0, warmup_steps=float(warm_steps.max(axis=0).sum()))
    if on_warmup is not None:
        from src.rsa_open.sampler import warmup_report

        warm_samples = {k: np.asarray(v) for k, v in mcmc.get_samples(group_by_chain=True).items()}
        on_warmup(dict(report=warmup_report(warm_samples, warm_steps, np.asarray(warm["adapt_state.step_size"]),
                                            names).to_json(), timing=timing))
    t1 = time.perf_counter()
    mcmc.run(mcmc.post_warmup_state.rng_key, theory, list(blocks),
             extra_fields=("diverging", "num_steps", "adapt_state.step_size"))
    extra = mcmc.get_extra_fields(group_by_chain=True)
    n_steps = np.asarray(extra["num_steps"])
    timing.update(sampling_sec=time.perf_counter() - t1, sampling_steps=float(n_steps.max(axis=0).sum()))
    samples = {k: np.asarray(v) for k, v in mcmc.get_samples(group_by_chain=True).items()}
    chains, draws = next(iter(samples.values())).shape[:2]
    flat = {k: v.reshape(chains * draws) for k, v in samples.items()}
    lls, offsets, start = [], {}, 0
    for block in blocks:
        lls.append(pointwise_log_lik(theory, block.link, block.data, block_draws(flat, theory, block)))
        offsets[block.name] = (start, start + block.n)
        start += block.n
    ll = np.concatenate(lls, axis=1)
    idata = az.from_dict(
        posterior=samples, log_likelihood={OBSERVED_SITE: ll.reshape(chains, draws, -1)},
        sample_stats={"diverging": np.asarray(extra["diverging"]), "n_steps": n_steps,
                      "step_size": np.asarray(extra["adapt_state.step_size"])})
    return CorpusFit(theory.name, [b.name for b in blocks], idata, names, convergence_problems(idata, names), offsets,
                     timing)


def heldout_lpd(theory: TheoryModel, fitted: CorpusFit, block: Block,
                max_draws: int = MAX_HELDOUT_DRAWS) -> np.ndarray:
    """(trials,) log posterior-mean probability of each of ``block``'s responses under a fit that
    did not see them; the block's link parameters are the fit's copy for that block."""
    if block.name not in fitted.blocks:
        raise ValueError(f"the fit has no link parameters for block {block.name} ({fitted.blocks})")
    flat = fitted.flat()
    n = len(next(iter(flat.values())))
    take = np.unique(np.linspace(0, n - 1, min(n, max_draws)).round().astype(int))
    draws = {k: v[take] for k, v in block_draws(flat, theory, block).items()}
    ll = pointwise_log_lik(theory, block.link, block.data, draws)
    return np.asarray(jax.scipy.special.logsumexp(ll, axis=0) - np.log(ll.shape[0]))


# --------------------------------------------------------------------------- cross-validation

def assign_folds(block: Block, k: int = DEFAULT_FOLDS, seed: int = 0) -> np.ndarray:
    """Fold of every trial of a block: whole units, balanced in trials within each source (the
    closed loop's rule, `src.rsa.loop.cv.assign_folds`)."""
    if k < 2:
        raise ValueError(f"grouped CV needs at least 2 folds, not {k}")
    units = np.asarray(block.units)
    sources = np.asarray(block.sources) if block.sources else np.full(block.n, block.name)
    rng = np.random.default_rng([seed, sum(block.name.encode())])
    fold_of_unit: Dict[str, int] = {}
    for src in sorted(set(sources)):
        names, sizes = np.unique(units[sources == src], return_counts=True)
        if len(names) < k:
            raise ValueError(f"block {block.name}, source {src}: {len(names)} units, fewer than the {k} folds")
        order = rng.permutation(len(names))
        load = np.zeros(k)
        for i in sorted(order, key=lambda i: -sizes[i]):  # largest first into the lightest fold
            f = int(np.argmin(load))
            fold_of_unit[names[i]] = f
            load[f] += sizes[i]
    return np.asarray([fold_of_unit[u] for u in units], dtype=int)


@dataclass
class CorpusCV:
    pointwise: Dict[str, np.ndarray]  # block -> (trials,) out-of-fold lpd
    converged: bool

    def elpd(self, block: str) -> float:
        return float(self.pointwise[block].sum())


def cv_corpus(theory: TheoryModel, blocks: Sequence[Block], folds: Mapping[str, np.ndarray], k: int,
              fit_fn=fit_corpus, settings: Optional[FitSettings] = None) -> CorpusCV:
    """Out-of-fold lpd of every trial of every block: fold f's fit trains on every block without
    its fold-f trials. ``fit_fn(theory, blocks, settings)`` is the fit (cached in the loop)."""
    check_blocks(blocks)
    out = {b.name: np.empty(b.n) for b in blocks}
    converged = True
    for f in range(k):
        train = [b.subset(np.where(folds[b.name] != f)[0]) for b in blocks]
        fitted = fit_fn(theory, train, settings) if settings is not None else fit_fn(theory, train)
        converged = converged and fitted.converged
        for b in blocks:
            rows = np.where(folds[b.name] == f)[0]
            if len(rows):
                out[b.name][rows] = heldout_lpd(theory, fitted, b.subset(rows))
    return CorpusCV(out, converged)


def standing(results: Mapping[str, CorpusCV], blocks: Sequence[Block], *, guard_k: float = GUARD_K) -> Dict[str, dict]:
    """Each theory's standing: ELPD-CV on the newest block (the last) and its gap to the best with
    the unit-clustered SE; on every earlier block its gap to that block's best and SE; and
    ``eligible``: within ``guard_k`` SEs of the best on every earlier block."""
    from src.models.clustered_se import cluster_dse

    check_blocks(blocks)
    if not results:
        raise ValueError("no theories to rank")
    out: Dict[str, dict] = {name: {"by_block": {}} for name in results}
    for i, block in enumerate(blocks):
        units = np.unique(np.asarray(block.units), return_inverse=True)[1]
        elpd = {n: r.elpd(block.name) for n, r in results.items()}
        top = max(elpd, key=elpd.get)
        for name, r in results.items():
            dse = 0.0 if name == top else float(cluster_dse(results[top].pointwise[block.name],
                                                            r.pointwise[block.name], units))
            out[name]["by_block"][block.name] = {"elpd_cv": elpd[name], "behind": elpd[top] - elpd[name], "dse": dse}
    newest = blocks[-1].name
    for name, o in out.items():
        new = o["by_block"][newest]
        o.update(sel_elpd=new["elpd_cv"], sel_diff=new["behind"], sel_dse=new["dse"],
                 cv_converged=results[name].converged)
        earlier = {b.name: o["by_block"][b.name] for b in blocks[:-1]}
        o["guard_failures"] = sorted(b for b, s in earlier.items() if s["behind"] > guard_k * s["dse"])
        o["eligible"] = not o["guard_failures"]
    return out
