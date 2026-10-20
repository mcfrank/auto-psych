"""Leave-one-participant-out ELPD by PSIS over people, person-level parameters integrated out.

The loop scores models by trial-level PSIS-LOO, which predicts a held-out
trial of a person whose other trials are in the fit. This scores each person's
whole set of responses as if they were new, from the fit to everyone:

    elpd_lopo = sum_i log p(y_i | y_-i),
    p(y_i | y_-i) ~ PSIS over posterior draws s of L_is = p(y_i | theta_s),

where ``L_is`` is the **marginal** likelihood of person i's trials given the
population-level draw ``theta_s``, the person's own parameters integrated out
against the population distribution. Leaving a person out with their own
parameters still in the model would leave nothing of them to predict with (and
the PSIS weights collapse); integrated out, dropping one of ~120 people barely
moves theta and PSIS is reliable, with its Pareto k per person as the check
(Merkle, Furr & Rabe-Hesketh 2019, "Bayesian comparison of latent variable
models: conditional versus marginal likelihoods").

The integral is over the model's person-level vectors (every free variable with
one entry per participant slot), each required to be a non-centred N(0, 1) z
(the model maps z to the person's parameters itself). It is estimated by
importance sampling, K draws per person shared across posterior draws:

    L_is ~ (1/K) sum_k prod_t p(y_it | theta_s, z_k) N(z_k) / q_i(z_k),

with q_i a defensive mixture of N(0, 1) (weight ``alpha``) and a normal at the
person's own posterior z (mean, sd inflated by ``inflate``) — sampling the
prior alone wastes most draws with 64 trials and up to six person-level
dimensions, and its downward bias would grow with the number of dimensions.
Each z_k is evaluated by writing it into an unused participant slot of the
thinned posterior (only the free variables are kept, so every deterministic is
recomputed from them) and predicting the person's trials as that slot. A
self-check first confirms that a slot holding a person's own z reproduces
their predictions.

A model without person-level vectors needs no integral: L_is is the product of
its per-trial likelihoods.

Also reported: ``elpd_in_sample`` = sum_i log mean_s L_is (the person included
in the fit, i.e. the new-participant score of person_level_analysis.py with a
better integral), and an MC stability check (the estimate from the first half
of the K draws).

Writes ``results/lopo_<scheme>_run<N>.csv`` and ``..._people.csv``.

Usage:
    uv run python data/results/human_experiment_sr-oct26/analysis/lopo_analysis.py --run 2
"""

from __future__ import annotations

import csv
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import tyro
from pyprojroot import here
from scipy.special import logsumexp
from scipy.stats import norm

sys.path.insert(0, str(here()))

# The pipeline sizes its concurrent fits by the scheduler's affinity mask
# (``allocated_cpus``), which only Linux has; on a laptop, every CPU.
if not hasattr(os, "sched_getaffinity"):
    os.sched_getaffinity = lambda pid: set(range(os.cpu_count()))

ANALYSIS_ROOT = here() / "data/results/human_experiment_sr-oct26/analysis"
N_SLOTS = 400
FIRST_FREE_SLOT = 120  # past every participant id of either run


@dataclass
class Args:
    """PSIS leave-one-participant-out ELPD with person-level parameters integrated out."""

    run: int = 1
    only: tuple[str, ...] = ()
    """Score only these models (default: every model of the scheme's models dir)."""
    seed_overrides: dict[str, int] = field(default_factory=dict)
    """Sampler seeds by model name, as person_level_analysis.py (the fit to load)."""
    draws: int = 1000
    """Posterior draws used (thinned evenly from the fit's 12,000)."""
    k: int = 100
    """Importance draws of the person-level z per person."""
    alpha: float = 0.1
    """Weight of N(0, 1) in the defensive proposal."""
    inflate: float = 1.5
    """Scale on the person's posterior sd of z in the proposal."""
    seed: int = 20261005
    out_dir: Path = ANALYSIS_ROOT / "results"
    cache_dir: Path = ANALYSIS_ROOT / "mcmc_cache"
    # The fits' settings (person_level_analysis.py).
    fit_draws: int = 3000
    tune: int = 2000
    chains: int = 4
    target_accept: float = 0.8
    random_seed: int = 20261003


def run_paths(run: int) -> tuple[Path, Path]:
    """(models dir, responses) of the all-data fits."""
    work = "fits/person_level" if run == 1 else f"fits/person_level_run{run}"
    responses = (
        here() / f"data/results/human_experiment_sr-oct26/collected/run{run}/subjective_randomness"
        / "experiment3/model_loop/responses.csv"
    )
    return ANALYSIS_ROOT / work / "all/models", responses


def slot_vectors(model) -> list[str]:
    """Names of the person-level free variables, each checked to be a N(0, 1) z."""
    import pytensor.tensor as pt
    from pytensor.tensor.random.basic import NormalRV

    point = model.initial_point()
    names = []
    for rv in model.free_RVs:
        shape = point[model.rvs_to_values[rv].name].shape
        if N_SLOTS not in shape:
            continue
        if shape != (N_SLOTS,):
            raise ValueError(f"{rv.name}: person-level variable of shape {shape}, expected ({N_SLOTS},)")
        if not isinstance(rv.owner.op, NormalRV):
            raise ValueError(f"{rv.name}: person-level variable is not Normal")
        mu, sigma = rv.owner.inputs[-2:]
        try:
            values = (pt.get_underlying_scalar_constant_value(mu), pt.get_underlying_scalar_constant_value(sigma))
        except pt.NotScalarConstantError as exc:
            raise ValueError(f"{rv.name}: person-level variable is not N(0, 1) (non-constant parameters)") from exc
        if values != (0.0, 1.0):
            raise ValueError(f"{rv.name}: person-level variable is N{values}, expected N(0, 1)")
        names.append(rv.name)
    return names


def free_posterior(fitted, draws: int):
    """The thinned posterior of the free variables only (deterministics recomputed)."""
    import arviz as az

    from src.models.pymc_inference import _thin_posterior

    post = _thin_posterior(fitted.idata, draws).posterior
    keep = [rv.name for rv in fitted.model.free_RVs]
    return az.InferenceData(posterior=post[keep].load())


def predict(fitted, idata, rows) -> np.ndarray:
    """(draws, rows) p_left of ``rows`` under ``idata``'s free-variable draws."""
    import pymc as pm

    from src.models.data_binding import make_stim_data
    from src.models.pymc_inference import _validated_p_left_draws

    with fitted.model:
        pm.set_data(make_stim_data(fitted.model, rows))
        pp = pm.sample_posterior_predictive(idata, var_names=["p_left"], random_seed=0, progressbar=False)
    arr = _validated_p_left_draws(pp.posterior_predictive["p_left"].values, context=fitted.name)
    out = arr.reshape(-1, arr.shape[-1])
    if out.shape[1] != len(rows):
        raise ValueError(f"{fitted.name}: p_left has {out.shape[1]} entries for {len(rows)} rows")
    return out


def loglik(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return np.where(y[None, :] == 1, np.log(p), np.log1p(-p))


def self_check(fitted, idata, vectors, rows_by_person, people) -> None:
    """A free slot holding a person's own z (per draw) predicts exactly as the person."""
    post = idata.posterior.copy(deep=True)
    checked = people[:: max(1, len(people) // 3)][:3]
    rows = []
    for j, who in enumerate(checked):
        slot = FIRST_FREE_SLOT + j
        for name in vectors:
            values = post[name].values
            values[..., slot] = values[..., int(who)]
            post[name].values = values
        rows += [{**r, "participant_id": slot} for r in rows_by_person[who]]
    import arviz as az

    moved = predict(fitted, az.InferenceData(posterior=post), rows)
    original = predict(fitted, idata, [r for who in checked for r in rows_by_person[who]])
    if not np.allclose(moved, original, atol=1e-9):
        raise AssertionError(f"{fitted.name}: a slot holding a person's z does not reproduce their predictions")


def proposal_moments(idata, vectors, people) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Each person's posterior mean and sd of their z (one entry per vector)."""
    post = idata.posterior
    out = {}
    for who in people:
        own = np.stack([post[name].values[..., int(who)].ravel() for name in vectors], axis=1)  # (draws, dims)
        out[who] = (own.mean(axis=0), own.std(axis=0))
    return out


def marginal_loglik(
    fitted, idata, vectors, rows_by_person, people, args, rng, *, moments=None, keep_draws=None, keep_k: int = 0
) -> tuple[np.ndarray, dict]:
    """(draws, people, K) log [prod_t p(y_it | theta_s, z_k) N(z_k) / q_i(z_k)].

    ``moments`` (person -> posterior mean and sd of z) centres each person's
    proposal; by default the person's own posterior in ``idata``. With
    ``keep_k`` > 0, ``info`` also holds ``trial_ll`` (people, len(keep_draws),
    keep_k, most trials of any person; float32), the per-trial
    log-likelihoods of the first ``keep_k`` z draws at posterior draws
    ``keep_draws`` (zero past a person's own trials: a person seen in two
    experiments has twice as many), and ``log_ratio`` (people, K), for
    reweighting trials (the bootstrap).
    """
    import arviz as az

    post = idata.posterior
    n_draws = post.sizes["chain"] * post.sizes["draw"]
    dims = len(vectors)
    if moments is None:
        moments = proposal_moments(idata, vectors, people)
    # Per person: the proposal and its K draws (dims each).
    z = np.empty((len(people), args.k, dims))
    log_ratio = np.empty((len(people), args.k))  # log N(z) - log q(z)
    for i, who in enumerate(people):
        m, s = moments[who]
        s = s * args.inflate
        prior = rng.random(args.k) < args.alpha
        draw = np.where(prior[:, None], rng.standard_normal((args.k, dims)), m + s * rng.standard_normal((args.k, dims)))
        log_prior = norm.logpdf(draw).sum(axis=1)
        log_q = np.logaddexp(np.log(args.alpha) + log_prior, np.log1p(-args.alpha) + norm.logpdf(draw, m, s).sum(axis=1))
        z[i], log_ratio[i] = draw, log_prior - log_q

    pairs = [(i, k) for i in range(len(people)) for k in range(args.k)]
    per_call = N_SLOTS - FIRST_FREE_SLOT
    out = np.empty((n_draws, len(people), args.k))
    trial_ll = None
    if keep_k:
        most = max(len(rows_by_person[p]) for p in people)
        trial_ll = np.zeros((len(people), len(keep_draws), keep_k, most), dtype=np.float32)
    for start in range(0, len(pairs), per_call):
        chunk = pairs[start : start + per_call]
        moved = post.copy(deep=True)
        rows, owner = [], []
        for j, (i, k) in enumerate(chunk):
            slot = FIRST_FREE_SLOT + j
            for d, name in enumerate(vectors):
                values = moved[name].values
                values[..., slot] = z[i, k, d]
                moved[name].values = values
            person_rows = rows_by_person[people[i]]
            rows += [{**r, "participant_id": slot} for r in person_rows]
            owner += [j] * len(person_rows)
        p = predict(fitted, az.InferenceData(posterior=moved), rows)
        ll = loglik(p, np.array([int(r["chose_left"]) for r in rows]))
        sums = np.zeros((n_draws, len(chunk)))
        np.add.at(sums.T, np.array(owner), ll.T)
        owner = np.array(owner)
        for j, (i, k) in enumerate(chunk):
            out[:, i, k] = sums[:, j] + log_ratio[i, k]
            if k < keep_k:
                cols = np.flatnonzero(owner == j)
                trial_ll[i, :, k, : len(cols)] = ll[np.ix_(keep_draws, cols)]
    # Importance-sampling quality: effective share of the K draws, per person
    # (averaged over posterior draws).
    w = np.exp(out - out.max(axis=2, keepdims=True))
    ess = (w.sum(axis=2) ** 2 / (w**2).sum(axis=2)).mean(axis=0) / args.k
    info = {"is_ess_share_min": float(ess.min()), "is_ess_share_median": float(np.median(ess))}
    if keep_k:
        info["trial_ll"], info["log_ratio"] = trial_ll, log_ratio
    return out, info


def psis_lopo(log_lik: np.ndarray, n_chains: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-person PSIS-LOO elpd and Pareto k from (draws, people) log-likelihoods."""
    import arviz as az

    # Smoothed log weights of leaving each person out, (people, draws) as
    # psislw takes them; thinned draws, so relative efficiency 1.
    log_weights, k = az.psislw(-log_lik.T, reff=1.0)
    if np.shape(k) != (log_lik.shape[1],):
        raise ValueError(f"Pareto k of shape {np.shape(k)} for {log_lik.shape[1]} people")
    return logsumexp(log_lik.T + log_weights, axis=1), np.asarray(k)


def main(args: Args) -> None:
    from src.models.pymc_inference import clear_fit_cache, convergence_problems_of, fit_model

    models_dir, responses = run_paths(args.run)
    names = [
        line.split("name: ")[1].strip()
        for line in (models_dir / "models_manifest.yaml").read_text().splitlines()
        if "name:" in line
    ]
    if args.only:
        unknown = sorted(set(args.only) - set(names))
        if unknown:
            raise ValueError(f"--only names unknown models: {unknown}")
        names = [n for n in names if n in args.only]
    with responses.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    people = sorted({r["participant_id"] for r in rows}, key=int)
    if int(people[-1]) >= FIRST_FREE_SLOT:
        raise ValueError(f"participant ids reach the free slots (from {FIRST_FREE_SLOT})")
    rows_by_person = {p: [r for r in rows if r["participant_id"] == p] for p in people}

    suffix = f"_run{args.run}" + ("_only" if args.only else "")
    records, person_rows = [], []
    for name in names:
        started = time.time()
        seed = args.seed_overrides.get(name, args.random_seed)
        fitted = fit_model(
            name, models_dir, responses, cache_dir=args.cache_dir, draws=args.fit_draws, tune=args.tune,
            chains=args.chains, target_accept=args.target_accept, random_seed=seed,
        )
        problems = convergence_problems_of(fitted)
        idata = free_posterior(fitted, args.draws)
        n_chains = idata.posterior.sizes["chain"]
        vectors = slot_vectors(fitted.model)
        rng = np.random.default_rng(args.seed)
        info: dict = {}
        if vectors:
            self_check(fitted, idata, vectors, rows_by_person, people)
            terms, info = marginal_loglik(fitted, idata, vectors, rows_by_person, people, args, rng)
            log_lik = logsumexp(terms, axis=2) - np.log(args.k)
            half = logsumexp(terms[:, :, : args.k // 2], axis=2) - np.log(args.k // 2)
        else:
            ordered = [r for p in people for r in rows_by_person[p]]
            ll = loglik(predict(fitted, idata, ordered), np.array([int(r["chose_left"]) for r in ordered]))
            owner = np.repeat(np.arange(len(people)), [len(rows_by_person[p]) for p in people])
            log_lik = np.zeros((ll.shape[0], len(people)))
            np.add.at(log_lik.T, owner, ll.T)
            half = log_lik
        elpd_i, k_i = psis_lopo(log_lik, n_chains)
        elpd_half, _ = psis_lopo(half, n_chains)
        in_sample = logsumexp(log_lik, axis=0) - np.log(log_lik.shape[0])
        records.append(
            {
                "model": name,
                "elpd_lopo": float(elpd_i.sum()),
                "se_lopo": float(np.sqrt(len(elpd_i) * elpd_i.var())),
                "elpd_in_sample": float(in_sample.sum()),
                "optimism": float(in_sample.sum() - elpd_i.sum()),
                "elpd_lopo_half_k": float(elpd_half.sum()),
                "max_pareto_k": float(k_i.max()),
                "n_k_over_0.7": int((k_i > 0.7).sum()),
                "n_person_vectors": len(vectors),
                "n_people": len(people),
                "draws": int(log_lik.shape[0]),
                "k": args.k if vectors else 0,
                **info,
                "convergence_problems": "; ".join(problems),
                "random_seed": seed,
                "seconds": round(time.time() - started),
            }
        )
        person_rows += [
            {"model": name, "participant_id": p, "elpd_lopo": e, "pareto_k": k, "in_sample": s}
            for p, e, k, s in zip(people, elpd_i, k_i, in_sample)
        ]
        print(
            f"[lopo] run {args.run} {name}: {elpd_i.sum():.1f} (in-sample {in_sample.sum():.1f}, "
            f"half-K {elpd_half.sum():.1f}, max k {k_i.max():.2f}, {records[-1]['seconds']} s)",
            flush=True,
        )
        del fitted, idata
        clear_fit_cache()

    table = pd.DataFrame(records).sort_values("elpd_lopo", ascending=False)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out_dir / f"lopo_all{suffix}.csv", index=False)
    pd.DataFrame(person_rows).to_csv(args.out_dir / f"lopo_all_people{suffix}.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(table.drop(columns=["convergence_problems"]).to_string(index=False, float_format="%.2f"))


if __name__ == "__main__":
    main(tyro.cli(Args))
