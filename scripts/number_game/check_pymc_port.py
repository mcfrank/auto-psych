"""Check the PyMC number-game seed against Eric Bigelow's numpyro model.

    uv run --with numpyro --with jax python -m scripts.number_game.check_pymc_port golden
    uv run --with numpyro --with jax python -m scripts.number_game.check_pymc_port sample
    uv run python -m scripts.number_game.sample_pymc_seed
    uv run --with numpyro --with jax python -m scripts.number_game.check_pymc_port compare

`golden` evaluates the reference (`numpyro_reference/`, `--prior-type basic`
or `interval`) in 64-bit at parameter points drawn from its prior, on every
cell of a spread of sets, and writes the joint log density and the predictive
per cell to `tests/fixtures/number_game/numpyro_<prior type>_golden.json`,
which `tests/test_number_game_seed_port.py` checks the PyMC seeds
(`bigelow_piantadosi_independent`, `bigelow_piantadosi_interval`) against.
For `interval` the reference's midpoint grid is replaced by the port's
(`corrected_midpoint_grid`): the reference reads grid point m + n + 1 of 200
for interval [m, n], off by three from its midpoint.

`sample` runs the reference's NUTS on the whole dataset's cell counts and
saves the draws of p_mix and p_rules; `sample_pymc_seed.py` does the same for
the seed on one row per response (the same posterior) in a process that never
imports jax (PyMC forks its chains, and forking after jax has started its
threads can hang). `compare` puts the two posteriors side by side, predicting every cell from thinned draws of each with the reference's
own matrices. The data are the reference's: untrimmed.
"""

from __future__ import annotations

import os

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

import json  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Literal  # noqa: E402

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import numpyro  # noqa: E402
import numpyro.distributions as dist  # noqa: E402
import tyro  # noqa: E402
from numpyro.infer.util import log_density  # noqa: E402
from scipy.special import gammaln, logsumexp  # noqa: E402

from scripts.number_game.numpyro_reference import infer  # noqa: E402
from scripts.number_game.numpyro_reference.np_utils import get_likelihoods_predictives  # noqa: E402
from src.number_game import data as ng_data  # noqa: E402
from src.number_game.metrics import binned_mean_r2  # noqa: E402
from src.runtime.config import REPO_ROOT  # noqa: E402

DATA_CSV = REPO_ROOT / "data" / "number-game" / "human experiment" / "numbergame_data.csv"
GOLDEN_DIR = REPO_ROOT / "tests" / "fixtures" / "number_game"
PRIORS = {"basic": infer.prior_eb_numbergame, "interval": infer.prior_eb_numbergame_interval}


def corrected_midpoint_grid() -> None:
    """Set the reference's lazily built interval grid (module globals) to the
    true midpoints: grid point m + n - 2 of 199 for interval [m, n]."""
    infer.interval_bounds = [(i, j) for i in range(1, 101) for j in range(1, 101) if i <= j]
    infer.interval_means = jnp.array([i + j - 2 for i, j in infer.interval_bounds], dtype=jnp.int16)
    infer.mean_range = (jnp.arange(199) / 198).clip(min=1e-5, max=1 - 1e-5)
SEEDS_DIR = REPO_ROOT / "src" / "pipelines" / "outer_loop" / "projects" / "number_game" / "seed_models"
SEED = "bigelow_piantadosi_independent"


def reference_inputs(cells):
    """The reference's model kwargs for `cells` (run_number_eb.main, verbatim logic)."""
    sets = list(dict.fromkeys(cells["set"]))
    h_likelihoods, y_predictives = map(jnp.array, get_likelihoods_predictives())
    max_len = max(len(s) for s in sets)
    obs_X = jnp.array([[x - 1 for x in s] + [-1] * (max_len - len(s)) for s in sets])
    obs_y = -np.ones((len(sets), 100, 2))
    index = {s: i for i, s in enumerate(sets)}
    for s, target, yes, n in zip(cells["set"], cells["target"], cells["yes"], cells["n"]):
        obs_y[index[s], target - 1] = [yes, n]
    X_ = obs_X[..., None]
    X_likelihoods = dist.Categorical(probs=h_likelihoods).mask(X_ != -1).log_prob(X_).sum(axis=-2)
    kwargs = dict(y_predictives=y_predictives, X_likelihoods=X_likelihoods, obs_y=jnp.array(obs_y))
    return kwargs, sets


@dataclass
class Args:
    command: tyro.conf.Positional[Literal["golden", "sample", "compare"]]
    prior_type: Literal["basic", "interval"] = "basic"
    n_sets: int = 12
    n_points: int = 3
    draws: int = 1000
    tune: int = 1000
    chains: int = 4
    n_predictive_draws: int = 200
    seed: int = 0
    out_dir: Path = REPO_ROOT / "data" / "number-game" / "port_check"


def golden(args: Args) -> None:
    if args.prior_type == "interval":
        corrected_midpoint_grid()
    all_cells = ng_data.cell_counts(ng_data.load_responses(DATA_CSV))
    every_set = list(dict.fromkeys(all_cells["set"]))
    chosen = set(every_set[:: len(every_set) // args.n_sets][: args.n_sets])
    cells = all_cells[all_cells["set"].isin(chosen)].reset_index(drop=True)
    kwargs, sets = reference_inputs(cells)
    index = {s: i for i, s in enumerate(sets)}
    rows, cols = [index[s] for s in cells["set"]], (cells["target"] - 1).tolist()

    points = []
    for key in jax.random.split(jax.random.PRNGKey(args.seed), args.n_points):
        prior = numpyro.handlers.trace(numpyro.handlers.seed(PRIORS[args.prior_type], key)).get_trace()
        params = {name: site["value"] for name, site in prior.items() if site["type"] == "sample"}
        joint, trace = log_density(infer.number_bpl_model, (), {**kwargs, "prior_type": args.prior_type}, params)
        predictive = np.asarray(trace["y"]["fn"].base_dist.probs).reshape(len(sets), 100)
        points.append({
            **{name: np.asarray(value).tolist() for name, value in params.items()},
            "log_density": float(joint),
            "post_predictive": predictive[rows, cols].tolist(),
        })

    yes, n = cells["yes"].to_numpy(), cells["n"].to_numpy()
    out = {
        "source": f"scripts/number_game/numpyro_reference, prior_type={args.prior_type!r}, jax x64",
        "cells": [[",".join(map(str, s)), int(t), int(y), int(m - y)] for s, t, y, m in zip(cells["set"], cells["target"], yes, n)],
        "log_binomial_coefficients": float((gammaln(n + 1) - gammaln(yes + 1) - gammaln(n - yes + 1)).sum()),
        "points": points,
    }
    path = GOLDEN_DIR / f"numpyro_{args.prior_type}_golden.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out))
    print(f"wrote {path}: {len(cells)} cells over {len(sets)} sets, {len(points)} points")


def _sample_numpyro(args: Args, cells):
    kwargs, _ = reference_inputs(cells)
    mcmc = numpyro.infer.MCMC(
        numpyro.infer.NUTS(infer.number_bpl_model),
        num_warmup=args.tune, num_samples=args.draws, num_chains=args.chains, chain_method="parallel",
    )
    mcmc.run(jax.random.PRNGKey(args.seed), extra_fields=("diverging",), **kwargs)
    samples = mcmc.get_samples(group_by_chain=True)
    divergent = int(np.asarray(mcmc.get_extra_fields()["diverging"]).sum())
    return np.asarray(samples["p_mix"]), np.asarray(samples["p_rules"]), divergent


def sample(args: Args) -> None:
    cells = ng_data.cell_counts(ng_data.load_responses(DATA_CSV))
    p_mix, p_rules, divergent = _sample_numpyro(args, cells)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / "numpyro_draws.npz"
    np.savez(out, p_mix=p_mix, p_rules=p_rules, divergent=divergent)
    print(f"wrote {out}: p_mix {p_mix.shape}, p_rules {p_rules.shape}, {divergent} divergent")


def predictive(p_mix, p_rules, cells):
    """P(yes) per cell for each draw, from the reference's matrices (numpy)."""
    h_likelihoods, y_predictives = get_likelihoods_predictives()
    sets = list(dict.fromkeys(cells["set"]))
    index = {s: i for i, s in enumerate(sets)}
    set_log_lik = np.stack([np.log(h_likelihoods[:, np.asarray(s) - 1]).sum(axis=1) for s in sets])
    n_rules = p_rules.shape[-1]
    rows, cols = np.array([index[s] for s in cells["set"]]), cells["target"].to_numpy() - 1
    out = []
    for mix, rules in zip(p_mix, p_rules):
        prior = np.concatenate([mix * rules / rules.sum(), np.full(h_likelihoods.shape[0] - n_rules, (1 - mix) / 5050)])
        log_post = np.log(prior / prior.sum())[None, :] + set_log_lik
        log_post -= logsumexp(log_post, axis=1, keepdims=True)
        out.append((np.exp(log_post) @ y_predictives)[rows, cols])
    return np.array(out)


def compare(args: Args) -> None:
    import arviz as az

    cells = ng_data.cell_counts(ng_data.load_responses(DATA_CSV))
    report = {}
    means = {}
    for framework in ("numpyro", "pymc"):
        draws = np.load(args.out_dir / f"{framework}_draws.npz")
        p_mix, p_rules = draws["p_mix"], draws["p_rules"]
        weights = p_rules / p_rules.sum(axis=-1, keepdims=True)
        rhat = az.rhat(az.convert_to_dataset({"p_mix": p_mix, "p_rules": p_rules}))
        flat_mix, flat_rules = p_mix.reshape(-1), p_rules.reshape(-1, p_rules.shape[-1])
        thin = np.linspace(0, len(flat_mix) - 1, args.n_predictive_draws).astype(int)
        p_yes = predictive(flat_mix[thin], flat_rules[thin], cells).mean(axis=0)
        means[framework] = {"p_mix": flat_mix, "weights": weights.reshape(-1, weights.shape[-1]), "p_yes": p_yes}
        report[framework] = {
            "p_mix_mean": float(flat_mix.mean()), "p_mix_sd": float(flat_mix.std()),
            "max_rhat": float(max(rhat["p_mix"].max(), rhat["p_rules"].max())),
            "divergent": int(draws["divergent"]),
            "binned_mean_r2": binned_mean_r2(p_yes, cells),
            "top_rules": np.argsort(-weights.reshape(-1, weights.shape[-1]).mean(axis=0))[:8].tolist(),
        }
    a, b = means["numpyro"], means["pymc"]
    report["difference"] = {
        "p_mix_mean_in_sds": float(abs(a["p_mix"].mean() - b["p_mix"].mean()) / a["p_mix"].std()),
        "max_rule_weight_mean_in_sds": float(np.max(np.abs(a["weights"].mean(0) - b["weights"].mean(0)) / a["weights"].std(0))),
        "max_abs_p_yes": float(np.max(np.abs(a["p_yes"] - b["p_yes"]))),
    }
    (args.out_dir / "compare.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    args = tyro.cli(Args)
    {"golden": golden, "sample": sample, "compare": compare}[args.command](args)
