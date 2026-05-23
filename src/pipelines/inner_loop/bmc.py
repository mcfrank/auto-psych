"""Bayesian model comparison over the zoo, with BIC penalty.

Replaces argmax-by-log-likelihood tournament selection with a posterior over
the full set of surviving candidates. The fitted log-likelihood overstates the
evidence for flexible models because it is the maximum, not the marginal —
BIC corrects for parameter count::

    marginal_log_likelihood ≈ log_likelihood − (n_params / 2) · log(N)

Where ``n_params`` and ``n_trials (== N)`` come from each ``FitResult``. Posterior
is the log-sum-exp normalisation of ``marginal_log_likelihood + log_prior``.

The "top-mass set" — the smallest collection of models whose cumulative
posterior exceeds a threshold — is what the critic targets: there is no value
in critiquing models the data has already ruled out.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable, Mapping

from src.pipelines.inner_loop.zoo import ZooEntry, iter_zoo


def _logsumexp(values: Iterable[float]) -> float:
    vs = list(values)
    if not vs:
        raise ValueError("logsumexp over empty sequence")
    m = max(vs)
    return m + math.log(sum(math.exp(v - m) for v in vs))


def compute_bmc(
    zoo_dir: Path,
    *,
    n_trials: int | None = None,
    log_prior: Mapping[str, float] | None = None,
    top_mass_threshold: float = 0.5,
) -> dict:
    """Score every entry in the zoo and return a posterior.

    Parameters
    ----------
    zoo_dir
        Directory written by ``src.pipelines.inner_loop.zoo``.
    n_trials
        Override the number of trials used in the BIC penalty. By default
        each entry's own ``n_trials`` is used; this is fine when all entries
        were fit on the same dataset (the standard case for the agentic
        loop). If entries disagree on ``n_trials`` and no override is given,
        a ``ValueError`` is raised.
    log_prior
        Optional dict mapping ``entry_id`` to a log prior. Missing entries
        receive a flat contribution. ``None`` (default) is equivalent to a
        uniform prior over the zoo.
    top_mass_threshold
        The smallest set of entries (sorted by posterior, descending) whose
        cumulative posterior exceeds this is reported as ``top_mass_set``.

    Returns
    -------
    A JSON-serialisable dict; see ``write_bmc`` for the on-disk shape.
    """
    entries: list[ZooEntry] = list(iter_zoo(zoo_dir))
    if not entries:
        raise ValueError(f"zoo at {zoo_dir} is empty")

    fits = {e.entry_id: e.load_fit() for e in entries}
    n_trials_per_entry = {eid: f.n_trials for eid, f in fits.items()}
    distinct_n = set(n_trials_per_entry.values())
    if n_trials is None:
        if len(distinct_n) > 1:
            raise ValueError(
                f"zoo entries disagree on n_trials: {n_trials_per_entry}; "
                "pass n_trials=... to override"
            )
        n_trials = next(iter(distinct_n))
    if n_trials <= 0:
        raise ValueError(f"n_trials must be positive, got {n_trials}")

    log_n = math.log(n_trials)
    log_likelihoods: dict[str, float] = {}
    n_params_per_entry: dict[str, int] = {}
    bic_penalties: dict[str, float] = {}
    marginal_lls: dict[str, float] = {}
    log_priors: dict[str, float] = {}
    log_posteriors_unnorm: dict[str, float] = {}

    for eid, fit in fits.items():
        log_likelihoods[eid] = float(fit.log_likelihood)
        n_params_per_entry[eid] = int(fit.n_params)
        penalty = 0.5 * fit.n_params * log_n
        bic_penalties[eid] = penalty
        marginal_lls[eid] = log_likelihoods[eid] - penalty
        lp = float(log_prior.get(eid, 0.0)) if log_prior is not None else 0.0
        log_priors[eid] = lp
        log_posteriors_unnorm[eid] = marginal_lls[eid] + lp

    log_z = _logsumexp(log_posteriors_unnorm.values())
    log_posteriors = {eid: v - log_z for eid, v in log_posteriors_unnorm.items()}
    posteriors = {eid: math.exp(v) for eid, v in log_posteriors.items()}

    ranked = sorted(posteriors.items(), key=lambda kv: kv[1], reverse=True)
    top_mass_set: list[str] = []
    cum = 0.0
    for eid, p in ranked:
        top_mass_set.append(eid)
        cum += p
        if cum >= top_mass_threshold:
            break

    return {
        "n_trials": n_trials,
        "n_params": n_params_per_entry,
        "log_likelihoods": log_likelihoods,
        "bic_penalties": bic_penalties,
        "marginal_log_likelihoods": marginal_lls,
        "log_priors": log_priors,
        "log_posteriors": log_posteriors,
        "posteriors": posteriors,
        "top_mass_threshold": top_mass_threshold,
        "top_mass_set": top_mass_set,
        "ranking": [eid for eid, _ in ranked],
    }


def write_bmc(zoo_dir: Path, out_path: Path, **kwargs) -> dict:
    """Compute BMC and write the result to ``out_path`` as JSON."""
    result = compute_bmc(zoo_dir, **kwargs)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    return result
