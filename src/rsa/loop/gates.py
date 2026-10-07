"""Admission of an agent-written memo candidate to the RSA loop's zoo.

A candidate directory must hold ``candidate.py`` (the model file) and
``hypothesis.md`` (its mechanism in prose); ``model_name.txt`` is optional.
It is admitted only if, in order:

1. the code gate passes (`src.rsa.loop.code_gate`);
2. it loads and meets the model contract (`src.rsa.model_file.check_contract`)
   on the training displays AND on every shape of the novelty pool, so a model
   that breaks on a 4 x 4 game is refused before any sampling;
3. it fits within the time limit, gives no observed choice probability zero,
   and passes the convergence gate (one refit at target_accept 0.95 with a
   fresh seed when the first fit fails it);
4. its ELPD-LOO is finite;
5. it is novel: its posterior-mean pool predictions are at least the
   threshold (RMSE) from every admitted model's.

Every refusal carries a reason written for the agent that will be asked to
repair it. Infrastructure failures raise instead.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

import numpy as np

from src.rsa.context import Context, group_by_shape, unique_contexts
from src.rsa.fit import FitSettings, RSAFit
from src.rsa.loop.code_gate import code_problems
from src.rsa.loop.fitting import ModelFailure, fit_cached
from src.rsa.loop.novelty import closest, posterior_mean_class_probs
from src.rsa.model_file import ModelContractViolation, RSAModel, check_contract

REFIT_TARGET_ACCEPT = 0.95


@dataclass
class Admission:
    admitted: bool
    reason: str
    fit: Optional[RSAFit] = None
    pool_preds: Optional[np.ndarray] = None
    elpd_loo: Optional[float] = None
    loo_reliable: Optional[bool] = None


@dataclass(frozen=True)
class GateConfig:
    responses_path: Path
    cache_dir: Path
    settings: FitSettings
    novelty_threshold: float
    time_limit_sec: Optional[float]


def _refuse(reason: str) -> Admission:
    return Admission(admitted=False, reason=reason)


def read_candidate(candidate_dir: Path) -> Optional[str]:
    """The reason a candidate directory is incomplete, or None."""
    for required in ("candidate.py", "hypothesis.md"):
        f = Path(candidate_dir) / required
        if not f.exists() or not f.read_text(encoding="utf-8").strip():
            return f"{required} is missing or empty"
    return None


def fit_with_refit(model_path: Path, name: str, cfg: GateConfig) -> RSAFit:
    fitted = fit_cached(model_path, name, cfg.responses_path, cfg.settings, cfg.cache_dir,
                        time_limit_sec=cfg.time_limit_sec)
    if fitted.converged:
        return fitted
    retry = replace(cfg.settings, target_accept=max(cfg.settings.target_accept, REFIT_TARGET_ACCEPT),
                    seed=cfg.settings.seed + 1)
    return fit_cached(model_path, name, cfg.responses_path, retry, cfg.cache_dir,
                      time_limit_sec=cfg.time_limit_sec)


def admit(
    candidate_dir: Path,
    name: str,
    *,
    cfg: GateConfig,
    training: Sequence[Context],
    pool: Sequence[Context],
    admitted_preds: Mapping[str, np.ndarray],
) -> Admission:
    candidate_dir = Path(candidate_dir)
    missing = read_candidate(candidate_dir)
    if missing:
        return _refuse(missing)
    model_path = candidate_dir / "candidate.py"
    source = model_path.read_text(encoding="utf-8")
    problems = code_problems(source)
    if problems:
        return _refuse(
            "the code gate refused candidate.py: " + ", ".join(problems)
            + ". Model files may import jax, memo, numpyro, numpy, a few stdlib modules "
            "and src.rsa.memo_kit only, and may not read files or reach the interpreter."
        )
    try:
        model = RSAModel(model_path, name=name)
        check_contract(model, group_by_shape(unique_contexts(training)[0]))
        check_contract(model, group_by_shape(pool))
    except ModelContractViolation as exc:
        return _refuse(f"contract: {exc}")
    except Exception as exc:  # memo compile errors, errors in the model's own code
        from src.rsa.loop.fitting import is_model_failure

        if is_model_failure(exc, model_path):
            return _refuse(f"the model failed to load or run: {type(exc).__name__}: {exc}")
        raise
    try:
        fitted = fit_with_refit(model_path, name, cfg)
    except ModelFailure as exc:
        return _refuse(f"fitting failed: {exc}")
    if not fitted.converged:
        return _refuse(
            "the fit did not converge even at target_accept 0.95: "
            + "; ".join(fitted.convergence_problems)
            + ". Reparameterise (e.g. sample on an unconstrained scale, tighten priors) "
            "rather than relying on smaller steps."
        )
    loo = fitted.loo()
    if not math.isfinite(loo.elpd_loo):
        return _refuse(f"ELPD-LOO is not finite ({loo.elpd_loo})")
    try:
        preds = posterior_mean_class_probs(model, fitted, pool)
    except ValueError as exc:
        return _refuse(str(exc))
    nearest, dist = closest(preds, admitted_preds)
    if dist < cfg.novelty_threshold:
        return Admission(
            admitted=False,
            reason=(
                f"not novel: its predictions are within RMSE {dist:.4f} of {nearest}'s "
                f"(threshold {cfg.novelty_threshold}) over the loop's novelty pool of "
                f"{len(pool)} displays. A re-parameterisation of an existing mechanism is "
                f"not a new hypothesis; change what the listener or speaker computes."
            ),
            fit=fitted, pool_preds=preds, elpd_loo=loo.elpd_loo, loo_reliable=not loo.unreliable,
        )
    return Admission(
        admitted=True, reason="admitted", fit=fitted, pool_preds=preds,
        elpd_loo=loo.elpd_loo, loo_reliable=not loo.unreliable,
    )


def pool_predictions_for(
    models: Mapping[str, RSAModel], fits: Mapping[str, RSAFit], pool: Sequence[Context]
) -> Dict[str, np.ndarray]:
    return {n: posterior_mean_class_probs(models[n], fits[n], pool) for n in models}
