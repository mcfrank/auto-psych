from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.optimize import minimize

from src.pipelines.inner_loop.core import Dataset, Likelihood, Sampler
from src.pipelines.inner_loop.likelihood import CategoricalLikelihood


@dataclass(init=False)
class FitResult:
    params: list[float]
    log_likelihood: float
    per_trial_ll: np.ndarray
    n_samples: int
    n_trials: int
    n_params: int | None = None
    sample_populations: list | None = None

    def __init__(
        self,
        params: list[float],
        log_likelihood: float,
        per_trial_ll: np.ndarray | list[float] | None = None,
        n_samples: int = 0,
        n_trials: int | None = None,
        n_params: int | None = None,
        sample_populations: list | None = None,
    ) -> None:
        if per_trial_ll is None:
            raise TypeError("FitResult requires per_trial_ll")
        self.params = params
        self.log_likelihood = log_likelihood
        self.per_trial_ll = np.asarray(per_trial_ll, dtype=float)
        self.n_samples = n_samples
        self.n_trials = n_trials if n_trials is not None else len(self.per_trial_ll)
        self.n_params = n_params
        self.sample_populations = sample_populations
        self.__post_init__()

    def __post_init__(self) -> None:
        self.per_trial_ll = np.asarray(self.per_trial_ll, dtype=float)
        if self.per_trial_ll.shape != (self.n_trials,):
            raise ValueError(
                f"per_trial_ll shape {self.per_trial_ll.shape} != (n_trials={self.n_trials},)"
            )
        if self.n_params is None:
            self.n_params = len(self.params)



def _rounded_param_key(params: list[float] | np.ndarray, cache_decimals: int) -> tuple[float, ...]:
    return tuple(round(float(param), cache_decimals) for param in params)


def sample_model_population(
    model_func: Callable,
    data: Dataset,
    params: list[float] | None,
    n_samples: int,
    *,
    sampler: Sampler | None = None,
    base_seed: int = 0,
    cache: dict | None = None,
    cache_decimals: int = 4,
    **kwargs,
) -> list:
    if sampler is None:
        raise ValueError("sample_model_population requires an experiment-specific sampler")
    cache_key = None
    if cache is not None:
        cache_key = (_rounded_param_key(params or [], cache_decimals), n_samples, base_seed)
        if cache_key in cache:
            return cache[cache_key]
    samples = sampler(model_func, data, params, n_samples, base_seed=base_seed, **kwargs)
    if cache is not None and cache_key is not None:
        cache[cache_key] = samples
    return samples


def _per_trial_ll(
    model_func: Callable,
    data: Dataset,
    params: list[float] | None,
    likelihood: Likelihood,
    **kwargs,
) -> np.ndarray:
    values = np.asarray(likelihood(model_func, data, params, **kwargs), dtype=float)
    if values.shape != (len(data),):
        raise ValueError(f"likelihood returned shape {values.shape}; expected ({len(data)},)")
    return values


def fit_model(
    model_func: Callable,
    data: Dataset,
    *,
    likelihood: Likelihood | None = None,
    sampler: Sampler | None = None,
    n_samples: int = 0,
    n_starts: int = 5,
    max_steps: int = 100,
    initial_params: list[float] | None = None,
    param_bounds: list[tuple[float, float]] | None = None,
    base_seed: int = 0,
    cache_decimals: int = 4,
) -> FitResult:
    """Fit a cognitive model against any per-trial likelihood.

    For fixed models, pass no ``param_bounds``/``initial_params`` and the
    function simply evaluates the likelihood. For parameterized models, provide
    continuous bounds and optional starts; optimization uses scipy's
    derivative-free Nelder-Mead with clipping inside the objective.
    """
    likelihood = likelihood or CategoricalLikelihood()
    sample_cache: dict = {}

    if not param_bounds:
        per_trial = _per_trial_ll(model_func, data, None, likelihood)
        samples = (
            sample_model_population(
                model_func,
                data,
                None,
                n_samples,
                sampler=sampler,
                base_seed=base_seed,
                cache=sample_cache,
                cache_decimals=cache_decimals,
            )
            if sampler is not None and n_samples > 0
            else None
        )
        return FitResult(
            params=[],
            log_likelihood=float(per_trial.sum()),
            per_trial_ll=per_trial,
            n_samples=n_samples,
            n_trials=len(data),
            n_params=0,
            sample_populations=samples,
        )

    bounds = list(param_bounds)
    if initial_params is None:
        initial_params = [(lo + hi) / 2 for lo, hi in bounds]
    if len(initial_params) != len(bounds):
        raise ValueError("initial_params and param_bounds must have the same length")

    objective_cache: dict[tuple[float, ...], float] = {}

    def clip_params(params: list[float] | np.ndarray) -> list[float]:
        return [max(lo, min(hi, float(value))) for value, (lo, hi) in zip(params, bounds)]

    def objective(params: list[float] | np.ndarray) -> float:
        clipped = clip_params(params)
        key = _rounded_param_key(clipped, cache_decimals)
        if key not in objective_cache:
            objective_cache[key] = -float(_per_trial_ll(model_func, data, clipped, likelihood).sum())
        return objective_cache[key]

    rng = np.random.default_rng(base_seed)
    starts = [list(initial_params)]
    for _ in range(max(0, n_starts - 1)):
        starts.append([float(rng.uniform(lo, hi)) for lo, hi in bounds])

    best_x = starts[0]
    best_val = float("inf")
    for x0 in starts:
        res = minimize(
            objective,
            x0,
            method="Nelder-Mead",
            options={"maxiter": max_steps, "xatol": 1e-3, "fatol": 1e-3},
        )
        if float(res.fun) < best_val:
            best_val = float(res.fun)
            best_x = clip_params(res.x)

    per_trial = _per_trial_ll(model_func, data, best_x, likelihood)
    samples = (
        sample_model_population(
            model_func,
            data,
            best_x,
            n_samples,
            sampler=sampler,
            base_seed=base_seed,
            cache=sample_cache,
            cache_decimals=cache_decimals,
        )
        if sampler is not None and n_samples > 0
        else None
    )
    return FitResult(
        params=best_x,
        log_likelihood=float(per_trial.sum()),
        per_trial_ll=per_trial,
        n_samples=n_samples,
        n_trials=len(data),
        n_params=len(best_x),
        sample_populations=samples,
    )
