from __future__ import annotations

import math
import json
from typing import Any, Mapping

import numpy as np

from src.pipelines.inner_loop.core import Dataset, ModelCallable, Params, normalize_probs


def predict(model: ModelCallable, stimulus: Any, options: list[str], params: Params = None) -> dict[str, float]:
    try:
        probs = model(stimulus, options, params)
    except TypeError:
        probs = model(stimulus, options)  # type: ignore[misc]
    return normalize_probs(probs, options)


def _cache_key(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True)
    except TypeError:
        return repr(value)


class CategoricalLikelihood:
    def __init__(self, clip: float = 1e-9):
        self.clip = clip

    def __call__(self, model: ModelCallable, data: Dataset, params: Params = None, **_: Any) -> np.ndarray:
        out = np.empty(len(data), dtype=float)
        cache: dict[str, dict[str, float]] = {}
        for i, trial in enumerate(data.trials):
            key = _cache_key(trial.stimulus)
            cache.setdefault(key, predict(model, trial.stimulus, data.response_options, params))
            p = min(1.0 - self.clip, max(self.clip, cache[key].get(trial.response, 0.0)))
            out[i] = math.log(p)
        return out


class CategoricalSampler:
    def __call__(
        self,
        model: ModelCallable,
        data: Dataset,
        params: Params = None,
        n_samples: int = 1,
        *,
        base_seed: int = 0,
        **_: Any,
    ) -> list[list[str]]:
        rng = np.random.default_rng(base_seed)
        draws = []
        for trial in data.trials:
            probs = predict(model, trial.stimulus, data.response_options, params)
            p = np.array([probs[option] for option in data.response_options])
            draws.append([str(x) for x in rng.choice(data.response_options, n_samples, p=p / p.sum())])
        return draws


class PyMCLikelihood:
    def __call__(self, model: ModelCallable, data: Dataset, params: Params = None, **kw: Any) -> np.ndarray:
        try:
            import pymc  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("Install PyMC support with `uv sync --group pymc`.") from exc

        result = model(data, params=params, **kw)  # type: ignore[call-arg]
        if isinstance(result, np.ndarray):
            return result.astype(float)
        if isinstance(result, Mapping) and "pointwise_log_likelihood" in result:
            return np.asarray(result["pointwise_log_likelihood"], dtype=float)
        raise TypeError("PyMC adapters must return an ndarray or {'pointwise_log_likelihood': ...}.")


ChoiceLikelihood = CategoricalLikelihood
ChoiceSampler = CategoricalSampler
call_choice_model = predict
