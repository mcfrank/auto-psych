from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol, runtime_checkable

import numpy as np


Params = list[float] | None
ModelCallable = Callable[[Any, list[str], Params], Mapping[str, float]]


@dataclass(frozen=True)
class Trial:
    stimulus: Any
    response: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class Dataset:
    trials: list[Trial]
    response_options: list[str]
    label: str = "dataset"

    def __len__(self) -> int:
        return len(self.trials)


@runtime_checkable
class Likelihood(Protocol):
    def __call__(self, model: ModelCallable, data: Dataset, params: Params = None, **kw: Any) -> np.ndarray:
        """Return per-trial log likelihoods with shape ``(len(data),)``."""


@runtime_checkable
class Sampler(Protocol):
    def __call__(
        self,
        model: ModelCallable,
        data: Dataset,
        params: Params = None,
        n_samples: int = 1,
        *,
        base_seed: int = 0,
        **kwargs: Any,
    ) -> list[list[Any]]:
        """Return synthetic observations grouped by trial."""


def normalize_probs(probs: Mapping[str, float], response_options: list[str]) -> dict[str, float]:
    values = {option: max(0.0, float(probs.get(option, 0.0))) for option in response_options}
    total = sum(values.values())
    if total <= 0:
        return {option: 1.0 / len(response_options) for option in response_options}
    return {option: value / total for option, value in values.items()}


ChoiceTrial = Trial
