import numpy as np

from src.pipelines.inner_loop.adapters import subjective_randomness_dataset
from src.pipelines.inner_loop.core import Dataset, Trial
from src.pipelines.inner_loop.diagnostics import build_candidate_diagnostics
from src.pipelines.inner_loop.fitting import fit_model
from src.pipelines.inner_loop.likelihood import CategoricalLikelihood, CategoricalSampler


def biased_model(stimulus, response_options, params=None):
    p_left = params[0] if params else 0.75
    return {response_options[0]: p_left, response_options[1]: 1 - p_left}


def test_categorical_loop_is_dataset_generic():
    data = Dataset(
        [Trial({"x": 1}, "left"), Trial({"x": 2}, "right")],
        ["left", "right"],
    )

    fit = fit_model(
        biased_model,
        data,
        likelihood=CategoricalLikelihood(),
        sampler=CategoricalSampler(),
        n_samples=2,
        initial_params=[0.5],
        param_bounds=[(0.01, 0.99)],
        n_starts=1,
        max_steps=5,
    )
    diagnostics = build_candidate_diagnostics("m", data, biased_model, fit.params)

    assert fit.per_trial_ll.shape == (2,)
    assert len(fit.sample_populations) == 2
    assert diagnostics.aggregate.n_trials == 2


def test_subjective_randomness_adapter_is_only_an_edge_mapping():
    data = subjective_randomness_dataset(
        [{"sequence_a": "HTHT", "sequence_b": "HHHH", "chose_left": "1"}]
    )

    assert data.response_options == ["left", "right"]
    assert data.trials[0].stimulus == {"sequence_a": "HTHT", "sequence_b": "HHHH"}
    assert data.trials[0].response == "left"
