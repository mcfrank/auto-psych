"""scripts/paper/parameter_summaries.py: what counts as a population and a
person-level parameter, and which participant slots are summarised."""

import arviz as az
import numpy as np

from scripts.paper.parameter_summaries import summarize_fit


def _idata(n_trials=6, n_slots=10):
    rng = np.random.default_rng(0)
    chains, draws = 2, 50
    person = rng.normal(size=(chains, draws, n_slots))
    person[..., :3] += np.array([1.0, 2.0, 3.0])  # slots 0-2 are in the data
    return az.from_dict(
        posterior={
            "beta": rng.normal(5.0, 0.1, size=(chains, draws)),
            "weights": rng.normal(size=(chains, draws, 2)),
            "ideal": person,
            "p_left": rng.uniform(size=(chains, draws, n_trials)),
        },
        constant_data={"participant_id": np.array([0, 0, 1, 1, 2, 2])},
    )


def test_population_rows_are_scalars_and_vector_elements_without_trial_predictions():
    population, _ = summarize_fit(_idata(), n_trials=6)
    assert list(population["parameter"]) == ["beta", "weights[0]", "weights[1]"]
    beta = population.set_index("parameter").loc["beta"]
    assert abs(beta["mean"] - 5.0) < 0.05


def test_person_level_rows_summarise_only_the_participants_in_the_data():
    _, person = summarize_fit(_idata(), n_trials=6)
    assert list(person["parameter"]) == ["ideal"]
    row = person.iloc[0]
    assert row["n_participants"] == 3
    assert abs(row["mean_of_means"] - 2.0) < 0.2


def test_a_model_without_participant_data_has_no_person_level_rows():
    idata = _idata()
    del idata.constant_data
    population, person = summarize_fit(idata, n_trials=6)
    assert person.empty
    assert "ideal[9]" in set(population["parameter"])
