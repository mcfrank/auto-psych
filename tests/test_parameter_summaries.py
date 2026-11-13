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


def test_the_number_game_fit_set_is_every_winner_and_seed_on_all_runs_pooled(tmp_path):
    """As ``sr_pooled``: the pooled figure data's winners and seeds, read from
    the figure-data work directory's ``pooled/`` at the pooled settings."""
    import json

    from scripts.paper import parameter_summaries as ps

    settings = {"method": "nuts_no_jitter", "tune": 4000, "draws": 1000, "chains": 4,
                "target_accept": 0.95, "random_seed": 20261018}
    figure = tmp_path / "number_game.json"
    figure.write_text(json.dumps({
        "pooled_fit_settings": settings,
        "pooled": {"winner_runs": {"w1": "run1", "w2": "run2"},
                   "elpd": {"w1": -1.0, "w2": -2.0, "s1": -3.0, "s2": -4.0}},
    }))
    args = ps.Args(sr_work_dir=tmp_path / "sr", number_game_work_dir=tmp_path / "ng",
                   number_game_figure_data=figure)

    fit_set = ps.number_game_fit_set(args)

    assert fit_set.label == "number_game_pooled"
    assert fit_set.roles == {"w1": "winner", "w2": "winner", "s1": "seed", "s2": "seed"}
    assert fit_set.models_dir == tmp_path / "ng" / "pooled" / "models"
    assert fit_set.responses == tmp_path / "ng" / "pooled" / "responses.csv"
    assert fit_set.cache_dir == tmp_path / "ng" / "pooled" / "fit_cache"
    assert fit_set.fit_kwargs == settings
