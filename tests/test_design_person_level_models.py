"""A design after data predicts a person-level model as new participants.

The design scores pairs for new participants, so it used to drop every model
that binds ``participant_id`` (a participant effect): a stimulus row carries no
participant. On real human data every leading model has one. In experiment 1
of the October 2026 live run the carried set was a single such model, and the
experiment-2 design stopped with "No models ... can be evaluated".

A design after data (experiments >= 2, posterior predictive) now predicts such
a model as ``DESIGN_NEW_PARTICIPANTS`` participants its data have not reached:
ids past every training id, whose parameters no data informed, so in each
posterior draw they are draws from the model's own population distribution.
Their average, draw by draw, marginalizes the person-level parameters: the
model's prediction for a new participant (user decision 2026-09-30, rather
than averaging over the training participants, which would substitute the
sample for the model's population). A model that cannot do this — no spare
slot, or a hook that renumbers participants — is left out of that design, on
record. The prior design (experiment 1) still screens such models out; a
model that binds ``trial_index`` is screened out either way.
"""

from __future__ import annotations

import json
import shutil

import numpy as np
import pytest

import src.models.pymc_inference as pymc_inference
from src.models.data_binding import as_participant, make_stim_data
from src.models.model_loading import load_pymc_model
from src.pipelines.outer_loop import eig as eig_mod
from tests.test_eig_pymc import _PARTICIPANT_MODEL

TRAINING = [0, 3, 7]
NEW = [8, 9, 10]  # past every training id, with DESIGN_NEW_PARTICIPANTS = 3


def _write_set(models_dir, models):
    models_dir.mkdir(parents=True)
    for name, source in models.items():
        (models_dir / f"{name}.py").write_text(source, encoding="utf-8")
    (models_dir / "models_manifest.yaml").write_text(
        "models:\n"
        + "".join(f"  - name: {n}\n    rationale: {n} hypothesis.\n" for n in models),
        encoding="utf-8",
    )


def _responses(tmp_path):
    path = tmp_path / "responses.csv"
    path.write_text(
        "sequence_a,sequence_b,participant_id,trial_index,chose_left\n"
        + "".join(f"HHT,HTH,{pid},0,1\nTTH,THT,{pid},1,0\n" for pid in TRAINING),
        encoding="utf-8",
    )
    return path


class _Fitted:
    """A fit whose per-draw p_left is a known function of the participant;
    ``slots`` participants have parameters (an index past them raises, as
    PyTensor does)."""

    slots = 64

    def __init__(self, name, models_dir):
        self.name = name
        self.model = load_pymc_model(name, models_dir)
        self.participants_predicted = []

    def predict_p_left_draws(self, stim_data, *, seed, max_draws):
        n_rows = len(stim_data["chose_left"])
        if "participant_id" not in stim_data:
            return np.full((2, n_rows), 0.5)
        (pid,) = set(np.asarray(stim_data["participant_id"]).tolist())
        if pid >= self.slots:
            raise IndexError(f"index {pid} is out of bounds for axis 0 with size {self.slots}")
        self.participants_predicted.append(pid)
        draw_offsets = np.array([[0.0], [0.1]])  # two posterior draws
        return np.full((2, n_rows), 0.1 + 0.01 * pid) + draw_offsets


@pytest.fixture
def stub_fits(monkeypatch):
    monkeypatch.setattr(eig_mod, "DESIGN_NEW_PARTICIPANTS", 3)
    fits = {}

    def fit_model(name, models_dir, responses_csv, **kwargs):
        fits[name] = _Fitted(name, models_dir)
        return fits[name]

    monkeypatch.setattr(pymc_inference, "fit_model", fit_model)
    return fits


def _person_and_population(tmp_path):
    from tests.paths import PYMC_MODEL_FIXTURES_DIR

    models_dir = tmp_path / "cognitive_models"
    _write_set(models_dir, {"person_a": _PARTICIPANT_MODEL})
    shutil.copyfile(
        PYMC_MODEL_FIXTURES_DIR / "bayesian_fair_coin.py",
        models_dir / "bayesian_fair_coin.py",
    )
    (models_dir / "models_manifest.yaml").write_text(
        "models:\n  - name: person_a\n    rationale: a.\n"
        "  - name: bayesian_fair_coin\n    rationale: b.\n",
        encoding="utf-8",
    )
    return models_dir


def _posterior_draws(models_dir, tmp_path, names=("person_a", "bayesian_fair_coin")):
    return eig_mod._posterior_p_left_draws(
        list(names),
        models_dir,
        [{"sequence_a": "HHT", "sequence_b": "HTH", "chose_left": 0}],
        responses_csv=_responses(tmp_path),
        fit_cache_dir=None,
        max_draws=2,
        seed=0,
    )


def test_a_design_after_data_predicts_a_person_level_model_as_new_participants(
    tmp_path, stub_fits, monkeypatch
):
    models_dir = tmp_path / "cognitive_models"
    _write_set(models_dir, {"person_a": _PARTICIPANT_MODEL, "person_b": _PARTICIPANT_MODEL})
    captured = {}

    def capture(draws, n_select, **kwargs):
        captured["draws"] = {m: np.array(d) for m, d in draws.items()}
        return [(i, 0.0, "eig") for i in range(n_select)]

    monkeypatch.setattr(eig_mod, "select_design_picks", capture)
    screened = tmp_path / "screened_out.json"

    stimuli = eig_mod.design_exhaustive(
        models_dir,
        lengths=(3,),
        n_select=2,
        n_samples=2,
        n_scenarios=10,
        n_responses=40,
        responses_csv=_responses(tmp_path),
        screened_out_path=screened,
    )

    # Every model was used (the carried set of run 1 was only such models).
    assert len(stimuli) == 2
    assert json.loads(screened.read_text(encoding="utf-8")) == []
    assert set(captured["draws"]) == {"person_a", "person_b"}
    # Predicted as participants past every training id, averaged draw by draw.
    assert stub_fits["person_a"].participants_predicted == NEW
    draws = captured["draws"]["person_a"]
    assert draws.shape == (2, 28)  # two draws over the 28 length-3 pairs
    np.testing.assert_allclose(draws[0], 0.1 + 0.01 * np.mean(NEW))
    np.testing.assert_allclose(draws[1], 0.2 + 0.01 * np.mean(NEW))


def test_a_population_model_beside_it_is_predicted_once(tmp_path, stub_fits):
    draws, screened = _posterior_draws(_person_and_population(tmp_path), tmp_path)

    assert screened == []
    np.testing.assert_allclose(draws["bayesian_fair_coin"], 0.5)
    assert stub_fits["bayesian_fair_coin"].participants_predicted == []
    np.testing.assert_allclose(
        draws["person_a"][:, 0], np.array([0.1, 0.2]) + 0.01 * np.mean(NEW)
    )


def test_a_model_without_spare_slots_is_screened_out_on_record(
    tmp_path, stub_fits, monkeypatch
):
    monkeypatch.setattr(_Fitted, "slots", 9)  # participants 0-8 only

    draws, screened = _posterior_draws(_person_and_population(tmp_path), tmp_path)

    assert list(draws) == ["bayesian_fair_coin"]
    (entry,) = screened
    assert entry["model"] == "person_a"
    assert "no slot for participant 9" in entry["reason"]
    assert "spare slots" in entry["reason"]


def test_an_index_error_for_a_training_participant_too_is_the_models_own(
    tmp_path, stub_fits, monkeypatch
):
    monkeypatch.setattr(_Fitted, "slots", 5)  # not even training participant 7

    with pytest.raises(IndexError):
        _posterior_draws(_person_and_population(tmp_path), tmp_path)


# Binds participants through a hook that numbers them 0..n-1 by first
# appearance: a new participant would land on a training participant's slot.
_RENUMBERING_MODEL = '''import numpy as np
import pymc as pm

def prepare_observed(rows):
    codes = {}
    for row in rows:
        codes.setdefault(int(row["participant_id"]), len(codes))
    return {
        "h_a": np.array([row["sequence_a"].count("H") for row in rows], dtype="int64"),
        "h_b": np.array([row["sequence_b"].count("H") for row in rows], dtype="int64"),
        "participant_id": np.array(
            [codes[int(row["participant_id"])] for row in rows], dtype="int64"
        ),
        "chose_left": np.array([int(row["chose_left"]) for row in rows], dtype="int64"),
    }

with pm.Model() as model:
    h_a = pm.Data("h_a", np.zeros(1, dtype="int64"))
    h_b = pm.Data("h_b", np.zeros(1, dtype="int64"))
    participant_id = pm.Data("participant_id", np.zeros(1, dtype="int64"))
    u = pm.Normal("u", mu=0.0, sigma=1.0, shape=64)
    tau = pm.HalfNormal("tau", sigma=2.0)
    p_left = pm.Deterministic(
        "p_left", pm.math.sigmoid(tau * (h_a - h_b) + u[participant_id])
    )
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
'''


def test_a_model_that_renumbers_participants_is_screened_out_on_record(tmp_path):
    models_dir = tmp_path / "cognitive_models"
    _write_set(models_dir, {"person_a": _PARTICIPANT_MODEL, "renumbers": _RENUMBERING_MODEL})
    row = {"sequence_a": "HTH", "sequence_b": "HHT", "chose_left": 0}

    usable, dropped = eig_mod._screen_usable_models(
        ["person_a", "renumbers"], models_dir, [row], responses_csv=_responses(tmp_path)
    )

    assert usable == ["person_a"]
    assert [d["model"] for d in dropped] == ["renumbers"]
    assert "participant_id through unchanged" in dropped[0]["reason"]


def test_the_prior_design_still_screens_out_a_person_level_model(tmp_path):
    """Before any data the design does not predict participants (yet)."""
    models_dir = tmp_path / "cognitive_models"
    _write_set(models_dir, {"person_a": _PARTICIPANT_MODEL})
    row = {"sequence_a": "HTH", "sequence_b": "HHT", "chose_left": 0}

    with pytest.raises(ValueError, match="No models"):
        eig_mod._screen_usable_models(["person_a"], models_dir, [row])


_TRIAL_MODEL = _PARTICIPANT_MODEL.replace(
    'participant_id = pm.Data("participant_id", np.zeros(1, dtype="int64"))',
    'participant_id = pm.Data("participant_id", np.zeros(1, dtype="int64"))\n'
    '    trial_index = pm.Data("trial_index", np.zeros(1, dtype="int64"))',
).replace("u[participant_id])", "u[participant_id] + 0.01 * trial_index)")


def test_a_model_that_binds_trial_index_is_still_screened_out_after_data(tmp_path):
    models_dir = tmp_path / "cognitive_models"
    _write_set(models_dir, {"person_a": _PARTICIPANT_MODEL, "by_trial": _TRIAL_MODEL})
    row = {"sequence_a": "HTH", "sequence_b": "HHT", "chose_left": 0}

    usable, dropped = eig_mod._screen_usable_models(
        ["person_a", "by_trial"], models_dir, [row], responses_csv=_responses(tmp_path)
    )

    assert usable == ["person_a"]
    assert [d["model"] for d in dropped] == ["by_trial"]
    assert "trial_index" in dropped[0]["missing"]


def test_an_unobserved_slot_predicts_with_its_own_parameters_and_one_past_the_end_raises(
    tmp_path,
):
    """The mechanics H rests on, with real PyTensor (no MCMC): predicting as a
    participant id uses that slot's posterior draws, and an id past the
    vector raises IndexError (what ``_new_participant_draws`` catches)."""
    import arviz as az

    from src.models.pymc_inference import FittedModel

    models_dir = tmp_path / "cognitive_models"
    _write_set(models_dir, {"person_a": _PARTICIPANT_MODEL})
    model = load_pymc_model("person_a", models_dir)
    n_draws = 4
    u = np.zeros((1, n_draws, 64))
    u[0, :, 9] = 2.0
    idata = az.from_dict(
        posterior={
            "sigma_u": np.ones((1, n_draws)),
            "u": u,
            "tau": np.ones((1, n_draws)),
        }
    )
    fitted = FittedModel(name="person_a", model=model, idata=idata, fingerprint="test")
    rows = [{"sequence_a": "HHT", "sequence_b": "HTT", "chose_left": 0}]  # h_a - h_b = 1

    def predict(pid):
        stim_data = make_stim_data(model, as_participant(rows, pid))
        return fitted.predict_p_left_draws(stim_data, seed=0, max_draws=n_draws)

    sigmoid = lambda x: 1 / (1 + np.exp(-x))  # noqa: E731
    np.testing.assert_allclose(predict(8), sigmoid(1.0))
    np.testing.assert_allclose(predict(9), sigmoid(3.0))
    with pytest.raises(IndexError):
        predict(64)
