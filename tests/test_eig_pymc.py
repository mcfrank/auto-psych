"""Tests for the PyMC EIG/design annotator.

`annotate` scores candidate stimuli by expected information gain over the PyMC
model set, using prior-predictive p_left (no MCMC fit). It featurizes each raw
stimulus via the project's `featurize_stimulus` before handing it to the models.
Uses prior-predictive sampling (fast-ish, no NUTS) — marked slow to be safe.
"""

from __future__ import annotations

import shutil

import pytest
import yaml

from src.pipelines.outer_loop import eig as eig_mod
from tests.paths import PYMC_MODEL_FIXTURES_DIR, REPO_ROOT

FEATURIZE = (
    REPO_ROOT / "src/pipelines/outer_loop/projects/subjective_randomness/preprocess.py"
)

# A model with a participant-level random effect: it needs a `participant_id`
# pm.Data column that stimulus feature rows (n_a/h_a/...) never carry. It fits
# fine on responses.csv (which has participant_id) but cannot be prior-predicted
# on a bare stimulus — the operation EIG/design needs — so it must be screened
# out of the EIG model set rather than crashing the whole annotation. This is the
# shape that killed the impossible-holdout cell run4/more_imbalance.
_PARTICIPANT_MODEL = """import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    h_a = pm.Data("h_a", np.zeros(1, dtype="int64"))
    h_b = pm.Data("h_b", np.zeros(1, dtype="int64"))
    participant_id = pm.Data("participant_id", np.zeros(1, dtype="int64"))

    sigma_u = pm.HalfNormal("sigma_u", sigma=1.0)
    u = pm.Normal("u", mu=0.0, sigma=sigma_u, shape=64)
    tau = pm.HalfNormal("tau", sigma=2.0)

    score = pt.cast(h_a - h_b, "float64")
    p_left = pm.Deterministic(
        "p_left", pm.math.sigmoid(tau * score + u[participant_id])
    )
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
"""


def _seed(tmp_path):
    models_dir = tmp_path / "cognitive_models"
    models_dir.mkdir(parents=True)
    for name in ("bayesian_fair_coin", "representativeness"):
        shutil.copyfile(PYMC_MODEL_FIXTURES_DIR / f"{name}.py", models_dir / f"{name}.py")
    shutil.copyfile(
        PYMC_MODEL_FIXTURES_DIR / "models_manifest.yaml", models_dir / "models_manifest.yaml"
    )
    return models_dir


def _seed_with_participant_model(tmp_path):
    """Seed set + one carried-forward model that requires a participant_id column."""
    models_dir = _seed(tmp_path)
    (models_dir / "participant_re.py").write_text(
        _PARTICIPANT_MODEL, encoding="utf-8"
    )
    manifest = yaml.safe_load(
        (models_dir / "models_manifest.yaml").read_text(encoding="utf-8")
    )
    manifest["models"].append(
        {"name": "participant_re", "rationale": "participant random effect."}
    )
    (models_dir / "models_manifest.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
    )
    return models_dir


def test_eig_rejects_a_manifest_with_a_missing_model_file(tmp_path):
    """EIG must never renormalize over only the model files that happen to exist."""
    models_dir = tmp_path / "cognitive_models"
    models_dir.mkdir()
    (models_dir / "present.py").write_text("", encoding="utf-8")
    (models_dir / "models_manifest.yaml").write_text(
        "models:\n  - name: present\n  - name: missing_model\n",
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError, match="missing_model"):
        eig_mod._load_model_names(models_dir)


def test_explicit_missing_registry_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="registry"):
        eig_mod._load_model_weights(tmp_path / "missing_registry.yaml")


def test_eig_registry_rejects_negative_weights(tmp_path):
    registry = tmp_path / "model_registry.yaml"
    registry.write_text(
        "theories:\n  a: -0.5\n  b: 1.5\nreserved_for_new: 0.0\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="a"):
        eig_mod._load_model_weights(registry)


def test_annotate_drops_model_that_cannot_bind_to_stimulus(
    tmp_path, monkeypatch, capsys
):
    """A model needing a non-stimulus column (participant_id) is screened out of
    the EIG set — loudly — instead of crashing the whole annotation; the EIG runs
    over the remaining, stimulus-predictable models. The batched prior-predictive
    pass is stubbed so the screen (a real load + make_stim_data bind check) is
    what's exercised, not sampling."""
    import numpy as np

    models_dir = _seed_with_participant_model(tmp_path)

    seen: dict = {}

    def fake_batch(model_names, models_dir, rows, **kwargs):
        seen["names"] = list(model_names)
        # Distinct means -> a known EIG: 1 - H_b(0.8) = 0.278072 bits.
        return {m: np.array([p] * len(rows)) for m, p in zip(model_names, (0.8, 0.2))}

    monkeypatch.setattr(
        "src.models.pymc_inference.prior_predict_p_left_batch", fake_batch
    )

    out = eig_mod.annotate(
        [{"sequence_a": "HHHT", "sequence_b": "HTHT"}],
        models_dir,
        featurize_path=FEATURIZE,
    )

    assert out[0]["eig"] == pytest.approx(0.278072)
    # The participant-requiring model was screened out before EIG; the others stay.
    assert "participant_re" not in seen["names"]
    assert "bayesian_fair_coin" in seen["names"]
    assert "representativeness" in seen["names"]
    assert "participant_re" in capsys.readouterr().out  # dropped loudly


@pytest.mark.parametrize(
    "error",
    [
        ImportError("No module named 'sklearn'"),
        SyntaxError("invalid syntax"),
        AttributeError("module 'pymc' has no attribute 'Nomral'"),
    ],
    ids=["import", "syntax", "attribute"],
)
def test_screen_raises_when_a_model_is_simply_broken(tmp_path, monkeypatch, error):
    """Screening exists to drop models that cannot *bind to a stimulus row*.

    Broken code is a different failure: dropping it would quietly shrink the
    hypothesis space and renormalize EIG over the survivors, so the researcher
    would never learn their model never ran. Those raise.
    """

    def boom(name, models_dir):
        raise error

    monkeypatch.setattr("src.models.pymc_inference.load_pymc_model_cached", boom)
    with pytest.raises(RuntimeError, match="broken"):
        eig_mod._screen_usable_models(["m"], tmp_path, {"sequence_a": "HT"})


def test_screen_still_drops_an_unbindable_model_loudly(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        "src.models.pymc_inference.load_pymc_model_cached", lambda name, d: name
    )

    def fake_bind(model, rows):
        if model == "needs_participant":
            raise KeyError("participant_id")

    monkeypatch.setattr("src.models.pymc_inference.make_stim_data", fake_bind)
    usable = eig_mod._screen_usable_models(
        ["needs_participant", "fine"], tmp_path, {"sequence_a": "HT"}
    )
    assert usable == ["fine"]
    assert "needs_participant" in capsys.readouterr().out


def test_annotate_raises_when_no_model_can_bind(tmp_path):
    """If no model can be evaluated on a stimulus, fail loudly rather than emit
    meaningless EIGs."""
    models_dir = tmp_path / "cognitive_models"
    models_dir.mkdir(parents=True)
    (models_dir / "participant_re.py").write_text(
        _PARTICIPANT_MODEL, encoding="utf-8"
    )
    (models_dir / "models_manifest.yaml").write_text(
        yaml.safe_dump(
            {"models": [{"name": "participant_re", "rationale": "p."}]},
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="no models|cannot be evaluated|stimulus"):
        eig_mod.annotate(
            [{"sequence_a": "HHHT", "sequence_b": "HTHT"}],
            models_dir,
            featurize_path=FEATURIZE,
        )


@pytest.mark.slow
def test_annotate_adds_nonnegative_eig_and_sorts(tmp_path):
    models_dir = _seed(tmp_path)
    candidates = [
        {"sequence_a": "HHHHH", "sequence_b": "HHHHH"},  # identical → low EIG
        {"sequence_a": "HHHHHHHH", "sequence_b": "HTHTHTHT"},  # discriminating
    ]
    out = eig_mod.annotate(
        candidates, models_dir, featurize_path=FEATURIZE, n_samples=100
    )

    assert len(out) == 2
    for item in out:
        assert "eig" in item
        assert 0.0 <= item["eig"] <= 1.0  # 2 models → ≤ log2(2) = 1 bit
        assert "sequence_a" in item and "sequence_b" in item
    # Sorted descending by EIG.
    assert out[0]["eig"] >= out[1]["eig"]


@pytest.mark.slow
def test_annotate_featurizes_so_models_can_read_columns(tmp_path):
    """Without featurization the models' pm.Data columns are absent → this
    proves the annotator derives n_a/h_a/... from raw sequences."""
    models_dir = _seed(tmp_path)
    candidates = [{"sequence_a": "HHHT", "sequence_b": "HTHT"}]
    out = eig_mod.annotate(
        candidates, models_dir, featurize_path=FEATURIZE, n_samples=100
    )
    assert out[0]["eig"] >= 0.0


def test_exhaustive_design_selects_joint_eig_set(tmp_path):
    """--exhaustive mode: enumerate the pair universe, score under the PyMC
    models, and greedily select the max-joint-EIG set — no candidates file and
    no agent-conjectured stimuli involved."""
    import json

    models_dir = _seed(tmp_path)
    out = tmp_path / "stimuli.json"
    args = eig_mod.Args(
        candidates=None,
        models_dir=models_dir,
        featurize=FEATURIZE,
        out=out,
        exhaustive=True,
        lengths=(3, 4),
        select=5,
        n_samples=25,
        n_scenarios=300,
    )
    eig_mod.main(args)

    stimuli = json.loads(out.read_text(encoding="utf-8"))
    assert len(stimuli) == 5
    keys = set()
    for rank, item in enumerate(stimuli, start=1):
        assert set("HT") >= set(item["sequence_a"]) and len(item["sequence_a"]) in (3, 4)
        assert set("HT") >= set(item["sequence_b"]) and len(item["sequence_b"]) in (3, 4)
        assert isinstance(item["eig"], float) and item["eig"] >= 0.0
        assert item["selection_rank"] == rank
        assert 0.0 <= item["joint_eig_bits"] <= 1.0 + 1e-9  # 2 models -> <= 1 bit
        keys.add((item["sequence_a"], item["sequence_b"]))
    assert len(keys) == 5
    # Joint EIG grows along the greedy order.
    assert stimuli[-1]["joint_eig_bits"] >= stimuli[0]["joint_eig_bits"] - 0.02

    # Deterministic: rerunning produces the same design.
    out2 = tmp_path / "stimuli2.json"
    eig_mod.main(
        eig_mod.Args(
            candidates=None, models_dir=models_dir, featurize=FEATURIZE, out=out2,
            exhaustive=True, lengths=(3, 4), select=5, n_samples=25, n_scenarios=300,
        )
    )
    assert json.loads(out2.read_text(encoding="utf-8")) == stimuli


@pytest.mark.slow
def test_exhaustive_design_posterior_mode_scores_from_fitted_models(tmp_path):
    """With a responses CSV, exhaustive design scores the pool from each
    model's *posterior*-predictive p_left (fit on those responses) instead of
    the prior — no pure-Python twin involved.

    Marked slow: unlike the prior-only tests above, this one runs real NUTS
    (two chains per model), which is what every other MCMC test in the suite is
    marked for."""
    import json

    models_dir = _seed(tmp_path)
    responses = PYMC_MODEL_FIXTURES_DIR / "responses.csv"
    cache = tmp_path / "fit_cache"

    stimuli = eig_mod.design_exhaustive(
        models_dir,
        featurize_path=FEATURIZE,
        lengths=(3, 4),
        n_select=4,
        n_samples=50,
        n_scenarios=300,
        responses_csv=responses,
        fit_cache_dir=cache,
        fit_draws=100,
        fit_tune=100,
        fit_chains=2,
    )

    assert len(stimuli) == 4
    for rank, item in enumerate(stimuli, start=1):
        assert isinstance(item["eig"], float) and item["eig"] >= 0.0
        assert item["selection_rank"] == rank
    # The models were actually fitted (cache holds one posterior per model).
    assert len(list(cache.glob("*.nc"))) == 2
    # Round-trips through JSON like the design stage requires.
    json.dumps(stimuli)


def test_exhaustive_design_posterior_mode_missing_responses_fails_loudly(tmp_path):
    models_dir = _seed(tmp_path)
    with pytest.raises(FileNotFoundError, match="responses"):
        eig_mod.design_exhaustive(
            models_dir,
            featurize_path=FEATURIZE,
            lengths=(3, 4),
            n_select=4,
            responses_csv=tmp_path / "nope.csv",
        )


def test_exhaustive_mode_argument_validation(tmp_path):
    models_dir = _seed(tmp_path)
    # --exhaustive needs --select.
    with pytest.raises(SystemExit):
        eig_mod.main(
            eig_mod.Args(candidates=None, models_dir=models_dir, exhaustive=True)
        )
    # Legacy mode needs --candidates.
    with pytest.raises(SystemExit):
        eig_mod.main(eig_mod.Args(candidates=None, models_dir=models_dir))
    # The two modes are mutually exclusive.
    with pytest.raises(SystemExit):
        eig_mod.main(
            eig_mod.Args(
                candidates=tmp_path / "candidates.json", models_dir=models_dir,
                exhaustive=True, select=5,
            )
        )
    # --responses (posterior design) only makes sense with --exhaustive.
    with pytest.raises(SystemExit):
        eig_mod.main(
            eig_mod.Args(
                candidates=tmp_path / "candidates.json", models_dir=models_dir,
                responses=PYMC_MODEL_FIXTURES_DIR / "responses.csv",
            )
        )


def test_missing_manifest_raises(tmp_path):
    (tmp_path / "cognitive_models").mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        eig_mod.annotate(
            [{"sequence_a": "H", "sequence_b": "T"}], tmp_path / "cognitive_models"
        )
