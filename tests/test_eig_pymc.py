"""Tests for the exhaustive PyMC EIG design (src.pipelines.outer_loop.eig).

`design_exhaustive` enumerates the full H/T pair universe over the given
lengths, scores every pair from each PyMC model's per-draw p_left (prior
predictive, or posterior predictive when a responses CSV is given), and
greedily selects the max-joint-EIG stimulus set. Each model computes its own
features from raw stimulus rows. This is the pipeline's only design mode —
there is no candidate-pool path. Uses prior-predictive sampling (fast-ish, no
NUTS) — heavier cases are marked slow to be safe.
"""

from __future__ import annotations

import shutil

import numpy as np
import pytest
import yaml

from src.pipelines.outer_loop import eig as eig_mod
from tests.paths import PYMC_MODEL_FIXTURES_DIR


def _design(*args, **kwargs):
    """``design_exhaustive`` with one response per stimulus: these tests check
    the design's mechanics, not the replicated-response objective
    (tests/test_eig_replicated_responses.py)."""
    return eig_mod.design_exhaustive(*args, n_responses=1, **kwargs)


# A model with a participant-level random effect: it needs a `participant_id`
# pm.Data column that stimulus feature rows (n_a/h_a/...) never carry. It fits
# fine on responses.csv (which has participant_id) but cannot be prior-predicted
# on a bare stimulus — the operation the exhaustive design needs — so it must be
# screened out of the EIG model set rather than crashing the whole design. This
# is the shape that killed the impossible-holdout cell run4/more_imbalance.
_PARTICIPANT_MODEL = """import numpy as np
import pymc as pm
import pytensor.tensor as pt

def compute_features(sequence_a, sequence_b):
    return {"h_a": sum(1 for c in sequence_a if c == "H"),
            "h_b": sum(1 for c in sequence_b if c == "H")}

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
        shutil.copyfile(
            PYMC_MODEL_FIXTURES_DIR / f"{name}.py", models_dir / f"{name}.py"
        )
    shutil.copyfile(
        PYMC_MODEL_FIXTURES_DIR / "models_manifest.yaml",
        models_dir / "models_manifest.yaml",
    )
    return models_dir


def _seed_with_participant_model(tmp_path):
    """Seed set + one carried-forward model that requires a participant_id column."""
    models_dir = _seed(tmp_path)
    (models_dir / "participant_re.py").write_text(_PARTICIPANT_MODEL, encoding="utf-8")
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


def test_design_drops_model_that_cannot_bind_to_stimulus(tmp_path, monkeypatch, capsys):
    """A model needing a non-stimulus column (participant_id) is screened out of
    the EIG set — loudly — instead of crashing the whole design; the selection
    runs over the remaining, stimulus-predictable models. The per-draw
    prior-predictive pass is stubbed so the screen (a real load + make_stim_data
    bind check) is what's exercised, not sampling."""
    import numpy as np

    models_dir = _seed_with_participant_model(tmp_path)

    seen: dict = {"names": []}
    # Distinct constant means -> a known EIG: 1 - H_b(0.8) = 0.278072 bits.
    means = {"bayesian_fair_coin": 0.8, "representativeness": 0.2}

    def fake_draws(model_names, models_dir, rows, *, n_samples=200, seed=42):
        seen["names"] += list(model_names)
        return {m: np.full((n_samples, len(rows)), means[m]) for m in model_names}

    monkeypatch.setattr(
        "src.models.pymc_inference.prior_predict_p_left_draws", fake_draws
    )

    out = _design(
        models_dir,
        lengths=(3,),
        n_select=2,
        n_samples=25,
        n_scenarios=200,
    )

    assert len(out) == 2
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

    A code error the model's own file did not raise (here the loader) means
    the harness is broken: dropping the model would quietly shrink the
    hypothesis space and renormalize EIG over the survivors, so the researcher
    would never learn their model never ran. Those raise. (A code error raised
    in the model's own file is the model's, and screened out on record: see
    tests/test_bad_candidate_never_ends_a_cell.py.)
    """

    def boom(name, models_dir):
        raise error

    monkeypatch.setattr("src.models.model_loading.load_pymc_model_cached", boom)
    with pytest.raises(RuntimeError, match="broken"):
        eig_mod._screen_usable_models(["m"], tmp_path, [{"sequence_a": "HT"}])


def test_screen_still_drops_an_unbindable_model_loudly(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        "src.models.model_loading.load_pymc_model_cached", lambda name, d: name
    )

    def fake_bind(model, rows):
        if model == "needs_participant":
            raise KeyError("participant_id")

    monkeypatch.setattr("src.models.data_binding.make_stim_data", fake_bind)
    usable, dropped = eig_mod._screen_usable_models(
        ["needs_participant", "fine"], tmp_path, [{"sequence_a": "HT"}]
    )
    assert usable == ["fine"]
    assert [d["model"] for d in dropped] == ["needs_participant"]
    assert "needs_participant" in capsys.readouterr().out


def test_design_raises_when_no_model_can_bind(tmp_path):
    """If no model can be evaluated on a stimulus, fail loudly rather than emit
    a meaningless design."""
    models_dir = tmp_path / "cognitive_models"
    models_dir.mkdir(parents=True)
    (models_dir / "participant_re.py").write_text(_PARTICIPANT_MODEL, encoding="utf-8")
    (models_dir / "models_manifest.yaml").write_text(
        yaml.safe_dump(
            {"models": [{"name": "participant_re", "rationale": "p."}]},
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="no models|cannot be evaluated|stimulus"):
        _design(models_dir, lengths=(3,), n_select=2)


def test_exhaustive_design_selects_joint_eig_set(tmp_path):
    """The CLI enumerates the pair universe, scores under the PyMC models, and
    greedily selects the max-joint-EIG set."""
    import json

    models_dir = _seed(tmp_path)
    out = tmp_path / "stimuli.json"
    args = eig_mod.Args(
        models_dir=models_dir,
        n_responses=1,
        out=out,
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
        assert set("HT") >= set(item["sequence_a"]) and len(item["sequence_a"]) in (
            3,
            4,
        )
        assert set("HT") >= set(item["sequence_b"]) and len(item["sequence_b"]) in (
            3,
            4,
        )
        assert len(item["sequence_a"]) == len(item["sequence_b"])
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
            n_responses=1,
            models_dir=models_dir,
            out=out2,
            lengths=(3, 4),
            select=5,
            n_samples=25,
            n_scenarios=300,
        )
    )
    assert json.loads(out2.read_text(encoding="utf-8")) == stimuli


def test_exhaustive_design_pure_random_no_eig(tmp_path):
    """n_select=0 -> a pure random-coverage set: N distinct pairs sampled from the
    pool with NO EIG computation (no model scoring), each marked source='random'
    with eig=None. This is the 64-random ablation."""
    models_dir = _seed(tmp_path)
    stimuli = _design(models_dir, lengths=(3, 4), n_select=0, n_random=6, seed=1)
    assert len(stimuli) == 6
    assert all(s["source"] == "random" for s in stimuli)
    assert all(s["eig"] is None for s in stimuli)
    keys = {(s["sequence_a"], s["sequence_b"]) for s in stimuli}
    assert len(keys) == 6  # distinct
    for s in stimuli:
        assert len(s["sequence_a"]) == len(s["sequence_b"]) and len(
            s["sequence_a"]
        ) in (3, 4)
    # deterministic given the seed
    again = _design(models_dir, lengths=(3, 4), n_select=0, n_random=6, seed=1)
    assert {(s["sequence_a"], s["sequence_b"]) for s in again} == keys


def test_exhaustive_design_eig_plus_random_split(tmp_path):
    """n_select>0 and n_random>0 -> EIG-selected half + random-coverage half,
    disjoint, EIG picks first with real eig, random picks tagged. This is the
    32-EIG + 32-random default."""
    models_dir = _seed(tmp_path)
    stimuli = _design(
        models_dir,
        lengths=(3, 4),
        n_select=3,
        n_random=4,
        n_samples=25,
        n_scenarios=300,
        seed=1,
    )
    assert len(stimuli) == 7
    eig_picks = [s for s in stimuli if s["source"] == "eig"]
    rand_picks = [s for s in stimuli if s["source"] == "random"]
    assert len(eig_picks) == 3 and len(rand_picks) == 4
    assert all(isinstance(s["eig"], float) and s["eig"] >= 0.0 for s in eig_picks)
    assert all(s["eig"] is None for s in rand_picks)
    keys = [(s["sequence_a"], s["sequence_b"]) for s in stimuli]
    assert len(set(keys)) == 7  # EIG and random halves are disjoint


def test_exhaustive_design_never_scores_cross_length_pairs(tmp_path, monkeypatch):
    import numpy as np

    models_dir = _seed(tmp_path)
    captured_rows = []

    def fake_draws(model_names, models_dir, rows, *, n_samples=200, seed=42):
        captured_rows.extend(rows)
        return {
            name: np.full((n_samples, len(rows)), probability)
            for name, probability in zip(model_names, (0.8, 0.2))
        }

    monkeypatch.setattr(
        "src.models.pymc_inference.prior_predict_p_left_draws", fake_draws
    )

    _design(
        models_dir,
        lengths=(3, 4),
        n_select=2,
        n_samples=5,
        n_scenarios=20,
    )

    assert captured_rows
    assert all(
        len(row["sequence_a"]) == len(row["sequence_b"]) for row in captured_rows
    )


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

    stimuli = _design(
        models_dir,
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
    # The models were actually fitted: the cache holds a posterior for each
    # (two, when a fit this short fails the convergence gate and is refit).
    assert {nc.name.split(".")[0] for nc in cache.glob("*.nc")} == {
        "bayesian_fair_coin",
        "representativeness",
    }
    # Round-trips through JSON like the design stage requires.
    json.dumps(stimuli)


def test_exhaustive_design_posterior_mode_missing_responses_fails_loudly(tmp_path):
    models_dir = _seed(tmp_path)
    with pytest.raises(FileNotFoundError, match="responses"):
        _design(
            models_dir,
            lengths=(3, 4),
            n_select=4,
            responses_csv=tmp_path / "nope.csv",
        )


def test_missing_manifest_raises(tmp_path):
    (tmp_path / "cognitive_models").mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        _design(tmp_path / "cognitive_models", lengths=(3,), n_select=2)


# ── screening: which drops are legitimate ───────────────────────────


def _probe_row(seq_a="HTH", seq_b="HHT"):
    """A raw stimulus row, as the design builds them."""
    return {"sequence_a": seq_a, "sequence_b": seq_b, "chose_left": 0}


def test_screen_drops_a_participant_level_model_and_reports_which(tmp_path, capsys):
    """The one legitimate drop: a model whose only unbindable columns are
    participant-level bookkeeping a stimulus row never carries. It is dropped,
    said out loud, AND returned so the caller can record it in the run
    artifacts."""
    models_dir = _seed_with_participant_model(tmp_path)

    usable, dropped = eig_mod._screen_usable_models(
        eig_mod._load_model_names(models_dir), models_dir, [_probe_row()]
    )

    assert "participant_re" not in usable
    assert "bayesian_fair_coin" in usable
    assert [d["model"] for d in dropped] == ["participant_re"]
    assert dropped[0]["missing"] == ["participant_id"]
    assert "participant_re" in capsys.readouterr().out


def test_screen_raises_when_model_needs_feature_columns_it_cannot_compute(tmp_path):
    """A model that expects precomputed feature columns (no compute_features
    hook) cannot bind a raw stimulus row. The screen must raise rather than
    silently dropping it, because dropping renormalizes EIG over whichever
    models happen to bind."""
    models_dir = tmp_path / "cognitive_models"
    models_dir.mkdir(parents=True)
    (models_dir / "no_hook.py").write_text(
        "import numpy as np, pymc as pm, pytensor.tensor as pt\n"
        "with pm.Model() as model:\n"
        "    n_a = pm.Data('n_a', np.zeros(1, dtype='int64'))\n"
        "    tau = pm.HalfNormal('tau', sigma=2.0)\n"
        "    p_left = pm.Deterministic('p_left', pm.math.sigmoid(tau * pt.cast(n_a, 'float64')))\n"
        "    chose_left = pm.Data('chose_left', np.zeros(1, dtype='int64'))\n"
        "    pm.Bernoulli('response', p=p_left, observed=chose_left)\n",
        encoding="utf-8",
    )
    (models_dir / "models_manifest.yaml").write_text(
        "models:\n  - name: no_hook\n    rationale: test model without compute_features\n",
        encoding="utf-8",
    )
    raw_row = {"sequence_a": "HTH", "sequence_b": "HHT", "chose_left": 0}

    with pytest.raises(RuntimeError, match="feature column"):
        eig_mod._screen_usable_models(
            eig_mod._load_model_names(models_dir), models_dir, [raw_row]
        )


def test_design_records_the_screened_out_models_as_an_artifact(tmp_path, monkeypatch):
    """A drop must survive in the run tree, not only in a log line nobody reads."""
    import numpy as np

    models_dir = _seed_with_participant_model(tmp_path)
    monkeypatch.setattr(
        "src.models.pymc_inference.prior_predict_p_left_draws",
        lambda names, d, rows, *, n_samples=200, seed=42: {
            m: np.full((n_samples, len(rows)), 0.5) for m in names
        },
    )
    out_path = tmp_path / "screened_out.json"

    _design(
        models_dir,
        lengths=(3,),
        n_select=1,
        n_samples=5,
        n_scenarios=20,
        screened_out_path=out_path,
    )

    import json

    recorded = json.loads(out_path.read_text(encoding="utf-8"))
    assert [d["model"] for d in recorded] == ["participant_re"]
    assert recorded[0]["missing"] == ["participant_id"]


# ── a model whose p_left is undefined on the design pool ──────────────
#
# A carried model can break on pairs unlike any it was trained on; one such
# model used to crash the design, identically on every retry. It is screened
# out of this design and recorded in screened_out.json with the reason.


def _undefined_on_long_pairs(names, d, rows, *, n_samples=200, seed=42):
    from src.models.pymc_inference import InvalidPredictions

    (name,) = names  # the design predicts one model at a time
    draws = np.full(
        (n_samples, len(rows)), 0.8 if name == "bayesian_fair_coin" else 0.2
    )
    if name == "representativeness":
        draws[:, [i for i, r in enumerate(rows) if len(r["sequence_a"]) == 3]] = np.nan
        raise InvalidPredictions("values must be finite and in [0, 1].", draws)
    return {name: draws}


def test_design_screens_out_a_model_with_an_undefined_p_left_and_records_why(
    tmp_path, monkeypatch, capsys
):
    import json

    models_dir = _seed_with_participant_model(tmp_path)
    monkeypatch.setattr(
        "src.models.pymc_inference.prior_predict_p_left_draws", _undefined_on_long_pairs
    )
    out_path = tmp_path / "screened_out.json"

    stimuli = _design(
        models_dir,
        lengths=(2, 3),
        n_select=2,
        n_samples=5,
        n_scenarios=20,
        screened_out_path=out_path,
    )

    assert len(stimuli) == 2
    recorded = {d["model"]: d for d in json.loads(out_path.read_text(encoding="utf-8"))}
    assert set(recorded) == {"participant_re", "representativeness"}
    bad = recorded["representativeness"]
    assert bad["invalid_pairs"] == 28  # every length-3 pair of the pool
    assert "prior-predictive p_left is undefined" in bad["reason"]
    assert "[screen] EIG: model 'representativeness'" in capsys.readouterr().out


def test_design_fails_loudly_when_no_model_has_a_defined_p_left(tmp_path, monkeypatch):
    from src.models.pymc_inference import InvalidPredictions

    models_dir = _seed_with_participant_model(tmp_path)

    def all_undefined(names, d, rows, *, n_samples=200, seed=42):
        raise InvalidPredictions("bad", np.full((n_samples, len(rows)), np.nan))

    monkeypatch.setattr(
        "src.models.pymc_inference.prior_predict_p_left_draws", all_undefined
    )
    with pytest.raises(ValueError, match="defined p_left"):
        _design(models_dir, lengths=(3,), n_select=2, n_samples=5, n_scenarios=20)


def test_posterior_design_screens_out_a_model_with_an_undefined_p_left(
    tmp_path, monkeypatch
):
    import src.models.data_binding as data_binding
    import src.models.pymc_inference as pymc_inference
    from src.models.pymc_inference import InvalidPredictions

    class Fitted:
        model = None

        def __init__(self, name):
            self.name = name

        def predict_p_left_draws(self, stim_data, *, seed, max_draws):
            if self.name == "bad":
                raise InvalidPredictions("bad", np.array([[np.nan, 0.5]]))
            return np.full((1, 2), 0.5)

    monkeypatch.setattr(pymc_inference, "fit_model", lambda name, *a, **k: Fitted(name))
    monkeypatch.setattr(pymc_inference, "model_sampler_settings", lambda n, d: {})
    monkeypatch.setattr(data_binding, "make_stim_data", lambda model, rows: {})
    rows = [
        {"sequence_a": "HT", "sequence_b": "HH"},
        {"sequence_a": "TT", "sequence_b": "TH"},
    ]

    draws, screened = eig_mod._posterior_p_left_draws(
        ["good", "bad"],
        tmp_path,
        rows,
        responses_csv=tmp_path / "r.csv",
        fit_cache_dir=None,
        max_draws=10,
        seed=0,
    )

    assert list(draws) == ["good"]
    (entry,) = screened
    assert entry["model"] == "bad" and entry["invalid_pairs"] == 1
    assert "HT vs HH" in entry["reason"]
