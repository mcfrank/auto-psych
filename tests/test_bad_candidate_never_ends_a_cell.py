"""One bad candidate must not end a cell (second audit B7; first audit R4, D5).

A candidate's own defects become a rejection with the reason — never an
exception that escapes the inner loop and ends the holdout cell, identically
on every retry:

- a code error in the candidate's own file (a ``NameError`` in its
  ``compute_features``) at the initial-point logp check, in admission and in
  the concurrent prefit of a wave;
- a missing or misshaped ``p_left`` in the novelty gate;
- a carried model whose features cannot be computed on short pairs: screened
  out of the design (lengths 2-3) on record, and left out of the evaluation
  (lengths 1-3) like an undefined ``p_left``.

A failure of the harness or the machine is not the model's, and still raises.
The models here are real PyMC files; only MCMC is replaced (a prior sample
stands in for the posterior), so no NUTS runs.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.models import pymc_inference
from src.models.model_loading import clear_model_cache, load_pymc_model
from src.models.pymc_inference import FittedModel, model_logp_is_finite
from src.pipelines.inner_loop import model_zoo
from src.pipelines.inner_loop.hypothesis_ledger import HypothesisLedger
from src.pipelines.outer_loop import eig as eig_mod
from src.subjective_randomness.holdout_eval import _eval_prediction

_GOOD = """
import numpy as np
import pymc as pm


def compute_features(sequence_a, sequence_b):
    return {"h_diff": (sequence_a.count("H") - sequence_b.count("H")) / len(sequence_a)}


with pm.Model() as model:
    h_diff = pm.Data("h_diff", np.zeros(1))
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    beta = pm.Normal("beta", 0.0, 1.0)
    p_left = pm.Deterministic("p_left", pm.math.sigmoid(beta * h_diff))
    pm.Bernoulli("obs", p=p_left, observed=chose_left)
"""

# A typo in the candidate's own hook: NameError at binding time, after the
# module loaded fine.
_TYPO = _GOOD.replace(
    'return {"h_diff": (sequence_a.count("H")',
    'return {"h_diff": (count_heads(sequence_a)',
)

# No p_left at all: the likelihood is fine, the prediction contract is not.
_NO_P_LEFT = _GOOD.replace('pm.Deterministic("p_left",', 'pm.Deterministic("prob",')

# One p_left for all trials instead of one per stimulus.
_SCALAR_P_LEFT = _GOOD.replace(
    'p_left = pm.Deterministic("p_left", pm.math.sigmoid(beta * h_diff))',
    'p_left = pm.Deterministic("p_left", pm.math.sigmoid(beta))',
)

# Indexes the fourth flip: fine at the novelty pool's lengths 4-8 and on the
# training data, an IndexError on pairs of length 1-3.
_SHORT_FAILS = _GOOD.replace(
    'return {"h_diff": (sequence_a.count("H") - sequence_b.count("H")) / len(sequence_a)}',
    'return {"h_diff": float(sequence_a[3] == "H") - float(sequence_b[3] == "H")}',
)


@pytest.fixture(autouse=True)
def _fresh_model_cache():
    clear_model_cache()
    yield
    clear_model_cache()


def _write_models(models_dir: Path, sources: dict) -> Path:
    models_dir.mkdir(parents=True, exist_ok=True)
    for name, source in sources.items():
        (models_dir / f"{name}.py").write_text(source, encoding="utf-8")
    (models_dir / "models_manifest.yaml").write_text(
        yaml.safe_dump(
            {"models": [{"name": n, "rationale": f"mechanism {n}"} for n in sources]}
        ),
        encoding="utf-8",
    )
    return models_dir


def _responses(tmp_path: Path) -> Path:
    path = tmp_path / "responses.csv"
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["sequence_a", "sequence_b", "participant_id", "trial_index", "chose_left"]
        )
        for i, (a, b, y) in enumerate(
            [("HHTTHT", "HTHTHT", 1), ("HHHHTT", "HTHTTH", 0), ("HTTHTH", "HHHTTT", 1)]
        ):
            writer.writerow([a, b, 0, i, y])
    return path


def _candidate(tmp_path: Path, source: str, slot: str = "candidate_0") -> Path:
    cand_dir = tmp_path / slot
    cand_dir.mkdir()
    (cand_dir / "candidate.py").write_text(source, encoding="utf-8")
    (cand_dir / "hypothesis.md").write_text("People count heads.\n", encoding="utf-8")
    return cand_dir / "candidate.py"


def _prior_fit(name: str, models_dir: Path) -> FittedModel:
    """A FittedModel whose 'posterior' is a prior sample: prediction runs for
    real, without MCMC."""
    import arviz as az
    import pymc as pm

    model = load_pymc_model(name, models_dir)
    with model:
        prior = pm.sample_prior_predictive(draws=20, random_seed=1)
    idata = az.InferenceData(
        posterior=prior.prior.drop_vars(["p_left", "prob"], errors="ignore")
    )
    return FittedModel(name=name, model=model, idata=idata, fingerprint=name)


# ── The initial-point logp check ─────────────────────────────────────────


def test_a_code_error_in_the_models_own_hook_is_a_logp_gate_failure(tmp_path):
    models_dir = _write_models(tmp_path / "models", {"typo": _TYPO})
    ok, reason = model_logp_is_finite("typo", models_dir, _responses(tmp_path))
    assert not ok
    assert "NameError" in reason and "count_heads" in reason


def test_a_code_error_raised_by_the_harness_still_raises(tmp_path, monkeypatch):
    """The same exception type, raised outside the model's file, is a broken
    harness: blaming every candidate for it would reject them all."""
    models_dir = _write_models(tmp_path / "models", {"good": _GOOD})

    def broken_binding(csv_path, model):
        raise NameError("name 'helper' is not defined")

    monkeypatch.setattr(pymc_inference, "extract_observed", broken_binding)
    with pytest.raises(NameError):
        model_logp_is_finite("good", models_dir, _responses(tmp_path))


def test_an_infrastructure_error_at_binding_still_raises(tmp_path, monkeypatch):
    models_dir = _write_models(tmp_path / "models", {"good": _GOOD})

    def full_disk(csv_path, model):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(pymc_inference, "extract_observed", full_disk)
    with pytest.raises(OSError):
        model_logp_is_finite("good", models_dir, _responses(tmp_path))


def test_admission_rejects_a_candidate_with_a_typo_in_its_hook(tmp_path):
    models_dir = _write_models(tmp_path / "models", {"good": _GOOD})
    ledger = HypothesisLedger.create(tmp_path / "ledger.jsonl", inherit_from=None)

    admission = model_zoo._admit_candidate_with_reason(
        _candidate(tmp_path, _TYPO),
        models_dir,
        "typo",
        _responses(tmp_path),
        ledger=ledger,
    )

    assert not admission.admitted
    assert "NameError" in admission.reason and "count_heads" in admission.reason
    assert not (models_dir / "typo.py").exists()
    (entry,) = ledger.entries()
    assert entry.outcome == "rejected" and "NameError" in entry.detail


def test_the_concurrent_prefit_skips_a_candidate_with_a_typo_in_its_hook(
    tmp_path, monkeypatch
):
    fitted = []
    monkeypatch.setattr(
        model_zoo,
        "fit_time_limited_concurrently",
        lambda names, *a, **k: fitted.extend(names),
    )
    ready = model_zoo.prefit_candidates(
        [
            (_candidate(tmp_path, _TYPO), "typo"),
            (_candidate(tmp_path, _GOOD, "candidate_1"), "good"),
        ],
        _responses(tmp_path),
        cache_dir=tmp_path / "cache",
    )
    assert ready == ["good"] and fitted == ["good"]


# ── The novelty gate ─────────────────────────────────────────────────────


def _stub_fit_and_scoring(monkeypatch):
    """Everything past the cheap gates, except the novelty gate's prediction,
    is stubbed; the novelty gate predicts with the real model code."""
    fits = {}

    def fit(name, models_dir, *a, **k):
        if name not in fits:
            fits[name] = _prior_fit(name, models_dir)
        return fits[name]

    monkeypatch.setattr(model_zoo, "fit_model", fit)
    monkeypatch.setattr(model_zoo, "convergence_problems_of", lambda fitted: [])
    monkeypatch.setattr(model_zoo, "log_likelihood", lambda *a, **k: -2.0)


@pytest.mark.parametrize(
    "source", [_NO_P_LEFT, _SCALAR_P_LEFT], ids=["no_p_left", "scalar_p_left"]
)
def test_the_novelty_gate_rejects_a_candidate_whose_p_left_cannot_be_predicted(
    tmp_path, monkeypatch, source
):
    models_dir = _write_models(tmp_path / "models", {"good": _GOOD})
    _stub_fit_and_scoring(monkeypatch)
    # The data-contract gate now rejects these before any fit
    # (test_model_contract.py); switched off here so the novelty gate's own
    # defence is still exercised.
    monkeypatch.setattr(model_zoo, "model_contract_violation", lambda *a, **k: None)
    ledger = HypothesisLedger.create(tmp_path / "ledger.jsonl", inherit_from=None)

    admission = model_zoo._admit_candidate_with_reason(
        _candidate(tmp_path, source),
        models_dir,
        "bad_p_left",
        _responses(tmp_path),
        ledger=ledger,
        novelty_pool=model_zoo.novelty_pool_rows()[:40],
    )

    assert not admission.admitted
    assert "could not be predicted on the 40-stimulus novelty pool" in admission.reason
    assert not (models_dir / "bad_p_left.py").exists()
    (entry,) = ledger.entries()
    assert entry.outcome == "rejected"


def test_the_novelty_gate_still_raises_an_infrastructure_error(tmp_path, monkeypatch):
    models_dir = _write_models(tmp_path / "models", {"good": _GOOD, "cand": _GOOD})

    class OutOfMemory:
        model = load_pymc_model("cand", models_dir)

        def predict_p_left(self, stim_data, **kwargs):
            raise MemoryError("cannot allocate")

    monkeypatch.setattr(model_zoo, "fit_model", lambda *a, **k: OutOfMemory())
    with pytest.raises(MemoryError):
        model_zoo._min_prediction_rmse(
            "cand",
            models_dir,
            _responses(tmp_path),
            pool_rows=model_zoo.novelty_pool_rows()[:10],
        )


# ── Short pairs: design (lengths 2-3) and evaluation (lengths 1-3) ───────


def test_a_carried_model_that_cannot_bind_short_pairs_is_screened_out_of_the_design(
    tmp_path, capsys
):
    models_dir = _write_models(
        tmp_path / "cognitive_models", {"good": _GOOD, "short_fails": _SHORT_FAILS}
    )
    screened = tmp_path / "design" / "screened_out.json"

    picks = eig_mod.design_exhaustive(
        models_dir,
        lengths=(2, 3, 4),
        n_select=3,
        n_samples=20,
        n_scenarios=50,
        screened_out_path=screened,
        n_responses=5,
        n_threads=1,
    )

    assert len(picks) == 3
    (record,) = json.loads(screened.read_text(encoding="utf-8"))
    assert record["model"] == "short_fails"
    assert "IndexError" in record["reason"] and "length [2, 3]" in record["reason"]
    assert "short_fails" in capsys.readouterr().out


def test_the_design_still_raises_when_the_harness_binding_is_broken(
    tmp_path, monkeypatch
):
    models_dir = _write_models(tmp_path / "cognitive_models", {"good": _GOOD})

    def broken(model, rows):
        raise AttributeError("module 'numpy' has no attribute 'asarry'")

    monkeypatch.setattr("src.models.data_binding.make_stim_data", broken)
    with pytest.raises(RuntimeError, match="harness is broken"):
        eig_mod._screen_usable_models(
            ["good"], models_dir, [{"sequence_a": "HT", "sequence_b": "TH"}]
        )


def _eval_rows(lengths):
    from src.subjective_randomness.stimulus_design import enumerate_all_pairs

    return [
        {**pair, "chose_left": 0}
        for pair in enumerate_all_pairs(list(lengths), same_length_only=True)
    ]


def test_evaluation_leaves_out_pairs_a_model_cannot_bind(tmp_path, capsys):
    models_dir = _write_models(tmp_path / "models", {"short_fails": _SHORT_FAILS})
    rows = _eval_rows((1, 2, 3, 4))
    short = np.array([len(row["sequence_a"]) < 4 for row in rows])

    pred = _eval_prediction(
        _prior_fit("short_fails", models_dir),
        rows,
        participant_ids=None,
        mask_invalid=True,
    )

    assert np.isnan(pred[short]).all()
    assert np.isfinite(pred[~short]).all()
    out = capsys.readouterr().out
    assert (
        f"cannot be bound to {int(short.sum())} of the {len(rows)} held-out pairs"
        in out
    )
    assert "lengths [1, 2, 3]" in out


def test_evaluation_without_masking_still_raises(tmp_path):
    models_dir = _write_models(tmp_path / "models", {"short_fails": _SHORT_FAILS})
    with pytest.raises(IndexError):
        _eval_prediction(
            _prior_fit("short_fails", models_dir),
            _eval_rows((2, 4)),
            participant_ids=None,
        )
