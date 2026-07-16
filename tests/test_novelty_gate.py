"""Novelty gate: a candidate that predicts like an existing model is rejected.

Without this gate, near-duplicates of the incumbents enter the zoo under fresh
names (run2's two winners were 0.029 RMSE apart in prediction space), splitting
posterior mass and wasting candidate slots. At admission — after the fit gate,
so every prediction reuses the cached fit — the candidate's posterior-mean
``p_left`` on the observed stimuli is compared to every admitted model's; a
minimum RMSE below ``novelty_rmse_threshold`` (default 0.02, ``0`` disables)
rejects the candidate, loudly naming its nearest neighbour.
"""

from __future__ import annotations

import numpy as np
import pytest
import yaml

from src.pipelines.inner_loop import pymc_orchestrator
from src.pipelines.inner_loop.pymc_orchestrator import (
    DEFAULT_NOVELTY_RMSE_THRESHOLD,
    _admit_candidate,
    _min_prediction_rmse,
)


class _FakeFitted:
    def __init__(self, p_left):
        self._p = np.asarray(p_left)
        self.model = object()

    def predict_p_left(self, stim_data, **kwargs):
        return self._p


def _models_dir(tmp_path, names):
    models_dir = tmp_path / "models"
    models_dir.mkdir(exist_ok=True)
    (models_dir / "models_manifest.yaml").write_text(
        yaml.safe_dump(
            {"models": [{"name": n, "rationale": f"mechanism {n}"} for n in names]}
        ),
        encoding="utf-8",
    )
    for n in names:
        (models_dir / f"{n}.py").write_text("# model\n", encoding="utf-8")
    return models_dir


def _responses(tmp_path):
    path = tmp_path / "responses.csv"
    path.write_text("n_a,chose_left\n4,1\n6,0\n", encoding="utf-8")
    return path


def test_min_prediction_rmse_finds_nearest_admitted_model(tmp_path, monkeypatch):
    models_dir = _models_dir(tmp_path, ["seed_a", "seed_b"])
    responses = _responses(tmp_path)
    predictions = {
        "candidate_x": np.array([0.5, 0.5]),
        "seed_a": np.array([0.5, 0.52]),  # RMSE ~0.014 — the nearest
        "seed_b": np.array([0.9, 0.1]),
    }
    monkeypatch.setattr(
        pymc_orchestrator,
        "fit_model",
        lambda name, *a, **k: _FakeFitted(predictions[name]),
    )
    monkeypatch.setattr(
        pymc_orchestrator, "make_stim_data", lambda model, rows: {"rows": len(rows)}
    )
    # candidate_x must be present in the manifest set for other names to skip it.
    name, rmse = _min_prediction_rmse(
        "candidate_x", models_dir, responses, cache_dir=None, fit_kwargs=None
    )
    assert name == "seed_a"
    assert rmse == pytest.approx(np.sqrt(np.mean([0.0, 0.02**2])), abs=1e-9)


def test_min_prediction_rmse_with_no_other_models(tmp_path, monkeypatch):
    models_dir = _models_dir(tmp_path, [])
    responses = _responses(tmp_path)
    monkeypatch.setattr(
        pymc_orchestrator, "fit_model", lambda *a, **k: _FakeFitted([0.5, 0.5])
    )
    monkeypatch.setattr(
        pymc_orchestrator, "make_stim_data", lambda model, rows: {}
    )
    name, rmse = _min_prediction_rmse(
        "candidate_x", models_dir, responses, cache_dir=None, fit_kwargs=None
    )
    assert name is None
    assert rmse == float("inf")


def _stub_admission_gates(monkeypatch):
    monkeypatch.setattr(
        pymc_orchestrator, "load_pymc_model", lambda name, models_dir: object()
    )
    monkeypatch.setattr(
        pymc_orchestrator, "model_logp_is_finite", lambda *a, **k: (True, "")
    )
    monkeypatch.setattr(pymc_orchestrator, "fit_model", lambda *a, **k: object())
    monkeypatch.setattr(pymc_orchestrator, "log_likelihood", lambda *a, **k: -10.0)


def _candidate(tmp_path):
    cand_dir = tmp_path / "candidate_0"
    cand_dir.mkdir(exist_ok=True)
    (cand_dir / "candidate.py").write_text("# candidate\n", encoding="utf-8")
    (cand_dir / "hypothesis.md").write_text("People use H.\n", encoding="utf-8")
    return cand_dir / "candidate.py"


def test_admission_rejects_near_duplicate(tmp_path, monkeypatch, capsys):
    models_dir = _models_dir(tmp_path, ["seed_a"])
    responses = _responses(tmp_path)
    _stub_admission_gates(monkeypatch)
    monkeypatch.setattr(
        pymc_orchestrator,
        "_min_prediction_rmse",
        lambda *a, **k: ("seed_a", 0.01),
    )
    admitted = _admit_candidate(
        _candidate(tmp_path), models_dir, "near_dup", responses
    )
    assert not admitted
    out = capsys.readouterr().out
    assert "seed_a" in out and "near_dup" in out
    assert not (models_dir / "near_dup.py").exists()
    manifest = yaml.safe_load(
        (models_dir / "models_manifest.yaml").read_text(encoding="utf-8")
    )
    assert [m["name"] for m in manifest["models"]] == ["seed_a"]


def test_admission_accepts_genuinely_novel_candidate(tmp_path, monkeypatch):
    models_dir = _models_dir(tmp_path, ["seed_a"])
    responses = _responses(tmp_path)
    _stub_admission_gates(monkeypatch)
    monkeypatch.setattr(
        pymc_orchestrator,
        "_min_prediction_rmse",
        lambda *a, **k: ("seed_a", 5 * DEFAULT_NOVELTY_RMSE_THRESHOLD),
    )
    assert _admit_candidate(_candidate(tmp_path), models_dir, "novel_model", responses)
    assert (models_dir / "novel_model.py").exists()


def test_threshold_zero_disables_the_gate(tmp_path, monkeypatch):
    models_dir = _models_dir(tmp_path, ["seed_a"])
    responses = _responses(tmp_path)
    _stub_admission_gates(monkeypatch)

    def tripwire(*a, **k):
        raise AssertionError("gate must not run when the threshold is 0")

    monkeypatch.setattr(pymc_orchestrator, "_min_prediction_rmse", tripwire)
    assert _admit_candidate(
        _candidate(tmp_path),
        models_dir,
        "any_model",
        responses,
        novelty_rmse_threshold=0.0,
    )
