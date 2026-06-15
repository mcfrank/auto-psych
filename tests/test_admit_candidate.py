"""Unit tests for candidate admission in the PyMC inner loop.

A candidate may enter the model set only if it ships a natural-language
hypothesis (``hypothesis.md``) alongside a loadable PyMC ``candidate.py``. The
hypothesis text becomes the model's manifest rationale and is copied next to the
model, so every model in the set carries the single hypothesis it tests. MCMC /
model loading is stubbed — these tests cover only the admission bookkeeping.
"""

from __future__ import annotations

import yaml

import src.pipelines.inner_loop.pymc_orchestrator as pymc_orchestrator
from src.pipelines.inner_loop.pymc_orchestrator import _admit_candidate


def _models_dir_with_seed(tmp_path):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "seed.py").write_text("# seed model\n", encoding="utf-8")
    (models_dir / "models_manifest.yaml").write_text(
        yaml.safe_dump(
            {"models": [{"name": "seed", "rationale": "People do X."}]},
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return models_dir


def _candidate_dir(tmp_path, *, hypothesis):
    cand_dir = tmp_path / "iter_0" / "candidate_0"
    cand_dir.mkdir(parents=True)
    (cand_dir / "candidate.py").write_text("# candidate\n", encoding="utf-8")
    if hypothesis is not None:
        (cand_dir / "hypothesis.md").write_text(hypothesis, encoding="utf-8")
    return cand_dir


def _manifest(models_dir):
    data = yaml.safe_load((models_dir / "models_manifest.yaml").read_text())
    return {e["name"]: e for e in data["models"]}


def _stub_fittable(monkeypatch, ok=True, reason=""):
    monkeypatch.setattr(
        pymc_orchestrator, "model_logp_is_finite", lambda *a, **k: (ok, reason)
    )


def test_admit_rejects_candidate_without_hypothesis(tmp_path, monkeypatch):
    monkeypatch.setattr(pymc_orchestrator, "load_pymc_model", lambda n, d: object())
    _stub_fittable(monkeypatch)
    models_dir = _models_dir_with_seed(tmp_path)
    cand_dir = _candidate_dir(tmp_path, hypothesis=None)

    ok = _admit_candidate(
        cand_dir / "candidate.py",
        models_dir,
        model_name="iter0_candidate0",
        responses_path=tmp_path / "responses.csv",
    )

    assert ok is False
    assert "iter0_candidate0" not in _manifest(models_dir)
    assert not (models_dir / "iter0_candidate0.py").exists()


def test_admit_rejects_candidate_with_empty_hypothesis(tmp_path, monkeypatch):
    monkeypatch.setattr(pymc_orchestrator, "load_pymc_model", lambda n, d: object())
    _stub_fittable(monkeypatch)
    models_dir = _models_dir_with_seed(tmp_path)
    cand_dir = _candidate_dir(tmp_path, hypothesis="   \n")

    ok = _admit_candidate(
        cand_dir / "candidate.py",
        models_dir,
        model_name="iter0_candidate0",
        responses_path=tmp_path / "responses.csv",
    )

    assert ok is False
    assert "iter0_candidate0" not in _manifest(models_dir)


def test_admit_rejects_unfittable_candidate(tmp_path, monkeypatch):
    """A candidate that loads but evaluates to non-finite logp is rejected."""
    monkeypatch.setattr(pymc_orchestrator, "load_pymc_model", lambda n, d: object())
    _stub_fittable(monkeypatch, ok=False, reason="non-finite logp (-inf)")
    models_dir = _models_dir_with_seed(tmp_path)
    cand_dir = _candidate_dir(tmp_path, hypothesis="People use heuristic H.\n")

    ok = _admit_candidate(
        cand_dir / "candidate.py",
        models_dir,
        model_name="iter0_candidate0",
        responses_path=tmp_path / "responses.csv",
    )

    assert ok is False
    assert "iter0_candidate0" not in _manifest(models_dir)
    assert not (models_dir / "iter0_candidate0.py").exists()


def test_admit_uses_hypothesis_as_rationale_and_copies_it(tmp_path, monkeypatch):
    monkeypatch.setattr(pymc_orchestrator, "load_pymc_model", lambda n, d: object())
    _stub_fittable(monkeypatch)
    models_dir = _models_dir_with_seed(tmp_path)
    hyp = "People judge a sequence as more random when it alternates more often."
    cand_dir = _candidate_dir(tmp_path, hypothesis=hyp + "\n")

    ok = _admit_candidate(
        cand_dir / "candidate.py",
        models_dir,
        model_name="iter0_candidate0",
        responses_path=tmp_path / "responses.csv",
    )

    assert ok is True
    manifest = _manifest(models_dir)
    assert manifest["iter0_candidate0"]["rationale"] == hyp
    copied = models_dir / "iter0_candidate0.hypothesis.md"
    assert copied.read_text(encoding="utf-8").strip() == hyp
    # The seed model's existing rationale must be preserved, not dropped.
    assert manifest["seed"]["rationale"] == "People do X."
