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
    # Admission ends with a real MCMC fit-gate; stub it to succeed so these
    # bookkeeping tests don't sample (the stub candidate isn't a real PyMC model).
    monkeypatch.setattr(pymc_orchestrator, "fit_model", lambda *a, **k: object())
    # Admission also gates on a finite ELPD-LOO (reuses the fit); stub it finite.
    monkeypatch.setattr(pymc_orchestrator, "log_likelihood", lambda *a, **k: -100.0)


def _stub_fit_raises(monkeypatch):
    """Make the admission fit-gate's MCMC sample raise (a NUTS-fragile model)."""

    def _boom(*a, **k):
        raise RuntimeError("NUTS diverged")

    monkeypatch.setattr(pymc_orchestrator, "fit_model", _boom)


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


def test_admit_rejects_candidate_with_nonfinite_elpd(tmp_path, monkeypatch):
    """A candidate that fits (sampling succeeds) but yields a non-finite ELPD-LOO
    is rejected at admission — so it never reaches scoring, where a NaN ELPD would
    crash model_posterior. Closes the gap the logp/real-fit gates miss."""
    import math

    monkeypatch.setattr(pymc_orchestrator, "load_pymc_model", lambda n, d: object())
    _stub_fittable(monkeypatch)  # logp finite + MCMC fit gate succeeds
    monkeypatch.setattr(pymc_orchestrator, "log_likelihood", lambda *a, **k: math.nan)
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


def test_admit_rejects_candidate_whose_fit_raises(tmp_path, monkeypatch):
    """A candidate that passes the initial-point logp check but whose MCMC fit
    raises (NaN once NUTS jitters off the initial point) is rejected — not
    admitted — so it can't abort the round's scoring pass."""
    monkeypatch.setattr(pymc_orchestrator, "load_pymc_model", lambda n, d: object())
    monkeypatch.setattr(
        pymc_orchestrator, "model_logp_is_finite", lambda *a, **k: (True, "")
    )
    _stub_fit_raises(monkeypatch)
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
    # Both the staged model file and its hypothesis sidecar must be cleaned up.
    assert not (models_dir / "iter0_candidate0.py").exists()
    assert not (models_dir / "iter0_candidate0.hypothesis.md").exists()
