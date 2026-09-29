"""The candidate agent can check its own model before finishing.

Before P36 the agent wrote a PyMC model blind: nothing told it how to confirm
that ``candidate.py`` loads as a module-level ``model: pm.Model`` and completes
a fit, so 17 of the 29 reasoned rejections in the 2026-09 sweep were things a
short local check would have caught. CONTEXT.md now documents one command —
the pipeline's own interpreter, ``src.pipelines.inner_loop.check_candidate``
— that runs the admission gates the agent can act on (import allowlist,
loadable model, finite logp, a short fit, finite ELPD-LOO) with a small
draws/tune setting. It is a smoke fit, never a production fit.
"""

from __future__ import annotations

import shutil
import sys

import pytest
import tyro
import yaml

import src.pipelines.inner_loop.check_candidate as check_candidate
from src.models.mcmc_defaults import (
    CANDIDATE_CHECK_CHAINS,
    CANDIDATE_CHECK_DRAWS,
    CANDIDATE_CHECK_TUNE,
    PRODUCTION_DRAWS,
    PRODUCTION_TUNE,
)
from src.pipelines.inner_loop.candidate_agent import _write_candidate_context
from src.pipelines.inner_loop.check_candidate import (
    Args,
    CandidateCheckFailed,
    check_candidate_command,
    run_candidate_check,
)
from tests.paths import PYMC_MODEL_FIXTURES_DIR


def _write_models_dir(tmp_path):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "models_manifest.yaml").write_text(
        yaml.safe_dump({"models": [{"name": "seed", "rationale": "People do X."}]}),
        encoding="utf-8",
    )
    return models_dir


# ── CONTEXT.md documents the command ──────────────────────────────────


def test_context_documents_the_check_command(tmp_path):
    models_dir = _write_models_dir(tmp_path)
    responses = tmp_path / "responses.csv"
    responses.write_text("sequence_a,sequence_b,chose_left\nHHT,HTH,1\n", encoding="utf-8")
    candidate_dir = tmp_path / "iter_0" / "candidate_0"

    docs = _write_candidate_context(
        candidate_dir, responses, models_dir, 0, 0, 3, current_posterior=None
    )

    context = (candidate_dir / "CONTEXT.md").read_text(encoding="utf-8")
    assert docs["context"] == context
    command = check_candidate_command(candidate_dir, responses)
    assert command in context
    # The exact interpreter, the module, and both paths — nothing to guess.
    assert command.startswith(sys.executable + " ")
    assert "-m src.pipelines.inner_loop.check_candidate" in command
    assert str(candidate_dir) in command
    assert str(responses) in command
    # The agent is told it is a short check, not a full fit.
    assert f"{CANDIDATE_CHECK_DRAWS} draws" in context
    assert "not a full" in context.lower() or "not a production" in context.lower()


# ── The check itself ──────────────────────────────────────────────────


def test_check_raises_without_a_candidate_file(tmp_path):
    candidate_dir = tmp_path / "candidate_0"
    candidate_dir.mkdir()
    with pytest.raises(CandidateCheckFailed, match="candidate.py"):
        run_candidate_check(candidate_dir, PYMC_MODEL_FIXTURES_DIR / "responses.csv")


def test_check_rejects_a_forbidden_import_before_loading(tmp_path):
    candidate_dir = tmp_path / "candidate_0"
    candidate_dir.mkdir()
    (candidate_dir / "candidate.py").write_text(
        "import pandas\nimport pymc as pm\nwith pm.Model() as model:\n    pass\n",
        encoding="utf-8",
    )
    with pytest.raises(CandidateCheckFailed, match="forbidden import: pandas"):
        run_candidate_check(candidate_dir, PYMC_MODEL_FIXTURES_DIR / "responses.csv")


def test_check_rejects_a_module_without_a_model(tmp_path):
    candidate_dir = tmp_path / "candidate_0"
    candidate_dir.mkdir()
    (candidate_dir / "candidate.py").write_text("x = 1\n", encoding="utf-8")
    with pytest.raises(CandidateCheckFailed, match="not a loadable PyMC model"):
        run_candidate_check(candidate_dir, PYMC_MODEL_FIXTURES_DIR / "responses.csv")


def test_check_fits_with_the_small_settings_and_no_cache(tmp_path, monkeypatch):
    """The fit is a smoke test: the small check settings, one core, no on-disk
    cache (a check must never seed the pipeline's fit cache)."""
    candidate_dir = tmp_path / "candidate_0"
    candidate_dir.mkdir()
    shutil.copyfile(
        PYMC_MODEL_FIXTURES_DIR / "bayesian_fair_coin.py", candidate_dir / "candidate.py"
    )
    calls = {}

    def fake_fit_model(name, models_dir, responses_path, **kwargs):
        calls["fit"] = (name, models_dir, kwargs)
        return object()

    def fake_log_likelihood(name, responses_path, models_dir, **kwargs):
        calls["elpd"] = (name, models_dir, kwargs)
        return -12.5

    monkeypatch.setattr(check_candidate, "model_logp_is_finite", lambda *a, **k: (True, ""))
    monkeypatch.setattr(check_candidate, "fit_model", fake_fit_model)
    monkeypatch.setattr(check_candidate, "log_likelihood", fake_log_likelihood)

    report = run_candidate_check(candidate_dir, PYMC_MODEL_FIXTURES_DIR / "responses.csv")

    assert calls["fit"][0] == "candidate" and calls["fit"][1] == candidate_dir
    fit_kwargs = calls["fit"][2]
    assert fit_kwargs["draws"] == CANDIDATE_CHECK_DRAWS < PRODUCTION_DRAWS
    assert fit_kwargs["tune"] == CANDIDATE_CHECK_TUNE < PRODUCTION_TUNE
    assert fit_kwargs["chains"] == CANDIDATE_CHECK_CHAINS
    assert fit_kwargs["cores"] == 1
    assert fit_kwargs.get("cache_dir") is None
    assert calls["elpd"][2]["draws"] == CANDIDATE_CHECK_DRAWS
    assert "-12.5" in report and "ELPD-LOO" in report


def test_check_reports_a_non_finite_logp(tmp_path, monkeypatch):
    candidate_dir = tmp_path / "candidate_0"
    candidate_dir.mkdir()
    shutil.copyfile(
        PYMC_MODEL_FIXTURES_DIR / "bayesian_fair_coin.py", candidate_dir / "candidate.py"
    )
    monkeypatch.setattr(
        check_candidate, "model_logp_is_finite", lambda *a, **k: (False, "logp is -inf")
    )
    with pytest.raises(CandidateCheckFailed, match="logp is -inf"):
        run_candidate_check(candidate_dir, PYMC_MODEL_FIXTURES_DIR / "responses.csv")


def test_check_reports_a_non_finite_elpd(tmp_path, monkeypatch):
    candidate_dir = tmp_path / "candidate_0"
    candidate_dir.mkdir()
    shutil.copyfile(
        PYMC_MODEL_FIXTURES_DIR / "bayesian_fair_coin.py", candidate_dir / "candidate.py"
    )
    monkeypatch.setattr(check_candidate, "model_logp_is_finite", lambda *a, **k: (True, ""))
    monkeypatch.setattr(check_candidate, "fit_model", lambda *a, **k: object())
    monkeypatch.setattr(check_candidate, "log_likelihood", lambda *a, **k: float("nan"))
    with pytest.raises(CandidateCheckFailed, match="non-finite ELPD-LOO"):
        run_candidate_check(candidate_dir, PYMC_MODEL_FIXTURES_DIR / "responses.csv")


def test_cli_parses_the_documented_command(tmp_path):
    args = tyro.cli(
        Args,
        args=[
            "--candidate-dir",
            str(tmp_path / "candidate_0"),
            "--responses",
            str(tmp_path / "responses.csv"),
        ],
    )
    assert args.candidate_dir == tmp_path / "candidate_0"
    assert args.responses == tmp_path / "responses.csv"


@pytest.mark.slow
def test_check_passes_a_real_model_end_to_end(tmp_path):
    """The documented command, as the agent would run it, on a real model."""
    import subprocess

    from tests.paths import REPO_ROOT

    candidate_dir = tmp_path / "candidate_0"
    candidate_dir.mkdir()
    shutil.copyfile(
        PYMC_MODEL_FIXTURES_DIR / "bayesian_fair_coin.py", candidate_dir / "candidate.py"
    )
    command = check_candidate_command(candidate_dir, PYMC_MODEL_FIXTURES_DIR / "responses.csv")
    proc = subprocess.run(
        command.replace("\\\n", " ").split(),
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "OK" in proc.stdout and "ELPD-LOO" in proc.stdout
