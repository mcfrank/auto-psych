"""Loop-start screening of carried-forward models for finite ELPD-LOO.

Admission (``_admit_candidate``) already rejects a *fresh candidate* whose
ELPD-LOO is non-finite. But a model admitted in experiment k is carried forward
into experiment k+1's model set, and a model that scored a finite ELPD on k's
responses can yield a NaN ELPD on k+1's *different* responses. The loop-start
guard ``_drop_unfittable_models`` only checks finite *logp* (the initial point),
not ELPD, so such a carried-forward model slips into ``_score`` and crashes
``model_posterior`` — taking down the whole multi-hour run (this is exactly what
killed the impossible-holdout cells run1/more_heads and run3/fewer_heads, whose
``squared_heads`` / ``log_heads_penalty`` carried-forward models NaN'd on the
new data).

``_drop_nonfinite_elpd_models`` closes that gap: before scoring, it drops any
model whose ELPD-LOO is non-finite on the current responses — loudly, keeping
the rest — and fails only if no model survives. MCMC / ELPD is stubbed here.
"""

from __future__ import annotations

import math

import pytest
import yaml

import src.pipelines.inner_loop.pymc_orchestrator as pymc_orchestrator


def _models_dir(tmp_path, names):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    for name in names:
        (models_dir / f"{name}.py").write_text("# model\n", encoding="utf-8")
    (models_dir / "models_manifest.yaml").write_text(
        yaml.safe_dump(
            {"models": [{"name": n, "rationale": "carried forward."} for n in names]},
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return models_dir


def _manifest_names(models_dir):
    data = yaml.safe_load((models_dir / "models_manifest.yaml").read_text())
    return [e["name"] for e in data["models"]]


def test_drops_carried_model_with_nonfinite_elpd_keeps_finite(
    tmp_path, monkeypatch, capsys
):
    models_dir = _models_dir(tmp_path, ["seed_good", "carried_nan"])
    elpd = {"seed_good": -100.0, "carried_nan": math.nan}
    monkeypatch.setattr(
        pymc_orchestrator, "log_likelihood", lambda m, *a, **k: elpd[m]
    )

    pymc_orchestrator._drop_nonfinite_elpd_models(
        models_dir, tmp_path / "responses.csv"
    )

    names = _manifest_names(models_dir)
    assert "seed_good" in names
    assert "carried_nan" not in names
    assert "carried_nan" in capsys.readouterr().out  # dropped loudly


@pytest.mark.parametrize("bad", [math.nan, float("-inf"), float("inf")])
def test_drops_each_kind_of_nonfinite_elpd(tmp_path, monkeypatch, bad):
    models_dir = _models_dir(tmp_path, ["seed_good", "bad"])
    elpd = {"seed_good": -100.0, "bad": bad}
    monkeypatch.setattr(
        pymc_orchestrator, "log_likelihood", lambda m, *a, **k: elpd[m]
    )

    pymc_orchestrator._drop_nonfinite_elpd_models(
        models_dir, tmp_path / "responses.csv"
    )

    assert _manifest_names(models_dir) == ["seed_good"]


def test_drops_model_whose_elpd_computation_raises(tmp_path, monkeypatch):
    """A model whose ELPD computation itself blows up (e.g. PSIS on a degenerate
    fit) is dropped too — it can't be scored, so it can't be kept."""
    models_dir = _models_dir(tmp_path, ["seed_good", "explodes"])

    def ll(m, *a, **k):
        if m == "explodes":
            raise RuntimeError("PSIS-LOO blew up")
        return -100.0

    monkeypatch.setattr(pymc_orchestrator, "log_likelihood", ll)

    pymc_orchestrator._drop_nonfinite_elpd_models(
        models_dir, tmp_path / "responses.csv"
    )

    assert _manifest_names(models_dir) == ["seed_good"]


def test_raises_when_no_model_survives(tmp_path, monkeypatch):
    models_dir = _models_dir(tmp_path, ["a", "b"])
    monkeypatch.setattr(
        pymc_orchestrator, "log_likelihood", lambda *a, **k: math.nan
    )

    with pytest.raises(ValueError, match="finite ELPD"):
        pymc_orchestrator._drop_nonfinite_elpd_models(
            models_dir, tmp_path / "responses.csv"
        )


def test_all_finite_keeps_every_model(tmp_path, monkeypatch):
    models_dir = _models_dir(tmp_path, ["a", "b", "c"])
    monkeypatch.setattr(
        pymc_orchestrator, "log_likelihood", lambda *a, **k: -50.0
    )

    pymc_orchestrator._drop_nonfinite_elpd_models(
        models_dir, tmp_path / "responses.csv"
    )

    assert _manifest_names(models_dir) == ["a", "b", "c"]
