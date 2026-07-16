"""Candidate agents name their own models descriptively.

Auto-generated ``iterN_candidateM`` names made two different discovered models
collide across runs (run1's and run3's winners were both ``iter1_candidate0``)
and carry no information into ``existing_hypotheses.md``. The agent now writes
``model_name.txt`` (a snake_case slug) alongside ``hypothesis.md``;
``_resolve_candidate_name`` validates it, uniquifies against the current model
set, and falls back to the auto name — loudly — when it is missing or invalid,
so a bad name never sinks an otherwise good candidate.
"""

from __future__ import annotations

import yaml

from src.pipelines.inner_loop.pymc_orchestrator import _resolve_candidate_name


def _models_dir(tmp_path, names):
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    (models_dir / "models_manifest.yaml").write_text(
        yaml.safe_dump(
            {"models": [{"name": n, "rationale": f"mechanism {n}"} for n in names]}
        ),
        encoding="utf-8",
    )
    return models_dir


def _candidate_dir(tmp_path, name_text=None):
    candidate_dir = tmp_path / "candidate_0"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    if name_text is not None:
        (candidate_dir / "model_name.txt").write_text(name_text, encoding="utf-8")
    return candidate_dir


FALLBACK = "iter0_candidate0"


def test_valid_slug_is_used(tmp_path):
    models_dir = _models_dir(tmp_path, ["seed_a"])
    cand = _candidate_dir(tmp_path, "recency_weighted_runs\n")
    assert (
        _resolve_candidate_name(cand, models_dir, fallback=FALLBACK)
        == "recency_weighted_runs"
    )


def test_missing_name_file_falls_back(tmp_path, capsys):
    models_dir = _models_dir(tmp_path, ["seed_a"])
    cand = _candidate_dir(tmp_path, None)
    assert _resolve_candidate_name(cand, models_dir, fallback=FALLBACK) == FALLBACK
    assert "model_name.txt" in capsys.readouterr().out


def test_invalid_slug_falls_back(tmp_path, capsys):
    models_dir = _models_dir(tmp_path, ["seed_a"])
    cand = _candidate_dir(tmp_path, "Recency Weighted!! Runs")
    assert _resolve_candidate_name(cand, models_dir, fallback=FALLBACK) == FALLBACK
    assert "invalid" in capsys.readouterr().out.lower()


def test_collision_with_existing_model_is_uniquified(tmp_path):
    models_dir = _models_dir(tmp_path, ["recency_weighted_runs", "seed_a"])
    cand = _candidate_dir(tmp_path, "recency_weighted_runs")
    assert (
        _resolve_candidate_name(cand, models_dir, fallback=FALLBACK)
        == "recency_weighted_runs_2"
    )


def test_zoo_pattern_name_falls_back(tmp_path, capsys):
    # An agent-chosen name that mimics the auto-name pattern would collide with
    # fallback semantics and trip the carried-manifest zoo check downstream.
    models_dir = _models_dir(tmp_path, ["seed_a"])
    cand = _candidate_dir(tmp_path, "iter3_candidate7")
    assert _resolve_candidate_name(cand, models_dir, fallback=FALLBACK) == FALLBACK


def test_reserved_name_falls_back(tmp_path):
    models_dir = _models_dir(tmp_path, ["seed_a"])
    cand = _candidate_dir(tmp_path, "inner_loop_model")
    assert _resolve_candidate_name(cand, models_dir, fallback=FALLBACK) == FALLBACK
