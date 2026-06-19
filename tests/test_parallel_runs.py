"""Parallel-run isolation: separate runs (e.g. cluster jobs) must not collide.

The output data root is overridable via AUTO_PSYCH_OUTPUT_DIR so each job writes
to its own tree and never pools another job's responses.
"""

from __future__ import annotations

from pathlib import Path

from src.pipelines.outer_loop.orchestrator import experiment_dir, outer_data_dir


def test_output_dir_defaults_under_repo(monkeypatch):
    monkeypatch.delenv("AUTO_PSYCH_OUTPUT_DIR", raising=False)
    assert outer_data_dir().parts[-2:] == ("data", "outer_loop")


def test_output_dir_respects_env_override(monkeypatch, tmp_path):
    job_dir = tmp_path / "job_123"
    monkeypatch.setenv("AUTO_PSYCH_OUTPUT_DIR", str(job_dir))
    assert outer_data_dir() == job_dir
    # Experiment dirs (and therefore pooling) live under the per-job root.
    assert experiment_dir("subjective_randomness", 1) == job_dir / "subjective_randomness" / "experiment1"


def test_parallel_jobs_get_isolated_experiment_dirs(monkeypatch, tmp_path):
    monkeypatch.setenv("AUTO_PSYCH_OUTPUT_DIR", str(tmp_path / "jobA"))
    a = experiment_dir("subjective_randomness", 1)
    monkeypatch.setenv("AUTO_PSYCH_OUTPUT_DIR", str(tmp_path / "jobB"))
    b = experiment_dir("subjective_randomness", 1)
    assert a != b
    assert Path(a).is_relative_to(tmp_path / "jobA")
    assert Path(b).is_relative_to(tmp_path / "jobB")
