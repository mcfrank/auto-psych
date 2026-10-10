"""The live run's modeling settings match the recovery sweep's.

The recovery sweep (`holdout_recovery_faithful.yaml`) is what validates the loop;
a live run that searched less, sampled differently or gave agents less time
would not be the loop the simulations tested. It had drifted (2026-09-29): the
live preset ran 2 rounds x 3 candidates instead of 5 x 6, fitted at PyMC's
production target_accept 0.99 instead of 0.8 (run.py had no option for it),
gave inner-loop agents 15 minutes instead of 30 (no option either) and ran on 4
CPUs instead of 16. Draws and tuning stay higher on purpose (user decision).
"""

from __future__ import annotations

import pytest
import yaml

import src.pipelines.outer_loop.run as outer_run
from tests.paths import REPO_ROOT

LIVE = REPO_ROOT / "scripts" / "outer_loop_live"
RECOVERY = REPO_ROOT / "scripts" / "subjective_randomness" / "configs" / "holdout_recovery_faithful.yaml"


def _load(path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_the_full_run_models_like_the_recovery_sweep():
    live = _load(LIVE / "full_run.yaml")
    recovery = _load(RECOVERY)
    modeling = live["modeling"]
    assert live["experiments"] == recovery["n_experiments"]
    assert live["prolific"]["participants"] == recovery["n_participants"]
    assert modeling["inner_loop_iterations"] == recovery["inner_loop"]["max_iterations"]
    assert modeling["inner_loop_candidates"] == recovery["inner_loop"]["candidate_count"]
    assert modeling["target_accept"] == recovery["fit"]["target_accept"]
    assert modeling["chains"] == recovery["fit"]["chains"]
    assert modeling["agent_timeout_sec"] == recovery["agent"]["timeout_sec"]
    # More draws than the sweep, deliberately: one live run, no repeats.
    assert modeling["draws"] >= recovery["fit"]["draws"]
    assert modeling["tune"] >= recovery["fit"]["tune"]


def test_the_live_job_has_the_recovery_cells_cpus():
    def cpus(path):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("#SBATCH --cpus-per-task="):
                return int(line.split("=", 1)[1].split()[0])
        raise AssertionError(f"{path} requests no CPUs")

    array = REPO_ROOT / "scripts" / "subjective_randomness" / "slurm" / "holdout_recovery_array.sbatch"
    assert cpus(LIVE / "run_live.sbatch") == cpus(array)


def test_the_launcher_exports_target_accept_and_agent_timeout(tmp_path, monkeypatch, capsys):
    import sys

    sys.path.insert(0, str(LIVE))
    import _pilot_config as pilot_config

    # The full run recruits for real; the bridge refuses that without the
    # confirmation flag, so read its modeling block through test mode.
    config = tmp_path / "run.yaml"
    config.write_text(
        (LIVE / "full_run.yaml").read_text(encoding="utf-8").replace(
            "prolific_mode: live", "prolific_mode: test"
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(pilot_config, "project_assets_dir", lambda pid: tmp_path / pid)
    # The launcher checks the Prolific token against the live API; this test
    # reads only the exported settings, and CI holds no token (nor should it
    # reach Prolific).
    monkeypatch.setattr(pilot_config, "get_me", lambda: ({"id": "test-researcher"}, None))
    (tmp_path / "subjective_randomness").mkdir()
    monkeypatch.setattr(sys, "argv", ["_pilot_config.py", str(config)])
    pilot_config.main()
    out = capsys.readouterr().out
    assert "export TARGET_ACCEPT=0.8" in out
    assert "export AGENT_TIMEOUT_SEC=1800" in out


def test_run_py_fits_at_the_stated_target_accept_and_gives_agents_the_stated_time(
    tmp_path, monkeypatch
):
    seen = {}
    monkeypatch.setenv("AUTO_PSYCH_OUTPUT_DIR", str(tmp_path / "output"))
    monkeypatch.setattr(outer_run, "_run_experiment", lambda *a, **k: seen.update(k))
    outer_run.main(outer_run.Args(
        project="subjective_randomness", experiment=1,
        mode="simulated_participants_nobrowser", coding_agent="opencode",
        target_accept=0.8, agent_timeout_sec=1800,
    ))
    assert seen["fit_kwargs"]["target_accept"] == 0.8
    assert seen["agent_timeout_sec"] == 1800


def test_the_model_stage_receives_the_agent_timeout(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(outer_run, "begin_model_loop_stage", lambda exp_dir: None)
    monkeypatch.setattr(outer_run, "finish_model_loop_stage", lambda exp_dir: None)
    monkeypatch.setattr(
        outer_run, "run_inner_model_loop_programmatic", lambda *a, **k: seen.update(k)
    )
    outer_run._run_agent(
        "5_model_loop", tmp_path, "subjective_randomness", 1,
        "simulated_participants_nobrowser", 40, None, False, agent_timeout_sec=1800,
    )
    assert seen["agent_timeout_sec"] == 1800


def test_the_launcher_refuses_qos_long(tmp_path, monkeypatch, capsys):
    """--qos=long is not on this Sherlock account ("Invalid qos specification",
    2026-10-07); a config asking for it stops before anything is submitted."""
    import sys

    sys.path.insert(0, str(LIVE))
    import _pilot_config as pilot_config

    config = tmp_path / "run.yaml"
    config.write_text(
        (LIVE / "full_run.yaml").read_text(encoding="utf-8")
        .replace("prolific_mode: live", "prolific_mode: test")
        .replace('qos: ""', "qos: long"),
        encoding="utf-8",
    )
    monkeypatch.setattr(pilot_config, "project_assets_dir", lambda pid: tmp_path / pid)
    monkeypatch.setattr(pilot_config, "get_me", lambda: ({"id": "test-researcher"}, None))
    (tmp_path / "subjective_randomness").mkdir()
    monkeypatch.setattr(sys, "argv", ["_pilot_config.py", str(config)])
    with pytest.raises(SystemExit) as exit_info:
        pilot_config.main()
    assert "qos: long" in str(exit_info.value) and "mcfrank" in str(exit_info.value)
