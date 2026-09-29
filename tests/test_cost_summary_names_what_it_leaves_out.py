"""The launchers' cost summary says it covers Prolific only.

``_pilot_config.py`` prints the pre-launch cost summary that ``run_pilot.sh``
and ``start_full_run.sh`` show before the typed "yes". It estimated the
Prolific reward and fee and called that the total, with no word that the
coding agents (the page, critiques, model proposals) bill a language-model
account separately. The repo holds no record of a live run's agent spend to
estimate from, so the summary states the omission instead of a number.
"""

from __future__ import annotations

import sys

import pytest
import yaml

from tests.paths import SCRIPTS_DIR, load_script_module


def _summary(tmp_path, monkeypatch, capsys, prolific_mode):
    pilot_config = load_script_module(SCRIPTS_DIR / "outer_loop_live" / "_pilot_config.py")
    monkeypatch.setattr(pilot_config, "get_me", lambda: ({"id": "researcher"}, None))
    config = tmp_path / "pilot.yaml"
    config.write_text(
        yaml.safe_dump({
            "project": "subjective_randomness",
            "run_label": "t",
            "experiments": 2,
            "coding_agent": "opencode",
            "prolific_mode": prolific_mode,
            "confirm_live_recruitment": True,
            "prolific": {"participants": 10, "reward_per_hour": 1200,
                         "estimated_completion_time": 7},
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr(sys, "argv", ["_pilot_config.py", str(config), "--check"])
    pilot_config.main()
    return capsys.readouterr().err


@pytest.mark.parametrize("prolific_mode", ["live", "test", "none"])
def test_the_summary_says_agent_costs_are_not_included(
    tmp_path, monkeypatch, capsys, prolific_mode
):
    summary = _summary(tmp_path, monkeypatch, capsys, prolific_mode)
    assert "AI agent costs NOT included" in summary
    assert "opencode" in summary
    assert "token_usage_summary.json" in summary


def test_the_live_total_is_labelled_as_prolific_only(tmp_path, monkeypatch, capsys):
    summary = _summary(tmp_path, monkeypatch, capsys, "live")
    assert "PROLIFIC TOTAL" in summary
    assert "ESTIMATED TOTAL" not in summary and "GRAND TOTAL" not in summary
