"""Tests for the no-browser collect mode and the implement-skip behavior.

Covers:
- _parse_participant_answer parses LLM-as-participant replies in canonical and lenient forms.
- run_experiment_implementer short-circuits when mode == simulated_participants_nobrowser
  (and when ground_truth_model is set), writing only a minimal config.json.
- validate_implementer_output accepts the minimal config.json (no index.html required) for
  skip-deploy modes, while still requiring index.html in the default mode.
- _collect_llm_participant produces correct CSV rows when invoke_llm is monkeypatched.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.agents.collect import (
    _collect_llm_participant,
    _parse_participant_answer,
)
from src.agents.experiment_implementer import run_experiment_implementer
from src.validation.validators import validate_implementer_output


# ---------- _parse_participant_answer --------------------------------------


@pytest.mark.parametrize(
    "reply,expected",
    [
        ("ANSWER: left", "left"),
        ("ANSWER: right", "right"),
        ("answer: LEFT", "left"),
        ("ANSWER:left\n(thought a bit about it)", "left"),
        ("Sure, here is my answer.\nANSWER: right", "right"),
        ("left", "left"),
        ("right", "right"),
        ("I think the left one looks more random.", "left"),
        ("The right side seems better here.", "right"),
        ("", None),
        ("I cannot decide.", None),
        ("both left and right look fine", None),
    ],
)
def test_parse_participant_answer(reply: str, expected) -> None:
    assert _parse_participant_answer(reply) == expected


# ---------- run_experiment_implementer short-circuit -----------------------


def _write_minimal_inputs(run_dir: Path) -> Dict[str, str]:
    """Create the inputs run_experiment_implementer expects in state."""
    (run_dir / "2_design").mkdir(parents=True, exist_ok=True)
    stimuli_path = run_dir / "2_design" / "stimuli.json"
    stimuli_path.write_text(
        json.dumps(
            [
                {"sequence_a": "HHTH", "sequence_b": "HTHT"},
                {"sequence_a": "HHHH", "sequence_b": "TTTT"},
            ]
        ),
        encoding="utf-8",
    )
    prob_path = run_dir / "problem_definition.md"
    prob_path.write_text("Which sequence looks more random?", encoding="utf-8")
    return {"stimuli_path": str(stimuli_path), "problem_definition_path": str(prob_path)}


def _make_state_for_implement(run_dir: Path, *, mode: str, ground_truth_model: str | None = None) -> Dict[str, Any]:
    paths = _write_minimal_inputs(run_dir)
    state: Dict[str, Any] = {
        "project_id": "test_project",
        "run_id": 1,
        "mode": mode,
        "problem_definition_path": paths["problem_definition_path"],
        "stimuli_path": paths["stimuli_path"],
        "simulated_n_participants": 4,
    }
    if ground_truth_model is not None:
        state["ground_truth_model"] = ground_truth_model
    return state


def _patch_projects_dir(monkeypatch: pytest.MonkeyPatch, projects_root: Path) -> None:
    """Redirect agent_dir_for_state's PROJECTS_DIR so writes land in tmp_path."""
    import src.config as config

    monkeypatch.setattr(config, "PROJECTS_DIR", projects_root)


def test_implement_skips_in_nobrowser_mode(tmp_path, monkeypatch) -> None:
    projects_root = tmp_path / "projects"
    run_dir = projects_root / "test_project" / "run1"
    run_dir.mkdir(parents=True)
    _patch_projects_dir(monkeypatch, projects_root)

    state = _make_state_for_implement(run_dir, mode="simulated_participants_nobrowser")
    result = run_experiment_implementer(state)

    impl_dir = run_dir / "3_implement"
    assert (impl_dir / "config.json").exists(), "config.json must be written in skip-deploy mode"
    assert not (impl_dir / "index.html").exists(), "index.html must not be written in skip-deploy mode"
    assert not (impl_dir / "stimuli.json").exists(), "no stimuli.json written when skipping"
    cfg = json.loads((impl_dir / "config.json").read_text(encoding="utf-8"))
    assert cfg.get("skipped") is True
    assert cfg.get("mode") == "simulated_participants_nobrowser"
    assert cfg.get("simulated_n_participants") == 4
    assert result["deployment_config_path"] == str(impl_dir / "config.json")
    # Validator accepts the skip-deploy output
    validation = validate_implementer_output(run_dir)
    assert validation.ok, validation.message


def test_implement_skips_when_ground_truth_model_set(tmp_path, monkeypatch) -> None:
    projects_root = tmp_path / "projects"
    run_dir = projects_root / "test_project" / "run1"
    run_dir.mkdir(parents=True)
    _patch_projects_dir(monkeypatch, projects_root)

    state = _make_state_for_implement(
        run_dir, mode="simulated_participants", ground_truth_model="alternation"
    )
    run_experiment_implementer(state)

    cfg = json.loads((run_dir / "3_implement" / "config.json").read_text(encoding="utf-8"))
    assert cfg.get("skipped") is True
    assert cfg.get("ground_truth_model") == "alternation"
    assert not (run_dir / "3_implement" / "index.html").exists()
    validation = validate_implementer_output(run_dir)
    assert validation.ok, validation.message


# ---------- validator: still strict in default mode ------------------------


def test_validator_requires_index_html_in_default_mode(tmp_path) -> None:
    impl_dir = tmp_path / "3_implement"
    impl_dir.mkdir()
    # Default-mode config.json (no skipped flag, mode=simulated_participants) and no index.html.
    (impl_dir / "config.json").write_text(
        json.dumps({"mode": "simulated_participants", "run_mode": "simulated_participants"}),
        encoding="utf-8",
    )
    validation = validate_implementer_output(tmp_path)
    assert not validation.ok
    assert "index.html" in validation.message


def test_validator_accepts_minimal_config_in_nobrowser(tmp_path) -> None:
    impl_dir = tmp_path / "3_implement"
    impl_dir.mkdir()
    (impl_dir / "config.json").write_text(
        json.dumps(
            {
                "mode": "simulated_participants_nobrowser",
                "run_mode": "simulated_participants_nobrowser",
                "skipped": True,
                "simulated_n_participants": 2,
            }
        ),
        encoding="utf-8",
    )
    validation = validate_implementer_output(tmp_path)
    assert validation.ok, validation.message


# ---------- _collect_llm_participant with monkeypatched LLM ----------------


def test_collect_llm_participant_emits_rows(tmp_path, monkeypatch) -> None:
    """Mock invoke_llm and load_prompt_for_run; verify rows have the right schema and counts."""
    out_dir = tmp_path / "4_collect"
    logs_dir = out_dir / "logs"
    logs_dir.mkdir(parents=True)

    stimuli: List[Dict[str, Any]] = [
        {"sequence_a": "HHTH", "sequence_b": "HTHT"},
        {"sequence_a": "HHHH", "sequence_b": "TTTT"},
    ]
    config = {"simulated_n_participants": 3}
    state: Dict[str, Any] = {
        "project_id": "test_project",
        "run_id": 1,
        "mode": "simulated_participants_nobrowser",
        "simulated_n_participants": 3,
    }

    # Alternate left/right replies so we exercise both parses.
    replies = ["ANSWER: left", "ANSWER: right"]
    counter = {"i": 0}

    def fake_invoke_llm(system: str, user: str, llm=None, timeout=None):
        r = replies[counter["i"] % len(replies)]
        counter["i"] += 1
        return r

    def fake_load_prompt_for_run(project_id, run_id, agent_key, state=None):
        assert agent_key == "4_collect_participant"
        return "You are a participant. Answer with ANSWER: left or ANSWER: right."

    def fake_get_llm(timeout=None):
        return object()  # _collect_llm_participant only passes this through to invoke_llm

    import src.agents.base as base_mod

    monkeypatch.setattr(base_mod, "invoke_llm", fake_invoke_llm)
    monkeypatch.setattr(base_mod, "load_prompt_for_run", fake_load_prompt_for_run)
    monkeypatch.setattr(base_mod, "get_llm", fake_get_llm)

    rows = _collect_llm_participant(state, config, out_dir, logs_dir, stimuli)

    assert len(rows) == 3 * len(stimuli), "expected n_participants * n_stimuli rows"
    expected_columns = {
        "participant_id",
        "trial_index",
        "sequence_a",
        "sequence_b",
        "chose_left",
        "chose_right",
        "model",
    }
    assert expected_columns <= set(rows[0].keys())
    for r in rows:
        assert r["model"] == "llm_participant"
        assert r["chose_left"] + r["chose_right"] == 1
    transcripts_dir = out_dir / "transcripts"
    assert transcripts_dir.is_dir()
    assert (transcripts_dir / "participant_000.md").exists()
    assert (logs_dir / "llm_participant_summary.txt").exists()
