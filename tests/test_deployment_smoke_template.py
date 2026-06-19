"""Tests for rendering the real jsPsych template into a smoke experiment.

These exercise the pure rendering helpers used by
``scripts/smoke_firebase_deploy.py`` — no browser, no Firebase, no Gemini — so
they prove the deployable artifact is well formed before any live deploy.
"""

from __future__ import annotations

import json

import pytest

from src.pipelines.outer_loop.deployment.smoke import (
    make_randomness_stimuli,
    render_template_experiment,
)


def test_make_randomness_stimuli_are_well_formed_ht_pairs():
    stimuli = make_randomness_stimuli(3)
    assert len(stimuli) == 3
    for stimulus in stimuli:
        assert set(stimulus) >= {"sequence_a", "sequence_b"}
        assert stimulus["sequence_a"] and set(stimulus["sequence_a"]) <= {"H", "T"}
        assert stimulus["sequence_b"] and set(stimulus["sequence_b"]) <= {"H", "T"}
        # A randomness judgment is only meaningful when the two sides differ.
        assert stimulus["sequence_a"] != stimulus["sequence_b"]


def test_make_randomness_stimuli_rejects_nonpositive_count():
    with pytest.raises(ValueError):
        make_randomness_stimuli(0)


def test_render_template_experiment_fills_every_placeholder(tmp_path):
    exp_dir = tmp_path / "smoke_experiment"
    experiment_dir = render_template_experiment(exp_dir, n_stimuli=2)

    assert experiment_dir == exp_dir / "experiment"
    index = (experiment_dir / "index.html").read_text(encoding="utf-8")

    # Every {{PLACEHOLDER}} must be substituted — a leftover would ship broken JS.
    assert "{{" not in index and "}}" not in index
    # Button trial choices (Sequence A / Sequence B) are injected as a JSON array.
    assert "Sequence A (left)" in index and "Sequence B (right)" in index
    # Each stimulus ships both side orderings so the template can counterbalance
    # which sequence is shown on the left per trial.
    assert "display_a_left" in index and "display_b_left" in index
    assert '"sequence_a"' in index and '"sequence_b"' in index
    # The real template's jsPsych bootstrap survives substitution.
    assert "initJsPsych" in index
    # Button responses ONLY — the modality the implement validator enforces; no
    # keyboard-response plugin should remain (the pipeline rejects keyboard).
    assert "jsPsychHtmlButtonResponse" in index
    assert "jsPsychHtmlKeyboardResponse" not in index
    # chose_left must come from the first button (index 0 = left / Sequence A).
    assert "data.response === 0" in index
    # Side counterbalancing: the trial records the presented order and flips sides.
    assert "presentedOrder" in index


def test_render_template_experiment_has_no_builtin_consent(tmp_path):
    """The template must NOT carry its own consent screen.

    Consent is injected at deployment from the approved verbatim IRB text
    (``ensure_consent_gate``). If the template also had a consent step, a
    participant would see two consent forms — and the template's wording is an
    unapproved paraphrase. So the rendered experiment starts at instructions.
    """
    exp_dir = tmp_path / "smoke_experiment"
    experiment_dir = render_template_experiment(exp_dir, n_stimuli=2)
    index = (experiment_dir / "index.html").read_text(encoding="utf-8")

    # The template's own consent step + "I agree" button must be gone; the only
    # consent is the gate injected by stage_experiment at deploy time.
    assert "I agree" not in index
    assert "consentHtml" not in index
    # The instructions screen (the real first screen) remains.
    assert "Instructions" in index


def test_render_template_experiment_writes_design_and_config(tmp_path):
    exp_dir = tmp_path / "smoke_experiment"
    render_template_experiment(exp_dir, n_stimuli=2)

    stimuli = json.loads((exp_dir / "design" / "stimuli.json").read_text(encoding="utf-8"))
    assert len(stimuli) == 2
    assert {"sequence_a", "sequence_b"} <= set(stimuli[0])

    config = json.loads((exp_dir / "experiment" / "config.json").read_text(encoding="utf-8"))
    # Deploy fills experiment_url later; before deploy it must be null, not absent.
    assert config["experiment_url"] is None


def test_render_template_experiment_embeds_custom_stimuli(tmp_path):
    exp_dir = tmp_path / "smoke_experiment"
    custom = [{"sequence_a": "HHTT", "sequence_b": "HTHT"}]
    experiment_dir = render_template_experiment(exp_dir, stimuli=custom)

    index = (experiment_dir / "index.html").read_text(encoding="utf-8")
    assert "HHTT" in index and "HTHT" in index
    stimuli = json.loads((exp_dir / "design" / "stimuli.json").read_text(encoding="utf-8"))
    assert stimuli == custom
