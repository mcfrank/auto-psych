"""Smoke experiment fixtures for deployment testing.

Two fixtures live here:

- :func:`write_smoke_experiment` — a tiny self-contained jsPsych experiment
  (button responses) used by the deployment unit tests.
- :func:`render_template_experiment` — renders the *real* participant-facing
  template (``templates/jspsych_experiment.html``: consent → instructions →
  button-response trials → debrief, with the live ``onFinish`` → ``/submit``
  bridge) over a small set of subjective-randomness stimuli. This is the artifact
  the Firebase + Prolific smoke test (``scripts/smoke_firebase_deploy.py``)
  deploys, so the simulated participants exercise exactly what a Prolific worker
  sees — and the same button-response modality the implement validator enforces.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.runtime.config import REPO_ROOT

TEMPLATE_PATH = REPO_ROOT / "templates" / "jspsych_experiment.html"

# Button labels the template's trial renders; the first button (index 0) is the
# LEFT / Sequence A option, the second is the RIGHT / Sequence B option. Matches
# the template's `data.chose_left = (response === 0)` (button responses only —
# the same modality the pipeline's implement validator enforces).
TRIAL_CHOICES = ["Sequence A (left)", "Sequence B (right)"]

# A handful of fixed H/T pairs that make the randomness judgment non-trivial
# (a streaky sequence vs. an alternating/typical-looking one). Sliced to the
# requested count, so the smoke test stays small and fast.
_RANDOMNESS_STIMULUS_PAIRS = [
    ("HHHTTT", "HTHTHT"),
    ("HHHHHT", "THTHHT"),
    ("HHHHHHHH", "HTHHTHTT"),
    ("HTHTHTHT", "HHTHTTHT"),
    ("TTTTTTTT", "HTHHTHTT"),
    ("HHHHTTTT", "HTTHTHHT"),
]

DEFAULT_INSTRUCTIONS_HTML = (
    "<h2>Instructions</h2>"
    '<p style="max-width:560px; text-align:left; margin:auto;">'
    "On each screen you will see two sequences of coin flips, on the left and "
    "the right. Decide which one looks <strong>more random</strong> to you, "
    "then press <strong>F</strong> for the left sequence or <strong>J</strong> "
    "for the right sequence. There are no right or wrong answers.</p>"
)

DEFAULT_DEBRIEF_HTML = (
    "<h2>Thank you!</h2>"
    '<p style="max-width:560px; text-align:left; margin:auto;">'
    "You have completed the study and your responses have been recorded. This "
    "study examines the systematic intuitions people hold about what random "
    "coin-flip sequences look like.</p>"
)


SMOKE_STIMULI = [
    {"sequence_a": "HHHTTT", "sequence_b": "HTHTHT"},
    {"sequence_a": "HHHHHT", "sequence_b": "THTHHT"},
]


def make_randomness_stimuli(n_stimuli: int) -> list[dict[str, str]]:
    """Return ``n_stimuli`` H/T sequence pairs for the randomness judgment.

    Drawn from a fixed pool (cycled if ``n_stimuli`` exceeds it) so the smoke
    test is deterministic. Each pair is ``{"sequence_a", "sequence_b"}`` of
    distinct H/T strings.
    """
    if n_stimuli <= 0:
        raise ValueError(f"n_stimuli must be positive, got {n_stimuli}")
    pairs = [
        _RANDOMNESS_STIMULUS_PAIRS[i % len(_RANDOMNESS_STIMULUS_PAIRS)]
        for i in range(n_stimuli)
    ]
    return [{"sequence_a": seq_a, "sequence_b": seq_b} for seq_a, seq_b in pairs]


def render_stimulus_display(sequence_a: str, sequence_b: str) -> str:
    """HTML shown above the choice buttons: the two sequences side by side.

    The trial is a two-button trial (``Sequence A (left)`` / ``Sequence B
    (right)``), rendered by jsPsych below this display. Uses inline styles (not
    CSS classes) so the side-by-side left/right layout survives regardless of the
    template's ``<style>`` block — the real template only styles the progress bar
    and buttons, so class-based styling here would silently collapse the two
    sequences into a vertical stack.
    """
    box_style = (
        "text-align:center; padding:20px 32px; background:#f5f5f5; "
        "border:2px solid #ccc; border-radius:10px; min-width:150px;"
    )
    label_style = (
        "font-size:14px; color:#666; margin-bottom:10px; "
        "text-transform:uppercase; letter-spacing:1px;"
    )
    seq_style = (
        "font-family:monospace; font-size:32px; font-weight:bold; "
        "letter-spacing:4px; color:#222;"
    )
    return (
        '<p style="font-size:20px; margin-bottom:10px;">Which sequence looks '
        "<strong>more random</strong>?</p>"
        '<div style="display:flex; justify-content:center; align-items:center; '
        'gap:60px; margin:30px 0;">'
        f'<div style="{box_style}"><div style="{label_style}">Sequence A (left)</div>'
        f'<div style="{seq_style}">{sequence_a}</div></div>'
        f'<div style="{box_style}"><div style="{label_style}">Sequence B (right)</div>'
        f'<div style="{seq_style}">{sequence_b}</div></div>'
        "</div>"
        '<p style="color:#555;">Click <strong>Sequence A</strong> for the left '
        "sequence or <strong>Sequence B</strong> for the right.</p>"
    )


def render_template_experiment(
    exp_dir: Path,
    *,
    n_stimuli: int = 4,
    stimuli: list[dict[str, str]] | None = None,
    instructions_html: str = DEFAULT_INSTRUCTIONS_HTML,
    debrief_html: str = DEFAULT_DEBRIEF_HTML,
) -> Path:
    """Render the real participant template into a deployable experiment dir.

    Substitutes the ``{{...}}`` placeholders in
    ``templates/jspsych_experiment.html`` and writes
    ``<exp_dir>/experiment/index.html``, ``experiment/config.json`` (with
    ``experiment_url: null`` — the deploy step fills it), and
    ``design/stimuli.json``. Returns the ``experiment`` directory.

    Pass ``stimuli`` to supply explicit ``{"sequence_a", "sequence_b"}`` pairs;
    otherwise ``n_stimuli`` pairs come from :func:`make_randomness_stimuli`.
    """
    if not TEMPLATE_PATH.exists():
        raise FileNotFoundError(f"jsPsych template not found at {TEMPLATE_PATH}")

    if stimuli is None:
        stimuli = make_randomness_stimuli(n_stimuli)
    if not stimuli:
        raise ValueError("render_template_experiment needs at least one stimulus")

    # Ship both side orderings precomputed; the template randomizes per trial
    # which one to show (side counterbalancing) and records the presented order.
    trial_variables: list[dict[str, Any]] = [
        {
            "sequence_a": s["sequence_a"],
            "sequence_b": s["sequence_b"],
            "display_a_left": render_stimulus_display(s["sequence_a"], s["sequence_b"]),
            "display_b_left": render_stimulus_display(s["sequence_b"], s["sequence_a"]),
        }
        for s in stimuli
    ]

    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    # Each placeholder is consumed as a JS value (e.g. `const x = {{X}};`), so
    # substitute JSON — strings become quoted JS string literals, lists become
    # JS array literals — which is exactly what the template expects.
    substitutions = {
        "{{INSTRUCTIONS_HTML}}": json.dumps(instructions_html),
        "{{STIMULI_JSON}}": json.dumps(trial_variables, indent=2),
        "{{TRIAL_CHOICES_JSON}}": json.dumps(TRIAL_CHOICES),
        "{{DEBRIEF_HTML}}": json.dumps(debrief_html),
    }
    rendered = template
    for placeholder, value in substitutions.items():
        if placeholder not in rendered:
            raise ValueError(
                f"Template {TEMPLATE_PATH} is missing expected placeholder {placeholder}"
            )
        rendered = rendered.replace(placeholder, value)
    if "{{" in rendered or "}}" in rendered:
        raise ValueError(
            "Rendered experiment still contains an unfilled {{...}} placeholder; "
            "the template gained a placeholder this renderer does not handle."
        )

    experiment_dir = exp_dir / "experiment"
    design_dir = exp_dir / "design"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    design_dir.mkdir(parents=True, exist_ok=True)

    (experiment_dir / "index.html").write_text(rendered, encoding="utf-8")
    (experiment_dir / "config.json").write_text(
        '{"experiment_url": null}\n', encoding="utf-8"
    )
    (design_dir / "stimuli.json").write_text(
        json.dumps([dict(s) for s in stimuli], indent=2) + "\n", encoding="utf-8"
    )
    return experiment_dir


def write_smoke_experiment(exp_dir: Path) -> Path:
    """Write a tiny self-contained jsPsych experiment for deployment smoke tests."""
    experiment_dir = exp_dir / "experiment"
    design_dir = exp_dir / "design"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    design_dir.mkdir(parents=True, exist_ok=True)

    stimuli_json = json.dumps(SMOKE_STIMULI, indent=2)
    (design_dir / "stimuli.json").write_text(stimuli_json + "\n", encoding="utf-8")
    (experiment_dir / "config.json").write_text(
        '{"experiment_url": null}\n', encoding="utf-8"
    )
    (experiment_dir / "index.html").write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Auto-psych Smoke Experiment</title>
  <script src="https://unpkg.com/jspsych@7.3.4"></script>
  <script src="https://unpkg.com/@jspsych/plugin-html-button-response@1.1.3"></script>
  <link rel="stylesheet" href="https://unpkg.com/jspsych@7.3.4/css/jspsych.css">
</head>
<body></body>
<script>
const stimuli = {stimuli_json};
const jsPsych = initJsPsych({{
  on_finish: function() {{
    window.__experimentData = jsPsych.data.get().values();
  }}
}});
const timeline = stimuli.map((s) => ({{
  type: jsPsychHtmlButtonResponse,
  stimulus: `<p>Which sequence looks more random?</p><p>A: ${{s.sequence_a}}</p><p>B: ${{s.sequence_b}}</p>`,
  choices: ["Sequence A", "Sequence B"],
  data: {{
    sequence_a: s.sequence_a,
    sequence_b: s.sequence_b
  }},
  on_finish: function(data) {{
    data.chose_left = data.response === 0;
    data.chose_right = data.response === 1;
  }}
}}));
jsPsych.run(timeline);
</script>
</html>
""",
        encoding="utf-8",
    )
    return experiment_dir
