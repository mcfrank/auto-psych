"""Smoke experiment fixture for deployment testing."""

from __future__ import annotations

import json
from pathlib import Path


SMOKE_STIMULI = [
    {"sequence_a": "HHHTTT", "sequence_b": "HTHTHT"},
    {"sequence_a": "HHHHHT", "sequence_b": "THTHHT"},
]


def write_smoke_experiment(exp_dir: Path) -> Path:
    """Write a tiny self-contained jsPsych experiment for deployment smoke tests."""
    experiment_dir = exp_dir / "experiment"
    design_dir = exp_dir / "design"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    design_dir.mkdir(parents=True, exist_ok=True)

    stimuli_json = json.dumps(SMOKE_STIMULI, indent=2)
    (design_dir / "stimuli.json").write_text(stimuli_json + "\n", encoding="utf-8")
    (experiment_dir / "config.json").write_text('{"experiment_url": null}\n', encoding="utf-8")
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
