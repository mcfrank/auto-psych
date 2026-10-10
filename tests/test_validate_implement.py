"""The implement-stage validator must judge only participant-facing content.

A `**bold**` example inside a code comment (e.g. the skeleton's own instruction
to the agent, which it copies into index.html) never reaches participants, so it
must NOT trip the raw-Markdown guard. Only real leaked Markdown in the rendered
text should fail validation.

Each participant's browser must shuffle the trial order. Every page of the
October 2026 live series ran ``STIMULI.forEach`` over the embedded array, so all
participants saw the stimuli in the design's selection order and a stimulus's
effect could not be told apart from its position (user decision 2026-10-10).
"""

from __future__ import annotations

import json
from pathlib import Path

from src.pipelines.outer_loop.orchestrator_validators import _validate_implement
from src.runtime.config import REPO_ROOT

IMPLEMENT_PROMPT = REPO_ROOT / "src/pipelines/outer_loop/prompts/3_implement.md"

# A minimal index.html that satisfies every other implement guardrail (jsPsych,
# button responses, the prose container, the data contract, a per-participant
# shuffle). ``{extra}`` lets a test inject more script content.
_BASE_INDEX = """<!DOCTYPE html>
<html><head>
  <script src="jspsych.js"></script>
  <style>.auto-psych-prose {{ max-width: 700px; }}</style>
</head><body>
<script>
  const STIMULI = [];
  jsPsych.randomization.shuffle(STIMULI).forEach(function (s) {{}});
  const trial = {{
    type: jsPsychHtmlButtonResponse,
    stimulus: "<div class='auto-psych-prose'>Which is more random?</div>",
    choices: ["Left", "Right"],
  }};
{extra}
  data.chose_left = 1; data.sequence_a = "HT"; data.sequence_b = "TH";
</script>
</body></html>
"""


def _make_experiment(tmp_path: Path, extra: str, *, trailing: str = "") -> Path:
    exp = tmp_path / "experiment1"
    (exp / "experiment").mkdir(parents=True)
    html = _BASE_INDEX.format(extra=extra) + trailing
    (exp / "experiment" / "index.html").write_text(html, encoding="utf-8")
    (exp / "experiment" / "config.json").write_text(
        json.dumps({"experiment_url": None}), encoding="utf-8"
    )
    return exp


def test_markdown_bold_inside_comments_is_not_flagged(tmp_path: Path):
    """The exact skeleton comment that broke pilot3 — plus block and HTML
    comments — must pass: comments never reach participants."""
    extra = (
        "  //    convert **bold** -> <strong>bold</strong>. NEVER emit raw Markdown (no `**`).\n"
        "  /* see **note** for details */\n"
    )
    exp = _make_experiment(
        tmp_path, extra, trailing="\n<!-- TODO: **review** wording -->\n"
    )
    ok, msg = _validate_implement(exp)
    assert ok, msg


def test_markdown_bold_in_participant_text_is_flagged(tmp_path: Path):
    """Real leaked Markdown in a rendered string must still fail (guards against
    over-stripping)."""
    extra = '  const instructions = "Please read **carefully** before you begin.";'
    exp = _make_experiment(tmp_path, extra)
    ok, msg = _validate_implement(exp)
    assert not ok
    assert "Markdown bold" in msg


def _write_page(tmp_path: Path, html: str) -> Path:
    exp = tmp_path / "experiment1"
    (exp / "experiment").mkdir(parents=True)
    (exp / "experiment" / "index.html").write_text(html, encoding="utf-8")
    (exp / "experiment" / "config.json").write_text(
        json.dumps({"experiment_url": None}), encoding="utf-8"
    )
    return exp


def _prompt_skeleton() -> str:
    text = IMPLEMENT_PROMPT.read_text(encoding="utf-8")
    return text.split("```html\n", 1)[1].split("```", 1)[0]


def test_the_prompts_fixed_structure_passes_the_validator(tmp_path: Path):
    ok, msg = _validate_implement(_write_page(tmp_path, _prompt_skeleton()))
    assert ok, msg


def test_a_page_that_runs_the_trials_in_the_embedded_order_is_rejected(tmp_path: Path):
    """The October 2026 pages: side counterbalanced, order fixed."""
    html = _prompt_skeleton().replace(
        "jsPsych.randomization.shuffle(STIMULI).forEach(", "STIMULI.forEach("
    )
    assert "STIMULI.forEach(" in html
    ok, msg = _validate_implement(_write_page(tmp_path, html))
    assert not ok
    assert "trial order" in msg


def test_a_shuffle_only_in_a_comment_is_rejected(tmp_path: Path):
    html = _BASE_INDEX.format(extra="").replace(
        "jsPsych.randomization.shuffle(STIMULI).forEach(",
        "// jsPsych.randomization.shuffle(STIMULI)\n  STIMULI.forEach(",
    )
    ok, msg = _validate_implement(_write_page(tmp_path, html))
    assert not ok
    assert "trial order" in msg


def test_shuffling_a_copy_but_running_the_embedded_array_is_rejected(tmp_path: Path):
    html = _BASE_INDEX.format(extra="").replace(
        "jsPsych.randomization.shuffle(STIMULI).forEach(",
        "var order = jsPsych.randomization.shuffle(STIMULI);\n  STIMULI.forEach(",
    )
    ok, msg = _validate_implement(_write_page(tmp_path, html))
    assert not ok
    assert "trial order" in msg
