"""The implement-stage validator must judge only participant-facing content.

A `**bold**` example inside a code comment (e.g. the skeleton's own instruction
to the agent, which it copies into index.html) never reaches participants, so it
must NOT trip the raw-Markdown guard. Only real leaked Markdown in the rendered
text should fail validation.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.pipelines.outer_loop.orchestrator import _validate_implement

# A minimal index.html that satisfies every other implement guardrail (jsPsych,
# button responses, the prose container, the data contract). ``{extra}`` lets a
# test inject more script content.
_BASE_INDEX = """<!DOCTYPE html>
<html><head>
  <script src="jspsych.js"></script>
  <style>.auto-psych-prose {{ max-width: 700px; }}</style>
</head><body>
<script>
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
    exp = _make_experiment(tmp_path, extra, trailing="\n<!-- TODO: **review** wording -->\n")
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
