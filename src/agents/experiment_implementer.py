"""Experiment implementer: fill generic jsPsych template with consent, instructions, and stimuli."""

from pathlib import Path
from typing import Any, Dict, List
import json

from src.config import REPO_ROOT, agent_dir
from src.console_log import agent_header, log_status


TEMPLATES_DIR = REPO_ROOT / "templates"
JSPSYCH_TEMPLATE_PATH = TEMPLATES_DIR / "jspsych_experiment.html"
CONSENT_PATH = TEMPLATES_DIR / "consent.txt"


def _text_to_html_paragraphs(text: str) -> str:
    """Convert plain text to HTML: each paragraph (blank-line-separated) in a <p> tag."""
    if not text or not text.strip():
        return "<p>Press any key to continue.</p>"
    parts = [f"<p>{p.strip()}</p>" for p in text.strip().split("\n\n") if p.strip()]
    return "\n      ".join(parts) if parts else "<p>Press any key to continue.</p>"


def _build_stimulus_display(sequence_a: str, sequence_b: str, response_prompt: str) -> str:
    """Build the HTML shown on one trial: question at top, then left/right sequences below."""
    seq_a_spaced = " ".join(sequence_a) if isinstance(sequence_a, str) else str(sequence_a)
    seq_b_spaced = " ".join(sequence_b) if isinstance(sequence_b, str) else str(sequence_b)
    return (
        f'<p style="font-size: 20px; font-weight: 600; margin-bottom: 1.5em; text-align: center;">{response_prompt}</p>'
        '<div style="display: flex; justify-content: space-around; align-items: center; gap: 3em; margin: 2em auto; max-width: 800px;">'
        f'<div style="flex: 1; text-align: center;"><p style="font-family: monospace; font-size: 22px; margin: 0.5em 0;">{seq_a_spaced}</p><p style="font-size: 14px; color: #555;">Left</p></div>'
        f'<div style="flex: 1; text-align: center;"><p style="font-family: monospace; font-size: 22px; margin: 0.5em 0;">{seq_b_spaced}</p><p style="font-size: 14px; color: #555;">Right</p></div>'
        '</div>'
    )


def _build_trial_variables(
    stimuli: List[Dict[str, Any]],
    response_prompt: str = "Which looks more random? Press <strong>F</strong> or <strong>Left</strong> for left, <strong>J</strong> or <strong>Right</strong> for right.",
) -> List[Dict[str, Any]]:
    """Build timeline_variables: each item has stimulus_display, sequence_a, sequence_b."""
    return [
        {
            "stimulus_display": _build_stimulus_display(s["sequence_a"], s["sequence_b"], response_prompt),
            "sequence_a": s["sequence_a"],
            "sequence_b": s["sequence_b"],
        }
        for s in stimuli
    ]


def run_experiment_implementer(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Load the generic jsPsych template and consent; build instructions and trial variables
    from the problem definition and designer stimuli; write index.html and stimuli.json.
    """
    project_id = state["project_id"]
    run_id = state["run_id"]
    agent_header("3experiment_implementer", run_id, state.get("total_runs"), state.get("mode"))
    if state.get("validation_retry_count", 0) > 0:
        log_status(f"Repeating due to validation failure (attempt {state['validation_retry_count']}/3)")
    out_dir = agent_dir(project_id, run_id, "3experiment_implementer")
    out_dir.mkdir(parents=True, exist_ok=True)

    stimuli_path = Path(state["stimuli_path"])
    raw_stimuli = json.loads(stimuli_path.read_text()) if stimuli_path.exists() else []
    stimuli_for_experiment = [
        {"sequence_a": s["sequence_a"], "sequence_b": s["sequence_b"]}
        for s in raw_stimuli
        if isinstance(s, dict) and "sequence_a" in s and "sequence_b" in s
    ]

    prob_path = Path(state["problem_definition_path"])
    problem_text = prob_path.read_text(encoding="utf-8").strip() if prob_path.exists() else ""
    # First substantive line/paragraph as short task description
    task_line = problem_text.split("\n")[0].strip() if problem_text else "Which sequence looks more random?"
    if not task_line or task_line.startswith("#"):
        task_line = "Which sequence looks more random?"

    # Experiment-specific instructions (agent can customize; here we use problem def + key mapping)
    instructions_html = f'<p><strong>Instructions</strong></p><p>{task_line}</p><p>On each trial you will see two sequences of coin flips (H and T).</p><p><strong>Press F or Left arrow</strong> for the left sequence.</p><p><strong>Press J or Right arrow</strong> for the right sequence.</p><p>Press any key to begin.</p>'

    if not JSPSYCH_TEMPLATE_PATH.exists():
        _write_minimal_experiment(out_dir, stimuli_for_experiment, task_line)
    else:
        consent_text = CONSENT_PATH.read_text(encoding="utf-8") if CONSENT_PATH.exists() else "Consent.\n\nPress any key to continue."
        consent_html = _text_to_html_paragraphs(consent_text)
        trial_variables = _build_trial_variables(stimuli_for_experiment)
        trial_choices = ["f", "j", "ArrowLeft", "ArrowRight"]
        debrief_html = "<p>Thank you for participating.</p><p>Press any key to finish.</p>"

        template_html = JSPSYCH_TEMPLATE_PATH.read_text(encoding="utf-8")
        # JSON-dump all so we get valid JS strings; escape </ to avoid breaking out of script
        def safe_js(s: str) -> str:
            return json.dumps(s).replace("</", "<\\/")
        replacements = {
            "{{CONSENT_HTML}}": safe_js(consent_html),
            "{{INSTRUCTIONS_HTML}}": safe_js(instructions_html),
            "{{STIMULI_JSON}}": json.dumps(trial_variables).replace("</", "<\\/"),
            "{{TRIAL_CHOICES_JSON}}": json.dumps(trial_choices),
            "{{DEBRIEF_HTML}}": safe_js(debrief_html),
        }
        for placeholder, value in replacements.items():
            template_html = template_html.replace(placeholder, value)
        (out_dir / "index.html").write_text(template_html, encoding="utf-8")

    (out_dir / "stimuli.json").write_text(json.dumps(stimuli_for_experiment, indent=2), encoding="utf-8")
    return {
        **state,
        "experiment_path": str(out_dir),
    }


def _write_minimal_experiment(out_dir: Path, stimuli: list, task_instruction: str) -> None:
    """Fallback if template is missing."""
    consent_text = CONSENT_PATH.read_text(encoding="utf-8") if CONSENT_PATH.exists() else "Consent. Press any key to continue."
    consent_html = _text_to_html_paragraphs(consent_text)
    trial_vars = _build_trial_variables(stimuli)
    stimuli_esc = json.dumps(trial_vars).replace("</", "<\\/")
    task_esc = json.dumps(task_instruction)
    consent_esc = json.dumps(consent_html).replace("</", "<\\/")
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>Experiment</title>
  <script src="https://unpkg.com/jspsych@8.2.3"></script>
  <script src="https://unpkg.com/@jspsych/plugin-html-keyboard-response@2.1.0"></script>
  <link href="https://unpkg.com/jspsych@8.2.3/css/jspsych.css" rel="stylesheet" />
  <script src="/jatos.js"></script>
</head>
<body></body>
<script>
  const trialVariables = {stimuli_esc};
  const jsPsych = initJsPsych({{ on_finish: () => typeof jatos !== 'undefined' ? jatos.startNextComponent(jsPsych.data.get().json()) : console.log(jsPsych.data.get().json()) }});
  const timeline = [];
  timeline.push({{ type: jsPsychHtmlKeyboardResponse, stimulus: {consent_esc}, choices: "ALL_KEYS" }});
  timeline.push({{ type: jsPsychHtmlKeyboardResponse, stimulus: '<p>' + {task_esc} + '</p><p>F/Left = left, J/Right = right. Press any key to begin.</p>', choices: "ALL_KEYS" }});
  trialVariables.forEach((v) => {{
    timeline.push({{ type: jsPsychHtmlKeyboardResponse, stimulus: v.stimulus_display, choices: ['f','j','ArrowLeft','ArrowRight'], data: {{ sequence_a: v.sequence_a, sequence_b: v.sequence_b }}, on_finish: data => {{ data.chose_left = (data.response === 'f' || data.response === 'ArrowLeft'); }} }});
  }});
  timeline.push({{ type: jsPsychHtmlKeyboardResponse, stimulus: '<p>Thank you. Press any key to finish.</p>', choices: "ALL_KEYS" }});
  jsPsych.run(timeline);
</script>
</html>
"""
    (out_dir / "index.html").write_text(html, encoding="utf-8")
