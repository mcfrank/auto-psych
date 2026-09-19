"""Output validators for the outer-loop pipeline stages.

Extracted from ``orchestrator.py``. Each stage's output (the seeded/carried
model set, the design, the implemented experiment, the collected data, the inner
model loop) is checked here, and the ``(ok, message)`` result drives the repair
loop in ``run.py``. A validator reports; it never raises — anything a reader
refuses becomes the feedback handed back to the agent.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from src.models.model_manifest import (
    manifest_path,
    read_manifest_entries,
    read_manifest_names,
)
from src.runtime.config import REPO_ROOT

# Inner-loop zoo candidate names (iterN_candidateM). An auto-named winner is
# exported under the stable `inner_loop_model` name, so a carried/validated model
# set must never contain a raw zoo name. Shared with orchestrator._export_*.
_ZOO_NAME_RE = re.compile(r"iter\d+_candidate\d+")


def validate_cc_output(agent_key: str, exp_dir: Path) -> tuple[bool, str]:
    """Validate agent output. Returns (ok, message)."""
    validators = {
        # "models" is not an agent stage: it validates the experiment's
        # cognitive_models/ set after seeding (experiment 1) or carry-forward
        # (experiments >= 2). New hypotheses enter only via the inner loop.
        "models": _validate_model_set,
        "2_design": _validate_design,
        "3_implement": _validate_implement,
        "4_collect": _validate_collect,
        "5_model_loop": _validate_model_loop,
    }
    fn = validators.get(agent_key)
    return fn(exp_dir) if fn else (True, "No validator for this agent")


def _validate_model_set(exp_dir: Path) -> tuple[bool, str]:
    """Validate that every manifest model is a loadable PyMC model.

    Each `<name>.py` must define a module-level `model: pm.Model` with exactly
    one observed-response container (so the inner loop can fit it). Validation
    only builds the model graph — it never samples.
    """
    sys.path.insert(0, str(REPO_ROOT))
    from src.models.model_loading import load_pymc_model, observed_response_data  # type: ignore

    models_dir = exp_dir / "cognitive_models"
    # A validator reports; it does not raise. Anything the manifest reader
    # refuses (missing file, unparseable YAML, a nameless entry) becomes the
    # feedback the repair loop hands back to the agent.
    try:
        entries = read_manifest_entries(models_dir)
    except (FileNotFoundError, ValueError) as e:
        return False, str(e)
    if not entries:
        return False, f"{manifest_path(models_dir)} has no models"

    # Each entry carries the model name and its rationale — the one-sentence
    # natural-language hypothesis the model implements. A model with no stated
    # hypothesis is rejected: every model must be a specific, testable claim.
    names = [entry["name"] for entry in entries]

    # Only the previous experiment's cognitive_models/ carries forward (its
    # protected seeds + the zoo survivors the export renamed where needed). The
    # inner loop's fallback-named candidates (`iterN_candidateM`) export under
    # `inner_loop_model[_k]` and must never appear under their zoo name — reject
    # them so the repair loop makes the agent drop them rather than silently
    # bloating every later experiment.
    zoo = [n for n in names if _ZOO_NAME_RE.fullmatch(n)]
    if zoo:
        return (
            False,
            f"models_manifest.yaml carries inner-loop zoo candidate(s) {zoo} from the "
            "previous experiment's model_loop/. Carry forward ONLY the previous "
            "experiment's cognitive_models/ (its seeds plus the exported survivors); "
            "never copy candidates from model_loop/models/ under their zoo names.",
        )

    for entry in entries:
        name = entry["name"]
        if not (entry.get("rationale") or "").strip():
            return (
                False,
                f"Model '{name}' states no hypothesis: every model needs a non-empty "
                "'rationale' in models_manifest.yaml naming the single cognitive "
                "hypothesis it implements",
            )
        if not (models_dir / f"{name}.py").exists():
            return (
                False,
                f"Model '{name}' has no {models_dir}/{name}.py (every manifest "
                f"entry needs its model file)",
            )
        try:
            model = load_pymc_model(name, models_dir)
        except Exception as e:
            return False, f"Model '{name}' is not a loadable PyMC model: {e}"
        try:
            observed_response_data(model)
        except Exception as e:
            return (
                False,
                f"Model '{name}' has no usable observed-response pm.Data container: {e}",
            )
    return True, f"Theory valid: {names}"


def _validate_design(exp_dir: Path) -> tuple[bool, str]:
    path = exp_dir / "design" / "stimuli.json"
    if not path.exists():
        return False, f"stimuli.json not found at {path}"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        return False, f"Invalid JSON: {e}"
    if not isinstance(data, list):
        return False, "stimuli.json is not a list"
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            return False, f"Stimulus {i} is not a dict"
        if "sequence_a" not in item or "sequence_b" not in item:
            return False, f"Stimulus {i} missing sequence_a or sequence_b"
        if "eig" not in item:
            return False, f"Stimulus {i} missing 'eig' field"
    return True, f"Design valid: {len(data)} stimuli"


def _strip_code_comments(text: str) -> str:
    """Remove HTML/JS/CSS comments so content checks see only participant-facing
    code. Comments are scaffolding (the skeleton's instructions to the agent, the
    agent's own notes) and never render to participants, so e.g. a `**bold**`
    example inside a comment must not trip the raw-Markdown guard.
    """
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)  # HTML comments
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)  # /* block */ (JS, CSS)
    text = re.sub(r"//[^\n]*", "", text)  # // line comments
    return text


def _validate_implement(exp_dir: Path) -> tuple[bool, str]:
    """Validate the implemented experiment AND enforce cross-experiment
    consistency: the ONLY thing that may differ between experiments (within or
    across runs) is the stimuli. Reject drift in response modality, the data
    contract, or the standard structure so every experiment looks/behaves alike.
    """
    index_path = exp_dir / "experiment" / "index.html"
    config_path = exp_dir / "experiment" / "config.json"
    if not index_path.exists():
        return False, f"index.html not found at {index_path}"
    text = index_path.read_text(encoding="utf-8")
    low = text.lower()
    if "jspsych" not in low:
        return False, "index.html does not mention jsPsych"
    if not config_path.exists():
        return False, f"config.json not found at {config_path}"
    try:
        json.loads(config_path.read_text(encoding="utf-8"))
    except Exception as e:
        return False, f"Invalid config.json: {e}"

    # --- consistency guardrails ------------------------------------------------
    # Response modality MUST be button responses ONLY (identical across
    # experiments) — no keyboard responses anywhere, so the modality can't drift.
    if "jspsychhtmlbuttonresponse" not in low:
        return False, (
            "consistency: the choice must be collected with jsPsychHtmlButtonResponse "
            "(button responses) so every experiment uses the same response modality — "
            "none found."
        )
    if "jspsychhtmlkeyboardresponse" in low or "jspsychkeyboardresponse" in low:
        return False, (
            "consistency: the experiment uses a keyboard-response plugin. Every "
            "experiment must use BUTTON responses only (jsPsychHtmlButtonResponse) — "
            "for fixations/spacing use a button trial with trial_duration/post_trial_gap, "
            "not a keyboard trial."
        )
    # Formatting consistency: instruction/debrief prose MUST render as HTML, never
    # leak raw Markdown. A literal `**bold**` (or `*emph*`/`__bold__`) means the
    # agent pasted the problem-definition Markdown verbatim instead of converting it
    # to <strong>/<em>; participants would see the asterisks. Reject it.
    # Check only participant-facing code: comments (e.g. the skeleton's own
    # `**bold**` instruction to the agent, which it copies verbatim) never reach
    # participants and must not trip this guard.
    # Paired `**…**` (opens with a letter) signals leaked Markdown bold without
    # tripping on JS exponentiation like `x**2` (no letter immediately after `**`).
    visible = _strip_code_comments(text)
    if re.search(r"\*\*[A-Za-z][^*]*?\*\*", visible):
        return False, (
            "formatting: index.html contains literal Markdown bold (`**...**`). The "
            "instruction/choice/debrief wording must be rendered as HTML — convert "
            "`**bold**` to <strong>bold</strong> and never emit raw asterisks to "
            "participants."
        )
    if re.search(r"(?<![\w*])__[A-Za-z].*?[A-Za-z]__(?![\w*])", visible):
        return False, (
            "formatting: index.html contains literal Markdown bold (`__...__`). Render "
            "emphasis as HTML (<strong>/<em>), not raw Markdown."
        )
    # Readability: long instruction/debrief text must sit in a constrained-width,
    # left-aligned prose container (the fixed `.auto-psych-prose` class from the
    # skeleton), so it does not stretch edge-to-edge across wide screens.
    if "auto-psych-prose" not in text:
        return False, (
            "readability: instructions and debrief must be wrapped in the fixed "
            "`<div class=\"auto-psych-prose\">…</div>` container (a max-width, "
            "left-aligned, line-height block) so prose does not span the full screen "
            "width. Copy the .auto-psych-prose rule and wrappers from the skeleton."
        )
    # Data contract: each trial must record chose_left (1=left/first sequence) plus
    # the raw sequences, or the collection/Firestore step cannot parse the data.
    for needed in ("chose_left", "sequence_a", "sequence_b"):
        if needed not in text:
            return False, (
                f"consistency/data contract: index.html never sets `{needed}` — every "
                "trial must record chose_left (1 if the LEFT/first sequence was chosen) "
                "plus sequence_a and sequence_b."
            )
    # The experiment must present exactly the design's stimuli — nothing added,
    # dropped, or altered. Each stimulus's raw sequences must appear verbatim.
    stim_path = exp_dir / "design" / "stimuli.json"
    if stim_path.exists():
        try:
            stimuli = json.loads(stim_path.read_text(encoding="utf-8"))
        except Exception as e:
            return False, f"design/stimuli.json is invalid: {e}"
        missing = [
            i
            for i, s in enumerate(stimuli)
            if isinstance(s, dict)
            and not (
                str(s.get("sequence_a", "\0")) in text
                and str(s.get("sequence_b", "\0")) in text
            )
        ]
        if missing:
            return False, (
                f"consistency: {len(missing)} of {len(stimuli)} design stimuli are not "
                f"embedded verbatim in index.html (e.g. indices {missing[:5]}). The "
                "experiment must present exactly the design's stimuli."
            )
    return True, "Implement valid (consistency guardrails passed)"


def _validate_collect(exp_dir: Path) -> tuple[bool, str]:
    path = exp_dir / "data" / "responses.csv"
    if not path.exists():
        return False, f"responses.csv not found at {path}"
    lines = path.read_text(encoding="utf-8").strip().split("\n")
    if len(lines) < 2:
        return False, "responses.csv has no data rows"
    header = {h.strip() for h in lines[0].split(",")}
    required = {
        "participant_id",
        "trial_index",
        "sequence_a",
        "sequence_b",
        "chose_left",
    }
    missing = required - header
    if missing:
        return False, f"responses.csv missing columns: {missing}"
    return True, f"Collect valid: {len(lines) - 1} rows"


def _validate_model_loop(exp_dir: Path) -> tuple[bool, str]:
    loop_dir = exp_dir / "model_loop"
    report = loop_dir / "report.md"
    posterior = loop_dir / "model_posterior.json"

    if not posterior.exists():
        return False, "model_loop/model_posterior.json not found"
    try:
        data = json.loads(posterior.read_text(encoding="utf-8"))
        if "posteriors" not in data:
            return False, "model_loop/model_posterior.json missing 'posteriors'"
        if not data.get("comparison"):
            # The registry updater needs the az.compare stacking weights.
            return False, "model_loop/model_posterior.json missing 'comparison'"
    except Exception as e:
        return False, f"Invalid model_posterior.json: {e}"
    if not report.exists():
        return False, "model_loop/report.md not found"
    if not report.read_text(encoding="utf-8").strip():
        return False, "model_loop/report.md is empty"

    # The best model must be in this experiment's model set — either it was
    # already there (a seed that won again) or the export copied it in. A
    # fallback auto-named winner exports under the legacy `inner_loop_model`.
    # Use the exported selection recorded by _export (the best *reliable* model,
    # which can differ from the raw posterior argmax when the argmax was excluded
    # as PSIS-LOO-unreliable); fall back to the argmax for older exports.
    best = data.get("best_model") or max(
        data["posteriors"], key=lambda m: data["posteriors"][m]
    )
    required = "inner_loop_model" if _ZOO_NAME_RE.fullmatch(best) else best
    models_dir = exp_dir / "cognitive_models"
    try:
        names = set(read_manifest_names(models_dir))
    except (FileNotFoundError, ValueError) as e:
        return False, str(e)
    if required not in names or not (models_dir / f"{required}.py").exists():
        return (
            False,
            f"best model {required!r} is not in cognitive_models (manifest + .py "
            f"file required — was the inner-loop export skipped?)",
        )
    return True, f"Model loop valid ({len(report.read_text())} char report)"
