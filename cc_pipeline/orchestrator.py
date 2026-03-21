"""
Orchestrator helpers for the Claude Code agent pipeline.

Responsibilities:
- Write CONTEXT.md before each agent run
- Spawn claude CLI as subprocess
- Run programmatic collect step directly
- Validate outputs
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"

# ─────────────────────────────────────────────
# Directory helpers
# ─────────────────────────────────────────────

def cc_projects_dir() -> Path:
    return REPO_ROOT / "cc_pipeline" / "projects"


def cc_project_dir(project_id: str) -> Path:
    return cc_projects_dir() / project_id


def experiment_dir(project_id: str, exp_num: int) -> Path:
    return cc_projects_dir() / project_id / f"experiment{exp_num}"


def get_ground_truth_models(project_id: str) -> Dict:
    """Load GROUND_TRUTH_MODELS from cc_pipeline/projects/<project>/ground_truth_models.py."""
    import importlib.util
    path = cc_project_dir(project_id) / "ground_truth_models.py"
    if not path.exists():
        return {}
    spec = importlib.util.spec_from_file_location(f"gt_{project_id}", path)
    if spec is None or spec.loader is None:
        return {}
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    registry = getattr(mod, "GROUND_TRUTH_MODELS", None)
    return dict(registry) if isinstance(registry, dict) else {}


def ensure_experiment_dirs(exp_dir: Path) -> None:
    for sub in ["cognitive_models", "design", "experiment", "data", "critique"]:
        (exp_dir / sub).mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────
# CONTEXT.md writer
# ─────────────────────────────────────────────

def write_context(
    exp_dir: Path,
    agent_key: str,
    project_id: str,
    exp_num: int,
    prev_exp_dir: Optional[Path] = None,
    extra: Optional[Dict[str, Any]] = None,
    complexity_prior_const: float = 0.0,
) -> Path:
    """Write CONTEXT.md into exp_dir for the given agent. Return path."""
    prob_path = REPO_ROOT / "projects" / project_id / "problem_definition.md"

    lines: List[str] = [
        f"# CONTEXT — experiment {exp_num}, agent {agent_key}",
        "",
        f"**Project:** {project_id}",
        f"**Experiment number:** {exp_num}",
        f"**Repo root:** {REPO_ROOT}",
        f"**This experiment directory:** {exp_dir}",
        "",
        "## Key paths",
        "",
        f"- Problem definition: `{prob_path}`",
        f"- Cognitive models dir: `{exp_dir / 'cognitive_models'}`",
        f"- Design dir: `{exp_dir / 'design'}`",
        f"- Experiment dir: `{exp_dir / 'experiment'}`",
        f"- Data dir: `{exp_dir / 'data'}`",
        f"- Responses: `{exp_dir / 'data' / 'responses.csv'}`",
        f"- Model registry: `{exp_dir / 'model_registry.yaml'}`",
        f"- Critique output dir: `{exp_dir / 'critique'}`",
    ]

    if prev_exp_dir and prev_exp_dir.exists():
        lines += ["", "## Previous experiment paths", ""]
        lines += [
            f"- Previous cognitive models: `{prev_exp_dir / 'cognitive_models'}`",
            f"- Previous model registry: `{prev_exp_dir / 'model_registry.yaml'}`",
            f"- Previous critique report: `{prev_exp_dir / 'critique' / 'report.md'}`",
            f"- Previous theory probabilities: `{prev_exp_dir / 'critique' / 'theory_probabilities.yaml'}`",
            f"- Previous model posterior: `{prev_exp_dir / 'critique' / 'model_posterior.json'}`",
        ]

    if agent_key == "5_critique":
        project_dir = exp_dir.parent
        all_responses = []
        all_stimuli = []
        for n in range(1, exp_num + 1):
            r = project_dir / f"experiment{n}" / "data" / "responses.csv"
            s = project_dir / f"experiment{n}" / "design" / "stimuli.json"
            if r.exists():
                all_responses.append(str(r))
            if s.exists():
                all_stimuli.append(str(s))
        lines += ["", "## All experiments (pooled data for posterior and PPCs)", ""]
        lines += ["Response files (all experiments, pass all to --responses):"]
        for p in all_responses:
            lines.append(f"- `{p}`")
        lines += ["", "Stimuli files (all experiments, pass all to --stimuli):"]
        for p in all_stimuli:
            lines.append(f"- `{p}`")
        # Embed the exact posterior command so the agent doesn't have to construct it
        responses_str = " \\\n        ".join(all_responses)
        complexity_flag = (
            f" \\\n    --complexity-prior {complexity_prior_const}"
            if complexity_prior_const != 0.0 else ""
        )
        posterior_cmd = (
            f"cd {REPO_ROOT} && python3 -m src.model_comparison.posterior \\\n"
            f"    --responses \\\n        {responses_str} \\\n"
            f"    --models-dir {exp_dir / 'cognitive_models'} \\\n"
            f"    --out {exp_dir / 'critique' / 'model_posterior.json'}"
            f"{complexity_flag}"
        )
        lines += ["", "## Posterior command (run this exactly)", "", "```bash", posterior_cmd, "```"]

    if extra:
        lines += ["", "## Additional context", ""]
        for k, v in extra.items():
            lines.append(f"- **{k}**: {v}")

    context_path = exp_dir / "CONTEXT.md"
    context_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return context_path


# ─────────────────────────────────────────────
# Claude Code agent spawner
# ─────────────────────────────────────────────

def spawn_cc_agent(
    agent_key: str,
    exp_dir: Path,
    allowed_dirs: Optional[List[Path]] = None,
    timeout_secs: int = 900,
) -> tuple[bool, str]:
    """
    Spawn a Claude Code agent for the given agent_key.
    Reads prompt from cc_pipeline/prompts/<agent_key>.md.
    Tells the agent to read CONTEXT.md and complete the task.
    File tool access is restricted to allowed_dirs (defaults to exp_dir only).
    Bash still runs from REPO_ROOT so python3 -m src.* imports work.
    Returns (success, output_or_error).
    """
    prompt_path = PROMPTS_DIR / f"{agent_key}.md"
    if not prompt_path.exists():
        return False, f"Prompt not found: {prompt_path}"

    context_path = exp_dir / "CONTEXT.md"
    prompt = (
        f"{prompt_path.read_text(encoding='utf-8')}\n\n"
        f"---\n\n"
        f"Read your task context from: `{context_path}`\n\n"
        f"Start by reading that file, then follow the instructions above.\n"
    )

    dirs = allowed_dirs if allowed_dirs is not None else [exp_dir]
    add_dir_args: List[str] = []
    for d in dirs:
        add_dir_args += ["--add-dir", str(d)]

    cmd = [
        "claude",
        "--print",
        "--dangerously-skip-permissions",
        *add_dir_args,
        "--model", "claude-sonnet-4-6",
        prompt,
    ]

    print(f"  [cc] Spawning agent {agent_key} ...", flush=True)
    try:
        result = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=timeout_secs,
        )
        output = result.stdout + result.stderr
        if result.returncode != 0:
            print(f"  [cc] Agent {agent_key} exited with code {result.returncode}", flush=True)
            return False, output
        print(f"  [cc] Agent {agent_key} completed.", flush=True)
        return True, output
    except subprocess.TimeoutExpired:
        msg = f"Agent {agent_key} timed out after {timeout_secs}s"
        print(f"  [cc] {msg}", flush=True)
        return False, msg
    except Exception as e:
        return False, str(e)


# ─────────────────────────────────────────────
# Programmatic: collect
# ─────────────────────────────────────────────

def run_collect_programmatic(
    exp_dir: Path,
    mode: str,
    n_participants: int,
    project_id: Optional[str] = None,
    ground_truth_model: Optional[str] = None,
) -> Path:
    """
    Run data collection directly (no CC agent).
    If ground_truth_model is set, samples all participants from that model
    (loaded from projects/<project_id>/ground_truth_models.py).
    Otherwise samples from the theorist's models.
    Writes exp_dir/data/responses.csv. Returns path to CSV.
    """
    sys.path.insert(0, str(REPO_ROOT))
    from src.agents.collect import _generate_from_models  # type: ignore
    from src.models.loader import get_model_names_from_manifest  # type: ignore

    stimuli_path = exp_dir / "design" / "stimuli.json"
    manifest_path = exp_dir / "cognitive_models" / "models_manifest.yaml"
    theorist_dir = exp_dir / "cognitive_models"

    stimuli: List[Dict[str, Any]] = []
    if stimuli_path.exists():
        stimuli = json.loads(stimuli_path.read_text(encoding="utf-8"))

    data_dir = exp_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    if ground_truth_model and project_id:
        model_registry = get_ground_truth_models(project_id)
        if ground_truth_model not in model_registry:
            print(f"  [collect] Warning: ground truth model {ground_truth_model!r} not found in registry", flush=True)
        print(f"  [collect] Using ground truth model: {ground_truth_model}", flush=True)
        rows = _generate_from_models(
            stimuli, [ground_truth_model], n_participants,
            model_registry=model_registry,
        )
    else:
        model_names: List[str] = []
        if manifest_path.exists():
            manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
            model_names = get_model_names_from_manifest(manifest, theorist_dir)
        if not model_names:
            print(f"  [collect] Warning: no loadable models in {theorist_dir} — responses will be random", flush=True)
        rows = _generate_from_models(stimuli, model_names, n_participants, theorist_dir=theorist_dir)

    csv_path = data_dir / "responses.csv"
    if rows:
        fieldnames = list(rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
    else:
        csv_path.write_text(
            "participant_id,trial_index,sequence_a,sequence_b,chose_left,chose_right,model\n",
            encoding="utf-8",
        )

    print(f"  [collect] Wrote {len(rows)} rows to {csv_path}", flush=True)
    return csv_path



# ─────────────────────────────────────────────
# Validation (adapted for new directory structure)
# ─────────────────────────────────────────────

def validate_cc_output(agent_key: str, exp_dir: Path) -> tuple[bool, str]:
    """Validate agent output. Returns (ok, message)."""
    validators = {
        "1_theory": _validate_theory,
        "2_design": _validate_design,
        "3_implement": _validate_implement,
        "4_collect": _validate_collect,
        "5_critique": _validate_critique,
    }
    fn = validators.get(agent_key)
    return fn(exp_dir) if fn else (True, "No validator for this agent")


def _validate_theory(exp_dir: Path) -> tuple[bool, str]:
    sys.path.insert(0, str(REPO_ROOT))
    from src.models.loader import get_model_callable, get_model_names_from_manifest  # type: ignore

    theorist_dir = exp_dir / "cognitive_models"
    manifest_path = theorist_dir / "models_manifest.yaml"
    if not manifest_path.exists():
        return False, f"models_manifest.yaml not found at {manifest_path}"
    try:
        data = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    except Exception as e:
        return False, f"Invalid YAML in models_manifest.yaml: {e}"
    if not isinstance(data, dict):
        return False, "models_manifest.yaml is not a dict"
    models = data.get("models") or []
    names = [m["name"] if isinstance(m, dict) else m for m in models]
    if not names:
        return False, "models_manifest.yaml has no models"
    loadable = get_model_names_from_manifest(data, theorist_dir)
    for name in names:
        if name not in loadable:
            return False, f"Model '{name}' has no {theorist_dir}/{name}.py (theorist must provide each model file)"
    # Test call each model
    test_stimulus = ("HHTHTTHT", "HTHTHTHT")
    response_options = ["left", "right"]
    for name in names:
        try:
            fn = get_model_callable(name, theorist_dir)
            preds = fn(test_stimulus, response_options)
        except Exception as e:
            return False, f"Model '{name}' raised: {e}"
        if not isinstance(preds, dict):
            return False, f"Model '{name}' did not return a dict"
        for k in response_options:
            if k not in preds:
                return False, f"Model '{name}' missing key '{k}'"
        total = sum(preds[k] for k in response_options)
        if abs(total - 1.0) > 1e-5:
            return False, f"Model '{name}' probabilities sum to {total}"
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


def _validate_implement(exp_dir: Path) -> tuple[bool, str]:
    index_path = exp_dir / "experiment" / "index.html"
    config_path = exp_dir / "experiment" / "config.json"
    if not index_path.exists():
        return False, f"index.html not found at {index_path}"
    text = index_path.read_text(encoding="utf-8")
    if "jsPsych" not in text and "jspsych" not in text.lower():
        return False, "index.html does not mention jsPsych"
    if not config_path.exists():
        return False, f"config.json not found at {config_path}"
    try:
        json.loads(config_path.read_text(encoding="utf-8"))
    except Exception as e:
        return False, f"Invalid config.json: {e}"
    return True, "Implement valid"


def _validate_collect(exp_dir: Path) -> tuple[bool, str]:
    path = exp_dir / "data" / "responses.csv"
    if not path.exists():
        return False, f"responses.csv not found at {path}"
    lines = path.read_text(encoding="utf-8").strip().split("\n")
    if len(lines) < 2:
        return False, "responses.csv has no data rows"
    header = {h.strip() for h in lines[0].split(",")}
    required = {"participant_id", "trial_index", "sequence_a", "sequence_b", "chose_left"}
    missing = required - header
    if missing:
        return False, f"responses.csv missing columns: {missing}"
    return True, f"Collect valid: {len(lines)-1} rows"


def _validate_critique(exp_dir: Path) -> tuple[bool, str]:
    critique_dir = exp_dir / "critique"
    report = critique_dir / "report.md"
    probs = critique_dir / "theory_probabilities.yaml"
    ppc = critique_dir / "ppc_results.json"
    posterior = critique_dir / "model_posterior.json"
    stats_dir = critique_dir / "test_stats"

    if not posterior.exists():
        return False, "critique/model_posterior.json not found"
    try:
        data = json.loads(posterior.read_text(encoding="utf-8"))
        if "posteriors" not in data or "log_likelihoods" not in data:
            return False, "critique/model_posterior.json missing 'posteriors' or 'log_likelihoods'"
    except Exception as e:
        return False, f"Invalid model_posterior.json: {e}"
    if not report.exists():
        return False, "critique/report.md not found"
    if not report.read_text(encoding="utf-8").strip():
        return False, "critique/report.md is empty"
    if not probs.exists():
        return False, "critique/theory_probabilities.yaml not found"
    try:
        yaml.safe_load(probs.read_text(encoding="utf-8"))
    except Exception as e:
        return False, f"Invalid theory_probabilities.yaml: {e}"
    if not ppc.exists():
        return False, "critique/ppc_results.json not found"
    n_stats = len(list(stats_dir.glob("*.py"))) if stats_dir.exists() else 0
    return True, f"Critique valid ({n_stats} test stats, {len(report.read_text())} char report)"


# ─────────────────────────────────────────────
# Registry helpers
# ─────────────────────────────────────────────

def init_registry(exp_dir: Path) -> None:
    """Write a fresh model_registry.yaml for this experiment."""
    sys.path.insert(0, str(REPO_ROOT))
    from src.registry import write_registry  # type: ignore

    registry_path = exp_dir / "model_registry.yaml"
    if not registry_path.exists():
        write_registry(registry_path, {})


def update_registry_from_interpretation(exp_dir: Path) -> None:
    """Update model_registry.yaml from critique/theory_probabilities.yaml."""
    sys.path.insert(0, str(REPO_ROOT))
    from src.registry import write_registry  # type: ignore

    prob_path = exp_dir / "critique" / "theory_probabilities.yaml"
    registry_path = exp_dir / "model_registry.yaml"

    if not prob_path.exists():
        return
    try:
        data = yaml.safe_load(prob_path.read_text(encoding="utf-8")) or {}
    except Exception:
        return

    theories = data.get("theories") or data.get("probabilities") or {}
    if not isinstance(theories, dict):
        return
    write_registry(registry_path, theories, reserved_for_new=0.0)
    print(f"  [registry] Updated model_registry.yaml from theory_probabilities.yaml", flush=True)
