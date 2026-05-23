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
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from src.runtime.coding_agent import run_coding_agent

REPO_ROOT = Path(__file__).resolve().parents[3]
PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"

# ─────────────────────────────────────────────
# Directory helpers
# ─────────────────────────────────────────────

def outer_projects_dir() -> Path:
    return REPO_ROOT / "src" / "pipelines" / "outer_loop" / "projects"


def outer_project_dir(project_id: str) -> Path:
    return outer_projects_dir() / project_id


def experiment_dir(project_id: str, exp_num: int) -> Path:
    return outer_projects_dir() / project_id / f"experiment{exp_num}"


def get_ground_truth_models(project_id: str) -> Dict:
    """Load GROUND_TRUTH_MODELS from src/pipelines/outer_loop/projects/<project>/ground_truth_models.py."""
    import importlib.util
    path = outer_project_dir(project_id) / "ground_truth_models.py"
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
    for sub in ["cognitive_models", "design", "experiment", "data", "model_loop"]:
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
) -> Path:
    """Write CONTEXT.md into exp_dir for the given agent. Return path."""
    prob_path = outer_project_dir(project_id) / "problem_definition.md"

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
        f"- Inner model loop dir: `{exp_dir / 'model_loop'}`",
    ]

    if prev_exp_dir and prev_exp_dir.exists():
        lines += ["", "## Previous experiment paths", ""]
        lines += [
            f"- Previous cognitive models: `{prev_exp_dir / 'cognitive_models'}`",
            f"- Previous model registry: `{prev_exp_dir / 'model_registry.yaml'}`",
            f"- Previous model loop report: `{prev_exp_dir / 'model_loop' / 'report.md'}`",
            f"- Previous model posterior: `{prev_exp_dir / 'model_loop' / 'model_posterior.json'}`",
        ]

    if extra:
        lines += ["", "## Additional context", ""]
        for k, v in extra.items():
            lines.append(f"- **{k}**: {v}")

    context_path = exp_dir / "CONTEXT.md"
    context_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return context_path


# ─────────────────────────────────────────────
# Coding-agent spawner
# ─────────────────────────────────────────────

def spawn_cc_agent(
    agent_key: str,
    exp_dir: Path,
    allowed_dirs: Optional[List[Path]] = None,
    timeout_secs: int = 900,
    backend: Optional[str] = None,
) -> tuple[bool, str]:
    """
    Spawn a coding agent (Claude Code or opencode) for the given agent_key.
    Reads prompt from src/pipelines/outer_loop/prompts/<agent_key>.md.
    Tells the agent to read CONTEXT.md and complete the task.
    File tool access is restricted to allowed_dirs (defaults to exp_dir only).
    Bash still runs from REPO_ROOT so python3 -m src.* imports work.
    Streams output to exp_dir/logs/<agent_key>.jsonl and prints live summaries.
    `backend` selects the agent CLI; None resolves via CODING_AGENT/default.
    Returns (success, final_result_text).
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
    log_path = exp_dir / "logs" / f"{agent_key}.jsonl"

    print(f"  [agent] Spawning {agent_key} (log: {log_path})", flush=True)
    success, final_result = run_coding_agent(
        prompt,
        cwd=REPO_ROOT,
        log_path=log_path,
        allowed_dirs=dirs,
        timeout_secs=timeout_secs,
        backend=backend,
    )
    if success:
        print(f"  [agent] {agent_key} completed.", flush=True)
    else:
        print(f"  [agent] {agent_key} finished without success.", flush=True)
    return success, final_result


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
    from src.pipelines.outer_loop.collect import _generate_from_models
    from src.models.theorist.loader import get_model_names_from_manifest  # type: ignore

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
# Programmatic: inner cognitive-model loop
# ─────────────────────────────────────────────

def _pooled_response_rows(exp_dir: Path) -> list[dict]:
    project_dir = exp_dir.parent
    current_num = int(exp_dir.name.removeprefix("experiment"))
    rows: list[dict] = []
    for exp_num in range(1, current_num + 1):
        path = project_dir / f"experiment{exp_num}" / "data" / "responses.csv"
        if path.exists():
            rows.extend(csv.DictReader(path.open(encoding="utf-8")))
    return rows


def _export_inner_loop_model(exp_dir: Path, loop_dir: Path, model_name: str = "inner_loop_model") -> Path:
    from src.pipelines.inner_loop.history import _load_fit_result

    best_model = loop_dir / "best_model.py"
    best_fit = _load_fit_result(loop_dir / "best_fit.json")
    out_dir = exp_dir / "cognitive_models"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / f"{model_name}.py"
    model_path.write_text(
        best_model.read_text(encoding="utf-8")
        + "\n\n"
        + f"FITTED_PARAMS = {best_fit.params!r}\n\n"
        + f"def {model_name}(stimulus, response_options):\n"
        + "    return cognitive_model(stimulus, response_options, FITTED_PARAMS)\n",
        encoding="utf-8",
    )

    manifest_path = out_dir / "models_manifest.yaml"
    manifest = {"models": []}
    if manifest_path.exists():
        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or manifest
    models = manifest.setdefault("models", [])
    models = [m for m in models if not (isinstance(m, dict) and m.get("name") == model_name)]
    models.append(
        {
            "name": model_name,
            "rationale": "Best model found by the inner model-improvement loop.",
        }
    )
    manifest["models"] = models
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    return model_path


def run_inner_model_loop_programmatic(
    exp_dir: Path,
    *,
    max_iterations: int,
    candidate_count: int,
    api_key: str | None = None,
) -> Path:
    """Run the abstract inner cognitive-model loop over pooled outer-loop data."""
    from src.pipelines.inner_loop.adapters import subjective_randomness_dataset
    from src.pipelines.inner_loop.orchestrator import run_pipeline as run_inner_loop

    rows = _pooled_response_rows(exp_dir)
    if not rows:
        raise ValueError(f"No response rows found for inner loop under {exp_dir.parent}")

    loop_dir = exp_dir / "model_loop"
    data = subjective_randomness_dataset(rows, label=f"{exp_dir.parent.name}:{exp_dir.name}")
    fit = run_inner_loop(
        data,
        loop_dir,
        max_iterations=max_iterations,
        candidate_count=candidate_count,
        api_key=api_key,
    )
    model_path = _export_inner_loop_model(exp_dir, loop_dir)

    from src.pipelines.inner_loop.bmc import compute_bmc

    bmc_path = loop_dir / "model_posterior.json"
    if not bmc_path.exists():
        bmc_path.write_text(
            json.dumps(compute_bmc(loop_dir / "model_zoo"), indent=2),
            encoding="utf-8",
        )
    (loop_dir / "report.md").write_text(
        "# Inner Model Loop Report\n\n"
        f"- Trials: {len(rows)}\n"
        f"- Best log likelihood: {fit.log_likelihood:.4f}\n"
        f"- Exported model: `{model_path}`\n"
        f"- Loop artifacts: `{loop_dir}`\n",
        encoding="utf-8",
    )
    print(f"  [inner-loop] Exported {model_path}", flush=True)
    return loop_dir



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
        "5_model_loop": _validate_model_loop,
    }
    fn = validators.get(agent_key)
    return fn(exp_dir) if fn else (True, "No validator for this agent")


def _validate_theory(exp_dir: Path) -> tuple[bool, str]:
    sys.path.insert(0, str(REPO_ROOT))
    from src.models.theorist.loader import get_model_callable, get_model_names_from_manifest  # type: ignore

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


def _validate_model_loop(exp_dir: Path) -> tuple[bool, str]:
    loop_dir = exp_dir / "model_loop"
    report = loop_dir / "report.md"
    posterior = loop_dir / "model_posterior.json"
    exported_model = exp_dir / "cognitive_models" / "inner_loop_model.py"

    if not posterior.exists():
        return False, "model_loop/model_posterior.json not found"
    try:
        data = json.loads(posterior.read_text(encoding="utf-8"))
        if "posteriors" not in data:
            return False, "model_loop/model_posterior.json missing 'posteriors'"
    except Exception as e:
        return False, f"Invalid model_posterior.json: {e}"
    if not report.exists():
        return False, "model_loop/report.md not found"
    if not report.read_text(encoding="utf-8").strip():
        return False, "model_loop/report.md is empty"
    if not exported_model.exists():
        return False, "cognitive_models/inner_loop_model.py not found"
    return True, f"Model loop valid ({len(report.read_text())} char report)"


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
    """Update model_registry.yaml from the inner model loop output."""
    sys.path.insert(0, str(REPO_ROOT))
    from src.registry import write_registry  # type: ignore

    posterior_path = exp_dir / "model_loop" / "model_posterior.json"
    registry_path = exp_dir / "model_registry.yaml"

    if not posterior_path.exists():
        return
    try:
        data = json.loads(posterior_path.read_text(encoding="utf-8"))
    except Exception:
        return

    if not isinstance(data.get("posteriors"), dict):
        return
    write_registry(registry_path, {"inner_loop_model": 1.0}, reserved_for_new=0.0)
    print("  [registry] Updated model_registry.yaml from inner model loop", flush=True)
