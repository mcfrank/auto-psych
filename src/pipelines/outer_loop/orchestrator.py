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
import importlib.util
import json
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import yaml

from src.runtime.coding_agent import run_coding_agent

REPO_ROOT = Path(__file__).resolve().parents[3]
PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"

# ─────────────────────────────────────────────
# Directory helpers
# ─────────────────────────────────────────────


def outer_projects_dir() -> Path:
    """Project *assets* (problem_definition.md, ground_truth_models.py, preprocess.py)."""
    return REPO_ROOT / "src" / "pipelines" / "outer_loop" / "projects"


def outer_project_dir(project_id: str) -> Path:
    return outer_projects_dir() / project_id


def outer_data_dir() -> Path:
    """Generated experiment *outputs* (one subtree per project)."""
    return REPO_ROOT / "data" / "outer_loop"


def experiment_dir(project_id: str, exp_num: int) -> Path:
    return outer_data_dir() / project_id / f"experiment{exp_num}"


def project_seed_models_dir(project_id: str) -> Path:
    """Return the optional project seed-model directory."""
    return outer_project_dir(project_id) / "seed_models"


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


def seed_experiment_models_from_project(exp_dir: Path, project_id: str) -> bool:
    """Copy project-level seed models into an empty experiment model directory.

    Projects can define ``seed_models/<name>.py`` plus ``models_manifest.yaml`` to
    specify the model set experiment 1 should start from. The copy is skipped if
    the experiment already has a manifest, which keeps ``--resume`` from
    overwriting models a user or previous agent created.
    """
    seed_dir = project_seed_models_dir(project_id)
    seed_manifest = seed_dir / "models_manifest.yaml"
    if not seed_manifest.exists():
        return False

    dest_dir = exp_dir / "cognitive_models"
    dest_manifest = dest_dir / "models_manifest.yaml"
    if dest_manifest.exists():
        return False

    manifest = yaml.safe_load(seed_manifest.read_text(encoding="utf-8")) or {}
    entries = manifest.get("models") or []
    if not entries:
        raise ValueError(f"Seed manifest has no models: {seed_manifest}")

    dest_dir.mkdir(parents=True, exist_ok=True)
    for entry in entries:
        name = entry.get("name") if isinstance(entry, dict) else entry
        if not name:
            continue
        src = seed_dir / f"{name}.py"
        if not src.exists():
            raise FileNotFoundError(f"Seed model {name!r} has no file at {src}")
        shutil.copyfile(src, dest_dir / f"{name}.py")
    shutil.copyfile(seed_manifest, dest_manifest)
    return True


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


def _collect_llm_participant_programmatic(
    stimuli: List[Dict[str, Any]],
    n_participants: int,
    project_id: Optional[str],
    data_dir: Path,
    *,
    participant_backend: str,
    participant_model: Optional[str],
) -> List[Dict[str, Any]]:
    """LLM-as-participant collection for the active (programmatic) outer loop.

    Resolves the participant prompt and the participant-model backend, then runs
    the shared generation loop. Failures (no stimuli, missing prompt, model init
    error) degrade to an empty result with a logged reason rather than raising,
    so one bad run does not abort the experiment.
    """
    from src.pipelines.outer_loop.collect import generate_llm_participant_rows
    from src.pipelines.outer_loop.llm import load_prompt_for_run
    from src.pipelines.outer_loop.participants import get_participant_model

    if not stimuli:
        print(
            "  [collect] no stimuli (design/stimuli.json missing or empty); nothing to collect",
            flush=True,
        )
        return []

    prompt_text = load_prompt_for_run(
        project_id or "", 1, "4_collect_participant", None
    )
    if not prompt_text.strip():
        print(
            "  [collect] no 4_collect_participant.md prompt found; cannot run no-browser mode",
            flush=True,
        )
        return []

    try:
        model = get_participant_model(participant_backend, participant_model)
    except Exception as exc:
        print(
            f"  [collect] failed to init participant model ({participant_backend}, {participant_model}): {exc}",
            flush=True,
        )
        return []

    print(
        f"  [collect] LLM participants via {model.name}: {n_participants} participant(s) x {len(stimuli)} stimuli",
        flush=True,
    )
    rows, stats = generate_llm_participant_rows(
        stimuli,
        n_participants,
        participant_model=model,
        prompt_text=prompt_text,
        transcripts_dir=data_dir / "transcripts",
    )
    print(
        f"  [collect] {stats['n_rows']} rows (unparseable={stats['n_unparseable']}, errors={stats['n_errors']})",
        flush=True,
    )
    return rows


def run_collect_programmatic(
    exp_dir: Path,
    mode: str,
    n_participants: int,
    project_id: Optional[str] = None,
    ground_truth_model: Optional[str] = None,
    participant_backend: str = "closed",
    participant_model: Optional[str] = None,
    prolific_mode: str = "none",
) -> Path:
    """
    Run data collection directly (no CC agent).

    Collection source, in priority order:
      - mode == "simulated_participants_nobrowser": LLM-as-participant. Each
        synthetic participant answers every stimulus via a participant model
        (``participant_backend`` "closed"=hosted API, "open"=Hugging Face;
        ``participant_model`` names the model). No browser, no Firebase.
      - ground_truth_model set: sample all participants from that project
        ground-truth callable (no browser).
      - otherwise: sample from the theorist's PyMC models' prior-predictive.

    Writes exp_dir/data/responses.csv. Returns path to CSV.
    """
    sys.path.insert(0, str(REPO_ROOT))
    from src.pipelines.outer_loop.collect import (
        _collect_from_firebase,
        _collect_live,
        _generate_from_models,
        _generate_from_pymc_models,
    )
    from src.models.theorist.loader import get_model_names_from_manifest  # type: ignore

    stimuli_path = exp_dir / "design" / "stimuli.json"
    manifest_path = exp_dir / "cognitive_models" / "models_manifest.yaml"
    theorist_dir = exp_dir / "cognitive_models"

    stimuli: List[Dict[str, Any]] = []
    if stimuli_path.exists():
        stimuli = json.loads(stimuli_path.read_text(encoding="utf-8"))

    data_dir = exp_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = data_dir / "logs"
    logs_dir.mkdir(exist_ok=True)
    config_path = exp_dir / "experiment" / "config.json"
    config: Dict[str, Any] = {}
    if config_path.exists():
        try:
            loaded = json.loads(config_path.read_text(encoding="utf-8"))
            config = loaded if isinstance(loaded, dict) else {}
        except Exception:
            config = {}

    run_match = re.search(r"experiment(\d+)$", exp_dir.name)
    run_id = int(run_match.group(1)) if run_match else 1
    state = {
        "project_id": project_id or exp_dir.parent.name,
        "run_id": run_id,
        "mode": mode,
        "deployment_config_path": str(config_path),
        "stimuli_path": str(stimuli_path),
        "theorist_manifest_path": str(manifest_path),
    }

    if mode == "simulated_participants_nobrowser":
        rows = _collect_llm_participant_programmatic(
            stimuli,
            n_participants,
            project_id,
            data_dir,
            participant_backend=participant_backend,
            participant_model=participant_model,
        )
    else:
        rows = None

    has_results_api = bool(config.get("results_api_url") or config.get("experiment_url"))
    if rows is None and not ground_truth_model and has_results_api:
        if prolific_mode != "none" or mode == "live" or config.get("prolific_study_id"):
            rows = _collect_live(state, config, data_dir, logs_dir)
        elif config.get("results_api_url"):
            rows = _collect_from_firebase(
                state,
                config,
                str(config["results_api_url"]),
                int(config.get("simulated_n_participants") or n_participants),
                data_dir,
                logs_dir,
            )

    if rows is None and ground_truth_model and project_id:
        # Ground-truth models are simple callables (data-generation tool used to
        # verify the loop recovers a known process); keep the callable path.
        model_registry = get_ground_truth_models(project_id)
        if ground_truth_model not in model_registry:
            print(
                f"  [collect] Warning: ground truth model {ground_truth_model!r} not found in registry",
                flush=True,
            )
        print(f"  [collect] Using ground truth model: {ground_truth_model}", flush=True)
        rows = _generate_from_models(
            stimuli,
            [ground_truth_model],
            n_participants,
            model_registry=model_registry,
        )
    elif rows is None:
        # Theorist models are PyMC models: sample synthetic responses from their
        # prior-predictive p_left, featurizing each stimulus first.
        model_names: List[str] = []
        if manifest_path.exists():
            manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
            model_names = get_model_names_from_manifest(manifest, theorist_dir)
        if not model_names:
            print(
                f"  [collect] Warning: no loadable models in {theorist_dir} — cannot generate data",
                flush=True,
            )
            rows = []
        else:
            # The featurizer is a project *asset* (src assets dir), not under the
            # data tree where exp_dir now lives.
            assets_dir = outer_project_dir(project_id or exp_dir.parent.name)
            featurize_path = assets_dir / "preprocess.py"
            rows = _generate_from_pymc_models(
                stimuli,
                model_names,
                n_participants,
                models_dir=theorist_dir,
                featurize_path=featurize_path if featurize_path.exists() else None,
            )

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


def run_deployment_programmatic(
    exp_dir: Path,
    project_id: str,
    run_id: int,
    deploy_target: str,
    prolific_mode: str,
    n_participants: int,
    collection_owner: str,
    firebase_project: Optional[str],
    firebase_region: str,
    backend: Optional[str],
) -> Path:
    """Run the deployment phase between implement and collect."""
    from src.pipelines.outer_loop.deployment import run_deployment

    manifest_path = run_deployment(
        exp_dir=exp_dir,
        project_id=project_id,
        run_id=run_id,
        deploy_target=deploy_target,
        prolific_mode=prolific_mode,
        agent_backend=backend or "unknown",
        collection_owner=collection_owner,
        firebase_project=firebase_project,
        firebase_region=firebase_region,
        n_participants=n_participants,
        repo_root=REPO_ROOT,
    )
    print(f"  [deploy] Wrote deployment manifest: {manifest_path}", flush=True)
    return manifest_path


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


def _load_project_featurizer(
    project_dir: Path,
) -> Optional[Callable[[str, str], Dict[str, Any]]]:
    """Return `featurize_stimulus` from `<project_dir>/preprocess.py` if present.

    A project supplies this to turn raw stimulus fields (e.g. H/T sequences)
    into the numeric feature columns its PyMC models read via `pm.Data`. Returns
    None if the project has no preprocess module — then responses are assumed to
    already carry the feature columns.
    """
    path = project_dir / "preprocess.py"
    if not path.exists():
        return None
    spec = importlib.util.spec_from_file_location(
        f"_preprocess_{project_dir.name}", path
    )
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, "featurize_stimulus", None)


def _write_feature_csv(
    rows: List[Dict[str, Any]],
    featurize: Optional[Callable[[str, str], Dict[str, Any]]],
    out_path: Path,
) -> Path:
    """Write pooled responses to `out_path`, merging in derived feature columns.

    If `featurize` is given and a row has `sequence_a`/`sequence_b`, its numeric
    features are added; otherwise the row is written as-is (already featurized).
    """
    out_rows: List[Dict[str, Any]] = []
    for r in rows:
        row = dict(r)
        if featurize is not None and "sequence_a" in r and "sequence_b" in r:
            row.update(featurize(r["sequence_a"], r["sequence_b"]))
        out_rows.append(row)
    if not out_rows:
        raise ValueError("No rows to write to feature CSV")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(out_rows[0].keys())
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)
    return out_path


def _export_inner_loop_model(
    exp_dir: Path, loop_dir: Path, model_name: str = "inner_loop_model"
) -> Path:
    """Copy the inner loop's best PyMC model into `cognitive_models/` + manifest.

    The exported file is the winning PyMC model verbatim (a module-level
    `model: pm.Model`), so the next experiment's theorist and the comparison
    machinery consume it under the same contract as any other model.
    """
    best_model = loop_dir / "best_model.py"
    if not best_model.exists():
        raise FileNotFoundError(f"Inner loop did not produce {best_model}")
    out_dir = exp_dir / "cognitive_models"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / f"{model_name}.py"
    shutil.copyfile(best_model, model_path)

    manifest_path = out_dir / "models_manifest.yaml"
    manifest = {"models": []}
    if manifest_path.exists():
        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or manifest
    models = manifest.setdefault("models", [])
    models = [
        m for m in models if not (isinstance(m, dict) and m.get("name") == model_name)
    ]
    models.append(
        {
            "name": model_name,
            "rationale": "Best PyMC model found by the inner model-improvement loop.",
        }
    )
    manifest["models"] = models
    manifest_path.write_text(
        yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
    )
    return model_path


def run_inner_model_loop_programmatic(
    exp_dir: Path,
    *,
    max_iterations: int,
    candidate_count: int,
    fit_kwargs: Optional[Dict[str, Any]] = None,
    backend: Optional[str] = None,
) -> Path:
    """Run the PyMC inner model loop over pooled outer-loop data.

    Pools responses across experiments, featurizes them (via the project's
    `preprocess.py` if present), seeds the model set from this experiment's
    `cognitive_models/` (the theorist's PyMC models), fits and compares them by
    ELPD-LOO, and exports the best model back into `cognitive_models/`.
    """
    from src.pipelines.inner_loop.pymc_orchestrator import run_pymc_inner_loop

    rows = _pooled_response_rows(exp_dir)
    if not rows:
        raise ValueError(
            f"No response rows found for inner loop under {exp_dir.parent}"
        )

    loop_dir = exp_dir / "model_loop"
    loop_dir.mkdir(parents=True, exist_ok=True)
    # The featurizer is a project *asset* (src assets dir), not under the data
    # tree where exp_dir now lives. exp_dir.parent.name is the project id.
    featurize = _load_project_featurizer(outer_project_dir(exp_dir.parent.name))
    responses_path = _write_feature_csv(rows, featurize, loop_dir / "responses.csv")

    seed_models_dir = exp_dir / "cognitive_models"
    run_pymc_inner_loop(
        responses_path,
        loop_dir,
        seed_models_dir=seed_models_dir,
        max_iterations=max_iterations,
        candidate_count=candidate_count,
        backend=backend,
        fit_kwargs=fit_kwargs,
    )
    model_path = _export_inner_loop_model(exp_dir, loop_dir)
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
    """Validate that every manifest model is a loadable PyMC model.

    Each `<name>.py` must define a module-level `model: pm.Model` with exactly
    one observed-response container (so the inner loop can fit it). Validation
    only builds the model graph — it never samples.
    """
    sys.path.insert(0, str(REPO_ROOT))
    from src.models.pymc_inference import load_pymc_model, observed_response_data  # type: ignore

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

    for name in names:
        if not (theorist_dir / f"{name}.py").exists():
            return (
                False,
                f"Model '{name}' has no {theorist_dir}/{name}.py (theorist must provide each model file)",
            )
        try:
            model = load_pymc_model(name, theorist_dir)
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
