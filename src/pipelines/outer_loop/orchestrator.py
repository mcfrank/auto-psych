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
import math
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

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
    """Generated experiment *outputs* (one subtree per project).

    Override with ``AUTO_PSYCH_OUTPUT_DIR`` so parallel runs (e.g. separate
    cluster jobs sharing one checkout) each write to their own output tree and
    don't collide on ``experiment{N}/`` dirs or pool each other's responses.
    """
    override = os.environ.get("AUTO_PSYCH_OUTPUT_DIR")
    return Path(override) if override else REPO_ROOT / "data" / "outer_loop"


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


def seed_experiment_models_from_project(
    exp_dir: Path, project_id: str, *, exclude: Sequence[str] = ()
) -> bool:
    """Copy project-level seed models into an empty experiment model directory.

    Projects can define ``seed_models/<name>.py`` plus ``models_manifest.yaml`` to
    specify the model set experiment 1 should start from. The copy is skipped if
    the experiment already has a manifest, which keeps ``--resume`` from
    overwriting models a user or previous agent created.

    ``exclude`` withholds the named seed models (e.g. one held out as a
    ground-truth generator). Unknown names or an exclusion that empties the
    seed set raise rather than silently seeding the wrong model set.
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

    names = [e.get("name") if isinstance(e, dict) else e for e in entries]
    unknown = set(exclude) - set(names)
    if unknown:
        raise ValueError(
            f"exclude names models not in the seed manifest: {sorted(unknown)} "
            f"(available: {sorted(n for n in names if n)})"
        )
    kept = [e for e, name in zip(entries, names) if name not in set(exclude)]
    if not kept:
        raise ValueError(
            f"Excluding {sorted(exclude)} empties the seed set from {seed_manifest}"
        )

    dest_dir.mkdir(parents=True, exist_ok=True)
    for entry in kept:
        name = entry.get("name") if isinstance(entry, dict) else entry
        if not name:
            continue
        src = seed_dir / f"{name}.py"
        if not src.exists():
            raise FileNotFoundError(f"Seed model {name!r} has no file at {src}")
        shutil.copyfile(src, dest_dir / f"{name}.py")
    if exclude:
        dest_manifest.write_text(
            yaml.safe_dump({"models": kept}, sort_keys=False), encoding="utf-8"
        )
    else:
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
    prompt_key: Optional[str] = None,
    repair_feedback: Optional[str] = None,
) -> tuple[bool, str]:
    """
    Spawn a coding agent (Claude Code or opencode) for the given agent_key.
    Reads prompt from src/pipelines/outer_loop/prompts/<agent_key>.md, or
    <prompt_key>.md when prompt_key is given (lets a caller swap the prompt
    while keeping the stage identity — logs/validation still use agent_key).
    Tells the agent to read CONTEXT.md and complete the task.
    File tool access is restricted to allowed_dirs (defaults to exp_dir only).
    Bash still runs from REPO_ROOT so python3 -m src.* imports work.
    Streams output to exp_dir/logs/<agent_key>.jsonl and prints live summaries.
    `backend` selects the agent CLI; None resolves via CODING_AGENT/default.
    Returns (success, final_result_text).
    """
    prompt_path = PROMPTS_DIR / f"{prompt_key or agent_key}.md"
    if not prompt_path.exists():
        return False, f"Prompt not found: {prompt_path}"

    context_path = exp_dir / "CONTEXT.md"
    prompt = (
        f"{prompt_path.read_text(encoding='utf-8')}\n\n"
        f"---\n\n"
        f"Read your task context from: `{context_path}`\n\n"
        f"Start by reading that file, then follow the instructions above.\n"
    )
    # On a repair pass, the agent's previous output is already on disk; feed it the
    # exact validation error and ask it to fix that in place rather than restart.
    if repair_feedback:
        prompt += (
            "\n---\n\n"
            "IMPORTANT — this is a REPAIR pass. Your previous attempt is already "
            "written in the task directory, but it FAILED automated validation with:\n\n"
            f"    {repair_feedback}\n\n"
            "Fix ONLY what is needed to resolve this specific error, then stop. Do "
            "not start over or change anything unrelated.\n"
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
        progress=lambda pid, n_rows: print(
            f"  [collect] participant {pid + 1}/{n_participants} done ({n_rows} responses)",
            flush=True,
        ),
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
        check_response_variation,
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
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Malformed experiment config at {config_path}: {exc}") from exc
        if not isinstance(loaded, dict):
            raise RuntimeError(f"Experiment config at {config_path} must be a JSON object")
        config = loaded

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

    # Track whether rows came from actual participants (browser / Firebase /
    # live / LLM-as-participant) vs. synthetic model sampling, so the
    # degenerate-data quality guard only fires on collected behavior.
    collected_from_participants = False
    if mode == "simulated_participants_nobrowser":
        rows = _collect_llm_participant_programmatic(
            stimuli,
            n_participants,
            project_id,
            data_dir,
            participant_backend=participant_backend,
            participant_model=participant_model,
        )
        collected_from_participants = True
    else:
        rows = None

    has_results_api = bool(config.get("results_api_url") or config.get("experiment_url"))
    if mode == "live" and not has_results_api:
        raise RuntimeError(
            "mode='live' requires a deployed experiment to collect from, but the "
            f"experiment config ({config_path}) has no results_api_url/experiment_url. "
            "Run a live pilot with: --deploy-target firebase --prolific-mode live "
            "(this deploys the experiment and creates the Prolific study). Refusing "
            "to silently fall back to synthetic data."
        )
    if rows is None and not ground_truth_model and has_results_api:
        if prolific_mode != "none" or mode == "live" or config.get("prolific_study_id"):
            rows = _collect_live(state, config, data_dir, logs_dir)
            collected_from_participants = True
        elif config.get("results_api_url"):
            rows = _collect_from_firebase(
                state,
                config,
                str(config["results_api_url"]),
                int(config.get("simulated_n_participants") or n_participants),
                data_dir,
                logs_dir,
            )
            collected_from_participants = True

    if rows is None and ground_truth_model and project_id:
        # Ground-truth models are simple callables (data-generation tool used to
        # verify the loop recovers a known process); keep the callable path.
        model_registry = get_ground_truth_models(project_id)
        if ground_truth_model not in model_registry:
            # The ground-truth registry is a known, enumerable set. A name that
            # isn't in it is a typo/config error — NOT a cue to silently fall back
            # to coin-flip data (which would flow into modeling as if it were the
            # named generative process). Fail loudly with the valid options.
            raise ValueError(
                f"--ground-truth-model {ground_truth_model!r} is not in the "
                f"{project_id} ground-truth registry. Available: "
                f"{sorted(model_registry)}."
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

    # Fail loudly on degenerate collected data: if real participants produced no
    # response variation (every trial chose the same side), the data carries no
    # signal for model comparison and almost always signals a broken collector.
    if collected_from_participants and not rows:
        raise RuntimeError(
            "Participant collection returned no data (0 rows). The deployed "
            f"experiment or results fetch failed; inspect {data_dir / 'logs'}. "
            "Refusing to write an empty responses.csv and proceed to modeling."
        )
    if collected_from_participants and rows:
        ok, qc_msg = check_response_variation(rows)
        if not ok:
            raise RuntimeError(
                f"Collected data failed the quality check: {qc_msg}. "
                f"Inspect {data_dir / 'logs'} and the deployed experiment; the "
                "data was NOT written for modeling."
            )

    csv_path = data_dir / "responses.csv"
    if rows:
        # Use the UNION of keys across all rows (not just rows[0]), preserving
        # first-seen order. Live/Firebase rows can be heterogeneous (a row missing
        # or carrying an extra column), and DictWriter raises ValueError on an
        # unexpected key; restval="" fills columns a row lacks.
        fieldnames: List[str] = []
        seen: set = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    fieldnames.append(key)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, restval="")
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
    run_label: Optional[str] = None,
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
        run_label=run_label,
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
    cache_dir: Optional[Path] = None,
    project_id: Optional[str] = None,
    agent_timeout_sec: int = 900,
    complexity_prior_const: Optional[float] = None,
    enable_critique: bool = True,
    n_critique_proposals: Optional[int] = None,
    critique_alpha: Optional[float] = None,
) -> Path:
    """Run the PyMC inner model loop over pooled outer-loop data.

    Pools responses across experiments, featurizes them (via the project's
    `preprocess.py` if present), seeds the model set from this experiment's
    `cognitive_models/` (the theorist's PyMC models), fits and compares them by
    ELPD-LOO, and exports the best model back into `cognitive_models/`.

    `project_id` locates the project assets; it defaults to `exp_dir.parent.name`
    (the standard `data/outer_loop/<project>/experimentN` layout) and must be
    passed explicitly when experiments live elsewhere. `cache_dir` shares the
    MCMC fit cache so later analyses can re-load the loop's fits for free.
    `complexity_prior_const` overrides the inner loop's default Occam line-count
    prior (leave None to use it; pass 0.0 to disable the penalty).
    `enable_critique` runs a CriticAL posterior-predictive critique of the
    incumbent before each candidate round (the critique feeds the candidate
    agents); `n_critique_proposals` (None ⇒ inner-loop default) sets how many test
    statistics the critique agent proposes; `critique_alpha` (None ⇒ inner-loop
    default) is the raw p threshold for flagging a discrepancy.
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
    # tree where exp_dir now lives.
    featurize = _load_project_featurizer(
        outer_project_dir(project_id or exp_dir.parent.name)
    )
    responses_path = _write_feature_csv(rows, featurize, loop_dir / "responses.csv")

    seed_models_dir = exp_dir / "cognitive_models"
    # None ⇒ inherit run_pymc_inner_loop's default Occam line-count prior.
    extra = (
        {}
        if complexity_prior_const is None
        else {"complexity_prior_const": complexity_prior_const}
    )
    # None ⇒ inherit run_pymc_inner_loop's default proposal count / critique alpha.
    if n_critique_proposals is not None:
        extra["n_critique_proposals"] = n_critique_proposals
    if critique_alpha is not None:
        extra["critique_significance_alpha"] = critique_alpha
    run_pymc_inner_loop(
        responses_path,
        loop_dir,
        seed_models_dir=seed_models_dir,
        max_iterations=max_iterations,
        candidate_count=candidate_count,
        cache_dir=cache_dir,
        agent_timeout_sec=agent_timeout_sec,
        backend=backend,
        fit_kwargs=fit_kwargs,
        enable_critique=enable_critique,
        **extra,
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
    if not models:
        return False, "models_manifest.yaml has no models"

    # Each entry carries the model name and its rationale — the one-sentence
    # natural-language hypothesis the model implements. A model with no stated
    # hypothesis is rejected: every model must be a specific, testable claim.
    entries = [
        (m.get("name"), (m.get("rationale") or "").strip())
        if isinstance(m, dict)
        else (m, "")
        for m in models
    ]
    names = [name for name, _ in entries]
    if not all(names):
        return False, "models_manifest.yaml has a model with no name"

    # Only the previous experiment's cognitive_models/ carries forward (its theory
    # models + the single exported best `inner_loop_model`). The inner loop's
    # intermediate candidates (`iterN_candidateM`) live only in model_loop/models/
    # and must never be copied into a theory set — reject them so the repair loop
    # makes the agent drop them rather than silently bloating every later experiment.
    zoo = [n for n in names if re.fullmatch(r"iter\d+_candidate\d+", n)]
    if zoo:
        return (
            False,
            f"models_manifest.yaml carries inner-loop zoo candidate(s) {zoo} from the "
            "previous experiment's model_loop/. Carry forward ONLY the previous "
            "experiment's cognitive_models/ (its theory models plus the single best "
            "`inner_loop_model`); never copy candidates from model_loop/models/.",
        )

    for name, rationale in entries:
        if not rationale:
            return (
                False,
                f"Model '{name}' states no hypothesis: every model needs a non-empty "
                "'rationale' in models_manifest.yaml naming the single cognitive "
                "hypothesis it implements",
            )
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
    """Record the inner model loop's posterior over models in model_registry.yaml.

    The inner loop writes ``model_loop/model_posterior.json`` with a ``posteriors``
    map (model_name -> probability). We copy those weights verbatim (renormalized
    to sum to 1) into this experiment's registry ``theories`` so downstream design
    — e.g. the posterior-weighted EIG of a later experiment — sees the *real*
    per-model posterior mass. (Previously this wrote a hard-coded
    ``{"inner_loop_model": 1.0}``, discarding the computed posterior and emitting a
    weight keyed by a model name with no pure-Python family twin, which broke the
    exhaustive posterior-design path.) No-op if the posterior file is missing or
    malformed.
    """
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

    posteriors = data.get("posteriors")
    if not isinstance(posteriors, dict) or not posteriors:
        return
    weights = {
        str(name): float(p)
        for name, p in posteriors.items()
        if isinstance(p, (int, float)) and math.isfinite(float(p)) and float(p) >= 0.0
    }
    total = sum(weights.values())
    if total <= 0:
        return
    weights = {name: p / total for name, p in weights.items()}
    write_registry(registry_path, weights, reserved_for_new=0.0)
    print(
        "  [registry] Recorded inner-loop posterior over models in model_registry.yaml",
        flush=True,
    )
