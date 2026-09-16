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
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

import yaml

from src.pipelines.inner_loop.hypothesis_ledger import LEDGER_FILENAME
from src.models.model_manifest import (
    manifest_path,
    read_loadable_model_names,
    read_manifest_entries,
    read_manifest_names,
)
from src.models.project.ground_truth import get_ground_truth_models
from src.pipelines.outer_loop.featurizer import Featurizer, load_featurizer

# Stage output validators live in orchestrator_validators.py; re-exported here
# so `from ...orchestrator import validate_cc_output / _validate_*` keeps working.
from src.pipelines.outer_loop.orchestrator_validators import (  # noqa: F401
    _ZOO_NAME_RE,
    _validate_collect,
    _validate_design,
    _validate_implement,
    _validate_model_loop,
    _validate_model_set,
    validate_cc_output,
)
from src.runtime.coding_agent import run_coding_agent
from src.runtime.config import PROJECT_ASSETS_DIR, REPO_ROOT

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"

# ─────────────────────────────────────────────
# Directory helpers
# ─────────────────────────────────────────────


def outer_projects_dir() -> Path:
    """Project *assets* (problem_definition.md, ground_truth_models.py, preprocess.py)."""
    return PROJECT_ASSETS_DIR


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
    seed_manifest = manifest_path(seed_dir)
    if not seed_manifest.exists():
        return False

    dest_dir = exp_dir / "cognitive_models"
    dest_manifest = manifest_path(dest_dir)
    if dest_manifest.exists():
        return False

    entries = read_manifest_entries(seed_dir)
    if not entries:
        raise ValueError(f"Seed manifest has no models: {seed_manifest}")

    names = [entry["name"] for entry in entries]
    unknown = set(exclude) - set(names)
    if unknown:
        raise ValueError(
            f"exclude names models not in the seed manifest: {sorted(unknown)} "
            f"(available: {sorted(names)})"
        )
    kept = [entry for entry in entries if entry["name"] not in set(exclude)]
    if not kept:
        raise ValueError(
            f"Excluding {sorted(exclude)} empties the seed set from {seed_manifest}"
        )

    dest_dir.mkdir(parents=True, exist_ok=True)
    for entry in kept:
        name = entry["name"]
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


def carry_forward_cognitive_models(prev_exp_dir: Path, exp_dir: Path) -> bool:
    """Copy the previous experiment's cognitive_models/ into a new experiment.

    This replaces the removed outer-loop theorist agent's one mechanical job:
    experiments >= 2 start from the previous experiment's model set (the live
    set the inner loop exported: the protected seeds plus every surviving zoo
    model, see ``_export_inner_loop_models``) together with its ledger of
    attempted hypotheses (``attempted_hypotheses.jsonl``, when present). New
    hypotheses enter only via the inner loop.

    Mirrors ``seed_experiment_models_from_project``: returns True on copy and
    False when the destination already has a manifest (so ``--resume`` never
    overwrites an existing model set). A missing or empty previous manifest, or
    a manifest entry without its ``.py`` file, raises — a later experiment must
    never start from a silently truncated model set.
    """
    prev_dir = Path(prev_exp_dir) / "cognitive_models"
    prev_manifest = manifest_path(prev_dir)
    if not prev_manifest.exists():
        raise FileNotFoundError(
            f"Cannot carry the model set forward: {prev_manifest} does not exist "
            f"(did experiment '{Path(prev_exp_dir).name}' complete?)"
        )

    dest_dir = Path(exp_dir) / "cognitive_models"
    dest_manifest = manifest_path(dest_dir)
    if dest_manifest.exists():
        return False

    entries = read_manifest_entries(prev_dir)
    if not entries:
        raise ValueError(f"Previous manifest has no models: {prev_manifest}")

    dest_dir.mkdir(parents=True, exist_ok=True)
    for entry in entries:
        name = entry["name"]
        src = prev_dir / f"{name}.py"
        if not src.exists():
            raise FileNotFoundError(
                f"Carried model {name!r} has no file at {src}; the previous "
                f"experiment's model set is incomplete."
            )
        shutil.copyfile(src, dest_dir / f"{name}.py")
    shutil.copyfile(prev_manifest, dest_manifest)
    prev_ledger = prev_dir / LEDGER_FILENAME
    if prev_ledger.exists():
        shutil.copyfile(prev_ledger, dest_dir / LEDGER_FILENAME)
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
    model: Optional[str] = None,
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
    `model` overrides the backend's default agent model (verbatim, per-backend
    format — e.g. opencode wants `provider/model`).
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
        model=model,
        usage_label=f"outer:{agent_key}",
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
    the shared generation loop. The result must contain one response for every
    participant/stimulus pair; partial collections are saved as diagnostics and
    rejected before they can reach model fitting.
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
    stats_path = data_dir / "collection_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    expected_rows = n_participants * len(stimuli)
    if stats["n_rows"] != expected_rows:
        # The rows that WERE collected are paid LLM output. Preserve them for
        # diagnosis (and possible salvage) under a name modeling never reads
        # before refusing the partial dataset.
        rejected_path = data_dir / "responses_rejected.csv"
        if rows:
            with rejected_path.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
        raise RuntimeError(
            "LLM participant collection was incomplete: "
            f"expected {expected_rows} responses but received {stats['n_rows']} "
            f"(unparseable={stats['n_unparseable']}, errors={stats['n_errors']}). "
            f"Diagnostics were written to {stats_path}"
            + (f"; partial rows saved to {rejected_path}" if rows else "")
            + "; refusing to model a partial dataset."
        )
    return rows


def run_design_programmatic(
    exp_dir: Path,
    project_id: str,
    *,
    exp_num: int = 1,
    prev_exp_dir: Optional[Path] = None,
    k: int = 32,
    n_random: int = 0,
    lengths: Sequence[int] = (2, 3, 4, 5, 6, 7, 8),
) -> None:
    """Select the design's stimuli by exhaustive enumeration (no design agent).

    Enumerates every H/T pair over the given lengths, scores it under the
    experiment's ACTUAL PyMC model set (batched per-draw p_left), and greedily
    picks the ``k`` stimuli with maximal joint EIG about model identity,
    writing ``design/stimuli.json``. Experiment 1 scores from the models'
    prior predictive with uniform model weights; experiments >= 2 fit each
    model on the previous experiment's responses and score from its posterior
    predictive, with model weights from the previous registry (weights over
    models absent here fall back to uniform, loudly). Works for any PyMC model
    in the set — no pure-Python family twin needed. Only implemented for
    subjective_randomness (H/T pair enumeration).
    """
    if project_id != "subjective_randomness":
        raise ValueError(
            "Exhaustive design is only implemented for subjective_randomness "
            f"(H/T pair enumeration); got {project_id!r}."
        )
    from src.pipelines.outer_loop import eig as eig_mod

    models_dir = exp_dir / "cognitive_models"
    featurize = outer_project_dir(project_id) / "preprocess.py"
    if exp_num <= 1 or prev_exp_dir is None:
        stimuli = eig_mod.design_exhaustive(
            models_dir,
            featurize_path=featurize,
            screened_out_path=exp_dir / "design" / "screened_out.json",
            lengths=tuple(lengths),
            n_select=k,
            n_random=n_random,
            random_seed=exp_num,
        )
        basis = "prior predictive + uniform model weights"
    else:
        stimuli = eig_mod.design_exhaustive(
            models_dir,
            prev_exp_dir / "model_registry.yaml",
            featurize_path=featurize,
            screened_out_path=exp_dir / "design" / "screened_out.json",
            lengths=tuple(lengths),
            n_select=k,
            n_random=n_random,
            random_seed=exp_num,
            responses_csv=prev_exp_dir / "data" / "responses.csv",
            fit_cache_dir=exp_dir / "design" / "_fit_cache",
        )
        basis = f"experiment {exp_num - 1} posterior (model weights + parameter posteriors)"

    design_dir = exp_dir / "design"
    design_dir.mkdir(parents=True, exist_ok=True)
    (design_dir / "stimuli.json").write_text(
        json.dumps(stimuli, indent=2), encoding="utf-8"
    )
    print(
        f"  [design] Exhaustive: enumerated all H/T pairs over lengths {tuple(lengths)}, "
        f"selected {len(stimuli)} jointly-informative pairs ({basis}) -> "
        f"{design_dir / 'stimuli.json'}",
        flush=True,
    )


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

    stimuli_path = exp_dir / "design" / "stimuli.json"
    theorist_dir = exp_dir / "cognitive_models"
    theorist_manifest = manifest_path(theorist_dir)

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
        "theorist_manifest_path": str(theorist_manifest),
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
        if theorist_manifest.exists():
            model_names = read_loadable_model_names(theorist_dir)
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


def _load_project_featurizer(project_dir: Path) -> Optional[Featurizer]:
    """Return `featurize_stimulus` from `<project_dir>/preprocess.py` if present.

    A project supplies this to turn raw stimulus fields (e.g. H/T sequences)
    into the numeric feature columns its PyMC models read via `pm.Data`. Returns
    None only if the project has no preprocess module — then responses are
    assumed to already carry the feature columns. A preprocess module that
    exists but cannot be loaded raises (see `featurizer.load_featurizer`).
    """
    path = project_dir / "preprocess.py"
    if not path.exists():
        return None
    return load_featurizer(path)


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


def _protected_seed_names(project_id: str, models_dir: Path) -> set:
    """The project's seed models present in ``models_dir``.

    These are the baselines every run reports against: the inner loop never
    prunes them and the export always carries them. A seed the project lists
    but this run holds out is simply absent. A project without a seed manifest
    cannot say which models are baselines, so that raises.
    """
    seed_dir = project_seed_models_dir(project_id)
    if not manifest_path(seed_dir).exists():
        raise FileNotFoundError(
            f"Project {project_id!r} has no seed manifest at {manifest_path(seed_dir)}; "
            "the inner loop needs it to know which models are protected baselines."
        )
    return set(read_manifest_names(seed_dir)) & set(read_manifest_names(models_dir))


def _export_inner_loop_models(
    exp_dir: Path, loop_dir: Path, *, best_model: str, protected_names: Iterable[str]
) -> Path:
    """Record the inner loop's live set in `cognitive_models/` + manifest.

    After the loop, ``cognitive_models/`` — the set the next experiment starts
    from — is exactly: every protected (project seed) model that was in it, in
    its original order, followed by every zoo survivor that was not already
    there, in zoo order. A carried, non-protected model the loop pruned or
    dropped is removed, file and entry: the carried set is the loop's
    uncertainty set (the seeds plus every model still within the pruning margin
    of the best), not an ever-growing archive. Before this change only the
    single best model was exported, so a rival statistically tied with it was
    left behind and re-proposed from scratch by the next experiment's agents.
    The ledger of attempted hypotheses (``attempted_hypotheses.jsonl``) is
    copied beside the manifest so the next experiment's loop continues it.

    Each exported model keeps its own descriptive name and its hypothesis as
    the manifest rationale; a model already in the set (a seed, or a model
    exported by an earlier experiment) is never duplicated under a second name.
    A fallback auto-named survivor (``iterN_candidateM`` — the agent wrote no
    usable ``model_name.txt``) exports under the legacy stable name
    ``inner_loop_model`` (``inner_loop_model_2``, ... for further ones),
    because zoo names must never enter the carried manifest (the model-set
    validator rejects them); the best model is named first so a zoo-named best
    maps to ``inner_loop_model`` as ``_validate_model_loop`` expects. Returns
    the best model's path in ``cognitive_models/``.
    """
    zoo_dir = loop_dir / "models"
    zoo_entries = read_manifest_entries(zoo_dir)
    rationales = {
        entry["name"]: (entry.get("rationale") or "").strip() for entry in zoo_entries
    }
    if best_model not in rationales or not (zoo_dir / f"{best_model}.py").exists():
        raise ValueError(
            f"Best model {best_model!r} is not in the inner-loop zoo "
            f"({zoo_dir}); cannot export it."
        )
    for name, rationale in rationales.items():
        if not rationale:
            raise ValueError(
                f"Zoo model {name!r} has an empty rationale in the zoo manifest; "
                "every carried model must state its hypothesis."
            )
        if not (zoo_dir / f"{name}.py").exists():
            raise FileNotFoundError(
                f"Zoo model {name!r} is in the manifest but has no file at "
                f"{zoo_dir / f'{name}.py'}; the zoo is incomplete."
            )

    out_dir = exp_dir / "cognitive_models"
    out_dir.mkdir(parents=True, exist_ok=True)
    protected = set(protected_names)
    previous = read_manifest_entries(out_dir, missing_ok=True)
    kept = [
        entry
        for entry in previous
        if entry["name"] in protected or entry["name"] in rationales
    ]
    removed = [entry["name"] for entry in previous if entry["name"] not in
               {kept_entry["name"] for kept_entry in kept}]
    for name in removed:
        (out_dir / f"{name}.py").unlink(missing_ok=True)

    existing = {entry["name"] for entry in kept}
    new_names = [entry["name"] for entry in zoo_entries if entry["name"] not in existing]
    # Assign export names best-first so a zoo-named best takes `inner_loop_model`.
    export_names = {}
    taken = set(existing)
    for name in sorted(new_names, key=lambda n: n != best_model):
        export_name = name
        if _ZOO_NAME_RE.fullmatch(name):
            export_name = "inner_loop_model"
            suffix = 2
            while export_name in taken:
                export_name = f"inner_loop_model_{suffix}"
                suffix += 1
        taken.add(export_name)
        export_names[name] = export_name
    for name in new_names:
        shutil.copyfile(zoo_dir / f"{name}.py", out_dir / f"{export_names[name]}.py")
        kept.append({"name": export_names[name], "rationale": rationales[name]})
    manifest_path(out_dir).write_text(
        yaml.safe_dump({"models": kept}, sort_keys=False), encoding="utf-8"
    )
    ledger = loop_dir / LEDGER_FILENAME
    if ledger.exists():
        shutil.copyfile(ledger, out_dir / LEDGER_FILENAME)

    best_export = export_names.get(best_model, best_model)
    print(
        f"  [inner-loop] Carried the live set into {out_dir}: best {best_export!r}; "
        f"added {[export_names[n] for n in new_names]}; removed {removed}; "
        f"set = {[entry['name'] for entry in kept]}",
        flush=True,
    )
    return out_dir / f"{best_export}.py"


def run_inner_model_loop_programmatic(
    exp_dir: Path,
    *,
    max_iterations: int,
    candidate_count: int,
    fit_kwargs: Optional[Dict[str, Any]] = None,
    backend: Optional[str] = None,
    agent_model: Optional[str] = None,
    cache_dir: Optional[Path] = None,
    project_id: Optional[str] = None,
    agent_timeout_sec: int = 900,
    complexity_prior_const: Optional[float] = None,
    enable_critique: bool = True,
    n_critique_proposals: Optional[int] = None,
    critique_alpha: Optional[float] = None,
    candidate_hints: Optional[List[str]] = None,
    novelty_rmse_threshold: Optional[float] = None,
    prune_dse_multiplier: Optional[float] = None,
    candidate_parallelism: Optional[int] = None,
) -> Path:
    """Run the PyMC inner model loop over pooled outer-loop data.

    Pools responses across experiments, featurizes them (via the project's
    `preprocess.py` if present), seeds the model set from this experiment's
    `cognitive_models/` (the carried set plus its ledger), fits and compares
    them by ELPD-LOO, and exports the surviving live set back into
    `cognitive_models/` (``_export_inner_loop_models``). Only the project's
    seed models are pruning-protected: a model carried from an earlier
    experiment can lose here and leave the set.

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
    protected = _protected_seed_names(project_id or exp_dir.parent.name, seed_models_dir)
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
    # None ⇒ inherit run_pymc_inner_loop's defaults for the exploration knobs.
    if candidate_hints is not None:
        extra["candidate_hints"] = list(candidate_hints)
    if novelty_rmse_threshold is not None:
        extra["novelty_rmse_threshold"] = novelty_rmse_threshold
    if prune_dse_multiplier is not None:
        extra["prune_dse_multiplier"] = prune_dse_multiplier
    if candidate_parallelism is not None:
        extra["candidate_parallelism"] = candidate_parallelism
    result = run_pymc_inner_loop(
        responses_path,
        loop_dir,
        seed_models_dir=seed_models_dir,
        max_iterations=max_iterations,
        candidate_count=candidate_count,
        cache_dir=cache_dir,
        agent_timeout_sec=agent_timeout_sec,
        backend=backend,
        agent_model=agent_model,
        fit_kwargs=fit_kwargs,
        enable_critique=enable_critique,
        protected_names=protected,
        ledger_context=exp_dir.name,
        **extra,
    )
    _export_inner_loop_models(
        exp_dir, loop_dir, best_model=result["best_model"], protected_names=protected
    )
    return loop_dir


# ─────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────
# The stage output validators live in orchestrator_validators.py; they are
# re-exported here (and imported at module top) so `from ...orchestrator import
# _validate_*` keeps working.


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
    """Write the next design's model prior: uniform over the carried model set.

    The registry is the model prior for the next experiment's EIG design, so it
    must (a) cover exactly the models that design will score — the
    ``cognitive_models/`` set the inner loop just exported — and (b) leave EIG
    something to discriminate. Neither held for the stacking weights recorded
    before: ``az.compare``'s weights are ensemble coefficients, not
    plausibility (a model 1.6 nats behind the best read 0.000 because its
    predictions were redundant with the best's; one 95 nats behind read 0.33
    because they differed), they were written over the whole zoo including
    models never carried, and in 15 of 40 next-experiment designs of the
    iteration-2 recovery sweep every model actually present had weight ~0 or a
    single model had weight 1.0 — a degenerate prior under which all 32
    EIG-selected stimuli had zero EIG (filler pairs). The carried set is, by
    construction, the protected seeds plus every model still within the pruning
    margin of the best, so a uniform prior over it asks the design to separate
    exactly the hypotheses the data have not yet resolved. The stacking weights
    stay in ``model_posterior.json``'s ``comparison`` block as a report field.

    This runs only after a model loop completed and exported, so a missing
    posterior export or an absent/empty carried set means the pipeline is
    broken — every such case raises loudly rather than leaving a stale registry
    to steer the next design.
    """
    sys.path.insert(0, str(REPO_ROOT))
    from src.registry import write_registry  # type: ignore

    posterior_path = exp_dir / "model_loop" / "model_posterior.json"
    registry_path = exp_dir / "model_registry.yaml"

    if not posterior_path.exists():
        raise FileNotFoundError(
            f"Cannot update the design registry: {posterior_path} does not exist "
            f"(the inner model loop should have exported it)."
        )
    names = read_manifest_names(exp_dir / "cognitive_models")
    if not names:
        raise ValueError(
            f"Cannot update the design registry: {manifest_path(exp_dir / 'cognitive_models')} "
            "lists no models."
        )
    weights = {name: 1.0 / len(names) for name in names}
    write_registry(registry_path, weights, reserved_for_new=0.0)
    print(
        f"  [registry] Recorded a uniform design prior over the {len(names)} carried "
        "models in model_registry.yaml",
        flush=True,
    )
