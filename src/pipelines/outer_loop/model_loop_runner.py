"""Inner-loop integration and registry helpers for the outer experiment loop.

These functions manage the interface between the inner model loop
(``src.pipelines.inner_loop``) and the outer experiment loop
(``src.pipelines.outer_loop``): pooling responses, protecting seed models,
exporting the live model set, and maintaining the design registry.

Extracted from ``orchestrator.py``; the original module re-exports every public
name here so existing ``from ...orchestrator import X`` keeps working.
"""

from __future__ import annotations

import csv
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml

from src.pipelines.inner_loop.hypothesis_ledger import LEDGER_FILENAME
from src.pipelines.inner_loop.task_description import TASK_DESCRIPTION_NAME
from src.models.model_manifest import (
    manifest_path,
    read_manifest_entries,
    read_manifest_names,
)
from src.pipelines.outer_loop.columns import (
    RAW_RESPONSE_COLUMNS,
    raw_response_rows,
    write_responses_csv,
)
from src.pipelines.outer_loop.orchestrator_validators import (
    EXPORT_RECORD_FILENAME,
    _ZOO_NAME_RE,
    ledger_sha256,
    validate_cc_output,
)
from src.runtime.atomic_files import (
    copy_directory_atomically,
    replace_directory,
    write_text_atomically,
)
from src.runtime.config import REPO_ROOT

# ─────────────────────────────────────────────
# Programmatic: inner cognitive-model loop
# ─────────────────────────────────────────────


def _pooled_response_rows(exp_dir: Path) -> list[dict[str, str]]:
    """Concatenate response rows from experiment 1 through ``exp_dir``, keeping
    only the raw columns (a run collected before collection dropped the others
    still has them in its ``data/responses.csv``)."""
    project_dir = exp_dir.parent
    current_num = int(exp_dir.name.removeprefix("experiment"))
    rows: list[dict] = []
    for exp_num in range(1, current_num + 1):
        path = project_dir / f"experiment{exp_num}" / "data" / "responses.csv"
        if path.exists():
            with path.open(encoding="utf-8") as f:
                rows.extend(raw_response_rows(csv.DictReader(f)))
    return rows


def write_task_description(project_id: str, loop_dir: Path) -> Path:
    """Copy the project's task description next to the inner loop's responses.

    Candidate and critique agents are told the task from it
    (``inner_loop.task_description``). Fails loudly if the project has none.
    """
    from src.pipelines.outer_loop.orchestrator import outer_project_dir

    source = outer_project_dir(project_id) / TASK_DESCRIPTION_NAME
    if not source.exists():
        raise FileNotFoundError(
            f"Project {project_id!r} has no {TASK_DESCRIPTION_NAME} at {source}; "
            "inner-loop agents must be told the task."
        )
    Path(loop_dir).mkdir(parents=True, exist_ok=True)
    return Path(shutil.copyfile(source, Path(loop_dir) / TASK_DESCRIPTION_NAME))


def agent_notes_dir(exp_dir: Path) -> Path:
    """Where a run's Claude agents keep notes for later agents of the same run.

    One directory per run, beside its ``experiment<N>/`` trees (the run is
    ``exp_dir.parent``, as for the pooled responses above), so notes carry
    across the run's rounds and experiments and are given to no other run's
    agents: each agent runs in a bubblewrap sandbox holding only its working
    tree and the directories it is given (``src.runtime.agent_sandbox``), so
    another run's notes are out of reach unless the output tree lies inside
    that working tree (``run.py``'s default ``REPO_ROOT/data/outer_loop``;
    the harness and the live launchers put it elsewhere).
    """
    return exp_dir.parent / "agent_notes"


# The run's starting models, recorded beside its experiment directories when
# experiment 1's model loop first runs (``run_starting_models``).
STARTING_MODELS_FILENAME = "starting_models.json"


def run_starting_models(exp_dir: Path, project_id: str) -> set[str]:
    """The models this run started from: the pruning-protected baselines.

    Recorded once, when experiment 1's model loop first runs, as the project
    seeds in experiment 1's ``cognitive_models/`` (``_protected_seed_names``;
    a seed held out of the run is absent), in
    ``<run>/starting_models.json``; read back by every later experiment and by
    a resumed experiment 1. Protection used to be recomputed every experiment
    from the project's full seed manifest intersected with the carried set,
    so a candidate that chose a seed's name — the held-out ground truth's,
    say — was carried as a protected, never-pruned starting model (second
    audit B10). A later experiment of a run with no record raises: it started
    on code that kept none, and its starting set cannot be told from names.
    """
    record = Path(exp_dir).parent / STARTING_MODELS_FILENAME
    if record.exists():
        names = json.loads(record.read_text(encoding="utf-8"))
        if not isinstance(names, list) or not all(isinstance(n, str) for n in names):
            raise ValueError(f"{record} must hold a JSON list of model names; got {names!r}")
        return set(names)
    if _experiment_number(exp_dir) != 1:
        raise FileNotFoundError(
            f"{record} does not exist. It records the run's starting models when "
            "experiment 1's model loop first runs; this run started on code that "
            "kept no record. Write the names experiment 1 started from (the seed "
            "step of experiment1/model_loop/history.json) there as a JSON list."
        )
    names = _protected_seed_names(project_id, Path(exp_dir) / "cognitive_models")
    write_text_atomically(record, json.dumps(sorted(names), indent=2) + "\n")
    return names


def _protected_seed_names(project_id: str, models_dir: Path) -> set[str]:
    """The project's seed models present in ``models_dir``.

    These are the baselines every run reports against: the inner loop never
    prunes them and the export always carries them. A seed the project lists
    but this run holds out is simply absent. A project without a seed manifest
    cannot say which models are baselines, so that raises.
    """
    # Lazy import to avoid a circular dependency with orchestrator.py, which
    # re-exports this module's names and defines project_seed_models_dir.
    from src.pipelines.outer_loop.orchestrator import project_seed_models_dir

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

    responses_csv = loop_dir / "responses.csv"
    is_raw = False
    if responses_csv.exists():
        with responses_csv.open(encoding="utf-8") as f:
            csv_header = [c.strip() for c in f.readline().strip().split(",")]
        is_raw = csv_header == list(RAW_RESPONSE_COLUMNS)
    if is_raw:
        from src.models.data_binding import MissingStimulusColumns, make_stim_data
        from src.models.model_loading import load_pymc_model

        raw_row = {c: "0" for c in RAW_RESPONSE_COLUMNS}
        raw_row["sequence_a"] = "HHT"
        raw_row["sequence_b"] = "THT"
        for name in rationales:
            try:
                m = load_pymc_model(name, zoo_dir)
                make_stim_data(m, [raw_row])
            except MissingStimulusColumns as exc:
                raise ValueError(
                    f"raw mode: exported model {name!r} cannot bind a raw "
                    f"row — missing columns: {list(exc.missing)}. In a raw "
                    f"run every model must compute its own features."
                ) from exc

    out_dir = exp_dir / "cognitive_models"
    protected = set(protected_names)
    previous = read_manifest_entries(out_dir, missing_ok=True)
    kept = [
        entry
        for entry in previous
        if entry["name"] in protected or entry["name"] in rationales
    ]
    removed = [entry["name"] for entry in previous if entry["name"] not in
               {kept_entry["name"] for kept_entry in kept}]

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
    exported = list(kept) + [
        {"name": export_names[name], "rationale": rationales[name]} for name in new_names
    ]
    ledger = loop_dir / LEDGER_FILENAME

    def build(staging: Path) -> None:
        for entry in kept:
            shutil.copyfile(out_dir / f"{entry['name']}.py", staging / f"{entry['name']}.py")
        for name in new_names:
            shutil.copyfile(zoo_dir / f"{name}.py", staging / f"{export_names[name]}.py")
        manifest_path(staging).write_text(
            yaml.safe_dump({"models": exported}, sort_keys=False), encoding="utf-8"
        )
        if ledger.exists():
            shutil.copyfile(ledger, staging / LEDGER_FILENAME)
        elif (out_dir / LEDGER_FILENAME).exists():
            shutil.copyfile(out_dir / LEDGER_FILENAME, staging / LEDGER_FILENAME)

    # Built beside the set and renamed over it: never a manifest that lists
    # deleted files or misses copied ones.
    replace_directory(out_dir, build)

    best_export = export_names.get(best_model, best_model)
    print(
        f"  [inner-loop] Carried the live set into {out_dir}: best {best_export!r}; "
        f"added {[export_names[n] for n in new_names]}; removed {removed}; "
        f"set = {[entry['name'] for entry in exported]}",
        flush=True,
    )
    return out_dir / f"{best_export}.py"


_EXPERIMENT_DIR_RE = re.compile(r"experiment(\d+)$")


def _experiment_number(exp_dir: Path) -> int:
    """The experiment number encoded in ``exp_dir``'s name (``experiment<k>``)."""
    match = _EXPERIMENT_DIR_RE.fullmatch(Path(exp_dir).name)
    if match is None:
        raise ValueError(
            f"Experiment directory {exp_dir} must be named experiment<k>; "
            f"got {Path(exp_dir).name!r}."
        )
    return int(match.group(1))


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
    agent_root: Optional[Path] = None,
) -> Path:
    """Run the PyMC inner model loop over pooled outer-loop data.

    Pools raw responses across experiments, seeds the model set from this
    experiment's `cognitive_models/` (the carried set plus its ledger), fits and
    compares them by ELPD-LOO, and exports the surviving live set back into
    `cognitive_models/` (``_export_inner_loop_models``). Each model computes its
    own features from raw stimulus rows via its hooks. Only the project's seed
    models are pruning-protected: a model carried from an earlier experiment can
    lose here and leave the set.

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
    from src.pipelines.inner_loop.model_zoo import _lens_offset
    from src.pipelines.inner_loop.pymc_orchestrator import run_pymc_inner_loop

    exp_num = _experiment_number(exp_dir)
    rows = _pooled_response_rows(exp_dir)
    if not rows:
        raise ValueError(
            f"No response rows found for inner loop under {exp_dir.parent}"
        )

    loop_dir = exp_dir / "model_loop"
    loop_dir.mkdir(parents=True, exist_ok=True)
    responses_path = write_responses_csv(rows, loop_dir / "responses.csv")
    write_task_description(project_id or exp_dir.parent.name, loop_dir)

    seed_models_dir = exp_dir / "cognitive_models"
    # Protected, and names no candidate may take: the run's starting models
    # (the loop protects those still in the carried set).
    protected = run_starting_models(exp_dir, project_id or exp_dir.parent.name)
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
    if agent_root is not None:
        extra["agent_root"] = agent_root
    result = run_pymc_inner_loop(
        responses_path,
        loop_dir,
        agent_notes_dir=agent_notes_dir(exp_dir),
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
        lens_offset=_lens_offset(
            exp_num,
            max_iterations=max_iterations,
            candidate_count=candidate_count,
        ),
        **extra,
    )
    _export_inner_loop_models(
        exp_dir, loop_dir, best_model=result["best_model"], protected_names=protected
    )
    return loop_dir


# ─────────────────────────────────────────────
# The model-loop stage: start, restart and completion
# ─────────────────────────────────────────────

# Beside cognitive_models/: the model set the experiment's model loop started
# from, recorded when the stage first starts. cognitive_models/ is both the
# loop's input and its export, so without this record a restarted loop could
# not tell the carried set from its own half-written export.
MODEL_LOOP_INPUT_DIRNAME = "cognitive_models_input"

# Beside cognitive_models/: the run's agent notes as they were when this
# experiment's model loop first started.
AGENT_NOTES_SNAPSHOT_DIRNAME = "agent_notes_at_start"


def experiment_input_models_dir(exp_dir: Path) -> Path:
    """The model set experiment ``exp_dir`` started from: the recorded input of
    its model loop once that has started, else ``cognitive_models/``."""
    recorded = Path(exp_dir) / MODEL_LOOP_INPUT_DIRNAME
    return recorded if recorded.is_dir() else Path(exp_dir) / "cognitive_models"


def begin_model_loop_stage(exp_dir: Path) -> None:
    """Start (or restart) the model-loop stage from exactly what it started from.

    On the first start, record the stage's input model set
    (``cognitive_models_input/``) and the run's agent notes
    (``agent_notes_at_start/``). On a restart — a previous attempt recorded
    them and did not complete — put both back: ``cognitive_models/`` may hold
    that attempt's export, and the notes describe candidates that no longer
    exist, so the notes written after the stage first started are discarded
    (not archived: anything kept in the run tree is readable by later agents).
    Either way, ``model_loop/`` is emptied and the registry reset to the
    placeholder ``init_registry`` writes, so the stage redoes everything from
    scratch; ``finish_model_loop_stage`` marks it complete.

    Each step is atomic and idempotent, so a crash in here is recovered by the
    next call. A ``model_loop/`` with content but no recorded input was left by
    code that kept no record: ``cognitive_models/`` may already be an export,
    so that raises rather than seeding a loop from it.
    """
    exp_dir = Path(exp_dir)
    models_dir = exp_dir / "cognitive_models"
    recorded_input = exp_dir / MODEL_LOOP_INPUT_DIRNAME
    loop_dir = exp_dir / "model_loop"
    notes = agent_notes_dir(exp_dir)
    notes_at_start = exp_dir / AGENT_NOTES_SNAPSHOT_DIRNAME

    if recorded_input.is_dir():
        print(
            f"  [model-loop] Restart: restoring {models_dir.name}/ and the agent notes "
            f"to what the stage started from; discarding the unfinished attempt",
            flush=True,
        )
        copy_directory_atomically(recorded_input, models_dir)
    else:
        if loop_dir.is_dir() and any(loop_dir.iterdir()):
            raise RuntimeError(
                f"{loop_dir} holds an earlier model loop but {recorded_input} does not "
                "exist: that loop ran on code that did not record its input, so "
                f"{models_dir} may already hold its export. Put back the set the "
                "experiment started from (for experiment 1 delete cognitive_models/ "
                "to re-seed; for a later one delete it to carry the previous "
                "experiment's set again), delete model_loop/, and resume."
            )
        copy_directory_atomically(models_dir, recorded_input)

    if notes_at_start.is_dir():
        copy_directory_atomically(notes_at_start, notes)
    elif notes.is_dir():
        copy_directory_atomically(notes, notes_at_start)
    else:
        replace_directory(notes_at_start, lambda staging: None)

    if loop_dir.exists():
        shutil.rmtree(loop_dir)
    loop_dir.mkdir(parents=True)
    (exp_dir / "model_registry.yaml").unlink(missing_ok=True)
    init_registry(exp_dir)


def finish_model_loop_stage(exp_dir: Path) -> None:
    """Complete the model-loop stage after the loop exported its live set.

    Writes the next design's registry, then the completion record
    ``model_loop/export_complete.json`` (the exported models and the ledger's
    hash), last, and validates the stage (``_validate_model_loop`` checks the
    set, the registry and the ledger against the record). The export and the
    registry update used to be two unvalidated writes after the loop: a crash
    between them left a stage that a resume skipped (its validator asked only
    whether the best model was in the set) with the placeholder registry
    steering the next design.
    """
    exp_dir = Path(exp_dir)
    update_registry_from_interpretation(exp_dir)
    models_dir = exp_dir / "cognitive_models"
    record = {
        "models": read_manifest_names(models_dir),
        "ledger_sha256": ledger_sha256(models_dir),
    }
    record_path = exp_dir / "model_loop" / EXPORT_RECORD_FILENAME
    write_text_atomically(record_path, json.dumps(record, indent=2) + "\n")
    ok, message = validate_cc_output("5_model_loop", exp_dir)
    if not ok:
        record_path.unlink()
        raise RuntimeError(f"5_model_loop output invalid in {exp_dir}: {message}")


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
