"""Map the data tree into structured run data.

A *run* is one outer-loop execution: a directory holding one or more experiments
(theory -> design -> implement -> collect -> model loop), or a single bare model
loop. Runs are discovered anywhere under the data root and identified by their
relative path, so the same structure is presented identically wherever it lives.

Design principle (research code): a *missing* stage is valid — a partial run
simply has empty/``None`` stages — but a *corrupt* artifact raises loudly with
the offending filename rather than being silently skipped.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import yaml
from pydantic import ValidationError


def _argmax_posteriors(posteriors: dict, source: object) -> str | None:
    """Argmax of a model-posterior map, failing loudly (as ValueError, which the
    server surfaces) on non-numeric values rather than raising an opaque TypeError.
    ``source`` names the offending artifact in the error."""
    if not posteriors:
        return None
    try:
        return max(posteriors, key=lambda k: float(posteriors[k]))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Non-numeric posterior values in {source}: {exc}") from exc

from src.viewer.models import (
    Candidate,
    CognitiveModel,
    CritiqueRound,
    CritiqueStat,
    DataSummary,
    DesignStage,
    Experiment,
    ExperimentStage,
    ModelLoopStage,
    Run,
    RunExperimentRef,
    RunIndex,
    RunRef,
    TheoryStage,
    TrajectoryStep,
)
from src.viewer.transcripts import strip_ansi

# Directories that are never experiments / runs in their own right.
_NON_EXPERIMENT_DIRS = {"analysis", "__pycache__"}
_CACHE_DIRS = {"mcmc_cache", "probe_cache", ".fit_cache", "__pycache__",
               ".DS_Store", "models", "test_stats"}
_MAX_PREVIEW_ROWS = 25
_MAX_RUN_DEPTH = 8


# ── small loud readers ───────────────────────────────────────────────────────
def _read_text(path: Path) -> str | None:
    return path.read_text() if path.is_file() else None


def _read_json(path: Path):
    """Parse JSON, raising ``ValueError`` (with the filename) on malformed input."""
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Corrupt JSON in {path.name}: {exc}") from exc


def _read_yaml(path: Path):
    if not path.is_file():
        return None
    try:
        return yaml.safe_load(path.read_text())
    except yaml.YAMLError as exc:
        raise ValueError(f"Corrupt YAML in {path.name}: {exc}") from exc


# ── theory stage ─────────────────────────────────────────────────────────────
_THEORY_SECTION = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)


def _parse_theory_hypotheses(report_md: str | None) -> dict[str, str]:
    """Extract the one-line ``**Hypothesis:**`` per ``## model_name`` section."""
    if not report_md:
        return {}
    hypotheses: dict[str, str] = {}
    sections = _THEORY_SECTION.split(report_md)
    for name, body in zip(sections[1::2], sections[2::2]):
        match = re.search(r"\*\*Hypothesis:\*\*\s*(.+)", body)
        if match:
            hypotheses[name.strip()] = match.group(1).strip()
    return hypotheses


def _scan_theory(exp_dir: Path) -> TheoryStage:
    cm_dir = exp_dir / "cognitive_models"
    if not cm_dir.is_dir():
        return TheoryStage()

    report_md = _read_text(cm_dir / "theory_report.md")
    hypotheses = _parse_theory_hypotheses(report_md)

    manifest = _read_yaml(cm_dir / "models_manifest.yaml") or {}
    entries = manifest.get("models", []) if isinstance(manifest, dict) else []

    models: list[CognitiveModel] = []
    for entry in entries:
        name = entry["name"]
        rationale = (entry.get("rationale") or "").strip()
        code = _read_text(cm_dir / f"{name}.py") or ""
        # Legacy runs exported each experiment's inner-loop winner as a copy
        # named `inner_loop_model` (with a boilerplate rationale) — flag those.
        # Since the descriptive-naming change (2026-07), winners are exported
        # under their own names with real hypotheses, which this heuristic
        # cannot distinguish from seeds; they render as ordinary models.
        is_carried = name == "inner_loop_model" or "inner model-improvement loop" in rationale.lower()
        models.append(
            CognitiveModel(
                name=name,
                rationale=rationale or None,
                hypothesis=hypotheses.get(name),
                code=code,
                origin="inner_loop" if is_carried else "seed",
            )
        )
    return TheoryStage(report_md=report_md, models=models)


# ── design stage ─────────────────────────────────────────────────────────────
def _scan_design(exp_dir: Path) -> DesignStage:
    design_dir = exp_dir / "design"
    stimuli = _read_json(design_dir / "stimuli.json")
    if stimuli is None:
        stimuli = _read_json(exp_dir / "experiment" / "stimuli.json")
    stimuli = stimuli or []
    candidates = _read_json(design_dir / "candidates.json") or []
    return DesignStage(
        rationale_md=_read_text(design_dir / "design_rationale.md"),
        n_stimuli=len(stimuli) if stimuli else None,
        n_candidates=len(candidates) if candidates else None,
        stimuli=stimuli,
    )


# ── implement (experiment) stage ─────────────────────────────────────────────
def _scan_experiment_stage(exp_dir: Path) -> ExperimentStage:
    experiment_dir = exp_dir / "experiment"
    config = _read_json(experiment_dir / "config.json")
    return ExperimentStage(
        config=config,
        has_index_html=(experiment_dir / "index.html").is_file(),
        experiment_url=(config or {}).get("experiment_url"),
    )


# ── collected data ───────────────────────────────────────────────────────────
def _scan_data(exp_dir: Path) -> DataSummary | None:
    responses = exp_dir / "data" / "responses.csv"
    if not responses.is_file():
        return None
    with responses.open(newline="") as fh:
        reader = csv.DictReader(fh)
        columns = reader.fieldnames or []
        rows = list(reader)

    participants = {r.get("participant_id") for r in rows if r.get("participant_id") not in (None, "")}
    p_chose_left = None
    if "chose_left" in columns:
        vals = [float(r["chose_left"]) for r in rows if r.get("chose_left") not in (None, "")]
        if vals:
            p_chose_left = sum(vals) / len(vals)

    return DataSummary(
        n_rows=len(rows),
        n_participants=len(participants),
        columns=list(columns),
        p_chose_left=p_chose_left,
        rows_preview=rows[:_MAX_PREVIEW_ROWS],
    )


# ── inner model loop ─────────────────────────────────────────────────────────
_ITER_DIR = re.compile(r"^iter_(\d+)$")
_CAND_DIR = re.compile(r"^candidate_(\d+)$")


def _scan_candidate(cand_dir: Path, iteration: int, index: int) -> Candidate:
    transcript = _read_text(cand_dir / "agent.jsonl")
    return Candidate(
        iteration=iteration,
        index=index,
        name=f"iter{iteration}_candidate{index}",
        hypothesis=(_read_text(cand_dir / "hypothesis.md") or "").strip() or None,
        brief=_read_text(cand_dir / "CANDIDATE_BRIEF.md"),
        code=_read_text(cand_dir / "candidate.py"),
        posterior=_read_json(cand_dir / "model_posterior.json"),
        transcript=strip_ansi(transcript) if transcript is not None else None,
    )


def _iter_dirs(ml_dir: Path) -> list[Path]:
    return sorted(
        (d for d in ml_dir.iterdir() if d.is_dir() and _ITER_DIR.match(d.name)),
        key=lambda d: int(_ITER_DIR.match(d.name).group(1)),
    )


def _scan_model_loop_at(ml_dir: Path) -> ModelLoopStage | None:
    """Read a model-loop directory (``model_loop/`` or a bare ``loop/``)."""
    if not ml_dir.is_dir():
        return None
    history = _read_json(ml_dir / "history.json") or []
    try:
        trajectory = [TrajectoryStep(**step) for step in history]
    except (TypeError, ValidationError) as exc:
        # Surface a malformed history.json loudly with its path (the server's
        # ValueError handler turns this into a clear 500), rather than letting a
        # pydantic ValidationError / TypeError escape as an opaque stack trace.
        raise ValueError(
            f"Malformed history.json in {ml_dir / 'history.json'}: {exc}"
        ) from exc

    candidates: list[Candidate] = []
    for iter_dir in _iter_dirs(ml_dir):
        iteration = int(_ITER_DIR.match(iter_dir.name).group(1))
        cand_dirs = sorted(
            (d for d in iter_dir.iterdir() if d.is_dir() and _CAND_DIR.match(d.name)),
            key=lambda d: int(_CAND_DIR.match(d.name).group(1)),
        )
        for cand_dir in cand_dirs:
            index = int(_CAND_DIR.match(cand_dir.name).group(1))
            candidates.append(_scan_candidate(cand_dir, iteration, index))

    return ModelLoopStage(
        report_md=_read_text(ml_dir / "report.md"),
        trajectory=trajectory,
        final_posterior=_read_json(ml_dir / "model_posterior.json"),
        candidates=candidates,
    )


def _scan_model_loop(exp_dir: Path) -> ModelLoopStage | None:
    return _scan_model_loop_at(exp_dir / "model_loop")


def _best_model(model_loop: ModelLoopStage | None) -> str | None:
    if model_loop is None:
        return None
    if model_loop.trajectory:
        return model_loop.trajectory[-1].best_model
    posteriors = (model_loop.final_posterior or {}).get("posteriors")
    return _argmax_posteriors(posteriors, "model_posterior.json")


def _quick_loop_stats(ml_dir: Path) -> tuple[str | None, int | None, int | None]:
    """Cheap best-model / iteration / candidate counts without reading transcripts."""
    history = _read_json(ml_dir / "history.json") or []
    final = _read_json(ml_dir / "model_posterior.json") or {}
    if history and history[-1].get("best_model"):
        best = history[-1]["best_model"]
    else:
        best = _argmax_posteriors(
            final.get("posteriors"), ml_dir / "model_posterior.json"
        )
    iters = _iter_dirs(ml_dir) if ml_dir.is_dir() else []
    n_cand = sum(
        1 for it in iters for c in it.iterdir() if c.is_dir() and _CAND_DIR.match(c.name)
    )
    return best, (len(iters) or None), (n_cand or None)


# ── critique (CriticAL posterior-predictive checks) ──────────────────────────
_STAT_NAME_RE = re.compile(r"^#\s*name:\s*(.+?)\s*$", re.MULTILINE)
_STAT_DESC_RE = re.compile(r"^#\s*description:\s*(.+?)\s*$", re.MULTILINE)
_DOCSTRING_RE = re.compile(r'"""(.*?)"""|\'\'\'(.*?)\'\'\'', re.DOTALL)


def _parse_test_stat(path: Path) -> tuple[str, str | None, str]:
    """Extract (name, description, code) from a test-statistic ``.py`` file.

    Prefers the ``# name:`` / ``# description:`` header comments the critique
    agent writes; falls back to the file stem and the function docstring.
    """
    code = path.read_text()
    name_match = _STAT_NAME_RE.search(code)
    desc_match = _STAT_DESC_RE.search(code)
    name = name_match.group(1).strip() if name_match else path.stem
    if desc_match:
        description = desc_match.group(1).strip()
    else:
        doc = _DOCSTRING_RE.search(code)
        description = " ".join((doc.group(1) or doc.group(2)).split()) if doc else None
    return name, description, code


def _stat_from_result(name, result, description=None, code=None) -> CritiqueStat:
    return CritiqueStat(
        name=name,
        description=description or result.get("description"),
        code=code or result.get("code"),
        t_observed=result.get("t_observed"),
        null_mean=result.get("null_mean"),
        null_std=result.get("null_std"),
        z_score=result.get("z_score"),
        p_value=result.get("p_value"),
        # ppc.py writes the Benjamini-Hochberg FDR q-value under ``p_value_fdr``
        # (``p_value_adjusted`` is the viewer's generic name for "multiplicity-
        # adjusted p"); read the key the harness actually emits.
        p_value_adjusted=result.get("p_value_fdr"),
        significant=result.get("significant"),
        error=result.get("error"),
        has_result=True,
    )


def _scan_critique_dir(cdir: Path, iteration: int | None) -> CritiqueRound:
    ppc = _read_json(cdir / "ppc_results.json") or {}
    results_by_name = {r["name"]: r for r in ppc.get("results", [])}

    stats: list[CritiqueStat] = []
    used: set[str] = set()
    stats_dir = cdir / "test_stats"
    if stats_dir.is_dir():
        for f in sorted(stats_dir.glob("*.py")):
            name, description, code = _parse_test_stat(f)
            result = results_by_name.get(name)
            if result is not None:
                used.add(name)
                stats.append(_stat_from_result(name, result, description, code))
            else:
                stats.append(CritiqueStat(name=name, description=description, code=code))
    # PPC results that had no matching source file are still surfaced.
    for name, result in results_by_name.items():
        if name not in used:
            stats.append(_stat_from_result(name, result))

    return CritiqueRound(
        iteration=iteration,
        context_md=_read_text(cdir / "CRITIQUE_CONTEXT.md"),
        model=ppc.get("model"),
        n_significant=ppc.get("n_significant"),
        n_replicates=ppc.get("n_replicates"),
        significance_alpha=ppc.get("significance_alpha"),
        stats=stats,
    )


def _scan_critiques(exp_dir: Path) -> list[CritiqueRound]:
    """Collect per-iteration (model_loop/iter_N/critique) and experiment-level critiques."""
    rounds: list[CritiqueRound] = []
    ml_dir = exp_dir / "model_loop"
    if ml_dir.is_dir():
        for iter_dir in _iter_dirs(ml_dir):
            cdir = iter_dir / "critique"
            if cdir.is_dir():
                rounds.append(_scan_critique_dir(cdir, int(_ITER_DIR.match(iter_dir.name).group(1))))
    exp_critique = exp_dir / "critique"
    if exp_critique.is_dir():
        rounds.append(_scan_critique_dir(exp_critique, None))
    return rounds


# ── one experiment ───────────────────────────────────────────────────────────
def scan_experiment_dir(exp_dir: Path, project: str, name: str) -> Experiment:
    """Read every stage of one experiment directory into an :class:`Experiment`."""
    exp_dir = Path(exp_dir)
    if not exp_dir.is_dir():
        raise FileNotFoundError(f"No such experiment: {project}/{name} ({exp_dir})")

    model_loop = _scan_model_loop(exp_dir)
    return Experiment(
        project=project,
        name=name,
        theory=_scan_theory(exp_dir),
        design=_scan_design(exp_dir),
        experiment=_scan_experiment_stage(exp_dir),
        data=_scan_data(exp_dir),
        model_loop=model_loop,
        critiques=_scan_critiques(exp_dir),
        best_model=_best_model(model_loop),
    )


def scan_loop_dir(loop_dir: Path, project: str, name: str) -> Experiment:
    """Wrap a bare model-loop directory (no experiment scaffolding) as an Experiment."""
    loop_dir = Path(loop_dir)
    if not loop_dir.is_dir():
        raise FileNotFoundError(f"No such loop: {project}/{name} ({loop_dir})")
    model_loop = _scan_model_loop_at(loop_dir)
    # A bare loop may still sit next to its own responses.csv and critique dir.
    return Experiment(
        project=project,
        name=name,
        data=_scan_data(loop_dir.parent) if (loop_dir.parent / "data").is_dir() else None,
        model_loop=model_loop,
        critiques=_scan_critiques(loop_dir),
        best_model=_best_model(model_loop),
    )


# ── run discovery ────────────────────────────────────────────────────────────
_EXPERIMENT_MARKERS = ("model_loop", "cognitive_models", "design")


def _is_experiment_dir(d: Path) -> bool:
    """A directory that holds the stages of one experiment."""
    return d.is_dir() and any((d / m).is_dir() for m in _EXPERIMENT_MARKERS)


def _is_loop_dir(d: Path) -> bool:
    """A bare model loop: a model posterior plus the scored model set."""
    return (d / "model_posterior.json").is_file() and (d / "models").is_dir()


def _exp_sort_key(name: str):
    match = re.match(r"experiment(\d+)$", name)
    return (0, int(match.group(1))) if match else (1, name)


def _run_units(run_dir: Path) -> list[tuple[str, str]]:
    """The (unit, kind) pairs inside a run dir; empty if the dir is not a run.

    ``unit`` is an experiment subdir name, ``"loop"`` for a ``loop/`` subdir, or
    ``"."`` when the run dir is itself the experiment / model loop.
    """
    exp_children = sorted(
        (c.name for c in run_dir.iterdir()
         if _is_experiment_dir(c) and c.name not in _NON_EXPERIMENT_DIRS),
        key=_exp_sort_key,
    )
    if exp_children:
        return [(name, "experiment") for name in exp_children]
    if _is_experiment_dir(run_dir):
        return [(".", "experiment")]
    if (run_dir / "loop").is_dir() and _is_loop_dir(run_dir / "loop"):
        return [("loop", "loop")]
    if _is_loop_dir(run_dir):
        return [(".", "loop")]
    return []


def _is_run(d: Path) -> bool:
    if not d.is_dir() or d.name.startswith(".") or d.name in _CACHE_DIRS:
        return False
    return bool(_run_units(d))


def find_runs(data_root: Path) -> list[RunRef]:
    """Walk the data tree and return every run, stopping at each run (no nesting)."""
    data_root = Path(data_root)
    if not data_root.is_dir():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")

    runs: list[RunRef] = []

    def walk(d: Path, rel: str, depth: int) -> None:
        if depth > _MAX_RUN_DEPTH or d.name.startswith(".") or d.name in _CACHE_DIRS:
            return
        if _is_run(d):
            units = _run_units(d)
            kind = "loop" if all(k == "loop" for _, k in units) else "experiments"
            runs.append(RunRef(path=rel, label=d.name, kind=kind, n_experiments=len(units)))
            return  # a run is a leaf — do not descend into its experiments
        for child in sorted(c for c in d.iterdir() if c.is_dir()):
            walk(child, f"{rel}/{child.name}" if rel else child.name, depth + 1)

    for child in sorted(c for c in data_root.iterdir() if c.is_dir()):
        walk(child, child.name, 1)
    return runs


def scan_index(data_root: Path) -> RunIndex:
    """List every run found under the data root."""
    return RunIndex(runs=find_runs(data_root))


# ── one run ──────────────────────────────────────────────────────────────────
def _unit_dir(run_dir: Path, unit: str) -> Path:
    return run_dir if unit == "." else run_dir / unit


def _unit_ml_dir(run_dir: Path, unit: str, kind: str) -> Path:
    udir = _unit_dir(run_dir, unit)
    return udir if kind == "loop" else udir / "model_loop"


def scan_run(data_root: Path, run_path: str) -> Run:
    """Summarize a run: its experiment units (with best model) and run-level figures."""
    run_dir = Path(data_root) / run_path
    if not run_dir.is_dir():
        raise FileNotFoundError(f"No such run: {run_path} ({run_dir})")
    units = _run_units(run_dir)
    if not units:
        raise FileNotFoundError(f"Not a run: {run_path}")

    figures = sorted(
        f"analysis/{f.name}" for f in (run_dir / "analysis").glob("*.png")
    ) if (run_dir / "analysis").is_dir() else []

    experiments = []
    for unit, kind in units:
        best, _n_iter, n_cand = _quick_loop_stats(_unit_ml_dir(run_dir, unit, kind))
        experiments.append(
            RunExperimentRef(
                unit=unit,
                name=("experiment" if unit == "." else unit),
                kind=kind,
                best_model=best,
                n_candidates=n_cand,
            )
        )
    return Run(path=run_path, label=run_dir.name, figures=figures, experiments=experiments)


def scan_run_experiment(data_root: Path, run_path: str, unit: str) -> Experiment:
    """Scan one experiment (or bare loop) unit of a run into an :class:`Experiment`."""
    run_dir = Path(data_root) / run_path
    units = dict(_run_units(run_dir))
    if unit not in units:
        raise FileNotFoundError(f"No such experiment unit: {run_path}/{unit}")
    kind = units[unit]
    target = _unit_dir(run_dir, unit)
    label = "experiment" if unit == "." else unit
    if kind == "loop":
        return scan_loop_dir(target, run_path, label)
    return scan_experiment_dir(target, run_path, label)
