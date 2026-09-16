"""Digest of holdout-recovery sweeps for the review agent.

A sweep root (a ``WORK_ROOT`` of ``scripts/subjective_randomness/slurm/``)
holds ``test_retest.{json,csv,png}``, ``run<r>/<gt>/holdout.{json,csv,png}``
and ``slurm_logs/holdout_recovery_<array>_<task>.out``. The agent could read
all of that itself, but a 20-task sweep is thousands of log lines; this module
condenses each sweep into one Markdown section — aggregate reliability, every
(ground truth, repeat) final fit next to its seed-only starting point, and each
task that crashed with its error line — plus a cross-sweep comparison table.
Nothing is silently skipped: a sweep with no aggregate, or a task with no
result, is reported as such.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src.runtime.formatting import format_number

AGGREGATE_NAME = "test_retest.json"
PER_RUN_CSV = "holdout.csv"
PER_RUN_JSON = "holdout.json"
TASK_LOG_GLOB = "slurm_logs/holdout_recovery_*.out"

_TASK_LINE = re.compile(r"^\[task (\d+)\] repeat=(\d+) gt=(\S+)")
_DONE_LINE = re.compile(r"^\[task \d+\] done ->")
_COST = re.compile(r"cost=\$([0-9]+(?:\.[0-9]+)?)")
# The last line matching this is the task's headline error.
_ERROR_LINE = re.compile(
    r"^(?:[\w.]*(?:Error|Exception|Exit)\b.*|ABORT:.*|ERROR:.*"
    r"|slurmstepd: error:.*|srun: error:.*)"
)
# The leakage-audit flags in holdout.json, with the short label the table shows.
_LEAKAGE_FLAGS = (
    ("any_identical", "identical-copy"),
    ("any_mention", "param-name"),
    ("any_value_mention", "param-value"),
    ("any_gt_named", "gt-named-file"),
)


@dataclass
class TaskOutcome:
    """What one array task's Slurm log says happened to it."""

    task: int
    log: Path
    repeat: Optional[int] = None
    gt: Optional[str] = None
    status: str = "incomplete"
    """``done`` (reached its final line), ``failed`` (an error line), or
    ``incomplete`` (neither: still running, killed, or cut off)."""
    error: str = ""
    cost_usd: Optional[float] = None


@dataclass
class RunRow:
    """One (ground truth, repeat) trajectory, reduced to its endpoints."""

    gt: str
    run: str
    r_seed: Optional[float]
    """Pearson r of the seed-set-only fit (global step 0)."""
    r_final: Optional[float]
    rmse_final: Optional[float]
    best_final: str
    n_steps: int
    leakage: str

    @property
    def delta(self) -> Optional[float]:
        if self.r_seed is None or self.r_final is None:
            return None
        return self.r_final - self.r_seed


@dataclass
class SweepSummary:
    label: str
    root: Path
    aggregate: Optional[dict] = None
    settings: dict = field(default_factory=dict)
    rows: list[RunRow] = field(default_factory=list)
    tasks: list[TaskOutcome] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    trees: dict[str, int] = field(default_factory=dict)
    """How many cells keep their experiment tree ``raw`` (``repo/_runs/<gt>/``),
    ``tarred`` (``agent_runs.tar.gz``) or ``missing`` — where the critiques,
    candidate models and agent transcripts live."""

    def count(self, status: str) -> int:
        return sum(1 for t in self.tasks if t.status == status)

    @property
    def total_cost_usd(self) -> Optional[float]:
        costs = [t.cost_usd for t in self.tasks if t.cost_usd is not None]
        return sum(costs) if costs else None


def _float(value: Optional[str]) -> Optional[float]:
    if value is None or value == "":
        return None
    return float(value)


def _run_number(run_dir_name: str) -> int:
    digits = "".join(ch for ch in run_dir_name if ch.isdigit())
    return int(digits) if digits else 0


def _leakage_label(holdout_json: Path) -> str:
    if not holdout_json.is_file():
        return "no holdout.json"
    payload = json.loads(holdout_json.read_text(encoding="utf-8"))
    gt_runs = payload.get("gt_runs") or []
    if not gt_runs:
        return "no gt_runs"
    leakage = gt_runs[0].get("leakage") or {}
    flags = [label for key, label in _LEAKAGE_FLAGS if leakage.get(key)]
    return ", ".join(flags) if flags else "clean"


def _run_row(csv_path: Path) -> RunRow:
    with csv_path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError(f"{csv_path} has no trajectory rows")
    rows.sort(key=lambda r: float(r.get("global_step") or r.get("step") or 0))
    first, last = rows[0], rows[-1]
    return RunRow(
        gt=csv_path.parent.name,
        run=csv_path.parent.parent.name,
        r_seed=_float(first.get("pearson_r")),
        r_final=_float(last.get("pearson_r")),
        rmse_final=_float(last.get("rmse")),
        best_final=last.get("best_model") or "?",
        n_steps=len(rows),
        leakage=_leakage_label(csv_path.parent / PER_RUN_JSON),
    )


def experiment_tree_kind(cell_dir: Path) -> str:
    """``raw`` / ``tarred`` / ``missing``: where a cell's experiment tree is."""
    raw = cell_dir / "repo" / "_runs" / cell_dir.name
    if raw.is_dir():
        return "raw"
    if (cell_dir / "agent_runs.tar.gz").is_file():
        return "tarred"
    return "missing"


def _settings_from(holdout_json: Path) -> dict:
    payload = json.loads(holdout_json.read_text(encoding="utf-8"))
    return {
        key: payload.get(key)
        for key in ("n_experiments", "n_participants", "inner_loop", "fit_kwargs", "design")
    }


def parse_task_log(log: Path) -> TaskOutcome:
    """Reduce one array task's log to (repeat, gt, status, error, cost)."""
    suffix = log.stem.rsplit("_", 1)[-1]
    outcome = TaskOutcome(task=int(suffix) if suffix.isdigit() else -1, log=log)
    last_error = ""
    for raw in log.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.rstrip()
        match = _TASK_LINE.match(line)
        if match:
            outcome.task = int(match.group(1))
            outcome.repeat = int(match.group(2))
            outcome.gt = match.group(3)
        if _DONE_LINE.match(line):
            outcome.status = "done"
        cost = _COST.search(line)
        if cost:
            outcome.cost_usd = float(cost.group(1))
        if _ERROR_LINE.match(line):
            last_error = line
    if outcome.status != "done" and last_error:
        outcome.status = "failed"
        outcome.error = last_error
    return outcome


def summarize_sweep(root: Path, label: Optional[str] = None) -> SweepSummary:
    """Read everything the digest reports about one sweep root."""
    root = Path(root)
    summary = SweepSummary(label=label or root.name, root=root)
    if not root.is_dir():
        summary.notes.append(f"sweep root {root} does not exist (sweep not started, or wrong path)")
        return summary

    aggregate_path = root / AGGREGATE_NAME
    if aggregate_path.is_file():
        summary.aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    else:
        summary.notes.append(
            f"no {AGGREGATE_NAME}: the analysis stage has not run (sweep unfinished or its "
            "array failed) — per-run rows and task logs below are all there is"
        )

    csv_paths = sorted(
        root.glob(f"run*/*/{PER_RUN_CSV}"),
        key=lambda p: (p.parent.name, _run_number(p.parent.parent.name)),
    )
    summary.rows = [_run_row(p) for p in csv_paths]
    for csv_path in csv_paths:
        kind = experiment_tree_kind(csv_path.parent)
        summary.trees[kind] = summary.trees.get(kind, 0) + 1
    if not csv_paths:
        summary.notes.append(f"no run*/<gt>/{PER_RUN_CSV} files: no task has finished")
    json_paths = [p.parent / PER_RUN_JSON for p in csv_paths if (p.parent / PER_RUN_JSON).is_file()]
    if json_paths:
        summary.settings = _settings_from(json_paths[0])

    logs = sorted(root.glob(TASK_LOG_GLOB), key=lambda p: parse_task_log(p).task)
    summary.tasks = [parse_task_log(p) for p in logs]
    if not logs:
        summary.notes.append("no slurm_logs/holdout_recovery_*.out: no array task has started")
    return summary


def _settings_line(settings: dict) -> str:
    if not settings:
        return "Settings: unknown (no per-run holdout.json yet)"
    inner = settings.get("inner_loop") or {}
    fit = settings.get("fit_kwargs") or {}
    design = settings.get("design")
    fit_text = "/".join(str(fit.get(k, "?")) for k in ("draws", "tune", "chains"))
    extra = ", ".join(f"{k}={v}" for k, v in fit.items() if k not in ("draws", "tune", "chains"))
    return (
        f"Settings: n_experiments={settings.get('n_experiments')}, "
        f"n_participants={settings.get('n_participants')}, "
        f"inner_loop={inner.get('max_iterations')}x{inner.get('candidate_count')}, "
        f"fit draws/tune/chains={fit_text}" + (f" ({extra})" if extra else "")
        + f", design={design if design else 'n/a (config default)'}"
    )


def render_sweep(summary: SweepSummary) -> str:
    lines = [f"## Sweep `{summary.label}`", "", f"Root: `{summary.root}`", ""]
    for note in summary.notes:
        lines.append(f"- NOTE: {note}")
    if summary.notes:
        lines.append("")
    lines.append(_settings_line(summary.settings))
    lines.append("")

    agg = summary.aggregate
    if agg:
        complete = agg.get("runs_in_complete_matrix") or []
        found = agg.get("runs_found") or []
        lines.append(
            f"Aggregate: ICC(2,1)={format_number(agg.get('icc_2_1'))}, "
            f"mean pairwise corr={format_number(agg.get('mean_pairwise_corr'))}, "
            f"repeats in complete matrix {len(complete)}/{len(found)}"
        )
        lines.append("")
        lines.append("| Ground truth | n | mean r | sd | min | max | modal best model | agreement |")
        lines.append("|---|---|---|---|---|---|---|---|")
        per_gt = agg.get("per_gt_model") or {}
        for gt in agg.get("gt_models") or sorted(per_gt):
            s = per_gt.get(gt, {})
            lines.append(
                f"| {gt} | {s.get('n_runs', 0)} | {format_number(s.get('mean'))} | "
                f"{format_number(s.get('sd'))} | {format_number(s.get('min'))} | "
                f"{format_number(s.get('max'))} | {s.get('modal_best_model') or 'n/a'} | "
                f"{format_number(s.get('best_model_agreement'), 2)} |"
            )
        lines.append("")

    if summary.rows:
        lines.append("Per (ground truth, repeat) — final fit vs. the seed-set-only start:")
        lines.append("")
        lines.append("| Ground truth | run | r seed-only | r final | delta | rmse final | final best model | steps | leakage audit |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for row in summary.rows:
            lines.append(
                f"| {row.gt} | {row.run} | {format_number(row.r_seed)} | "
                f"{format_number(row.r_final)} | {format_number(row.delta)} | "
                f"{format_number(row.rmse_final)} | {row.best_final} | {row.n_steps} | {row.leakage} |"
            )
        lines.append("")
        where = {
            "raw": "raw under `run<r>/<gt>/repo/_runs/<gt>/experiment<k>/`",
            "tarred": "in `run<r>/<gt>/agent_runs.tar.gz`",
            "missing": "missing",
        }
        lines.append(
            "Experiment trees (critiques, candidate models, agent transcripts): "
            + "; ".join(f"{n} cell(s) {where[k]}" for k, n in sorted(summary.trees.items()))
        )
        lines.append("")

    if summary.tasks:
        cost = summary.total_cost_usd
        lines.append(
            f"Tasks: {summary.count('done')} done, {summary.count('failed')} failed, "
            f"{summary.count('incomplete')} incomplete"
            + (f"; agent spend ${cost:.2f} (tasks that reported it)" if cost is not None else "")
        )
        problems = [t for t in summary.tasks if t.status != "done"]
        if problems:
            lines.append("")
            lines.append("Tasks that did not finish:")
            for t in problems:
                where = f"run{t.repeat}/{t.gt}" if t.repeat is not None else "cell unknown"
                detail = t.error if t.error else "no error line (still running, killed, or cut off)"
                lines.append(f"- task {t.task} ({where}) {t.status}: {detail} — log `{t.log}`")
        lines.append("")
    return "\n".join(lines)


def _cell(agg: Optional[dict], gt: str) -> str:
    if not agg:
        return "n/a"
    s = (agg.get("per_gt_model") or {}).get(gt)
    if not s:
        return "n/a"
    return f"{format_number(s.get('mean'))} ± {format_number(s.get('sd'))} (n={s.get('n_runs', 0)})"


def _metric_cell(agg: Optional[dict], metric: str, gt: str) -> str:
    if not agg:
        return "n/a"
    per_metric = agg.get("per_metric") or {}
    m_data = per_metric.get(metric, {}).get("per_gt_model", {}).get(gt)
    if not m_data:
        return "n/a"
    return f"{format_number(m_data.get('mean'))} ± {format_number(m_data.get('sd'))}"


def _metric_mean(agg: Optional[dict], metric: str) -> str:
    if not agg:
        return "n/a"
    per_metric = agg.get("per_metric") or {}
    m_data = per_metric.get(metric, {}).get("per_gt_model", {})
    values = [v.get("mean") for v in m_data.values() if v.get("mean") is not None]
    return format_number(sum(values) / len(values)) if values else "n/a"


def render_comparison(summaries: list[SweepSummary]) -> str:
    """Cross-sweep table: per ground truth, RMSE first, then Pearson r."""
    gts: list[str] = []
    for s in summaries:
        for gt in (s.aggregate or {}).get("gt_models") or []:
            if gt not in gts:
                gts.append(gt)
    labels = [s.label for s in summaries]
    lines = ["## Comparison across sweeps (mean ± sd across repeats)", ""]
    if not gts:
        lines.append("No sweep has an aggregate yet; nothing to compare.")
        lines.append("")
        return "\n".join(lines)

    lines.append("### RMSE (primary)")
    lines.append("")
    lines.append("| Ground truth | " + " | ".join(f"`{l}`" for l in labels) + " |")
    lines.append("|---|" + "---|" * len(labels))
    for gt in gts:
        lines.append(
            f"| {gt} | "
            + " | ".join(_metric_cell(s.aggregate, "rmse", gt) for s in summaries)
            + " |"
        )
    lines.append(
        "| **mean** | "
        + " | ".join(_metric_mean(s.aggregate, "rmse") for s in summaries)
        + " |"
    )
    lines.append("")

    lines.append("### Pearson r (descriptive)")
    lines.append("")
    lines.append("| Ground truth | " + " | ".join(f"`{l}`" for l in labels) + " |")
    lines.append("|---|" + "---|" * len(labels))
    for gt in gts:
        lines.append(f"| {gt} | " + " | ".join(_cell(s.aggregate, gt) for s in summaries) + " |")
    means = []
    for s in summaries:
        per_gt = (s.aggregate or {}).get("per_gt_model") or {}
        values = [v.get("mean") for v in per_gt.values() if v.get("mean") is not None]
        means.append(format_number(sum(values) / len(values)) if values else "n/a")
    lines.append("| **mean** | " + " | ".join(means) + " |")
    lines.append(
        "| ICC(2,1) | "
        + " | ".join(format_number((s.aggregate or {}).get("icc_2_1")) for s in summaries)
        + " |"
    )
    lines.append("")
    return "\n".join(lines)


def build_digest(roots: list[tuple[str, Path]], *, primary_label: str) -> str:
    """Markdown digest of ``roots`` (label, path) with ``primary_label`` first.

    ``primary_label`` names the sweep the agent is asked to act on (the most
    recent one); the others are context.
    """
    labels = [label for label, _ in roots]
    if primary_label not in labels:
        raise ValueError(f"primary label {primary_label!r} is not among {labels}")
    summaries = [summarize_sweep(path, label) for label, path in roots]
    parts = [
        "# Sweep digest",
        "",
        f"Primary sweep to review: `{primary_label}`. "
        f"Sweeps included ({len(summaries)}): " + ", ".join(f"`{l}`" for l in labels) + ".",
        "",
    ]
    if len(summaries) > 1:
        parts.append(render_comparison(summaries))
    for summary in summaries:
        parts.append(render_sweep(summary))
    return "\n".join(parts)
