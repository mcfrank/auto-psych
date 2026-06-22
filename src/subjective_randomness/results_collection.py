"""Collect a finished holdout test-retest run's results into the repo.

A holdout test-retest run (driven by ``scripts/subjective_randomness/slurm/``)
leaves the following under a scratch ``WORK_ROOT``::

    test_retest.{json,csv,png}            aggregate reliability across repeats
    run<r>/<gt>/holdout.{json,csv,png}    per-(repeat, ground-truth) trajectory

next to heavy material we never want in git: per-task repo copies, MCMC ``.nc``
caches, the shared venv, and agent XDG/session state. This module copies only
the lightweight result artifacts into the repo and renders a compact
``SUMMARY.md`` from the aggregate JSON. It fails loudly if an expected artifact
is missing or the destination is already populated.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

# Artifact naming written by the Slurm pipeline (see slurm/holdout_analysis.sbatch
# and slurm/holdout_recovery_array.sbatch).
AGGREGATE_STEM = "test_retest"
PER_RUN_STEM = "holdout"
ARTIFACT_EXTENSIONS = ("json", "csv", "png")


@dataclass
class CollectionReport:
    """What a single collection run copied."""

    source: Path
    dest: Path
    aggregate_paths: list[Path]
    per_run_paths: list[Path]
    summary_path: Path

    @property
    def n_per_run_artifacts(self) -> int:
        return len(self.per_run_paths)


def discover_per_run_artifacts(source: Path) -> list[Path]:
    """Per-(repeat, ground-truth) ``holdout.{json,csv,png}`` files under ``source``.

    Globs the exact ``run<r>/<gt>/`` depth rather than recursing, so we never
    descend into the per-task repo copies or MCMC caches on Lustre.
    """
    paths: set[Path] = set()
    for ext in ARTIFACT_EXTENSIONS:
        paths.update(source.glob(f"run*/*/{PER_RUN_STEM}.{ext}"))
    return sorted(paths)


def collect_results(
    source: Path | str,
    dest: Path | str,
    *,
    include_per_run: bool = True,
    overwrite: bool = False,
) -> CollectionReport:
    """Copy a finished test-retest run's lightweight artifacts into ``dest``.

    Copies the aggregate ``test_retest.{json,csv,png}`` and (unless
    ``include_per_run`` is False) every per-run ``holdout.{json,csv,png}``,
    preserving the ``run<r>/<gt>/`` layout, then writes a generated
    ``SUMMARY.md``. Raises rather than copying partial or surprising trees.
    """
    source = Path(source)
    dest = Path(dest)

    if not source.is_dir():
        raise FileNotFoundError(f"Source run root is not a directory: {source}")

    aggregate_paths = [source / f"{AGGREGATE_STEM}.{ext}" for ext in ARTIFACT_EXTENSIONS]
    missing = [p for p in aggregate_paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Aggregate {AGGREGATE_STEM}.* artifacts missing under {source}: "
            f"{', '.join(p.name for p in missing)}. "
            "Has the analysis stage (holdout_analysis.sbatch) run?"
        )

    per_run_sources = discover_per_run_artifacts(source) if include_per_run else []
    if include_per_run and not per_run_sources:
        raise FileNotFoundError(
            f"No per-run {PER_RUN_STEM}.* artifacts under run*/ in {source}. "
            "Pass include_per_run=False to collect only the aggregate summary."
        )

    if dest.exists() and any(dest.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Destination {dest} already exists and is not empty. "
            "Pass overwrite=True to replace its contents."
        )

    dest.mkdir(parents=True, exist_ok=True)

    copied_aggregate = []
    for src_path in aggregate_paths:
        out_path = dest / src_path.name
        shutil.copy2(src_path, out_path)
        copied_aggregate.append(out_path)

    copied_per_run = []
    for src_path in per_run_sources:
        relative = src_path.relative_to(source)  # run<r>/<gt>/holdout.<ext>
        out_path = dest / relative
        out_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, out_path)
        copied_per_run.append(out_path)

    summary = json.loads((source / f"{AGGREGATE_STEM}.json").read_text(encoding="utf-8"))
    summary_path = dest / "SUMMARY.md"
    summary_path.write_text(render_test_retest_summary(summary), encoding="utf-8")

    return CollectionReport(
        source=source,
        dest=dest,
        aggregate_paths=copied_aggregate,
        per_run_paths=copied_per_run,
        summary_path=summary_path,
    )


def _fmt(value: object, ndigits: int = 3) -> str:
    """Format a number to ``ndigits`` decimals, or ``n/a`` for missing values."""
    if value is None:
        return "n/a"
    if isinstance(value, (int, float)):
        return f"{value:.{ndigits}f}"
    return str(value)


def render_test_retest_summary(summary: dict) -> str:
    """Render a compact Markdown summary from a ``test_retest.json`` payload."""
    metric = summary.get("metric", "pearson_r")
    runs_found = summary.get("runs_found", [])
    gt_models = summary.get("gt_models", [])

    lines: list[str] = []
    lines.append("# Holdout recovery — test-retest summary")
    lines.append("")
    lines.append(f"- Source run root: `{summary.get('runs_root', 'unknown')}`")
    lines.append(f"- Recovery metric: `{metric}`")
    lines.append(
        f"- Repeats: {len(runs_found)}"
        + (f" ({', '.join(runs_found)})" if runs_found else "")
    )
    lines.append(
        f"- Ground-truth models: {len(gt_models)}"
        + (f" ({', '.join(gt_models)})" if gt_models else "")
    )
    lines.append("")

    lines.append("## Across-repeat reliability")
    lines.append("")
    lines.append(f"- ICC(2,1): {_fmt(summary.get('icc_2_1'))}")
    lines.append(f"- Mean pairwise correlation: {_fmt(summary.get('mean_pairwise_corr'))}")
    lines.append("")

    lines.append(f"## Per ground-truth model (final-step {metric} across repeats)")
    lines.append("")
    lines.append(
        "| Ground truth | n | mean | sd | cv | min | max | modal best model | agreement |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    per_gt = summary.get("per_gt_model", {})
    for gt in gt_models or sorted(per_gt):
        stats = per_gt.get(gt, {})
        lines.append(
            f"| {gt} | {stats.get('n_runs', 0)} | {_fmt(stats.get('mean'))} | "
            f"{_fmt(stats.get('sd'))} | {_fmt(stats.get('cv'))} | "
            f"{_fmt(stats.get('min'))} | {_fmt(stats.get('max'))} | "
            f"{stats.get('modal_best_model') or 'n/a'} | "
            f"{_fmt(stats.get('best_model_agreement'), 2)} |"
        )
    lines.append("")
    return "\n".join(lines)
