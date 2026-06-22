"""Collect a finished live (human) outer-loop run's results into the repo.

A live run (``scripts/outer_loop_live/run_live.sbatch``) writes, under a scratch
``WORK_ROOT`` (e.g. ``$SCRATCH/auto-psych/outer_loop_live``), one tree per run
label::

    <run_label>/data/<project>/experiment<N>/
        data/responses.csv          collected human responses
        design/                     stimuli + design rationale
        cognitive_models/           the seeded + inner-loop model code
        model_loop/                 model_posterior.json, history.json, report.md,
                                    best_model.py, per-candidate code, agent logs
        logs/, deployment/, experiment/  agent transcripts + deploy artifacts

next to heavy, non-result material we never commit: per-run repo worktrees
(``runs/<label>/repo`` with node_modules), the shared venv, language caches, and
the giant per-fit MCMC ``.nc`` traces under ``model_loop/**/.fit_cache``.

This module copies only the result trees of the *real* runs (pilots and
``_validate*`` runs are skipped by default), strips the heavy material, and
**scrubs every Prolific ID** before anything lands in the repo:

* the raw Prolific worker id is the ``participant_id_str`` column of every
  responses.csv — that column is dropped (the anonymized integer
  ``participant_id`` index is kept);
* Prolific worker/study ids also appear as bare 24-hex tokens in logs,
  deployment manifests, configs and agent transcripts — every such token is
  redacted in all copied text files; and
* as a fail-loud backstop, the copied tree is re-scanned and collection raises
  rather than leave any surviving Prolific id on disk.
"""

from __future__ import annotations

import csv
import io
import json
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Directories / files that are never results. ``.fit_cache`` holds the multi-100MB
# MCMC ``.nc`` traces; the rest are build/output cruft that can reappear in a tree.
EXCLUDE_DIR_NAMES = {".fit_cache", "node_modules", "__pycache__", ".git"}
EXCLUDE_FILE_SUFFIXES = (".nc",)

# Run-label prefixes/names that are not the "real" human loop.
PILOT_PREFIX = "pilot"
INFRA_RUN_NAMES = {"runs", "slurm_logs", "venv", "worktrees"}

# Prolific worker and study ids are 24-char lowercase-hex tokens (Mongo
# ObjectId format). This shape never occurs in the H/T-sequence research data,
# so matching it is safe; the ``{{%PROLIFIC_PID%}}`` URL placeholder has no hex
# and is left intact.
PROLIFIC_ID_PATTERN = re.compile(r"\b[0-9a-f]{24}\b")
REDACTION = "[REDACTED_PROLIFIC_ID]"

# CSV columns carrying participant-identifying info. The integer ``participant_id``
# (an anonymized 0..N index) is deliberately NOT here — it is what makes the data
# analyzable.
PII_COLUMN_DENYLIST = {
    "participant_id_str",
    "prolific_pid",
    "prolific_id",
    "worker_id",
    "session_id",
    "study_id",
}

# File suffixes we treat as text for redaction / the verification scan.
TEXT_SUFFIXES = {
    ".csv", ".tsv", ".json", ".jsonl", ".log", ".md", ".txt",
    ".py", ".yaml", ".yml", ".html", ".htm", ".js", ".css",
}


@dataclass
class HumanCollectionReport:
    """What a single live-run collection copied and scrubbed."""

    source: Path
    dest: Path
    runs: list[str]
    experiment_paths: list[Path] = field(default_factory=list)
    summary_path: Optional[Path] = None
    n_ids_scrubbed: int = 0

    @property
    def n_experiments(self) -> int:
        return len(self.experiment_paths)


# --- Prolific-ID scrubbing (pure) ------------------------------------------


def find_prolific_ids(text: str) -> set[str]:
    """Return every 24-hex Prolific-id token in ``text``."""
    return set(PROLIFIC_ID_PATTERN.findall(text))


def redact_text(text: str) -> str:
    """Replace every Prolific-id token with the redaction marker."""
    return PROLIFIC_ID_PATTERN.sub(REDACTION, text)


def redact_csv_text(text: str, drop_columns: set[str] = PII_COLUMN_DENYLIST) -> str:
    """Drop PII columns from CSV ``text``, then redact any residual id tokens."""
    reader = csv.reader(io.StringIO(text))
    rows = list(reader)
    if not rows:
        return redact_text(text)
    header = rows[0]
    keep = [i for i, name in enumerate(header) if name.strip().lower() not in drop_columns]
    out = io.StringIO()
    writer = csv.writer(out)
    for row in rows:
        writer.writerow([row[i] for i in keep if i < len(row)])
    return redact_text(out.getvalue())


def _scrub_file(path: Path) -> int:
    """Scrub a copied file in place. Returns the number of ids removed."""
    if path.suffix.lower() not in TEXT_SUFFIXES:
        return 0
    original = path.read_text(encoding="utf-8", errors="ignore")
    n_ids = len(find_prolific_ids(original))
    if path.suffix.lower() in (".csv", ".tsv"):
        cleaned = redact_csv_text(original)
    else:
        cleaned = redact_text(original)
    if cleaned != original:
        path.write_text(cleaned, encoding="utf-8")
    return n_ids


# --- run discovery ----------------------------------------------------------


def _has_experiments(run_dir: Path, project: str) -> bool:
    project_dir = run_dir / "data" / project
    return project_dir.is_dir() and any(project_dir.glob("experiment*"))


def discover_runs(
    source: Path | str, *, project: str, include_pilots: bool = False
) -> list[str]:
    """Sorted labels of runs under ``source`` that hold real experiment data.

    Skips infra dirs (the worktree copies, venv, caches), ``_validate*`` runs,
    and — unless ``include_pilots`` — anything named ``pilot*``.
    """
    source = Path(source)
    labels = []
    for child in source.iterdir():
        if not child.is_dir():
            continue
        name = child.name
        if name.startswith(".") or name.startswith("_") or name in INFRA_RUN_NAMES:
            continue
        if name.startswith(PILOT_PREFIX) and not include_pilots:
            continue
        if _has_experiments(child, project):
            labels.append(name)
    return sorted(labels)


# --- copy + summarize -------------------------------------------------------


def _copy_ignore(_dir: str, names: list[str]) -> set[str]:
    ignored = {n for n in names if n in EXCLUDE_DIR_NAMES}
    ignored |= {n for n in names if n.endswith(EXCLUDE_FILE_SUFFIXES)}
    return ignored


def _count_responses(responses_csv: Path) -> tuple[int, Optional[int]]:
    """(#response rows, #unique participants) from a responses.csv."""
    with responses_csv.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    n_responses = len(rows)
    n_participants = (
        len({r["participant_id"] for r in rows}) if rows and "participant_id" in rows[0] else None
    )
    return n_responses, n_participants


def _experiment_record(run: str, exp_dir: Path) -> dict:
    """Build one summary record from an experiment's on-disk results."""
    record: dict = {"run": run, "experiment": exp_dir.name}

    posterior_path = exp_dir / "model_loop" / "model_posterior.json"
    if posterior_path.is_file():
        record.update(summarize_model_posterior(json.loads(posterior_path.read_text())))
    else:
        record["incomplete"] = True

    responses_path = exp_dir / "data" / "responses.csv"
    if responses_path.is_file():
        record["n_responses"], record["n_participants"] = _count_responses(responses_path)
    return record


def collect_human_results(
    source: Path | str,
    dest: Path | str,
    *,
    project: str,
    runs: Optional[list[str]] = None,
    include_pilots: bool = False,
    overwrite: bool = False,
) -> HumanCollectionReport:
    """Copy the real runs' scrubbed result trees into ``dest`` and summarize them."""
    source = Path(source)
    dest = Path(dest)

    if not source.is_dir():
        raise FileNotFoundError(f"Source run root is not a directory: {source}")

    selected = runs if runs is not None else discover_runs(
        source, project=project, include_pilots=include_pilots
    )
    if not selected:
        raise FileNotFoundError(
            f"No live runs with {project}/experiment* data found under {source}."
        )

    if dest.exists() and any(dest.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Destination {dest} already exists and is not empty. "
            "Pass overwrite=True to replace its contents."
        )
    dest.mkdir(parents=True, exist_ok=True)

    experiment_paths: list[Path] = []
    records: list[dict] = []
    for run in selected:
        src_project = source / run / "data" / project
        if not src_project.is_dir():
            raise FileNotFoundError(f"Expected {src_project} for run {run!r}, not found.")
        dst_project = dest / run / project  # drop the redundant data/ level
        shutil.copytree(
            src_project, dst_project, ignore=_copy_ignore, dirs_exist_ok=overwrite
        )
        for exp_dir in sorted(src_project.glob("experiment*")):
            experiment_paths.append(dst_project / exp_dir.name)
            records.append(_experiment_record(run, exp_dir))

    # Scrub every copied text file, then verify nothing slipped through.
    n_ids = 0
    for path in dest.rglob("*"):
        if path.is_file():
            n_ids += _scrub_file(path)
    _verify_no_prolific_ids(dest)

    summary_path = dest / "SUMMARY.md"
    summary_path.write_text(
        render_human_experiment_summary(records, source=str(source)), encoding="utf-8"
    )

    return HumanCollectionReport(
        source=source,
        dest=dest,
        runs=list(selected),
        experiment_paths=experiment_paths,
        summary_path=summary_path,
        n_ids_scrubbed=n_ids,
    )


def _verify_no_prolific_ids(dest: Path) -> None:
    """Fail loudly if any Prolific id survived into the destination tree."""
    leaked = []
    for path in dest.rglob("*"):
        if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES:
            if find_prolific_ids(path.read_text(encoding="utf-8", errors="ignore")):
                leaked.append(path)
    if leaked:
        listing = "\n  ".join(str(p) for p in leaked)
        raise RuntimeError(
            "Prolific ids survived redaction in:\n  "
            f"{listing}\nRefusing to leave PII in the repo."
        )


# --- summary rendering (pure) ----------------------------------------------


def summarize_model_posterior(posterior: dict) -> dict:
    """Distill a ``model_posterior.json`` payload into headline numbers."""
    posteriors = posterior["posteriors"]
    best_model = max(posteriors, key=posteriors.get)
    comparison = posterior.get("comparison", {})

    by_rank = sorted(comparison.items(), key=lambda kv: kv[1].get("rank", 1e9))
    top_elpd_model = by_rank[0][0] if by_rank else None
    runner_up = by_rank[1][0] if len(by_rank) > 1 else None
    runner_up_stats = by_rank[1][1] if len(by_rank) > 1 else {}

    return {
        "best_model": best_model,
        "best_posterior": posteriors[best_model],
        "n_models": len(posteriors),
        "n_trials": posterior.get("n_trials"),
        "top_elpd_model": top_elpd_model,
        "runner_up": runner_up,
        "runner_up_delta_elpd": runner_up_stats.get("elpd_diff"),
        "runner_up_dse": runner_up_stats.get("dse"),
    }


def _fmt(value: object, ndigits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, (int, float)):
        return f"{value:.{ndigits}f}"
    return str(value)


def render_human_experiment_summary(records: list[dict], source: str) -> str:
    """Render a compact Markdown summary of a live (human) outer-loop run."""
    runs = sorted({r["run"] for r in records})
    experiments = sorted({r["experiment"] for r in records})

    lines: list[str] = []
    lines.append("# Live (human) outer-loop — results summary")
    lines.append("")
    lines.append(f"- Source run root: `{source}`")
    lines.append(f"- Runs: {len(runs)} ({', '.join(runs)})")
    lines.append(f"- Experiments per run: {', '.join(experiments)}")
    lines.append("- Prolific worker/study ids scrubbed (no PII committed).")
    lines.append("")

    lines.append("## Winning model per (run, experiment)")
    lines.append("")
    lines.append(
        "| run | experiment | best model | P(best) | participants | trials | "
        "runner-up | Δelpd | dse |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in sorted(records, key=lambda x: (x["run"], x["experiment"])):
        if r.get("incomplete"):
            lines.append(
                f"| {r['run']} | {r['experiment']} | **INCOMPLETE** "
                "(no model_posterior.json) | n/a | "
                f"{r.get('n_participants', 'n/a')} | n/a | n/a | n/a | n/a |"
            )
            continue
        lines.append(
            f"| {r['run']} | {r['experiment']} | {r.get('best_model', 'n/a')} | "
            f"{_fmt(r.get('best_posterior'))} | {r.get('n_participants', 'n/a')} | "
            f"{r.get('n_trials', 'n/a')} | {r.get('runner_up', 'n/a')} | "
            f"{_fmt(r.get('runner_up_delta_elpd'), 2)} | {_fmt(r.get('runner_up_dse'), 2)} |"
        )
    lines.append("")

    # Cross-run agreement: did the runs converge on the same winner per experiment?
    lines.append("## Winning-model agreement across runs")
    lines.append("")
    for exp in experiments:
        winners = [
            r.get("best_model")
            for r in records
            if r["experiment"] == exp and not r.get("incomplete")
        ]
        unique = sorted(set(winners))
        agree = "yes" if len(unique) == 1 else "no"
        lines.append(f"- {exp}: agreement={agree} → {', '.join(unique) if unique else 'n/a'}")
    lines.append("")
    return "\n".join(lines)
