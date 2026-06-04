"""Batch-level correlation CSV, plot (correlations by run), and runs summary in batch root."""

import csv
import json
from pathlib import Path
from typing import Dict, List


def write_batch_runs_summary(batch_dir: Path) -> Path:
    """Write batch_dir/runs_summary.json with run_ids and n_runs (from run* subdirs). Return path."""
    run_ids: List[int] = []
    for d in sorted(batch_dir.iterdir()):
        if d.is_dir() and d.name.startswith("run") and d.name[3:].isdigit():
            try:
                run_ids.append(int(d.name[3:]))
            except ValueError:
                continue
    run_ids.sort()
    summary = {"run_ids": run_ids, "n_runs": len(run_ids)}
    path = batch_dir / "runs_summary.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return path


def append_correlations_to_batch_csv(batch_dir: Path, run_id: int, correlations: Dict[str, float]) -> None:
    """Append one row (run_id + model correlations) to batch_dir/correlations.csv."""
    csv_path = batch_dir / "correlations.csv"
    new_row = {"run_id": run_id, **correlations}

    existing_rows: List[Dict[str, str]] = []
    existing_columns: List[str] = ["run_id"]
    if csv_path.exists():
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            existing_columns = list(reader.fieldnames or ["run_id"])
            existing_rows = [dict(row) for row in reader]

    all_models = sorted(set(existing_columns) - {"run_id"} | set(correlations.keys()))
    columns = ["run_id"] + all_models

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in existing_rows:
            writer.writerow({c: row.get(c, "") for c in columns})
        writer.writerow({c: new_row.get(c, "") for c in columns})


def plot_correlations_by_run(batch_dir: Path) -> Path:
    """Generate correlations_by_run.png from batch_dir/correlations.csv. Always writes a PNG (placeholder if no data). Returns path."""
    out_path = batch_dir / "correlations_by_run.png"
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return out_path

    csv_path = batch_dir / "correlations.csv"
    run_ids: List[int] = []
    model_names: List[str] = []
    data: Dict[str, List[float]] = {}

    if csv_path.exists():
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            model_names = [c for c in fieldnames if c != "run_id"]
            for row in reader:
                try:
                    run_ids.append(int(row.get("run_id", 0)))
                except (ValueError, TypeError):
                    continue
                for m in model_names:
                    data.setdefault(m, [])
                    try:
                        data[m].append(float(row.get(m, 0)))
                    except (ValueError, TypeError):
                        data[m].append(0.0)

    fig, ax = plt.subplots()
    if run_ids and model_names:
        for m in model_names:
            if m in data and len(data[m]) == len(run_ids):
                ax.plot(run_ids, data[m], marker="o", label=m)
        ax.set_xlabel("Run")
        ax.set_ylabel("Correlation (r)")
        ax.set_title("Model–data correlation by run (merged aggregate)")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No correlation data yet", ha="center", va="center", transform=ax.transAxes)
        ax.set_xlabel("Run")
        ax.set_ylabel("Correlation (r)")
        ax.set_title("Model–data correlation by run (merged aggregate)")
    fig.savefig(out_path, dpi=120)
    plt.close()
    return out_path
