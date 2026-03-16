"""Batch-level correlation CSV and plot (correlations by run)."""

import csv
from pathlib import Path
from typing import Dict, List


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
    """Generate correlations_by_run.png from batch_dir/correlations.csv. Returns path to plot."""
    csv_path = batch_dir / "correlations.csv"
    if not csv_path.exists():
        return batch_dir / "correlations_by_run.png"

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return batch_dir / "correlations_by_run.png"

    run_ids: List[int] = []
    model_names: List[str] = []
    data: Dict[str, List[float]] = {}

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

    if not run_ids or not model_names:
        return batch_dir / "correlations_by_run.png"

    fig, ax = plt.subplots()
    for m in model_names:
        if m in data and len(data[m]) == len(run_ids):
            ax.plot(run_ids, data[m], marker="o", label=m)
    ax.set_xlabel("Run")
    ax.set_ylabel("Correlation (r)")
    ax.set_title("Model–data correlation by run (merged aggregate)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    out_path = batch_dir / "correlations_by_run.png"
    fig.savefig(out_path, dpi=120)
    plt.close()
    return out_path
