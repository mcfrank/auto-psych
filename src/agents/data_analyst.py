"""Data analyst: aggregate participant data (from CSV) and produce summary stats. Python only."""

from pathlib import Path
from typing import Any, Dict
import csv
import json
from collections import defaultdict

from src.config import agent_dir
from src.console_log import agent_header, log_status


def run_data_analyst(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Read participant data from the path provided by the collect step (CSV).
    Aggregate by stimulus, compute summary statistics; write aggregate.csv and summary_stats.json.
    """
    project_id = state["project_id"]
    run_id = state["run_id"]
    agent_header("5_analyze", run_id, state.get("total_runs"), state.get("mode"))
    if state.get("validation_retry_count", 0) > 0:
        log_status(f"Repeating due to validation failure (attempt {state['validation_retry_count']}/3)")
    out_dir = agent_dir(project_id, run_id, "5_analyze")
    out_dir.mkdir(parents=True, exist_ok=True)

    data_path = state.get("simulated_data_path")
    if data_path and Path(data_path).exists():
        aggregate, summary = _aggregate_csv(Path(data_path))
        (out_dir / "aggregate.csv").write_text(aggregate, encoding="utf-8")
        (out_dir / "summary_stats.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    else:
        (out_dir / "aggregate.csv").write_text("sequence_a,sequence_b,chose_left_pct,n\n", encoding="utf-8")
        (out_dir / "summary_stats.json").write_text(
            json.dumps({"n_stimuli": 0, "n_responses_total": 0, "mean_chose_left": 0.0, "note": "No data path or file missing"}),
            encoding="utf-8",
        )

    return {
        **state,
        "summary_stats_path": str(out_dir / "summary_stats.json"),
        "aggregate_csv_path": str(out_dir / "aggregate.csv"),
    }


def _aggregate_csv(csv_path: Path) -> tuple[str, dict]:
    """Aggregate by (sequence_a, sequence_b): proportion chose_left, n."""
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    key_to_left: Dict[tuple, list] = defaultdict(list)
    for r in rows:
        key = (r.get("sequence_a", ""), r.get("sequence_b", ""))
        key_to_left[key].append(int(r.get("chose_left", 0)))
    lines = ["sequence_a,sequence_b,chose_left_pct,n\n"]
    for (sa, sb), lefts in sorted(key_to_left.items()):
        pct = sum(lefts) / len(lefts) if lefts else 0
        lines.append(f"{sa},{sb},{pct:.4f},{len(lefts)}\n")
    aggregate = "".join(lines)
    summary = {
        "n_stimuli": len(key_to_left),
        "n_responses_total": len(rows),
        "mean_chose_left": sum(int(r.get("chose_left", 0)) for r in rows) / len(rows) if rows else 0.0,
    }
    return aggregate, summary
