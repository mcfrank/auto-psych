"""Data analyst: aggregate participant data (from CSV), summary stats, and model–data correlations. Python only."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import csv
import json
import yaml
from collections import defaultdict

from src.config import agent_dir, DEFAULT_MAX_VALIDATION_RETRIES
from src.console_log import agent_header, log_status
from src.observability import agent_log
from src.models.loader import get_model_names_from_manifest
from src.models.randomness import MODEL_LIBRARY, get_model_predictions

RESPONSE_OPTIONS = ["left", "right"]


def _pearson_r(x: List[float], y: List[float]) -> float:
    """Pearson correlation between two lists. Returns 0.0 if undefined."""
    n = len(x)
    if n != len(y) or n < 2:
        return 0.0
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    var_x = sum((v - mean_x) ** 2 for v in x)
    var_y = sum((v - mean_y) ** 2 for v in y)
    if var_x == 0 or var_y == 0:
        return 0.0
    cov = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
    return round(cov / (var_x * var_y) ** 0.5, 4)


def _compute_model_correlations(
    aggregate_lines: List[str],
    model_names: List[str],
    theorist_dir: Optional[Path],
) -> Dict[str, float]:
    """Pearson r between each model's predicted P(left) and observed chose_left_pct per stimulus."""
    observed = []
    stimuli = []
    for line in aggregate_lines[1:]:  # skip header
        parts = line.strip().split(",")
        if len(parts) >= 3:
            try:
                seq_a, seq_b, pct = parts[0], parts[1], float(parts[2])
                observed.append(pct)
                stimuli.append((seq_a, seq_b))
            except ValueError:
                continue
    if not stimuli or not model_names:
        return {m: 0.0 for m in model_names}
    model_predictions: Dict[str, List[float]] = {m: [] for m in model_names}
    for stim in stimuli:
        preds = get_model_predictions(stim, RESPONSE_OPTIONS, model_names, theorist_dir)
        for m in model_names:
            model_predictions[m].append(preds.get(m, {}).get("left", 0.5))
    return {m: _pearson_r(model_predictions[m], observed) for m in model_names}


def run_data_analyst(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Read participant data from the path provided by the collect step (CSV).
    Aggregate by stimulus, compute summary statistics; write aggregate.csv and summary_stats.json.
    """
    project_id = state["project_id"]
    run_id = state["run_id"]
    # Reset validation retry state when this agent is entered from a different agent's validation
    if state.get("last_validated_agent") != "5_analyze":
        state = {**state, "validation_retry_count": 0, "validation_feedback": ""}
    if state.get("validation_retry_count", 0) == 0:
        agent_header("5_analyze", run_id, state.get("total_runs"), state.get("mode"))
    elif state.get("validation_retry_count", 0) > 0:
        max_r = state.get("max_validation_retries", DEFAULT_MAX_VALIDATION_RETRIES)
        log_status(f"Repeating due to validation failure (attempt {state['validation_retry_count']}/{max_r})")
    out_dir = agent_dir(project_id, run_id, "5_analyze")
    out_dir.mkdir(parents=True, exist_ok=True)
    attempt = (state.get("validation_retry_count") or 0) + 1
    validation_feedback = (state.get("validation_feedback") or "").strip()
    agent_log(out_dir, "=== 5_analyze start ===")
    agent_log(out_dir, f"project_id={project_id!r} run_id={run_id} attempt={attempt}")
    if validation_feedback:
        agent_log(out_dir, f"Validation feedback: {validation_feedback[:500]}")

    data_path = state.get("simulated_data_path")
    if data_path and Path(data_path).exists():
        agent_log(out_dir, f"data_path={data_path!r}")
        aggregate, summary = _aggregate_csv(Path(data_path))
        (out_dir / "aggregate.csv").write_text(aggregate, encoding="utf-8")
        (out_dir / "summary_stats.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        # Model–data correlations: Pearson r per model (predicted P(left) vs observed proportion per stimulus)
        manifest_path = Path(state.get("theorist_manifest_path", ""))
        theorist_dir = manifest_path.parent if manifest_path else None
        model_names = []
        if manifest_path and manifest_path.exists():
            manifest = yaml.safe_load(manifest_path.read_text()) or {}
            model_names = get_model_names_from_manifest(manifest, theorist_dir)
        if not model_names:
            model_names = list(MODEL_LIBRARY.keys())
        agg_lines = aggregate.strip().split("\n")
        if agg_lines:
            correlations = _compute_model_correlations(agg_lines, model_names, theorist_dir)
            (out_dir / "model_correlations.yaml").write_text(
                yaml.dump({"correlations": correlations}, default_flow_style=False, sort_keys=False),
                encoding="utf-8",
            )
        agent_log(out_dir, f"wrote aggregate.csv, summary_stats.json, model_correlations.yaml (n_stimuli={summary.get('n_stimuli', 0)})")
    else:
        agent_log(out_dir, f"no data path or file missing: simulated_data_path={data_path!r}")
        (out_dir / "aggregate.csv").write_text("sequence_a,sequence_b,chose_left_pct,n\n", encoding="utf-8")
        (out_dir / "summary_stats.json").write_text(
            json.dumps({"n_stimuli": 0, "n_responses_total": 0, "mean_chose_left": 0.0, "note": "No data path or file missing"}),
            encoding="utf-8",
        )
    agent_log(out_dir, "=== 5_analyze end ===")

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
