"""Interpreter agent: compare theory to data and write plain-language report."""

from pathlib import Path
from typing import Any, Dict, List, Tuple
import json
import yaml
from collections import defaultdict

from src.config import agent_dir, run_dir, DEFAULT_MAX_VALIDATION_RETRIES
from src.agents.base import load_prompt_for_run, invoke_llm
from src.console_log import agent_header, log_status
from src.observability import agent_log, write_transcript
from src.agents.llm_output_parsing import extract_yaml_from_response, ensure_str
from src.models.loader import get_model_names_from_manifest
from src.models.randomness import MODEL_LIBRARY, get_model_predictions
from src.registry import (
    DEFAULT_RESERVED_FOR_NEW,
    write_registry,
)

RESPONSE_OPTIONS = ["left", "right"]


def _pearson_r(x: List[float], y: List[float]) -> float:
    """Pearson correlation between two lists. Returns 0.0 if undefined (e.g. constant input)."""
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
    return cov / (var_x * var_y) ** 0.5


def _correlations_from_aggregate_and_predictions(
    model_predictions: Dict[str, List[float]], aggregate_lines: List[str]
) -> Dict[str, float]:
    """Compute Pearson r between each model's predicted P(left) and observed chose_left_pct per stimulus."""
    # aggregate_lines[0] is header; rest are "seq_a,seq_b,pct,n"
    observed = []
    for line in aggregate_lines[1:]:
        parts = line.strip().split(",")
        if len(parts) >= 3:
            try:
                observed.append(float(parts[2]))
            except ValueError:
                continue
    if len(observed) == 0:
        return {m: 0.0 for m in model_predictions}
    correlations = {}
    for model, preds in model_predictions.items():
        if len(preds) != len(observed):
            preds = preds[: len(observed)] if len(preds) > len(observed) else preds + [0.5] * (len(observed) - len(preds))
        correlations[model] = round(_pearson_r(preds, observed), 4)
    return correlations


def _format_correlations_for_prompt(
    model_predictions: Dict[str, List[float]], aggregate_lines: List[str]
) -> str:
    """Format model–data correlations as readable text for the LLM prompt."""
    corr = _correlations_from_aggregate_and_predictions(model_predictions, aggregate_lines)
    return "\n".join(f"- {m}: {r}" for m, r in sorted(corr.items()))


def _analyst_correlations_section(project_id: str, run_id: int) -> str:
    """If the analyze step wrote model_correlations.yaml for this run, include it in the prompt."""
    corr_path = run_dir(project_id, run_id) / "5_analyze" / "model_correlations.yaml"
    if not corr_path.exists():
        return ""
    try:
        data = yaml.safe_load(corr_path.read_text()) or {}
        corr = data.get("correlations") or {}
        if not corr:
            return ""
        lines = ["\n## Correlations from analyze step (this run)", "\n".join(f"- {m}: {r}" for m, r in sorted(corr.items()))]
        return "\n".join(lines)
    except Exception:
        return ""


def _merge_aggregates_and_summary(project_id: str, run_id: int) -> Tuple[str, List[str], dict]:
    """
    Merge aggregate.csv and summary_stats from runs 1..run_id.
    Returns (merged_aggregate_csv_string, aggregate_lines for prompt, merged_summary_dict).
    """
    key_to_pct_n: Dict[Tuple[str, str], List[Tuple[float, int]]] = defaultdict(list)
    total_responses = 0
    total_left = 0.0
    for r in range(1, run_id + 1):
        agg_path = run_dir(project_id, r) / "5_analyze" / "aggregate.csv"
        if not agg_path.exists():
            continue
        lines = agg_path.read_text(encoding="utf-8").strip().split("\n")
        if not lines:
            continue
        header = lines[0]
        for line in lines[1:]:
            parts = line.split(",")
            if len(parts) >= 4:
                seq_a, seq_b = parts[0], parts[1]
                try:
                    pct = float(parts[2])
                    n = int(parts[3])
                except (ValueError, IndexError):
                    continue
                key_to_pct_n[(seq_a, seq_b)].append((pct, n))
                total_responses += n
                total_left += pct * n
    # Build merged aggregate: one row per (seq_a, seq_b) with combined n and weighted pct
    out_lines = ["sequence_a,sequence_b,chose_left_pct,n\n"]
    for (seq_a, seq_b), pairs in sorted(key_to_pct_n.items()):
        total_n = sum(n for _, n in pairs)
        if total_n == 0:
            continue
        weighted_pct = sum(pct * n for pct, n in pairs) / total_n
        out_lines.append(f"{seq_a},{seq_b},{weighted_pct:.4f},{total_n}\n")
    merged_csv = "".join(out_lines)
    aggregate_lines = out_lines[0].strip().split("\n") + [l.strip() for l in out_lines[1:]]
    summary = {
        "n_stimuli": len(key_to_pct_n),
        "n_responses_total": total_responses,
        "mean_chose_left": total_left / total_responses if total_responses else 0,
        "runs_merged": run_id,
    }
    return merged_csv, aggregate_lines, summary


def _fallback_report(summary: dict, aggregate_lines: list, model_names: list) -> str:
    """When LLM is unavailable, write a short data summary."""
    return f"""# Experiment report (template)

## Summary statistics
- n_stimuli: {summary.get('n_stimuli', 'N/A')}
- n_responses_total: {summary.get('n_responses_total', 'N/A')}
- mean_chose_left: {summary.get('mean_chose_left', 'N/A')}

## Models compared
{chr(10).join('- ' + m for m in model_names)}

## Aggregate data (sample)
{chr(10).join(aggregate_lines[:5]) if aggregate_lines else 'No data'}

Run with GOOGLE_API_KEY set for an LLM-generated interpretation.
"""


def run_interpreter(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Read summary stats and aggregate CSV (merged over runs 1..run_id); compare to model predictions;
    write plain-language report to 6_interpret/report.md.
    """
    project_id = state["project_id"]
    run_id = state["run_id"]
    # Reset validation retry state when this agent is entered from a different agent's validation
    if state.get("last_validated_agent") != "6_interpret":
        state = {**state, "validation_retry_count": 0, "validation_feedback": ""}
    if state.get("validation_retry_count", 0) == 0:
        agent_header("6_interpret", run_id, state.get("total_runs"), state.get("mode"))
    elif state.get("validation_retry_count", 0) > 0:
        max_r = state.get("max_validation_retries", DEFAULT_MAX_VALIDATION_RETRIES)
        log_status(f"Repeating due to validation failure (attempt {state['validation_retry_count']}/{max_r})")
    out_dir = agent_dir(project_id, run_id, "6_interpret")
    out_dir.mkdir(parents=True, exist_ok=True)
    attempt = (state.get("validation_retry_count") or 0) + 1
    validation_feedback = (state.get("validation_feedback") or "").strip()
    agent_log(out_dir, "=== 6_interpret start ===")
    agent_log(out_dir, f"project_id={project_id!r} run_id={run_id} attempt={attempt}")
    if validation_feedback:
        agent_log(out_dir, f"Validation feedback: {validation_feedback[:500]}")

    # Merge data from runs 1..run_id so interpreter sees all data
    merged_csv, aggregate_lines, summary = _merge_aggregates_and_summary(project_id, run_id)
    merged_aggregate_path = out_dir / "aggregate_merged.csv"
    merged_aggregate_path.write_text(merged_csv, encoding="utf-8")
    merged_summary_path = out_dir / "summary_merged.json"
    merged_summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    manifest_path = Path(state.get("theorist_manifest_path", ""))
    registry_path = Path(state.get("registry_path", ""))
    # Prefer current run's registry for theory list and probabilities; fall back to manifest
    model_names: List[str] = []
    theorist_dir = manifest_path.parent if manifest_path else None
    if registry_path and registry_path.exists():
        reg = yaml.safe_load(registry_path.read_text()) or {}
        theories = reg.get("theories") or reg.get("probabilities") or {}
        if isinstance(theories, dict):
            model_names = [k for k in theories.keys() if k != "reserved_for_new"]
        else:
            model_names = []
    if not model_names and manifest_path and manifest_path.exists():
        manifest = yaml.safe_load(manifest_path.read_text()) or {}
        model_names = get_model_names_from_manifest(manifest, theorist_dir)
    if not model_names:
        model_names = list(MODEL_LIBRARY.keys())

    # Compute model predictions for each stimulus in aggregate
    model_predictions = {}
    for line in aggregate_lines[1:]:  # skip header
        parts = line.split(",")
        if len(parts) >= 3:
            seq_a, seq_b = parts[0], parts[1]
            stim = (seq_a, seq_b)
            preds = get_model_predictions(stim, RESPONSE_OPTIONS, model_names, theorist_dir)
            for m, probs in preds.items():
                model_predictions.setdefault(m, []).append(probs.get("left", 0.5))

    # Pearson correlation per model (predicted P(left) vs observed proportion chose left)
    correlations = _correlations_from_aggregate_and_predictions(model_predictions, aggregate_lines)
    (out_dir / "model_correlations.yaml").write_text(
        yaml.dump({"correlations": correlations}, default_flow_style=False, sort_keys=False),
        encoding="utf-8",
    )

    prompt = load_prompt_for_run(project_id, run_id, "6_interpret")
    validation_feedback = (state.get("validation_feedback") or "").strip()
    user_content = ""
    if validation_feedback:
        user_content += f"""## Validation feedback (previous attempt failed)

{validation_feedback}

Please fix the report so report.md is non-empty and passes validation.

"""
    user_content += f"""## Run context

This is **Run {run_id}** of the pipeline. Your report and the theory probabilities (YAML block) will be used by the **theorist in Run {run_id + 1}** to update the model set and design the next experiment. Write with that audience in mind.

## Summary statistics (merged from runs 1..{run_id})

{json.dumps(summary, indent=2)}

## Aggregate data (first 15 lines)

{chr(10).join(aggregate_lines[:15])}

## Models used

{chr(10).join('- ' + m for m in model_names)}

## Model predictions (mean P(left) per stimulus, for each model)

{json.dumps({m: (sum(v)/len(v) if v else 0) for m, v in model_predictions.items()}, indent=2)}

## Model–data correlations (Pearson r: predicted P(left) vs observed proportion chose left, per stimulus)

{_format_correlations_for_prompt(model_predictions, aggregate_lines)}
{_analyst_correlations_section(project_id, run_id)}

Write the report in **formatted Markdown** (use headers, bullet lists, bold/italic). Do not output JSON for the report body. Write a short report (2–4 paragraphs) that:
1. Summarizes what was tested (subjective randomness: which sequence looks more random).
2. Describes the data (e.g. mean proportion chose left, number of stimuli and responses; data spans runs 1–{run_id}).
3. Compares the data to the model predictions and states which model(s) fit best or worst.
4. Suggests what could be done next (e.g. run more participants, try different stimuli, or revise models).

Then output a YAML block with your updated probability distribution over the theories. Use this exact format (replace model names and values):

---BEGIN THEORY PROBABILITIES---
probabilities:
  model_a: 0.45
  model_b: 0.30
reserved_for_new: 0.25
---END THEORY PROBABILITIES---

The probabilities should sum to (1 - reserved_for_new). Reserve 0.25 for new theories the theorist may add. Use the model names listed under "Models used" above.
"""

    agent_log(out_dir, "invoking LLM (interpret)...")
    try:
        full_response = invoke_llm(system=prompt, user=user_content)
    except Exception as e:
        agent_log(out_dir, f"LLM invoke error: {type(e).__name__}: {e}")
        full_response = _fallback_report(summary, aggregate_lines, model_names)
    full_response = ensure_str(full_response)
    agent_log(out_dir, f"LLM response length={len(full_response)} chars")
    write_transcript(
        out_dir, attempt,
        system=prompt, user=user_content, response=full_response[:100_000],
        validation_feedback=validation_feedback,
    )
    # Strip out the YAML block for the report body
    report = full_response
    if "---BEGIN THEORY PROBABILITIES---" in full_response and "---END THEORY PROBABILITIES---" in full_response:
        report = full_response.split("---BEGIN THEORY PROBABILITIES---")[0].strip()
    (out_dir / "report.md").write_text(report, encoding="utf-8")

    # Parse theory probabilities and update this run's registry
    prob_data = extract_yaml_from_response(
        full_response,
        begin_marker="---BEGIN THEORY PROBABILITIES---",
        end_marker="---END THEORY PROBABILITIES---",
    )
    registry_path = Path(state.get("registry_path", ""))
    if registry_path and prob_data:
        probs = prob_data.get("probabilities") or prob_data.get("theories") or {}
        if isinstance(probs, dict):
            reserved = prob_data.get("reserved_for_new", DEFAULT_RESERVED_FOR_NEW)
            try:
                reserved = float(reserved)
            except (TypeError, ValueError):
                reserved = DEFAULT_RESERVED_FOR_NEW
            # Normalize so sum(probs) + reserved = 1
            total_p = sum(v for k, v in probs.items() if isinstance(v, (int, float)))
            if total_p > 0:
                target = max(0.0, 1.0 - reserved)
                probs = {k: (float(v) / total_p) * target for k, v in probs.items() if isinstance(v, (int, float))}
            (out_dir / "theory_probabilities.yaml").write_text(
                yaml.dump({"probabilities": probs, "reserved_for_new": reserved}, default_flow_style=False),
                encoding="utf-8",
            )
            write_registry(registry_path, probs, reserved_for_new=reserved)

    agent_log(out_dir, "wrote report.md, theory_probabilities.yaml")
    agent_log(out_dir, "=== 6_interpret end ===")
    return {
        **state,
        "interpreter_report_path": str(out_dir / "report.md"),
    }
