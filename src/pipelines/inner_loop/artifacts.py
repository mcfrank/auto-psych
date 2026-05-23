from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path

from src.pipelines.inner_loop.diagnostics import DiagnosticsBundle, write_population_jsonl
from src.pipelines.inner_loop.fitting import FitResult


def _write_json(path: Path, payload: dict | list) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _format_recent_rounds(history: list) -> list[str]:
    if not history:
        return ["No previous rounds yet."]
    lines = []
    for round_summary in history[-3:]:
        candidate_scores = ", ".join(
            (
                f"{candidate.candidate_id}: {candidate.log_likelihood:.4f}"
                if candidate.status == "ok"
                else f"{candidate.candidate_id}: failed"
            )
            for candidate in round_summary.candidate_summaries
        )
        if round_summary.winner_candidate_id is None:
            lines.append(f"- Round {round_summary.iteration}: no successful candidate; {candidate_scores}")
        else:
            lines.append(
                f"- Round {round_summary.iteration}: winner={round_summary.winner_candidate_id} "
                f"LL={round_summary.winner_log_likelihood:.4f}; candidates: {candidate_scores}"
            )
    return lines


def _format_worst_trials(diagnostics: DiagnosticsBundle | None, n_trials: int = 8) -> list[str]:
    if diagnostics is None or not diagnostics.metrics_rows:
        return ["No incumbent diagnostics available yet."]
    rows = sorted(diagnostics.metrics_rows, key=lambda row: row.log_likelihood)[:n_trials]
    return [
        (
            f"- trial_idx={row.trial_idx} stimulus={row.stimulus} response={row.response} "
            f"p={row.response_probability:.3f} LL={row.log_likelihood:.3f} "
            f"predicted={row.predicted_response}"
        )
        for row in rows
    ]


def _write_candidate_brief(candidate_dir: Path, candidate_idx: int, candidate_count: int) -> None:
    hints = [
        "Candidate 0: make the most evidence-backed refinement to the incumbent.",
        "Candidate 1: try a moderate structural variant if residuals suggest a missing feature.",
        "Candidate 2: try a higher-variance alternative or mixture model if progress has stalled.",
    ]
    lines = [
        "# Candidate Brief",
        "",
        f"You are generating candidate {candidate_idx} of {candidate_count}.",
        hints[candidate_idx % len(hints)],
        "Return one coherent parameterized choice model.",
    ]
    (candidate_dir / "CANDIDATE_BRIEF.md").write_text("\n".join(lines), encoding="utf-8")


def _write_top_mass_models(target_dir: Path, entries: list[dict]) -> None:
    if not entries:
        return
    lines = ["# Top-mass models", ""]
    for entry in entries:
        lines += [
            f"## {entry['entry_id']} posterior={entry['posterior']:.4f} "
            f"marginal_LL={entry['marginal_ll']:.3f} params={entry['n_params']}",
            "",
            "```python",
            entry["model_code"].strip(),
            "```",
            "",
        ]
    (target_dir / "top_mass_models.md").write_text("\n".join(lines), encoding="utf-8")


def _write_shared_round_artifacts(
    target_dir: Path,
    incumbent_model_code: str,
    history: list,
    incumbent_fit: FitResult | None,
    incumbent_diagnostics: DiagnosticsBundle | None,
    agent_timeout_sec: int,
    iteration: int,
    top_mass_models: list[dict] | None = None,
) -> None:
    from src.pipelines.inner_loop.history import _round_summary_to_dict

    lines = [
        "# Cognitive Model Search Round",
        "",
        f"Round: {iteration}",
        f"Candidate timeout seconds: {agent_timeout_sec}",
        "",
        "## Incumbent",
    ]
    if incumbent_fit is None:
        lines.append("No fitted incumbent yet. Start from the initial model.")
    else:
        aggregate = incumbent_diagnostics.aggregate if incumbent_diagnostics else None
        lines += [
            f"- Aggregate log-likelihood: {incumbent_fit.log_likelihood:.4f}",
            f"- Fitted params: {json.dumps([round(value, 4) for value in incumbent_fit.params])}",
        ]
        if aggregate is not None:
            lines += [
                f"- Accuracy: {aggregate.accuracy:.3f}",
                f"- Mean response probability: {aggregate.mean_response_probability:.3f}",
                f"- Mean prediction entropy: {aggregate.mean_prediction_entropy:.3f}",
            ]

    lines += [
        "",
        "## Recent History",
        *_format_recent_rounds(history),
        "",
        "## Worst-Fit Trials",
        *_format_worst_trials(incumbent_diagnostics),
        "",
        "## Output Contract",
        "- Write `cognitive_model.py` in this directory.",
        "- Define `PARAM_NAMES`, `PARAM_BOUNDS`, `INITIAL_PARAMS`.",
        "- Define `cognitive_model(stimulus, response_options, params=None) -> dict[str, float]`.",
        "- Probabilities must be finite and sum to 1.0.",
        "",
        "## Incumbent Model Code",
        "```python",
        incumbent_model_code.strip(),
        "```",
    ]
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / "CONTEXT.md").write_text("\n".join(lines), encoding="utf-8")
    (target_dir / "best_model.py").write_text(incumbent_model_code, encoding="utf-8")
    _write_json(
        target_dir / "history.json",
        [_round_summary_to_dict(round_summary) for round_summary in history],
    )
    if top_mass_models:
        _write_top_mass_models(target_dir, top_mass_models)
    if incumbent_fit is not None:
        _write_json(
            target_dir / "incumbent_fit.json",
            {
                "params": incumbent_fit.params,
                "log_likelihood": incumbent_fit.log_likelihood,
                "per_trial_ll": incumbent_fit.per_trial_ll.tolist(),
                "n_samples": incumbent_fit.n_samples,
                "n_trials": incumbent_fit.n_trials,
                "n_params": incumbent_fit.n_params,
            },
        )
    if incumbent_diagnostics is not None:
        _write_csv(
            target_dir / "incumbent_per_trial_metrics.csv",
            [asdict(row) for row in incumbent_diagnostics.metrics_rows],
        )
        write_population_jsonl(
            incumbent_diagnostics.population_rows,
            target_dir / "incumbent_trial_predictions.jsonl",
        )
