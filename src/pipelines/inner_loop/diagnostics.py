from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from src.pipelines.inner_loop.core import Dataset, ModelCallable
from src.pipelines.inner_loop.likelihood import predict


@dataclass
class TrialMetricsRow:
    candidate_id: str
    trial_idx: int
    stimulus: object
    response: str
    response_probability: float
    log_likelihood: float
    prediction_entropy: float
    predicted_response: str
    correct_prediction: bool


@dataclass
class AggregateDiagnostics:
    candidate_id: str
    n_trials: int
    aggregate_log_likelihood: float
    mean_log_likelihood: float
    mean_response_probability: float
    mean_prediction_entropy: float
    accuracy: float


@dataclass
class DiagnosticsBundle:
    metrics_rows: list[TrialMetricsRow]
    population_rows: list[dict]
    aggregate: AggregateDiagnostics


GraphMetricsRow = TrialMetricsRow
GraphPopulationRow = dict
ProblemDiagnosticsRow = dict


def _entropy(probs: dict[str, float]) -> float:
    values = np.asarray(list(probs.values()), dtype=float)
    values = values[values > 0]
    return float(-(values * np.log(values)).sum())


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def build_candidate_diagnostics(
    candidate_id: str,
    data: Dataset,
    model: ModelCallable,
    params: list[float] | None = None,
) -> DiagnosticsBundle:
    metrics_rows: list[TrialMetricsRow] = []
    population_rows: list[dict] = []

    for trial_idx, trial in enumerate(data.trials):
        probs = predict(model, trial.stimulus, data.response_options, params)
        p_response = max(1e-9, min(1.0 - 1e-9, probs.get(trial.response, 0.0)))
        predicted_response = max(probs, key=probs.get)
        ll = float(np.log(p_response))
        metrics_rows.append(
            TrialMetricsRow(
                candidate_id=candidate_id,
                trial_idx=trial_idx,
                stimulus=trial.stimulus,
                response=trial.response,
                response_probability=p_response,
                log_likelihood=ll,
                prediction_entropy=_entropy(probs),
                predicted_response=predicted_response,
                correct_prediction=predicted_response == trial.response,
            )
        )
        population_rows.append(
            {
                "candidate_id": candidate_id,
                "trial_idx": trial_idx,
                "stimulus": trial.stimulus,
                "response": trial.response,
                "probabilities": probs,
                "metadata": dict(trial.metadata),
            }
        )

    aggregate = AggregateDiagnostics(
        candidate_id=candidate_id,
        n_trials=len(metrics_rows),
        aggregate_log_likelihood=float(sum(row.log_likelihood for row in metrics_rows)),
        mean_log_likelihood=_mean([row.log_likelihood for row in metrics_rows]),
        mean_response_probability=_mean([row.response_probability for row in metrics_rows]),
        mean_prediction_entropy=_mean([row.prediction_entropy for row in metrics_rows]),
        accuracy=_mean([1.0 if row.correct_prediction else 0.0 for row in metrics_rows]),
    )
    return DiagnosticsBundle(metrics_rows=metrics_rows, population_rows=population_rows, aggregate=aggregate)


def write_population_jsonl(rows: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def write_diagnostics_json(bundle: DiagnosticsBundle, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "per_trial_metrics.json").write_text(
        json.dumps([asdict(row) for row in bundle.metrics_rows], indent=2),
        encoding="utf-8",
    )
    (output_dir / "summary_stats.json").write_text(
        json.dumps(asdict(bundle.aggregate), indent=2),
        encoding="utf-8",
    )
