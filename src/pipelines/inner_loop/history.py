import json
from dataclasses import asdict, dataclass
from pathlib import Path

from src.pipelines.inner_loop.diagnostics import AggregateDiagnostics, TrialMetricsRow
from src.pipelines.inner_loop.fitting import FitResult


@dataclass
class CandidateSummary:
    candidate_id: str
    status: str
    log_likelihood: float | None = None
    params: list[float] | None = None
    aggregate: AggregateDiagnostics | None = None
    worst_trials: list[TrialMetricsRow] | None = None
    error: str | None = None


@dataclass
class RoundSummary:
    iteration: int
    winner_candidate_id: str | None
    winner_log_likelihood: float | None
    candidate_summaries: list[CandidateSummary]


def _fit_result_to_dict(fit_result: FitResult) -> dict:
    return {
        "params": fit_result.params,
        "log_likelihood": fit_result.log_likelihood,
        "per_trial_ll": fit_result.per_trial_ll.tolist(),
        "n_samples": fit_result.n_samples,
        "n_trials": fit_result.n_trials,
        "n_params": fit_result.n_params,
    }


def _round_summary_to_dict(round_summary: RoundSummary) -> dict:
    return asdict(round_summary)


def _round_summary_from_dict(payload: dict) -> RoundSummary:
    return RoundSummary(
        iteration=payload["iteration"],
        winner_candidate_id=payload["winner_candidate_id"],
        winner_log_likelihood=payload["winner_log_likelihood"],
        candidate_summaries=[
            CandidateSummary(
                candidate_id=candidate_payload["candidate_id"],
                status=candidate_payload["status"],
                log_likelihood=candidate_payload.get("log_likelihood"),
                params=candidate_payload.get("params"),
                aggregate=(
                    AggregateDiagnostics(**candidate_payload["aggregate"])
                    if candidate_payload.get("aggregate") is not None
                    else None
                ),
                worst_trials=[
                    TrialMetricsRow(**row_payload)
                    for row_payload in candidate_payload.get("worst_trials") or []
                ]
                or None,
                error=candidate_payload.get("error"),
            )
            for candidate_payload in payload["candidate_summaries"]
        ],
    )


def _load_fit_result(path: Path) -> FitResult:
    return FitResult(**json.loads(path.read_text()))


def _load_round_summary(round_dir: Path) -> RoundSummary:
    return _round_summary_from_dict(
        json.loads((round_dir / "fit_summary.json").read_text())
    )


def _round_winner(round_summary: RoundSummary) -> tuple[str, float]:
    winner = round_summary.winner_candidate_id
    if winner is None:
        raise ValueError("Round has no successful candidate.")
    winner_score = next(
        candidate.log_likelihood
        for candidate in round_summary.candidate_summaries
        if candidate.candidate_id == winner
    )
    return winner, winner_score


def _initial_history() -> list[RoundSummary]:
    return []


def _save_history(results_dir: Path, history: list[RoundSummary]) -> None:
    (results_dir / "history.json").write_text(
        json.dumps([_round_summary_to_dict(rs) for rs in history], indent=2)
    )


def _load_history(results_dir: Path) -> list[RoundSummary]:
    history_path = results_dir / "history.json"
    if not history_path.exists():
        return _initial_history()
    return [
        _round_summary_from_dict(payload)
        for payload in json.loads(history_path.read_text())
    ]
