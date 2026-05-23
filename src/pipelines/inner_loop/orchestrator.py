from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from src.pipelines.inner_loop.agent import _spawn_agent, _validate_model
from src.pipelines.inner_loop.artifacts import (
    _write_candidate_brief,
    _write_csv,
    _write_json,
    _write_shared_round_artifacts,
)
from src.pipelines.inner_loop.core import Dataset, Likelihood, Sampler
from src.pipelines.inner_loop.diagnostics import build_candidate_diagnostics, write_diagnostics_json
from src.pipelines.inner_loop.fitting import FitResult, fit_model
from src.pipelines.inner_loop.history import (
    CandidateSummary,
    RoundSummary,
    _fit_result_to_dict,
    _load_fit_result,
    _load_history,
    _load_round_summary,
    _round_summary_to_dict,
    _save_history,
)
from src.pipelines.inner_loop.initial_model import INITIAL_PARAMS, PARAM_BOUNDS, cognitive_model
from src.pipelines.inner_loop.zoo import candidate_entry_id, get_entry, record_candidate, record_initial, zoo_size
from src.pipelines.inner_loop.bmc import compute_bmc
from src.pipelines.inner_loop.likelihood import CategoricalLikelihood, CategoricalSampler


def _top_mass_digest(zoo_dir: Path, bmc: dict) -> list[dict]:
    out = []
    for entry_id in bmc["top_mass_set"]:
        entry = get_entry(zoo_dir, entry_id)
        if entry:
            out.append(
                {
                    "entry_id": entry_id,
                    "posterior": bmc["posteriors"][entry_id],
                    "marginal_ll": bmc["marginal_log_likelihoods"][entry_id],
                    "n_params": bmc["n_params"][entry_id],
                    "model_code": entry.model_path.read_text(encoding="utf-8"),
                }
            )
    return out


def _best_from_bmc(zoo_dir: Path, bmc: dict | None = None) -> tuple[str, str, FitResult, dict]:
    bmc = bmc or compute_bmc(zoo_dir)
    entry_id = bmc["ranking"][0]
    entry = get_entry(zoo_dir, entry_id)
    if entry is None:
        raise RuntimeError(f"BMC selected missing zoo entry {entry_id!r}")
    return entry_id, entry.model_path.read_text(encoding="utf-8"), entry.load_fit(), bmc


def _fit_and_record(
    model_path: Path,
    data: Dataset,
    likelihood: Likelihood,
    sampler: Sampler | None,
    *,
    n_samples: int,
    n_starts: int,
    max_fit_steps: int,
    base_seed: int,
) -> tuple[FitResult, object]:
    model, bounds = _validate_model(model_path)
    fit = fit_model(
        model,
        data,
        likelihood=likelihood,
        sampler=sampler,
        n_samples=n_samples,
        n_starts=n_starts,
        max_steps=max_fit_steps,
        initial_params=model.__globals__.get("INITIAL_PARAMS"),
        param_bounds=bounds,
        base_seed=base_seed,
    )
    return fit, model


def run_pipeline(
    data: Dataset,
    results_dir: Path,
    *,
    max_iterations: int = 5,
    candidate_count: int = 3,
    likelihood: Likelihood | None = None,
    sampler: Sampler | None = None,
    n_samples: int = 50,
    n_starts: int = 3,
    max_fit_steps: int = 100,
    overwrite: bool = False,
    agent_timeout_sec: int = 900,
    base_seed: int = 0,
    api_key: str | None = None,
) -> FitResult:
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    zoo_dir = results_dir / "model_zoo"
    likelihood = likelihood or CategoricalLikelihood()
    sampler = sampler or CategoricalSampler()

    history = _load_history(results_dir)
    best_model_path = results_dir / "best_model.py"
    best_fit_path = results_dir / "best_fit.json"
    incumbent_code = (Path(__file__).parent / "initial_model.py").read_text(encoding="utf-8")
    incumbent_fit = None
    incumbent_diagnostics = None

    if best_model_path.exists() and best_fit_path.exists():
        incumbent_code = best_model_path.read_text(encoding="utf-8")
        incumbent_fit = _load_fit_result(best_fit_path)
        model, _ = _validate_model(best_model_path)
        incumbent_diagnostics = build_candidate_diagnostics("incumbent", data, model, incumbent_fit.params)
        record_initial(zoo_dir, best_model_path, incumbent_fit)
    else:
        initial_fit = fit_model(
            cognitive_model,
            data,
            likelihood=likelihood,
            sampler=sampler,
            n_samples=n_samples,
            n_starts=n_starts,
            max_steps=max_fit_steps,
            initial_params=INITIAL_PARAMS,
            param_bounds=PARAM_BOUNDS,
            base_seed=base_seed,
        )
        best_model_path.write_text(incumbent_code, encoding="utf-8")
        _write_json(best_fit_path, _fit_result_to_dict(initial_fit))
        record_initial(zoo_dir, best_model_path, initial_fit)
        incumbent_fit = initial_fit
        incumbent_diagnostics = build_candidate_diagnostics("incumbent", data, cognitive_model, initial_fit.params)

    for iteration in range(max_iterations):
        round_dir = results_dir / f"iter_{iteration}"
        round_dir.mkdir(parents=True, exist_ok=True)
        if not overwrite and (round_dir / "fit_summary.json").exists():
            history.append(_load_round_summary(round_dir))
            continue

        top_mass = _top_mass_digest(zoo_dir, compute_bmc(zoo_dir)) if zoo_size(zoo_dir) else None
        _write_shared_round_artifacts(
            round_dir,
            incumbent_code,
            history,
            incumbent_fit,
            incumbent_diagnostics,
            agent_timeout_sec,
            iteration,
            top_mass_models=top_mass,
        )

        summaries = []
        for idx in range(candidate_count):
            candidate_id = f"candidate_{idx}"
            candidate_dir = round_dir / candidate_id
            candidate_dir.mkdir(parents=True, exist_ok=True)
            _write_shared_round_artifacts(
                candidate_dir,
                incumbent_code,
                history,
                incumbent_fit,
                incumbent_diagnostics,
                agent_timeout_sec,
                iteration,
                top_mass_models=top_mass,
            )
            _write_candidate_brief(candidate_dir, idx, candidate_count)
            try:
                _spawn_agent(candidate_dir, agent_timeout_sec, api_key)
                fit, model = _fit_and_record(
                    candidate_dir / "cognitive_model.py",
                    data,
                    likelihood,
                    sampler,
                    n_samples=n_samples,
                    n_starts=n_starts,
                    max_fit_steps=max_fit_steps,
                    base_seed=base_seed,
                )
                diagnostics = build_candidate_diagnostics(candidate_id, data, model, fit.params)
                _write_json(candidate_dir / "fit_result.json", _fit_result_to_dict(fit))
                _write_csv(candidate_dir / "per_trial_metrics.csv", [asdict(r) for r in diagnostics.metrics_rows])
                write_diagnostics_json(diagnostics, candidate_dir)
                record_candidate(zoo_dir, iteration, candidate_id, candidate_dir / "cognitive_model.py", fit)
                summaries.append(
                    CandidateSummary(
                        candidate_id=candidate_id,
                        status="ok",
                        log_likelihood=fit.log_likelihood,
                        params=fit.params,
                        aggregate=diagnostics.aggregate,
                        worst_trials=sorted(diagnostics.metrics_rows, key=lambda r: r.log_likelihood)[:5],
                    )
                )
            except Exception as exc:
                summaries.append(CandidateSummary(candidate_id=candidate_id, status="failed", error=str(exc)))

        successful = [s for s in summaries if s.status == "ok"]
        bmc = compute_bmc(zoo_dir) if successful else None
        winner = None
        if bmc:
            winner = max(successful, key=lambda s: bmc["posteriors"].get(candidate_entry_id(iteration, s.candidate_id), -1))
            best_id, incumbent_code, incumbent_fit, bmc = _best_from_bmc(zoo_dir, bmc)
            incumbent_fit._zoo_entry_id = best_id
            best_model_path.write_text(incumbent_code, encoding="utf-8")
            _write_json(best_fit_path, _fit_result_to_dict(incumbent_fit))
            _write_json(round_dir / "model_posterior.json", bmc)
            _write_json(results_dir / "model_posterior.json", bmc)
            model, _ = _validate_model(best_model_path)
            incumbent_diagnostics = build_candidate_diagnostics("incumbent", data, model, incumbent_fit.params)

        summary = RoundSummary(
            iteration=iteration,
            winner_candidate_id=winner.candidate_id if winner else None,
            winner_log_likelihood=winner.log_likelihood if winner else None,
            candidate_summaries=summaries,
        )
        _write_json(round_dir / "fit_summary.json", _round_summary_to_dict(summary))
        history.append(summary)
        _save_history(results_dir, history)

    return incumbent_fit


_bmc_global_best = lambda zoo_dir, *_args, bmc_result=None, **_kw: _best_from_bmc(zoo_dir, bmc_result)
