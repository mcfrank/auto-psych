"""One pooled number-game model's per-condition predictions, of the same kind as the figure data's
``conditions`` (``number_game_figure_data.py``): the posterior-mean P(yes) of its fit on all runs' final
data pooled, at the pooled sampler settings, averaged over each (example set, target) cell.

The figure data keep only the pooled best model's and best seed's predictions. Figure 3 draws the number
game with run 3's winner instead (``fig3.COLUMN_MODEL``), so this script writes that model's predictions
beside the figure data (``data/paper/figure_data/number_game_model_conditions.json``).

The fit is read from a copy of the figure data's pooled fit cache, never sampled: the file is
``<model>.<fingerprint>.nc``, whose fingerprint (``fit_fingerprint``) this script prints. To get it, copy
that one file from the figure data's work directory on Sherlock
(``$SCRATCH/auto-psych/paper/number_game_figure_data_nov26/pooled/fit_cache/``) into ``--cache-dir``.

Usage:
    uv run python -m scripts.paper.number_game_model_conditions
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro
from pyprojroot import here

from scripts.paper import number_game_figure_data as ng
from scripts.paper import sr_figure_data as sr


@dataclass
class Args:
    model: str = "exception_weighted_rule_rejection"
    """The pooled model whose predictions to write (one of the runs' winners or the seeds)."""
    cache_dir: Path = here() / "data" / "paper" / "work" / "ng_pooled_fit_cache"
    """A local copy of the model's full-data fit from the figure data's pooled fit cache."""
    work_dir: Path = here() / "data" / "paper" / "work" / "number_game_model_conditions"
    """Where the pooled responses and the model's source are written (as the figure data wrote them)."""
    out: Path = here() / "data" / "paper" / "figure_data" / "number_game_model_conditions.json"
    figure_data: Path = here() / "data" / "paper" / "figure_data" / "number_game.json"
    """The figure data, whose pooled settings and conditions these predictions must match."""
    prediction_draws: int = ng.Args.prediction_draws
    """Posterior draws (thinned) behind each posterior-mean prediction, as in the figure data."""


def main(args: Args) -> None:
    from src.models.data_binding import make_stim_data
    from src.models.pymc_inference import load_cached_fit
    from src.tasks.number_game import NUMBER_GAME

    defaults = ng.Args(work_dir=args.work_dir)
    settings = ng.shared_run_settings(defaults.harness_root, defaults.runs)
    infos = [ng.run_models(defaults, run, settings["fit_kwargs"], settings["selection"]) for run in defaults.runs]
    responses = ng.write_pooled_responses(defaults, infos)
    model_set = ng.pooled_model_set(infos)
    if args.model not in model_set["files"]:
        raise ValueError(f"{args.model!r} is not a pooled model: {sorted(model_set['files'])}")
    fit_kwargs = ng.pooled_fit_settings(defaults, settings["fit_kwargs"])
    figure = json.loads(args.figure_data.read_text())
    if figure["pooled_fit_settings"] != fit_kwargs:
        raise RuntimeError(f"pooled settings {fit_kwargs} are not the figure data's {figure['pooled_fit_settings']}")
    models_dir = args.work_dir / ng.POOLED_DIR / "models"
    sr.write_model_set(models_dir, {args.model: model_set["files"][args.model]})

    fitted = load_cached_fit(args.model, models_dir, responses, cache_dir=args.cache_dir, **fit_kwargs)
    print(f"{args.model}: fit {fitted.fingerprint} from {args.cache_dir}", flush=True)
    rows = sr._read_rows(responses)
    draws = fitted.predict_p_left_draws(make_stim_data(fitted.model, rows), var_name=NUMBER_GAME.prediction_var,
                                        seed=0, max_draws=args.prediction_draws)
    p_yes = np.asarray(draws, dtype="float64").mean(axis=0)
    conditions = ng.condition_table(rows, {"p_yes": p_yes})
    expected = [(c["number_set"], c["target"], c["n"], c["observed"]) for c in figure["conditions"]]
    if [(c["number_set"], c["target"], c["n"], c["observed"]) for c in conditions] != expected:
        raise RuntimeError(f"the pooled responses' conditions are not those of {args.figure_data}")

    args.out.write_text(json.dumps({
        "model": args.model,
        "run": model_set["winners"].get(args.model),
        "prediction": "posterior-mean P(yes) of the fit on all runs' final data pooled (in sample), "
                      "averaged over each (example set, target) cell, as the figure data's conditions",
        "fit_fingerprint": fitted.fingerprint,
        "fit_settings": fit_kwargs,
        "prediction_draws": args.prediction_draws,
        "conditions": conditions,
    }, indent=1))
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main(tyro.cli(Args))
