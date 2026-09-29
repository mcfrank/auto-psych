"""
Exhaustive stimulus design by Expected Information Gain (EIG) over PyMC models.

Enumerate EVERY sequence pair over the given lengths, score all of them in one
batched per-draw pass per PyMC model (module-level `model: pm.Model`), and
greedily select the set with maximal *joint* EIG about model identity
(src.models.eig_selection). Each model computes its own features from raw
stimulus rows via its ``compute_features`` or ``prepare_observed`` hook.

Usage (CLI):
    python3 -m src.pipelines.outer_loop.eig \\
        --select 32 --lengths 4 5 6 7 8 \\
        --models-dir PATH/cognitive_models \\
        --registry   PATH/model_registry.yaml \\
        --out        PATH/design/stimuli.json

    # --out defaults to stdout if omitted
    # --registry is optional (uniform prior over models if omitted)
    # --responses PREV/model_loop/responses.csv (all data so far) scores from
    # the posterior predictive; --n-responses is the participants per experiment
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

import tyro
from pyprojroot import here

# Ensure repo root on path so "import src..." works when run as a module/script.
# Must precede the (function-level) src imports below, hence here() rather than
# the canonical src.runtime.config.REPO_ROOT (same resolution).
sys.path.insert(0, str(here()))


def _load_model_names(models_dir: Path) -> List[str]:
    """Model names from the manifest that have a matching ``.py`` file."""
    from src.models.model_manifest import read_loadable_model_names  # type: ignore

    model_names = read_loadable_model_names(models_dir)
    if not model_names:
        raise ValueError(f"No loadable models found in {models_dir}")
    return model_names


def _load_model_weights(registry_path: Optional[Path]) -> Dict[str, float]:
    """Load the design prior from a registry YAML, or return empty (uniform)."""
    if registry_path is None:
        return {}
    from src.registry.io import load_registry  # type: ignore

    path = Path(registry_path)
    if not path.is_file():
        raise FileNotFoundError(f"Explicit model registry does not exist: {path}")
    return dict(load_registry(path)["theories"])


def _screen_usable_models(
    model_names: List[str], models_dir: Path, rows: List[Dict[str, Any]]
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Drop models that cannot be evaluated on the design pool's stimulus rows.

    E.g. a carried-forward model with a participant-level pm.Data
    (participant_id) that stimulus rows never carry, or one whose features
    cannot be computed for some pairs (a hook that indexes past the end of a
    length-2 sequence: admission never binds pairs shorter than 4). One such
    model would otherwise raise inside the predictive pass and abort the
    design — identically on every retry. Bind each model to every row, drop
    the ones that fail loudly and on record, and keep the rest; fail only if
    none can be evaluated.

    A failure is the model's when its own code raised it, or when it is not a
    code error at all (``is_model_failure``). A code error raised by the
    harness (``BROKEN_MODEL_CODE_ERRORS`` outside the model's file) and an
    infrastructure error raise: dropping every model for a broken harness
    would renormalize EIG over whichever models happen to load.
    """
    from src.models.data_binding import (  # type: ignore
        MissingStimulusColumns,
        NON_STIMULUS_COLUMNS,
        make_stim_data,
    )
    from src.models.model_loading import load_pymc_model_cached  # type: ignore
    from src.models.pymc_inference import (  # type: ignore
        INFRASTRUCTURE_ERRORS,
        is_model_failure,
    )

    usable: List[str] = []
    dropped: List[Dict[str, Any]] = []
    for name in model_names:
        model = None
        try:
            model = load_pymc_model_cached(name, models_dir)
            make_stim_data(model, rows)
        except MissingStimulusColumns as e:
            if not e.only_non_stimulus:
                raise RuntimeError(
                    f"model {name!r} in {models_dir} needs feature column(s) "
                    f"{[c for c in e.missing if c not in NON_STIMULUS_COLUMNS]} "
                    f"that the design rows do not carry (available: "
                    f"{list(e.available)}). That is a configuration error — "
                    "the rows lack columns this model needs — "
                    "not a participant-level mismatch. Dropping it would "
                    "renormalize EIG over whichever models happen to bind."
                ) from e
            reason = f"needs response-row column(s) {list(e.missing)}"
            dropped.append(
                {"model": name, "missing": list(e.missing), "reason": reason}
            )
            print(
                f"  [drop] EIG: model {name!r} {reason}; excluding it from EIG.",
                flush=True,
            )
            continue
        except INFRASTRUCTURE_ERRORS:
            raise
        except Exception as e:
            if not is_model_failure(e, Path(models_dir) / f"{name}.py"):
                raise RuntimeError(
                    f"model {name!r} in {models_dir} could not be evaluated "
                    f"({type(e).__name__}: {e}), and the model's own code did not "
                    "raise it: the harness is broken. Not dropping "
                    "the model — EIG must not silently renormalize over the "
                    "models that happen to load."
                ) from e
            reason = f"cannot be evaluated on a stimulus ({type(e).__name__}: {e})"
            if model is not None:
                reason += (
                    f"; fails on pairs of length {_failing_pair_lengths(model, rows)}"
                )
            dropped.append({"model": name, "missing": [], "reason": reason})
            print(
                f"  [drop] EIG: model {name!r} {reason}; excluding it from EIG.",
                flush=True,
            )
            continue
        usable.append(name)
    if not usable:
        raise ValueError(
            f"No models in {models_dir} can be evaluated on the design pool's stimulus rows "
            "(every model requires columns absent from stimuli, e.g. "
            "participant_id, or fails on its pairs); cannot compute EIG."
        )
    return usable, dropped


def _failing_pair_lengths(model: Any, rows: List[Dict[str, Any]]) -> List[int]:
    """The pair lengths among ``rows`` on which ``model`` cannot be bound."""
    from src.models.data_binding import make_stim_data  # type: ignore

    by_length: Dict[int, List[Dict[str, Any]]] = {}
    for row in rows:
        by_length.setdefault(len(row["sequence_a"]), []).append(row)
    failing = []
    for length, length_rows in sorted(by_length.items()):
        try:
            make_stim_data(model, length_rows)
        except Exception:  # noqa: BLE001 — only which lengths fail is wanted here
            failing.append(length)
    return failing


def _invalid_predictions_entry(
    name: str, exc: Any, rows: List[Dict[str, Any]], basis: str
) -> Dict[str, Any]:
    """The ``screened_out.json`` record of a model whose ``p_left`` is undefined
    (NaN or outside [0, 1]) on some design-pool pairs (``exc`` is its
    ``InvalidPredictions``)."""
    bad = [int(i) for i in exc.invalid_stimuli().nonzero()[0]]
    examples = [f"{rows[i]['sequence_a']} vs {rows[i]['sequence_b']}" for i in bad[:5]]
    reason = (
        f"{basis} p_left is undefined (NaN or outside [0, 1]) on {len(bad)} of "
        f"{len(rows)} design-pool pairs (e.g. {', '.join(examples)})"
    )
    return {"model": name, "missing": [], "reason": reason, "invalid_pairs": len(bad)}


def _screen_invalid_predictions(
    draws_of: Callable[[str], Any],
    model_names: List[str],
    rows: List[Dict[str, Any]],
    basis: str,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """``{name: draws_of(name)}`` over the models whose ``p_left`` is a
    probability on every pair, and a ``screened_out.json`` record for each
    model whose ``p_left`` is not.

    A carried model can break on pairs unlike any it was trained on (the
    recovery evaluation met one, commit 4de536c), and one such model used to
    crash the design — identically on every retry, since a resume reuses the
    same models and fits. It is left out of this design, loudly and on
    record, like a model that cannot bind a stimulus row.
    """
    from src.models.pymc_inference import InvalidPredictions  # type: ignore

    draws: Dict[str, Any] = {}
    screened: List[Dict[str, Any]] = []
    for name in model_names:
        try:
            draws[name] = draws_of(name)
        except InvalidPredictions as exc:
            entry = _invalid_predictions_entry(name, exc, rows, basis)
            screened.append(entry)
            print(
                f"  [screen] EIG: model {name!r} excluded from this design: "
                f"{entry['reason']}.",
                flush=True,
            )
    return draws, screened


def _raw_row(item: Dict[str, Any]) -> Dict[str, Any]:
    """Build a raw stimulus row: sequence_a, sequence_b, chose_left (dummy)."""
    row: Dict[str, Any] = dict(item)
    row.setdefault("chose_left", 0)
    return row


def _posterior_p_left_draws(
    model_names: List[str],
    models_dir: Path,
    rows: List[Dict[str, Any]],
    *,
    responses_csv: Path,
    fit_cache_dir: Optional[Path],
    max_draws: int,
    seed: int,
    fit_draws: Optional[int] = None,
    fit_tune: Optional[int] = None,
    fit_chains: Optional[int] = None,
) -> Dict[str, Any]:
    """Per-draw posterior-predictive p_left over ``rows`` for each model.

    Fits every model on ``responses_csv`` (design-time MCMC settings from
    ``src.models.mcmc_defaults`` unless overridden; target_accept is the
    model's own declared value, else DESIGN_TWIN_TARGET_ACCEPT) and predicts
    p_left draws for the stimulus pool, thinned to ``max_draws`` posterior
    samples. Returns the draws of every model whose p_left is defined on the
    whole pool, and the ``screened_out.json`` records of the others
    (``_screen_invalid_predictions``).
    """
    from src.models.mcmc_defaults import (  # type: ignore
        DESIGN_TWIN_CHAINS,
        DESIGN_TWIN_DRAWS,
        DESIGN_TWIN_TARGET_ACCEPT,
        DESIGN_TWIN_TUNE,
    )
    from src.models.data_binding import make_stim_data  # type: ignore
    from src.models.pymc_inference import fit_model, model_sampler_settings  # type: ignore

    def posterior_draws(name: str) -> Any:
        fitted = fit_model(
            name,
            models_dir,
            responses_csv,
            cache_dir=fit_cache_dir,
            draws=fit_draws if fit_draws is not None else DESIGN_TWIN_DRAWS,
            tune=fit_tune if fit_tune is not None else DESIGN_TWIN_TUNE,
            chains=fit_chains if fit_chains is not None else DESIGN_TWIN_CHAINS,
            target_accept=model_sampler_settings(name, models_dir).get(
                "target_accept", DESIGN_TWIN_TARGET_ACCEPT
            ),
        )
        stim_data = make_stim_data(fitted.model, rows)
        return fitted.predict_p_left_draws(stim_data, seed=seed, max_draws=max_draws)

    return _screen_invalid_predictions(
        posterior_draws, model_names, rows, "posterior-predictive"
    )


# How the design searches for the max-joint-EIG set (src/models/eig_selection.py):
# lazy batched greedy — a full pass over the pool every DESIGN_REFRESH_EVERY
# picks, and in between only the best-ranked candidates re-scored in batches of
# DESIGN_LAZY_BATCH_SIZE — in float32. Exact greedy (a full pass per pick, in
# float64) took 11-13 h per later-experiment design; this takes ~3 min on 16
# CPUs. Validated on two experiment-2 designs of the September 2026 sweep
# (scripts/subjective_randomness/validate_lazy_eig.py, 2026-09-27): exact
# greedy in float32 picked sets with the same joint EIG as in float64 (59-64 of
# 64 pairs shared, the rest near-ties), and lazy greedy's out-of-sample joint EIG
# was within Monte Carlo noise of exact greedy's (design 1, over three
# scenario seeds each: 2.078 vs 2.080 bits; design 2: 0.0006 bits lower, of
# 2.69), while exact greedy itself moved by up to 0.008 bits between scenario
# seeds. It is an approximation all the same (joint EIG is not submodular);
# exact greedy stays available (lazy=False, scoring_dtype "float64", or the
# eig CLI's --no-lazy --scoring-dtype float64).
DESIGN_LAZY_SEARCH = True
DESIGN_LAZY_BATCH_SIZE = 512
DESIGN_REFRESH_EVERY = 16
DESIGN_SCORING_DTYPE = "float32"
# Each Monte Carlo scenario's likelihood average leaves out the draw that
# generated the scenario (src/models/eig_selection.py; first audit, C5): an
# average that includes it rewards the true model for memorising its own draw.
# Measured on the two experiment-2 designs of the validation above (lazy
# float32, seeds 42 and 43, 2026-09-27): the same run time (167 s vs 167 s;
# 205 s vs 196-203 s), recorded joint_eig_bits lower by 0.06-0.10 and
# 0.02-0.03 bits (the inflation removed), and sets whose joint EIG on 50,000
# fresh scenarios, scored without the generating draw, is within the
# selection's own seed-to-seed spread (0.000 to -0.005 bits of 2.0 and 2.7).
DESIGN_LEAVE_ONE_OUT = True


def select_design_picks(
    draws: Dict[str, Any],
    n_select: int,
    *,
    model_weights: Optional[Dict[str, float]],
    n_scenarios: int,
    seed: int,
    n_responses: int,
    lazy: bool = DESIGN_LAZY_SEARCH,
    dtype: str = DESIGN_SCORING_DTYPE,
    n_threads: Optional[int] = None,
    leave_one_out: bool = DESIGN_LEAVE_ONE_OUT,
) -> List[Tuple[int, float, str]]:
    """The design's ``n_select`` picks as ``(pool index, joint EIG bits, source)``.

    With all ``n_responses`` answers counted, a few picks can identify the
    model; after that every gain is Monte Carlo noise, so selection stops at
    its noise floor and single-response EIG, conditioned on the picks so far,
    fills the remaining slots (user decision 2026-09-26). ``source`` says which
    objective chose a pick (``"eig"`` or ``"eig_single_response_fill"``), and
    its joint EIG is in that objective's units. ``n_threads`` (``None`` ⇒ the
    CPUs this process is allocated) score candidates concurrently; the picks
    do not depend on it. ``leave_one_out``: the estimator (see
    ``DESIGN_LEAVE_ONE_OUT``).
    """
    from src.models.eig_selection import select_n_joint_eig  # type: ignore
    from src.models.pymc_inference import allocated_cpus  # type: ignore

    search = dict(
        n_scenarios=n_scenarios,
        lazy=lazy,
        lazy_batch_size=DESIGN_LAZY_BATCH_SIZE,
        refresh_every=DESIGN_REFRESH_EVERY,
        dtype=dtype,
        n_threads=n_threads if n_threads is not None else allocated_cpus(),
        leave_one_out=leave_one_out,
    )
    print(
        f"  [design] joint-EIG search: {'lazy batched' if lazy else 'exact'} greedy"
        + (
            f" (batches of {DESIGN_LAZY_BATCH_SIZE}, full pass every "
            f"{DESIGN_REFRESH_EVERY} picks)"
            if lazy
            else ""
        )
        + f", {dtype} scoring on {search['n_threads']} thread(s), "
        + (
            "generating draw left out of each scenario's likelihood average."
            if leave_one_out
            else "generating draw included in each likelihood average."
        ),
        flush=True,
    )
    selection = select_n_joint_eig(
        draws,
        n_select,
        model_weights=model_weights,
        seed=seed,
        n_responses=n_responses,
        stop_below_noise=True,
        **search,
    )
    picks = [
        (idx, bits, "eig")
        for idx, bits in zip(selection.indices, selection.joint_eig_bits)
    ]
    if len(selection.indices) < n_select:
        fill = select_n_joint_eig(
            draws,
            n_select - len(selection.indices),
            model_weights=model_weights,
            seed=seed + 1,
            n_responses=1,
            preselected=selection.indices,
            **search,
        )
        picks += [
            (idx, bits, "eig_single_response_fill")
            for idx, bits in zip(fill.indices, fill.joint_eig_bits)
        ]
        print(
            f"  [design] {n_responses}-response EIG reached its noise floor after "
            f"{len(selection.indices)} pick(s); {len(fill.indices)} filled by "
            "single-response EIG.",
            flush=True,
        )
    return picks


def design_exhaustive(
    models_dir: Path,
    registry_path: Optional[Path] = None,
    *,
    lengths: tuple = (4, 5, 6, 7, 8),
    n_select: int = 32,
    n_random: int = 0,
    n_samples: int = 200,
    n_scenarios: int = 1000,
    seed: int = 42,
    random_seed: Optional[int] = None,
    screened_out_path: Optional[Path] = None,
    responses_csv: Optional[Path] = None,
    fit_cache_dir: Optional[Path] = None,
    fit_draws: Optional[int] = None,
    fit_tune: Optional[int] = None,
    fit_chains: Optional[int] = None,
    n_responses: int,
    lazy: bool = DESIGN_LAZY_SEARCH,
    scoring_dtype: str = DESIGN_SCORING_DTYPE,
    n_threads: Optional[int] = None,
    leave_one_out: bool = DESIGN_LEAVE_ONE_OUT,
) -> List[Dict[str, Any]]:
    """Select the max-joint-EIG stimulus set from the FULL pair universe.

    Enumerates every distinct same-length sequence pair over ``lengths``,
    scores all of them in one batched per-draw pass per
    model, and greedily selects the ``n_select`` stimuli with maximal joint
    EIG about model identity. No candidates file — the pool is the whole
    space, so nothing an agent could conjecture is outside it.

    Without ``responses_csv``, per-draw p_left comes from each model's
    **prior** predictive (experiment 1). With ``responses_csv``, each model is
    first fitted on those responses (MCMC, at the design-time settings from
    ``src.models.mcmc_defaults`` unless fit_* override them) and per-draw
    p_left comes from its **posterior** predictive, thinned to ``n_samples``
    draws — sequential design informed by the data so far.

    ``n_responses`` is how many responses each selected stimulus receives (the
    experiment's participant count): the joint EIG scores the count of "left"
    choices, Binomial(n_responses, p_left), not a single response.

    ``lazy``, ``scoring_dtype``, ``n_threads`` and ``leave_one_out`` set the
    greedy search (:func:`select_design_picks`); by default
    ``DESIGN_LAZY_SEARCH``, ``DESIGN_SCORING_DTYPE`` and
    ``DESIGN_LEAVE_ONE_OUT`` on every allocated CPU.

    Returns stimuli in selection (greedy) order, each with:
      - "eig": the stimulus's marginal EIG (bits);
      - "selection_rank": 1-based greedy pick order;
      - "joint_eig_bits": in-sample joint EIG of the set up to this stimulus.
    """
    from src.models.pymc_inference import (  # type: ignore
        eig_from_prior_means,
        prior_predict_p_left_draws,
    )
    from src.subjective_randomness.stimulus_design import (  # type: ignore
        enumerate_all_pairs,
    )

    import random as _random

    models_dir = Path(models_dir)
    if responses_csv is not None and not Path(responses_csv).exists():
        raise FileNotFoundError(
            f"Posterior exhaustive design needs responses at {responses_csv}, "
            "but the file is missing."
        )
    if n_select < 0 or n_random < 0:
        raise ValueError(f"n_select/n_random must be >= 0; got {n_select}, {n_random}.")
    if n_select == 0 and n_random == 0:
        raise ValueError("design_exhaustive needs n_select > 0 or n_random > 0.")

    pool = enumerate_all_pairs(list(lengths), same_length_only=True)
    rows = [_raw_row(item) for item in pool]

    results: List[Dict[str, Any]] = []
    chosen: set = set()

    # EIG-selected half: greedily pick the most jointly-informative pairs. Skipped
    # entirely when n_select == 0 (no model scoring needed for a pure-random set).
    if n_select > 0:
        model_names = _load_model_names(models_dir)
        model_weights = _load_model_weights(registry_path)
        model_names, screened_out = _screen_usable_models(model_names, models_dir, rows)
        if model_weights and not any(
            model_weights.get(n, 0.0) > 0 for n in model_names
        ):
            print(
                f"  [design] registry weights over {sorted(model_weights)} do not "
                f"overlap this model set {model_names}; using a uniform model prior.",
                flush=True,
            )
        basis = "prior predictive" if responses_csv is None else "posterior predictive"
        print(
            f"Exhaustive design: {len(pool):,d} pairs over lengths {list(lengths)}, "
            f"{len(model_names)} models ({basis}), selecting {n_select} by EIG "
            f"+ {n_random} random.",
            flush=True,
        )
        if responses_csv is None:
            draws, invalid = _screen_invalid_predictions(
                lambda name: prior_predict_p_left_draws(
                    [name], models_dir, rows, n_samples=n_samples, seed=seed
                )[name],
                model_names,
                rows,
                "prior-predictive",
            )
        else:
            draws, invalid = _posterior_p_left_draws(
                model_names,
                models_dir,
                rows,
                responses_csv=Path(responses_csv),
                fit_cache_dir=fit_cache_dir,
                max_draws=n_samples,
                seed=seed,
                fit_draws=fit_draws,
                fit_tune=fit_tune,
                fit_chains=fit_chains,
            )
        screened_out += invalid
        if screened_out_path is not None:
            # Written even when empty: "the screen ran and dropped nothing" and
            # "nobody looked" must not be the same absent file.
            Path(screened_out_path).parent.mkdir(parents=True, exist_ok=True)
            Path(screened_out_path).write_text(
                json.dumps(screened_out, indent=2), encoding="utf-8"
            )
        if not draws:
            raise ValueError(
                f"No model in {models_dir} has a defined p_left on the design pool "
                f"({[e['model'] for e in invalid]} screened out); cannot compute EIG."
            )
        picks = select_design_picks(
            draws,
            n_select,
            model_weights=model_weights or None,
            n_scenarios=n_scenarios,
            seed=seed,
            n_responses=n_responses,
            lazy=lazy,
            dtype=scoring_dtype,
            n_threads=n_threads,
            leave_one_out=leave_one_out,
        )
        means = {m: arr.mean(axis=0) for m, arr in draws.items()}
        for rank, (idx, joint_bits, source) in enumerate(picks, start=1):
            preds = {m: float(means[m][idx]) for m in means}
            results.append(
                {
                    **pool[idx],
                    "eig": round(eig_from_prior_means(preds, model_weights or None), 6),
                    "selection_rank": rank,
                    "joint_eig_bits": round(joint_bits, 6),
                    "source": source,
                }
            )
            chosen.add(int(idx))
    else:
        print(
            f"Exhaustive design: {len(pool):,d} pairs over lengths {list(lengths)}, "
            f"{n_random} random (no EIG selection).",
            flush=True,
        )

    # Random-coverage half: uniform over the pool (minus the EIG picks), sampling
    # the flat middle of the space the EIG selection deliberately avoids — so the
    # selected model must fit broadly, not only the discriminating extremes.
    if n_random > 0:
        remaining = [i for i in range(len(pool)) if i not in chosen]
        if n_random > len(remaining):
            raise ValueError(
                f"n_random={n_random} exceeds the {len(remaining)} pairs left after "
                f"EIG selection over lengths {list(lengths)}."
            )
        rng = _random.Random(random_seed if random_seed is not None else seed)
        for offset, idx in enumerate(sorted(rng.sample(remaining, n_random)), start=1):
            results.append(
                {
                    **pool[idx],
                    "eig": None,
                    "selection_rank": len(chosen) + offset,
                    "joint_eig_bits": None,
                    "source": "random",
                }
            )

    return results


@dataclass
class Args:
    """Exhaustively enumerate the pair universe and select the max-joint-EIG set."""

    models_dir: Path
    """Path to the cognitive_models/ directory."""
    n_responses: int
    """Responses each selected stimulus receives (participants per experiment)."""
    registry: Optional[Path] = None
    """Path to model_registry.yaml (optional; uniform prior if omitted)."""
    out: Optional[Path] = None
    """Output JSON file path (default: stdout)."""
    n_samples: int = 200
    """Per-draw p_left samples per model (prior- or posterior-predictive)."""
    lengths: tuple = (4, 5, 6, 7, 8)
    """Sequence lengths for the exhaustive pair universe."""
    select: int = 32
    """Stimulus-set size for the joint-EIG selection."""
    n_scenarios: int = 1000
    """Monte Carlo scenarios for joint-EIG gain estimation."""
    seed: int = 42
    """Seed for predictive draws and selection scenarios."""
    responses: Optional[Path] = None
    """Responses so far (the previous experiment's model_loop/responses.csv):
    fit each model on them and design from the POSTERIOR predictive instead of
    the prior."""
    fit_cache: Optional[Path] = None
    """Cache dir for the design-time MCMC fits (with --responses)."""
    lazy: bool = DESIGN_LAZY_SEARCH
    """Lazy batched greedy search instead of exact greedy (an approximation;
    see src/models/eig_selection.py)."""
    scoring_dtype: Literal["float32", "float64"] = DESIGN_SCORING_DTYPE
    """Precision of candidate scoring."""
    leave_one_out: bool = DESIGN_LEAVE_ONE_OUT
    """Leave each scenario's generating draw out of its likelihood average
    (--no-leave-one-out: the estimator used before 2026-09-27, which
    includes it)."""


def _write_output(stimuli: List[Dict[str, Any]], out: Optional[Path]) -> None:
    """Write the selected stimuli to ``out`` (or stdout if ``None``)."""
    output = json.dumps(stimuli, indent=2)
    if out:
        out.write_text(output, encoding="utf-8")
        eig_vals = [s["eig"] for s in stimuli]
        print(
            f"Wrote {len(stimuli)} stimuli to {out} "
            f"(EIG range: {min(eig_vals):.4f} – {max(eig_vals):.4f})",
            flush=True,
        )
    else:
        print(output)


def main(args: Args) -> None:
    """CLI entry point: run exhaustive EIG design and write selected stimuli."""
    selected = design_exhaustive(
        models_dir=args.models_dir,
        registry_path=args.registry,
        lengths=tuple(args.lengths),
        n_select=args.select,
        n_samples=args.n_samples,
        n_scenarios=args.n_scenarios,
        seed=args.seed,
        responses_csv=args.responses,
        fit_cache_dir=args.fit_cache,
        n_responses=args.n_responses,
        lazy=args.lazy,
        scoring_dtype=args.scoring_dtype,
        leave_one_out=args.leave_one_out,
    )
    _write_output(selected, args.out)


if __name__ == "__main__":
    main(tyro.cli(Args))
