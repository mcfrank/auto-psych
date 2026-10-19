"""
Exhaustive stimulus design by Expected Information Gain (EIG) over PyMC models.

Take EVERY stimulus in the task's design pool (for the coin-flip task every
same-length H/T pair at lengths 2-8; for the number game every (set, target)
cell), score all of them in one
batched per-draw pass per PyMC model (module-level `model: pm.Model`), and
greedily select the set with maximal *joint* EIG about model identity
(src.models.eig_selection). Each model computes its own features from raw
stimulus rows via its ``compute_features`` or ``prepare_observed`` hook.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple



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


# A design after data predicts a model with a participant effect as this many
# participants its data have not reached (``_new_participant_draws``): ids
# past every training id, whose parameters no data informed, so in each
# posterior draw they are draws from the model's own population distribution
# (its hyperparameters as fitted). Averaging over them marginalizes the
# person-level parameters, the model's prediction for a new participant; 40
# gives the Monte Carlo precision of averaging over a 40-person sample (user
# decision 2026-09-30).
DESIGN_NEW_PARTICIPANTS = 40


class NoNewParticipant(ValueError):
    """A model with a participant effect that cannot predict a participant its
    data have not reached: its per-participant parameters have no slot past the
    training ids, or its hook renumbers participants. Left out of the design,
    on record (``screened_out.json``)."""

    def __init__(self, model: str, reason: str):
        super().__init__(reason)
        self.model = model
        self.reason = reason


def _new_participant_ids(
    error: Any, responses_csv: Optional[Path]
) -> Optional[List[int]]:
    """The participants a model that failed to bind with ``error`` (a
    ``MissingStimulusColumns``) is predicted as: ``DESIGN_NEW_PARTICIPANTS``
    ids past every id in ``responses_csv``, when its only missing column is
    ``participant_id`` and the design has responses; else ``None``. The
    responses are read only for such a model, so a design whose models need no
    participant does not depend on the column."""
    if responses_csv is None or set(error.missing) != {"participant_id"}:
        return None
    from src.models.data_binding import participant_ids_in  # type: ignore

    training = participant_ids_in(Path(responses_csv))
    if not training:
        return None
    return list(range(max(training) + 1, max(training) + 1 + DESIGN_NEW_PARTICIPANTS))


def _require_own_slot(name: str, stim_data: Dict[str, Any], participant_id: int) -> None:
    """Raise ``NoNewParticipant`` unless ``participant_id`` reaches the model
    as given: a hook that renumbers participants (or binds them under another
    name) would put a new participant on a training participant's parameters."""
    import numpy as np

    bound = stim_data.get("participant_id")
    if bound is None or not np.all(np.asarray(bound) == participant_id):
        raise NoNewParticipant(
            name,
            "its hook does not pass participant_id through unchanged, so a "
            "participant its data have not reached cannot be given a slot of "
            "its own (bind participant_id directly and index parameters by it)",
        )


def _new_participant_draws(
    name: str,
    fitted: Any,
    rows: List[Dict[str, Any]],
    new_ids: Sequence[int],
    *,
    var_name: str,
    seed: int,
    max_draws: int,
) -> Any:
    """Per-draw ``var_name`` (the task's prediction) over ``rows`` of a model with a participant effect, as
    a new participant: predicted as each of ``new_ids`` (slots no data
    reached, so draws from the model's population distribution) and averaged
    draw by draw, over the same thinned draws each time (``_thin_posterior``
    is deterministic). New participants answer one pair independently, so each
    pair's Binomial(n, p) in the EIG is exact under the model; one person's
    answers to different pairs are correlated, which the joint EIG ignores.

    A model whose per-participant parameters stop before ``new_ids`` raises an
    index error for them while a training participant (``new_ids[0] - 1``)
    predicts: ``NoNewParticipant``. An error for the training participant too
    is the model's own and propagates.
    """
    from src.models.data_binding import as_participant, make_stim_data  # type: ignore

    total = None
    for pid in new_ids:
        stim_data = make_stim_data(fitted.model, as_participant(rows, pid))
        _require_own_slot(name, stim_data, pid)
        try:
            draws = fitted.predict_p_left_draws(
                stim_data, var_name=var_name, seed=seed, max_draws=max_draws
            )
        except IndexError as exc:
            training_id = new_ids[0] - 1
            fitted.predict_p_left_draws(
                make_stim_data(fitted.model, as_participant(rows, training_id)),
                var_name=var_name,
                seed=seed,
                max_draws=max_draws,
            )
            raise NoNewParticipant(
                name,
                f"its per-participant parameters have no slot for participant "
                f"{pid} (the data reach participant {training_id}), so it cannot "
                "predict a new participant (size them with spare slots, e.g. 400)",
            ) from exc
        total = draws.astype("float64") if total is None else total + draws
    return total / len(new_ids)


def _screen_usable_models(
    model_names: List[str],
    models_dir: Path,
    rows: List[Dict[str, Any]],
    task: Any,
    *,
    responses_csv: Optional[Path] = None,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Drop models that cannot be evaluated on the design pool's stimulus rows.

    E.g. a model that binds ``trial_index``, or a participant-level pm.Data
    (``participant_id``) that stimulus rows never carry when there are no
    responses to take participants from (the prior design), or one whose
    features cannot be computed for some pairs (a hook that indexes past the
    end of a length-2 sequence: admission never binds pairs shorter than 4).
    One such model would otherwise raise inside the predictive pass and abort
    the design — identically on every retry. Bind each model to every row, drop
    the ones that fail loudly and on record, and keep the rest; fail only if
    none can be evaluated.

    With ``responses_csv`` (a design after data), a model whose only
    unbindable column is ``participant_id`` is kept: it is bound as a new
    participant here (and dropped, on record, if its hook renumbers
    participants), and ``_posterior_p_left_draws`` predicts it as new
    participants (``_new_participant_draws``): dropping person-level models
    can leave a design with no model at all.

    A failure is the model's when its own code raised it, or when it is not a
    code error at all (``is_model_failure``). A code error raised by the
    harness (``BROKEN_MODEL_CODE_ERRORS`` outside the model's file) and an
    infrastructure error raise: dropping every model for a broken harness
    would renormalize EIG over whichever models happen to load.
    """
    from src.models.data_binding import (  # type: ignore
        MissingStimulusColumns,
        NON_STIMULUS_COLUMNS,
        as_participant,
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
            try:
                make_stim_data(model, rows)
            except MissingStimulusColumns as e:
                new_ids = _new_participant_ids(e, responses_csv)
                if not new_ids:
                    raise
                _require_own_slot(
                    name, make_stim_data(model, as_participant(rows, new_ids[0])), new_ids[0]
                )
                print(
                    f"  [design] model {name!r} has a participant effect: "
                    f"predicted as {len(new_ids)} new participants (ids "
                    f"{new_ids[0]}-{new_ids[-1]}) and averaged.",
                    flush=True,
                )
        except NoNewParticipant as e:
            dropped.append({"model": name, "missing": [], "reason": e.reason})
            print(
                f"  [drop] EIG: model {name!r} {e.reason}; excluding it from EIG.",
                flush=True,
            )
            continue
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
                failing = [task.stimulus_label(row) for row in _failing_rows(model, rows)]
                reason += f"; fails on e.g. {', '.join(failing)}"
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


def _failing_rows(model: Any, rows: List[Dict[str, Any]], limit: int = 3) -> List[Dict[str, Any]]:
    """Up to ``limit`` rows on which ``model`` cannot be bound, found by
    bisection (binding every row on its own would take a pass per row)."""
    from src.models.data_binding import make_stim_data  # type: ignore

    def binds(chunk: List[Dict[str, Any]]) -> bool:
        try:
            make_stim_data(model, chunk)
            return True
        except Exception:  # noqa: BLE001 — only which rows fail is wanted here
            return False

    failing: List[Dict[str, Any]] = []
    pending = [rows]
    while pending and len(failing) < limit:
        chunk = pending.pop(0)
        if binds(chunk):
            continue
        if len(chunk) == 1:
            failing.append(chunk[0])
        else:
            pending += [chunk[: len(chunk) // 2], chunk[len(chunk) // 2 :]]
    return failing


def _invalid_predictions_entry(
    name: str, exc: Any, rows: List[Dict[str, Any]], basis: str, task: Any
) -> Dict[str, Any]:
    """The ``screened_out.json`` record of a model whose prediction is undefined
    (NaN or outside [0, 1]) on some design-pool stimuli (``exc`` is its
    ``InvalidPredictions``)."""
    bad = [int(i) for i in exc.invalid_stimuli().nonzero()[0]]
    examples = [task.stimulus_label(rows[i]) for i in bad[:5]]
    reason = (
        f"{basis} {task.prediction_var} is undefined (NaN or outside [0, 1]) on {len(bad)} of "
        f"{len(rows)} design-pool stimuli (e.g. {', '.join(examples)})"
    )
    return {"model": name, "missing": [], "reason": reason, "invalid_pairs": len(bad)}


def _screen_invalid_predictions(
    draws_of: Callable[[str], Any],
    model_names: List[str],
    rows: List[Dict[str, Any]],
    basis: str,
    task: Any,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """``{name: draws_of(name)}`` over the models whose prediction is a
    probability on every pair, and a ``screened_out.json`` record for each
    model whose prediction is not.

    A carried model can break on pairs unlike any it was trained on, and would
    crash the design identically on every retry (a resume reuses the same
    models and fits). It is left out of this design, loudly and on record,
    like a model that cannot bind a stimulus row.
    """
    from src.models.pymc_inference import InvalidPredictions  # type: ignore

    draws: Dict[str, Any] = {}
    screened: List[Dict[str, Any]] = []
    for name in model_names:
        try:
            draws[name] = draws_of(name)
        except InvalidPredictions as exc:
            entry = _invalid_predictions_entry(name, exc, rows, basis, task)
            screened.append(entry)
            print(
                f"  [screen] EIG: model {name!r} excluded from this design: "
                f"{entry['reason']}.",
                flush=True,
            )
        except NoNewParticipant as exc:
            screened.append({"model": name, "missing": [], "reason": exc.reason})
            print(
                f"  [screen] EIG: model {name!r} excluded from this design: "
                f"{exc.reason}.",
                flush=True,
            )
    return draws, screened


def _posterior_p_left_draws(
    model_names: List[str],
    models_dir: Path,
    rows: List[Dict[str, Any]],
    *,
    responses_csv: Path,
    fit_cache_dir: Optional[Path],
    max_draws: int,
    seed: int,
    task: Any,
    fit_method: Optional[str] = None,
) -> Dict[str, Any]:
    """Per-draw posterior-predictive prediction (the task's probability) over ``rows`` for each model.

    Fits every model on ``responses_csv`` (design-time MCMC settings from
    ``src.models.mcmc_defaults``; target_accept is the
    model's own declared value, else DESIGN_TWIN_TARGET_ACCEPT) and predicts
    prediction draws (the task's ``prediction_var``) for the stimulus pool,
    thinned to ``max_draws`` posterior samples; a model with a participant
    effect is predicted as new participants and averaged
    (``_new_participant_draws``; one that cannot be is screened out on record).
    Returns the draws of every model whose prediction is defined on the whole
    pool, and the ``screened_out.json`` records of the others
    (``_screen_invalid_predictions``).
    """
    from src.models.mcmc_defaults import (  # type: ignore
        DESIGN_TWIN_CHAINS,
        DESIGN_TWIN_DRAWS,
        DESIGN_TWIN_TARGET_ACCEPT,
        DESIGN_TWIN_TUNE,
    )
    from src.models.data_binding import MissingStimulusColumns, make_stim_data  # type: ignore
    from src.models.pymc_inference import fit_model, model_sampler_settings  # type: ignore

    def posterior_draws(name: str) -> Any:
        fitted = fit_model(
            name,
            models_dir,
            responses_csv,
            cache_dir=fit_cache_dir,
            draws=DESIGN_TWIN_DRAWS,
            tune=DESIGN_TWIN_TUNE,
            chains=DESIGN_TWIN_CHAINS,
            target_accept=model_sampler_settings(name, models_dir).get(
                "target_accept", DESIGN_TWIN_TARGET_ACCEPT
            ),
            method=fit_method,
        )
        try:
            stim_data = make_stim_data(fitted.model, rows)
        except MissingStimulusColumns as e:
            new_ids = _new_participant_ids(e, responses_csv)
            if not new_ids:
                raise
            return _new_participant_draws(
                name,
                fitted,
                rows,
                new_ids,
                var_name=task.prediction_var,
                seed=seed,
                max_draws=max_draws,
            )
        return fitted.predict_p_left_draws(
            stim_data, var_name=task.prediction_var, seed=seed, max_draws=max_draws
        )

    return _screen_invalid_predictions(
        posterior_draws, model_names, rows, "posterior-predictive", task
    )


# How the design searches for the max-joint-EIG set (src/models/eig_selection.py):
# lazy batched greedy — a full pass over the pool every DESIGN_REFRESH_EVERY
# picks, and in between only the best-ranked candidates re-scored in batches of
# DESIGN_LAZY_BATCH_SIZE — in float32. Exact greedy (a full pass per pick, in
# float64) takes 11-13 h per later-experiment design; this takes ~3 min on 16
# CPUs, and its joint EIG is within exact greedy's own scenario-to-scenario
# spread (scripts/subjective_randomness/validate_lazy_eig.py). It is an
# approximation all the same (joint EIG is not submodular); exact greedy stays
# available (lazy=False, scoring_dtype "float64").
DESIGN_LAZY_SEARCH = True
DESIGN_LAZY_BATCH_SIZE = 512
DESIGN_REFRESH_EVERY = 16
DESIGN_SCORING_DTYPE = "float32"
# Each Monte Carlo scenario's likelihood average leaves out the draw that
# generated the scenario (src/models/eig_selection.py; first audit, C5): an
# average that includes it rewards the true model for memorising its own draw
# and inflates joint_eig_bits (by 0.02-0.1 bits on the validation designs), at
# no extra cost.
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
    task: Any,
    pool: Optional[List[Dict[str, Any]]] = None,
    n_select: int = 32,
    n_random: int = 0,
    n_samples: int = 200,
    n_scenarios: int = 1000,
    seed: int = 42,
    screened_out_path: Optional[Path] = None,
    responses_csv: Optional[Path] = None,
    fit_cache_dir: Optional[Path] = None,
    fit_method: Optional[str] = None,
    n_responses: int,
    lazy: bool = DESIGN_LAZY_SEARCH,
    scoring_dtype: str = DESIGN_SCORING_DTYPE,
    n_threads: Optional[int] = None,
    leave_one_out: bool = DESIGN_LEAVE_ONE_OUT,
) -> List[Dict[str, Any]]:
    """Select the max-joint-EIG stimulus set from the task's FULL design pool.

    Takes every stimulus in ``task.design_pool()`` (or in ``pool``, a part of
    it), scores all of them in one
    batched per-draw pass per
    model, and greedily selects the ``n_select`` stimuli with maximal joint
    EIG about model identity. No candidates file — the pool is the whole
    space, so nothing an agent could conjecture is outside it.

    Without ``responses_csv``, per-draw predictions come from each model's
    **prior** predictive (experiment 1). With ``responses_csv``, each model is
    first fitted on those responses (MCMC, at the design-time settings from
    ``src.models.mcmc_defaults``) and per-draw
    predictions come from its **posterior** predictive, thinned to ``n_samples``
    draws — sequential design informed by the data so far.

    ``n_responses`` is how many responses each selected stimulus receives (the
    experiment's participant count): the joint EIG scores the count of
    positive responses, Binomial(n_responses, p), not a single response.

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

    pool = list(pool) if pool is not None else task.design_pool()
    rows = task.stimulus_rows(pool)

    results: List[Dict[str, Any]] = []
    chosen: set = set()

    # EIG-selected half: greedily pick the most jointly-informative pairs. Skipped
    # entirely when n_select == 0 (no model scoring needed for a pure-random set).
    if n_select > 0:
        model_names = _load_model_names(models_dir)
        model_weights = _load_model_weights(registry_path)
        # After data, a model with a participant effect is predicted as the
        # training participants (_screen_usable_models); before, it cannot be.
        model_names, screened_out = _screen_usable_models(
            model_names, models_dir, rows, task, responses_csv=responses_csv
        )
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
            f"Exhaustive design: {len(pool):,d} {task.name} stimuli, "
            f"{len(model_names)} models ({basis}), selecting {n_select} by EIG "
            f"+ {n_random} random.",
            flush=True,
        )
        if responses_csv is None:
            draws, invalid = _screen_invalid_predictions(
                lambda name: prior_predict_p_left_draws(
                    [name], models_dir, rows, n_samples=n_samples, seed=seed,
                    var_name=task.prediction_var,
                )[name],
                model_names,
                rows,
                "prior-predictive",
                task,
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
                task=task,
                fit_method=fit_method,
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
                f"No model in {models_dir} has a defined {task.prediction_var} on the design pool "
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
            f"Exhaustive design: {len(pool):,d} {task.name} stimuli, "
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
                f"n_random={n_random} exceeds the {len(remaining)} stimuli left after "
                f"EIG selection."
            )
        rng = _random.Random(seed)
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
