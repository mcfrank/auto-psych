"""PyMC-native inner model loop.

The inner loop improves a cognitive model by repeatedly asking a coding agent
to write a new candidate **PyMC model** (`candidate.py` with a module-level
`with pm.Model() as model:` block), fitting every surviving model to the
observed responses via MCMC, and scoring them by ELPD-LOO. The Bayesian
posterior over models (softmax of ELPD-LOO, optionally with a complexity prior)
selects the incumbent; the best model is exported for the outer loop.

This replaces the earlier maximum-likelihood + BIC tournament: models are now
probabilistic programs fit by inference, not callables fit by external
optimization.

Layout under ``results_dir``::

    results_dir/
        models/                 # the surviving model set (the "zoo")
            <name>.py           # one PyMC model per surviving candidate
            models_manifest.yaml
            pruned/             # models that lost (audit trail, still readable)
        attempted_hypotheses.jsonl  # ledger: every candidate/prune event
        iter_0/candidate_0/     # per-candidate agent working dirs; a slot's
                                # retry (after an empty attempt) works in
                                # candidate_0_retry_1/, its repair (after a
                                # rejection) in candidate_0_repair_1/
        model_posterior.json    # ELPD-LOO posterior over models/
        history.json            # best model + posterior after every scoring step,
                                # with each round's critique status
        best_model.py           # copy of the exported (best reliable) model
        report.md
"""

from __future__ import annotations

import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from src.pipelines.inner_loop.hypothesis_ledger import (
    LEDGER_FILENAME,
    HypothesisLedger,
)
from src.pipelines.inner_loop.candidate_agent import (
    DEFAULT_CANDIDATE_HINTS,
    _repair_note,
    _retry_note,
    _spawn_candidate_agent,
    _write_candidate_context,
)
from src.pipelines.inner_loop.model_zoo import (
    DEFAULT_NOVELTY_RMSE_THRESHOLD,
    DEFAULT_PRUNE_DSE_MULTIPLIER,
    MAX_EMPTY_ROUND_RETRIES,
    NOVELTY_POOL_FILENAME,
    SLOT_EXPLORE,
    SLOT_REFINE_CHOSEN,
    SLOT_REFINE_INCUMBENT,
    AllCandidatesNoFileError,
    _NO_FILE_DETAIL,
    _admit_candidate_with_reason,
    _drop_nonfinite_elpd_models,
    _drop_unfittable_models,
    _is_all_no_file_round,
    _lens_index,
    _manifest_names,
    _cap_live_set,
    _prune_losers,
    _record,
    _resolve_candidate_name,
    _seed_model_set,
    name_clash_note,
    novelty_pool_rows,
    prefit_candidates,
    reserved_names,
    slot_roles,
)

from src.pipelines.inner_loop.critique_round import (
    CRITIQUE_N_PROPOSALS,
    CRITIQUE_PPC_REPLICATES,
    CRITIQUE_SIGNIFICANCE_ALPHA,
    _run_critique_round,
    critique_disabled_status,
)
from src.pipelines.inner_loop.scoring import (
    DEFAULT_COMPLEXITY_PRIOR_CONST,
    _compare,
    _export,
    _record_end_of_experiment_retirements,
    _record_history_step,
    _score,
)


# ─────────────────────────────────────────────
# Candidate slots
# ─────────────────────────────────────────────

# The files a rejected attempt hands to its repair as a starting point.
_CANDIDATE_FILES = ("candidate.py", "hypothesis.md", "model_name.txt")


@dataclass
class _Slot:
    """One candidate slot of a round, across its (at most three) agent attempts.

    A slot's first attempt works in ``candidate_<i>/``. An attempt that writes
    no ``candidate.py`` (or whose agent process fails) is followed by one
    retry in ``candidate_<i>_retry_1/``; an attempt whose candidate is
    rejected at admission is followed by one repair in
    ``candidate_<i>_repair_1/``, with the rejection reason in its prompt and
    the rejected files copied in. A repair that writes nothing, or is rejected
    again, is final — so every slot ends admitted or with a recorded reason the
    agent had a chance to act on. In the 2026-09 sweep 29% of slots ended as
    "no candidate.py written" and no rejected candidate ever saw its reason.

    A slot has a ``role`` (``model_zoo.slot_roles``), fixed across its
    attempts: an exploratory slot works the lens ``lens``; a refinement slot
    has no lens and refines the round's incumbent or a model of the agent's
    choosing.
    """

    idx: int
    role: str
    lens: Optional[int] = None
    directory: Path = field(init=False)
    docs: Dict[str, Optional[str]] = field(init=False)
    ledger_context: str = field(init=False)
    # The ledger name of this slot's previous attempt, left out of the next
    # attempt's "already tried — do not re-propose" list.
    previous_name: Optional[str] = None
    retried: bool = False
    repaired: bool = False
    result: Optional[Dict[str, str]] = None


def _slot_context(slot: _Slot, incumbent: str) -> str:
    """The ledger-context fragment naming a slot and its assignment."""
    if slot.role == SLOT_EXPLORE:
        return f"candidate {slot.idx} lens {slot.lens}"
    if slot.role == SLOT_REFINE_INCUMBENT:
        return f"candidate {slot.idx} refine incumbent {incumbent}"
    if slot.role == SLOT_REFINE_CHOSEN:
        return f"candidate {slot.idx} refine chosen"
    raise ValueError(f"Unknown slot role {slot.role!r} for candidate {slot.idx}.")


def _copy_rejected_attempt(previous_dir: Path, repair_dir: Path) -> None:
    """Copy a rejected attempt's candidate files into its repair directory."""
    for filename in _CANDIDATE_FILES:
        source = previous_dir / filename
        if source.exists():
            shutil.copyfile(source, repair_dir / filename)


# ─────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────


def run_pymc_inner_loop(
    responses_path: Path,
    results_dir: Path,
    *,
    seed_models_dir: Path,
    max_iterations: int = 3,
    candidate_count: int = 3,
    complexity_prior_const: float = DEFAULT_COMPLEXITY_PRIOR_CONST,
    cache_dir: Optional[Path] = None,
    agent_timeout_sec: int = 900,
    backend: Optional[str] = None,
    agent_model: Optional[str] = None,
    fit_kwargs: Optional[Dict[str, Any]] = None,
    enable_critique: bool = True,
    n_critique_proposals: int = CRITIQUE_N_PROPOSALS,
    critique_significance_alpha: float = CRITIQUE_SIGNIFICANCE_ALPHA,
    n_critique_replicates: int = CRITIQUE_PPC_REPLICATES,
    candidate_hints: Optional[List[str]] = None,
    novelty_rmse_threshold: float = DEFAULT_NOVELTY_RMSE_THRESHOLD,
    prune_dse_multiplier: float = DEFAULT_PRUNE_DSE_MULTIPLIER,
    candidate_parallelism: Optional[int] = None,
    starting_models: Optional[Iterable[str]] = None,
    ledger_context: str = "",
    lens_offset: int = 0,
    agent_root: Optional[Path] = None,
    agent_notes_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run the PyMC inner model loop and export the best model.

    Parameters
    ----------
    responses_path
        Preprocessed responses CSV. Its columns are the `pm.Data` inputs that
        candidate models read (one container per column they use).
    results_dir
        Output directory (created if absent).
    seed_models_dir
        Directory with the starting model set (`<name>.py` + manifest), e.g. the
        previous experiment's `cognitive_models/`. An ``attempted_hypotheses.jsonl``
        beside the manifest (the ledger a previous experiment carried) seeds
        this run's ledger.
    starting_models
        The run's starting models: names no candidate may take
        (``reserved_names``), whether or not they are still in the set. They
        are not protected — pruned, retired and dropped like any other model
        (user decision 2026-09-28). ``None`` means the models in
        ``seed_models_dir`` (right when that directory *is* the seed set); the
        outer loop passes the names it recorded at experiment 1
        (``run_starting_models``). A starting model that breaks the data
        contract still raises (``_drop_unfittable_models``).
    ledger_context
        Prefix for the ledger's ``context`` field (e.g. ``"experiment2"``).
    max_iterations
        Number of candidate-generation rounds. ``0`` only fits/compares the seed
        set (no agent is spawned).
    candidate_count
        Candidate slots per round. Slots have roles (``model_zoo.slot_roles``):
        with four or more, ``candidate_count - 3`` exploratory slots walk the
        lens battery, two refine the incumbent (named in the brief) and one
        refines a non-incumbent model of the agent's choosing from a menu of
        the live and pruned models; at three, one of each; at two, one
        exploratory and one incumbent slot; at one, exploratory only.
    complexity_prior_const
        Passed through to ``model_posterior`` (negative penalises complex models).
    fit_kwargs
        Extra kwargs for MCMC (e.g. ``{"draws": 500, "tune": 500, "chains": 2}``).
    enable_critique
        When True, run a CriticAL posterior-predictive critique of the incumbent
        (best) model before each candidate round and feed the resulting
        ``critiques.md`` to the candidate agents (see ``src/critique/ppc.py``).
        A critique agent that writes no usable statistic is retried once; a
        round can end with no critique, and every round's ``history.json``
        entry records its critique status either way.
    n_critique_proposals, critique_significance_alpha, n_critique_replicates
        Test statistics the critique agent proposes per round, the raw p-value
        threshold for a significant discrepancy, and the posterior-predictive replicates
        forming each statistic's null distribution.
    candidate_hints
        Exploration lenses cycled across a round's exploratory slots (``None``
        ⇒ ``DEFAULT_CANDIDATE_HINTS``). With no more exploratory slots than
        lenses, every exploratory slot in a round works a distinct lens.
    novelty_rmse_threshold
        Reject a candidate whose posterior-mean ``p_left`` is within this RMSE
        of an admitted model's on the loop's novelty pool — a broad stimulus
        pool the loop generates from its own seed (``novelty_pool_rows``) and
        records as ``novelty_pool.json``, not the training stimuli (``0``
        disables).
    prune_dse_multiplier
        At the end of the experiment, drop every model (starting models
        included) that is statistically distinguishable from the best trusted
        one (``elpd_diff > multiplier·dse_clustered``); ``0`` disables pruning.
    candidate_parallelism
        Concurrent candidate agents per round (``None`` ⇒ all of the round's
        candidates at once; ``1`` ⇒ sequential). Agents are CLI subprocesses,
        so this is a pure wall-clock lever; admission is always sequential in
        candidate order, keeping runs deterministic. With a ``cache_dir``, the
        candidates of a wave are fitted concurrently before admission
        (``prefit_candidates``, as many at once as the allocated CPUs hold),
        and admission loads those fits.
    lens_offset
        Starting position in the lens battery. The outer loop
        passes ``_lens_offset(exp_num, ...)`` so experiment k+1 continues the
        walk where experiment k stopped.
    agent_notes_dir
        Where Claude agents keep notes for later agents of the same run (their
        auto-memory; ``None`` gives them none). The outer loop passes one
        directory per run.

    Returns a dict with ``best_model``, ``posteriors``, ``elpd_loo``,
    ``live_models`` (the surviving zoo, in manifest order) and paths.
    """
    responses_path = Path(responses_path)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir = results_dir / "models"

    seeded_entries = _seed_model_set(Path(seed_models_dir), models_dir)
    seeded_names = {e.get("name") for e in seeded_entries if e.get("name")}
    # Every starting model, carried or not, is a name no candidate may take
    # (with those pruned or retired so far; see reserved_names).
    starting_models = (
        set(starting_models) if starting_models is not None else set(seeded_names)
    )
    # The loop's memory: every hypothesis tried, continuing the ledger the
    # previous experiment carried beside its model set.
    ledger = HypothesisLedger.create(
        results_dir / LEDGER_FILENAME,
        inherit_from=Path(seed_models_dir) / LEDGER_FILENAME,
    )
    _drop_unfittable_models(
        models_dir,
        responses_path,
        ledger=ledger,
        ledger_context=ledger_context,
        starting_models=starting_models,
    )
    fit_kwargs = fit_kwargs or {}
    # A carried-forward model can score a finite ELPD on a prior experiment's data
    # yet NaN on this one's; drop those before scoring so a single one can't crash
    # model_posterior and abort the whole run.
    _drop_nonfinite_elpd_models(
        models_dir,
        responses_path,
        cache_dir=cache_dir,
        fit_kwargs=fit_kwargs,
        ledger=ledger,
        ledger_context=ledger_context,
    )
    # The novelty pool: generated once from the loop's own seed and recorded in
    # the run tree, so every candidate this run gates is compared on the same
    # stimuli and what the gate saw is auditable. None when the gate is off.
    novelty_pool: Optional[List[Dict[str, str]]] = None
    if novelty_rmse_threshold > 0:
        novelty_pool = novelty_pool_rows()
        (results_dir / NOVELTY_POOL_FILENAME).write_text(
            json.dumps(novelty_pool), encoding="utf-8"
        )
    # Each model's posterior-mean p_left on the pool, computed once for the
    # whole loop, not at every admission: the data and every fit are fixed
    # within it (_min_prediction_rmse says why this matters).
    novelty_predictions: Dict[tuple, Any] = {}
    n_lenses = (
        len(candidate_hints)
        if candidate_hints is not None
        else len(DEFAULT_CANDIDATE_HINTS)
    )
    if max_iterations > 0 and n_lenses < 1:
        raise ValueError(
            "The lens battery is empty — pass at least one exploration lens "
            "via candidate_hints, or use the defaults."
        )

    posterior = _score(
        responses_path, models_dir, complexity_prior_const, cache_dir, fit_kwargs
    )
    # The comparison table (ELPD rank + PSIS-LOO reliability) is what selects
    # the best model at every step; it reuses the fits _score just made.
    comparison = _compare(responses_path, models_dir, cache_dir, fit_kwargs)
    history: List[Dict[str, Any]] = []
    _record_history_step(history, results_dir, posterior, comparison, iteration=None)

    rounds_abandoned = 0
    for iteration in range(max_iterations):
        round_dir = results_dir / f"iter_{iteration}"
        critique_path: Optional[Path] = None
        critique_status = critique_disabled_status()
        if enable_critique:
            critique = _run_critique_round(
                round_dir,
                responses_path=responses_path,
                models_dir=models_dir,
                posterior=posterior,
                comparison=comparison,
                cache_dir=cache_dir,
                fit_kwargs=fit_kwargs,
                n_proposals=n_critique_proposals,
                significance_alpha=critique_significance_alpha,
                n_replicates=n_critique_replicates,
                agent_timeout_sec=agent_timeout_sec,
                backend=backend,
                agent_model=agent_model,
                agent_root=agent_root,
                notes_dir=agent_notes_dir,
            )
            critique_path = critique.critiques_md
            critique_status = critique.status

        round_context = f"{ledger_context} round {iteration}".strip()
        round_results: List[Dict[str, str]] = []
        # The incumbent the refinement slots work on: the best model of the
        # latest scoring step, exactly as history.json records it (and as the
        # critique above critiqued), so the brief, the record and the
        # critique name the same model.
        incumbent = history[-1]["best_model"]
        roles = slot_roles(candidate_count)
        n_exploratory = roles.count(SLOT_EXPLORE)

        for attempt in range(1 + MAX_EMPTY_ROUND_RETRIES):
            if attempt == 0:
                attempt_dir = round_dir
            else:
                attempt_dir = results_dir / f"iter_{iteration}_retry_{attempt}"
                print(
                    f"  [retry] {round_context}: all candidates no-file on "
                    f"attempt {attempt}/{1 + MAX_EMPTY_ROUND_RETRIES}, retrying",
                    flush=True,
                )

            def prepare(
                slot: _Slot,
                directory: Path,
                *,
                attempt_note: Optional[str] = None,
                context_suffix: str = "",
            ) -> None:
                """Write the context documents for the slot's next attempt."""
                slot.directory = directory
                slot.docs = _write_candidate_context(
                    directory,
                    responses_path,
                    models_dir,
                    iteration,
                    slot.idx,
                    candidate_count,
                    posterior,
                    critique_path=critique_path,
                    hints=candidate_hints,
                    ledger=ledger,
                    comparison=comparison,
                    lens_index=slot.lens,
                    attempt_note=attempt_note,
                    omit_from_attempted=(
                        [slot.previous_name] if slot.previous_name else ()
                    ),
                    role=slot.role,
                    incumbent=incumbent,
                )
                slot.ledger_context = (
                    f"{round_context} {_slot_context(slot, incumbent)}{context_suffix}"
                )

            slots: List[_Slot] = []
            exploratory_idx = 0
            for idx, role in enumerate(roles):
                lens: Optional[int] = None
                if role == SLOT_EXPLORE:
                    lens = _lens_index(
                        lens_offset, iteration, n_exploratory, exploratory_idx, n_lenses
                    )
                    exploratory_idx += 1
                slot = _Slot(idx=idx, role=role, lens=lens)
                prepare(slot, attempt_dir / f"candidate_{idx}")
                slots.append(slot)

            def spawn(slot: _Slot) -> bool:
                return _spawn_candidate_agent(
                    slot.directory,
                    slot.docs,
                    models_dir=models_dir,
                    responses_path=responses_path,
                    agent_timeout_sec=agent_timeout_sec,
                    backend=backend,
                    agent_model=agent_model,
                    agent_root=agent_root,
                    notes_dir=agent_notes_dir,
                )

            def settle(slot: _Slot, spawned_ok: bool) -> bool:
                """Admit the slot's attempt; True when the slot gets another attempt."""
                previous_dir = slot.directory
                fallback = f"iter{iteration}_candidate{slot.idx}"
                this_attempt_is_a_repair = slot.repaired
                left_a_candidate = (previous_dir / "candidate.py").exists()
                if not spawned_ok and left_a_candidate:
                    # The agent process failed (timeout, non-zero exit) after
                    # writing a candidate — Opus was often still checking one
                    # when its time ran out. What it left goes through the
                    # usual admission gates below; a half-written file fails them.
                    print(
                        f"  [timeout] {slot.ledger_context}: agent process failed; "
                        f"admitting the candidate it left through the usual gates",
                        flush=True,
                    )
                if not spawned_ok and not left_a_candidate:
                    # The agent process failed and wrote nothing. Recorded like
                    # every attempt.
                    _record(
                        ledger,
                        name=fallback,
                        outcome="rejected",
                        detail="agent process failed — nothing admitted",
                        hypothesis="",
                        context=slot.ledger_context,
                    )
                    slot.previous_name = fallback
                    wrote_file = False
                    outcome = {
                        "outcome": "spawn_failed",
                        "detail": "agent process failed",
                    }
                else:
                    name = _resolve_candidate_name(
                        previous_dir,
                        models_dir,
                        fallback=fallback,
                        taken=reserved_names(models_dir, ledger, starting_models),
                    )
                    admission = _admit_candidate_with_reason(
                        previous_dir / "candidate.py",
                        models_dir,
                        model_name=name,
                        responses_path=responses_path,
                        cache_dir=cache_dir,
                        fit_kwargs=fit_kwargs,
                        novelty_rmse_threshold=novelty_rmse_threshold,
                        novelty_pool=novelty_pool,
                        ledger=ledger,
                        ledger_context=slot.ledger_context,
                        name_note=name_clash_note(previous_dir, name),
                        novelty_predictions=novelty_predictions,
                    )
                    slot.previous_name = name
                    if admission.admitted:
                        slot.result = {"outcome": "admitted", "detail": ""}
                        return False
                    wrote_file = (previous_dir / "candidate.py").exists()
                    outcome = {
                        "outcome": "rejected",
                        "detail": (
                            "rejected after file written"
                            if wrote_file
                            else _NO_FILE_DETAIL
                        ),
                    }

                if this_attempt_is_a_repair:
                    # The slot did write a candidate (that is what got repaired),
                    # so whatever the repair did, this is not an unfilled slot for
                    # the all-slots-empty round guard. A repair is always final.
                    slot.result = {
                        "outcome": "rejected",
                        "detail": f"final after repair: {outcome['detail']}",
                    }
                    return False
                slot.result = outcome

                if not wrote_file:
                    # An empty attempt gets one retry.
                    if slot.retried:
                        return False
                    slot.retried = True
                    print(
                        f"  [retry] {slot.ledger_context}: no candidate.py written "
                        f"— re-spawning the slot once",
                        flush=True,
                    )
                    prepare(
                        slot,
                        attempt_dir / f"candidate_{slot.idx}_retry_1",
                        attempt_note=_retry_note(previous_dir),
                        context_suffix=" retry 1",
                    )
                    return True

                # A rejected candidate gets one repair, with the reason verbatim.
                slot.repaired = True
                print(
                    f"  [repair] {slot.ledger_context}: re-spawning the slot once "
                    f"with the rejection reason",
                    flush=True,
                )
                prepare(
                    slot,
                    attempt_dir / f"candidate_{slot.idx}_repair_1",
                    attempt_note=_repair_note(previous_dir, admission.reason),
                    context_suffix=" repair 1",
                )
                _copy_rejected_attempt(previous_dir, slot.directory)
                return True

            def prefit(wave: List[_Slot]) -> None:
                """Fit the wave's candidates concurrently before admission.

                Names are predicted in slot order as admission will resolve
                them (a name claimed earlier in the wave is uniquified). The
                fits land in the cache; admission, below, is unchanged.
                """
                if cache_dir is None:
                    return  # nowhere to hand the fits over; admission fits one at a time
                claimed: List[str] = sorted(
                    reserved_names(models_dir, ledger, starting_models)
                )
                candidates = []
                for slot in wave:
                    candidate_file = slot.directory / "candidate.py"
                    if not candidate_file.exists():
                        continue
                    name = _resolve_candidate_name(
                        slot.directory,
                        models_dir,
                        fallback=f"iter{iteration}_candidate{slot.idx}",
                        taken=claimed,
                        announce=False,
                    )
                    claimed.append(name)
                    candidates.append((candidate_file, name))
                prefit_candidates(
                    candidates,
                    responses_path,
                    cache_dir=cache_dir,
                    fit_kwargs=fit_kwargs,
                )

            # Each wave spawns its attempts concurrently, fits their candidates
            # concurrently, then settles them sequentially in slot order
            # (admission mutates the manifest and runs the novelty gate, so
            # order keeps runs deterministic; the fits it needs are cached by
            # then). A settled slot that earned a retry or repair joins the
            # next wave.
            workers = min(candidate_parallelism or candidate_count, candidate_count)
            pending = list(slots)
            while pending:
                if workers > 1 and len(pending) > 1:
                    with ThreadPoolExecutor(
                        max_workers=min(workers, len(pending))
                    ) as pool:
                        spawn_ok = list(pool.map(spawn, pending))
                else:
                    spawn_ok = [spawn(slot) for slot in pending]
                prefit(pending)
                pending = [
                    slot for slot, ok in zip(pending, spawn_ok) if settle(slot, ok)
                ]

            round_results = [slot.result for slot in slots if slot.result is not None]
            if len(round_results) != len(slots):
                raise RuntimeError(
                    f"{round_context}: {len(slots) - len(round_results)} slot(s) ended "
                    "without a recorded result — every slot must settle."
                )
            if not _is_all_no_file_round(round_results):
                break

        if _is_all_no_file_round(round_results):
            rounds_abandoned += 1
            _record(
                ledger,
                name="__round__",
                outcome="round_abandoned",
                detail=(
                    f"abandoned after {1 + MAX_EMPTY_ROUND_RETRIES} attempts "
                    f"({len(round_results)} slots, all no-file)"
                ),
                hypothesis="",
                context=round_context,
            )
            print(
                f"  [abandon] {round_context}: abandoned after "
                f"{1 + MAX_EMPTY_ROUND_RETRIES} attempts — continuing to the "
                f"next round.",
                flush=True,
            )
            continue

        posterior = _score(
            responses_path, models_dir, complexity_prior_const, cache_dir, fit_kwargs
        )
        comparison = _compare(responses_path, models_dir, cache_dir, fit_kwargs)
        _record_history_step(
            history,
            results_dir,
            posterior,
            comparison,
            iteration=iteration,
            pruned=[],
            critique=critique_status,
        )

    if rounds_abandoned == max_iterations and max_iterations > 0:
        raise AllCandidatesNoFileError(
            f"Every round of the experiment ended with all candidates failing "
            f"to write files ({rounds_abandoned} of {max_iterations} rounds "
            f"abandoned after {1 + MAX_EMPTY_ROUND_RETRIES} attempts each). "
            f"This is a systemic failure, not a transient hiccup."
        )

    # Pruning happens once, at the end of the experiment (user decision
    # 2026-09-26): it used to run after every round against the max of a
    # growing zoo. Then the live set is capped. Both apply to every model,
    # starting models included (user decision 2026-09-28). Neither can
    # change the best trusted model or empty the set: pruning measures every
    # margin against the best trusted model, and the cap retires untrusted
    # fits first, then from the bottom, keeping at least one model.
    end_context = f"{ledger_context} end of experiment".strip()
    retired = _prune_losers(
        models_dir,
        responses_path,
        cache_dir=cache_dir,
        fit_kwargs=fit_kwargs,
        dse_multiplier=prune_dse_multiplier,
        ledger=ledger,
        ledger_context=end_context,
    )
    retired += _cap_live_set(
        models_dir,
        responses_path,
        cache_dir=cache_dir,
        fit_kwargs=fit_kwargs,
        ledger=ledger,
        ledger_context=end_context,
    )
    if retired:
        posterior = _score(
            responses_path, models_dir, complexity_prior_const, cache_dir, fit_kwargs
        )
        comparison = _compare(responses_path, models_dir, cache_dir, fit_kwargs)
        _record_end_of_experiment_retirements(history, results_dir, retired)

    result = _export(results_dir, models_dir, posterior, comparison)
    result["history"] = history
    result["history_path"] = str(results_dir / "history.json")
    result["live_models"] = _manifest_names(models_dir)
    result["ledger_path"] = str(ledger.path)
    return result
