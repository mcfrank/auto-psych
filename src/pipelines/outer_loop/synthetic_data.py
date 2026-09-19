"""Synthetic data generation: LLM-as-participant and model-based responses."""

from __future__ import annotations

import random
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

from src.models.probability import validate_probability_distribution
from src.models.theorist.predictions import get_model_predictions

RESPONSE_OPTIONS = ["left", "right"]

# Attempts per trial (1 + one retry). The completeness gate downstream rejects
# any collection missing a response, so a trial gets a single second chance at
# a transient backend error or an uncommitted reply before it is counted.
_TRIAL_ATTEMPTS = 2


def _parse_participant_answer(text: str) -> str | None:
    """Parse a participant reply into ``"left"``/``"right"``, or ``None`` if it did
    not clearly commit to one.

    Only the explicit ``ANSWER: left|right`` form (the format the participant is
    instructed to use) or a reply that is exactly ``left``/``right`` counts. A
    loose "does the word 'left' appear anywhere?" scan is deliberately NOT used:
    chain-of-thought that merely mentions a side ("the left column looks…",
    "I'd lean left but it's close") would be miscoded as a committed choice, and
    because presentation side is fixed that miscoding would bias the data toward
    one side. An ambiguous reply is better counted as unparseable (and dropped)
    than silently mis-sided.
    """
    if not text or not isinstance(text, str):
        return None
    match = re.search(r"ANSWER\s*:\s*(left|right)\b", text, re.IGNORECASE)
    if match:
        return match.group(1).lower()
    normalized = text.strip().lower()
    if normalized in ("left", "right"):
        return normalized
    return None


def _present_sides(seq_a: str, seq_b: str, swap: bool) -> tuple[str, str]:
    """Return the presented ``(left, right)`` order: canonical, or swapped.

    Side counterbalancing: each trial independently randomizes which sequence is
    shown on the LEFT, and we record the *presented* order into
    ``sequence_a`` (= left) / ``sequence_b`` (= right). The cognitive models treat
    ``sequence_a`` as the left option, so randomizing the side per trial decouples
    content from physical side — making each model's ``side_bias`` a genuine,
    identifiable physical-left preference — with no model or schema change.
    """
    return (seq_b, seq_a) if swap else (seq_a, seq_b)


def generate_llm_participant_rows(
    stimuli: list[dict[str, Any]],
    n_participants: int,
    *,
    participant_model: Any,
    prompt_text: str,
    transcripts_dir: Path | None = None,
    max_workers: int = 8,
    progress: Callable[[int, int], None] | None = None,
    seed: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Run an LLM-as-participant over ``stimuli`` and return ``(rows, stats)``.

    Model-agnostic: ``participant_model`` is any object exposing
    ``.answer(system, user) -> str`` and a ``.name`` (see
    ``participants.ParticipantModel``), so the closed (API) and open (Hugging
    Face) backends share this one loop. A backend may expose a positive integer
    ``max_concurrency`` to limit simultaneous calls on its shared instance. Each
    participant answers every stimulus;
    unparseable replies and per-trial errors are counted but never abort the run.
    If ``transcripts_dir`` is given, one Markdown transcript per participant is
    written there.

    Participants run concurrently (up to ``max_workers``) since each one's API
    calls are independent and I/O-bound; output rows are reassembled in
    participant order so the result is deterministic regardless of finish order.
    ``progress(participant_id, n_rows)`` is called as each participant completes.

    ``stats`` carries ``n_participants``, ``n_stimuli``, ``n_rows``,
    ``n_unparseable``, ``n_errors``.
    """
    model_name = getattr(participant_model, "name", "llm_participant")

    def _run_participant(participant_id: int) -> tuple[list[dict[str, Any]], int, int]:
        rows_p: list[dict[str, Any]] = []
        unparseable = 0
        errors = 0
        # Per-participant seeded RNG so the side counterbalancing is reproducible
        # regardless of the (concurrent) finish order of participants.
        prng = random.Random(seed + participant_id * 1_000_003)
        transcript = None
        if transcripts_dir is not None:
            transcripts_dir.mkdir(parents=True, exist_ok=True)
            transcript = (
                transcripts_dir / f"participant_{participant_id:03d}.md"
            ).open("w", encoding="utf-8")
            transcript.write(
                f"# Participant {participant_id} transcript ({model_name})\n\n"
            )
        try:
            for trial_index, stimulus in enumerate(stimuli):
                # A malformed stimulus must fail here, at its source: defaulting
                # a missing key to "" would show the participant a blank option
                # and record an empty sequence as if it were a real trial.
                missing = [
                    key
                    for key in ("sequence_a", "sequence_b")
                    if not stimulus.get(key)
                ]
                if missing:
                    raise ValueError(
                        f"stimulus {trial_index} is missing/empty {missing}: "
                        f"{stimulus!r}. Fix the design's stimuli.json rather than "
                        "presenting a blank option."
                    )
                # Randomize which sequence is shown on the left per trial; the
                # presented order is what we show AND what we record (a = left).
                swap = prng.random() < 0.5
                left, right = _present_sides(
                    str(stimulus["sequence_a"]),
                    str(stimulus["sequence_b"]),
                    swap,
                )
                user_msg = (
                    "Stimulus pair (left vs right):\n"
                    f"  Left:  {left}\n"
                    f"  Right: {right}\n\n"
                    "Reply with exactly one line: `ANSWER: left` or `ANSWER: right`."
                )
                # One retry per trial: the downstream completeness gate rejects
                # any collection with a missing response, so a single transient
                # backend error or uncommitted reply would otherwise abort the
                # whole (paid) collection. ``swap`` was drawn before the
                # attempts, so a retry re-presents the identical trial and the
                # counterbalancing stays reproducible.
                choice = None
                for attempt in range(1, _TRIAL_ATTEMPTS + 1):
                    retry_note = "" if attempt == 1 else f" (retry {attempt - 1})"
                    try:
                        response = participant_model.answer(prompt_text, user_msg)
                    except Exception as exc:
                        if transcript is not None:
                            transcript.write(
                                f"## Trial {trial_index}{retry_note}\n- left: `{left}`\n- right: `{right}`\n\n"
                                f"**Model error:** {exc}\n\n"
                            )
                        if attempt == _TRIAL_ATTEMPTS:
                            errors += 1
                        continue
                    choice = _parse_participant_answer(response)
                    if transcript is not None:
                        transcript.write(
                            f"## Trial {trial_index}{retry_note}\n- left: `{left}`\n- right: `{right}`\n\n"
                            f"**Reply:**\n```\n{response.strip()}\n```\n\n"
                            f"**Parsed:** {choice if choice else 'UNPARSEABLE'}\n\n"
                        )
                    if choice is not None:
                        break
                    if attempt == _TRIAL_ATTEMPTS:
                        unparseable += 1
                if choice is None:
                    continue
                chose_left = choice == "left"
                rows_p.append(
                    {
                        "participant_id": participant_id,
                        "trial_index": trial_index,
                        "sequence_a": left,
                        "sequence_b": right,
                        "chose_left": int(chose_left),
                        "chose_right": int(not chose_left),
                        "model": model_name,
                    }
                )
        finally:
            if transcript is not None:
                transcript.close()
        if progress is not None:
            progress(participant_id, len(rows_p))
        return rows_p, unparseable, errors

    results: list[tuple[list[dict[str, Any]], int, int] | None] = [None] * n_participants
    model_limit = getattr(participant_model, "max_concurrency", max_workers)
    if isinstance(model_limit, bool) or not isinstance(model_limit, int) or model_limit < 1:
        raise ValueError(
            f"participant model max_concurrency must be a positive integer, got "
            f"{model_limit!r}"
        )
    workers = max(1, min(max_workers, model_limit, n_participants))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_run_participant, pid): pid for pid in range(n_participants)
        }
        for future in as_completed(futures):
            results[futures[future]] = future.result()

    rows: list[dict[str, Any]] = []
    n_unparseable = 0
    n_errors = 0
    for result in results:
        assert result is not None  # every participant future populated its slot
        rows_p, unparseable, errors = result
        rows.extend(rows_p)
        n_unparseable += unparseable
        n_errors += errors

    stats = {
        "n_participants": n_participants,
        "n_stimuli": len(stimuli),
        "n_rows": len(rows),
        "n_unparseable": n_unparseable,
        "n_errors": n_errors,
    }
    return rows, stats


def _generate_from_pymc_models(
    stimuli: list[dict[str, Any]],
    model_names: list[str],
    n_participants: int,
    *,
    models_dir: Path,
    n_samples: int = 200,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Generate synthetic responses by sampling each model's prior-predictive p_left.

    Each participant is assigned a random cognitive model; for every stimulus the
    model's prior-predictive mean p_left is the choice probability for a binary
    draw. The side each sequence is shown on is randomized per trial (see
    :func:`_present_sides`), and the model is evaluated on the *presented* order so
    a generative ``side_bias`` biases toward the physical left. Each model
    computes its own features via its hooks. No MCMC fit — the prior is the
    generative process for synthetic participants.
    """
    from src.models.pymc_inference import prior_predict_p_left

    rng = random.Random(seed)

    p_left_cache: dict[tuple[str, int, bool], float] = {}

    def _raw_row(left: str, right: str) -> dict[str, Any]:
        return {"sequence_a": left, "sequence_b": right, "chose_left": 0}

    def _p_left(model_name: str, stim_idx: int, left: str, right: str, swap: bool) -> float:
        key = (model_name, stim_idx, swap)
        if key not in p_left_cache:
            preds = prior_predict_p_left(
                [model_name], models_dir, _raw_row(left, right),
                n_samples=n_samples, seed=seed,
            )
            p_left_cache[key] = preds[model_name]
        return p_left_cache[key]

    rows: list[dict[str, Any]] = []
    for participant_id in range(n_participants):
        model_name = rng.choice(model_names)
        for trial_index, stimulus in enumerate(stimuli):
            swap = rng.random() < 0.5
            left, right = _present_sides(
                stimulus["sequence_a"], stimulus["sequence_b"], swap
            )
            p_left = _p_left(model_name, trial_index, left, right, swap)
            chose_left = rng.random() < p_left
            rows.append(
                {
                    "participant_id": participant_id,
                    "trial_index": trial_index,
                    "sequence_a": left,
                    "sequence_b": right,
                    "chose_left": int(chose_left),
                    "chose_right": int(not chose_left),
                    "model": model_name,
                }
            )
    return rows


def _generate_from_models(
    stimuli: list[dict[str, Any]],
    model_names: list[str],
    n_participants: int,
    cognitive_models_dir: Path | None = None,
    model_registry: dict[str, Any] | None = None,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Generate synthetic responses from ground-truth models (non-PyMC path).

    Each participant draws a random model; each trial draws a binary choice
    from that model's predicted left-probability. If ``model_registry``
    maps ``model_name`` to a callable, that callable is used directly;
    otherwise ``get_model_predictions`` loads the model from
    ``cognitive_models_dir``. Uses a seeded RNG for reproducibility.
    """
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    for participant_id in range(n_participants):
        model_name = rng.choice(model_names)
        for trial_index, stimulus in enumerate(stimuli):
            # Randomize which sequence is shown on the left per trial; evaluate the
            # ground-truth model on the PRESENTED order and record it (a = left).
            swap = rng.random() < 0.5
            seq_a, seq_b = _present_sides(
                stimulus["sequence_a"], stimulus["sequence_b"], swap
            )
            stimulus_tuple = (seq_a, seq_b)
            if model_registry is not None and model_name in model_registry:
                fn = model_registry[model_name]
                preds = {model_name: fn(stimulus_tuple, RESPONSE_OPTIONS)}
            else:
                preds = get_model_predictions(
                    stimulus_tuple, RESPONSE_OPTIONS, [model_name], cognitive_models_dir
                )
            if not preds:
                # Backstop. get_model_predictions now raises rather than dropping
                # a model, so this fires only if a registry callable returns
                # nothing. Substituting a coin flip would emit pure noise labeled
                # as this model's ground-truth data and feed it straight into model
                # comparison. Fail loudly instead of fabricating data.
                raise RuntimeError(
                    f"ground-truth model {model_name!r} produced no prediction for "
                    f"stimulus {stimulus_tuple}; refusing to substitute random "
                    "responses. Check the model loads and returns a left/right "
                    "distribution."
                )
            distribution = validate_probability_distribution(
                preds[model_name],
                RESPONSE_OPTIONS,
                context=f"ground-truth model {model_name!r}",
            )
            p_left = distribution["left"]
            chose_left = rng.random() < p_left
            rows.append(
                {
                    "participant_id": participant_id,
                    "trial_index": trial_index,
                    "sequence_a": seq_a,
                    "sequence_b": seq_b,
                    "chose_left": int(chose_left),
                    "chose_right": int(not chose_left),
                    "model": model_name,
                }
            )
    return rows
