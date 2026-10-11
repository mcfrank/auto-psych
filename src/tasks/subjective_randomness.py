"""The coin-flip task: which of two same-length H/T sequences looks more random."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping

import numpy as np

from src.runtime.config import REPO_ROOT
from src.tasks import Row, TaskSpec

DESIGN_LENGTHS = (2, 3, 4, 5, 6, 7, 8)
NOVELTY_POOL_SEED = 20260919
NOVELTY_POOL_N_PAIRS = 512
NOVELTY_POOL_LENGTHS = (4, 5, 6, 7, 8)

# The exploratory slots' lenses (see src/pipelines/inner_loop/candidate_agent.py).
CANDIDATE_HINTS = [
    "Propose a mechanism from a genuinely different psychological process "
    "family than anything in the current set — one the current models cannot "
    "express, not a variant of them.",
    "Propose a single mechanism whose predictions would disagree most sharply "
    "with the current best model somewhere in the stimulus space — and say in "
    "your hypothesis where that disagreement lives.",
    "Build a mechanism around information the current models ignore. Derive "
    "the exact statistic your hypothesis needs from the raw sequences with "
    "`compute_features` (order, position, recency, specific sub-sequences).",
    "Propose a process-level account — a memory limit, attention window, "
    "encoding cost, or sequential-updating process — rather than another "
    "statistical summary of the stimulus.",
    "Take a normative account (e.g. Bayesian inference over candidate "
    "generators) and add exactly one principled distortion: a bias, a "
    "resource limit, or a mis-weighting.",
    "Propose something simpler or higher-variance than anything in the set — "
    "if it is wrong, the model comparison will say so loudly, and that is "
    "informative.",
    "Propose a mechanism in which the two sequences are not judged one at a "
    "time: the comparison itself does the work — a contrast effect, anchoring "
    "on whichever sequence is read first, or a shared reference point both are "
    "judged against — so the same sequence would be judged differently beside "
    "a different partner.",
    "Propose a mechanism driven by the single most salient local feature — the "
    "longest run, the most lopsided window, or the presence of one specific "
    "motif — a maximum over sub-sequences rather than a sum or average over the "
    "whole sequence, so one striking stretch decides the judgment.",
    "Propose a mechanism at the level of the decision rule rather than the "
    "evidence: a lapse rate, a bias toward one response side, an indifference "
    "band within which the choice is a coin flip, or probability matching — "
    "with the evidence itself kept as simple as the simplest current model.",
    "Propose a mechanism in which people track a running tally as they read "
    "the sequence — the cumulative lead of heads over tails, or how far and how "
    "often it drifts from balance before correcting — so the path to the final "
    "count matters, not the count.",
    "Propose an exemplar account: people carry a few remembered examples of "
    "what random and designed sequences look like and judge a new sequence by "
    "its similarity to the nearest ones — define the prototypes and the "
    "similarity explicitly, as the single mechanism.",
]


def _design_pool() -> List[Row]:
    from src.subjective_randomness.stimulus_design import enumerate_all_pairs

    return enumerate_all_pairs(list(DESIGN_LENGTHS), same_length_only=True)


def _novelty_pool() -> List[Row]:
    from src.subjective_randomness.stimulus_design import generate_candidate_pool

    return generate_candidate_pool(NOVELTY_POOL_N_PAIRS, lengths=NOVELTY_POOL_LENGTHS, seed=NOVELTY_POOL_SEED)


def _eval_pool(cfg: Mapping[str, Any]) -> List[Row]:
    from src.subjective_randomness.stimulus_design import enumerate_all_pairs, generate_candidate_pool

    if cfg["exhaustive"]:
        return enumerate_all_pairs(list(cfg["lengths"]), same_length_only=True)
    return generate_candidate_pool(cfg["n_pairs"], lengths=tuple(cfg["lengths"]), seed=cfg["seed"])


def _simulate(predict, stimuli, n_participants, rng, participant_id_offset) -> List[Row]:
    """Left/right is counterbalanced as in the human experiments: each
    participant sees each pair in a random order (a fair coin per trial), the
    row records the pair as displayed, and ``chose_left`` is drawn from the
    ground truth's ``p_left`` for that displayed order."""
    swapped = [{"sequence_a": s["sequence_b"], "sequence_b": s["sequence_a"]} for s in stimuli]
    p_left = np.asarray(predict(SUBJECTIVE_RANDOMNESS.stimulus_rows(stimuli)))
    p_left_swapped = np.asarray(predict(SUBJECTIVE_RANDOMNESS.stimulus_rows(swapped)))
    rows: List[Dict[str, Any]] = []
    for participant in range(n_participants):
        swap = rng.random(len(stimuli)) < 0.5
        p_shown = np.where(swap, p_left_swapped, p_left)
        chose_left = rng.random(len(stimuli)) < p_shown
        for trial_index, stim in enumerate(stimuli):
            shown = swapped[trial_index] if swap[trial_index] else stim
            rows.append({
                "sequence_a": shown["sequence_a"],
                "sequence_b": shown["sequence_b"],
                "participant_id": participant_id_offset + participant,
                "trial_index": trial_index,
                "chose_left": int(chose_left[trial_index]),
            })
    return rows


SUBJECTIVE_RANDOMNESS = TaskSpec(
    name="subjective_randomness",
    raw_columns=("sequence_a", "sequence_b", "participant_id", "trial_index", "chose_left"),
    stimulus_columns=("sequence_a", "sequence_b"),
    response_column="chose_left",
    prediction_var="p_left",
    stimulus_key=lambda row: tuple(sorted((row["sequence_a"], row["sequence_b"]))),
    stimulus_label=lambda row: f"{row['sequence_a']} vs {row['sequence_b']}",
    probe_row={"sequence_a": "HHT", "sequence_b": "THT", "participant_id": "0", "trial_index": "0", "chose_left": "0"},
    design_pool=_design_pool,
    novelty_pool=_novelty_pool,
    eval_pool=_eval_pool,
    simulate=_simulate,
    candidate_hints=tuple(CANDIDATE_HINTS),
    theory_prompt=REPO_ROOT / "src" / "pipelines" / "inner_loop" / "prompts" / "pymc_theory.md",
    critique_prompt=REPO_ROOT / "src" / "pipelines" / "inner_loop" / "prompts" / "critique.md",
)
