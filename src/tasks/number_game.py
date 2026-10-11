"""The number game: would a hidden program that produced a set of example
numbers also produce the target? (Bigelow & Piantadosi's 2016 dataset.)"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any, List, Mapping

import numpy as np

from src.runtime.config import PROJECT_ASSETS_DIR
from src.tasks import Row, TaskSpec, bernoulli_simulation

NUMBER_SETS_PATH = PROJECT_ASSETS_DIR / "number_game" / "number_sets.json"
TARGETS = range(1, 101)
NOVELTY_POOL_SEED = 20261011
NOVELTY_POOL_N_CELLS = 512

# The exploratory slots' lenses (see src/pipelines/inner_loop/candidate_agent.py):
# one mechanism per model, each a different way into the hypothesis space.
CANDIDATE_HINTS = (
    "Propose a mechanism from a genuinely different psychological process "
    "family than anything in the current set — one the current models cannot "
    "express, not a variant of them.",
    "Propose a single mechanism whose predictions would disagree most sharply "
    "with the current best model somewhere in the (set, target) space — and say "
    "in your hypothesis where that disagreement lives.",
    "Build a mechanism around information the current models ignore. Derive "
    "the exact statistic your hypothesis needs from the raw columns with "
    "`compute_features` or `prepare_observed` (the examples' order, spacing, "
    "digits, magnitude, how many there are).",
    "Propose a process-level account — a memory limit, attention to some "
    "examples over others, a cost of considering more complex concepts, or a "
    "sequential update as the examples are read — rather than another "
    "summary of the stimulus.",
    "Take a normative account (e.g. Bayesian inference over candidate "
    "concepts) and add exactly one principled distortion: a bias, a resource "
    "limit, or a mis-weighting.",
    "Propose something simpler or higher-variance than anything in the set — "
    "if it is wrong, the model comparison will say so loudly, and that is "
    "informative.",
    "Propose a mechanism in which the order of the examples matters: the first "
    "or the last example is weighted more, or each new example revises a "
    "running guess about the concept.",
    "Propose a mechanism built on similarity: the target is judged by how close "
    "it is to the examples on some explicit metric (numerical distance, shared "
    "digits, parity, magnitude band), with the metric and its gradient stated "
    "as the single mechanism.",
    "Propose a mechanism at the level of the decision rule rather than the "
    "inference: a lapse rate, a bias toward answering yes or no, a threshold on "
    "how sure one must be, or probability matching — with the inference itself "
    "kept as simple as the simplest current model.",
    "Propose a change to the space of concepts people consider: concepts the "
    "current models lack (arithmetic progressions, digit sums, near-misses of a "
    "rule, unions or exceptions), stated explicitly with how likely each is a "
    "priori.",
    "Propose an exemplar or prototype account: people compare the target to "
    "stored examples or prototypes of number concepts rather than inferring a "
    "concept — define the stored items and the comparison explicitly, as the "
    "single mechanism.",
)


@lru_cache(maxsize=1)
def number_sets() -> List[str]:
    """The dataset's 255 example sets, as comma-separated strings in shown order."""
    return [",".join(map(str, s)) for s in json.loads(NUMBER_SETS_PATH.read_text())]


def _design_pool() -> List[Row]:
    # Strings, as the responses CSV gives them: a model's hooks see one type.
    return [{"number_set": s, "target": str(t)} for s in number_sets() for t in TARGETS]


def _novelty_pool() -> List[Row]:
    pool = _design_pool()
    picks = np.random.default_rng(NOVELTY_POOL_SEED).choice(len(pool), NOVELTY_POOL_N_CELLS, replace=False)
    return [pool[i] for i in sorted(picks)]


def _eval_pool(cfg: Mapping[str, Any]) -> List[Row]:
    return _design_pool()


def _simulate(predict, stimuli, n_participants, rng, participant_id_offset) -> List[Row]:
    return bernoulli_simulation(predict, stimuli, n_participants, rng, participant_id_offset, NUMBER_GAME)


NUMBER_GAME = TaskSpec(
    name="number_game",
    raw_columns=("number_set", "target", "participant_id", "trial_index", "rating"),
    stimulus_columns=("number_set", "target"),
    response_column="rating",
    prediction_var="p_yes",
    stimulus_key=lambda row: (str(row["number_set"]), int(row["target"])),
    stimulus_label=lambda row: f"{{{row['number_set']}}} -> {row['target']}",
    probe_row={"number_set": "16,8", "target": "4", "participant_id": "0", "trial_index": "0", "rating": "0"},
    design_pool=_design_pool,
    novelty_pool=_novelty_pool,
    eval_pool=_eval_pool,
    simulate=_simulate,
    candidate_hints=CANDIDATE_HINTS,
    theory_prompt=PROJECT_ASSETS_DIR / "number_game" / "prompts" / "pymc_theory.md",
    critique_prompt=PROJECT_ASSETS_DIR / "number_game" / "prompts" / "critique.md",
)
