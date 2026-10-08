"""Simulated participants for the RSA outer loop, through the real page path.

A simulated participant runs one trial list exactly as the page would (the
practice trial, the designed displays with their random items, words and
screen order, the catch trials), choosing on each trial from a ground-truth
model's choice-class probabilities for that display, then clicking a screen
position that shows an object of the chosen class. Its record is the page's
jsPsych data (`page_records`), converted by `src.rsa.experiment.convert` as
live data are, so the simulated pipeline exercises the same bookkeeping.

The ground truth is a model file fitted to the data so far; every simulated
participant answers from its posterior-mean probabilities (the models are
population-level: one set of parameters for everyone).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Sequence

import numpy as np
import pandas as pd

from src.rsa.experiment.convert import convert
from src.rsa.experiment.design import trial_context
from src.rsa.fit import RSAFit, mean_probs, posterior_flat
from src.rsa.model_file import RSAModel

RECORD_KEYS = (
    "phase", "trial_number", "n_test_trials", "condition", "is_catch", "catch_target", "spec_index", "item",
    "feature_names", "objects", "roles", "utterance", "word", "query", "messages", "display_order", "bases",
)  # fmt: skip


def page_records(doc: dict, list_index: int, positions: Sequence[int], participant_id: str = "SIM") -> List[dict]:
    """The page's jsPsych data for a participant on list ``list_index`` who clicks
    screen position ``positions[i]`` on trial i (as `template.html` writes it)."""
    lst = doc["lists"][list_index]
    common = dict(participant_id=participant_id, list_index=list_index, list_seed=lst["seed"], list_assignment="url",
                  design_name=doc["design_name"], design_sha256=doc["design_sha256"])
    records = [dict(common, task=task, response=0, rt=900) for task in ("welcome", "consent", "instructions")]
    for trial, pos in zip(lst["trials"], positions):
        records.append(dict(common, task="rsa_choice", **{k: trial[k] for k in RECORD_KEYS}, response=int(pos), rt=1500,
                            choice_position=int(pos), choice=trial["display_order"][int(pos)],
                            chosen_alt=trial["screen"][int(pos)]["alt"], trial_type="html-button-response"))
        if trial["phase"] == "practice":
            records.append(dict(common, task="practice_done", response=0, rt=500))
    for j, r in enumerate(records):
        r["trial_index"], r["time_elapsed"] = j, 1000 * j
    return records


def simulate_participants(doc: dict, model_path: Path, fitted: RSAFit, n_participants: int, *, experiment: str,
                          seed: int, first_id: int = 0) -> pd.DataFrame:
    """Canonical rows for ``n_participants`` simulated people, participant k on list k mod n_lists."""
    model = RSAModel(model_path)
    flat = posterior_flat(fitted)
    rng = np.random.default_rng(seed)
    lists = doc["lists"]
    contexts = [trial_context(t) for lst in lists for t in lst["trials"]]
    probs = iter(mean_probs(model, flat, contexts, 200, by_class=True))
    by_list = [[next(probs) for _ in lst["trials"]] for lst in lists]
    frames = []
    for k in range(n_participants):
        li = k % len(lists)
        positions = []
        for trial, p in zip(lists[li]["trials"], by_list[li]):
            ctx = trial_context(trial)
            classes = ctx.choice_classes()
            p = np.clip(np.asarray(p, float), 0, None)
            cls = int(rng.choice(len(p), p=p / p.sum()))
            members = [o for o in range(len(classes)) if classes[o] == cls]
            obj = members[int(rng.integers(len(members)))]
            positions.append(trial["display_order"].index(obj))
        records = page_records(doc, li, positions)
        frames.append(convert(json.dumps(records), participant_id=f"sim{first_id + k}", experiment=experiment))
    return pd.concat(frames, ignore_index=True)
