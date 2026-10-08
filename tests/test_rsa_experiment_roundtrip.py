"""From a designed display to the screen, the click, the data row and back (PI request 2026-10-08).

Every display is shown with a random item, random words for its columns, a
random screen order and random base images. None of that may reach a model:
the row a click becomes must be the designed display, with the clicked
object's canonical index, so the model's probability of the response is the
probability of the abstract object that was clicked. And the counts per
designed display and choice class must be the clicks, tallied in the design's
terms.
"""

import json
from collections import Counter

import numpy as np

from src.rsa.dataset import context_from_row
from src.rsa.design.counts import display_counts
from src.rsa.design.run import design_pool
from src.rsa.experiment.convert import convert
from src.rsa.experiment.design import Design, TrialSpec, trial_lists
from src.rsa.fit import mean_probs
from src.rsa.model_file import RSAModel
from src.runtime.config import PROJECT_ASSETS_DIR

SEEDS = PROJECT_ASSETS_DIR / "rsa_reference" / "seed_models"
RECORD_KEYS = (
    "phase", "trial_number", "n_test_trials", "condition", "is_catch", "catch_target", "spec_index", "item",
    "feature_names", "objects", "roles", "utterance", "word", "query", "messages", "display_order", "bases",
)  # fmt: skip


def click(doc, k, positions):
    """jsPsych data of a participant on list k who clicks screen position positions[i] on trial i."""
    lst = doc["lists"][k]
    common = dict(participant_id="X", list_index=k, list_seed=lst["seed"], list_assignment="url",
                  design_name=doc["design_name"], design_sha256=doc["design_sha256"])
    records = []
    for i, (trial, pos) in enumerate(zip(lst["trials"], positions)):
        records.append(dict(common, task="rsa_choice", **{key: trial[key] for key in RECORD_KEYS}, response=pos, rt=900,
                            choice_position=pos, choice=trial["display_order"][pos], trial_type="html-button-response"))
        if trial["phase"] == "practice":
            records.append(dict(common, task="practice_done", response=0, rt=500))
    for j, r in enumerate(records):
        r["trial_index"], r["time_elapsed"] = j, 1000 * j
    return json.dumps(records)


def design_from_pool(n=30, seed=0):
    """Displays of every kind the EIG can pick: twins, 4x4 games, prior queries."""
    pool = design_pool()
    rng = np.random.default_rng(seed)
    picks = list(rng.choice(len(pool), size=n - 2, replace=False))
    picks.append(next(i for i, c in enumerate(pool) if len(set(c.objects)) < len(c.objects)))
    picks.append(next(i for i, c in enumerate(pool) if c.utterance is None and c.shape[0] == 4))
    specs = tuple(TrialSpec.from_context(pool[int(i)], label=f"eig_{j:02d}") for j, i in enumerate(picks))
    return Design(name="roundtrip", specs=specs)


def test_a_click_becomes_the_designed_display_and_the_clicked_abstract_object():
    design = design_from_pool()
    doc = trial_lists(design, seed=3, n_lists=10, n_catch=2, n_trials=12)  # 10 lists x 12 = 4 x 30
    model = RSAModel(SEEDS / "rsa_l1.py")
    params = {"alpha": np.array([2.0, 2.0]), "lapse": np.array([0.05, 0.05])}
    rng = np.random.default_rng(1)
    clicked = Counter()
    frames = []
    for k, lst in enumerate(doc["lists"]):
        positions = [int(rng.integers(len(t["objects"]))) for t in lst["trials"]]
        frame = convert(click(doc, k, positions), participant_id=f"p{k}", experiment="roundtrip")
        frames.append(frame)
        tests = [(t, p) for t, p in zip(lst["trials"], positions) if t["phase"] == "test"]
        for (_, row), (trial, pos) in zip(frame.iterrows(), tests):
            row_ctx = context_from_row(row)
            if trial["is_catch"]:
                continue
            spec = design.specs[trial["spec_index"]]
            abstract = spec.context()
            obj = trial["display_order"][pos]  # the canonical object at the clicked position
            assert int(row["choice"]) == obj
            # The model sees the same display: every array it reads is equal.
            for a, b in zip(row_ctx.arrays(), abstract.arrays()):
                assert np.array_equal(np.asarray(a), np.asarray(b))
            # And gives the response the probability of the abstract object clicked.
            p_row = mean_probs(model, params, [row_ctx], 2, by_class=True)[0]
            p_abs = mean_probs(model, params, [abstract], 2, by_class=True)[0]
            cls = abstract.choice_classes()[obj]
            assert row_ctx.choice_classes()[int(row["choice"])] == cls
            assert np.isclose(p_row[cls], p_abs[cls])
            clicked[(trial["spec_index"], cls)] += 1
    # The counts per designed display and class are the clicks.
    import pandas as pd

    counts = display_counts(pd.concat(frames), design)
    assert counts.sum() == sum(clicked.values()) == 10 * 12
    for (i, cls), n in clicked.items():
        assert counts[i, cls] == n
    assert set(counts.sum(1)) == {4}  # balanced: every display seen by 4 of the 10 lists


def test_items_words_and_positions_vary_for_one_display():
    design = design_from_pool()
    doc = trial_lists(design, seed=4, n_lists=60, n_catch=0, n_trials=12)
    shown = [t for lst in doc["lists"] for t in lst["trials"] if t["phase"] == "test" and t["spec_index"] == 0]
    assert len({t["item"] for t in shown}) > 1
    assert len({tuple(t["feature_names"]) for t in shown}) > 1
    assert len({tuple(t["display_order"]) for t in shown}) > 1
