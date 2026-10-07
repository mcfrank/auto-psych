"""Acceptance tests for the Mayn & Demberg ingest (``src/rsa/ingest/mayn_demberg.py``).

* 2026 (PLoS One, CC-BY; CSV committed): the paper's participant counts
  (300 in the first session, 8 below 80% on the unambiguous trials, 292 left,
  23 and 15 of those excluded for misunderstood_instr / odd_one_out, N = 254),
  its strategy-annotation counts and its "70% vs 30%" around the diagonal of
  Fig 3; these run in CI.
* 2023 (Open Mind) and 2022 (CogSci), whose repositories have no licence: the
  CSV lives only in the gitignored cache, so these skip unless it was built
  (``uv run python -m src.rsa.ingest.run --sources ...``) or
  ``RSA_INGEST_FETCH=1``.
* Rebuilding any of them from the pinned raw files reproduces the pinned
  sha256 (skips without the raw files, the same way).
"""

from __future__ import annotations

import hashlib
import json

import pandas as pd
import pytest

from src.rsa.dataset import load_forced_choice
from src.rsa.ingest.common import check_frame
from src.rsa.ingest.mayn_demberg import (
    ALL_MESSAGES,
    MAYN_DEMBERG_2022,
    MAYN_DEMBERG_2023,
    MAYN_DEMBERG_2026,
    ORIGINAL_MESSAGES,
    REMAPPED_MESSAGES,
)
from tests.rsa_ingest_fixtures import build_csv, derived_csv

SOURCES = (MAYN_DEMBERG_2026, MAYN_DEMBERG_2023, MAYN_DEMBERG_2022)


def _participants(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("participant_id").agg(
        experiment=("experiment", "first"),
        included=("included", "first"),
        reason=("exclusion_reason", lambda s: s.fillna("").iloc[0]),
        covariates=("covariates", "first"),
        n=("trial_index", "size"),
    )


@pytest.fixture(scope="module")
def md2026() -> pd.DataFrame:
    return pd.read_csv(MAYN_DEMBERG_2026.csv_path())


@pytest.fixture(scope="module")
def md2023(tmp_path_factory) -> pd.DataFrame:
    return pd.read_csv(derived_csv(MAYN_DEMBERG_2023, tmp_path_factory.mktemp("md2023")))


@pytest.fixture(scope="module")
def md2022(tmp_path_factory) -> pd.DataFrame:
    return pd.read_csv(derived_csv(MAYN_DEMBERG_2022, tmp_path_factory.mktemp("md2022")))


# ---------------------------------------------------------------- 2026


def test_2026_reproduces_the_papers_participant_counts(md2026):
    p = _participants(md2026)
    assert (p.n == 66).all()
    paper_sample = p[~p.reason.str.contains("not_in_paper_sample")]
    assert len(p) == 306 and len(paper_sample) == 300
    accuracy_fail = paper_sample.reason.str.contains("unambiguous_accuracy_below_0.8")
    assert accuracy_fail.sum() == 8
    kept_292 = paper_sample[~accuracy_fail]
    assert len(kept_292) == 292
    misunderstood = kept_292.reason.str.contains("strategy_misunderstood_instr")
    odd = kept_292.reason.str.contains("strategy_odd_one_out")
    assert misunderstood.sum() == 23
    assert (odd & ~misunderstood).sum() == 15  # one participant has both tags
    assert p.included.sum() == 254
    # The scripts' main_exclusions() ignores sample membership and keeps 259.
    scripts_rule = p[~p.reason.str.contains("unambiguous|strategy_")]
    assert len(scripts_rule) == 259


def test_2026_strategy_annotations_match_the_paper(md2026):
    p = _participants(md2026)
    kept_292 = p[~p.reason.str.contains("not_in_paper_sample|unambiguous_accuracy")]
    tags = pd.DataFrame([json.loads(c) for c in kept_292.covariates])
    # "50.7% (148 out of 292) vs. 34.2% (100 out of 292)" correct_reasoning.
    assert (tags.strategy_tag_simple == "correct_reasoning").sum() == 148
    assert (tags.strategy_tag_complex == "correct_reasoning").sum() == 100
    # The paper's guess counts are 73 and 129; the published annotations
    # give 72 and 130 (one participant's simple/complex guess tags differ).
    assert (tags.strategy_tag_simple == "guess").sum() == 72
    assert (tags.strategy_tag_complex == "guess").sum() == 130


def test_2026_seventy_percent_of_participants_are_on_or_below_the_diagonal(md2026):
    inc = md2026[md2026.included & md2026.condition.isin(["target simple", "target complex"])]
    target = [json.loads(r)[c] == "target" for r, c in zip(inc.object_roles, inc.choice)]
    by = inc.assign(target=target).groupby(["participant_id", "condition"]).target.mean().unstack()
    share = (by["target complex"] <= by["target simple"]).mean()
    assert round(share * 100) == 70


def test_2026_item_structures_and_message_set(md2026):
    assert set(md2026.messages) == {json.dumps(list(ORIGINAL_MESSAGES)).replace(" ", "")}
    first = md2026.drop_duplicates("item")
    assert len(first) == 66
    counts = first.condition.value_counts().to_dict()
    assert counts == {
        "target simple": 12, "target complex": 12, "filler type c": 12, "filler ambiguous": 9,
        "filler unambiguous": 9, "filler type a": 6, "filler type b": 6,
    }
    # Every item shows the same display to everyone (objects, word).
    assert md2026.groupby("item")[["objects", "utterance"]].nunique().max().max() == 1
    # No 2026 label needed correcting (its errors are in targetpos/answer_which, unused).
    assert md2026.notes.isna().all()


def test_2026_loads_as_forced_choice_trials(md2026):
    trials = load_forced_choice(MAYN_DEMBERG_2026.csv_path())
    assert len(trials.contexts) == 254 * 66
    simple = [c for c, (_, r) in zip(trials.contexts, trials.frame.iterrows()) if r.condition == "target simple"]
    # Square and blue have no word: the simple target cannot be named alone.
    assert {c.utterance_names[-1] for c in simple} <= {"red", "green", "triangle", "circle", "<sink>"}


# ---------------------------------------------------------------- 2023


def test_2023_reproduces_the_papers_ns(md2023):
    p = _participants(md2023)
    assert (p.n == 66).all()
    total = p.groupby("experiment").size().to_dict()
    kept = p[p.included].groupby("experiment").size().to_dict()
    # Table 1: replication (57), remapped (55), all messages (56), shapes (60).
    assert total == {"md2023_e1_replication": 59, "md2023_e2_remapped": 59,
                     "md2023_e3_all_messages": 59, "md2023_e4_shapes": 60}
    assert kept == {"md2023_e1_replication": 57, "md2023_e2_remapped": 55,
                    "md2023_e3_all_messages": 56, "md2023_e4_shapes": 60}


def test_2023_annotation_exclusions_match_the_paper(md2023):
    p = _participants(md2023)
    p = p[p.included]
    tags = pd.DataFrame([json.loads(c) for c in p.covariates], index=p.index).assign(experiment=p.experiment)

    def n(exp, kind, tag):
        return int((tags[tags.experiment == exp][f"strategy_tag_{kind}"] == tag).sum())

    # "1 response labeled exclude and 8 responses labeled unclear out of 57" (Exp. 1, simple)
    assert (n("md2023_e1_replication", "simple", "exclude"), n("md2023_e1_replication", "simple", "unclear")) == (1, 8)
    # Exp. 2: simple 4 exclude + 6 unclear; complex 3 exclude + 10 unclear (of 55)
    assert (n("md2023_e2_remapped", "simple", "exclude"), n("md2023_e2_remapped", "simple", "unclear")) == (4, 6)
    assert (n("md2023_e2_remapped", "complex", "exclude"), n("md2023_e2_remapped", "complex", "unclear")) == (3, 10)
    # Exp. 3: simple 1 exclude + 7 unclear; complex 6 unclear (of 56)
    assert (n("md2023_e3_all_messages", "simple", "exclude"), n("md2023_e3_all_messages", "simple", "unclear")) == (1, 7)
    assert (n("md2023_e3_all_messages", "complex", "exclude"), n("md2023_e3_all_messages", "complex", "unclear")) == (0, 6)


def test_2023_message_sets_and_item_20_relabelling(md2023):
    msgs = md2023.groupby("experiment").messages.unique().to_dict()
    assert {k: [json.loads(x) for x in v] for k, v in msgs.items()} == {
        "md2023_e1_replication": [list(ORIGINAL_MESSAGES)],
        "md2023_e2_remapped": [list(REMAPPED_MESSAGES)],
        "md2023_e3_all_messages": [list(ALL_MESSAGES)],
        "md2023_e4_shapes": [list(ORIGINAL_MESSAGES)],
    }
    relabelled = md2023[md2023.notes.fillna("").str.contains("labelled_target_is_not_the_feature_target")]
    assert set(relabelled["item"].str.split("_item").str[1]) == {"20"}
    assert len(relabelled) == 59 * 3 + 60
    # Exps. 1 and 4 are the same game drawn differently: identical displays per item.
    e1 = md2023[md2023.experiment == "md2023_e1_replication"].drop_duplicates("item")
    e4 = md2023[md2023.experiment == "md2023_e4_shapes"].drop_duplicates("item")
    key = lambda d: dict(zip(d["item"].str.split("_item").str[1], zip(d.objects, d.utterance)))
    assert key(e1) == key(e4)


def test_2023_meets_the_contract_and_loads(md2023, tmp_path):
    check_frame(md2023, "mayn_demberg_2023")
    path = tmp_path / "md2023.csv"
    md2023.to_csv(path, index=False)
    assert len(load_forced_choice(path).contexts) == (57 + 55 + 56 + 60) * 66


# ---------------------------------------------------------------- 2022


def test_2022_has_the_pilot_and_main_samples(md2022):
    p = _participants(md2022)
    assert p.groupby("experiment").size().to_dict() == {"md2022_main": 68, "md2022_pilot": 47}
    assert p.included.all()
    assert set(md2022.batch) == {"pilot", "main"}


def test_2022_corrects_the_item_20_and_51_labels(md2022):
    relabelled = md2022[md2022.notes.fillna("").str.contains("labelled_target_is_not_the_feature_target")]
    by_item = relabelled.groupby("item").experiment.value_counts().to_dict()
    assert by_item == {
        ("monsters_item20", "md2022_main"): 68,
        ("monsters_item51", "md2022_main"): 68,
        ("monsters_item51", "md2022_pilot"): 47,
    }


def test_2022_meets_the_contract_and_loads(md2022, tmp_path):
    check_frame(md2022, "mayn_demberg_2022")
    path = tmp_path / "md2022.csv"
    md2022.to_csv(path, index=False)
    assert len(load_forced_choice(path).contexts) == 115 * 66


# ---------------------------------------------------------------- rebuilding


@pytest.mark.parametrize("source", SOURCES, ids=lambda s: s.name)
def test_rebuilding_from_the_pinned_files_reproduces_the_pinned_csv(source, tmp_path):
    path = build_csv(source, tmp_path)
    pinned = json.loads(source.provenance_path.read_text())["output"]["sha256"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == pinned
