"""Acceptance tests for the Sikos et al. (2021) ingest (``src/rsa/ingest/sikos2021.py``).

The committed CSV reproduces every N the paper reports: per experiment the
recruited participants, the exclusions by reason (counted in the paper's
order: language, then attention, then literal listener choice) and the
remaining participants per task.
"""

from __future__ import annotations

import hashlib
import json

import pandas as pd
import pytest

from src.rsa.dataset import load_forced_choice
from src.rsa.ingest.sikos2021 import SIKOS_2021
from tests.rsa_ingest_fixtures import build_csv

CSV = SIKOS_2021.csv_path()

# (recruited, non-native/non-fluent, attention, literal; kept per task)
PAPER = {
    "sikos2021_e1": (4642, 1137, 118, 13, {"speaker": 1143, "listener": 1098, "salience": 1133}),
    "sikos2021_e2": (1671, 142, 77, 12, {"listener": 960, "salience": 480}),
    "sikos2021_e3": (1175, 265, 96, 3, {"listener": 405, "salience": 406}),
}


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    return pd.read_csv(CSV)


@pytest.mark.parametrize("experiment", sorted(PAPER))
def test_reproduces_the_papers_ns(df, experiment):
    recruited, language, attention, literal, kept = PAPER[experiment]
    e = df[df.experiment == experiment]
    assert len(e) == e.participant_id.nunique() == recruited
    reasons = e.exclusion_reason.fillna("")
    first = reasons.str.split(";").str[0]
    assert (first == "non_native_or_non_fluent").sum() == language
    assert (first == "attention_check_failed").sum() == attention
    assert (first == "listener_choice_not_literally_true").sum() == literal
    assert e[e.included].groupby("batch").size().to_dict() == kept


def test_tasks_map_to_dv_and_query(df):
    by_task = df.groupby("batch")[["dv", "query"]].agg(lambda s: sorted(set(s))).to_dict("index")
    assert by_task == {
        "listener": {"dv": ["forced_choice"], "query": ["utterance"]},
        "salience": {"dv": ["forced_choice"], "query": ["prior"]},
        "speaker": {"dv": ["production"], "query": ["production"]},
    }


def test_speaker_rows_keep_the_word_and_the_target(df):
    sp = df[df.dv == "production"]
    assert len(sp) == 1550 and (sp.experiment == "sikos2021_e1").all()
    for _, row in sp.iterrows():
        response = json.loads(row.response)
        objects = json.loads(row.objects)
        names = json.loads(row.feature_names)
        assert names[response["feature"]] == response["word"] in response["options"]
        assert objects[int(row.referent)][response["feature"]] == 1
        assert json.loads(row.object_roles)[int(row.referent)] == "target"


def test_every_feature_is_nameable_and_included_listeners_chose_literally(df):
    for _, row in df.iterrows():
        assert json.loads(row.messages) == list(range(len(json.loads(row.feature_names))))
    li = df[(df.batch == "listener") & df.included]
    objects = li.objects.map(json.loads)
    assert all(o[int(c)][int(u)] for o, c, u in zip(objects, li.choice, li.utterance))


def test_no_free_text_or_browser_strings(df):
    assert set(json.loads(df.covariates.iloc[0])) <= {"display", "side", "stimulus_type", "task_likelihood"}
    assert not df.astype(str).apply(lambda s: s.str.contains("Mozilla")).any().any()


def test_loads_as_forced_choice_trials():
    trials = load_forced_choice(CSV)
    assert len(trials.contexts) == 1098 + 1133 + 960 + 480 + 405 + 406
    assert sum(c.utterance is None for c in trials.contexts) == 1133 + 480 + 406


def test_experiment_2_codes_both_stimulus_types(df):
    e2 = df[df.experiment == "sikos2021_e2"]
    assert set(e2["item"]) == {"iconic", "geometric"}
    assert set(e2.condition) == {"d.2s2c.b.c", "d.2s2c.b.s"}


def test_rebuilding_from_the_pinned_files_reproduces_the_committed_csv(tmp_path):
    path = build_csv(SIKOS_2021, tmp_path)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == hashlib.sha256(CSV.read_bytes()).hexdigest()
