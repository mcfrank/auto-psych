"""The external-dataset ingest's shared machinery: pins, the column contract, combining.

These run in CI: they read only committed files (pragmods and the CC-BY
sources) and write only to temporary directories.
"""

from __future__ import annotations

import json
import hashlib

import pandas as pd
import pytest

from src.rsa import pragmods_ingest
from src.rsa.dataset import DEFAULT_TRIALS_CSV, load_forced_choice
from src.rsa.ingest import combine as combine_mod
from src.rsa.ingest.common import COLUMNS, check_frame
from src.rsa.ingest.fetch import PinMismatch, SourceFile, fetch
from src.rsa.ingest.mayn_demberg import MAYN_DEMBERG_2026, derive_roles
from src.rsa.ingest.run import NOT_INGESTED, SOURCES
from src.rsa.ingest.sikos2021 import SIKOS_2021
from tests.paths import REPO_ROOT

COMMITTED = (MAYN_DEMBERG_2026, SIKOS_2021)


def test_the_contract_is_the_pragmods_columns_plus_source_messages_covariates():
    assert COLUMNS == ["source", *pragmods_ingest.COLUMNS, "messages", "covariates"]


def test_a_cached_file_with_the_wrong_hash_raises_and_is_not_replaced(tmp_path):
    pinned = SourceFile("x.csv", "https://example.invalid/x.csv", hashlib.sha256(b"right").hexdigest(), 5)
    path = tmp_path / "src" / "raw" / "x.csv"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"wrong")
    with pytest.raises(PinMismatch, match="delete"):
        fetch(pinned, tmp_path, "src", offline=True)
    assert path.read_bytes() == b"wrong"


def test_a_missing_file_offline_raises(tmp_path):
    pinned = SourceFile("x.csv", "https://example.invalid/x.csv", "0" * 64, 1)
    with pytest.raises(FileNotFoundError, match="offline"):
        fetch(pinned, tmp_path, "src", offline=True)


def test_a_correctly_cached_file_is_used(tmp_path):
    pinned = SourceFile("x.csv", "https://example.invalid/x.csv", hashlib.sha256(b"ok").hexdigest(), 2)
    path = tmp_path / "src" / "raw" / "x.csv"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"ok")
    assert fetch(pinned, tmp_path, "src", offline=True) == path


@pytest.mark.parametrize("source", COMMITTED, ids=lambda s: s.name)
def test_committed_csvs_meet_the_contract_and_their_pin(source):
    path = source.csv_path()
    check_frame(pd.read_csv(path), source.name)
    prov = json.loads(source.provenance_path.read_text())
    assert prov["output"]["committed"] is True
    assert prov["output"]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert [f["sha256"] for f in prov["source_files"]] == [f.sha256 for f in source.files]


@pytest.mark.parametrize("source", [s for s in SOURCES.values() if not s.commit_csv], ids=lambda s: s.name)
def test_unlicensed_sources_commit_only_a_pin_file(source):
    assert not (REPO_ROOT / "src" / "pipelines" / "outer_loop" / "projects" / "rsa_reference" / "data"
                / source.csv_name).exists()
    prov = json.loads(source.provenance_path.read_text())
    assert prov["output"]["committed"] is False
    assert prov["pinned_commit"] and len(prov["output"]["sha256"]) == 64
    assert all(f["url"].startswith("https://raw.githubusercontent.com/") and prov["pinned_commit"] in f["url"]
               for f in prov["source_files"])


@pytest.mark.parametrize("source", COMMITTED, ids=lambda s: s.name)
def test_load_forced_choice_reads_every_included_listener_row(source):
    path = source.csv_path()
    df = pd.read_csv(path)
    trials = load_forced_choice(path)
    expected = df[(df.dv == "forced_choice") & df.included]
    assert len(trials.contexts) == len(expected) > 0
    for ctx, choice in zip(trials.contexts, trials.choices):
        assert 0 <= choice < len(ctx.objects)
        assert ctx.messages is not None
        if ctx.utterance is not None:
            assert ctx.utterance in ctx.messages
            assert any(obj[ctx.utterance] for obj in ctx.objects)


def test_combine_pragmods_with_the_committed_sources(tmp_path):
    combined = combine_mod.combine(["pragmods", "mayn_demberg_2026", "sikos_2021"])
    assert list(combined.columns) == COLUMNS
    assert combined["participant_id"].str.split(":").str[0].tolist() == combined["source"].tolist()
    out = tmp_path / "combined.csv"
    combined.to_csv(out, index=False)
    trials = load_forced_choice(out)
    parts = (DEFAULT_TRIALS_CSV, MAYN_DEMBERG_2026.csv_path(), SIKOS_2021.csv_path())
    assert len(trials.contexts) == sum(len(load_forced_choice(p).contexts) for p in parts)
    pragmods_rows = trials.frame[trials.frame["source"] == "pragmods"]
    assert pragmods_rows["messages"].isna().all()  # every feature is a word


def test_combine_refuses_a_duplicate_or_unknown_source():
    with pytest.raises(ValueError, match="twice"):
        combine_mod.combine(["sikos_2021", "sikos_2021"])
    with pytest.raises(ValueError, match="unknown source"):
        combine_mod.combine(["franke_degen_2016"])


def test_combine_names_the_command_for_an_unbuilt_source(tmp_path):
    with pytest.raises(FileNotFoundError, match="src.rsa.ingest.run --sources mayn_demberg_2023"):
        combine_mod.combine(["mayn_demberg_2023"], cache_dir=tmp_path)


def test_duff_et_al_is_documented_as_not_ingested_for_its_feedback():
    assert "duff_mayn_demberg_2026" not in SOURCES
    assert "feedback after every" in NOT_INGESTED["duff_mayn_demberg_2026"]


# derive_roles: shapes columns (circle, triangle, square, green, red, blue),
# original message set {circle, triangle, green, red}.
ORIG = (0, 1, 3, 4)


def _o(*features):
    v = [0] * 6
    for f in features:
        v[f] = 1
    return tuple(v)


def test_simple_implicature_target_is_the_object_without_a_nameable_unique_feature():
    # message red (4): red square (target, square unnameable) vs red triangle (triangle unique).
    objects = [_o(1, 4), _o(2, 4), _o(0, 3)]  # labelled order deliberately wrong
    assert derive_roles(objects, 4, ORIG) == (1, 0, 2)


def test_complex_implicature_target_shares_its_other_feature_with_the_distractor():
    # message red: red circle (target) / red triangle (competitor) / green circle (distractor)
    objects = [_o(0, 4), _o(1, 4), _o(0, 3)]
    assert derive_roles(objects, 4, ORIG) == (0, 1, 2)


def test_twins_and_an_unambiguous_filler_with_swapped_labels():
    twins = [_o(0, 3), _o(0, 3), _o(1, 4)]
    assert derive_roles(twins, 3, ORIG) == (0, 1, 2)
    # item 20 of the 2023 study: labelled target is really the distractor
    item20 = [_o(2, 5), _o(1, 3), _o(0, 4)]
    assert derive_roles(item20, 4, ORIG) == (2, 1, 0)
