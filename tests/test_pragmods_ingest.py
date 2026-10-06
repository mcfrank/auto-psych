"""Acceptance tests for the pragmods ingest (``src/rsa/pragmods_ingest.py``).

The committed ``pragmods_trials.csv`` is checked against the paper's own
analysis outputs, copied into ``data/reference/`` from langcog/pragmods
``models/data/``:

* ``levels.csv`` / ``prior.csv`` — the counts levels.Rmd and prior.Rmd write,
  by the experiment's own role labels (target / logical / foil);
* ``models.csv`` / ``prior_counts.csv`` — the same counts relabelled by hand
  onto the ``models/matrices.R`` matrices, which the paper's model fits read;

and against Table 1 (``tab:expts``) of ``writeup/pragmods.tex``. Every cell is
reproduced exactly except the ones listed (with the reason) in
``KNOWN_MODELS_CSV_RELABELLING`` and ``TABLE1_WITHOUT_DEDUPLICATION``.

These tests read only the committed files, so they run in CI. The tests that
regenerate the CSV need a clone of github.com/langcog/pragmods named by the
``PRAGMODS_DIR`` environment variable and skip without it.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import pytest

from src.rsa import pragmods_ingest as ingest
from tests.paths import REPO_ROOT

DATA_DIR = REPO_ROOT / "src" / "pipelines" / "outer_loop" / "projects" / "rsa_reference" / "data"
CSV_PATH = DATA_DIR / ingest.CSV_NAME
PROVENANCE_PATH = DATA_DIR / ingest.PROVENANCE_NAME
REFERENCE_DIR = DATA_DIR / "reference"
REFERENCE_FILES = ("models.csv", "prior_counts.csv", "levels.csv", "prior.csv")

JSON_COLUMNS = ("feature_names", "objects", "object_roles", "display_order")

# models/matrices.R: rows (referents) and columns (messages) of each matrix.
MATRICES_R: Dict[str, Tuple[Tuple[str, ...], Tuple[str, ...], Tuple[Tuple[int, ...], ...]]] = {
    "simple": (("foil", "target", "logical"), ("hat", "glasses"), ((0, 0), (0, 1), (1, 1))),
    "complex": (("foil", "target", "logical"), ("hat", "glasses", "mustache"), ((0, 0, 1), (0, 1, 1), (1, 1, 0))),
    "oddman": (("foil", "target", "logical"), ("hat", "glasses", "mustache"), ((1, 1, 0), (1, 0, 1), (0, 1, 1))),
    "twins": (("foil", "target", "logical"), ("hat", "glasses", "mustache"), ((0, 1, 1), (1, 0, 1), (1, 0, 1))),
}
# Which matrices.R message each canonical CSV feature column is. The CSV keeps
# the experiment code's column order; matrices.R names them for the face item.
# Oddman: the uttered column is "glasses"; the other two are fixed by
# levels.Rmd's twin_1 -> logical, twin_2 -> foil.
CSV_COLUMNS_AS_R = {
    "simple": ("hat", "glasses"),
    "complex": ("hat", "glasses", "mustache"),
    "twins": ("hat", "glasses", "mustache"),
    "oddman": ("hat", "mustache", "glasses"),
}

# models.csv cells whose hand relabelling (models/data/levels_mod.csv) does not
# follow the stimuli. The source data say, for complex level 1 ("mustache"):
# 44 chose M, 11 GM (mustache: literally true) and 3 HG (no mustache);
# models.csv puts the 11 on HG ("logical") and the 3 on GM ("target"). For
# simple level 0 ("hat"): 48 chose HG, 1 the featureless face, 0 G; models.csv
# puts the 1 on G ("target") and 0 on the featureless face ("foil"). In
# levels_mod.csv one cell was given the other's permutation (a swap where a
# rotation was needed and vice versa). Map: our object -> models.csv object.
KNOWN_MODELS_CSV_RELABELLING = {
    ("simple", "0"): {"foil": "target", "target": "foil", "logical": "logical"},
    ("complex", "1"): {"foil": "foil", "target": "logical", "logical": "target"},
}

# Table 1 of pragmods.tex: (N_total, N_include).
TABLE1 = {
    "E1_dv": (689, 554),
    "E2_manip_check": (580, 513),
    "E3_ling_frame": (100, 89),
    "E4_prior_frame": (200, 175),
    "E5_baserate": (800, 488),
    "E6_valence": (550, 502),
    "E7_color": (300, 267),
    "E8_levels": (416, 362),
    "E9_twins": (220, 194),
    "E10_oddman": (300, 270),
}
# For these Table 1 counts participants passing the manipulation and name
# checks without removing repeat participants; the analyses (and models.csv,
# which the ingest reproduces) do remove them. N_include = included + the
# participants excluded only as duplicates: 478+24, 264+3, 345+17, 193+1, 269+1.
TABLE1_WITHOUT_DEDUPLICATION = frozenset({"E6_valence", "E7_color", "E8_levels", "E9_twins", "E10_oddman"})

# The commented-out rows of tab:expts, (N_total, N_include) per participant.
# Their sequence labels are swapped: "Level 1 x3 200/193" is the two level-2
# files and "Level 2 100/93" the two x3 files. The production "Level 1" N_total
# of 450 is nominal; its three files hold 453 participants.
COMMENTED_ROWS = {
    "size": ({"size"}, None, (1750, 1368)),
    "seqs-1": ({"sequences"}, {"4-sequences/pragmods_seq.anondata.csv"}, (200, 191)),
    "seqs-level2 (labelled 1x3)": (
        {"sequences"},
        {"4-sequences/pragmods_seq2.anondata.csv", "4-sequences/pragmods_L2second.anondata.tsv"},
        (200, 193),
    ),
    "seqs-x3 (labelled 2)": (
        {"sequences"},
        {"4-sequences/pragmods_wx3.anondata.csv", "4-sequences/pragmods_bx3.anondata.csv"},
        (100, 93),
    ),
    "prod-1": ({"speakers"}, {"L1"}, (453, 383)),
    "prod-1seq": ({"speakers"}, {"seq_L1"}, (453, 394)),
    "prod-2seq": ({"speakers"}, {"seq_L2"}, (450, 389)),
}


# --------------------------------------------------------------------------
# fixtures and helpers
# --------------------------------------------------------------------------


def _load_trials(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    for col in JSON_COLUMNS:
        df[col] = df[col].map(json.loads)
    for col in ("familiarization", "grayscale", "response"):
        df[col] = df[col].map(lambda v: json.loads(v) if v else None)
    for col in ("utterance", "choice", "referent"):
        df[col] = pd.Series([int(v) if v else None for v in df[col]], index=df.index, dtype=object)
    df["trial_index"] = df["trial_index"].astype(int)
    if not set(df["included"]) <= {"True", "False"}:
        raise ValueError(f"included is not boolean: {sorted(set(df['included']))}")
    df["included"] = df["included"] == "True"
    return df


@pytest.fixture(scope="module")
def trials() -> pd.DataFrame:
    return _load_trials(CSV_PATH)


def _reference(name: str) -> pd.DataFrame:
    return pd.read_csv(REFERENCE_DIR / name)


def _pragmods_dir() -> Path:
    value = os.environ.get("PRAGMODS_DIR")
    if not value:
        pytest.skip("PRAGMODS_DIR is not set: no clone of github.com/langcog/pragmods to regenerate from")
    path = Path(value)
    if not (path / "data").is_dir():
        pytest.skip(f"PRAGMODS_DIR={value} is not a clone of langcog/pragmods (no data/)")
    return path


def _paper_matrices() -> Dict[str, str]:
    """paper_cond -> matrices.R matrix, from models.csv's own matrix column."""
    models = _reference("models.csv")
    out = {}
    for row in models.itertuples():
        out[f"models:{row.expt}/{row.cond}"] = row.matrix
        out[f"prior_counts:{row.prior}"] = row.matrix
    return out


def _as_r_vector(matrix: str, vec: List[int]) -> Tuple[int, ...]:
    cols = CSV_COLUMNS_AS_R[matrix]
    r_cols = MATRICES_R[matrix][1]
    return tuple(vec[cols.index(c)] for c in r_cols)


def _semantic_counts(trials: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Included forced choices per paper cell, by matrices.R object.

    The chosen object is matched to the matrices.R row(s) with the same
    features; a choice of one of two identical objects (twins) counts half
    for each, as models.csv does.
    """
    matrices = _paper_matrices()
    counts: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    rows = trials[(trials.paper_cond != "") & trials.included]
    assert (rows.dv == "forced_choice").all()
    for row in rows.itertuples():
        matrix = matrices[row.paper_cond]
        names, _, r_rows = MATRICES_R[matrix]
        vec = _as_r_vector(matrix, row.objects[row.choice])
        match = [n for n, r in zip(names, r_rows) if r == vec]
        assert match, f"{row.source_file} {row.participant_id}: chosen object {vec} is not a {matrix} object"
        for n in match:
            counts[row.paper_cond][n] += 1 / len(match)
    return counts


# --------------------------------------------------------------------------
# (a) the paper's counts
# --------------------------------------------------------------------------


def test_reference_copies_are_the_paper_files() -> None:
    clone = _pragmods_dir()
    for name in REFERENCE_FILES:
        assert (REFERENCE_DIR / name).read_bytes() == (clone / "models" / "data" / name).read_bytes(), name


def test_every_paper_cell_uses_its_matrices_r_matrix(trials: pd.DataFrame) -> None:
    """Each paper cell's display is its matrices.R matrix and its word the cell's query."""
    matrices = _paper_matrices()
    models = _reference("models.csv").drop_duplicates(["expt", "cond"])
    query = {f"models:{r.expt}/{r.cond}": r.query for r in models.itertuples()}
    rows = trials[trials.paper_cond != ""]
    assert set(rows.paper_cond) == set(matrices), "paper cells with no rows, or rows with an unknown cell"
    for cell, group in rows.groupby("paper_cond"):
        matrix = matrices[cell]
        for objects in {json.dumps(o) for o in group.objects}:
            got = sorted(_as_r_vector(matrix, o) for o in json.loads(objects))
            assert got == sorted(MATRICES_R[matrix][2]), (cell, objects)
        if cell in query:
            utt = group.utterance.unique()
            assert len(utt) == 1, cell
            assert CSV_COLUMNS_AS_R[matrix][utt[0]] == query[cell], cell
        else:
            assert (group["query"] == "prior").all(), cell


def test_aggregation_reproduces_models_csv(trials: pd.DataFrame) -> None:
    counts = _semantic_counts(trials)
    models = _reference("models.csv")
    mismatched = set()
    for row in models.itertuples():
        cell = f"models:{row.expt}/{row.cond}"
        total = sum(counts[cell].values())
        assert total == row.n, f"{cell}: {total} included choices, models.csv n={row.n}"
        relabel = KNOWN_MODELS_CSV_RELABELLING.get((row.expt, str(row.cond)))
        if relabel is None:
            assert counts[cell][row.object] == row.count, (cell, row.object, counts[cell][row.object], row.count)
            continue
        ours = {relabel[obj]: c for obj, c in counts[cell].items()}
        assert ours.get(row.object, 0) == row.count, (cell, row.object, ours, row.count)
        if counts[cell][row.object] != row.count:
            mismatched.add((row.expt, str(row.cond)))
    # the documented cells really do differ (the relabelling is not a no-op)
    assert mismatched == set(KNOWN_MODELS_CSV_RELABELLING)


def test_aggregation_reproduces_prior_counts_csv(trials: pd.DataFrame) -> None:
    counts = _semantic_counts(trials)
    prior_counts = _reference("prior_counts.csv")
    assert len(prior_counts) == 13
    for row in prior_counts.itertuples():
        cell = f"prior_counts:{row.prior}"
        for obj in ("foil", "logical", "target"):
            assert counts[cell][obj] == getattr(row, obj), (cell, obj, dict(counts[cell]))


# levels.Rmd / prior.Rmd output rows, by (expt, question, cond) -> paper_cond.
RMD_OUTPUT_CELLS = {
    ("complex", "inference", "0"): "models:complex/0",
    ("complex", "inference", "1"): "models:complex/1",
    ("complex", "inference", "2"): "models:complex/2",
    ("complex", "prior", "0"): "prior_counts:0-complex",
    ("simple", "inference", "0"): "models:simple/0",
    ("simple", "inference", "1"): "models:simple/1",
    ("twins", "prior", "prior"): "prior_counts:prior-twins",
    ("twins", "inference", "twin"): "models:twins/twin",
    ("twins", "inference", "uniform"): "models:twins/uniform",
    ("oddman", "inference", "patch"): "models:oddman/patch",
    ("oddman", "prior", "prior"): "prior_counts:prior-oddman",
    ("oddman", "inference", "word"): "models:oddman/word",
    ("prior", "prior", "prior"): "prior_counts:prior-prior",
    ("lang", "inference", "favorite"): "models:lang/favorite",
    ("lang", "inference", "least favorite"): "models:lang/least",
    ("lang", "prior", "favorite"): "prior_counts:favorite-lang",
    ("lang", "prior", "least favorite"): "prior_counts:least-lang",
}
for _ref in ("foil", "logical", "target"):
    RMD_OUTPUT_CELLS[("color", "prior", _ref)] = f"prior_counts:{_ref}-color"
for _ref in ("foil", "logical", "target", "none"):
    RMD_OUTPUT_CELLS[("color", "inference", _ref)] = f"models:color/{_ref}"
for _rate in ("0.11", "0.33", "0.44", "0.77"):
    RMD_OUTPUT_CELLS[("baserate", "inference", _rate)] = f"models:baserate/{_rate}"
    RMD_OUTPUT_CELLS[("baserate", "prior", _rate)] = f"prior_counts:{_rate}-baserate"

# levels.Rmd's renaming of the twins / oddman choices onto target/logical/foil.
RMD_ROLE_AS = {
    "single": {"target": 1.0},
    "twin": {"logical": 0.5, "foil": 0.5},
    "odd_one": {"target": 1.0},
    "twin_1": {"logical": 1.0},
    "twin_2": {"foil": 1.0},
}


def test_role_counts_reproduce_the_rmd_outputs(trials: pd.DataFrame) -> None:
    """levels.csv and prior.csv (the Rmds' own output) match by role label, all cells."""
    counts: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for row in trials[(trials.paper_cond != "") & trials.included].itertuples():
        role = row.object_roles[row.choice]
        for name, weight in RMD_ROLE_AS.get(role, {role: 1.0}).items():
            counts[row.paper_cond][name] += weight
    seen = set()
    for name in ("levels.csv", "prior.csv"):
        ref = _reference(name)
        for row in ref.itertuples():
            key = (row.expt, row.question, str(row.cond))
            cell = RMD_OUTPUT_CELLS[key]
            seen.add(cell)
            for obj in ("foil", "logical", "target"):
                expected = getattr(row, obj)
                expected = 0 if pd.isna(expected) else expected
                assert counts[cell][obj] == expected, (name, key, obj, dict(counts[cell]))
    assert seen == set(RMD_OUTPUT_CELLS.values())


# --------------------------------------------------------------------------
# (b) Table 1
# --------------------------------------------------------------------------


def _participants(trials: pd.DataFrame) -> pd.DataFrame:
    return trials[trials.trial_index == 0]


def test_table1_participant_counts(trials: pd.DataFrame) -> None:
    people = _participants(trials)
    for experiment, (n_total, n_include) in TABLE1.items():
        rows = people[people.experiment == experiment]
        assert len(rows) == n_total, experiment
        included = int(rows.included.sum())
        if experiment in TABLE1_WITHOUT_DEDUPLICATION:
            duplicates_only = int((rows.exclusion_reason == "duplicate_participant").sum())
            assert included < n_include, experiment
            assert included + duplicates_only == n_include, experiment
        else:
            assert included == n_include, experiment


def test_commented_out_table1_rows(trials: pd.DataFrame) -> None:
    people = _participants(trials)
    for label, (experiments, scope, (n_total, n_include)) in COMMENTED_ROWS.items():
        rows = people[people.experiment.isin(experiments)]
        if scope is not None:
            in_scope = rows.source_file.isin(scope) | rows.condition.str.split(":").str[0].isin(scope)
            rows = rows[in_scope]
        assert (len(rows), int(rows.included.sum())) == (n_total, n_include), label


def test_multi_trial_participants_share_inclusion(trials: pd.DataFrame) -> None:
    for _, rows in trials.groupby(["source_file", "participant_id"]):
        assert rows.included.nunique() == 1
        assert list(rows.trial_index) == list(range(len(rows)))


# --------------------------------------------------------------------------
# (c) schema
# --------------------------------------------------------------------------


def test_columns_are_the_contract() -> None:
    header = CSV_PATH.read_text(encoding="utf-8").splitlines()[0]
    assert header.split(",") == ingest.COLUMNS


def test_schema(trials: pd.DataFrame) -> None:
    assert (trials.participant_id != "").all()
    assert set(trials.dv) == {"forced_choice", "betting", "likert", "production"}
    assert set(trials["query"]) == {"utterance", "prior", "production"}
    for row in trials.itertuples():
        where = f"{row.source_file} {row.participant_id} trial {row.trial_index}"
        assert row.included == (row.exclusion_reason == ""), where
        if not row.objects:  # the one recorded gap: speaker level-1 listener trials
            assert "speaker_seq_level1_listener_matrix_unknown" in row.notes, where
            assert row.choice is None and row.response["choice_role"] in ("target", "logical", "foil"), where
            continue
        n_obj, n_feat = len(row.objects), len(row.objects[0])
        assert all(len(o) == n_feat for o in row.objects), where
        assert all(v in (0, 1) for o in row.objects for v in o), where
        assert all(any(o[c] for o in row.objects) for c in range(n_feat)), f"{where}: a feature on no object"
        assert len(row.feature_names) == n_feat and len(row.object_roles) == n_obj, where
        known = [n for n in row.feature_names if n is not None]
        assert len(set(known)) == len(known), where
        if row.display_order:
            assert len(row.display_order) == n_obj, where
            placed = [r for r in row.display_order if r is not None]
            assert len(set(placed)) == len(placed) and all(0 <= r < n_obj for r in placed), where
        if row.query == "utterance":
            assert row.utterance is not None and 0 <= row.utterance < n_feat, where
            assert any(o[row.utterance] for o in row.objects), f"{where}: utterance true of no object"
        else:
            assert row.utterance is None, where
        if row.dv == "forced_choice":
            assert row.choice is not None and 0 <= row.choice < n_obj, where
        else:
            assert row.choice is None, where
        if row.dv in ("betting", "likert"):
            assert len(row.response) == n_obj, where
        if row.dv == "betting":
            assert sum(row.response) == 100, where
        if row.dv == "production":
            assert row.referent is not None and 0 <= row.referent < n_obj, where
        if row.familiarization is not None:
            assert len(row.familiarization) == n_obj and sum(row.familiarization) == 9, where
        if row.grayscale is not None:
            assert len(row.grayscale) == n_obj and sum(row.grayscale) in (0, n_obj - 1), where


def test_forced_choice_rows_are_consistent_with_their_display(trials: pd.DataFrame) -> None:
    """Literal semantics hold for almost every listener choice in the paper cells.

    38 of 1092 such choices (3.5%) name an object the word is false of — the
    foils in models.csv's word cells.
    """
    rows = trials[(trials["query"] == "utterance") & trials.included & (trials.paper_cond != "")]
    rows = rows[rows.framing != "points_to_color_patch"]  # the oddman patch probes non-literal choice
    rows = rows[rows.condition != "word"]  # so does oddman "word"
    false = sum(1 for r in rows.itertuples() if not r.objects[r.choice][r.utterance])
    assert false / len(rows) < 0.04


def test_no_free_text_or_hit_metadata(trials: pd.DataFrame) -> None:
    """Comments, demographics and HIT fields never reach the CSV; descriptions only as speaker responses."""
    for row in trials.itertuples():
        if row.response is not None and isinstance(row.response, dict) and "description" in row.response:
            assert row.dv == "production"
    text = CSV_PATH.read_text(encoding="utf-8").lower()
    for marker in ("what is he talking about? (1-2 mins)", "reviewable", "notreviewed"):
        assert marker not in text


def test_provenance_matches_csv() -> None:
    prov = json.loads(PROVENANCE_PATH.read_text(encoding="utf-8"))
    assert len(prov["source_commit"]) == 40
    assert prov["output"]["sha256"] == ingest._sha256(CSV_PATH)
    df = pd.read_csv(CSV_PATH, dtype=str, keep_default_na=False)
    assert prov["output"]["rows"] == len(df)
    assert set(df.source_file) == {f["path"] for f in prov["source_files"]}


# --------------------------------------------------------------------------
# regeneration (needs the clone)
# --------------------------------------------------------------------------


def test_regenerating_reproduces_the_committed_csv(tmp_path: Path) -> None:
    clone = _pragmods_dir()
    prov = json.loads(PROVENANCE_PATH.read_text(encoding="utf-8"))
    commit, _ = ingest.git_commit(clone)
    if commit != prov["source_commit"]:
        pytest.skip(f"PRAGMODS_DIR is at {commit}, the committed CSV was built from {prov['source_commit']}")
    csv_path, _ = ingest.write_outputs(clone, tmp_path, command="test")
    assert csv_path.read_bytes() == CSV_PATH.read_bytes()


def test_unused_files_are_not_ingested() -> None:
    clone = _pragmods_dir()
    files = ingest.participant_files(clone / "data")
    assert not [f for f in files if set(Path(f).parts) & ingest.EXCLUDED_DIR_NAMES]
    assert any("unused" in str(p) for p in (clone / "data").rglob("*.csv"))
