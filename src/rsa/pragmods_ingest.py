"""Ingest the de-identified pragmods MTurk data into one canonical trial-level CSV.

Source: a clone of github.com/langcog/pragmods (MIT; Frank, Emilsson, Peloquin,
Goodman & Potts, "Rational speech act models of pragmatic reasoning in
reference games"). Every participant file under ``data/`` (except ``unused/``,
``originals/`` and ``unused-size/``) is read, recoded the way the analysis Rmds
in ``analysis/`` code it, and written as one row per response to
``src/pipelines/outer_loop/projects/rsa_reference/data/pragmods_trials.csv``
with a ``pragmods_trials.provenance.json`` sidecar.

Run::

    uv run python -m src.rsa.pragmods_ingest --pragmods-dir <clone of langcog/pragmods>

What is transcribed here rather than read (the clone holds no machine-readable
copy of it):

* the stimulus matrices, object roles and the queried / counted feature of
  every ``scale_and_level`` condition, from ``pragmods_parameter_setter_c1.js``
  of langcog/pragmods-expts (commit ``PRAGMODS_EXPTS_COMMIT``);
* the per-analysis file lists, condition coding and exclusion criteria, from
  ``analysis/{1-prelims,2-prior,3-levels,4-sequences,5-speakers}/*.Rmd`` and
  ``analysis/helper.R``;
* the size-experiment matrices, from ``models/matrices.R`` (the per-condition
  table ``data/3-levels/size/scale_and_level.csv`` *is* read).

The exclusion code reproduces R's semantics where they matter: ``duplicated``
over each analysis' bound rows (in the Rmd's file order, before any other
filter), string comparison against R's own column typing (``mc.targ == "2"``)
where the Rmd compares strings, and ``as.numeric`` where it converts. Anything
the data cannot pin down is left empty and named in the row's ``notes``; a
source row that contradicts its own coding raises.

The column contract and every coding decision are documented in the data
directory's ``README.md``.
"""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import pandas as pd
import tyro
from pyprojroot import here

DEFAULT_OUT_DIR = here() / "src" / "pipelines" / "outer_loop" / "projects" / "rsa_reference" / "data"
CSV_NAME = "pragmods_trials.csv"
PROVENANCE_NAME = "pragmods_trials.provenance.json"

# The experiment code the c1 matrices below were transcribed from.
PRAGMODS_EXPTS_REPO = "https://github.com/langcog/pragmods-expts"
PRAGMODS_EXPTS_COMMIT = "8ff10c31e47e0daf00c1f48ded583b955eb56747"
PRAGMODS_REPO = "https://github.com/langcog/pragmods"

EXCLUDED_DIR_NAMES = frozenset({"unused", "originals", "unused-size"})
NON_PARTICIPANT_FILES = frozenset({"all_experiments.csv", "3-levels/size/scale_and_level.csv"})

COLUMNS = [
    "participant_id",
    "batch",
    "source_file",
    "series",
    "experiment",
    "condition",
    "paper_cond",
    "trial_index",
    "item",
    "feature_names",
    "objects",
    "object_roles",
    "display_order",
    "query",
    "query_detail",
    "utterance",
    "framing",
    "familiarization",
    "grayscale",
    "dv",
    "choice",
    "response",
    "referent",
    "included",
    "exclusion_reason",
    "notes",
]

POSITIONS3 = ("left", "middle", "right")

# linguistic_framing codes (pragmods_parameter_setter_c1.js).
FRAMING_LABELS = {
    0: "my_favorite_X_has",
    1: "one_word",
    2: "my_least_favorite_X_has",
    3: "most_beautiful_X_has",
    4: "most_ugly_X_has",
    5: "most_cheerful_X_has",
    6: "most_depressing_X_has",
    7: "silent_favorite",
    8: "silent_least_favorite",
    9: "points_to_color_patch",
    10: "one_word",
    11: "tricky_guy",
    12: "pure_randomness",
    13: "my_X_has",
    14: "the_X_has",
}

# question_type codes (pragmods_parameter_setter_c1.js / control_flow_c1.js).
QUESTION_TYPE_PRIOR_DETAIL = {
    1: "mumble: 'the X I will Y next has mumblemumble' (you couldn't hear)",
    2: "which_favorite: 'Which X is Bob's favorite?'",
    3: "mumble_one_word: 'Bob can only say one word ... which X he will Y next, and he says mumblemumble'",
    4: "which_next: 'Which X will Bob Y next?'",
}

# Familiarization (fam_dists in the c1 code): per unpermuted row (foil, target,
# logical in scale_and_level 1), how many of the nine familiarization images
# showed that object, by familiarization_cond.
FAMILIARIZATION_COUNTS = {0: [1, 1, 7], 1: [1, 3, 5], 2: [1, 5, 3], 3: [1, 7, 1]}
# models.csv / prior.Rmd labels for those conditions (sic: 5/9 is labelled 0.44).
BASERATE_LABELS = {0: "0.11", 1: "0.33", 2: "0.44", 3: "0.77"}


@dataclass(frozen=True)
class C1Condition:
    """One ``scale_and_level`` value of pragmods_parameter_setter_c1.js."""

    matrix: str
    rows: Tuple[Tuple[int, int, int], ...]  # the unpermuted ``expt`` rows
    roles: Tuple[str, ...]  # choice_names_unpermuted
    target_row: int  # target_unpermuted
    distractor_row: int  # distractor_unpermuted (its position is ``logical_position``)
    target_prop: int  # JS column of the uttered word (``target_property``)
    distractor_prop: int  # JS column counted by manip_check_dist (``logical_property``)
    foil_prop: int  # JS column named ``foil_property``

    @property
    def present_cols(self) -> List[int]:
        return [c for c in range(3) if any(r[c] for r in self.rows)]

    @property
    def objects(self) -> List[List[int]]:
        cols = self.present_cols
        return [[r[c] for c in cols] for r in self.rows]

    def canonical_col(self, js_col: int) -> Optional[int]:
        cols = self.present_cols
        return cols.index(js_col) if js_col in cols else None


_SIMPLE = ((0, 0, 0), (0, 0, 1), (0, 1, 1))
_COMPLEX = ((0, 0, 1), (0, 1, 1), (1, 1, 0))
_TWINS = ((0, 1, 1), (1, 0, 1), (1, 0, 1))
_ODDMAN = ((0, 1, 1), (1, 0, 1), (1, 1, 0))
C1 = {
    0: C1Condition("simple", _SIMPLE, ("foil", "logical", "target"), 2, 1, 1, 2, 0),
    1: C1Condition("simple", _SIMPLE, ("foil", "target", "logical"), 1, 2, 2, 1, 0),
    2: C1Condition("complex", _COMPLEX, ("foil", "logical", "target"), 2, 1, 0, 1, 2),
    3: C1Condition("complex", _COMPLEX, ("target", "logical", "foil"), 0, 1, 2, 1, 0),
    4: C1Condition("complex", _COMPLEX, ("foil", "target", "logical"), 1, 2, 1, 0, 2),
    5: C1Condition("twins", _TWINS, ("single", "twin", "twin"), 0, 1, 1, 0, 2),
    6: C1Condition("twins", _TWINS, ("single", "twin", "twin"), 1, 2, 0, 1, 2),
    7: C1Condition("twins", _TWINS, ("single", "twin", "twin"), 1, 2, 2, 1, 0),
    8: C1Condition("oddman", _ODDMAN, ("twin_1", "twin_2", "odd_one"), 2, 1, 2, 1, 0),
}

# models/matrices.R size{objects}.{features}; rows ref1..refN, columns hat,
# glasses, moustache, bowtie (first N features).
SIZE_FEATURES = ("hat", "glasses", "moustache", "bowtie")
SIZE_MATRICES = {
    (2, 2): ((0, 1), (1, 1)),
    (3, 2): ((0, 0), (0, 1), (1, 1)),
    (4, 2): ((0, 0), (0, 0), (0, 1), (1, 1)),
    (2, 3): ((0, 1, 1), (1, 1, 1)),
    (3, 3): ((0, 0, 1), (0, 1, 1), (1, 1, 1)),
    (4, 3): ((0, 0, 1), (0, 0, 1), (0, 1, 1), (1, 1, 1)),
    (2, 4): ((0, 1, 1, 1), (1, 1, 1, 1)),
    (3, 4): ((0, 0, 1, 1), (0, 1, 1, 1), (1, 1, 1, 1)),
    (4, 4): ((0, 0, 1, 1), (0, 0, 1, 1), (0, 1, 1, 1), (1, 1, 1, 1)),
}
# scale_and_level.csv numbers its matrices 1..9 in this order; the prior file's
# matrix_number is the same list 0-indexed (object_k_items confirm it).
SIZE_MATRIX_ORDER = [(2, 2), (3, 2), (4, 2), (2, 3), (3, 3), (4, 3), (2, 4), (3, 4), (4, 4)]


# --------------------------------------------------------------------------
# Reading, with R's read.table typing
# --------------------------------------------------------------------------

_R_LOGICAL = {"T", "F", "TRUE", "FALSE", "true", "false", "True", "False"}


def _parses_as_number(value: str) -> bool:
    try:
        x = float(value)
    except ValueError:
        return False
    return not math.isnan(x)


def _r_column_kind(raw_values: Iterable[str]) -> str:
    """The type R's read.table/type.convert gives a column: lgl, num or chr."""
    values = [v for v in raw_values if v not in ("NA", "")]
    if values and all(v in _R_LOGICAL for v in values):
        return "lgl"
    if all(_parses_as_number(v) for v in values):
        return "num"
    return "chr"


@dataclass
class SourceTable:
    rel: str  # path relative to pragmods/data
    frame: pd.DataFrame  # lower-cased names without 'answer.', '"' stripped (helper.R)
    kinds: Dict[str, str]  # column -> lgl / num / chr as R typed it before stripping

    def r_str(self, i: int, col: str) -> Optional[str]:
        """The value as R's ``==`` against a string literal sees it (None = NA)."""
        value = self.frame.at[i, col]
        if value == "NA":
            return None
        kind = self.kinds[col]
        if kind == "num":
            if value == "":
                return None
            x = float(value)
            return str(int(x)) if x.is_integer() else repr(x)
        if kind == "lgl":
            if value == "":
                return None
            return "TRUE" if value in ("T", "TRUE", "true", "True") else "FALSE"
        return value

    def r_num(self, i: int, col: str) -> Optional[float]:
        """R's ``as.numeric`` of the (quote-stripped) value (None = NA)."""
        value = self.frame.at[i, col]
        if value == "NA" or value.strip() == "":
            return None
        if self.kinds[col] == "lgl":
            return 1.0 if self.r_str(i, col) == "TRUE" else 0.0
        try:
            x = float(value)
        except ValueError:
            return None
        return None if math.isnan(x) else x

    def get(self, i: int, col: str) -> str:
        if col not in self.frame.columns:
            raise KeyError(f"{self.rel}: no column {col!r}")
        return self.frame.at[i, col]

    def has(self, col: str) -> bool:
        return col in self.frame.columns

    def __len__(self) -> int:
        return len(self.frame)


def read_source(data_dir: Path, rel: str) -> SourceTable:
    path = data_dir / rel
    if not path.is_file():
        raise FileNotFoundError(f"pragmods data file missing: {path}")
    sep = "," if "csv" in path.name else "\t"  # analysis/helper.R: grepl("csv", fname)
    raw = pd.read_csv(path, sep=sep, dtype=str, keep_default_na=False, quoting=csv.QUOTE_MINIMAL)
    names = [re.sub(r"answer.", "", c.lower(), count=1) for c in raw.columns]
    if len(set(names)) != len(names):
        raise ValueError(f"{rel}: duplicate column names after helper.R renaming: {names}")
    kinds = {name: _r_column_kind(raw[col]) for name, col in zip(names, raw.columns)}
    frame = raw.copy()
    frame.columns = names
    for col in names:
        frame[col] = frame[col].str.replace('"', "", regex=False)
    frame = frame.reset_index(drop=True)
    return SourceTable(rel=rel, frame=frame, kinds=kinds)


def batch_code(rel: str) -> str:
    stem = Path(rel).name
    stem = re.sub(r"\.(tsv|csv)$", "", stem)
    codes = [t for t in stem.split("_") if len(t) >= 4 and t.isupper() and t.isalnum()]
    if codes:
        return codes[-1]
    return re.sub(r"\.results(\d*)$", r"\1", re.sub(r"\.anondata$", "", stem))


# --------------------------------------------------------------------------
# Display reconstruction
# --------------------------------------------------------------------------


@dataclass
class Display:
    display_order: List[Optional[int]]  # row index at each position (None = unknown)
    feature_names: List[Optional[str]]  # name of each canonical column (None = unknown)
    chosen_row: Optional[int] = None


class DisplayInconsistent(ValueError):
    """The recorded positions / names / items admit no layout of the matrix."""


def solve_display(
    objects: Sequence[Sequence[int]],
    positions: Sequence[str],
    row_at: Dict[str, Set[int]],
    col_names: Dict[int, str],
    items_at: Optional[Dict[str, frozenset]] = None,
    chosen_pos: Optional[str] = None,
    chosen_rows: Optional[Set[int]] = None,
) -> Display:
    """Every layout of ``objects`` over ``positions`` that the records allow.

    A position (or column name) is filled only when every allowed layout agrees
    on it. Layouts that differ only by swapping identical objects (twins) or
    identically distributed features are the same layout, resolved
    deterministically.
    """
    n = len(objects)
    n_feat = len(objects[0])
    if len(positions) != n:
        raise ValueError(f"{len(positions)} positions for {n} objects")
    feats = [frozenset(c for c in range(n_feat) if objects[r][c]) for r in range(n)]
    col_vec = [tuple(objects[r][c] for r in range(n)) for c in range(n_feat)]
    names: List[str] = sorted(set().union(*items_at.values())) if items_at is not None else []
    if items_at is not None:
        if set(items_at) != set(positions):
            raise ValueError(f"items given for {sorted(items_at)}, positions are {list(positions)}")
        if len(names) != n_feat:
            raise DisplayInconsistent(f"items name {len(names)} features {names}, matrix has {n_feat}")
    for c, nm in col_names.items():
        if items_at is not None and nm not in names:
            raise DisplayInconsistent(f"recorded feature {nm!r} not on any displayed object {names}")
    solutions: List[Tuple[Tuple[int, ...], Dict[int, str]]] = []
    for perm in itertools.permutations(range(n)):
        at = dict(zip(positions, perm))
        if any(at[p] not in allowed for p, allowed in row_at.items()):
            continue
        if chosen_pos is not None and at[chosen_pos] not in (chosen_rows or set()):
            continue
        if items_at is None:
            solutions.append((perm, dict(col_names)))
            continue
        for assign in itertools.permutations(range(n_feat)):
            name_to_col = dict(zip(names, assign))
            if any(name_to_col[nm] != c for c, nm in col_names.items()):
                continue
            if all(frozenset(name_to_col[nm] for nm in items_at[p]) == feats[at[p]] for p in positions):
                solutions.append((perm, {c: nm for nm, c in name_to_col.items()}))
    if not solutions:
        raise DisplayInconsistent("no layout of the matrix satisfies the recorded display")
    layouts: Dict[tuple, List[Tuple[Tuple[int, ...], Dict[int, str]]]] = {}
    for perm, cn in solutions:
        key = (
            tuple(tuple(objects[r]) for r in perm),
            frozenset((nm, col_vec[c]) for c, nm in cn.items()),
        )
        layouts.setdefault(key, []).append((perm, cn))
    reps = [min(v, key=lambda s: (s[0], sorted(s[1].items()))) for v in layouts.values()]
    order: List[Optional[int]] = []
    for p in range(n):
        rows = {rep[0][p] for rep in reps}
        order.append(rows.pop() if len(rows) == 1 else None)
    feature_names: List[Optional[str]] = []
    for c in range(n_feat):
        nms = {rep[1].get(c) for rep in reps}
        feature_names.append(nms.pop() if len(nms) == 1 else None)
    chosen_row = None
    if chosen_pos is not None:
        idx = list(positions).index(chosen_pos)
        rows = {rep[0][idx] for rep in reps}
        chosen_row = rows.pop() if len(rows) == 1 else None
    return Display(order, feature_names, chosen_row)


def _items(value: str) -> frozenset:
    return frozenset(t.strip() for t in value.split(",") if t.strip())


# --------------------------------------------------------------------------
# Row records
# --------------------------------------------------------------------------


def _json(value) -> str:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


@dataclass
class Record:
    participant_id: str
    batch: str
    source_file: str
    series: str
    experiment: str
    condition: str
    paper_cond: str = ""
    trial_index: int = 0
    item: str = ""
    feature_names: list = field(default_factory=list)
    objects: list = field(default_factory=list)
    object_roles: list = field(default_factory=list)
    display_order: list = field(default_factory=list)
    query: str = ""
    query_detail: str = ""
    utterance: Optional[int] = None
    framing: str = ""
    familiarization: Optional[list] = None
    grayscale: Optional[list] = None
    dv: str = ""
    choice: Optional[int] = None
    response: object = None
    referent: Optional[int] = None
    included: bool = False
    exclusion_reason: str = ""
    notes: List[str] = field(default_factory=list)

    def as_row(self) -> Dict[str, object]:
        return {
            "participant_id": self.participant_id,
            "batch": self.batch,
            "source_file": self.source_file,
            "series": self.series,
            "experiment": self.experiment,
            "condition": self.condition,
            "paper_cond": self.paper_cond,
            "trial_index": self.trial_index,
            "item": self.item,
            "feature_names": _json(self.feature_names),
            "objects": _json(self.objects),
            "object_roles": _json(self.object_roles),
            "display_order": _json(self.display_order),
            "query": self.query,
            "query_detail": self.query_detail,
            "utterance": "" if self.utterance is None else self.utterance,
            "framing": self.framing,
            "familiarization": "" if self.familiarization is None else _json(self.familiarization),
            "grayscale": "" if self.grayscale is None else _json(self.grayscale),
            "dv": self.dv,
            "choice": "" if self.choice is None else self.choice,
            "response": "" if self.response is None else _json(self.response),
            "referent": "" if self.referent is None else self.referent,
            "included": self.included,
            "exclusion_reason": self.exclusion_reason,
            "notes": ";".join(self.notes),
        }


# --------------------------------------------------------------------------
# Exclusion groups (one per analysis block of the Rmds)
# --------------------------------------------------------------------------

Criteria = Callable[[SourceTable, int], List[str]]


@dataclass
class GroupMember:
    rel: str
    keep: Callable[[SourceTable, int], bool] = lambda t, i: True  # pre-bind filter (SCALESBASE)


@dataclass
class Group:
    """One ``d <- bind_rows(...); dc <- d %>% filter(...)`` block of an Rmd."""

    experiment: str
    members: List[GroupMember]
    criteria: Criteria
    dedupe: bool = True


def _mc_str(target: str, dist: str) -> Criteria:
    def crit(t: SourceTable, i: int) -> List[str]:
        reasons = []
        if t.r_str(i, "manip_check_target") != target:
            reasons.append("manip_check_target")
        if t.r_str(i, "manip_check_dist") != dist:
            reasons.append("manip_check_dist")
        if t.r_str(i, "name_check_correct") != "TRUE":
            reasons.append("name_check")
        return reasons

    return crit


def _mc_num(target: Optional[float], dist: float, extra_ok: Sequence[float] = ()) -> Criteria:
    def crit(t: SourceTable, i: int) -> List[str]:
        reasons = []
        mt, md = t.r_num(i, "manip_check_target"), t.r_num(i, "manip_check_dist")
        if target is not None and (mt is None or (mt != target and mt not in extra_ok)):
            reasons.append("manip_check_target")
        if md is None or (md != dist and md not in extra_ok):
            reasons.append("manip_check_dist")
        if t.r_str(i, "name_check_correct") != "TRUE":
            reasons.append("name_check")
        return reasons

    return crit


def _name_only(t: SourceTable, i: int) -> List[str]:
    return [] if t.r_str(i, "name_check_correct") == "TRUE" else ["name_check"]


def _name_and_overspec(t: SourceTable, i: int) -> List[str]:
    reasons = _name_only(t, i)
    if t.r_num(i, "overspec") is None:
        reasons.append("overspec_missing")
    return reasons


# levels.Rmd `mcs`: (cond, level) -> (mc.targ.ans, mc.dist.ans)
LEVELS_MC = {
    ("simple", 0): (1.0, 2.0),
    ("simple", 1): (None, 1.0),
    ("complex", 0): (1.0, 2.0),
    ("complex", 1): (2.0, 2.0),
    ("complex", 2): (2.0, 1.0),
    ("complex prior", -1): (2.0, 1.0),
}
# levels.Rmd twins `mcs`
TWINS_MC = {5: (1.0, 2.0), 6: (2.0, 1.0), 7: (3.0, 1.0)}


def _levels_cell(t: SourceTable, i: int) -> Tuple[str, int]:
    sl = int(t.r_num(i, "scale_and_levels_condition"))
    if t.rel.endswith("_SCAL.tsv"):
        return "complex", sl - 2
    if t.rel.endswith("_OSCA.tsv"):
        return "simple", sl
    if t.rel.endswith("_SCALESBASE.tsv"):
        return "complex prior", -1
    raise ValueError(f"{t.rel} is not a levels file")


def _levels_criteria(t: SourceTable, i: int) -> List[str]:
    targ, dist = LEVELS_MC[_levels_cell(t, i)]
    return _mc_num(targ, dist)(t, i)


def _twins_criteria(t: SourceTable, i: int) -> List[str]:
    targ, dist = TWINS_MC[int(t.r_num(i, "scale_and_levels_condition"))]
    return _mc_num(targ, dist)(t, i)


def _qt(value: int) -> Callable[[SourceTable, int], bool]:
    return lambda t, i: t.r_num(i, "question_type_condition") == value


DV_FILES = {
    "forced_choice": [
        "1-prelims/dv/forced_choice_no_fam_boat_oneword_7_december_LFBN.tsv",
        "1-prelims/dv/forced_choice_no_fam_friend_oneword_1_december_AGSK.tsv",
        "1-prelims/dv/forced_choice_no_fam_snowman_oneword_8_december_LSBN.tsv",
        "1-prelims/dv/forced_choice_no_fam_sundaes_oneword_8_december_LSBZ.tsv",
    ],
    "betting": [
        "1-prelims/dv/betting_no_fam_boat_oneword_8_december_LZZZ.tsv",
        "1-prelims/dv/betting_no_fam_friend_oneword_1_december_AGHP.tsv",
        "1-prelims/dv/betting_no_fam_snowman_oneword_8_december_LSSS.tsv",
        "1-prelims/dv/betting_no_fam_sundae_oneword_8_december_BSUN.tsv",
    ],
    "likert": [
        "1-prelims/dv/likert_no_fam_boat_oneword_9_december_FUNU.tsv",
        "1-prelims/dv/likert_no_fam_friend_oneword_2_december_JSLD.tsv",
        "1-prelims/dv/likert_no_fam_snowman_oneword_9_december_FINA.tsv",
        "1-prelims/dv/likert_no_fam_sundaes_oneword_9_december_LSUN.tsv",
    ],
}

SCALESBASE = "3-levels/levels/forced_choice_no_fam_6random_count_onewordmumble_22jan2015_SCALESBASE.tsv"

GROUPS: List[Group] = [
    # prelims.Rmd, "Dependent Variables": files in dir() order within measure.
    Group("E1_dv", [GroupMember(f) for m in ("forced_choice", "betting", "likert") for f in DV_FILES[m]], _mc_str("2", "1")),
    # prelims.Rmd, "Manipulation check": -1 = not asked (ALNC).
    Group(
        "E2_manip_check",
        [
            GroupMember("1-prelims/manip/forced_choice_no_fam_6random_count_ALLS.tsv"),
            GroupMember("1-prelims/manip/forced_choice_no_fam_6random_NOcount_ALNC.tsv"),
        ],
        _mc_num(2.0, 1.0, extra_ok=(-1.0,)),
    ),
    Group(
        "E3_ling_frame",
        [GroupMember("1-prelims/linguistic_framing/forced_choice_no_fam_6random_count_baseratesmy_14jan2015_BASERATES.tsv")],
        _mc_str("2", "1"),
    ),
    Group(
        "E4_prior_frame",
        [GroupMember("2-prior/measurement/forced_choice_no_fam_6random_count_onewordmumble_21jan2015_WORDMUMBLE.tsv")],
        _mc_num(2.0, 1.0),
    ),
    Group(
        "E5_baserate",
        [
            GroupMember("2-prior/baserates/scale_6stimuli_yes_fam_oneword_25_february_FAMO.tsv"),
            GroupMember("2-prior/baserates/scale_6stimuli_yes_fam_oneword_25_february_FAMO2.tsv"),
            GroupMember("2-prior/baserates/scale_6stimuli_yes_fam_mumblemumble_26_february_FMMM.tsv"),
        ],
        _mc_num(2.0, 1.0),
    ),
    Group(
        "E6_valence",
        [
            GroupMember("2-prior/language/forced_choice_no_fam_friends_30_november.tsv"),
            GroupMember("2-prior/language/forced_choice_no_fam_boat_1_december_XYZQ_incomplete.tsv"),
            GroupMember("2-prior/language/forced_choice_no_fam_pizza_30_november.tsv"),
            GroupMember("2-prior/language/forced_choice_no_fam_snowman_30_november.tsv"),
            GroupMember("2-prior/language/forced_choice_no_fam_4random_count_12_january_least_LEAS.tsv"),
            GroupMember("2-prior/language/scale_6stimuli_no_fam_prior_5_april_PRIOLF.tsv"),
        ],
        _mc_num(2.0, 1.0),
    ),
    Group(
        "E7_color",
        [
            GroupMember("2-prior/color/forced_choice_no_fam_6random_2count_oneword_4mar2015_COLORBASE.tsv"),
            GroupMember("2-prior/color/forced_choice_no_fam_6random_2count_oneword_23feb2015_COLORSALIENCE.tsv"),
        ],
        _mc_num(2.0, 1.0),
    ),
    # Not read by prior.Rmd (a second color prior batch); coded like E7, deduped on its own.
    Group(
        "color_prior_rerun",
        [GroupMember("2-prior/color/forced_choice_no_fam_6random_2count_oneword_10april2015_COLORBASE2.tsv")],
        _mc_num(2.0, 1.0),
    ),
    Group(
        "E8_levels",
        [
            GroupMember("3-levels/levels/scale_plus_6stimuli_3levels_no_fam_24_january_SCAL.tsv"),
            GroupMember("3-levels/levels/scales_6stimuli_3levels_no_fam_25_january_OSCA.tsv"),
            GroupMember(SCALESBASE, keep=_qt(3)),
        ],
        _levels_criteria,
    ),
    # SCALESBASE's question_type 4 half, which levels.Rmd drops before binding.
    Group("levels_prior_action", [GroupMember(SCALESBASE, keep=_qt(4))], _levels_criteria),
    Group(
        "E9_twins",
        [
            GroupMember("3-levels/twins/scaleweird_6stimuli_no_fam_oneword_19_february_WERD.tsv"),
            GroupMember("3-levels/twins/forced_choice_no_fam_6random_count_onewordmumble_11feb2015_TWINBASE.tsv"),
        ],
        _twins_criteria,
    ),
    Group(
        "E10_oddman",
        [
            GroupMember("3-levels/oddman/patch_oddone_no_fam_14_may_PATCH.tsv"),
            GroupMember("3-levels/oddman/forced_choice_no_fam_6random_3count_onewordmumble_11feb2015_ODDBASE.tsv"),
        ],
        _mc_num(2.0, 2.0),
    ),
    Group(
        "size",
        [
            GroupMember("3-levels/size/pragmods_distractions.results1.tsv"),
            GroupMember("3-levels/size/pragmods_distractions.results2.tsv"),
            GroupMember("3-levels/size/pragmods_distractions.results3.tsv"),
            GroupMember("3-levels/size/pragmods_salience_priors.results.tsv"),
        ],
        _name_only,
    ),
    Group(
        "sequences",
        [
            GroupMember("4-sequences/pragmods_seq.anondata.csv"),
            GroupMember("4-sequences/pragmods_wx3.anondata.csv"),
            GroupMember("4-sequences/pragmods_bx3.anondata.csv"),
            GroupMember("4-sequences/pragmods_seq2.anondata.csv"),
            GroupMember("4-sequences/pragmods_L2second.anondata.tsv"),
        ],
        _name_only,
        dedupe=False,
    ),
    Group(
        "speakers",
        [
            GroupMember("5-speakers/pragmods_overspec_baseline.results.csv"),
            GroupMember("5-speakers/pragmods_overspec_checkbox.results.csv"),
            GroupMember("5-speakers/pragmods_overspec_virtual.results.csv"),
        ],
        _name_and_overspec,
        dedupe=False,
    ),
    Group(
        "speakers",
        [
            GroupMember("5-speakers/pragmods_overspec_seq_baseline_lvl1.results.csv"),
            GroupMember("5-speakers/pragmods_overspec_seq_checkbox_lvl1.results.csv"),
            GroupMember("5-speakers/pragmods_overspec_seq_virtual_lvl1.results.csv"),
        ],
        _name_and_overspec,
        dedupe=False,
    ),
    Group(
        "speakers",
        [
            GroupMember("5-speakers/pragmods_overspec_seq_baseline_lvl2.results.csv"),
            GroupMember("5-speakers/pragmods_overspec_seq_checkbox_lvl2.results.csv"),
            GroupMember("5-speakers/pragmods_overspec_seq_virtual_lvl2.results.csv"),
        ],
        _name_and_overspec,
        dedupe=False,
    ),
]

# Files whose workerid was a within-file row number before de-identification
# (data/README.md): their ids cannot be linked to other files.
UNLINKABLE_ID_FILES = frozenset(
    {m.rel for g in GROUPS if g.experiment == "sequences" for m in g.members}
    | {"5-speakers/pragmods_overspec_baseline.results.csv"}
)


# --------------------------------------------------------------------------
# Builders
# --------------------------------------------------------------------------


@dataclass
class Context:
    table: SourceTable
    i: int
    experiment: str
    included: bool
    reasons: List[str]

    def base(self, condition: str, trial_index: int = 0) -> Record:
        t = self.table
        worker = t.get(self.i, "workerid")
        batch = batch_code(t.rel)
        pid = f"{batch}:{worker}" if t.rel in UNLINKABLE_ID_FILES else worker
        return Record(
            participant_id=pid,
            batch=batch,
            source_file=t.rel,
            series=str(Path(t.rel).parent),
            experiment=self.experiment,
            condition=condition,
            trial_index=trial_index,
            included=self.included,
            exclusion_reason=";".join(self.reasons),
        )


def _int(t: SourceTable, i: int, col: str) -> int:
    x = t.r_num(i, col)
    if x is None or not float(x).is_integer():
        raise ValueError(f"{t.rel} row {i}: {col}={t.get(i, col)!r} is not an integer")
    return int(x)


def _c1_condition_labels(experiment: str, t: SourceTable, i: int, sl: int) -> Tuple[str, str]:
    """(condition, paper_cond) for a one-shot c1 trial, following the Rmds."""
    lf = _int(t, i, "linguistic_framing_condition")
    qt = _int(t, i, "question_type_condition")
    if experiment == "E1_dv":
        return {0: "forced_choice", 1: "betting", 2: "likert"}[_int(t, i, "participant_response_type_condition")], ""
    if experiment == "E2_manip_check":
        return ("manip_check" if t.rel.endswith("_ALLS.tsv") else "no_manip_check"), ""
    if experiment == "E3_ling_frame":
        return {1: "one_word", 13: "my_X_has"}[lf], ""
    if experiment == "E4_prior_frame":
        return {3: "mumble", 4: "action"}[qt], "prior_counts:prior-prior"
    if experiment == "E5_baserate":
        fam = _int(t, i, "familiarization_cond")
        label = BASERATE_LABELS[fam]
        if t.rel.endswith("_FMMM.tsv"):
            return f"prior_baserate_{label}", f"prior_counts:{label}-baserate"
        return f"inference_baserate_{label}", f"models:baserate/{label}"
    if experiment == "E6_valence":
        return {
            0: ("favorite", "models:lang/favorite"),
            2: ("least_favorite", "models:lang/least"),
            7: ("favorite_prior", "prior_counts:favorite-lang"),
            8: ("least_favorite_prior", "prior_counts:least-lang"),
        }[lf]
    if experiment in ("E7_color", "color_prior_rerun"):
        ref = t.get(i, "referent_with_color")
        if experiment == "color_prior_rerun":
            return f"prior_color_{ref}", ""
        if lf == 13:
            return f"prior_color_{ref}", f"prior_counts:{ref}-color"
        return f"inference_color_{ref}", f"models:color/{ref}"
    if experiment in ("E8_levels", "levels_prior_action"):
        cond, level = _levels_cell(t, i)
        if experiment == "levels_prior_action":
            return "complex_prior_action", ""
        if cond == "complex prior":
            return "complex_prior", "prior_counts:0-complex"
        return f"{cond}_L{level}", f"models:{cond}/{level}"
    if experiment == "E9_twins":
        return {
            5: ("prior", "prior_counts:prior-twins"),
            6: ("twin", "models:twins/twin"),
            7: ("uniform", "models:twins/uniform"),
        }[sl]
    if experiment == "E10_oddman":
        if t.rel.endswith("_ODDBASE.tsv"):
            return "prior", "prior_counts:prior-oddman"
        return {9: ("patch", "models:oddman/patch"), 10: ("word", "models:oddman/word")}[lf]
    raise ValueError(f"no c1 condition coding for {experiment}")


def _choice_row(roles: Sequence[str], label: str, where: str) -> Tuple[int, bool]:
    rows = [r for r, role in enumerate(roles) if role == label]
    if not rows:
        raise ValueError(f"{where}: choice {label!r} is not a role of {list(roles)}")
    return rows[0], len(rows) > 1


def build_c1(ctx: Context) -> List[Record]:
    t, i = ctx.table, ctx.i
    sl = _int(t, i, "scale_and_levels_condition") if t.has("scale_and_levels_condition") else 1
    cond = C1[sl]
    where = f"{t.rel} row {i}"
    condition, paper_cond = _c1_condition_labels(ctx.experiment, t, i, sl)
    rec = ctx.base(condition)
    rec.paper_cond = paper_cond
    rec.item = t.get(i, "item")
    rec.objects = cond.objects
    rec.object_roles = list(cond.roles)
    if not t.has("scale_and_levels_condition"):
        rec.notes.append("scale_and_level_1_from_all_experiments_csv")

    lf = _int(t, i, "linguistic_framing_condition")
    qt = _int(t, i, "question_type_condition")
    rec.framing = FRAMING_LABELS[lf]
    if qt == 0 and lf not in (7, 8):
        rec.query = "utterance"
        rec.utterance = cond.canonical_col(cond.target_prop)
        if rec.utterance is None:
            raise ValueError(f"{where}: uttered column {cond.target_prop} is on no object")
        rec.query_detail = "color_patch_pointing" if lf == 9 else "word"
    elif qt == 0:
        rec.query = "prior"
        rec.query_detail = {7: "silent: which is Bob's favorite", 8: "silent: which is Bob's least favorite"}[lf]
    else:
        rec.query = "prior"
        rec.query_detail = QUESTION_TYPE_PRIOR_DETAIL[qt]

    # display: positions, names, items
    row_at: Dict[str, Set[int]] = {}
    col_names: Dict[int, str] = {}

    def name_col(js_col: int, column: str) -> None:
        if not t.has(column):
            return
        c = cond.canonical_col(js_col)
        if c is not None:
            col_names[c] = t.get(i, column)

    name_col(cond.target_prop, "target_property")
    name_col(cond.distractor_prop, "logical_property")
    name_col(cond.foil_prop, "foil_property")

    def pin(column: str, rows: Set[int]) -> None:
        if not t.has(column):
            return
        pos = t.get(i, column)
        if pos not in POSITIONS3:
            raise ValueError(f"{where}: {column}={pos!r}")
        row_at[pos] = row_at.get(pos, {0, 1, 2}) & rows

    pin("target_position", {cond.target_row})
    pin("logical_position", {cond.distractor_row})

    items_at = None
    if t.has("left_items"):
        items_at = {p: _items(t.get(i, f"{p}_items")) for p in POSITIONS3}

    grayscale = None
    if t.has("referent_with_color"):
        ref = t.get(i, "referent_with_color")
        salience = _int(t, i, "color_salience")
        if ref == "none":
            if salience != 0 or t.get(i, "position_with_color") != "none":
                raise ValueError(f"{where}: referent_with_color none but color_salience={salience}")
            grayscale = [0, 0, 0]
            rec.notes.append("color_none_condition_all_colored_per_c1_code_paper_says_none_colored")
        else:
            colored = [r for r, role in enumerate(cond.roles) if role == ref]
            if len(colored) != 1:
                raise ValueError(f"{where}: referent_with_color={ref!r}")
            grayscale = [0 if r == colored[0] else 1 for r in range(3)]
            pin("position_with_color", set(colored))
    rec.grayscale = grayscale

    if t.has("familiarization_present_in_study") and _int(t, i, "familiarization_present_in_study") == 1:
        fam = _int(t, i, "familiarization_cond")
        if cond.matrix != "simple" or sl != 1:
            raise ValueError(f"{where}: familiarization outside scale_and_level 1")
        rec.familiarization = FAMILIARIZATION_COUNTS[fam]

    dv_code = _int(t, i, "participant_response_type_condition")
    chosen_pos = chosen_rows = None
    if dv_code == 0:
        rec.dv = "forced_choice"
        label = t.get(i, "choice")
        row, ambiguous = _choice_row(cond.roles, label, where)
        if t.has("position_chosen"):
            chosen_pos = t.get(i, "position_chosen")
            chosen_rows = {r for r, role in enumerate(cond.roles) if role == label}
            if items_at is not None and _items(t.get(i, "items_chosen")) != items_at[chosen_pos]:
                raise ValueError(f"{where}: items_chosen disagrees with {chosen_pos}_items")
    elif dv_code == 1:
        rec.dv = "betting"
        rec.response = [_int(t, i, f"money_allocated_to_{role}") for role in cond.roles]
        if sum(rec.response) != 100:
            raise ValueError(f"{where}: bets {rec.response} do not sum to 100")
    elif dv_code == 2:
        rec.dv = "likert"
        rec.response = [_int(t, i, f"likert_value_{role}") for role in cond.roles]
    else:
        raise ValueError(f"{where}: participant_response_type_condition={dv_code}")

    display = solve_display(rec.objects, POSITIONS3, row_at, col_names, items_at, chosen_pos, chosen_rows)
    rec.display_order = display.display_order
    rec.feature_names = display.feature_names
    if rec.dv == "forced_choice":
        if display.chosen_row is not None:
            rec.choice = display.chosen_row
        else:
            rec.choice = row
            if ambiguous:
                rec.notes.append("choice_is_one_of_identical_objects")
    if any(p is None for p in rec.display_order):
        rec.notes.append("display_order_partial")
    if any(n is None for n in rec.feature_names):
        rec.notes.append("feature_names_partial")
    return [rec]


# ---- size (3-levels/size) ------------------------------------------------


@dataclass(frozen=True)
class SizeCondition:
    matrix_index: int  # 1..9 in scale_and_level.csv
    targets: Tuple[int, ...]  # 0-indexed rows
    uttered: Tuple[int, ...]  # 0-indexed columns, identical extensions


def read_size_conditions(data_dir: Path) -> Dict[int, SizeCondition]:
    path = data_dir / "3-levels" / "size" / "scale_and_level.csv"
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")
    rows = list(csv.DictReader(text.splitlines()))
    out: Dict[int, SizeCondition] = {}
    for r in rows:
        sl = int(r["scale_and_levels_condition"])
        m = int(r["matrix"])
        n_obj, n_feat = SIZE_MATRIX_ORDER[m - 1]
        if (int(r["objects"]), int(r["features"])) != (n_obj, n_feat):
            raise ValueError(f"scale_and_level.csv row {sl}: matrix {m} is not {n_obj}x{n_feat}")
        tokens = [tok.strip() for tok in re.split(r",| and ", r["referent-feature pair"]) if tok.strip()]
        tokens = [re.sub(r"^and ", "", tok) for tok in tokens]
        targets = tuple(int(tok) - 1 for tok in tokens if tok.isdigit())
        uttered = tuple(SIZE_FEATURES.index(tok) for tok in tokens if not tok.isdigit())
        matrix = SIZE_MATRICES[(n_obj, n_feat)]
        ext = {tuple(row[c] for row in matrix) for c in uttered}
        if len(ext) != 1:
            raise ValueError(f"scale_and_level.csv row {sl}: features {uttered} differ in extension")
        if not all(matrix[r][uttered[0]] for r in targets):
            raise ValueError(f"scale_and_level.csv row {sl}: a target lacks the uttered feature")
        if int(r["targ.features"]) != sum(row[uttered[0]] for row in matrix):
            raise ValueError(f"scale_and_level.csv row {sl}: targ.features disagrees with matrices.R")
        out[sl] = SizeCondition(m, targets, uttered)
    return out


def build_size_inference(ctx: Context, conditions: Dict[int, SizeCondition]) -> List[Record]:
    t, i = ctx.table, ctx.i
    where = f"{t.rel} row {i}"
    sl = _int(t, i, "scale_and_levels_condition")
    sc = conditions[sl]
    n_obj, n_feat = SIZE_MATRIX_ORDER[sc.matrix_index - 1]
    matrix = SIZE_MATRICES[(n_obj, n_feat)]
    rec = ctx.base(f"inference_{n_obj}obj_{n_feat}feat_sl{sl}")
    rec.item = t.get(i, "item")
    rec.objects = [list(row) for row in matrix]
    non_targets = [r for r in range(n_obj) if r not in sc.targets]
    roles = ["target" if r in sc.targets else "" for r in range(n_obj)]
    for k, r in enumerate(non_targets, start=1):
        roles[r] = f"distractor{k}"
    rec.object_roles = roles
    rec.query, rec.query_detail = "utterance", "word"
    rec.utterance = sc.uttered[0]
    if len(sc.uttered) > 1:
        rec.notes.append("utterance_is_one_of_identical_feature_columns")
    rec.framing = FRAMING_LABELS[_int(t, i, "linguistic_framing_condition")]
    names: List[Optional[str]] = [None] * n_feat
    names[rec.utterance] = t.get(i, "target_property")
    rec.feature_names = names
    rec.notes.append("feature_names_partial")
    rec.display_order = []
    rec.dv = "forced_choice"
    label = t.get(i, "choice")
    if label == "target":
        rec.choice = sc.targets[0]
        if len(sc.targets) > 1:
            rec.notes.append("choice_is_one_of_identical_objects")
    else:
        m = re.fullmatch(r"distractor(\d)", label)
        if m is None or int(m.group(1)) > len(non_targets):
            raise ValueError(f"{where}: choice {label!r} with {len(non_targets)} non-target objects")
        rec.choice = non_targets[int(m.group(1)) - 1]
    if len(non_targets) > 1:
        rec.notes.append("size_distractor_numbering_inferred")
    return [rec]


def build_size_prior(ctx: Context) -> List[Record]:
    t, i = ctx.table, ctx.i
    where = f"{t.rel} row {i}"
    n_obj, n_feat = SIZE_MATRIX_ORDER[_int(t, i, "matrix_number")]
    matrix = SIZE_MATRICES[(n_obj, n_feat)]
    slots = [_items(t.get(i, f"object_{k}_items")) for k in range(1, 5)]
    if any(slots[n_obj:]):
        raise ValueError(f"{where}: objects beyond the {n_obj} of matrix {n_obj}x{n_feat}")
    rec = ctx.base(f"prior_{n_obj}obj_{n_feat}feat")
    rec.item = t.get(i, "item")
    rec.objects = [list(row) for row in matrix]
    rec.object_roles = [f"ref{r + 1}" for r in range(n_obj)]
    rec.query = "prior"
    rec.query_detail = "size_salience_prior (prompt not in repo; linguistic_framing_condition=11)"
    rec.framing = FRAMING_LABELS[_int(t, i, "linguistic_framing_condition")]
    rec.dv = "forced_choice"
    positions = [f"slot{k}" for k in range(1, n_obj + 1)]
    chosen = _int(t, i, "position_chosen")
    if not 0 <= chosen < n_obj:
        raise ValueError(f"{where}: position_chosen={chosen} with {n_obj} objects")
    if _items(t.get(i, "items_chosen")) != slots[chosen]:
        raise ValueError(f"{where}: items_chosen disagrees with object_{chosen + 1}_items")
    items_at = {p: slots[k] for k, p in enumerate(positions)}
    all_rows = set(range(n_obj))
    display = solve_display(rec.objects, positions, {}, {}, items_at, positions[chosen], all_rows)
    rec.display_order = display.display_order
    rec.feature_names = display.feature_names
    if display.chosen_row is None:
        raise ValueError(f"{where}: chosen object not identifiable")
    rec.choice = display.chosen_row
    rec.notes.append("slot_order_object_1_to_n_assumed_left_to_right")
    if any(p is None for p in rec.display_order):
        rec.notes.append("display_order_partial")
    return [rec]


# ---- sequences (4-sequences) ----------------------------------------------

# Sequences.Rmd: the level of each trial by sequence_condition.
SEQUENCE_LEVELS = {
    "1w0w1b": [1, 0, 1],
    "0w1w1b": [0, 1, 1],
    "(0w1w)x3": [0, 1, 0, 1, 0, 1],
    "(0b1b)x3": [0, 1, 0, 1, 0, 1],
    "0w1w2w": [0, 1, 2],
    "2w1w0w": [2, 1, 0],
    "0w2w1w": [0, 2, 1],
    "1w2w0w": [1, 2, 0],
}
# The sequences were run by a later version of the experiment code (not in
# pragmods-expts). Its recorded target/distractor words and positions pin each
# level's display: the files with a level-2 trial put all three levels on one
# complex display, the others put levels 0 and 1 on one simple display (their
# words and positions swap between the two trials, as only the simple matrix
# allows). Levels 1 and 2 (complex) and 0 and 1 (simple) agree with the c1
# coding. Complex level 0 does not: its recorded distractor is the
# mustache-only object (M) with the mustache column, on all 200 rows, where the
# c1 code has GM and the glasses column. In the c1 code the distractor is
# always the "logical" object, so that is taken as this version's "logical"
# (an inference: no chosen position is recorded; it decides only which of the
# two literally false objects a level-0 "logical"/"foil" choice was).
SEQ_COMPLEX_L0 = C1Condition("complex", _COMPLEX, ("logical", "foil", "target"), 2, 0, 0, 2, 1)
SEQUENCE_CONDITIONS = {
    "simple": {0: (0, C1[0]), 1: (1, C1[1])},
    "complex": {0: (2, SEQ_COMPLEX_L0), 1: (3, C1[3]), 2: (4, C1[4])},
}
SEQUENCE_FAMILY = {
    "pragmods_seq": "simple",
    "pragmods_wx3": "simple",
    "pragmods_bx3": "simple",
    "pragmods_seq2": "complex",
    "pragmods_L2second": "complex",
}


@dataclass
class TrialRecord:
    """What one trial recorded about its display."""

    cond: C1Condition
    target_position: str
    target_prop: str
    distractor_position: Optional[str]  # None when the trial's distractor fields are unusable
    distractor_prop: Optional[str]


def _trial_record(t: SourceTable, i: int, k: int, cond: C1Condition) -> TrialRecord:
    tp, dp = t.get(i, f"target_prop_{k}"), t.get(i, f"distractor_prop_{k}")
    usable = tp != dp  # some versions recorded the target word as the distractor's
    return TrialRecord(
        cond,
        t.get(i, f"target_position_{k}"),
        tp,
        t.get(i, f"distractor_position_{k}") if usable else None,
        dp if usable else None,
    )


def _try_solve(objects, trials: Sequence[TrialRecord], positions: bool, names: bool, distractors: bool) -> Optional[Display]:
    row_at: Dict[str, Set[int]] = {}
    col_names: Dict[int, Set[str]] = {}
    for tr in trials:
        pins = [(tr.target_position, tr.cond.target_row, tr.cond.target_prop, tr.target_prop)]
        if distractors and tr.distractor_position is not None:
            pins.append((tr.distractor_position, tr.cond.distractor_row, tr.cond.distractor_prop, tr.distractor_prop))
        for pos, row, js_col, word in pins:
            if positions:
                if pos not in POSITIONS3:
                    return None
                row_at[pos] = row_at.get(pos, {0, 1, 2}) & {row}
            c = tr.cond.canonical_col(js_col)
            if names and c is not None:
                col_names.setdefault(c, set()).add(word)
    if any(not v for v in row_at.values()) or any(len(v) != 1 for v in col_names.values()):
        return None
    single = {c: next(iter(v)) for c, v in col_names.items()}
    if len(set(single.values())) != len(single):
        return None
    try:
        return solve_display(objects, POSITIONS3, row_at, single)
    except DisplayInconsistent:
        return None


def solve_trials(objects, trials: Sequence[TrialRecord]) -> Tuple[Optional[Display], List[str]]:
    """The display shared by ``trials``, dropping the least reliable records first.

    All target and distractor positions and words; else without the distractor
    words; else targets only. Returns the notes naming what was dropped.
    """
    attempts = [
        ((True, True, True), []),
        ((True, False, True), ["distractor_prop_inconsistent_ignored"]),
        ((True, True, False), ["distractor_fields_inconsistent_ignored"]),
    ]
    for (positions, names, distractors), notes in attempts:
        if not names:
            display = _try_solve(objects, trials, positions, False, distractors)
            if display is None:
                continue
            # target words alone still name their columns
            target_names = _try_solve(objects, trials, positions, True, False)
            if target_names is not None:
                display.feature_names = target_names.feature_names
            return display, notes
        display = _try_solve(objects, trials, positions, names, distractors)
        if display is not None:
            return display, notes
    return None, ["recorded_positions_inconsistent_with_level_labels"]


def _attach_display(rec: Record, display: Optional[Display], notes: List[str]) -> None:
    rec.notes.extend(notes)
    if display is None:
        rec.display_order = []
        rec.feature_names = [None] * len(rec.objects[0])
        return
    rec.display_order = display.display_order
    rec.feature_names = display.feature_names
    if any(p is None for p in rec.display_order):
        rec.notes.append("display_order_partial")
    if any(nm is None for nm in rec.feature_names):
        rec.notes.append("feature_names_partial")


def build_sequence(ctx: Context) -> List[Record]:
    t, i = ctx.table, ctx.i
    where = f"{t.rel} row {i}"
    seq = t.get(i, "sequence_condition")
    levels = SEQUENCE_LEVELS[seq]
    conditions = SEQUENCE_CONDITIONS[SEQUENCE_FAMILY[batch_code(t.rel)]]
    n = len(levels)
    for k in range(n + 1, 7):
        if t.has(f"choice_{k}") and t.get(i, f"choice_{k}") not in ("NA", "null"):
            raise ValueError(f"{where}: trial {k} recorded beyond the {n} of {seq}")
    # Trials on one item share one display (their words and positions agree).
    by_item: Dict[str, List[int]] = {}
    for k in range(1, n + 1):
        by_item.setdefault(t.get(i, f"item_{k}"), []).append(k)
    solved: Dict[int, Tuple[Optional[Display], List[str]]] = {}
    for ks in by_item.values():
        trials = [_trial_record(t, i, k, conditions[levels[k - 1]][1]) for k in ks]
        result = solve_trials(conditions[levels[ks[0] - 1]][1].objects, trials)
        for k in ks:
            solved[k] = result
    records = []
    for k in range(1, n + 1):
        level = levels[k - 1]
        sl, cond = conditions[level]
        rec = ctx.base(seq, trial_index=k - 1)
        rec.item = t.get(i, f"item_{k}")
        rec.objects = cond.objects
        rec.object_roles = list(cond.roles)
        rec.query, rec.query_detail = "utterance", f"word; sequence level {level} (scale_and_level {sl})"
        rec.utterance = cond.canonical_col(cond.target_prop)
        rec.framing = FRAMING_LABELS[_int(t, i, "linguistic_framing_condition")]
        rec.dv = "forced_choice"
        rec.choice, _ = _choice_row(cond.roles, t.get(i, f"choice_{k}"), f"{where} trial {k}")
        if cond is SEQ_COMPLEX_L0:
            rec.notes.append("sequence_complex_L0_logical_role_inferred_from_distractor")
        if t.get(i, f"target_prop_{k}") == t.get(i, f"distractor_prop_{k}"):
            rec.notes.append("distractor_prop_recorded_equal_to_target_prop")
        display, notes = solved[k]
        _attach_display(rec, display, list(notes))
        records.append(rec)
    return records


# ---- speakers (5-speakers) -------------------------------------------------

SPEAKER_MODALITY = {"baseline": "text", "checkbox": "checkbox", "virtual": "virtual_keyboard"}
CHECKBOX_COLUMNS = (
    "checkbox_target_positive",
    "checkbox_target_negative",
    "checkbox_distractor_positive",
    "checkbox_distractor_negative",
)


def build_speaker(ctx: Context) -> List[Record]:
    t, i = ctx.table, ctx.i
    where = f"{t.rel} row {i}"
    name = Path(t.rel).name
    modality = next(SPEAKER_MODALITY[k] for k in SPEAKER_MODALITY if f"_{k}" in name)
    sequential = "_seq_" in name
    level = 2 if "lvl2" in name else 1
    series = f"seq_L{level}" if sequential else "L1"
    simple = C1[1]
    n_features = _int(t, i, "features_in_referent_to_describe")
    rows = [r for r, obj in enumerate(simple.objects) if sum(obj) == n_features]
    if len(rows) != 1:
        raise ValueError(f"{where}: features_in_referent_to_describe={n_features}")

    prod = ctx.base(f"{series}:{modality}", trial_index=0)
    prod.item = t.get(i, "item_1" if sequential else "item")
    prod.objects = simple.objects
    prod.object_roles = list(simple.roles)
    prod.feature_names = [None, None]
    prod.referent = rows[0]
    prod.query = "production"
    prod.query_detail = "speaker: describe the referent for a listener"
    prod.framing = FRAMING_LABELS[_int(t, i, "linguistic_framing_condition")]
    prod.dv = "production"
    overspec = t.r_num(i, "overspec")
    response: Dict[str, object] = {"modality": modality}
    if modality == "checkbox":
        for col in CHECKBOX_COLUMNS:
            response[col] = t.r_str(i, col) == "TRUE"
    else:
        response["description"] = t.get(i, "free_response" if modality == "text" else "keyboard")
    response["overspec"] = None if overspec is None else int(overspec)
    prod.response = response
    prod.notes.append("speaker_display_simple_matrix_from_features_in_referent_0_1_2")
    if sequential:
        # target_position_1 is the one-feature object, distractor_position_1 the
        # two-feature one: position_to_describe agrees with it on every row.
        row_at: Dict[str, Set[int]] = {}
        for column, row in (("target_position_1", 1), ("distractor_position_1", 2)):
            pos = t.get(i, column)
            row_at[pos] = row_at.get(pos, {0, 1, 2}) & {row}
        display = solve_display(simple.objects, POSITIONS3, row_at, {})
        if t.has("position_to_describe"):
            describe = t.get(i, "position_to_describe")
            if display.display_order[POSITIONS3.index(describe)] != rows[0]:
                raise ValueError(f"{where}: position_to_describe disagrees with the recorded positions")
        else:
            prod.notes.append("display_order_from_target_distractor_positions_unverified")
        prod.display_order = display.display_order
    else:
        prod.display_order = []
    records = [prod]

    if sequential:
        lis = ctx.base(f"{series}:{modality}", trial_index=1)
        lis.item = t.get(i, "item_2")
        lis.framing = prod.framing
        lis.dv = "forced_choice"
        label = t.get(i, "choice_2")
        broken = t.get(i, "target_prop_2") == t.get(i, "distractor_prop_2")
        if broken:
            lis.notes.append("distractor_prop_recorded_equal_to_target_prop")
        if level == 2:
            cond = C1[4]
            lis.objects = cond.objects
            lis.object_roles = list(cond.roles)
            lis.query, lis.query_detail = "utterance", "word; level 2 (scale_and_level 4)"
            lis.utterance = cond.canonical_col(cond.target_prop)
            lis.choice, _ = _choice_row(cond.roles, label, where)
            lis.notes.append("speaker_seq_listener_matrix_complex_because_level_2")
            display, notes = solve_trials(cond.objects, [_trial_record(t, i, 2, cond)])
            _attach_display(lis, display, notes)
        else:
            lis.query, lis.query_detail = "utterance", "word; level 1 (matrix not recorded)"
            lis.response = {"choice_role": label}
            lis.notes.append("speaker_seq_level1_listener_matrix_unknown")
        records.append(lis)
    return records


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------


def _builder(experiment: str, rel: str, size_conditions: Dict[int, SizeCondition]) -> Callable[[Context], List[Record]]:
    if experiment == "size":
        if "salience_priors" in rel:
            return build_size_prior
        return lambda ctx: build_size_inference(ctx, size_conditions)
    if experiment == "sequences":
        return build_sequence
    if experiment == "speakers":
        return build_speaker
    return build_c1


def participant_files(data_dir: Path) -> List[str]:
    out = []
    for path in sorted(data_dir.rglob("*")):
        if not path.is_file() or path.suffix not in (".csv", ".tsv"):
            continue
        rel = path.relative_to(data_dir).as_posix()
        if EXCLUDED_DIR_NAMES & set(Path(rel).parts[:-1]):
            continue
        if rel in NON_PARTICIPANT_FILES:
            continue
        out.append(rel)
    return out


def build_records(data_dir: Path) -> Tuple[List[Record], Dict[str, int]]:
    size_conditions = read_size_conditions(data_dir)
    tables: Dict[str, SourceTable] = {}
    used: Set[str] = set()
    records: List[Record] = []
    for group in GROUPS:
        bound: List[Tuple[SourceTable, int]] = []
        for member in group.members:
            if member.rel not in tables:
                tables[member.rel] = read_source(data_dir, member.rel)
            t = tables[member.rel]
            used.add(member.rel)
            bound.extend((t, i) for i in range(len(t)) if member.keep(t, i))
        seen: Set[str] = set()
        for t, i in bound:
            reasons = group.criteria(t, i)
            worker = t.get(i, "workerid")
            if group.dedupe:
                if worker in seen:
                    reasons.append("duplicate_participant")
                seen.add(worker)
            ctx = Context(t, i, group.experiment, included=not reasons, reasons=reasons)
            records.extend(_builder(group.experiment, t.rel, size_conditions)(ctx))
    on_disk = set(participant_files(data_dir))
    if on_disk != used:
        raise ValueError(
            f"participant files not ingested: {sorted(on_disk - used)}; listed but absent: {sorted(used - on_disk)}"
        )
    row_counts = {rel: len(tables[rel]) for rel in sorted(used)}
    return records, row_counts


def git_commit(repo: Path) -> Tuple[str, bool]:
    try:
        sha = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain"], check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"{repo} is not a readable git clone of {PRAGMODS_REPO}: {exc}") from exc
    return sha, bool(dirty)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_outputs(pragmods_dir: Path, out_dir: Path, command: str) -> Tuple[Path, Path]:
    pragmods_dir = pragmods_dir.resolve()
    data_dir = pragmods_dir / "data"
    if not data_dir.is_dir():
        raise FileNotFoundError(f"{pragmods_dir} has no data/ directory; pass a clone of {PRAGMODS_REPO}")
    commit, dirty = git_commit(pragmods_dir)
    records, row_counts = build_records(data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / CSV_NAME
    frame = pd.DataFrame([r.as_row() for r in records], columns=COLUMNS)
    tmp = csv_path.with_suffix(".csv.tmp")
    frame.to_csv(tmp, index=False, lineterminator="\n")
    tmp.replace(csv_path)
    provenance = {
        "source_repo": PRAGMODS_REPO,
        "source_commit": commit,
        "source_worktree_dirty": dirty,
        "stimulus_code_repo": PRAGMODS_EXPTS_REPO,
        "stimulus_code_commit_transcribed": PRAGMODS_EXPTS_COMMIT,
        "generation_command": command,
        "excluded_directories": sorted(EXCLUDED_DIR_NAMES),
        "source_files": [
            {"path": rel, "rows": n, "sha256": _sha256(data_dir / rel)} for rel, n in row_counts.items()
        ],
        "auxiliary_files": [
            {"path": rel, "sha256": _sha256(data_dir / rel)} for rel in sorted(NON_PARTICIPANT_FILES)
        ],
        "output": {
            "path": CSV_NAME,
            "rows": len(frame),
            "included_rows": int(frame["included"].sum()),
            "sha256": _sha256(csv_path),
        },
    }
    prov_path = out_dir / PROVENANCE_NAME
    prov_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    return csv_path, prov_path


@dataclass
class Args:
    """Build the canonical pragmods trial CSV from a clone of langcog/pragmods."""

    pragmods_dir: Path
    """Path to a git clone of github.com/langcog/pragmods."""
    out_dir: Path = DEFAULT_OUT_DIR
    """Directory for pragmods_trials.csv and its provenance sidecar."""


def main(args: Args) -> None:
    command = "uv run python -m src.rsa.pragmods_ingest " + " ".join(shlex.quote(a) for a in sys.argv[1:])
    csv_path, prov_path = write_outputs(args.pragmods_dir, args.out_dir, command)
    print(f"wrote {csv_path}")
    print(f"wrote {prov_path}")


if __name__ == "__main__":
    main(tyro.cli(Args))
