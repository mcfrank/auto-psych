"""Sikos, Venhuizen, Drenhaus & Crocker (2021), "Reevaluating pragmatic reasoning in language games".

PLoS One 16(3): e0248388 (CC-BY 4.0); data are the article's supporting files
S1-S3 (``exp{1,2,3}_data``). A one-shot Frank & Goodman (2012) replication on
MTurk: each participant did one trial of one task on a display of three
coloured objects (colour + shape, e.g. ``orange.fish``):

* Listener — Robert says one word (a colour or a shape); pick his object
  (``query=utterance``);
* Salience — Robert says something incomprehensible; pick his object
  (``query=prior``);
* Speaker — describe the middle object to Robert with one of its two words
  (Exp. 1 only; ``dv=production``, the word in ``response``, the object in
  ``referent``).

Every feature on the screen is a word, so ``messages`` lists every feature
column. Columns are the display's colours, then its shapes, in order of first
appearance left to right. ``o1, o2, o3`` are the objects left to right (the
speaker's target, ``targ``, is always ``o2`` in Exps. 1 and 3: "a target
object that appeared in the middle position"); ``task.resp`` A/B/C is o1/o2/o3
(the A=o1 reading makes 98% of listener choices literally true; the reversed
one 77%).

Exclusions reproduce the paper's Ns exactly (checked in
``tests/test_rsa_ingest_sikos2021.py``): self-identified non-native or
non-fluent English (``language``/``nativeLang`` must be ``English`` and
``fluency`` ``fluent``), a wrong answer to either attention question
(``attnQ.acc``), and, for listeners, a choice the word is not literally true
of. Exp. 2's ``task.resp`` is the response (its ``task.resp1`` is not used).
Demographics, browser/OS strings, timings and all free text are dropped.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from src.rsa.ingest.common import Source, Trial
from src.rsa.ingest.fetch import SourceFile

SOURCE = "sikos_2021"
_PLOS = "https://journals.plos.org/plosone/article/file?type=supplementary&id=10.1371/journal.pone.0248388"
FILES = (
    SourceFile(
        "exp1_data_s001.csv",
        f"{_PLOS}.s001",
        "3e39b2ccbab76f0a345423ec9b4fdeff9c00c4bb149d52de4a577e9210e4f766",
        1982544,
    ),
    SourceFile(
        "exp2_data_s002.csv",
        f"{_PLOS}.s002",
        "a2b7140af017b627c61a79522cc40a520126510e787e4463929f9bcfaa9dcb43",
        670778,
    ),
    SourceFile(
        "exp3_data_s003.csv",
        f"{_PLOS}.s003",
        "10c1d093f02e2d43bfad3acdcf996364ece9a9104f10103b109d6cc168976b09",
        511365,
    ),
)
RESPONSE_OBJECT = {"A": 0, "B": 1, "C": 2}
TASKS = ("listener", "salience", "speaker")
FRAMING = "one_word"
DETAIL = {
    "listener": "word",  # Robert says one word (a colour or a shape)
    "salience": "mumble",  # Robert says something incomprehensible
    "speaker": "speaker_choose_word",  # describe the middle object with one of its two words
}
STIMULUS_TYPE = {"C3": "iconic", "FG": "geometric"}


@dataclass(frozen=True)
class Experiment:
    file: str
    name: str
    language_col: str
    context_col: str


EXPERIMENTS = (
    Experiment("exp1_data_s001.csv", "sikos2021_e1", "language", "cond"),
    Experiment("exp2_data_s002.csv", "sikos2021_e2", "nativeLang", "cond"),
    Experiment("exp3_data_s003.csv", "sikos2021_e3", "language", "context"),
)


def display_of(codes: List[str]) -> Tuple[Tuple[str, ...], Tuple[Tuple[int, ...], ...]]:
    """Feature names (colours, then shapes, by first appearance) and 0/1 objects."""
    parts = []
    for code in codes:
        bits = code.split(".")
        if len(bits) != 2 or not all(bits):
            raise ValueError(f"object {code!r} is not colour.shape")
        parts.append(bits)
    colours = list(dict.fromkeys(p[0] for p in parts))
    shapes = list(dict.fromkeys(p[1] for p in parts))
    if set(colours) & set(shapes):
        raise ValueError(f"a word is both a colour and a shape in {codes}")
    names = tuple(colours + shapes)
    objects = tuple(tuple(int(n in p) for n in names) for p in parts)
    return names, objects


def _word_index(word: str, names: Tuple[str, ...]) -> int:
    if word not in names:
        raise ValueError(f"word {word!r} names no feature of the display {names}")
    return names.index(word)


def build_experiment(exp: Experiment, path: Path) -> List[Trial]:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    trials: List[Trial] = []
    for i, row in df.reset_index(drop=True).iterrows():
        task = row["task"]
        if task not in TASKS:
            raise ValueError(f"{exp.file} row {i}: unknown task {task!r}")
        codes = [row["o1"], row["o2"], row["o3"]]
        names, objects = display_of(codes)
        target = {"o1": 0, "o2": 1, "o3": 2}[row["targ"]]
        roles = ["target" if j == target else "other" for j in range(3)]
        if "c.comp" in df.columns:
            if row["target"] != codes[target]:
                raise ValueError(f"{exp.file} row {i}: target {row['target']!r} is not {row['targ']}")
            for col, role in (("c.comp", "colour_competitor"), ("s.comp", "shape_competitor")):
                roles[codes.index(row[col])] = role
        covariates: Dict[str, object] = {"display": row["display"]}
        if "side" in df.columns:
            covariates["side"] = row["side"]
        if "stimType" in df.columns:
            covariates["stimulus_type"] = STIMULUS_TYPE[row["stimType"]]
        if "task.likelihood" in df.columns:
            covariates["task_likelihood"] = int(row["task.likelihood"])
        utterance: Optional[int] = None
        choice: Optional[int] = None
        response = None
        referent: Optional[int] = None
        if task == "speaker":
            word = row["task.resp"]
            options = [row["speak.word.L"], row["speak.word.R"]]
            if word not in options:
                raise ValueError(f"{exp.file} row {i}: speaker word {word!r} not among {options}")
            if not all(objects[target][_word_index(w, names)] for w in options):
                raise ValueError(f"{exp.file} row {i}: speaker options {options} are not the target's words")
            response = {"word": word, "feature": _word_index(word, names), "options": options}
            referent = target
            query, dv = "production", "production"
        else:
            choice = RESPONSE_OBJECT[row["task.resp"]]
            dv = "forced_choice"
            if task == "listener":
                query = "utterance"
                utterance = _word_index(row["listen.word"], names)
            else:
                query = "prior"
        reasons = []
        if not (row[exp.language_col] == "English" and row["fluency"] == "fluent"):
            reasons.append("non_native_or_non_fluent")
        if row["attnQ.acc"] != "1":
            reasons.append("attention_check_failed")
        if task == "listener" and not objects[choice][utterance]:
            reasons.append("listener_choice_not_literally_true")
        trials.append(
            Trial(
                source=SOURCE,
                participant_id=f"{SOURCE}:{exp.name.rsplit('_', 1)[1]}:{i + 1:04d}",
                batch=task,
                source_file=exp.file,
                series="plos:10.1371/journal.pone.0248388",
                experiment=exp.name,
                condition=row[exp.context_col],
                trial_index=0,
                item=covariates.get("stimulus_type", "iconic"),
                feature_names=names,
                objects=objects,
                object_roles=roles,
                display_order=(0, 1, 2),
                query=query,
                query_detail=DETAIL[task],
                framing=FRAMING,
                dv=dv,
                messages=tuple(range(len(names))),
                included=not reasons,
                exclusion_reason=";".join(reasons),
                utterance=utterance,
                choice=choice,
                response=response,
                referent=referent,
                covariates=covariates,
            )
        )
    return trials


def build(paths: Dict[str, Path]) -> List[Trial]:
    trials: List[Trial] = []
    for exp in EXPERIMENTS:
        trials.extend(build_experiment(exp, paths[exp.file]))
    return trials


SIKOS_2021 = Source(
    name=SOURCE,
    citation=(
        "Sikos, L., Venhuizen, N. J., Drenhaus, H., & Crocker, M. W. (2021). Reevaluating pragmatic "
        "reasoning in language games. PLoS One, 16(3), e0248388. https://doi.org/10.1371/journal.pone.0248388"
    ),
    licence="CC-BY-4.0 (PLoS One article and supporting information)",
    landing_url="https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0248388",
    files=FILES,
    build=build,
    commit_csv=True,
)
