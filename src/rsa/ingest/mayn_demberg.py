"""Mayn & Demberg's reference games (the Franke & Degen 2016 paradigm).

Three sources, one trial coding:

* ``mayn_demberg_2026`` — Mayn & Demberg (2026), PLoS One 21(2): e0339899,
  "Sources of individual variability in a pragmatic reference game", OSF
  5ab3f (CC-BY 4.0). Shapes and colours, 300 + 6 participants x 66 trials.
* ``mayn_demberg_2023`` — Mayn & Demberg (2023), Open Mind 7: 156-178,
  "High performance on a pragmatic task may not be the result of successful
  reasoning", github.com/sashamayn/refgame_stimuli_methods (no licence).
  Four experiments: replication (monsters), remapped message set, all
  messages available, shapes and colours.
* ``mayn_demberg_2022`` — Mayn & Demberg (2022), CogSci 44: 3016-3022,
  "Individual differences in a pragmatic reference game",
  github.com/sashamayn/refgame_cogsci22 (no licence). Monsters, a pilot (47)
  and the main study (68).

Every trial shows three objects, each one value on each of two dimensions
(shape/creature, colour/accessory), and one message naming a feature. Some
features have no message (in the original game: square/robot and blue/scarf),
which is what makes the critical trials solvable. A feature is a 0/1 column
of a fixed six-column vocabulary per stimulus family, in an order under which
the two families correspond (the 2023 paper's Exp. 4 mapping: square = robot,
triangle = green monster, circle = purple monster, blue = scarf, green = red
hat, red = blue hat):

    shapes:   circle, triangle, square, green, red, blue
    monsters: purple_monster, green_monster, robot, red_hat, blue_hat, scarf

so the original message set is columns {0, 1, 3, 4} in both.

Object roles are derived from the features, never from the files' label
columns (``target``/``competitor``/``distractor``, ``targetpos``,
``answer_which``, ``correct``), which are wrong in places: item 20's target
and distractor labels are swapped in every 2023 experiment and in the 2022
main study, item 51's target and competitor in all of 2022, ``targetpos`` is
wrong for item 20 and for some ambiguous fillers in 2026, and ``answer_which``
is arbitrary when the two identical objects of an ambiguous filler are chosen.
The chosen object is read from ``answer_order`` (its screen position in
``presentation_order``), checked against ``answer``.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import pandas as pd

from src.rsa.ingest.common import Source, Trial
from src.rsa.ingest.fetch import SourceFile

SHAPES_CODES = ("ci", "tr", "sq", "gr", "re", "bl")
SHAPES_NAMES = ("circle", "triangle", "square", "green", "red", "blue")
MONSTER_CODES = ("pm", "gm", "ro", "rh", "bh", "sc")
MONSTER_NAMES = ("purple_monster", "green_monster", "robot", "red_hat", "blue_hat", "scarf")
FAMILIES = {"shapes": (SHAPES_CODES, SHAPES_NAMES), "monsters": (MONSTER_CODES, MONSTER_NAMES)}
DIMENSION = (0, 0, 0, 1, 1, 1)  # shape/creature columns, then colour/accessory columns

ORIGINAL_MESSAGES = (0, 1, 3, 4)  # Franke & Degen: no square/robot, no blue/scarf
REMAPPED_MESSAGES = (1, 2, 3, 5)  # 2023 Exp. 2: green monster, robot, red hat, scarf
ALL_MESSAGES = (0, 1, 2, 3, 4, 5)  # 2023 Exp. 3

UNAMBIGUOUS_FILLERS = ("filler unambiguous", "filler type a", "filler type b", "filler type c")
ITEM_TYPES = ("target simple", "target complex", "filler ambiguous", *UNAMBIGUOUS_FILLERS)
ACCURACY_THRESHOLD = 0.8
EXCLUDING_STRATEGIES = ("misunderstood_instr", "odd_one_out")
FRAMING = "message_icon"


@dataclass(frozen=True)
class Game:
    """One experiment's version of the game."""

    experiment: str
    family: str
    messages: Tuple[int, ...]  # feature columns the speaker could name
    design_messages: Tuple[int, ...]  # the message set the items were designed for (roles)


# The message is shown as a picture of one feature: a shape contour or a colour
# tube (shapes), a creature or an accessory without the rest (monsters).
QUERY_DETAIL = "picture_of_feature"


@dataclass(frozen=True)
class Coded:
    """One response of the Franke & Degen game, coded."""

    objects: Tuple[Tuple[int, ...], ...]
    roles: Tuple[str, ...]
    display_order: Tuple[int, ...]
    utterance: int
    choice: int
    notes: Tuple[str, ...]


def features_of(code: str, family: str) -> Tuple[int, ...]:
    codes = FAMILIES[family][0]
    parts = code.strip().split("_")
    if len(parts) != 2 or any(p not in codes for p in parts):
        raise ValueError(f"object code {code!r} is not two {family} feature codes {codes}")
    vec = [0] * len(codes)
    for p in parts:
        vec[codes.index(p)] = 1
    if [DIMENSION[i] for i, v in enumerate(vec) if v] != [0, 1]:
        raise ValueError(f"object code {code!r} is not one value on each dimension")
    return tuple(vec)


def derive_roles(objects: Sequence[Tuple[int, ...]], utterance: int, design_messages: Sequence[int]) -> Tuple[int, ...]:
    """Indices (into ``objects``, the labelled target/competitor/distractor) in role order.

    Returns the order (target, competitor, distractor), or for two identical
    objects the message is true of (twins) (twin, twin, distractor). The
    target of a critical trial is the object the message is true of that
    cannot be named unambiguously; its competitor can (it has a nameable
    feature no other object has). Unambiguous fillers keep the labelled
    competitor where it is not the target.
    """
    true_of = [i for i, o in enumerate(objects) if o[utterance]]
    if len(true_of) == 1:
        target = true_of[0]
        rest = [i for i in (1, 2, 0) if i != target]
        return (target, *rest)
    if len(true_of) != 2:
        raise ValueError(f"message column {utterance} is true of {len(true_of)} of {objects}")
    a, b = true_of
    other = 3 - a - b
    if objects[a] == objects[b]:
        return (a, b, other)

    def nameable_alone(i: int) -> bool:
        return any(objects[i][f] and sum(o[f] for o in objects) == 1 for f in design_messages)

    if nameable_alone(a) == nameable_alone(b):
        raise ValueError(f"cannot tell the target from the competitor in {objects} (message {utterance})")
    target, competitor = (b, a) if nameable_alone(a) else (a, b)
    return (target, competitor, other)


def code_trial(row: pd.Series, game: Game) -> Coded:
    codes = FAMILIES[game.family][0]
    message = str(row["message"]).strip()
    if message not in codes:
        raise ValueError(f"message {message!r} is not a {game.family} feature code")
    utterance = codes.index(message)
    if utterance not in game.messages:
        raise ValueError(f"{game.experiment}: message {message!r} is not in the message set {game.messages}")
    labelled = [str(row[c]).strip() for c in ("target", "competitor", "distractor")]
    labelled_vecs = [features_of(c, game.family) for c in labelled]
    order = derive_roles(labelled_vecs, utterance, game.design_messages)
    canon = [labelled[i] for i in order]
    twins = canon[0] == canon[1]
    roles = ("twin_1", "twin_2", "distractor") if twins else ("target", "competitor", "distractor")
    notes: List[str] = []
    if not twins and canon[0] != labelled[0]:
        notes.append("labelled_target_is_not_the_feature_target")
    layout = [x.strip() for x in str(row["presentation_order"]).split(",")]
    if sorted(layout) != sorted(canon):
        raise ValueError(f"presentation_order {layout} is not the objects {canon}")
    display: List[int] = []
    for code in layout:
        idx = next(i for i, c in enumerate(canon) if c == code and i not in display)
        display.append(idx)
    position = int(row["answer_order"]) - 1
    if layout[position] != str(row["answer"]).strip():
        raise ValueError(f"answer_order {position + 1} of {layout} is not the answer {row['answer']!r}")
    return Coded(
        objects=tuple(features_of(c, game.family) for c in canon),
        roles=roles,
        display_order=tuple(display),
        utterance=utterance,
        choice=display[position],
        notes=tuple(notes),
    )


def _read(path: Path) -> pd.DataFrame:
    # Untrusted CSV text: read as strings, no type inference beyond pandas' own.
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def _game_trials(
    source: str,
    df: pd.DataFrame,
    *,
    pid_col: str,
    game_of: Dict[str, Game],
    exp_col: Optional[str],
    trial_offset: int,
    source_file: str,
    series: str,
    batch_col: Optional[str] = None,
    pid_prefix: str = "",
) -> Tuple[List[Trial], Dict[Tuple[str, str], List[Trial]]]:
    """Code every row; return the trials and them grouped by (experiment key, participant)."""
    trials: List[Trial] = []
    by_participant: Dict[Tuple[str, str], List[Trial]] = {}
    for _, row in df.iterrows():
        key = row[exp_col] if exp_col else ""
        game = game_of[key]
        itemtype = row["itemtype"]
        if itemtype not in ITEM_TYPES:
            raise ValueError(f"{source}: unknown itemtype {itemtype!r}")
        coded = code_trial(row, game)
        names = FAMILIES[game.family][1]
        pid = str(row[pid_col])
        trial = Trial(
            source=source,
            participant_id=f"{source}:{pid_prefix}{pid}",
            batch=row[batch_col] if batch_col else "",
            source_file=source_file,
            series=series,
            experiment=game.experiment,
            condition=itemtype,
            trial_index=int(row["trialid"]) - trial_offset,
            item=f"{game.family}_item{int(row['itemid'])}",
            feature_names=names,
            objects=coded.objects,
            object_roles=coded.roles,
            display_order=coded.display_order,
            query="utterance",
            query_detail=QUERY_DETAIL,
            framing=FRAMING,
            dv="forced_choice",
            messages=game.messages,
            included=True,
            utterance=coded.utterance,
            choice=coded.choice,
            notes=list(coded.notes),
        )
        trials.append(trial)
        by_participant.setdefault((key, pid), []).append(trial)
    for (key, pid), ts in by_participant.items():
        if len(ts) != 66 or len({t.trial_index for t in ts}) != 66:
            raise ValueError(f"{source}: participant {pid} ({key}) has {len(ts)} trials, not 66 distinct")
        if sorted(t.trial_index for t in ts) != list(range(66)):
            raise ValueError(f"{source}: participant {pid} trial numbers are not 0..65")
    return trials, by_participant


def unambiguous_accuracy(trials: Sequence[Trial]) -> float:
    """Share of unambiguous-filler trials on which the (feature-derived) target was chosen."""
    fillers = [t for t in trials if t.condition in UNAMBIGUOUS_FILLERS]
    if len(fillers) != 33:
        raise ValueError(f"expected 33 unambiguous fillers, got {len(fillers)}")
    return sum(t.object_roles[t.choice] == "target" for t in fillers) / len(fillers)


def _exclude(trials: Sequence[Trial], reasons: Sequence[str]) -> None:
    for t in trials:
        t.included = not reasons
        t.exclusion_reason = ";".join(reasons)


def _strategy_tags(ann: pd.DataFrame, pid_col: str, keys: Sequence[str]) -> Dict[Tuple[str, ...], Dict[str, str]]:
    """{(key..., participant): {"strategy_tag_simple": tag, "strategy_tag_complex": tag}}."""
    tags: Dict[Tuple[str, ...], Dict[str, str]] = {}
    for _, row in ann.iterrows():
        kind = {"target simple": "simple", "target complex": "complex"}.get(row["itemtype"])
        if kind is None:
            raise ValueError(f"annotation for unexpected itemtype {row['itemtype']!r}")
        k = tuple(str(row[c]) for c in keys) + (str(row[pid_col]),)
        slot = tags.setdefault(k, {})
        if f"strategy_tag_{kind}" in slot:
            raise ValueError(f"two {kind} annotations for {k}")
        slot[f"strategy_tag_{kind}"] = row["tag_both"]
    return tags


# --------------------------------------------------------------------------
# Mayn & Demberg 2026 (PLoS One; OSF 5ab3f; CC-BY 4.0)
# --------------------------------------------------------------------------

MD2026_FILES = (
    SourceFile(
        "data/main_task_data.csv",
        "https://osf.io/download/u2xjf/",
        "56c6793268934faa155eac9b412b8e649977d580d786b2fa56cc0c6ad4a084ce",
        2050067,
    ),
    SourceFile(
        "data/ID_scores_with_exclusions.csv",
        "https://osf.io/download/53br2/",
        "9f4efa39678e3b2e4f70e31caa48bf786fbffd66b40edad350db929e8aaa1527",
        26375,
    ),
    SourceFile(
        "data/annotations.csv",
        "https://osf.io/download/dh8b6/",
        "969547baf45e550eb68837a329f10217b446243fdc1e367398910b945cbf7451",
        133008,
    ),
)
MD2026_GAME = Game("md2026_shapes", "shapes", ORIGINAL_MESSAGES, ORIGINAL_MESSAGES)


def build_2026(paths: Dict[str, Path]) -> List[Trial]:
    source = "mayn_demberg_2026"
    main = _read(paths["data/main_task_data.csv"])
    ids = _read(paths["data/ID_scores_with_exclusions.csv"])
    ann = _read(paths["data/annotations.csv"])
    trials, groups = _game_trials(
        source,
        main,
        pid_col="participantId",
        game_of={"": MD2026_GAME},
        exp_col=None,
        trial_offset=0,
        source_file="data/main_task_data.csv",
        series="osf:5ab3f",
    )
    main_exclude = dict(zip(ids["participantId"], ids["main_exclude"]))
    tags = _strategy_tags(ann, "participantId", ())
    for (_, pid), ts in groups.items():
        reasons: List[str] = []
        acc = unambiguous_accuracy(ts)
        if pid not in main_exclude:
            # Not among the 300 of the paper's first session (ids 301-306 have
            # no individual-difference scores); the paper's N = 254 is of the 300.
            reasons.append("not_in_paper_sample")
        else:
            flagged = main_exclude[pid] == "1"
            if flagged != (acc < ACCURACY_THRESHOLD):
                raise ValueError(f"{source}: main_exclude={main_exclude[pid]} for {pid} but accuracy {acc:.3f}")
            if flagged:
                reasons.append("unambiguous_accuracy_below_0.8")
        tag = tags.get((pid,))
        if tag is None or len(tag) != 2:
            raise ValueError(f"{source}: participant {pid} lacks two strategy annotations")
        for strategy in EXCLUDING_STRATEGIES:
            if strategy in tag.values():
                reasons.append(f"strategy_{strategy}")
        _exclude(ts, reasons)
        for t in ts:
            t.covariates = dict(tag)
    return trials


MAYN_DEMBERG_2026 = Source(
    name="mayn_demberg_2026",
    citation=(
        "Mayn, A., & Demberg, V. (2026). Sources of individual variability in a pragmatic reference game: "
        "Effects of logical reasoning and Theory of Mind. PLoS One, 21(2), e0339899. "
        "https://doi.org/10.1371/journal.pone.0339899"
    ),
    licence="CC-BY-4.0 (OSF project 5ab3f)",
    landing_url="https://osf.io/5ab3f/",
    files=MD2026_FILES,
    build=build_2026,
    commit_csv=True,
)


# --------------------------------------------------------------------------
# Mayn & Demberg 2023 (Open Mind; GitHub refgame_stimuli_methods; no licence)
# --------------------------------------------------------------------------

MD2023_REPO = "https://github.com/sashamayn/refgame_stimuli_methods"
MD2023_COMMIT = "d7d4aecb55b0460bc32e366d3e0b2e4552b6015c"
_MD2023_RAW = f"https://raw.githubusercontent.com/sashamayn/refgame_stimuli_methods/{MD2023_COMMIT}"
MD2023_FILES = (
    SourceFile(
        "data/all_experiments.csv",
        f"{_MD2023_RAW}/data/all_experiments.csv",
        "81e3d5d6c85c5c84aa4874791a5df7575a12bc2baeaee85a3a2c8d4c44b5bf6b",
        2122090,
    ),
    SourceFile(
        "data/all_annotations.csv",
        f"{_MD2023_RAW}/data/all_annotations.csv",
        "7f8343a491e9e6d6ecaf73d95a45a52a631239fbdd2ca0e8552dcdf353a7d4c8",
        76744,
    ),
)
MD2023_GAMES = {
    "1": Game("md2023_e1_replication", "monsters", ORIGINAL_MESSAGES, ORIGINAL_MESSAGES),
    "2": Game("md2023_e2_remapped", "monsters", REMAPPED_MESSAGES, REMAPPED_MESSAGES),
    "3": Game("md2023_e3_all_messages", "monsters", ALL_MESSAGES, ORIGINAL_MESSAGES),
    "4": Game("md2023_e4_shapes", "shapes", ORIGINAL_MESSAGES, ORIGINAL_MESSAGES),
}


def build_2023(paths: Dict[str, Path]) -> List[Trial]:
    source = "mayn_demberg_2023"
    df = _read(paths["data/all_experiments.csv"])
    ann = _read(paths["data/all_annotations.csv"])
    unknown = set(df["experiment"]) - set(MD2023_GAMES)
    if unknown:
        raise ValueError(f"{source}: unknown experiments {sorted(unknown)}")
    # The strategy_* columns are a second response, after the 66 trials, to a
    # re-shown simple or complex item; they are not trials here (README).
    trials, groups = _game_trials(
        source,
        df,
        pid_col="participantid",
        game_of=MD2023_GAMES,
        exp_col="experiment",
        trial_offset=1,
        source_file="data/all_experiments.csv",
        series="github:sashamayn/refgame_stimuli_methods",
    )
    ids_per_exp: Dict[str, Set[str]] = {}
    for key, pid in groups:
        ids_per_exp.setdefault(pid, set()).add(key)
    if any(len(v) > 1 for v in ids_per_exp.values()):
        raise ValueError(f"{source}: a participant id recurs across experiments; ids would collide")
    tags = _strategy_tags(ann, "participantid", ("experiment",))
    for (key, pid), ts in groups.items():
        acc = unambiguous_accuracy(ts)
        _exclude(ts, ["unambiguous_accuracy_below_0.8"] if acc < ACCURACY_THRESHOLD else [])
        tag = tags.get((key, pid))
        if tag is None or len(tag) != 2:
            raise ValueError(f"{source}: experiment {key} participant {pid} lacks two strategy annotations")
        for t in ts:
            t.covariates = dict(tag)
    return trials


MAYN_DEMBERG_2023 = Source(
    name="mayn_demberg_2023",
    citation=(
        "Mayn, A., & Demberg, V. (2023). High performance on a pragmatic task may not be the result of "
        "successful reasoning: On the importance of eliciting participants' reasoning strategies. "
        "Open Mind, 7, 156-178. https://doi.org/10.1162/opmi_a_00077"
    ),
    licence="none stated (GitHub repository without a licence file): derived CSV is not committed",
    landing_url=MD2023_REPO,
    files=MD2023_FILES,
    build=build_2023,
    commit_csv=False,
    pinned_commit=MD2023_COMMIT,
)


# --------------------------------------------------------------------------
# Mayn & Demberg 2022 (CogSci; GitHub refgame_cogsci22; no licence)
# --------------------------------------------------------------------------

MD2022_REPO = "https://github.com/sashamayn/refgame_cogsci22"
MD2022_COMMIT = "b3a2b5e033cd703e8dc437466c066761d4f41672"
MD2022_FILES = (
    SourceFile(
        "main_task_data.csv",
        f"https://raw.githubusercontent.com/sashamayn/refgame_cogsci22/{MD2022_COMMIT}/main_task_data.csv",
        "11d001543ff612e8a24a08e687841ff7ed710153670f4923c8928c59cda8e371",
        830035,
    ),
)
MD2022_GAMES = {
    "pilot": Game("md2022_pilot", "monsters", ORIGINAL_MESSAGES, ORIGINAL_MESSAGES),
    "main": Game("md2022_main", "monsters", ORIGINAL_MESSAGES, ORIGINAL_MESSAGES),
}


def build_2022(paths: Dict[str, Path]) -> List[Trial]:
    source = "mayn_demberg_2022"
    df = _read(paths["main_task_data.csv"])
    unknown = set(df["study"]) - set(MD2022_GAMES)
    if unknown:
        raise ValueError(f"{source}: unknown study {sorted(unknown)}")
    trials, groups = _game_trials(
        source,
        df,
        pid_col="participant_id",
        game_of=MD2022_GAMES,
        exp_col="study",
        trial_offset=0,
        source_file="main_task_data.csv",
        series="github:sashamayn/refgame_cogsci22",
        batch_col="study",
    )
    seen: Counter = Counter(pid for _, pid in groups)
    if any(n > 1 for n in seen.values()):
        raise ValueError(f"{source}: a participant id recurs across pilot and main")
    # The analysis scripts exclude no one (and every participant reaches 0.8
    # on the unambiguous fillers); all are included.
    for ts in groups.values():
        _exclude(ts, [])
    return trials


MAYN_DEMBERG_2022 = Source(
    name="mayn_demberg_2022",
    citation=(
        "Mayn, A., & Demberg, V. (2022). Individual differences in a pragmatic reference game. "
        "Proceedings of the Annual Meeting of the Cognitive Science Society, 44, 3016-3022."
    ),
    licence="none stated (GitHub repository without a licence file): derived CSV is not committed",
    landing_url=MD2022_REPO,
    files=MD2022_FILES,
    build=build_2022,
    commit_csv=False,
    pinned_commit=MD2022_COMMIT,
)
