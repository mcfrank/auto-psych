"""A frozen paradigm and its design -> what each participant sees, and the data back
(docs/auto_open/PLAN.md §4: the page, the pilot and the data all come from these lists).

The plan had an implementer agent write each paradigm's jsPsych page. Every spec uses one of
six response widgets with SVG or text materials, so the page is instead one generic, tested
player (`src/rsa_open/page.html`, `src.rsa_open.page`) of the lists built here, and the
simulated pilot (`src.rsa_open.pilot`) reads the same lists: what LLM participants see is
what people see.

**Trial lists** (`trial_lists`): one list per participant, a pure function of (paradigm, design,
seed, list index), so randomisation happens here, not in the page:
- **test trials**: the design's units (`src.rsa_open.design`: a scene and, by response type, a
  probed object, a target, a judged message). List ``p`` takes ``trials_per_participant``
  consecutive units of one shuffled order of the design, starting at ``p x trials``, so every
  unit is answered equally often (to within one) across lists; the list shuffles them.
- **catch trials** (``design.catch_trials``) at random positions, and one **practice** trial
  first: a scene with a word true of exactly one object (the closed campaign's
  `_catch_spec`), each with what passing means (`CATCH_RULES`). A judgment's catch trials
  alternate that word for its object (answer yes) and for another object, of which it is false
  (answer no), so answering one way fails.
- **domains** rotate (least used first, not the previous trial's when another can serve), and
  each trial maps the scene's abstract features onto distinct features of its domain at random;
  the objects' screen order is a random permutation (``display_order[p]`` is the scene's object
  at screen position ``p``).
- **rendering**: each object is drawn as an SVG (the domain's base, then its features' fragments,
  in a 100 x 100 box) or written as a phrase ("a house with a garden and with a pool"); the
  prompt shows the message as the speaker's word. A production trial's buttons are the scene's
  words in a random order; a slider or rating names its ends.

**Data**: the page records, per trial, the trial's scene, its list fields and ``response``
in the link's form (`src.rsa_open.links.LinkLikelihood.encode`), with objects and messages as
scene indices, never screen positions. `scenes_and_responses` turns test rows back into scenes
(the design pool's abstract feature names, as the theories saw in design) and responses, and
`catch_passed` scores a catch trial.
"""

from __future__ import annotations

import hashlib
import html
import json
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from src.rsa.context import Context
from src.rsa.design_space import feature_names as abstract_feature_names

SCHEMA = "rsa_open_lists/1"
PRACTICE_SHAPE = (3, 2)
SUBSET_STREAM = 2**31 - 1
UNIT_FIELDS = ("probe", "target", "utterance")
PHASES = ("practice", "test", "catch")


# --------------------------------------------------------------------------- the design

def design_units(report: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """The design's units from an EIG report (`src.rsa_open.design.score_paradigm`)."""
    units = []
    for i, u in enumerate(report["design"]):
        unit = {"objects": [list(map(int, row)) for row in u["objects"]], "heard": u["heard"],
                **{k: int(u[k]) for k in UNIT_FIELDS if k in u}}
        scene_of(unit)  # a unit that is not a valid scene fails here
        units.append(unit)
    if not units:
        raise ValueError("the design has no units")
    return units


def design_sha256(units: Sequence[Mapping[str, Any]]) -> str:
    blob = json.dumps(list(units), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


def scene_of(trial: Mapping[str, Any]) -> Context:
    """The abstract scene of a unit or trial: the design pool's feature names."""
    objects = tuple(tuple(int(v) for v in row) for row in trial["objects"])
    return Context(objects=objects, feature_names=abstract_feature_names(len(objects[0])),
                   utterance=trial["heard"])


# --------------------------------------------------------------------------- catch trials

def catch_scene(shape: Tuple[int, int], rng: np.random.Generator) -> Tuple[Tuple[Tuple[int, ...], ...], int, int]:
    """An unambiguous scene of this shape: (objects, a word true of exactly one object, that object).

    The closed campaign's catch trials (`src.rsa.experiment.design._catch_spec`, drawing the same
    from ``rng``), here because the agents' tree leaves the closed experiment build out: the
    designer's ``validate`` crashed on a missing module in run B's experiment 3 (2026-10-24)."""
    from src.rsa.design_space import games

    n_obj, n_feat = shape
    candidates = []
    for synonyms in (False, True):
        candidates = [(matrix, word) for matrix in games(n_obj, n_feat, synonyms) for word in range(n_feat)
                      if sum(row[word] for row in matrix) == 1]
        if candidates:
            break
    if not candidates:
        raise ValueError(f"no {n_obj}x{n_feat} game has a word true of exactly one object")
    matrix, word = candidates[int(rng.integers(len(candidates)))]
    objects = tuple(tuple(int(v) for v in row) for row in matrix)
    return objects, word, next(i for i, row in enumerate(objects) if row[word])


def _catch(shape: Tuple[int, int], rng: np.random.Generator, response_type: str, k: int,
           options: Mapping[str, int]) -> Dict[str, Any]:
    """An unambiguous trial and what passing it means, for the ``k``-th catch trial."""
    scene, word, target = catch_scene(shape, rng)
    objects = [list(r) for r in scene]
    trial: Dict[str, Any] = {"objects": objects, "heard": word}
    if response_type == "choice":
        trial["expect"] = {"choice": target}
    elif response_type == "rating":
        points = options["scale_points"]
        trial.update(probe=target, expect={"min_rating": (points + 1) // 2 + 1})
    elif response_type == "slider":
        top = options["total_points"]
        trial.update(probe=target, expect={"min_value": top // 2 + 1})
    elif response_type == "bet":
        top = options["total_points"]
        trial["expect"] = {"object": target, "min_points": top // 2 + 1}
    elif response_type == "production":
        true_words = [f for f, v in enumerate(scene[target]) if v]
        trial.update(target=target, expect={"utterance_in": true_words})
    elif response_type == "binary_judgment":
        if k % 2 == 1:  # the word is false of every other object
            others = [i for i in range(len(objects)) if i != target]
            trial.update(target=others[int(rng.integers(len(others)))], utterance=word, expect={"answer": False})
        else:
            trial.update(target=target, utterance=word, expect={"answer": True})
    else:
        raise KeyError(response_type)
    return trial


def catch_passed(row: Mapping[str, Any]) -> bool:
    """Whether a catch or practice trial's response meets its rule."""
    e, r = row["expect"], row["response"]
    if "choice" in e:
        return r["choice"] == e["choice"]
    if "min_rating" in e:
        return r["rating"] >= e["min_rating"]
    if "min_value" in e:
        return r["value"] >= e["min_value"]
    if "min_points" in e:
        return r["points"][e["object"]] >= e["min_points"]
    if "utterance_in" in e:
        return r["utterance"] in e["utterance_in"]
    if "answer" in e:
        return r["answer"] is e["answer"]
    raise KeyError(f"unknown catch rule {e}")


CATCH_RULES = {
    "choice": "click the one object the word is true of",
    "rating": "rate that object above the scale's middle",
    "slider": "put the slider above its middle for that object",
    "bet": "give that object more than half the points",
    "production": "pick a word that is true of the marked object",
    "binary_judgment": "answer yes for the word true of only the marked object, no when it is false of it",
}


# --------------------------------------------------------------------------- rendering

def _object_html(paradigm, domain, feats: Sequence[int], label: str) -> str:
    if paradigm.materials.kind == "svg":
        parts = [domain.base] + [domain.features[f].svg for f in feats]
        return (f'<svg class="ro-svg" viewBox="0 0 100 100" role="img" aria-label="{html.escape(label)}" '
                f'xmlns="http://www.w3.org/2000/svg">{"".join(parts)}</svg>')
    return f'<span class="ro-text">{html.escape(label)}</span>'


def _object_label(paradigm, domain, feats: Sequence[int]) -> str:
    """The object in words: a text material's phrase, an SVG's accessible name."""
    if paradigm.materials.kind == "text":
        phrases = [domain.features[f].phrase for f in feats]
        return " ".join([domain.base, " and ".join(phrases)]).strip()
    words = [domain.features[f].word for f in feats]
    return f"a {domain.name} item" + (f" with {', '.join(words)}" if words else " with nothing extra")


def _word_html(word: str) -> str:
    return f'<span class="ro-word">“{html.escape(word)}”</span>'


def _prompt_html(paradigm, trial: Mapping[str, Any], words: Sequence[str]) -> str:
    r = paradigm.response
    if r.type == "production":
        text = html.escape(r.prompt)
    elif r.type == "binary_judgment":
        text = html.escape(r.prompt).replace(html.escape("{message}"), _word_html(words[trial["utterance"]]))
    elif trial["heard"] is None:
        text = html.escape(r.prior_prompt)
    else:
        text = html.escape(r.prompt).replace(html.escape("{message}"), _word_html(words[trial["heard"]]))
    return f'<p class="ro-prompt">{text}</p>'


def _widget(paradigm, trial: Mapping[str, Any], position_of: Mapping[int, int], words: Sequence[str],
            rng: np.random.Generator) -> Dict[str, Any]:
    r = paradigm.response
    kind = r.type
    w: Dict[str, Any] = {"type": kind}
    if kind in ("rating", "slider"):
        w.update(marked=position_of[trial["probe"]], labels=list(r.scale_labels))
        w["points" if kind == "rating" else "max"] = r.scale_points if kind == "rating" else r.total_points
    elif kind == "bet":
        w["total"] = r.total_points
    elif kind == "production":
        n_feat = len(trial["objects"][0])
        order = [int(f) for f in rng.permutation(n_feat)]
        w.update(marked=position_of[trial["target"]],
                 options=[{"label": words[f], "utterance": f} for f in order])
    elif kind == "binary_judgment":
        w.update(marked=position_of[trial["target"]], options=["Yes", "No"])
    return w


def _rotate_domain(paradigm, n_features: int, counts: List[int], previous: Optional[int]) -> int:
    eligible = [i for i, d in enumerate(paradigm.materials.domains) if len(d.features) >= n_features]
    if not eligible:
        raise ValueError(f"no domain has {n_features} features")
    if previous in eligible and len(eligible) > 1:
        eligible.remove(previous)
    return min(eligible, key=lambda i: (counts[i], i))


def _concrete(paradigm, trial: Dict[str, Any], rng: np.random.Generator, counts: List[int],
              previous: Optional[int]) -> Dict[str, Any]:
    n_obj, n_feat = len(trial["objects"]), len(trial["objects"][0])
    d = _rotate_domain(paradigm, n_feat, counts, previous)
    counts[d] += 1
    domain = paradigm.materials.domains[d]
    feature_map = [int(f) for f in rng.permutation(len(domain.features))[:n_feat]]
    words = [domain.features[f].word for f in feature_map]
    display_order = [int(i) for i in rng.permutation(n_obj)]
    position_of = {obj: p for p, obj in enumerate(display_order)}
    items, labels = [], []
    for obj in display_order:
        feats = [feature_map[j] for j, v in enumerate(trial["objects"][obj]) if v]
        label = _object_label(paradigm, domain, feats)
        labels.append(label)
        items.append(_object_html(paradigm, domain, feats, label))
    return {**trial, "domain": d, "domain_name": domain.name, "feature_map": feature_map, "words": words,
            "display_order": display_order, "items_html": items, "item_labels": labels,
            "prompt_html": _prompt_html(paradigm, trial, words),
            "widget": _widget(paradigm, trial, position_of, words, rng)}


# --------------------------------------------------------------------------- lists

def _options(paradigm) -> Dict[str, int]:
    r = paradigm.response
    return {k: v for k, v in (("scale_points", r.scale_points), ("total_points", r.total_points)) if v is not None}


def _abstract_test_trial(paradigm, unit: Mapping[str, Any], rng: np.random.Generator) -> Dict[str, Any]:
    trial = {"objects": unit["objects"], "heard": unit["heard"], **{k: unit[k] for k in UNIT_FIELDS if k in unit}}
    missing = [k for k in _needed_fields(paradigm.response.type) if k not in trial]
    if missing:
        raise ValueError(f"a {paradigm.response.type} design unit needs {missing}: {dict(unit)}")
    return trial


def _needed_fields(response_type: str) -> Tuple[str, ...]:
    return {"rating": ("probe",), "slider": ("probe",), "production": ("target",),
            "binary_judgment": ("target", "utterance")}.get(response_type, ())


def trial_list(paradigm, units: Sequence[Mapping[str, Any]], seed: int, index: int) -> Dict[str, Any]:
    """List ``index``: one participant's practice, test and catch trials, fully concrete."""
    d = paradigm.design
    order = np.random.default_rng([seed, SUBSET_STREAM]).permutation(len(units))
    T = d.trials_per_participant
    chosen = [int(order[(index * T + k) % len(units)]) for k in range(T)]
    rng = np.random.default_rng([seed, index])
    rng.shuffle(chosen)
    kind = paradigm.response.type
    options = _options(paradigm)
    tests = [{"phase": "test", "unit": u, **_abstract_test_trial(paradigm, units[u], rng)} for u in chosen]
    shapes = [(len(t["objects"]), len(t["objects"][0])) for t in tests]
    catches = [{"phase": "catch", "unit": None,
                **_catch(shapes[int(rng.integers(len(shapes)))], rng, kind, k, options)}
               for k in range(d.catch_trials)]
    sequence = list(tests)
    for c in catches:
        sequence.insert(int(rng.integers(len(sequence) + 1)), c)
    practice = {"phase": "practice", "unit": None, **_catch(PRACTICE_SHAPE, rng, kind, 0, options)}
    counts = [0] * len(paradigm.materials.domains)
    out, previous = [], None
    for i, trial in enumerate([practice] + sequence):
        concrete = _concrete(paradigm, trial, rng, counts, previous)
        previous = concrete["domain"]
        concrete["trial_number"] = i
        out.append(concrete)
    return {"list_index": index, "seed": [seed, index], "trials": out}


def trial_lists(paradigm, units: Sequence[Mapping[str, Any]], seed: int,
                n_lists: Optional[int] = None) -> Dict[str, Any]:
    """Every participant's list (``n_participants`` by default), with the intro screens."""
    n = paradigm.design.n_participants if n_lists is None else n_lists
    if len(units) != paradigm.design.distinct_trials:
        raise ValueError(f"the design has {len(units)} units; the paradigm says distinct_trials = "
                         f"{paradigm.design.distinct_trials}")
    ins = paradigm.instructions
    return {
        "schema": SCHEMA, "paradigm_sha256": paradigm.sha256, "paradigm_name": paradigm.name,
        "design_sha256": design_sha256(units), "seed": seed, "response_type": paradigm.response.type,
        "minutes": paradigm.design.minutes, "materials_kind": paradigm.materials.kind,
        "intro": {"welcome": ins.welcome, "task": ins.task,
                  "checks": [c.model_dump() for c in ins.comprehension_checks]},
        "lists": [trial_list(paradigm, units, seed, i) for i in range(n)],
    }


# --------------------------------------------------------------------------- data back

def scenes_and_responses(rows: Sequence[Mapping[str, Any]]) -> Tuple[List[Context], List[dict]]:
    """The test trials' scenes and responses, for `src.rsa_open.fit.OpenData.build`."""
    tests = [r for r in rows if r.get("phase") == "test"]
    return [scene_of(r) for r in tests], [dict(r["response"]) for r in tests]


def page_rows(data: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """The trial rows of one participant's jsPsych data (the page's ``task == 'rsa_open_trial'``)."""
    rows = [dict(r) for r in data if r.get("task") == "rsa_open_trial"]
    for r in rows:
        if r.get("phase") not in PHASES:
            raise ValueError(f"trial {r.get('trial_number')}: unknown phase {r.get('phase')!r}")
        if not isinstance(r.get("response"), dict):
            raise ValueError(f"trial {r.get('trial_number')}: no response recorded")
    return rows
