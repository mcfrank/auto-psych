"""Abstract reference-game contexts -> concrete per-participant trial lists.

A design is a list of `TrialSpec`s: an object x feature 0/1 matrix (rows are
objects, columns abstract features), the column the speaker said (or None:
a prior query, Bob mumbles) and optionally a message set and role labels.
`trial_list` turns it into what one participant sees:

* **item domain** — each trial is drawn as one of the pragmods items
  (`DOMAINS`). Domains rotate across a participant's trials (least-used
  eligible domain first, never the previous trial's when another is
  eligible, ties broken by a per-participant shuffled order). A domain is
  eligible only when it has at least as many features as the context has
  columns: 4-feature contexts go to friend, snowman or sundae, 5-feature
  ones to sundae, and a context wider than every domain raises.
* **feature words** — the abstract columns are mapped to distinct features
  of the domain, at random (an injective map, a bijection onto the features
  used), so which word plays which role varies across trials and people.
* **screen order** — the objects' left-to-right order is a random
  permutation (`display_order[p]` is the object at position p, the schema's
  convention).
* **bases** — each object is drawn on one of the item's three base images
  (pragmods drew position i on base i+1); the bases of a trial are distinct
  up to three objects, and a fourth object reuses one.
* **catch trials** — `n_catch` unambiguous trials are inserted at random
  positions: a game of the shape (objects x features) of a random design
  trial, from `src.rsa.design_space.games` (with redundant features only when
  the shape has no other game), with a word true of exactly one
  object (the `catch_target`). They are labelled `condition="catch"`.
* **practice** — one unambiguous trial (same rule as a catch trial, shape
  `PRACTICE_SHAPE`) before the test trials, recorded as `phase="practice"`.

**Randomisation happens here, in Python, not in the page.** A list is a
pure function of (design, seed, list index) (`numpy.random.default_rng([seed,
list_index])`), so it is reproducible and testable without a browser, and
the page needs no RNG of its own. `build` bakes `n_lists` lists into the page,
which runs the one named by its `?list=<k>` URL parameter or, without one, a
uniformly random one, and records the list index on every trial. Every
displayed detail (words, screen order, bases, images) is also recorded on
each trial, so converting the data never needs the list file.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import tyro

from src.rsa.context import Context
from src.rsa.design_space import feature_names as abstract_feature_names
from src.rsa.design_space import games
from src.runtime.config import PROJECT_ASSETS_DIR

EXPERIMENT_ASSETS_DIR = PROJECT_ASSETS_DIR / "rsa_reference" / "experiment"
IMAGES_DIR = EXPERIMENT_ASSETS_DIR / "images"

TRIAL_LISTS_SCHEMA = "rsa_trial_lists/1"
SUBSET_STREAM = 2**31 - 1  # the seed stream of the design's shuffled order for balanced subsets
N_BASES = 3
PRACTICE_SHAPE = (3, 2)


@dataclass(frozen=True)
class Feature:
    word: str  # what Bob says
    image: str  # overlay file in IMAGES_DIR
    phrase: str  # for alt text: "friend with a hat and glasses"


@dataclass(frozen=True)
class Domain:
    name: str  # the schema's `item`
    plural: str
    slug: str  # image file prefix
    features: Tuple[Feature, ...]

    def base_image(self, base: int) -> str:
        if not 1 <= base <= N_BASES:
            raise ValueError(f"base {base} is not 1..{N_BASES}")
        return f"{self.slug}-base{base}.png"


def _domain(name: str, plural: str, features: Sequence[Tuple[str, str, str]]) -> Domain:
    slug = name.lower().replace(" ", "_")
    return Domain(
        name=name,
        plural=plural,
        slug=slug,
        features=tuple(
            Feature(word=word, image=f"{slug}-{part.replace(' ', '_')}.png", phrase=phrase)
            for word, part, phrase in features
        ),
    )


# Words are pragmods' one-word forms (`stims_single_words` in
# pragmods_parameter_setter_c1.js); bowtie, belt, banana and sprinkles are the
# size experiment's fourth/fifth features.
DOMAINS: Dict[str, Domain] = {
    d.name: d
    for d in (
        _domain(
            "friend",
            "friends",
            [
                ("hat", "hat", "a hat"),
                ("glasses", "glasses", "glasses"),
                ("mustache", "mustache", "a mustache"),
                ("bowtie", "bowtie", "a bow tie"),
            ],
        ),
        _domain(
            "snowman",
            "snowmen",
            [
                ("hat", "hat", "a hat"),
                ("scarf", "scarf", "a scarf"),
                ("mittens", "mittens", "mittens"),
                ("belt", "belt", "a belt"),
            ],
        ),
        _domain(
            "sundae",
            "sundaes",
            [
                ("cherry", "cherry", "a cherry"),
                ("whipped-cream", "whipped cream", "whipped cream"),
                ("chocolate", "chocolate", "chocolate sauce"),
                ("banana", "banana", "banana slices"),
                ("sprinkles", "sprinkles", "sprinkles"),
            ],
        ),
        _domain(
            "pizza",
            "pizzas",
            [
                ("mushrooms", "mushrooms", "mushrooms"),
                ("olives", "olives", "olives"),
                ("peppers", "peppers", "peppers"),
            ],
        ),
        _domain(
            "boat",
            "boats",
            [("cabin", "cabin", "a cabin"), ("sail", "sail", "a sail"), ("motor", "motor", "a motor")],
        ),
        _domain(
            "Christmas tree",
            "Christmas trees",
            [("lights", "lights", "lights"), ("ornaments", "ornaments", "ornaments"), ("star", "star", "a star")],
        ),
    )
}


def all_images() -> List[str]:
    """Every image file a trial can use (bases and overlays of every domain)."""
    files: List[str] = []
    for d in DOMAINS.values():
        files += [d.base_image(b) for b in range(1, N_BASES + 1)]
        files += [f.image for f in d.features]
    return files


@dataclass(frozen=True)
class TrialSpec:
    """One abstract trial of a design. ``utterance`` is a column index (None: prior query)."""

    objects: Tuple[Tuple[int, ...], ...]
    utterance: Optional[int]
    label: str
    roles: Optional[Tuple[str, ...]] = None
    messages: Optional[Tuple[int, ...]] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "objects", tuple(tuple(int(v) for v in r) for r in self.objects))
        if self.roles is not None:
            object.__setattr__(self, "roles", tuple(str(r) for r in self.roles))
            if len(self.roles) != len(self.objects):
                raise ValueError(f"{self.label}: roles {self.roles} need one per object")
        if self.messages is not None:
            object.__setattr__(self, "messages", tuple(int(m) for m in self.messages))
        if not self.label:
            raise ValueError("a trial spec needs a label (its condition)")
        self.context()  # validates the matrix, the word and the message set

    @property
    def shape(self) -> Tuple[int, int]:
        return (len(self.objects), len(self.objects[0]))

    def context(self, names: Optional[Sequence[str]] = None) -> Context:
        names = tuple(names) if names is not None else abstract_feature_names(self.shape[1])
        return Context(objects=self.objects, feature_names=names, utterance=self.utterance, messages=self.messages)

    @classmethod
    def from_context(cls, ctx: Context, label: str, roles: Optional[Sequence[str]] = None) -> "TrialSpec":
        return cls(objects=ctx.objects, utterance=ctx.utterance, label=label, roles=roles, messages=ctx.messages)

    @classmethod
    def from_json(cls, d: dict) -> "TrialSpec":
        unknown = set(d) - {"objects", "utterance", "label", "roles", "messages"}
        if unknown:
            raise ValueError(f"unknown trial spec keys {sorted(unknown)}")
        return cls(
            objects=d["objects"],
            utterance=d["utterance"],
            label=d["label"],
            roles=d.get("roles"),
            messages=d.get("messages"),
        )

    def to_json(self) -> dict:
        return {
            "label": self.label,
            "objects": [list(r) for r in self.objects],
            "utterance": self.utterance,
            "roles": None if self.roles is None else list(self.roles),
            "messages": None if self.messages is None else list(self.messages),
        }


@dataclass(frozen=True)
class Design:
    name: str
    specs: Tuple[TrialSpec, ...]

    def __post_init__(self) -> None:
        if not self.specs:
            raise ValueError(f"design {self.name!r} has no trials")
        reserved = sorted({s.label for s in self.specs} & {"catch", "practice"})
        if reserved:
            raise ValueError(f"design {self.name!r} uses reserved condition labels {reserved}")
        widest = max(s.shape[1] for s in self.specs)
        if widest > max(len(d.features) for d in DOMAINS.values()):
            raise ValueError(f"design {self.name!r} has a {widest}-feature context; no domain has that many")

    def to_json(self) -> dict:
        return {"name": self.name, "trials": [s.to_json() for s in self.specs]}

    @property
    def sha256(self) -> str:
        blob = json.dumps(self.to_json(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()

    @classmethod
    def from_json(cls, d: dict) -> "Design":
        unknown = set(d) - {"name", "trials"}
        if unknown:
            raise ValueError(f"unknown design keys {sorted(unknown)}")
        return cls(name=d["name"], specs=tuple(TrialSpec.from_json(t) for t in d["trials"]))

    @classmethod
    def load(cls, path: Path) -> "Design":
        return cls.from_json(json.loads(Path(path).read_text(encoding="utf-8")))


def _alt_text(domain: Domain, feats: Sequence[Feature]) -> str:
    if not feats:
        return f"a {domain.name} with nothing extra"
    phrases = [f.phrase for f in feats]
    joined = phrases[0] if len(phrases) == 1 else ", ".join(phrases[:-1]) + " and " + phrases[-1]
    return f"a {domain.name} with {joined}"


def _rotate_domain(n_features: int, order: Sequence[str], counts: Dict[str, int], previous: Optional[str]) -> str:
    eligible = [name for name in order if len(DOMAINS[name].features) >= n_features]
    if not eligible:
        raise ValueError(f"no domain has {n_features} features")
    if previous in eligible and len(eligible) > 1:
        eligible.remove(previous)
    return min(eligible, key=lambda name: (counts[name], order.index(name)))


def _catch_spec(shape: Tuple[int, int], rng: np.random.Generator) -> Tuple[TrialSpec, int]:
    """An unambiguous trial of this shape: a word true of exactly one object.

    From the games without redundant features when the shape has one (as every
    list built before 2026-10-10), else with them (two objects, four features)."""
    n_obj, n_feat = shape
    candidates = []
    for synonyms in (False, True):
        candidates = [
            (matrix, word)
            for matrix in games(n_obj, n_feat, synonyms)
            for word in range(n_feat)
            if sum(row[word] for row in matrix) == 1
        ]
        if candidates:
            break
    if not candidates:
        raise ValueError(f"no {n_obj}x{n_feat} game has a word true of exactly one object")
    matrix, word = candidates[int(rng.integers(len(candidates)))]
    target = next(i for i, row in enumerate(matrix) if row[word])
    roles = tuple("target" if i == target else "other" for i in range(n_obj))
    return TrialSpec(objects=matrix, utterance=word, label="catch", roles=roles), target


def _concrete(
    spec: TrialSpec,
    domain: Domain,
    rng: np.random.Generator,
    *,
    phase: str,
    condition: str,
    spec_index: Optional[int],
    catch_target: Optional[int],
) -> dict:
    n_obj, n_feat = spec.shape
    picked = rng.permutation(len(domain.features))[:n_feat]
    feats = [domain.features[int(i)] for i in picked]
    display_order = [int(i) for i in rng.permutation(n_obj)]
    base_perm = rng.permutation(N_BASES) + 1
    bases = [int(base_perm[i % N_BASES]) for i in range(n_obj)]
    screen = []
    for obj in display_order:
        on = [feats[c] for c in range(n_feat) if spec.objects[obj][c]]
        screen.append(
            {
                "object": obj,
                "images": [domain.base_image(bases[obj])] + [f.image for f in on],
                "alt": _alt_text(domain, on),
            }
        )
    return {
        "phase": phase,
        "condition": condition,
        "is_catch": condition == "catch",
        "catch_target": catch_target,
        "spec_index": spec_index,
        "item": domain.name,
        "item_plural": domain.plural,
        "feature_names": [f.word for f in feats],
        "objects": [list(r) for r in spec.objects],
        "roles": list(spec.roles) if spec.roles is not None else [""] * n_obj,
        "utterance": spec.utterance,
        "word": None if spec.utterance is None else feats[spec.utterance].word,
        "query": "prior" if spec.utterance is None else "utterance",
        "messages": None if spec.messages is None else list(spec.messages),
        "display_order": display_order,
        "bases": bases,
        "screen": screen,
    }


def trial_list(
    design: Design,
    *,
    seed: int,
    list_index: int,
    n_catch: int,
    n_trials: Optional[int] = None,
    shuffle: bool = True,
) -> dict:
    """One participant's trials: a practice trial, then the test trials in order."""
    if n_catch < 0:
        raise ValueError(f"n_catch must be >= 0: {n_catch}")
    rng = np.random.default_rng([seed, list_index])
    indices = list(range(len(design.specs)))
    if n_trials is not None:
        if not 1 <= n_trials <= len(indices):
            raise ValueError(f"n_trials {n_trials} is not 1..{len(indices)}")
        # Balanced subsets: one shuffled order of the design per seed, and list k
        # takes the next n_trials of it (cyclically), so every display is seen by
        # the same number of lists over each D / gcd(D, n_trials) consecutive lists.
        order = np.random.default_rng([seed, SUBSET_STREAM]).permutation(len(indices))
        start = list_index * n_trials
        indices = sorted(int(order[(start + j) % len(indices)]) for j in range(n_trials))
    if shuffle:
        indices = [indices[int(i)] for i in rng.permutation(len(indices))]
    # (spec, spec_index or None, catch_target or None), in presentation order
    entries: List[Tuple[TrialSpec, Optional[int], Optional[int]]] = [(design.specs[i], i, None) for i in indices]
    for _ in range(n_catch):
        shape = design.specs[indices[int(rng.integers(len(indices)))]].shape
        spec, target = _catch_spec(shape, rng)
        entries.insert(int(rng.integers(len(entries) + 1)), (spec, None, target))

    order = [str(n) for n in rng.permutation(sorted(DOMAINS))]
    counts = {name: 0 for name in DOMAINS}
    practice_spec, practice_target = _catch_spec(PRACTICE_SHAPE, rng)
    practice_domain = order[int(rng.integers(len(order)))]
    trials = [
        _concrete(
            practice_spec,
            DOMAINS[practice_domain],
            rng,
            phase="practice",
            condition="practice",
            spec_index=None,
            catch_target=practice_target,
        )
    ]
    previous: Optional[str] = practice_domain
    for spec, spec_index, target in entries:
        name = _rotate_domain(spec.shape[1], order, counts, previous)
        counts[name] += 1
        previous = name
        trials.append(
            _concrete(
                spec,
                DOMAINS[name],
                rng,
                phase="test",
                condition="catch" if target is not None else spec.label,
                spec_index=spec_index,
                catch_target=target,
            )
        )
    n_test = len(entries)
    for number, trial in enumerate(t for t in trials if t["phase"] == "test"):
        trial["trial_number"] = number
        trial["n_test_trials"] = n_test
    trials[0]["trial_number"] = 0
    trials[0]["n_test_trials"] = n_test
    out = {"list_index": list_index, "seed": seed, "trials": trials}
    check_trial_list(out)
    return out


def trial_lists(
    design: Design,
    *,
    seed: int,
    n_lists: int,
    n_catch: int,
    n_trials: Optional[int] = None,
    shuffle: bool = True,
) -> dict:
    """``n_lists`` participant lists as one JSON document (what `build` bakes in)."""
    if n_lists < 1:
        raise ValueError(f"n_lists must be >= 1: {n_lists}")
    return {
        "schema": TRIAL_LISTS_SCHEMA,
        "design_name": design.name,
        "design_sha256": design.sha256,
        "design": design.to_json(),
        "seed": seed,
        "n_catch": n_catch,
        "n_trials": n_trials,
        "shuffle": shuffle,
        "lists": [
            trial_list(design, seed=seed, list_index=k, n_catch=n_catch, n_trials=n_trials, shuffle=shuffle)
            for k in range(n_lists)
        ],
    }


def trial_context(trial: dict) -> Context:
    """The trial as a `Context` with its concrete words (raises when it is not a valid one)."""
    return Context(
        objects=tuple(tuple(r) for r in trial["objects"]),
        feature_names=tuple(trial["feature_names"]),
        utterance=trial["utterance"],
        item=trial["item"],
        messages=None if trial["messages"] is None else tuple(trial["messages"]),
    )


def check_trial(trial: dict) -> None:
    """Raise unless a concrete trial is internally consistent."""
    ctx = trial_context(trial)  # rectangular 0/1 matrix, word true of some object, message set
    domain = DOMAINS.get(trial["item"])
    if domain is None:
        raise ValueError(f"unknown item {trial['item']!r}")
    n_obj, n_feat = len(ctx.objects), len(ctx.feature_names)
    words = [f.word for f in domain.features]
    if n_feat > len(words):
        raise ValueError(f"{trial['item']} has {len(words)} features, the context {n_feat}")
    if len(set(trial["feature_names"])) != n_feat or not set(trial["feature_names"]) <= set(words):
        raise ValueError(f"feature words {trial['feature_names']} are not distinct {trial['item']} features")
    if sorted(trial["display_order"]) != list(range(n_obj)):
        raise ValueError(f"display_order {trial['display_order']} is not a permutation of the objects")
    if trial["query"] != ("prior" if ctx.utterance is None else "utterance"):
        raise ValueError(f"query {trial['query']!r} does not match utterance {ctx.utterance}")
    if trial["word"] != (None if ctx.utterance is None else ctx.feature_names[ctx.utterance]):
        raise ValueError(f"word {trial['word']!r} is not the uttered column's word")
    if len(trial["bases"]) != n_obj or len(trial["roles"]) != n_obj:
        raise ValueError("bases and roles need one entry per object")
    by_word = {f.word: f for f in domain.features}
    for pos, cell in enumerate(trial["screen"]):
        obj = trial["display_order"][pos]
        if cell["object"] != obj:
            raise ValueError(f"screen position {pos} shows object {cell['object']}, display_order says {obj}")
        expected = [domain.base_image(trial["bases"][obj])] + [
            by_word[trial["feature_names"][c]].image for c in range(n_feat) if ctx.objects[obj][c]
        ]
        if cell["images"] != expected:
            raise ValueError(f"screen position {pos} images {cell['images']} are not object {obj}'s {expected}")
    if trial["condition"] in ("catch", "practice"):
        target = trial["catch_target"]
        if ctx.utterance is None:
            raise ValueError(f"a {trial['condition']} trial needs a word")
        true_of = [i for i, row in enumerate(ctx.objects) if row[ctx.utterance]]
        if true_of != [target]:
            raise ValueError(f"{trial['condition']} word {trial['word']!r} is true of {true_of}, not only {target}")
    elif trial["catch_target"] is not None:
        raise ValueError("only catch and practice trials have a catch_target")


def check_trial_list(lst: dict) -> None:
    trials = lst["trials"]
    phases = [t["phase"] for t in trials]
    if phases[:1] != ["practice"] or set(phases[1:]) != {"test"}:
        raise ValueError(f"a list is one practice trial then test trials, got phases {phases}")
    numbers = [t["trial_number"] for t in trials[1:]]
    if numbers != list(range(len(numbers))):
        raise ValueError(f"test trial numbers {numbers} are not 0..{len(numbers) - 1}")
    for t in trials:
        check_trial(t)


def load_trial_lists(path: Path) -> dict:
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    if doc.get("schema") != TRIAL_LISTS_SCHEMA:
        raise ValueError(f"{path}: schema {doc.get('schema')!r} is not {TRIAL_LISTS_SCHEMA!r}")
    if not doc["lists"]:
        raise ValueError(f"{path}: no lists")
    for lst in doc["lists"]:
        check_trial_list(lst)
    return doc


@dataclass
class Args:
    """Write per-participant trial lists for a design (JSON, the input of `build`)."""

    design: Path
    """Design JSON: {"name": ..., "trials": [{"label", "objects", "utterance", "roles"?, "messages"?}]}."""
    out: Path
    """Where to write the trial lists JSON."""
    seed: int
    """Base seed; list k uses numpy default_rng([seed, k])."""
    n_lists: int
    """Number of participant lists to generate."""
    n_catch: int = 2
    """Catch trials (a word true of exactly one object) per list."""
    n_trials: Optional[int] = None
    """Design trials per list, drawn without replacement (default: all of them)."""
    shuffle: bool = True
    """Randomise the order of the design trials per list."""


def main(args: Args) -> None:
    doc = trial_lists(
        Design.load(args.design),
        seed=args.seed,
        n_lists=args.n_lists,
        n_catch=args.n_catch,
        n_trials=args.n_trials,
        shuffle=args.shuffle,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    n = len(doc["lists"][0]["trials"]) - 1
    print(f"wrote {len(doc['lists'])} lists of {n} test trials (+1 practice) to {args.out}")


if __name__ == "__main__":
    main(tyro.cli(Args))
