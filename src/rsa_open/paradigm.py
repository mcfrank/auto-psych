"""The paradigm spec the designer writes, and its validator (docs/auto_open/PLAN.md §3, §4, §8).

A paradigm is one JSON file: what the experiment asks (the charter question), its
instructions and comprehension checks, its materials (SVG or text only), the space of scenes
its trials are drawn from, the response type and the link that scores it, and the design (N,
trials, length, pay). Every trial maps onto an abstract scene, a `src.rsa.context.Context`:
referents with features and a heard message or none. Theories see only those scenes.

`load_paradigm` validates a spec and raises `ParadigmInvalid` listing **every** problem at once,
so an agent fixing a spec sees them all. It checks:
- structure and types, with unknown keys refused (a typo fails loudly);
- the design against the campaign's bounds: the closed campaign's budget per experiment
  (`BUDGET_USD`), at least `MIN_PARTICIPANTS`, at most `MAX_MINUTES` per person, and pay at the
  closed campaign's hourly rate (`MIN_HOURLY_USD`);
- the scene space: at most `MAX_OBJECTS` objects and `MAX_FEATURES` features, and at most
  `MAX_POOL` scenes (the EIG tools score every one);
- materials: every domain names enough features for the largest scene; SVG is well-formed and
  contains only drawing elements (no script, foreignObject, images, links, styles, event
  handlers, or references outside the fragment); text materials give each feature a phrase;
- content (PI 2026-10-20): no deception, no affectively charged stimuli or language, no URLs
  or email addresses;
- the response's link exists in the library (`src.rsa_open.links`) and scores its type.

A paradigm's `sha256` is that of its canonical JSON, so formatting does not change it and any
content does; it is recorded with every trial collected under the paradigm.

    python -m src.rsa_open.paradigm --spec paradigm.json
    python -m src.rsa_open.paradigm --schema-out schema.json   # the JSON schema, for the designer
"""

from __future__ import annotations

import hashlib
import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import tyro
from pydantic import BaseModel, ConfigDict, ValidationError

# The closed campaign's experiment: 200 people x $0.50 for 3 minutes, plus Prolific's fee of a
# third (PLAN.md, decisions of 2026-10-21: equal dollars per experiment).
PROLIFIC_FEE_MULTIPLIER = 4.0 / 3.0
BUDGET_USD = 200 * 0.50 * PROLIFIC_FEE_MULTIPLIER
MIN_HOURLY_USD = 0.50 / 3 * 60
MIN_PARTICIPANTS = 100
MAX_MINUTES = 15
MAX_OBJECTS = 6
MAX_FEATURES = 4
MAX_POOL = 6000
MAX_SVG_CHARS = 20_000
RESPONSE_TYPES = ("choice", "rating", "slider", "production", "binary_judgment")

SVG_NS = "http://www.w3.org/2000/svg"
SVG_ELEMENTS = {
    "svg", "g", "rect", "circle", "ellipse", "line", "polyline", "polygon", "path", "text",
    "tspan", "title", "desc", "defs", "linearGradient", "radialGradient", "stop", "clipPath", "use",
}
# A crude backstop to the PI's review of each chain's first paradigm, not a content classifier.
CHARGED = re.compile(
    r"\b(blood\w*|kill\w*|dead|death\w*|die[sd]?|dying|murder\w*|suicid\w*|abus\w*|violen\w*|"
    r"weapon\w*|guns?|knife|knives|sex\w*|nude|naked|drugs?|alcohol\w*|racis\w*|terror\w*|"
    r"tortur\w*|injur\w*|wound\w*|corpse\w*|hate\w*|disease\w*|cancer|vomit\w*|gore)\b",
    re.IGNORECASE,
)
URL = re.compile(r"https?://|www\.", re.IGNORECASE)
EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
WORD = re.compile(r"^[a-z][a-z' -]{0,30}$")


class ParadigmInvalid(ValueError):
    """A paradigm spec that cannot be used; the message lists every problem."""

    def __init__(self, path: str, problems: List[str]) -> None:
        self.problems = problems
        super().__init__(f"{path}: {len(problems)} problem(s):\n" + "\n".join(f"- {p}" for p in problems))


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Charter(_Strict):
    question: str
    """The question about pragmatic reference this paradigm asks (docs/auto_open/PLAN.md §2)."""


class ComprehensionCheck(_Strict):
    question: str
    options: List[str]
    correct: int


class Instructions(_Strict):
    welcome: str
    task: str
    comprehension_checks: List[ComprehensionCheck]


class Feature(_Strict):
    word: str
    svg: Optional[str] = None
    """An SVG fragment drawn over the domain's base (materials.kind 'svg')."""
    phrase: Optional[str] = None
    """How the feature is written (materials.kind 'text'), e.g. 'with a garden'."""


class Domain(_Strict):
    name: str
    base: str
    """The object without features: an SVG fragment, or a noun phrase for text."""
    features: List[Feature]


class Materials(_Strict):
    kind: Literal["svg", "text"]
    domains: List[Domain]


class SceneSpace(_Strict):
    sizes: List[List[int]]
    """[n_objects, n_features] pairs; every valid game of each size is in the pool."""
    prior_queries: bool
    """Include trials with no informative message (the listener's prior)."""
    synonyms: bool
    """Include games in which two features have the same extension."""


class Response(_Strict):
    type: Literal["choice", "rating", "slider", "production", "binary_judgment"]
    link: str
    prompt: str


class Design(_Strict):
    n_participants: int
    trials_per_participant: int
    catch_trials: int
    minutes: float
    pay_usd: float


class Content(_Strict):
    deception: bool
    affective_stimuli: bool


class Paradigm(_Strict):
    schema_version: Literal[1]
    name: str
    summary: str
    charter: Charter
    instructions: Instructions
    materials: Materials
    scene_space: SceneSpace
    response: Response
    design: Design
    content: Content

    @property
    def cost_usd(self) -> float:
        return self.design.n_participants * self.design.pay_usd * PROLIFIC_FEE_MULTIPLIER

    @cached_property
    def sha256(self) -> str:
        canonical = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"),
                               ensure_ascii=False)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def max_features(self) -> int:
        return max(f for _, f in self.scene_space.sizes)


def _svg_problems(fragment: str, where: str) -> List[str]:
    if len(fragment) > MAX_SVG_CHARS:
        return [f"{where}: SVG is {len(fragment)} characters, over the {MAX_SVG_CHARS} limit"]
    if re.search(r"<!(DOCTYPE|ENTITY)", fragment, re.IGNORECASE):
        return [f"{where}: SVG may not declare a DOCTYPE or entities"]
    try:
        root = ET.fromstring(f'<svg xmlns="{SVG_NS}">{fragment}</svg>')
    except ET.ParseError as err:
        return [f"{where}: SVG is not well-formed ({err})"]
    problems = []
    for el in root.iter():
        tag = el.tag.split("}")[-1]
        if tag not in SVG_ELEMENTS:
            problems.append(f"{where}: SVG element <{tag}> is not allowed (drawing elements only)")
        for attr, value in el.attrib.items():
            name = attr.split("}")[-1]
            if name.lower().startswith("on"):
                problems.append(f"{where}: SVG event handler attribute {name} is not allowed")
            if name == "href" and not value.startswith("#"):
                problems.append(f"{where}: SVG href {value!r} points outside the fragment")
            if "url(" in value.replace(" ", "") and not re.fullmatch(r"\s*url\(#[\w-]+\)\s*", value):
                problems.append(f"{where}: SVG attribute {name} references {value!r} outside the fragment")
    return problems


def _texts(p: Paradigm) -> List[tuple]:
    """Every piece of designer-written text a participant can see, with where it is."""
    out = [("name", p.name), ("summary", p.summary), ("charter.question", p.charter.question),
           ("instructions.welcome", p.instructions.welcome), ("instructions.task", p.instructions.task),
           ("response.prompt", p.response.prompt)]
    for i, check in enumerate(p.instructions.comprehension_checks):
        out.append((f"instructions.comprehension_checks[{i}].question", check.question))
        out += [(f"instructions.comprehension_checks[{i}].options[{j}]", o) for j, o in enumerate(check.options)]
    for d, domain in enumerate(p.materials.domains):
        out.append((f"materials.domains[{d}].name", domain.name))
        if p.materials.kind == "text":
            out.append((f"materials.domains[{d}].base", domain.base))
        for f, feature in enumerate(domain.features):
            out.append((f"materials.domains[{d}].features[{f}].word", feature.word))
            if feature.phrase:
                out.append((f"materials.domains[{d}].features[{f}].phrase", feature.phrase))
    return out


def _svg_texts(p: Paradigm) -> List[tuple]:
    """The text drawn inside SVG fragments (<text>, <tspan>, <title>)."""
    out = []
    if p.materials.kind != "svg":
        return out
    for d, domain in enumerate(p.materials.domains):
        fragments = [(f"materials.domains[{d}].base", domain.base)] + [
            (f"materials.domains[{d}].features[{f}].svg", feat.svg or "") for f, feat in enumerate(domain.features)]
        for where, fragment in fragments:
            try:
                root = ET.fromstring(f'<svg xmlns="{SVG_NS}">{fragment}</svg>')
            except ET.ParseError:
                continue
            text = " ".join(t for t in root.itertext() if t.strip())
            if text:
                out.append((where, text))
    return out


def _problems(p: Paradigm) -> List[str]:
    from src.rsa_open.links import LINKS

    problems: List[str] = []
    d = p.design
    if d.n_participants < MIN_PARTICIPANTS:
        problems.append(f"design.n_participants {d.n_participants} is below the minimum of {MIN_PARTICIPANTS}")
    if not 0 < d.minutes <= MAX_MINUTES:
        problems.append(f"design.minutes {d.minutes:g} is outside (0, {MAX_MINUTES}], the maximum per person")
    if d.minutes > 0 and d.pay_usd / d.minutes * 60 < MIN_HOURLY_USD - 1e-9:
        problems.append(f"design.pay_usd ${d.pay_usd:.2f} for {d.minutes:g} minutes is "
                        f"${d.pay_usd / d.minutes * 60:.2f} per hour, below ${MIN_HOURLY_USD:.2f} per hour")
    if p.cost_usd > BUDGET_USD + 0.01:
        problems.append(f"design costs ${p.cost_usd:.2f} (n_participants x pay_usd x "
                        f"{PROLIFIC_FEE_MULTIPLIER:.4g}), over the ${BUDGET_USD:.2f} budget per experiment")
    if d.trials_per_participant < 1:
        problems.append("design.trials_per_participant must be at least 1")
    if d.catch_trials < 1:
        problems.append("design.catch_trials must be at least 1 (attention is checked in every paradigm)")

    sizes_ok = True
    if not p.scene_space.sizes:
        problems.append("scene_space.sizes is empty")
        sizes_ok = False
    for size in p.scene_space.sizes:
        if len(size) != 2:
            problems.append(f"scene_space.sizes entry {size} must be [n_objects, n_features]")
            sizes_ok = False
            continue
        n_obj, n_feat = size
        if not 2 <= n_obj <= MAX_OBJECTS:
            problems.append(f"scene_space.sizes {size}: between 2 and {MAX_OBJECTS} objects")
            sizes_ok = False
        if not 2 <= n_feat <= MAX_FEATURES:
            problems.append(f"scene_space.sizes {size}: between 2 and {MAX_FEATURES} features")
            sizes_ok = False
    if sizes_ok:
        n_pool = len(scene_pool(p))
        if n_pool > MAX_POOL:
            problems.append(f"scene_space gives {n_pool} scenes, over the {MAX_POOL} the EIG tools score")
        elif n_pool == 0:
            problems.append("scene_space gives no scenes")

    kind = p.materials.kind
    if not p.materials.domains:
        problems.append("materials.domains is empty")
    for i, domain in enumerate(p.materials.domains):
        where = f"materials.domains[{i}] ({domain.name})"
        if sizes_ok and len(domain.features) < p.max_features():
            problems.append(f"{where} names {len(domain.features)} features; scenes need {p.max_features()} features")
        words = [f.word for f in domain.features]
        if len(set(words)) != len(words):
            problems.append(f"{where}: feature words must differ ({words})")
        for j, feature in enumerate(domain.features):
            fwhere = f"{where}.features[{j}] ({feature.word})"
            if not WORD.match(feature.word):
                problems.append(f"{fwhere}: a word is 1-30 lowercase letters, spaces, hyphens or apostrophes")
            if kind == "svg":
                if not feature.svg:
                    problems.append(f"{fwhere}: svg materials need an svg fragment per feature")
                else:
                    problems += _svg_problems(feature.svg, fwhere)
            elif not feature.phrase:
                problems.append(f"{fwhere}: text materials need a phrase per feature")
        if kind == "svg":
            problems += _svg_problems(domain.base, f"{where}.base")

    for where, text in _texts(p) + _svg_texts(p):
        if CHARGED.search(text):
            problems.append(f"{where}: affectively charged language ({CHARGED.search(text).group(0)!r})")
        if URL.search(text):
            problems.append(f"{where}: no URLs")
        if EMAIL.search(text):
            problems.append(f"{where}: no email addresses")

    if p.content.deception:
        problems.append("content.deception: deception is not allowed (PI 2026-10-20)")
    if p.content.affective_stimuli:
        problems.append("content.affective_stimuli: affectively charged stimuli are not allowed (PI 2026-10-20)")

    link = LINKS.get(p.response.link)
    if link is None:
        problems.append(f"response.link {p.response.link!r} is not in the link library ({', '.join(sorted(LINKS))})")
    elif link.response_type != p.response.type:
        problems.append(f"response.link {link.name!r} scores {link.response_type!r} responses, not {p.response.type!r}")
    if not p.response.prompt.strip():
        problems.append("response.prompt is empty")

    checks = p.instructions.comprehension_checks
    if not checks:
        problems.append("instructions.comprehension_checks: at least one is required")
    for i, check in enumerate(checks):
        if len(check.options) < 2:
            problems.append(f"instructions.comprehension_checks[{i}] needs at least two options")
        elif not 0 <= check.correct < len(check.options):
            problems.append(f"instructions.comprehension_checks[{i}].correct is not an option index")
    for field in ("welcome", "task"):
        if not getattr(p.instructions, field).strip():
            problems.append(f"instructions.{field} is empty")
    if len(p.charter.question.strip()) < 15:
        problems.append("charter.question must state the question about pragmatic reference the paradigm asks")
    return problems


def parse_paradigm(spec: Dict[str, Any], where: str = "paradigm") -> Paradigm:
    try:
        paradigm = Paradigm.model_validate(spec)
    except ValidationError as err:
        raise ParadigmInvalid(where, [
            f"{'.'.join(str(x) for x in e['loc']) or '(top level)'}: {e['msg']}" for e in err.errors()
        ]) from None
    problems = _problems(paradigm)
    if problems:
        raise ParadigmInvalid(where, problems)
    return paradigm


def load_paradigm(path: Path) -> Paradigm:
    path = Path(path)
    try:
        spec = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as err:
        raise ParadigmInvalid(str(path), [f"not valid JSON ({err})"]) from None
    return parse_paradigm(spec, str(path))


def scene_pool(paradigm: Paradigm) -> list:
    """The abstract scenes the paradigm's trials can show (`src.rsa.context.Context`)."""
    from src.rsa.design_space import context_pool

    return context_pool([tuple(s) for s in paradigm.scene_space.sizes],
                        include_prior_queries=paradigm.scene_space.prior_queries,
                        synonyms=paradigm.scene_space.synonyms)


def main_schema(out: Path) -> dict:
    schema = Paradigm.model_json_schema()
    Path(out).write_text(json.dumps(schema, indent=2))
    return schema


@dataclass
class Args:
    spec: Optional[Path] = None
    """A paradigm spec to validate."""
    schema_out: Optional[Path] = None
    """Write the spec's JSON schema here."""


def main(args: Args) -> None:
    if args.spec is None and args.schema_out is None:
        raise SystemExit("give --spec and/or --schema-out")
    if args.schema_out is not None:
        main_schema(args.schema_out)
        print(f"schema -> {args.schema_out}")
    if args.spec is not None:
        p = load_paradigm(args.spec)
        print(f"{args.spec}: valid. {len(scene_pool(p))} scenes; {p.response.type} via link {p.response.link!r}; "
              f"N={p.design.n_participants}, {p.design.minutes:g} min, ${p.cost_usd:.2f}; sha256 {p.sha256}")


if __name__ == "__main__":
    main(tyro.cli(Args))
