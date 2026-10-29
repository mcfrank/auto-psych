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
- the response's link exists in the library (`src.rsa_open.links`) and scores its type; a
  rating gives its scale points and a slider its total points.

A paradigm's `sha256` is that of its canonical JSON, so formatting does not change it and any
content does; it is recorded with every trial collected under the paradigm.

    python -m src.rsa_open.paradigm --spec paradigm.json
    python -m src.rsa_open.paradigm --schema-out schema.json   # the JSON schema, for the designer
"""

from __future__ import annotations

import functools
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

# Twice the closed campaign's experiment (200 people x $0.50 for 3 minutes, plus Prolific's fee of
# a third), for every experiment of both arms (PLAN.md, decisions of 2026-10-21: equal dollars per
# experiment; 2026-10-26: twice the closed campaign's).
PROLIFIC_FEE_MULTIPLIER = 4.0 / 3.0
BUDGET_USD = 2 * 200 * 0.50 * PROLIFIC_FEE_MULTIPLIER
MIN_HOURLY_USD = 0.50 / 3 * 60
MIN_PARTICIPANTS = 100
MAX_MINUTES = 15
MAX_OBJECTS = 6
MAX_FEATURES = 4
MAX_POOL = 6000
MAX_SVG_CHARS = 20_000
MAX_SCALE_POINTS = 11
MAX_TOTAL_POINTS = 1000
RESPONSE_TYPES = ("choice", "rating", "slider", "bet", "production", "binary_judgment")
LISTENER_TYPES = ("choice", "rating", "slider", "bet")
MESSAGE = "{message}"
PLACEHOLDER = re.compile(r"\{[^{}]*\}")
QUOTED_MESSAGE = re.compile(r"[\"'“”‘’«»]\s*\{message\}|\{message\}\s*[\"'“”‘’«»]")

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
# Raster images are the harness's own curated stimuli only (the fixed arm's pragmods pictures,
# PI-approved; `src.rsa_open.fixed`): an SVG <image> must embed one of these files byte for byte. The
# files are not in the agents' tree, so a designer's spec can never carry one.
TRUSTED_IMAGES_DIR = Path(__file__).resolve().parents[1] / "pipelines/outer_loop/projects/rsa_reference/experiment/images"
PNG_DATA = re.compile(r"^data:image/png;base64,([A-Za-z0-9+/=]+)$")


@functools.lru_cache(maxsize=1)
def _trusted_image_shas() -> frozenset:
    if not TRUSTED_IMAGES_DIR.is_dir():
        return frozenset()
    return frozenset(hashlib.sha256(f.read_bytes()).hexdigest() for f in TRUSTED_IMAGES_DIR.glob("*.png"))


def _trusted_png(href: str) -> bool:
    import base64
    import binascii

    m = PNG_DATA.match(href)
    if not m:
        return False
    try:
        data = base64.b64decode(m.group(1), validate=True)
    except (binascii.Error, ValueError):
        return False
    return hashlib.sha256(data).hexdigest() in _trusted_image_shas()


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
    """An SVG fragment drawn over the domain's base, in the same 100 x 100 box (materials.kind
    'svg')."""
    phrase: Optional[str] = None
    """How the feature is written (materials.kind 'text'), e.g. 'with a garden'."""


class Domain(_Strict):
    name: str
    base: str
    """The object without features: an SVG fragment drawn in a 100 x 100 box (viewBox
    "0 0 100 100"), or a noun phrase for text."""
    features: List[Feature]
    variants: Optional[List[str]] = None
    """Alternative bases (SVG), one drawn per object in place of ``base``, distinct within a trial
    up to their number (the closed campaign drew each object on one of three base pictures, so
    objects with the same features are still different individuals)."""


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
    type: Literal["choice", "rating", "slider", "bet", "production", "binary_judgment"]
    link: str
    prompt: str
    """Shown on every trial. A listener trial (choice, rating, slider, bet) and a judgment show
    the message where the prompt says ``{message}`` (required), in quotation marks the page adds;
    a production trial shows none."""
    prior_prompt: Optional[str] = None
    """A listener trial with no informative message (scene_space.prior_queries): its prompt,
    without ``{message}``. Required exactly when such trials exist."""
    scale_labels: Optional[List[str]] = None
    """rating and slider: the labels of the scale's two ends, [low, high]."""
    scale_points: Optional[int] = None
    """rating: the number of points on the scale (2-11), frozen with the paradigm."""
    total_points: Optional[int] = None
    """slider: its maximum (a 0-max slider on one referent); bet: the points split over the
    referents. 2-1000, frozen with the paradigm."""


class Design(_Strict):
    n_participants: int
    trials_per_participant: int
    distinct_trials: int
    """D, the trial units the design picks (`src.rsa_open.design`); each participant answers a
    balanced subset, so each unit gets n_participants x trials_per_participant / D responses."""
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
    second_response: Optional[Response] = None
    """A second question about the same screen (`src.rsa_open.links`, "Two measures"): a choice or
    a rating of at most 7 points after a listener response, a yes/no judgment after a speaker one.
    Its prompt is shown once the first answer is given; its probed or judged object and message
    are design fields of their own."""
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

    @property
    def harness_materials(self) -> bool:
        """Whether the pictures are the harness's own curated images (the fixed arm's), which the
        legibility check, meant for a designer's drawings, does not test."""
        return self.materials.kind == "svg" and any("<image" in d.base for d in self.materials.domains)

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
        trusted_image = tag == "image" and any(a.split("}")[-1] == "href" and _trusted_png(v)
                                               for a, v in el.attrib.items())
        if tag not in SVG_ELEMENTS and not trusted_image:
            problems.append(f"{where}: SVG element <{tag}> is not allowed (drawing elements only)")
        for attr, value in el.attrib.items():
            name = attr.split("}")[-1]
            if name.lower().startswith("on"):
                problems.append(f"{where}: SVG event handler attribute {name} is not allowed")
            if name == "href" and trusted_image:
                continue
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
    for where, r in (("response", p.response), ("second_response", p.second_response)):
        if r is None:
            continue
        if where != "response":
            out.append((f"{where}.prompt", r.prompt))
        if r.prior_prompt:
            out.append((f"{where}.prior_prompt", r.prior_prompt))
        out += [(f"{where}.scale_labels[{i}]", x) for i, x in enumerate(r.scale_labels or [])]
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
    if d.distinct_trials < max(d.trials_per_participant, 1):
        problems.append(f"design.distinct_trials {d.distinct_trials} must be at least trials_per_participant "
                        f"({d.trials_per_participant})")
    elif d.n_participants * d.trials_per_participant < d.distinct_trials:
        problems.append("design.distinct_trials is more than the responses collected "
                        "(n_participants x trials_per_participant): some units would get none")
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
    if sizes_ok and p.materials.domains:
        # Each scene is drawn in a domain with enough features (`src.rsa_open.screens` rotates among
        # those), so every size needs one, and every domain must serve the narrowest scenes.
        widest = max(len(d.features) for d in p.materials.domains)
        if widest < p.max_features():
            problems.append(f"no domain names {p.max_features()} features; the widest scenes need them")
        narrowest = min(f for _, f in p.scene_space.sizes)
    for i, domain in enumerate(p.materials.domains):
        where = f"materials.domains[{i}] ({domain.name})"
        if sizes_ok and len(domain.features) < narrowest:
            problems.append(f"{where} names {len(domain.features)} features; every scene needs at least {narrowest}")
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
            for k, variant in enumerate(domain.variants or []):
                problems += _svg_problems(variant, f"{where}.variants[{k}]")
        elif domain.variants:
            problems.append(f"{where}: variants are SVG bases (materials.kind 'svg')")

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

    problems += _response_problems(p.response, p, "response")
    if p.second_response is not None:
        problems += _response_problems(p.second_response, p, "second_response")
        problems += _pair_problems(p)

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


def _response_problems(r: "Response", p: "Paradigm", where: str) -> List[str]:
    """A response block's problems (the first measure's, or the second's)."""
    from src.rsa_open.links import LINKS

    problems = []
    link = LINKS.get(r.link)
    if link is None:
        problems.append(f"{where}.link {r.link!r} is not in the link library ({', '.join(sorted(LINKS))})")
    elif link.response_type != r.type:
        problems.append(f"{where}.link {link.name!r} scores {link.response_type!r} responses, not {r.type!r}")
    if not r.prompt.strip():
        problems.append(f"{where}.prompt is empty")
    listener = r.type in LISTENER_TYPES
    for field in ("prompt", "prior_prompt"):
        text = getattr(r, field) or ""
        other = sorted(set(PLACEHOLDER.findall(text)) - {MESSAGE})
        if other:
            problems.append(f"{where}.{field}: unknown placeholders {other}; the only one is {MESSAGE}")
    wants_message = listener or r.type == "binary_judgment"
    if wants_message and r.prompt.count(MESSAGE) != 1:
        problems.append(f"{where}.prompt must show the message once, where it says {MESSAGE}")
    if QUOTED_MESSAGE.search(r.prompt):
        problems.append(f"{where}.prompt: no quotation marks around {MESSAGE}: the page shows the word in "
                        f"quotation marks itself")
    if not wants_message and MESSAGE in r.prompt:
        problems.append(f"{where}.prompt: a {r.type} trial has no message to show ({MESSAGE})")
    if listener and p.scene_space.prior_queries:
        if not (r.prior_prompt or "").strip():
            problems.append(f"{where}.prior_prompt is required: scene_space.prior_queries gives trials with "
                            "no informative message")
        elif MESSAGE in r.prior_prompt:
            problems.append(f"{where}.prior_prompt: those trials have no message to show ({MESSAGE})")
    elif r.prior_prompt is not None:
        problems.append(f"{where}.prior_prompt applies only to listener responses with scene_space.prior_queries")
    if r.type in ("rating", "slider"):
        if r.scale_labels is None or len(r.scale_labels) != 2 or not all(x.strip() for x in r.scale_labels):
            problems.append(f"{where}.scale_labels must be [low, high] for a {r.type} response")
    elif r.scale_labels is not None:
        problems.append(f"{where}.scale_labels applies only to rating and slider responses")
    if r.type == "rating":
        if r.scale_points is None or not 2 <= r.scale_points <= MAX_SCALE_POINTS:
            problems.append(f"{where}.scale_points must be 2-{MAX_SCALE_POINTS} for a rating response")
    elif r.scale_points is not None:
        problems.append(f"{where}.scale_points applies only to rating responses")
    if r.type in ("slider", "bet"):
        if r.total_points is None or not 2 <= r.total_points <= MAX_TOTAL_POINTS:
            problems.append(f"{where}.total_points must be 2-{MAX_TOTAL_POINTS} for a {r.type} response")
    elif r.total_points is not None:
        problems.append(f"{where}.total_points applies only to slider and bet responses")

    return problems


def _outcome_count(r: "Response", p: "Paradigm") -> int:
    """The most outcomes one trial of this measure can have (the design tool's slots)."""
    from src.rsa_open.links import bet_compositions, design_bet_total, slider_bin_edges

    objects = max(n for n, _ in p.scene_space.sizes)
    return {"choice": objects, "rating": r.scale_points or 0, "binary_judgment": 2,
            "production": max(f for _, f in p.scene_space.sizes),
            "slider": len(slider_bin_edges(r.total_points or 2)) + 1,
            "bet": max((len(bet_compositions(n, design_bet_total(n, r.total_points or 2)))
                        for n, _ in p.scene_space.sizes), default=0)}[r.type]


def _pair_problems(p: "Paradigm") -> List[str]:
    """A second measure's own rules (`src.rsa_open.links`, "Two measures")."""
    from src.rsa_open.links import MAX_PAIR_OUTCOMES, MAX_SECOND_POINTS, SECOND_TYPES

    first, second = p.response, p.second_response
    side = "listener" if first.type in LISTENER_TYPES else "speaker"
    problems = []
    if second.type not in SECOND_TYPES[side]:
        problems.append(f"second_response.type: after a {first.type} response the second is one of "
                        f"{', '.join(SECOND_TYPES[side])}, not {second.type}")
        return problems
    if second.type == "choice" and first.type == "choice":
        problems.append("second_response: a second choice on the same screen repeats the first")
    if second.type == "rating" and (second.scale_points or 0) > MAX_SECOND_POINTS:
        problems.append(f"second_response.scale_points: a second rating has at most {MAX_SECOND_POINTS} points")
    if not problems:
        joint = _outcome_count(first, p) * _outcome_count(second, p)
        if joint > MAX_PAIR_OUTCOMES:
            problems.append(f"the two measures have up to {joint} joint outcomes per trial, over the "
                            f"{MAX_PAIR_OUTCOMES} the EIG tool scores: make one of them coarser")
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


MIN_DOMAINS = 3
MAX_TEST_TRIALS = 30
"""A designed paradigm's floor on variety and cap on length (PI 2026-10-29). Run C′'s designer chose
one robot, a 10-point bet typed per object and 24 near-identical trials: a design scored on
information per dollar gains nothing from variety, and every extra domain is one more drawing the
legibility check can fail. The closed campaign showed 6 domains with 3 base pictures each in 13
trials. Checked on a designer's spec (`load_designed`), not on every spec the harness reads."""


def experience_problems(p: "Paradigm") -> List[str]:
    """What a designed paradigm owes its participants: varied materials, a bounded length."""
    problems = []
    if p.materials.kind == "svg" and len(p.materials.domains) < MIN_DOMAINS:
        problems.append(f"materials: {len(p.materials.domains)} domain(s); a picture paradigm needs at least "
                        f"{MIN_DOMAINS}, so that people see varied things (the page rotates domains over trials)")
    if p.design.trials_per_participant > MAX_TEST_TRIALS:
        problems.append(f"design.trials_per_participant is {p.design.trials_per_participant}; at most "
                        f"{MAX_TEST_TRIALS} test trials a person (spend the budget on more people instead)")
    return problems


def load_designed(path: Path) -> Paradigm:
    """`load_paradigm`, plus `experience_problems`: a spec a designer may submit."""
    p = load_paradigm(path)
    problems = experience_problems(p)
    if problems:
        raise ParadigmInvalid(str(path), problems)
    return p


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


def paradigm_likelihood(paradigm: Paradigm):
    """The paradigm's link likelihood, with its frozen options (`src.rsa_open.links`)."""
    from src.rsa_open.links import likelihood

    def part(r):
        return dict(name=r.link, **{k: v for k, v in (("scale_points", r.scale_points),
                                                      ("total_points", r.total_points)) if v is not None})

    if paradigm.second_response is None:
        return likelihood(**part(paradigm.response))
    return likelihood("pair", first=part(paradigm.response), second=part(paradigm.second_response))


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
