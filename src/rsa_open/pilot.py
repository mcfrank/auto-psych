"""The simulated pilot (docs/auto_open/PLAN.md §4): LLM participants take a paradigm's trial
lists screen by screen, and the pilot gate decides whether the paradigm may go live.

    uv run python -m src.rsa_open.pilot --paradigm p.json --lists lists.json --out pilot/

**Participants** are Gemini 3.8 Flash (PI 2026-10-21, `PILOT_MODEL`), one chat per participant.
They see what the page shows (`src.rsa_open.screens`): the welcome and instructions, each
comprehension check (a wrong answer shows the instructions once more, as the page does), the
practice trial and every trial of one list, in order. A trial with SVG materials is shown as one
image of its objects, left to right, the marked one in an orange frame (rendered with resvg);
text materials are written out. Each screen asks for a reply line ``ANSWER: ...`` in the form
its widget takes (an object number, a rating, a slider value, points per object, a word, yes or
no); an unreadable reply is asked once more, then recorded as unreadable.

**The gate** (`gate`): a paradigm passes when
- comprehension: at least ``FLOORS['comprehension']`` of participants answer every check right
  by their second reading of the instructions;
- attention: at least ``FLOORS['catch']`` of catch trials are passed (`screens.catch_passed`);
- readability: at least ``FLOORS['readable']`` of trials get a usable answer;
- the test responses parse as the paradigm's link's data (`src.rsa_open.fit.OpenData.build`);
- the time estimate (`estimated_minutes`: `src.rsa_open.timing`'s model, calibrated on people's
  real times in finished experiments) is within ``TIME_TOLERANCE`` of the declared minutes;
- the test answers are not all one answer;
- legibility (SVG materials, `materials_check`): ``LEGIBILITY_CHECKERS`` fresh chats name each
  domain's base freely (a name other than the domain's goes to a judge chat: "is 'box' a
  reasonable name for robots?"), pick which of the domain's words names the part a single
  feature adds, and say exactly which objects of ``TRUTH_ITEMS`` real test trials a word is true
  of; at least ``FLOORS['legibility']`` of answers are right, and no feature or trial is misread
  by half of them or more (PI 2026-10-21: the first designer drew a bare box for a featureless
  robot, and a multiple-choice naming question passed it by elimination). A base named otherwise
  by half or more counts against the rate but alone is a warning, not a failure (PI 2026-10-21:
  a plain badge is fairly called a circle; the features and the words' truth carry the
  inference).
LLM participants are not people: the pilot finds instructions, prompts and materials that a
capable reader cannot follow, and data that do not parse; it says nothing about effect sizes.
"""

from __future__ import annotations

import html
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Protocol, Sequence, Tuple

import tyro

PILOT_MODEL = "gemini-3.8-flash"
PILOT_PARTICIPANTS = 12
FLOORS = {"comprehension": 0.8, "catch": 0.8, "readable": 0.95, "legibility": 0.9}
LEGIBILITY_CHECKERS = 3
CHOICE_ASK = "Reply ANSWER: <the number of your choice>."
NAME_ASK = "Reply ANSWER: <your name for it>."
TRUTH_ASK = "Reply ANSWER: <their numbers, separated by commas>, or ANSWER: none."
TRUTH_ITEMS = 6
TIME_TOLERANCE = 0.30
SYSTEM = ("You are taking part in an online research study. Read each screen carefully and respond as a "
          "thoughtful participant would. End every reply with one line of the form `ANSWER: ...`, in the "
          "format the screen asks for.")
IMAGE_DROPPED = "(The picture shown with this screen is no longer shown.)"
ANSWER = re.compile(r"ANSWER:\s*(.+)", re.IGNORECASE)
FRAME, MARK = "#d7dbe1", "#d9822b"


class Participant(Protocol):
    name: str

    def reply(self, text: str, images: Sequence[bytes] = ()) -> str:
        """Send one screen (text and images) in this participant's conversation; their reply."""


class GeminiParticipant:
    """One conversation with a Gemini model; usage is recorded per call (`src.runtime.token_usage`).

    Only the current screen's picture is sent: an earlier turn keeps its words and the reply,
    with ``IMAGE_DROPPED`` where its picture was. A chat that resent every earlier picture made
    each call's input grow with the trial count (a 3-person pilot: 4.1M input tokens for 195
    calls); a person, too, remembers the task, not every earlier screen."""

    def __init__(self, model: str = PILOT_MODEL, temperature: float = 1.0, retries: int = 3, client=None):
        from google.genai import types

        self.name, self._types, self.retries = model, types, retries
        if client is None:
            from google import genai

            from src.pipelines.outer_loop.llm import resolve_google_api_key

            client = genai.Client(api_key=resolve_google_api_key(require=True))
        # Held: a client nothing refers to is closed when collected, under its calls.
        self._client = client
        self._config = types.GenerateContentConfig(system_instruction=SYSTEM, temperature=temperature)
        self._history: List[Any] = []

    def reply(self, text: str, images: Sequence[bytes] = ()) -> str:
        from src.runtime.token_usage import record_usage

        t = self._types
        pictures = [t.Part.from_bytes(data=png, mime_type="image/png") for png in images]
        turn = t.Content(role="user", parts=pictures + [t.Part.from_text(text=text)])
        for attempt in range(self.retries):
            try:
                response = self._client.models.generate_content(model=self.name, contents=self._history + [turn],
                                                                config=self._config)
                break
            except Exception as err:  # an API error is the run's, not the paradigm's
                if attempt == self.retries - 1:
                    raise RuntimeError(f"{self.name}: the pilot participant's API call failed: {err}") from err
                time.sleep(2 ** (attempt + 1))
        usage = getattr(response, "usage_metadata", None)
        if usage is None:
            record_usage(source="rsa_open:pilot", backend="google-genai", model=self.name, usage_missing=True)
        else:
            record_usage(source="rsa_open:pilot", backend="google-genai", model=self.name,
                         input_tokens=int(usage.prompt_token_count or 0),
                         output_tokens=int(usage.candidates_token_count or 0),
                         reasoning_tokens=int(getattr(usage, "thoughts_token_count", 0) or 0))
        answer = response.text or ""
        kept = [t.Part.from_text(text=text)] + ([t.Part.from_text(text=IMAGE_DROPPED)] if images else [])
        self._history += [t.Content(role="user", parts=kept), t.Content(role="model", parts=[t.Part.from_text(text=answer)])]
        return answer


# --------------------------------------------------------------------------- screens as text

def _plain(fragment: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", fragment)).strip()


def screen_png(trial: Mapping[str, Any]) -> bytes:
    """One image of an SVG trial's objects, left to right, the marked one framed in orange."""
    return row_png(trial["items_html"], trial["widget"].get("marked"))


def row_png(items_html: Sequence[str], marked: Optional[int] = None) -> bytes:
    """SVG objects drawn in a row of cards, as the page shows them."""
    import resvg_py

    cells = []
    for p, item in enumerate(items_html):
        inner = re.sub(r"^<svg[^>]*>", "", item)
        inner = inner[: inner.rfind("</svg>")]
        x = 10 + p * 120
        stroke, width = (MARK, 6) if p == marked else (FRAME, 3)
        cells.append(f'<rect x="{x - 4}" y="6" width="108" height="108" rx="10" fill="#ffffff" '
                     f'stroke="{stroke}" stroke-width="{width}"/>'
                     f'<svg x="{x}" y="10" width="100" height="100" viewBox="0 0 100 100">{inner}</svg>')
    w = 20 + 120 * len(cells)
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="120" viewBox="0 0 {w} 120">'
           f'<rect width="{w}" height="120" fill="#f3f4f6"/>{"".join(cells)}</svg>')
    return bytes(resvg_py.svg_to_bytes(svg_string=svg, zoom=2))


def _step(trial: Mapping[str, Any], second: bool) -> Mapping[str, Any]:
    """The trial as one question sees it: a two-measure trial's second question (``widget2``) with
    its own probed or judged object and message under the first's names."""
    if not second:
        return trial
    return {**trial, "widget": trial["widget2"], "prompt_html": trial["prompt2_html"],
            **{k: trial["second_" + k] for k in ("probe", "target", "utterance") if "second_" + k in trial}}


DESIGN_FIELDS = ("probe", "target", "utterance", "second_probe", "second_target", "second_utterance")


def trial_screen(trial: Mapping[str, Any], kind: str, second: bool = False) -> Tuple[str, List[bytes]]:
    """A trial as the participant reads it: text, and its image for SVG materials (the second
    question of a two-measure trial with ``second``)."""
    trial = _step(trial, second)
    w, n = trial["widget"], len(trial["items_html"])
    lines = ["(Practice question.)"] if trial["phase"] == "practice" and not second else []
    if second:
        lines.append("A second question about the same picture.")
    lines.append(_plain(trial["prompt_html"]))
    images: List[bytes] = []
    if kind == "svg":
        images.append(screen_png(trial))
        lines.append(f"The picture shows {n} objects, numbered 1 to {n} from left to right.")
    else:
        lines += [f"Object {p + 1}: {label}" for p, label in enumerate(trial["item_labels"])]
    if w.get("marked") is not None:
        frame = "with an orange frame" if kind == "svg" else "marked"
        lines.append(f"The question is about object {w['marked'] + 1}, which is {frame}.")
    t = w["type"]
    if t == "choice":
        lines.append(f"Choose one object: reply ANSWER: <the object's number, 1 to {n}>.")
    elif t == "rating":
        lo, hi = w["labels"]
        lines.append(f"Answer on a scale from 1 ({lo}) to {w['points']} ({hi}): reply ANSWER: <1 to {w['points']}>.")
    elif t == "slider":
        lo, hi = w["labels"]
        lines.append(f"Set a slider from 0 ({lo}) to {w['max']} ({hi}): reply ANSWER: <a whole number, 0 to {w['max']}>.")
    elif t == "bet":
        lines.append(f"Split {w['total']} points among the objects: reply ANSWER: <{n} whole numbers, one per object "
                     f"in order, separated by commas, adding up to {w['total']}>.")
    elif t == "production":
        words = ", ".join(o["label"] for o in w["options"])
        lines.append(f"Choose one of these words: {words}. Reply ANSWER: <the word>.")
    else:
        lines.append("Reply ANSWER: yes or ANSWER: no.")
    return "\n".join(lines), images


def parse_answer(trial: Mapping[str, Any], reply: str, second: bool = False) -> Tuple[Optional[dict], str]:
    """The response the page would record for this reply, or (None, why not)."""
    trial = _step(trial, second)
    found = ANSWER.findall(reply)
    if not found:
        return None, "no line starting with ANSWER:"
    a = found[-1].strip().strip("`*").strip().rstrip(".")
    w, n = trial["widget"], len(trial["items_html"])
    order = trial["display_order"]
    t = w["type"]
    try:
        if t == "choice":
            k = int(a)
            if not 1 <= k <= n:
                return None, f"the object number must be 1 to {n}"
            return {"choice": order[k - 1]}, ""
        if t == "rating":
            v = int(a)
            if not 1 <= v <= w["points"]:
                return None, f"the rating must be 1 to {w['points']}"
            return {"probe": trial["probe"], "rating": v}, ""
        if t == "slider":
            v = int(a)
            if not 0 <= v <= w["max"]:
                return None, f"the value must be 0 to {w['max']}"
            return {"probe": trial["probe"], "value": v}, ""
        if t == "bet":
            pts = [int(x) for x in re.split(r"[,\s]+", a) if x]
            if len(pts) != n or any(x < 0 for x in pts) or sum(pts) != w["total"]:
                return None, f"give {n} non-negative whole numbers adding up to {w['total']}"
            by_object = [0] * n
            for p, x in enumerate(pts):
                by_object[order[p]] = x
            return {"points": by_object}, ""
        if t == "production":
            words = {o["label"].lower(): o["utterance"] for o in w["options"]}
            word = a.strip("\"'“”").lower()
            if word not in words:
                return None, f"the word must be one of {', '.join(o['label'] for o in w['options'])}"
            return {"target": trial["target"], "utterance": words[word]}, ""
        if a.lower() in ("yes", "no"):
            return {"target": trial["target"], "utterance": trial["utterance"], "answer": a.lower() == "yes"}, ""
        return None, "answer yes or no"
    except ValueError:
        return None, "the answer is not in the form asked for"


# --------------------------------------------------------------------------- legibility

@dataclass(frozen=True)
class LegibilityItem:
    """One question about the materials (`legibility_items`)."""

    key: str  # "houses: the base" / "houses: chimney" / "trial 3: chimney"
    kind: str  # "name": free naming; "choice": one of ``options``; "truth": the objects a word is true of
    text: str
    image: bytes
    expected: Any  # the domain's name; the right option; the set of object numbers
    options: Tuple[str, ...] = ()


def _plural_key(word: str) -> str:
    w = word.strip().lower().strip("\"'“”.!")
    for suffix in ("ies", "es", "s"):
        if w.endswith(suffix) and len(w) > len(suffix) + 2:
            return w[: -len(suffix)] + ("y" if suffix == "ies" else "")
    return w


def legibility_items(paradigm, lists: Optional[Mapping[str, Any]] = None, seed: int = 0) -> List[LegibilityItem]:
    """The materials check's questions (SVG materials; text materials say what they are):

    - each domain's base alone, named freely ("name");
    - the base beside the base with one feature, which of the domain's words names the added part
      ("choice");
    - with ``lists``, up to ``TRUTH_ITEMS`` real test trials, unmarked: which objects have a word
      true of some but not all of them ("truth", exact)."""
    import numpy as np

    from src.rsa_open.screens import _object_html

    if paradigm.materials.kind != "svg":
        return []
    rng = np.random.default_rng(seed)
    items: List[LegibilityItem] = []
    for domain in paradigm.materials.domains:
        base = _object_html(paradigm, domain, [], domain.name)
        items.append(LegibilityItem(f"{domain.name}: the base", "name",
                                    f"What is pictured? Name it in one or two words.\n{NAME_ASK}",
                                    row_png([base]), domain.name))
        words = [f.word for f in domain.features]
        for j, feature in enumerate(domain.features):
            order = [int(i) for i in rng.permutation(len(words))]
            shown = tuple(words[i] for i in order)
            listing = "\n".join(f"{i + 1}) {o}" for i, o in enumerate(shown))
            items.append(LegibilityItem(
                f"{domain.name}: {feature.word}", "choice",
                f"Object 2 is object 1 with one part added. Which word names the added part?\n{listing}\n{CHOICE_ASK}",
                row_png([base, _object_html(paradigm, domain, [j], domain.name)]), feature.word, shown))
    if lists is not None:
        trials = [t for lst in lists["lists"] for t in lst["trials"] if t["phase"] == "test"]
        seen = set()
        for k in rng.permutation(len(trials)):
            t = trials[int(k)]
            n = len(t["objects"])
            mixed = [j for j in range(len(t["words"])) if 0 < sum(row[j] for row in t["objects"]) < n]
            key = (json.dumps(t["objects"]), t["domain"])
            if not mixed or key in seen:
                continue
            seen.add(key)
            j = mixed[int(rng.integers(len(mixed)))]
            word = t["words"][j]
            having = {p + 1 for p, obj in enumerate(t["display_order"]) if t["objects"][obj][j]}
            items.append(LegibilityItem(
                f"trial {len(seen)} ({t['domain_name']}): {word}", "truth",
                f"The picture shows {n} objects, numbered 1 to {n} from left to right. Which of them have "
                f"{word}?\n{TRUTH_ASK}", row_png(t["items_html"]), frozenset(having)))
            if len(seen) == TRUTH_ITEMS:
                break
    return items


def _answer(reply: str) -> Optional[str]:
    found = ANSWER.findall(reply)
    return found[-1].strip().strip("`*").strip().rstrip(".") if found else None


def _judge_name(judge: Participant, given: str, intended: str) -> bool:
    reply = judge.reply(f"Someone shown a simple drawing named it \"{given}\". The drawing was meant to show one of "
                        f"the {intended}. Is \"{given}\" a reasonable name for that kind of thing (the same kind of "
                        f"object, a synonym, or a more specific kind)? Reply ANSWER: yes or ANSWER: no.")
    return (_answer(reply) or "").lower() == "yes"


def materials_check(paradigm, checkers: Sequence[Participant], out: Optional[Path] = None, *,
                    lists: Optional[Mapping[str, Any]] = None, judge: Optional[Participant] = None) -> Dict[str, Any]:
    """Each checker (a fresh chat) answers every legibility question; an item is misread when no
    more than half of them get it right: ``illegible`` for a feature or a trial, and
    ``unrecognised_bases`` (a warning) for a base. A free name that is not the domain's name
    (singular or plural) goes to ``judge`` (a fresh chat); without a judge it is wrong."""
    # The harness's own pictures (the fixed arm's, used with people in the closed campaign) are not
    # tested: the check exists for a designer's drawings.
    items = [] if paradigm.harness_materials else legibility_items(paradigm, lists)
    if not items:
        return {"items": 0, "legibility": None, "illegible": [], "unrecognised_bases": []}
    given: Dict[str, List[Optional[str]]] = {it.key: [] for it in items}
    right: Dict[str, List[bool]] = {it.key: [] for it in items}
    for c, checker in enumerate(checkers):
        transcript = []
        for it in items:
            reply = checker.reply(it.text, [it.image])
            transcript += [f"## {it.key}\n\n{it.text}", f"## Reply\n\n{reply}"]
            a = _answer(reply)
            if it.kind == "name":
                ok = a is not None and (_plural_key(a) == _plural_key(it.expected) or
                                        (judge is not None and _judge_name(judge, a, it.expected)))
                shown = a
            elif it.kind == "choice":
                try:
                    k = int(a) - 1 if a is not None else -1
                except ValueError:
                    k = -1
                shown = it.options[k] if 0 <= k < len(it.options) else a
                ok = shown == it.expected
            else:
                nums = set()
                if a is not None and a.lower() != "none":
                    try:
                        nums = {int(x) for x in re.split(r"[,\s]+", a) if x}
                    except ValueError:
                        nums = {-1}
                shown = ", ".join(map(str, sorted(nums))) or "none"
                ok = nums == set(it.expected)
            given[it.key].append(shown)
            right[it.key].append(ok)
        if out is not None:
            (out / "legibility").mkdir(parents=True, exist_ok=True)
            (out / "legibility" / f"checker_{c}.md").write_text("\n\n".join(transcript))
    rows = []
    for it in items:
        expected = (", ".join(map(str, sorted(it.expected))) or "none") if it.kind == "truth" else it.expected
        rows.append({"item": it.key, "kind": it.kind, "right": sum(right[it.key]), "of": len(right[it.key]),
                     "answer": expected,
                     "wrong_answers": sorted({g for g, ok in zip(given[it.key], right[it.key]) if not ok and g})})
    total = sum(r["of"] for r in rows)
    misread = [r for r in rows if r["right"] * 2 <= r["of"]]
    return {"items": len(items), "legibility": sum(r["right"] for r in rows) / total,
            "illegible": [r for r in misread if r["kind"] != "name"],
            "unrecognised_bases": [r for r in misread if r["kind"] == "name"], "by_item": rows}


# --------------------------------------------------------------------------- one participant

@dataclass
class Session:
    list_index: int
    checks: List[List[bool]] = field(default_factory=list)  # per reading of the instructions
    rows: List[Dict[str, Any]] = field(default_factory=list)
    transcript: List[str] = field(default_factory=list)


def _ask(participant: Participant, session: Session, text: str, images: Sequence[bytes] = ()) -> str:
    session.transcript.append(f"## Screen\n\n{text}" + (f"\n\n[{len(images)} image]" if images else ""))
    reply = participant.reply(text, images)
    session.transcript.append(f"## Reply\n\n{reply}")
    return reply


def _check(participant: Participant, session: Session, check: Mapping[str, Any], prefix: str) -> bool:
    options = "\n".join(f"{i + 1}) {o}" for i, o in enumerate(check["options"]))
    text = f"{prefix}{check['question']}\n{options}\nReply ANSWER: <the option's number>."
    found = ANSWER.findall(_ask(participant, session, text))
    try:
        return bool(found) and int(found[-1].strip().strip("`*").rstrip(".")) - 1 == check["correct"]
    except ValueError:
        return False


def run_participant(lists: Mapping[str, Any], index: int, participant: Participant) -> Session:
    lst = lists["lists"][index]
    intro, kind = lists["intro"], lists["materials_kind"]
    session = Session(index)
    lead = (f"Welcome. {intro['welcome']}\nIt takes about {lists['minutes']} minutes.\n\n"
            f"Instructions:\n{intro['task']}\n\n")
    for reading in range(2):
        answers = []
        for i, check in enumerate(intro["checks"]):
            prefix = lead if i == 0 else ""
            answers.append(_check(participant, session, check, prefix))
        session.checks.append(answers)
        if all(answers):
            break
        lead = f"Not quite: please read the instructions again.\n\nInstructions:\n{intro['task']}\n\n"
    def answer(trial, second):
        text, images = trial_screen(trial, kind, second)
        response, why = parse_answer(trial, _ask(participant, session, text, images), second)
        if response is None:
            response, why = parse_answer(trial, _ask(
                participant, session, f"Your answer could not be read: {why}. Please answer again.\n\n{text}", images),
                second)
        return response, why

    for trial in lst["trials"]:
        response, why = answer(trial, False)
        if response is not None and "widget2" in trial:  # the page asks the second once the first is answered
            second, why = answer(trial, True)
            response = None if second is None else {**response, "second": second}
        session.rows.append({"phase": trial["phase"], "unit": trial["unit"], "trial_number": trial["trial_number"],
                             "objects": trial["objects"], "heard": trial["heard"], "expect": trial.get("expect"),
                             **{k: trial[k] for k in DESIGN_FIELDS if k in trial},
                             "response": response, "unreadable": why if response is None else None})
    return session


# --------------------------------------------------------------------------- the gate

def _records(timing: Optional[Sequence]) -> Sequence:
    from src.rsa_open.timing import DEFAULT_RECORDS

    return DEFAULT_RECORDS if timing is None else timing


def estimated_minutes(lists: Mapping[str, Any], timing: Optional[Sequence] = None) -> float:
    """A participant's minutes under these lists, calibrated on ``timing`` records (default: the
    closed campaign's and run D2's; `src.rsa_open.timing.estimate`)."""
    from src.rsa_open.timing import estimate

    return estimate(lists, _records(timing))["estimated_minutes"]


def gate(paradigm, lists: Mapping[str, Any], sessions: Sequence[Session],
         timing: Optional[Sequence] = None, legibility: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    from src.rsa_open.fit import OpenData
    from src.rsa_open.paradigm import paradigm_likelihood
    from src.rsa_open.screens import catch_passed, scenes_and_responses

    rows = [r for s in sessions for r in s.rows]
    readable = [r for r in rows if r["response"] is not None]
    catches = [r for r in readable if r["phase"] == "catch"]
    n_catch = sum(r["phase"] == "catch" for r in rows)
    report: Dict[str, Any] = {
        "paradigm_sha256": paradigm.sha256, "design_sha256": lists["design_sha256"],
        "n_participants": len(sessions),
        "comprehension_first": sum(all(s.checks[0]) for s in sessions) / len(sessions),
        "comprehension_final": sum(all(s.checks[-1]) for s in sessions) / len(sessions),
        "catch_pass": sum(catch_passed(r) for r in catches) / max(n_catch, 1),
        "practice_pass": (sum(catch_passed(r) for r in readable if r["phase"] == "practice")
                          / max(sum(r["phase"] == "practice" for r in rows), 1)),
        "readable": len(readable) / max(len(rows), 1),
        "unreadable_examples": [r["unreadable"] for r in rows if r["response"] is None][:5],
        "declared_minutes": paradigm.design.minutes,
    }
    from src.rsa_open.timing import estimate

    t = estimate(lists, _records(timing))
    report.update(estimated_minutes=t["estimated_minutes"], time_model_minutes=t["model_minutes"],
                  time_calibration=t["calibration"], time_calibration_basis=t["calibration_basis"])
    if legibility is not None:
        report["legibility"] = legibility
    problems, warnings = [], []
    if report["comprehension_final"] < FLOORS["comprehension"]:
        problems.append(f"comprehension: {report['comprehension_final']:.0%} answered every check right by the "
                        f"second reading (floor {FLOORS['comprehension']:.0%})")
    if report["catch_pass"] < FLOORS["catch"]:
        problems.append(f"attention: {report['catch_pass']:.0%} of catch trials passed (floor {FLOORS['catch']:.0%})")
    if report["readable"] < FLOORS["readable"]:
        problems.append(f"readability: {report['readable']:.0%} of trials got a usable answer "
                        f"(floor {FLOORS['readable']:.0%}); e.g. {report['unreadable_examples'][:2]}")
    tests = [r for r in readable if r["phase"] == "test"]
    try:
        scenes, responses = scenes_and_responses(tests)
        OpenData.build(scenes, responses, paradigm_likelihood(paradigm))
        report["data_parse"] = "ok"
    except ValueError as err:
        report["data_parse"] = str(err)
        problems.append(f"data: the test responses do not parse as the link's data ({err})")
    ratio = report["estimated_minutes"] / paradigm.design.minutes
    report["time_ratio"] = ratio
    if abs(ratio - 1.0) > TIME_TOLERANCE:
        problems.append(f"time: estimated {report['estimated_minutes']:.1f} minutes against the declared "
                        f"{paradigm.design.minutes:g} (more than {TIME_TOLERANCE:.0%} apart): change the trials or "
                        f"the minutes (and the pay)")
    from src.rsa_open.paradigm import MIN_HOURLY_USD
    from src.rsa_open.timing import hourly_usd

    report["hourly_usd"] = hourly_usd(paradigm.design.pay_usd, paradigm.design.minutes, report["estimated_minutes"])
    if report["hourly_usd"] < MIN_HOURLY_USD - 1e-9:
        problems.append(f"pay: ${paradigm.design.pay_usd:.2f} for the estimated {report['estimated_minutes']:.1f} "
                        f"minutes is ${report['hourly_usd']:.2f} per hour, below ${MIN_HOURLY_USD:.2f} (the floor holds "
                        f"at the larger of the declared and the estimated minutes): raise the pay or shorten the study")
    if legibility is not None and legibility["legibility"] is not None:
        bad = "; ".join(f"{r['item']} ({r['right']} of {r['of']} chose '{r['answer']}'"
                        + (f"; others chose {', '.join(r['wrong_answers'])}" if r["wrong_answers"] else "") + ")"
                        for r in legibility["illegible"])
        bases = "; ".join(f"the base of {r['item'].split(':')[0]} was not recognised ({r['right']} of {r['of']} "
                          f"named it '{r['answer']}' or a fair synonym"
                          + (f"; others said {', '.join(r['wrong_answers'])}" if r["wrong_answers"] else "") + ")"
                          for r in legibility.get("unrecognised_bases", []))
        if legibility["legibility"] < FLOORS["legibility"] or legibility["illegible"]:
            problems.append(f"legibility: {legibility['legibility']:.0%} of the materials questions answered right "
                            f"(floor {FLOORS['legibility']:.0%})" + (f"; misread: {bad}" if bad else "")
                            + (f"; {bases}" if bases else ""))
        elif bases:
            warnings.append(f"legibility: {bases}")
    distinct = {json.dumps(r["response"], sort_keys=True) for r in tests}
    if len(tests) > 1 and len(distinct) == 1:
        problems.append(f"degenerate: every test answer was {distinct.pop()}")
    report["problems"] = problems
    report["warnings"] = warnings
    report["passed"] = not problems
    return report


def run_pilot(paradigm, lists: Mapping[str, Any], out: Path, *, n_participants: int = PILOT_PARTICIPANTS,
              participant_factory: Optional[Callable[[int], Participant]] = None,
              max_workers: int = 6, timing: Optional[Sequence] = None) -> Dict[str, Any]:
    """Pilot ``n_participants`` lists (the first ones) and gate the paradigm; writes the record to ``out``."""
    if lists["paradigm_sha256"] != paradigm.sha256:
        raise ValueError("the lists were not built from this paradigm")
    n = min(n_participants, len(lists["lists"]))
    factory = participant_factory or (lambda i: GeminiParticipant())
    participants = [factory(i) for i in range(n)]
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        sessions = list(pool.map(lambda i: run_participant(lists, i, participants[i]), range(n)))
    out = Path(out)
    # The materials check: fresh chats, so no participant's trials colour it.
    legibility = materials_check(paradigm, [factory(n + c) for c in range(LEGIBILITY_CHECKERS)], out, lists=lists,
                                 judge=factory(n + LEGIBILITY_CHECKERS))
    (out / "transcripts").mkdir(parents=True, exist_ok=True)
    for s in sessions:
        (out / "transcripts" / f"participant_{s.list_index:03d}.md").write_text("\n\n".join(s.transcript))
    with (out / "rows.jsonl").open("w") as f:
        for s in sessions:
            for r in s.rows:
                f.write(json.dumps({"list_index": s.list_index, **r}) + "\n")
    report = gate(paradigm, lists, sessions, timing, legibility)
    report["participant_model"] = participants[0].name
    (out / "pilot.json").write_text(json.dumps(report, indent=2))
    return report


@dataclass
class Args:
    paradigm: Path
    lists: Path
    out: Path
    n_participants: int = PILOT_PARTICIPANTS


def main(args: Args) -> Dict[str, Any]:
    from src.rsa_open.paradigm import load_paradigm

    report = run_pilot(load_paradigm(args.paradigm), json.loads(args.lists.read_text()), args.out,
                       n_participants=args.n_participants)
    print(f"pilot {'PASSED' if report['passed'] else 'FAILED'}: " + ("; ".join(report["problems"]) or "no problems"))
    return report


if __name__ == "__main__":
    main(tyro.cli(Args))
