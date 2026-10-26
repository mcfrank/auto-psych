"""How long experiments take people (PI 2026-10-21): a digest for the designer's brief, and the
calibration of the pilot's time estimate, from the same records.

A designer must declare its paradigm's minutes, and the pilot gate checks them against an
estimate (`src.rsa_open.pilot`). The estimate's model (`model_seconds`: an overhead, reading at
``READING_WPM``, a look at each object and an answer per trial) is a guess; people's real times
correct it. Each `TimingRecord` is one finished experiment, as aggregates only: what people did
(response type, materials, trials, objects and words per trial, instruction length) and how
long they took (median, p10, p90 seconds, from Prolific's completion times or the page's own
clock). Two sources:
- **the closed campaign** (`CLOSED_CAMPAIGN`): its 9 live experiments, from the campaign
  reports (`docs/auto_rsa/CAMPAIGN_C*_REPORT.md`): 16 forced choices (13 + 2 catch + practice)
  on SVG displays averaging 3.44 objects. These are completion times only, never responses: the
  PI ruled that durations are not closed-campaign data in the sense of the drift rule (§5).
- **this campaign's finished experiments**, every chain's (`record_from_sessions`).

**Calibration** (`calibration`): the ratio of people's median to the model's estimate, as a
median over the records of the same response type when there are any, else over every record.
`estimated_minutes` is the model times that ratio; the record says which basis it used. With
the closed campaign alone, a choice paradigm's estimate is scaled by its own records, and any
other response type by the same choice records, until this campaign has run one of its kind.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from html import unescape
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

READING_WPM = 200
SECONDS_PER_OBJECT = 1.5
SECONDS_TO_ANSWER = {"choice": 2.0, "rating": 3.0, "slider": 4.0, "bet": 3.0, "production": 3.0,
                     "binary_judgment": 2.0}
SECONDS_PER_BET_OBJECT = 3.0
SECONDS_OVERHEAD = 30.0  # consent, page loads, the end screen


@dataclass(frozen=True)
class TimingRecord:
    source: str
    response_type: str
    materials: str
    trials: int  # every trial a person answers: practice, test and catch
    objects_per_trial: float
    words_per_trial: float  # the prompt, and written objects for text materials
    instruction_words: int  # welcome, task and comprehension checks
    n_people: int
    median_s: float
    p10_s: float
    p90_s: float

    @classmethod
    def from_json(cls, d: Mapping[str, Any]) -> "TimingRecord":
        return cls(**d)


def _closed(chain: int, experiment: int, n: int, med: float, p10: float, p90: float) -> TimingRecord:
    # The closed page: 18 + 100 instruction words (welcome, "Meet Bob"); 24-word prompts.
    return TimingRecord(f"closed campaign, chain {chain}, experiment {experiment}", "choice", "svg", 16, 3.44,
                        24.0, 118, n, med, p10, p90)


# docs/auto_rsa/CAMPAIGN_C0_REPORT.md, CAMPAIGN_C1_REPORT.md, CAMPAIGN_C2_REPORT.md: completers.
CLOSED_CAMPAIGN: Tuple[TimingRecord, ...] = (
    _closed(0, 1, 200, 144, 86, 290), _closed(0, 2, 201, 124, 77, 200), _closed(0, 3, 201, 135, 85, 248),
    _closed(1, 1, 200, 119, 79, 212), _closed(1, 2, 201, 162, 96, 340), _closed(1, 3, 200, 126, 82, 242),
    _closed(2, 1, 196, 115, 74, 230), _closed(2, 2, 200, 129, 83, 254), _closed(2, 3, 200, 104, 69, 198),
)


def model_seconds(response_type: str, materials: str, trials: int, objects_per_trial: float,
                  words_per_trial: float, instruction_words: float) -> float:
    """The uncalibrated estimate of one person's time."""
    per_trial = (60.0 * words_per_trial / READING_WPM + SECONDS_PER_OBJECT * objects_per_trial
                 # a two-measure trial ("choice+rating") answers both
                 + sum(SECONDS_TO_ANSWER[t] for t in response_type.split("+"))
                 + (SECONDS_PER_BET_OBJECT * objects_per_trial if response_type.split("+")[0] == "bet" else 0.0))
    return SECONDS_OVERHEAD + 60.0 * instruction_words / READING_WPM + trials * per_trial


def _words(s: str) -> int:
    return len(unescape(re.sub(r"<[^>]+>", " ", s)).split())


def features(lists: Mapping[str, Any]) -> Dict[str, Any]:
    """What a person does under these trial lists (averaged over the lists), for the model."""
    intro = lists["intro"]
    instruction = _words(intro["welcome"]) + _words(intro["task"]) + sum(
        _words(c["question"]) + sum(_words(o) for o in c["options"]) for c in intro["checks"])
    trials = [t for lst in lists["lists"] for t in lst["trials"]]
    text = lists["materials_kind"] == "text"
    words = [_words(t["prompt_html"]) + _words(t.get("prompt2_html", "")) + (sum(len(x.split()) for x in t["item_labels"]) if text else 0)
             for t in trials]
    return {"response_type": lists["response_type"], "materials": lists["materials_kind"],
            "trials": len(lists["lists"][0]["trials"]),
            "objects_per_trial": sum(len(t["objects"]) for t in trials) / len(trials),
            "words_per_trial": sum(words) / len(words), "instruction_words": instruction}


def calibration(records: Sequence[TimingRecord], response_type: str) -> Tuple[float, str]:
    """(people's median / the model's estimate, the basis), over records of this type if any."""
    def ratio(r: TimingRecord) -> float:
        return r.median_s / model_seconds(r.response_type, r.materials, r.trials, r.objects_per_trial,
                                          r.words_per_trial, r.instruction_words)

    same = [r for r in records if r.response_type == response_type]
    use = same or list(records)
    if not use:
        return 1.0, "no timing records: the model's estimate, uncalibrated"
    basis = (f"{len(use)} {response_type} experiment(s)" if same
             else f"{len(use)} experiment(s) of other response types (none of {response_type} yet)")
    return float(median(ratio(r) for r in use)), basis


def estimate(lists: Mapping[str, Any], records: Sequence[TimingRecord]) -> Dict[str, Any]:
    """The calibrated estimate of a person's minutes under these lists, with how it was made."""
    f = features(lists)
    raw = model_seconds(**f) / 60.0
    k, basis = calibration(records, f["response_type"])
    return {"estimated_minutes": raw * k, "model_minutes": raw, "calibration": k, "calibration_basis": basis,
            **{f"features_{key}": value for key, value in f.items()}}


def record_from_sessions(source: str, lists: Mapping[str, Any], seconds: Sequence[float]) -> TimingRecord:
    """A finished experiment's record: its lists' features and its people's completion times."""
    if not seconds:
        raise ValueError(f"{source}: no completion times")
    s = sorted(float(x) for x in seconds)

    def q(p: float) -> float:
        return s[min(len(s) - 1, max(0, round(p * (len(s) - 1))))]

    f = features(lists)
    return TimingRecord(source, f["response_type"], f["materials"], f["trials"], round(f["objects_per_trial"], 2),
                        round(f["words_per_trial"], 1), f["instruction_words"], len(s), q(0.5), q(0.1), q(0.9))


def write_records(path: Path, records: Iterable[TimingRecord]) -> None:
    Path(path).write_text(json.dumps([asdict(r) for r in records], indent=2))


def read_records(path: Path) -> List[TimingRecord]:
    return [TimingRecord.from_json(d) for d in json.loads(Path(path).read_text())]


def digest_md(records: Sequence[TimingRecord]) -> str:
    """The records as the designer reads them."""
    lines = ["# How long experiments took people", "",
             "Completion times of finished experiments (aggregates only: how long, never what people "
             "answered). Use them to judge how many trials fit in your minutes. The pilot's time check uses "
             "the same records: its estimate is a simple model (reading at 200 words a minute, a look at "
             "each object, an answer per trial) scaled by people's real times on experiments of the "
             "same response type, or of every type when none has run yet.", "",
             "| experiment | response | materials | trials | objects/trial | words/trial | people | median | "
             "p10-p90 | seconds/trial (median) |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in records:
        per_trial = (r.median_s - SECONDS_OVERHEAD - 60.0 * r.instruction_words / READING_WPM) / r.trials
        lines.append(f"| {r.source} | {r.response_type} | {r.materials} | {r.trials} | {r.objects_per_trial:g} | "
                     f"{r.words_per_trial:g} | {r.n_people} | {r.median_s / 60:.1f} min | "
                     f"{r.p10_s / 60:.1f}-{r.p90_s / 60:.1f} min | {per_trial:.1f} |")
    lines += ["", "Seconds per trial: the median, less a 30 s overhead and the instructions' reading time, "
              "over the trials. Prolific pays by the minutes you declare: declare about the median.", ""]
    return "\n".join(lines)
