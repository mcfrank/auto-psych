"""The fixed arm's paradigm (PI 2026-10-26; docs/auto_open/PREREG_DRAFT.md §2).

    uv run python -m src.rsa_open.fixed            # writes src/rsa_open/fixed_paradigm.json
    uv run python -m src.rsa_open.fixed --pair     # the two-measure rehearsal variant (fixed_pair_paradigm.json)

The two arms differ only in who chooses each experiment. The open arm's designer writes a new
paradigm every time; the fixed arm runs the closed campaign's forced-choice reference game every
time, as one frozen spec, with its displays chosen by the same EIG tool (`src.rsa_open.chain`, arm
``fixed``). This module writes that spec from the closed campaign's own materials:
- the pragmods pictures and words (`src.rsa.experiment.design.DOMAINS`, PI-approved stimuli),
  each object drawn on one of an item's three base pictures (the spec's base ``variants``);
- Bob's wording, from the closed page (`src/rsa/experiment/template.html`);
- the closed scene space (2-4 objects × 2-4 features, prior queries, synonyms;
  `src.rsa_open.design.REFERENCE`): 13 test trials of 52 designed displays, 2 catch trials;
- $0.50 for 3 minutes, as the closed campaign paid, for **400 people** (the per-experiment budget,
  twice the closed campaign's).

The page around it is the open arm's, like everything else: the consent gate, comprehension
checks, practice, catch trials and data. The closed page had no comprehension checks; here they
are the same kind of check every open paradigm has.

The pictures are embedded as PNG data in SVG ``<image>`` elements, which the spec validator allows
only for these exact files (`src.rsa_open.paradigm.TRUSTED_IMAGES_DIR`). Harness only: neither this
module nor its spec is in the agents' tree (`scripts/rsa/slurm/agent_tree.exclude`).
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import tyro

SPEC_PATH = Path(__file__).resolve().parent / "fixed_paradigm.json"
PAIR_SPEC_PATH = Path(__file__).resolve().parent / "fixed_pair_paradigm.json"

WELCOME = "In this short study we are interested in how people figure out what others are talking about."
TASK = ("Meet Bob. Bob wants to tell you about some of his things: his friends, his snowmen, the sundaes and "
        "pizzas he makes, his boats and his Christmas trees. On each screen you will see a few pictures. Bob can "
        "only say one word to tell you which one he means. Click on the picture you think Bob is talking about. "
        "Sometimes Bob mumbles and you won't be able to hear his word. Then just click on the one you think he is "
        "talking about. There are no right or wrong answers; go with your first impression.")
PROMPT = "Bob can only say one word to communicate with you and he says {message}. Click on the one you think Bob is talking about."
PRIOR_PROMPT = ("Bob can only say one word to communicate with you, but he mumbles: you couldn't hear what he said. "
                "Click on the one you think Bob is talking about.")
CHECKS = [
    {"question": "What do you do on each screen?",
     "options": ["Click on the picture you think Bob is talking about", "Type the word Bob said",
                 "Rate how much you like each picture"], "correct": 0},
    {"question": "How many words can Bob say on each screen?",
     "options": ["One", "Two", "As many as he likes"], "correct": 0},
]
N_PARTICIPANTS = 400


def _image(path: Path) -> str:
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f'<image href="data:image/png;base64,{data}" x="0" y="0" width="100" height="100"/>'


def build() -> Dict[str, Any]:
    from src.rsa.experiment.design import DOMAINS, IMAGES_DIR, N_BASES
    from src.rsa_open.design import REFERENCE

    domains = []
    for d in DOMAINS.values():
        variants = [_image(IMAGES_DIR / d.base_image(b)) for b in range(1, N_BASES + 1)]
        domains.append({"name": d.name, "base": variants[0], "variants": variants,
                        "features": [{"word": f.word, "svg": _image(IMAGES_DIR / f.image)} for f in d.features]})
    return {
        "schema_version": 1,
        "name": "Which one does Bob mean? (forced choice)",
        "summary": ("The closed campaign's reference game: participants see a few pictures, Bob says one word or "
                    "mumbles, and they click the picture they think he means."),
        "charter": {"question": "How do listeners use the alternatives a speaker could have said to pick a referent?"},
        "instructions": {"welcome": WELCOME, "task": TASK, "comprehension_checks": CHECKS},
        "materials": {"kind": "svg", "domains": domains},
        "scene_space": {"sizes": [list(s) for s in REFERENCE["sizes"]], "prior_queries": REFERENCE["prior_queries"],
                        "synonyms": REFERENCE["synonyms"]},
        "response": {"type": "choice", "link": REFERENCE["link"], "prompt": PROMPT, "prior_prompt": PRIOR_PROMPT},
        "design": {"n_participants": N_PARTICIPANTS, "trials_per_participant": REFERENCE["trials_per_participant"],
                   "distinct_trials": REFERENCE["distinct_trials"], "catch_trials": 2, "minutes": 3, "pay_usd": 0.50},
        "content": {"deception": False, "affective_stimuli": False},
    }


def build_pair() -> Dict[str, Any]:
    """The fixed paradigm with a second question about the same screen (rehearsal only, PI
    2026-10-26: a designer chooses its own format, so the two-measure path is rehearsed on a spec
    the harness writes): after each choice, a 5-point rating of how likely Bob means the marked
    picture. 288 people (72 responses a display) for 4 minutes keep it within the budget."""
    spec = build()
    spec["name"] = "Which one does Bob mean, and how likely is the marked one? (two measures, rehearsal)"
    spec["summary"] += " Then Bob marks one picture, and they rate how likely it is that he meant that one."
    spec["instructions"]["task"] += (" After each click, one of the pictures is marked: say how likely it is that Bob "
                                     "means that one, from 1 (not at all) to 5 (certainly).")
    spec["second_response"] = {"type": "rating", "link": "rating",
                               "prompt": "Bob said {message}. How likely is it that he means the marked one?",
                               "prior_prompt": "Bob mumbled. How likely is it that he means the marked one?",
                               "scale_points": 5, "scale_labels": ["not at all", "certainly"]}
    spec["design"].update(n_participants=288, minutes=4, pay_usd=0.67)
    return spec


def write(path: Path = SPEC_PATH, pair: bool = False) -> Path:
    from src.rsa_open.paradigm import parse_paradigm

    spec = build_pair() if pair else build()
    parse_paradigm(spec, "the fixed paradigm")  # raises with every problem
    Path(path).write_text(json.dumps(spec, indent=1, ensure_ascii=False) + "\n")
    return Path(path)


@dataclass
class Args:
    out: Path = SPEC_PATH
    pair: bool = False
    """The two-measure rehearsal variant (write it to PAIR_SPEC_PATH)."""


if __name__ == "__main__":
    a = tyro.cli(Args)
    print(write(PAIR_SPEC_PATH if a.pair and a.out == SPEC_PATH else a.out, pair=a.pair))
