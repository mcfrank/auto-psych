"""The theorist's documents for the open loop (the closed loop's `src.rsa.loop.brief`, for
theories judged on every paradigm of a chain).

The slot texts (exploratory, refine the incumbent, refine one of your choice), the existing
and attempted hypotheses, the refinement menu, the repair and retry notes and the lenses are
the closed loop's, with the theory's API (``listener``) in place of ``choice_probs``. What is
new is the context (`context_md`): the chain's paradigms, each with its response type, its
link, its data file (`write_block_data`: one row per trial, the scene and the response as
JSON) and, for an experiment, the paradigm's own words; and the self-check command
(`src.rsa_open.loop.check_candidate`).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

from src.rsa.loop import brief as closed
from src.rsa_open.corpus import Block
from src.rsa_open.sampler import CORRELATION_FLAG

PROMPTS = Path(__file__).with_name("prompts")
THEORY_PROMPT = PROMPTS / "open_theory.md"
HANDBOOK = closed.HANDBOOK
LENSES = [lens.replace("click", "response") for lens in closed.PLAIN_RSA_LENSES]
CHECK_COMMAND = (sys.executable + " -m src.rsa_open.loop.check_candidate {candidate_dir} --corpus {corpus}"
                 " --full-corpus {full_corpus} --limit-minutes {limit_minutes}")
CHECK_SHELL_TIMEOUT_MS = 1_200_000
NO_NETWORK_NOTE = closed.NO_NETWORK_NOTE
DEADLINE_FILE = "deadline.json"
DRAFT_MINUTES = 15
URGENT_SECONDS = 10 * 60
"""A theorist's session has a clock (PI 2026-10-28): run C's Gemini 4 theorists were cut off with no
candidate.py in 19 of 23 sessions, all still exploring. The brief asks for a complete first version
within ``DRAFT_MINUTES`` and states when the session ends; every self-check prints the time left."""


def write_deadline(candidate_dir: Path, timeout_sec: float, now: Optional[float] = None) -> float:
    """The session's end (epoch seconds) in ``candidate_dir/deadline.json``, for `time_note`."""
    import time

    end = (time.time() if now is None else now) + timeout_sec
    Path(candidate_dir).mkdir(parents=True, exist_ok=True)
    (Path(candidate_dir) / DEADLINE_FILE).write_text(json.dumps({"deadline": end, "clock": time.strftime(
        "%H:%M %Z", time.localtime(end))}))
    return end


def time_note(candidate_dir: Path, now: Optional[float] = None) -> Optional[str]:
    """How long the theorist's session has left (None without a deadline)."""
    import time

    path = Path(candidate_dir) / DEADLINE_FILE
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    left = d["deadline"] - (time.time() if now is None else now)
    note = f"Time: {max(0, left) / 60:.0f} min left in this session (it ends at {d['clock']})."
    if left < URGENT_SECONDS:
        note += (" Finish now: make sure candidate.py, hypothesis.md and model_name.txt in your directory are the "
                 "version you want judged. Whatever is there when the session ends is what the loop admits.")
    return note

RESPONSE_WORDS = {
    "choice": "click the object the speaker means (the listener)",
    "rating": "rate how likely the speaker means a marked object (the listener)",
    "slider": "set a slider for how likely the speaker means a marked object (the listener)",
    "bet": "split points over the objects by how likely the speaker means each (the listener)",
    "production": "choose the word to say for a marked object (the speaker)",
    "binary_judgment": "say whether a speaker would use a word for a marked object (the speaker)",
}


def _api(text: str) -> str:
    return text.replace("choice_probs", "listener").replace("Models in the set", "Theories in the set")


# The theorists' self-check fits a fixed fraction of each large block's displays (the literature),
# whole displays at a time, and every other block whole: a quick smoke fit for their feedback.
# Admission fits all the data (PI 2026-10-22).
CHECK_LARGE_BLOCK = 5000
CHECK_FRACTION = 0.2


def self_check_blocks(blocks: Sequence[Block], seed: int = 0) -> List[Block]:
    import numpy as np

    out = []
    for b in blocks:
        if b.n <= CHECK_LARGE_BLOCK:
            out.append(b)
            continue
        units = sorted(set(b.units))
        rng = np.random.default_rng(seed)
        keep = set(rng.choice(len(units), max(1, round(CHECK_FRACTION * len(units))), replace=False).tolist())
        kept = {units[i] for i in keep}
        out.append(b.subset([i for i, u in enumerate(b.units) if u in kept]))
    return out


def write_block_data(directory: Path, blocks: Sequence[Block]) -> Dict[str, Path]:
    """One JSON-lines file per block for the theorists to analyse: every trial's scene (objects
    as 0/1 feature rows, the heard word's index or null), its response in the link's form, its
    unit and source."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = {}
    for b in blocks:
        path = directory / f"{b.name}.jsonl"
        with path.open("w") as f:
            for i, (ctx, response, unit) in enumerate(zip(b.contexts, b.responses, b.units)):
                f.write(json.dumps({"trial": i, "objects": [list(r) for r in ctx.objects], "heard": ctx.utterance,
                                    "response": dict(response), "unit": unit,
                                    "source": b.sources[i] if b.sources else b.name}) + "\n")
        paths[b.name] = path
    return paths


def context_md(*, candidate_dir: Path, corpus_path: Path, full_corpus_path: Path,
               admission_limit_sec: Optional[float], blocks: Sequence[Block], data_paths: Mapping[str, Path],
               paradigm_notes: Mapping[str, str], round_index: int, n_rounds: int, network: bool = False,
               session_minutes: Optional[float] = None) -> str:
    lines = ["# Context", "", f"- Candidate directory (write your three files here): `{candidate_dir}`",
             f"- Inner-loop round {round_index + 1} of {n_rounds}.", ""]
    if session_minutes is not None:
        lines += ["## Your time", "",
                  f"This session ends after **{session_minutes:.0f} minutes**; every self-check prints the time "
                  f"left. **Write a complete first version of your three files within {DRAFT_MINUTES} minutes**, "
                  "run the self-check on it, then improve the files in place. Whatever is in your directory when the "
                  "session ends is what the loop judges; a session that ends with no candidate.py has wasted its "
                  "slot. Do not spend the session on fitting scripts of your own: the quick fit below answers what "
                  "they would, in seconds.", ""]
    lines += ["## The paradigms of this chain", "",
             "Your theory is fitted to all of them at once; the newest is last. The data files are one JSON "
             "object per line (`objects`: one row of 0/1 features per object; `heard`: the index of the word "
             "heard, `null` on a prior trial; `response` in its paradigm's form). You may analyse them from "
             "the shell; your theory file may not read them.", ""]
    for b in blocks:
        resp = ", then ".join(RESPONSE_WORDS.get(t, t) for t in b.link.response_type.split("+"))
        opts = ", ".join(f"{k} {v}" for k, v in b.link.options.items() if not isinstance(v, dict))
        if b.link.name == "pair":
            opts = (f"two questions per screen: {b.link.first.name} then {b.link.second.name}, each with its own "
                    f"link parameters (the second's prefixed q2_)")
        lines += [f"### {b.name}: {b.n} trials, people {resp}",
                  f"- link `{b.link.name}`{f' ({opts})' if opts else ''}; data `{data_paths[b.name]}`"]
        if b.fixed:
            held = ", ".join(f"`{k}` = {v:g}" for k, v in b.fixed_params.items())
            lines.append(f"- the reference paradigm: its link's {held} (not fitted); later paradigms' are "
                         "fitted relative to it")
        if paradigm_notes.get(b.name):
            lines.append(f"- {paradigm_notes[b.name]}")
        lines.append("")
    limit_minutes = "inf" if admission_limit_sec is None else f"{admission_limit_sec / 60:.0f}"
    if admission_limit_sec is not None:
        lines += ["## The admission limit", "",
                  f"The loop's fit of your theory to all the data must finish within **{limit_minutes} minutes** "
                  f"(twice the median fit time of the theories this loop started from), or it is rejected as "
                  f"too slow.", ""]
    lines += ["## Self-check", "",
              f"`{CHECK_COMMAND.format(candidate_dir=candidate_dir, corpus=corpus_path, full_corpus=full_corpus_path, limit_minutes=limit_minutes)}`",
              f"It compiles your theory and fits it briefly on every paradigm, the literature (E) on a fixed "
              f"{CHECK_FRACTION:.0%} of its displays, in a few minutes; the loop's admission fit uses all the "
              f"data. It reports how hard the posterior was to sample (steps per draw, step size, and any pair "
              f"of parameters correlated above {CORRELATION_FLAG}) and projects the admission fit's time from "
              f"the short fit. Give the shell tool a timeout of {CHECK_SHELL_TIMEOUT_MS} milliseconds for it. Run it once "
              f"per change.", "",
              "Add `--quick` to the same command for the quick fit instead (seconds, no sampling): the best-fitting "
              "values from several random starts, how sharply the data pin each parameter down (a Laplace "
              "approximation), which parameters trade off, and whether the starts disagree (more than one optimum). "
              "It is for exploring; the loop scores theories with full sampling, never with it.", ""]
    if not network:
        lines.append(NO_NETWORK_NOTE.rstrip())
    lines.append("- Write and test your files **in your candidate directory only**.")
    return "\n".join(lines) + "\n"


def write_docs(candidate_dir: Path, *, role: str, lens: Optional[str], context: str, live, pruned,
               incumbent: Optional[str], ledger, attempt_note: Optional[str] = None,
               critiques: Optional[str] = None) -> Dict[str, Optional[str]]:
    """The closed loop's documents (`src.rsa.loop.brief.write_docs`), for a theory; ``critiques``
    is the round's critiques.md (`src.rsa_open.loop.critique`), or None."""
    docs = closed.write_docs(candidate_dir, role=role, lens=lens, context=context, live=live, pruned=pruned,
                             incumbent=incumbent, ledger=ledger, attempt_note=attempt_note, critiques=critiques)
    docs = {k: (_api(v) if isinstance(v, str) and k in ("brief", "existing_hypotheses", "menu") else v)
            for k, v in docs.items()}
    for fname, key in (("CANDIDATE_BRIEF.md", "brief"), ("existing_hypotheses.md", "existing_hypotheses"),
                       ("refinement_menu.md", "menu")):
        if docs.get(key):
            (Path(candidate_dir) / fname).write_text(docs[key], encoding="utf-8")
    return docs


def build_prompt(candidate_dir: Path, docs: Mapping[str, Optional[str]]) -> str:
    prompt = closed.build_prompt(candidate_dir, docs)
    old = closed.THEORY_PROMPT.read_text(encoding="utf-8")
    if not prompt.startswith(old):
        raise RuntimeError("the closed brief's prompt changed shape; update src.rsa_open.loop.brief")
    return THEORY_PROMPT.read_text(encoding="utf-8") + prompt[len(old):]


SAMPLING_REJECTIONS = ("did not converge", "did not finish within")
REPARAMETERISE = """

## Making the posterior easier to sample

The rejection is about inference, not about the hypothesis. Before you change the mechanism:
1. Run the self-check with `--quick` (seconds): it shows the best values, which parameters trade off, which
   the data cannot pin down, and whether different starts end at different optima.
2. A pair that trades off: the data pin down a combination of them. Keep one and fix the other, or sample
   the combination (a sum and a difference, a product and a ratio) instead of the two.
3. A parameter the data cannot pin down (a wide Laplace interval, or a singular curvature): fix it, drop it,
   or give it a tighter prior, and say so in hypothesis.md.
4. Sample positive scales on the log scale and weights on the logit scale, not with hard bounds the
   sampler runs into.
5. Several optima: the theory has two near-equivalent explanations of the same data. Break the symmetry
   (order the parameters, fix a sign) or drop one of them.
Smaller steps do not help: the loop refits only near misses."""


def repair_note(reason: str) -> str:
    """The closed loop's repair note; after a rejection for sampling, a reparameterisation checklist
    (PI 2026-10-28: the cheap version of an inference debugger)."""
    note = closed.repair_note(reason)
    return note + REPARAMETERISE if any(k in reason for k in SAMPLING_REJECTIONS) else note


retry_note = closed.retry_note
