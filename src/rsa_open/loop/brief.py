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

PROMPTS = Path(__file__).with_name("prompts")
THEORY_PROMPT = PROMPTS / "open_theory.md"
HANDBOOK = closed.HANDBOOK
LENSES = [lens.replace("click", "response") for lens in closed.PLAIN_RSA_LENSES]
CHECK_COMMAND = sys.executable + " -m src.rsa_open.loop.check_candidate {candidate_dir} --corpus {corpus}"
CHECK_SHELL_TIMEOUT_MS = 1_200_000
NO_NETWORK_NOTE = closed.NO_NETWORK_NOTE

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


def context_md(*, candidate_dir: Path, corpus_path: Path, blocks: Sequence[Block], data_paths: Mapping[str, Path],
               paradigm_notes: Mapping[str, str], round_index: int, n_rounds: int, network: bool = False) -> str:
    lines = ["# Context", "", f"- Candidate directory (write your three files here): `{candidate_dir}`",
             f"- Inner-loop round {round_index + 1} of {n_rounds}.", "", "## The paradigms of this chain", "",
             "Your theory is fitted to all of them at once; the newest is last. The data files are one JSON "
             "object per line (`objects`: one row of 0/1 features per object; `heard`: the index of the word "
             "heard, `null` on a prior trial; `response` in its paradigm's form). You may analyse them from "
             "the shell; your theory file may not read them.", ""]
    for b in blocks:
        resp = RESPONSE_WORDS.get(b.link.response_type, b.link.response_type)
        opts = ", ".join(f"{k} {v}" for k, v in b.link.options.items())
        lines += [f"### {b.name}: {b.n} trials, people {resp}",
                  f"- link `{b.link.name}`{f' ({opts})' if opts else ''}; data `{data_paths[b.name]}`"]
        if paradigm_notes.get(b.name):
            lines.append(f"- {paradigm_notes[b.name]}")
        lines.append("")
    lines += ["## Self-check", "",
              f"`{CHECK_COMMAND.format(candidate_dir=candidate_dir, corpus=corpus_path)}`",
              f"It takes 3-15 minutes (it compiles and fits your theory on every paradigm). Give the shell tool a "
              f"timeout of {CHECK_SHELL_TIMEOUT_MS} milliseconds for it. Run it once per change.", ""]
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


repair_note = closed.repair_note
retry_note = closed.retry_note
