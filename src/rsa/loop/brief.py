"""Candidate briefs and prompts for the RSA loop's agents.

Every slot gets the same documents as the PyMC domain's loop (CONTEXT.md,
CANDIDATE_BRIEF.md, existing_hypotheses.md, and attempted_hypotheses.md for
an exploratory slot or refinement_menu.md for a refinement slot), written
into its candidate directory and inlined into its prompt, plus the memo
Handbook. Slot roles and the lens walk are the PyMC loop's
(`src.pipelines.inner_loop.model_zoo.slot_roles`).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

from src.pipelines.inner_loop.hypothesis_ledger import HypothesisLedger
from src.pipelines.inner_loop.model_zoo import (
    SLOT_EXPLORE,
    SLOT_REFINE_CHOSEN,
    SLOT_REFINE_INCUMBENT,
)

PROMPTS = Path(__file__).with_name("prompts")
THEORY_PROMPT = PROMPTS / "rsa_theory.md"
HANDBOOK = PROMPTS / "memo_handbook.md"

# The exploration lenses, walked by exploratory slots (one per slot).
DEFAULT_RSA_LENSES = [
    "Propose a mechanism from a genuinely different family than anything in "
    "the current set: one the current models cannot express, not a variant.",
    "Propose a single mechanism whose predictions would disagree most sharply "
    "with the current best model on some display, and say in your hypothesis "
    "which display and why.",
    "Model the speaker's costs or alternatives: which words a speaker would "
    "consider, how costly each is, or what a speaker who cannot name something "
    "does, and let the listener reason about that speaker.",
    "Model what the listener expects before hearing anything: a prior over "
    "objects grounded in a psychological account (salience, typicality, what "
    "the speaker is likely to talk about), used in a principled place.",
    "Propose a resource-limited listener: a bounded reasoning depth, an "
    "approximate or sampled inference, or limited attention to the display, "
    "as the single mechanism.",
    "Propose that the meaning of the words is uncertain or soft: lexical "
    "uncertainty, graded or noisy truth values, or a listener who is unsure "
    "which feature a word picks out.",
    "Explain the framing manipulations: why describing a 'favorite' or 'least "
    "favorite' object, or showing one object in colour, changes the choice, "
    "with one mechanism that also covers the neutral trials.",
    "Propose a non-Bayesian heuristic listener (e.g. pick the object with the "
    "fewest features the word is true of, or the most distinctive one) that "
    "is simple and could be wrong loudly.",
    "Propose a mechanism at the level of the decision rule: how people turn "
    "beliefs into a click (probability matching, a softmax over beliefs, a "
    "structured lapse), with the inference itself kept simple.",
    "Model the speaker's goal or question under discussion: the speaker is "
    "informative about some aspect of the object rather than its identity, "
    "and the listener reasons about that goal.",
    "Propose that listeners differ: a mixture of a small number of listener "
    "types (e.g. literal and pragmatic) as ONE hypothesis about population "
    "structure, with the type proportions as parameters.",
]

# The live phase studies plain displays only (PI 2026-10-10): no lens asks
# for framing or colour mechanisms, which its experiments cannot test.
PLAIN_RSA_LENSES = [lens for lens in DEFAULT_RSA_LENSES if not lens.startswith("Explain the framing")]
PLAIN_SCOPE_NOTE = (
    "- **Scope: plain displays.** Every trial in the responses and every display the next "
    "experiments can show is plain: valence 0 (no favourite / least favourite framing), no "
    "familiarization and no greyscale. Mechanisms that act through those fields cannot be "
    "tested or rewarded here; spend your proposal on how listeners reason about the words "
    "and the display.\n"
    "- **What the next experiments show** (PI 2026-10-10). The displays are chosen by "
    "expected information over the current models, within fixed quotas (of 40 displays, at "
    "least 6 with two objects, 10 with three, 8 mumble trials and 20 with a word); the rest "
    "go where the models disagree most, which is often the largest displays (4 x 4). Sizes "
    "run from 2 objects x 2 features to 4 x 4, and two features may be on exactly the same "
    "objects (e.g. two objects sharing three features, one of them with a fourth): to a "
    "participant these are different things on the objects, and to a speaker two words, "
    "either of which could be said, so a redundant feature changes what a speaker who weighs "
    "her alternatives would say. The existing data are mostly 3-object displays without "
    "redundant features, so a 4 x 4 display, a two-object display or a redundant feature asks "
    "a model to extrapolate: a mechanism that only fits the familiar displays, or behaves "
    "oddly as displays grow, will be found out there. Make your model's predictions sensible "
    "across display sizes and feature overlaps, and say in your hypothesis how it scales.\n"
    "- **The speaker's alternatives are yours to propose.** Which utterances a speaker "
    "considers is part of a hypothesis: for instance, saying nothing, or a word for a "
    "feature no object has, as a way out when every available word is ambiguous. Such a "
    "speaker uses an ambiguous word less often, which changes what a listener infers from "
    "one. (A display never shows a feature no object has, and the word heard is always "
    "true of some object.)\n"
    "- **Mumble trials** can be part of any experiment (they are in the pool the displays "
    "are chosen from, and are chosen where the models' priors disagree): the speaker's "
    "word is not heard "
    "(`query` is `prior`, `is_prior` in the model's context), and the click shows which "
    "object the listener expected the speaker to mean before any word: the prior over "
    "referents (salience, distinctiveness, what speakers tend to talk about). A model's "
    "prior is half of its account of the word trials too, so it is worth getting right; "
    "the data have such trials (pragmods, Sikos 2021) to learn it from.\n"
)

# The harness's own interpreter, as in the PyMC loop: sandboxed agents have no
# `uv` (it lives in ~/.local/bin, which the private home hides) and `uv run`
# would sync the read-only venv; the 2026-10-07 smoke test found the `uv run`
# form unrunnable in the sandbox.
CHECK_COMMAND = sys.executable + " -m src.rsa.loop.check_candidate {candidate_dir} --responses {responses}"
# The shell timeout agents must give the self-check: its fit may take up to
# CHECK_TIME_LIMIT_SEC (10 min) after loading and the contract. opencode's
# shell tool kills a command after 120 s by default, which in the 2026-10-06
# smoke test killed every self-check before it printed anything.
CHECK_SHELL_TIMEOUT_MS = 900_000


@dataclass
class ZooModel:
    name: str
    hypothesis: str
    source: Path
    standing: str  # e.g. "best" or "12.3 ± 4.0 nats behind the best"


NO_NETWORK_NOTE = """- You have no internet access: shell commands cannot open network connections and
  the web tools are disabled. Everything you need is in this repository, your
  candidate directory and the responses file; do not try to install packages.
"""


def context_md(*, candidate_dir: Path, responses_path: Path, round_index: int,
               n_rounds: int, n_trials: int, experiments: Sequence[str], network: bool = False,
               scope_note: str = "") -> str:
    return f"""# Context

- Candidate directory (write your three files here): `{candidate_dir}`
- Responses (read-only): `{responses_path}`: {n_trials} forced-choice trials,
  one row per click, from experiments {", ".join(experiments)}. Columns
  include `experiment`, `condition`, `objects` (JSON list of 0/1 feature
  lists, one per object), `feature_names`, `query` (`utterance` or
  `prior`), `utterance` (feature index heard), `familiarization`,
  `grayscale`, `framing`, `choice` (object index clicked), `participant_id`
  and `trial_index`. You may analyse it (e.g. with pandas, from the shell) to
  find what the current models miss; your model file may not read it.
- Inner-loop round {round_index + 1} of {n_rounds}.
- Self-check: `{CHECK_COMMAND.format(candidate_dir=candidate_dir, responses=responses_path)}`
  It takes about 1-3 minutes (it compiles and fits your model). Give the shell
  tool a timeout of {CHECK_SHELL_TIMEOUT_MS} milliseconds for it (the tool's
  `timeout` parameter); with the default of 120 s the check is killed before it
  prints anything. Run it once and read its result; do not re-run it unchanged.
- Do not refit the models in the set: their standings above are from the
  loop's own fits. Write memo code in a `.py` file (memo reads its source
  back, so `python -c` raises "couldn't find your memo source code").
{scope_note}{"" if network else NO_NETWORK_NOTE}- Write and test your files **in your candidate directory only**: no drafts in
  `/tmp`, the repository root or anywhere else. The loop reads only
  `{candidate_dir}`; a candidate.py anywhere else counts as no candidate.
"""


def existing_md(models: Sequence[ZooModel]) -> str:
    lines = ["# Models in the set", ""]
    for m in models:
        lines += [f"## {m.name} ({m.standing})", "", m.hypothesis.strip(), "", f"Source: `{m.source}`", ""]
    return "\n".join(lines)


def menu_md(live: Sequence[ZooModel], pruned: Sequence[ZooModel], incumbent: str) -> str:
    lines = [
        "# Refinement menu",
        "",
        "Choose ONE model to refine (not the incumbent). Your hypothesis must name it "
        "and state the single change you make. Live models first, best first, then "
        "models pruned earlier, closest to the best first (a pruned mechanism may "
        "come back with a substantive change).",
        "",
    ]
    for m in [x for x in live if x.name != incumbent] + list(pruned):
        lines += [f"## {m.name} ({m.standing})", "", m.hypothesis.strip(), "", f"Source: `{m.source}`", ""]
    return "\n".join(lines)


def brief_md(role: str, *, lens: Optional[str], incumbent: Optional[ZooModel]) -> str:
    if role == SLOT_EXPLORE:
        return (
            "# This slot: exploratory\n\n"
            "Propose a NEW mechanism, not a refinement of a model in the set and not "
            "anything in attempted_hypotheses.md.\n\n"
            f"Lens for this slot: {lens}\n"
        )
    if role == SLOT_REFINE_INCUMBENT:
        assert incumbent is not None
        return (
            "# This slot: refine the incumbent\n\n"
            f"Improve the current best model, **{incumbent.name}** ({incumbent.standing}), "
            "with ONE stated change to its mechanism (a different functional form, prior, "
            "depth, or one component taken from another model). Your hypothesis must say "
            "what you changed and why it should fit better, and list EVERY difference "
            "between your candidate.py and its source: recursion depth (which L<k> "
            "choice_probs calls), each parameter added or removed, and every other "
            "term. A hypothesis that misdescribes its code misleads every later round.\n\n"
            f"Its hypothesis: {incumbent.hypothesis.strip()}\n\nIts source: `{incumbent.source}`\n"
        )
    if role == SLOT_REFINE_CHOSEN:
        return (
            "# This slot: refine a model of your choice\n\n"
            "Pick one model from refinement_menu.md (any but the incumbent"
            + (f", {incumbent.name}" if incumbent else "")
            + ") and improve it with ONE stated change. Name the model you refine in your "
            "hypothesis and list EVERY difference between your candidate.py and its "
            "source: recursion depth (which L<k> choice_probs calls), each parameter "
            "added or removed, and every other term.\n"
        )
    raise ValueError(f"unknown slot role {role!r}")


def write_docs(
    candidate_dir: Path,
    *,
    role: str,
    lens: Optional[str],
    context: str,
    live: Sequence[ZooModel],
    pruned: Sequence[ZooModel],
    incumbent: Optional[str],
    ledger: HypothesisLedger,
    attempt_note: Optional[str] = None,
) -> Dict[str, Optional[str]]:
    candidate_dir.mkdir(parents=True, exist_ok=True)
    inc = next((m for m in live if m.name == incumbent), None)
    docs: Dict[str, Optional[str]] = {
        "context": context,
        "brief": brief_md(role, lens=lens, incumbent=inc),
        "existing_hypotheses": existing_md(live),
        "attempted": ledger.render_markdown([m.name for m in live]) if role == SLOT_EXPLORE else None,
        "menu": menu_md(live, pruned, incumbent or "") if role == SLOT_REFINE_CHOSEN else None,
        "attempt_note": attempt_note,
    }
    files = {
        "CONTEXT.md": "context", "CANDIDATE_BRIEF.md": "brief",
        "existing_hypotheses.md": "existing_hypotheses", "attempted_hypotheses.md": "attempted",
        "refinement_menu.md": "menu",
    }
    for fname, key in files.items():
        if docs[key]:
            (candidate_dir / fname).write_text(docs[key], encoding="utf-8")
    (candidate_dir / "memo_handbook.md").write_text(HANDBOOK.read_text(encoding="utf-8"), encoding="utf-8")
    return docs


def build_prompt(candidate_dir: Path, docs: Mapping[str, Optional[str]]) -> str:
    sections: List[str] = [
        THEORY_PROMPT.read_text(encoding="utf-8"),
        "---",
        f"Your working directory for this candidate is `{candidate_dir}`.\n"
        "Write `hypothesis.md`, `model_name.txt` and `candidate.py` there. Your shell "
        "runs from the repository root, NOT the candidate directory, so always write "
        "to the absolute paths:\n"
        f"  {candidate_dir}/hypothesis.md\n  {candidate_dir}/model_name.txt\n"
        f"  {candidate_dir}/candidate.py\n\n"
        "Use the bash tool with a heredoc to create each file, e.g.\n"
        f"```bash\ncat << 'EOF' > {candidate_dir}/hypothesis.md\nYour hypothesis.\nEOF\n```\n"
        "Then run the self-check in CONTEXT.md and fix what it reports.",
    ]
    if docs.get("attempt_note"):
        sections.append(str(docs["attempt_note"]))
    sections += [
        f"## CONTEXT.md\n\n{docs['context']}",
        f"## CANDIDATE_BRIEF.md\n\n{docs['brief']}",
        f"## existing_hypotheses.md\n\n{docs['existing_hypotheses']}",
    ]
    if docs.get("attempted"):
        sections.append(f"## attempted_hypotheses.md\n\n{docs['attempted']}")
    if docs.get("menu"):
        sections.append(f"## refinement_menu.md\n\n{docs['menu']}")
    sections.append(f"## memo_handbook.md\n\n{HANDBOOK.read_text(encoding='utf-8')}")
    return "\n\n".join(sections) + "\n"


def repair_note(reason: str) -> str:
    return (
        "## REPAIR: your previous attempt was rejected\n\n"
        "Your files from the previous attempt are in this directory. The pipeline "
        f"rejected them for this reason:\n\n> {reason}\n\n"
        "Fix exactly that (keep the hypothesis unless the reason shows it cannot work), "
        "run the self-check, and write the corrected files. This is the final attempt."
    )


def retry_note() -> str:
    return (
        "## RETRY: the previous attempt wrote no candidate.py\n\n"
        "Write all three files to the absolute paths above this time."
    )
