"""One designer cycle (docs/auto_open/PLAN.md §4): the brief, the agent, the harness's checks of
what it submits, the pilot, and the frozen paradigm.

**A cycle** (`run_designer`): the harness lays out a workspace (`prepare_workspace`: the design
context with the floor's reference already scored, the spec's schema, the link library, the
chain's earlier paradigms) and spawns the designer agent (Gemini 4, `DESIGNER_MODEL`) with the
brief (`brief`). The agent works with the tools (`src.rsa_open.tools`) and submits a spec. The
harness then checks the submission itself (`check_submission`), never trusting the agent's
reports:
1. the spec validates and is the one submitted (its sha256);
2. its design is re-scored with a seed of the harness's own, and must meet the floor; this is
   the design that runs;
3. its link is re-checked for recovery in the paradigm, with the harness's seed;
4. its trial lists are built and piloted with LLM participants (`src.rsa_open.pilot`).
A submission that fails any step goes back to the agent with the report, in a new session on the
same workspace (its notes and drafts are still in ``work/``), up to ``MAX_SUBMISSIONS`` (3) per
cycle; a spec that already failed is refused unchanged. A cycle also ends when the agent's
wall-clock budget (``CYCLE_SECONDS``, 2 hours over its sessions) or its tool calls (40) are spent,
or when a session ends without a new submission.

**A failed cycle gets one retry** (PI 2026-10-21), on a fresh workspace with the failure's
reason in the brief; a second failure raises `DesignerFailed` and the chain stops. There is no
fallback design.

**The frozen paradigm** (`freeze`): ``frozen/`` holds the paradigm, its design (the harness's
re-score), the trial lists, the page, the link check, the pilot and ``record.json`` (hashes,
chain, experiment, and ``review_required``: the PI reviews the first paradigm of each chain).
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from src.rsa_open.timing import CLOSED_CAMPAIGN, TimingRecord, digest_md, write_records
from src.rsa_open.tools import MAX_TOOL_CALLS, Workspace, seed_for

DESIGNER_MODEL = "google/gemini-4-argon"
CYCLE_SECONDS = 2 * 3600
MAX_SUBMISSIONS = 3
MAX_CYCLES = 2
MIN_SESSION_SECONDS = 300  # a session shorter than this is not started
TOOL_SHELL_TIMEOUT_MS = 1_800_000  # link-check fits three models; eig can take minutes
TOOLS_COMMAND = sys.executable + " -m src.rsa_open.tools"
SpawnFn = Callable[[Workspace, str, int, Path], bool]  # (workspace, prompt, timeout_sec, log) -> ok


class DesignerFailed(RuntimeError):
    """Both cycles failed: the chain stops (no fallback design)."""


@dataclass(frozen=True)
class EarlierParadigm:
    """An earlier paradigm of the chain, as the designer sees it: the spec and per-theory fits."""

    name: str
    spec_path: Path
    fits_md: str  # a table of each live theory's fit to that paradigm's data (no participant data)


# --------------------------------------------------------------------------- the workspace

def links_md() -> str:
    from src.rsa_open.links import LINKS, likelihood

    lines = ["# The link library", "",
             "Each link scores one response type. Every link has a decision `temperature` (sharpens or "
             "flattens the theory's probabilities) and a `lapse` (a uniform guess); some have more. "
             "Their parameters are fitted jointly with the theories', shared by every theory on the "
             "paradigm. The paradigm fixes the options (scale points, total points).", ""]
    for name, link in LINKS.items():
        lk = likelihood(name, scale_points=5, total_points=100)
        priors = ", ".join(f"`{k}` ~ {type(d).__name__}" for k, d in lk.params.items())
        lines += [f"## `{name}` (response type `{link.response_type}`)", "", link.description, "",
                  f"Parameters: {priors}.", ""]
    lines += ["The EIG tool scores choice, rating, slider (binned: a lower bound), production and "
              "binary_judgment paradigms; it cannot yet score bets (allocation).", ""]
    return "\n".join(lines)


def format_example() -> Dict[str, Any]:
    """A spec that shows the format only: placeholder content, not a suggestion."""
    square = '<rect x="35" y="35" width="30" height="30" fill="#777777"/>'
    return {
        "schema_version": 1, "name": "FORMAT EXAMPLE ONLY", "summary": "Placeholder text showing the format.",
        "charter": {"question": "Placeholder: the question about pragmatic reference this paradigm asks."},
        "instructions": {"welcome": "Placeholder welcome.", "task": "Placeholder task description.",
                         "comprehension_checks": [{"question": "Placeholder question?",
                                                   "options": ["right option", "wrong option"], "correct": 0}]},
        "materials": {"kind": "svg", "domains": [{"name": "things", "base": '<circle cx="50" cy="50" r="40" fill="#dddddd"/>',
                                                  "features": [{"word": w, "svg": square} for w in ("alpha", "beta")]}]},
        "scene_space": {"sizes": [[2, 2], [3, 2]], "prior_queries": False, "synonyms": False},
        "response": {"type": "choice", "link": "choice", "prompt": "Placeholder prompt with the word {message}."},
        "design": {"n_participants": 100, "trials_per_participant": 10, "distinct_trials": 20, "catch_trials": 2,
                   "minutes": 3, "pay_usd": 0.5},
        "content": {"deception": False, "affective_stimuli": False},
    }


def prepare_workspace(root: Path, *, theories: Mapping[str, Tuple[Path, Mapping[str, np.ndarray]]],
                      history: Sequence[EarlierParadigm], timing: Sequence[TimingRecord]) -> Workspace:
    """Lay out a designer workspace; the floor's reference is scored here, once."""
    from src.rsa_open.design import DEFAULT_DRAWS, DEFAULT_SCENARIOS, load_context, reference_report, write_context
    from src.rsa_open.paradigm import Paradigm

    ws = Workspace(Path(root))
    if ws.root.exists():
        raise FileExistsError(f"{ws.root} exists: a workspace is laid out once")
    write_context(ws.context, theories)
    reference_report(load_context(ws.context), n_draws=DEFAULT_DRAWS, n_scenarios=DEFAULT_SCENARIOS, seed=0)
    ws.inputs.mkdir(parents=True)
    (ws.inputs / "paradigm_schema.json").write_text(json.dumps(Paradigm.model_json_schema(), indent=2))
    (ws.inputs / "links.md").write_text(links_md())
    (ws.inputs / "format_example.json").write_text(json.dumps(format_example(), indent=2))
    write_records(ws.timing, timing)  # the tools' time estimates and the pilot's read these
    (ws.inputs / "timing.md").write_text(digest_md(timing))
    hist = ws.inputs / "history"
    hist.mkdir()
    for k, item in enumerate(history):
        shutil.copyfile(item.spec_path, hist / f"{k + 1}_paradigm.json")
        (hist / f"{k + 1}_fits.md").write_text(item.fits_md)
    ws.work.mkdir()
    (ws.root / "checks").mkdir()  # the harness's checks of each submission; the agent may read them
    (ws.root / "logs").mkdir()  # briefs and agent transcripts; the agent may not
    return ws


# --------------------------------------------------------------------------- the brief

CHARTER = """Anchored in Grice (1975) and Goodman & Frank (2016), "Pragmatic language interpretation as
probabilistic inference".
- In scope: how people use a message, its alternatives and the shared context to infer what a
  speaker means (a referent, an implied meaning) or to choose what a speaker should say: tasks in
  which a message is interpreted or chosen given a context both parties know.
- Out of scope: perception or memory limits as the object of study; learning over many trials;
  social evaluation of speakers; anything needing deception or emotionally charged material.
- Every paradigm maps each trial onto the abstract scene: the referents and their features, the
  message (or its absence), the alternatives the speaker had, and what the participant is asked to
  report."""


def brief(ws: Workspace, *, chain: str, experiment: int, hypotheses: Mapping[str, str],
          history: Sequence[EarlierParadigm], feedback: str = "", submissions_left: int = MAX_SUBMISSIONS,
          seconds_left: int = CYCLE_SECONDS) -> str:
    from src.rsa_open.design import FLOOR_SHARE
    from src.rsa_open.paradigm import (BUDGET_USD, MAX_FEATURES, MAX_MINUTES, MAX_OBJECTS, MAX_POOL,
                                       MIN_HOURLY_USD, MIN_PARTICIPANTS)
    from src.rsa_open.pilot import FLOORS, TIME_TOLERANCE

    calls_left = MAX_TOOL_CALLS - len(ws.calls())
    theories = "\n".join(f"- **{name}** (`{ws.context / 'theories' / (name + '.py')}`): {text}"
                         for name, text in hypotheses.items())
    if history:
        earlier = "\n".join(f"- {k + 1}. {item.name}: `{ws.inputs / 'history' / f'{k + 1}_paradigm.json'}`, "
                            f"fits in `{ws.inputs / 'history' / f'{k + 1}_fits.md'}`"
                            for k, item in enumerate(history))
    else:
        earlier = "- None: this is the chain's first experiment."
    tool = f"{TOOLS_COMMAND} {{command}} --workspace {ws.root} --spec {ws.work}/<your spec>.json"
    return f"""# You design the next experiment

You are the experiment designer of chain `{chain}`, experiment {experiment}, in a study of how
people interpret and choose referring expressions. Theories of this ability are judged by how well
they predict people **across paradigms**: a theory fitted to one task can be overfit to that task
(Yarkoni, 2020, "The generalizability crisis"). Your job: design the experiment that matters most
for arriving at theories that generalize, given the earlier paradigms in this chain and the
current theories. You choose the task, the materials, the instructions and the response (any of
the library's response types); nothing requires your paradigm to resemble or differ from earlier
ones. Decide what is most important to learn next, and say why.

## The charter (every paradigm must fit it)

{CHARTER}

## What a paradigm is

A paradigm is one JSON spec: schema `{ws.inputs / 'paradigm_schema.json'}`, a format example (placeholder
content only) `{ws.inputs / 'format_example.json'}`. Every trial is an abstract scene: a few objects
(rows) with binary features (columns); each feature is a word the speaker can say. Your domains
give each feature its word and its drawing (an SVG fragment in a 100 x 100 box, drawn over the
domain's base) or its phrase (text materials). The response types and the links that score them:
`{ws.inputs / 'links.md'}`. The theories are written over abstract scenes and never see your
words or drawings; your materials decide whether people perceive the scene the theories assume.
The page shows the speaker's word in quotation marks where your prompt says `{{message}}`: do not
add your own.

**Materials people can read, and want to look at.** Scenes include objects with no features, and
objects with every feature, so:
- the base must be a complete, recognisable object on its own (a robot with a face and a body,
  not a bare box): it is what an object with no features looks like;
- each feature is a separate, nameable part, clearly visible at card size (about 180 px), that does
  not look like part of the base or like another feature, and that can be added to the base
  alongside any other features without hiding them;
- each word is a common, concrete word that a person would use for that part of the drawing;
- make them engaging: colour and a little character, nothing cluttered or abstract.
The pilot checks this: its participants are shown each domain's base alone and asked what it is,
and the base with each single feature and asked which word names the added part (see "After you
submit").

The harness builds the page from your spec: one list per participant (balanced over your design),
a practice trial, your comprehension checks (shown again once after a wrong answer), catch trials
with one right answer, objects in a random screen order, words mapped to features at random per
trial. You write no page code.

## The current theories

Their files (read-only) are in `{ws.context / 'theories'}`, their posterior draws on the data so far in
`{ws.context / 'draws'}`.
{theories}

## Earlier paradigms in this chain

{earlier}

## Bounds

- Budget: ${BUDGET_USD:.2f} per experiment (participants x pay x 4/3, Prolific's fee). At least
  {MIN_PARTICIPANTS} participants, at most {MAX_MINUTES} minutes each, at least ${MIN_HOURLY_USD:.2f} per hour.
- Scenes: 2-{MAX_OBJECTS} objects, 2-{MAX_FEATURES} features, at most {MAX_POOL} scenes in the space.
- Content (IRB): no deception, no affectively charged stimuli or language, SVG or text materials
  only, no links or requests for personal information.

## Your tools ({calls_left} calls left; each command is one call)

    {tool.format(command="validate")}     # every problem with the spec, at once; the time estimate
    {tool.format(command="eig")}          # bits per dollar under the theories, the floor; the time estimate
    {tool.format(command="eig")} --quick  # the same, approximate, in a fraction of the time: for exploring
    {tool.format(command="link-check")}   # can the link's parameters be recovered in this paradigm?
    {tool.format(command="submit")}       # submit (needs a passing eig and link-check of this exact spec)

- Give the shell tool a timeout of {TOOL_SHELL_TIMEOUT_MS} ms for `eig` and `link-check`: `eig` takes
  5-15 minutes, `link-check` fits three models (5-20 minutes). With the default timeout they are
  killed before they print anything. Compare candidate paradigms with `eig --quick` (1-2
  minutes) and run the full `eig` on the one you mean to submit.
- `link-check` simulates from the live theory with the most parameters; `--theory <name>` names
  another. A theory that cannot move a link parameter (the literal listener is flat over the
  objects a word is true of, so a listener link's temperature changes nothing) cannot recover it.
- **Time:** how long people took in finished experiments, and how many trials fit in a minute:
  `{ws.inputs / 'timing.md'}`. `validate` and `eig` report the pilot's time estimate for your spec
  against your declared minutes; the pilot's check uses the same estimate.
- **The floor:** `eig` reports your paradigm's expected information gain about the theories per
  dollar; `submit` needs at least {FLOOR_SHARE:.0%} of what the closed campaign's forced-choice
  paradigm gives under the same theories.
- Every report is saved in `{ws.reports}`. Keep notes in `{ws.work}/notes.md`: a later session of this
  cycle starts with no memory but your files.

## After you submit

The harness re-scores your spec with its own seed (the design that runs is its), re-checks the
link, builds the page and pilots it with LLM participants. The pilot passes when at least
{FLOORS['comprehension']:.0%} of them understand the instructions (by their second reading), at least
{FLOORS['catch']:.0%} of catch trials are passed, {FLOORS['readable']:.0%} of answers are usable, the data parse, its
time estimate is within {TIME_TOLERANCE:.0%} of your declared minutes, and (SVG materials) at least {FLOORS['legibility']:.0%} of
its materials questions are answered right, with no base or feature misread by half of them. If anything fails you get the report
and may revise: {submissions_left} submission(s) left in this cycle, about {seconds_left // 60} minutes of your time.

## What to write

1. Your spec(s) in `{ws.work}`; submit one.
2. `{ws.work}/rationale.md`: what this experiment is for, which disagreements among the theories it
   targets, why this task and response type, and what you expect each theory to predict.

You have no internet access. Work only in `{ws.work}`.
{feedback}"""


# --------------------------------------------------------------------------- the harness's checks

@dataclass
class SubmissionResult:
    sha256: str
    passed: bool
    problems: List[str]
    dir: Path
    details: Dict[str, Any] = field(default_factory=dict)

    def feedback(self) -> str:
        lines = [f"Submission {self.sha256[:12]} did not pass:"] + [f"- {p}" for p in self.problems]
        return "\n".join(lines)


def check_submission(ws: Workspace, submitted: Path, out: Path, *, pilot_participants: int,
                     participant_factory=None) -> SubmissionResult:
    """The harness's own checks of a submission record (validation, floor, link, pilot)."""
    from src.rsa_open.paradigm import ParadigmInvalid, load_paradigm
    from src.rsa_open.pilot import run_pilot
    from src.rsa_open.screens import design_units, trial_lists
    from src.rsa_open.tools import eig_report, link_check_report

    record = json.loads(Path(submitted).read_text())
    out.mkdir(parents=True)
    spec = out / "paradigm.json"
    shutil.copyfile(record["spec"], spec)
    sha = record["sha256"]
    try:
        paradigm = load_paradigm(spec)
    except ParadigmInvalid as err:
        return SubmissionResult(sha, False, [f"the spec is not valid: {err}"], out)
    if paradigm.sha256 != sha:
        return SubmissionResult(sha, False, ["the spec changed after it was submitted (its sha256 differs)"], out)
    eig = eig_report(ws, spec, seed=seed_for(sha, "harness_eig"))
    (out / "eig.json").write_text(json.dumps(eig, indent=2))
    if not eig["meets_floor"]:
        return SubmissionResult(sha, False, [
            f"below the floor on the harness's re-score: {eig['bits_per_dollar']:.5f} bits/$ against "
            f"{eig['floor_bits_per_dollar']:.5f}"], out, {"eig": eig})
    link = link_check_report(ws, spec, seed=seed_for(sha, "harness_link_check"))
    (out / "link_check.json").write_text(json.dumps(link, indent=2))
    if not (link["converged"] and link["recovered"]):
        return SubmissionResult(sha, False, [
            f"the link check failed on the harness's re-run: converged={link['converged']}, "
            f"misses={link['misses']}"], out, {"eig": eig, "link": link})
    units = design_units(eig)
    lists = trial_lists(paradigm, units, seed=seed_for(sha, "lists"))
    (out / "lists.json").write_text(json.dumps(lists))
    pilot = run_pilot(paradigm, lists, out / "pilot", n_participants=pilot_participants,
                      participant_factory=participant_factory, timing=ws.timing_records())
    if not pilot["passed"]:
        return SubmissionResult(sha, False, [f"pilot: {p}" for p in pilot["problems"]], out,
                                {"eig": eig, "link": link, "pilot": pilot})
    return SubmissionResult(sha, True, [], out, {"eig": eig, "link": link, "pilot": pilot, "lists": lists,
                                                 "paradigm": paradigm, "units": units})


def _sha_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze(result: SubmissionResult, out: Path, *, chain: str, experiment: int,
           code_commit: Optional[str]) -> Dict[str, Any]:
    """Write the paradigm that passed, and its record, to ``out`` (once)."""
    from src.rsa_open.page import render
    from src.rsa_open.screens import design_sha256

    if out.exists():
        raise FileExistsError(f"{out} exists: a paradigm is frozen once")
    out.mkdir(parents=True)
    d = result.details
    paradigm = d["paradigm"]
    (out / "paradigm.json").write_text(json.dumps(paradigm.model_dump(mode="json"), indent=2, ensure_ascii=False))
    (out / "design.json").write_text(json.dumps({"units": d["units"], "eig": d["eig"]}, indent=2))
    (out / "lists.json").write_text(json.dumps(d["lists"]))
    (out / "page_preview.html").write_text(render(d["lists"], preview=True))
    (out / "link_check.json").write_text(json.dumps(d["link"], indent=2))
    shutil.copytree(result.dir / "pilot", out / "pilot")
    record = {
        "chain": chain, "experiment": experiment, "paradigm_sha256": paradigm.sha256,
        "design_sha256": design_sha256(d["units"]), "lists_sha256": _sha_file(out / "lists.json"),
        "context_sha256": d["eig"]["context_sha256"], "joint_eig_bits": d["eig"]["joint_eig_bits"],
        "bits_per_dollar": d["eig"]["bits_per_dollar"], "floor_bits_per_dollar": d["eig"]["floor_bits_per_dollar"],
        "pilot": {k: d["pilot"][k] for k in ("comprehension_final", "catch_pass", "readable", "estimated_minutes",
                                             "participant_model")},
        "review_required": experiment == 1, "frozen_at": time.time(), "code_commit": code_commit,
    }
    (out / "record.json").write_text(json.dumps(record, indent=2))
    return record


# --------------------------------------------------------------------------- a cycle, and the retry

def run_designer(root: Path, *, chain: str, experiment: int,
                 theories: Mapping[str, Tuple[Path, Mapping[str, np.ndarray]]], hypotheses: Mapping[str, str],
                 history: Sequence[EarlierParadigm], spawn: SpawnFn, pilot_participants: int = 12,
                 participant_factory=None, code_commit: Optional[str] = None,
                 cycle_seconds: int = CYCLE_SECONDS,
                 timing: Sequence[TimingRecord] = CLOSED_CAMPAIGN) -> Dict[str, Any]:
    """Design one experiment: up to ``MAX_CYCLES`` cycles; returns the frozen record. ``timing``
    is every finished experiment's completion times (`src.rsa_open.timing`): the closed
    campaign's, plus this campaign's from every chain."""
    root = Path(root)
    failures: List[str] = []
    for cycle in range(MAX_CYCLES):
        ws = prepare_workspace(root / f"cycle_{cycle + 1}", theories=theories, history=history, timing=timing)
        feedback = ""
        if failures:
            feedback = ("\n## The previous attempt failed\n\nA previous designer cycle for this experiment "
                        f"failed: {failures[-1]}\nThis is the last attempt.\n")
        spent, failed_shas = 0.0, set()
        reason = None
        for k in range(MAX_SUBMISSIONS):
            left = int(cycle_seconds - spent)
            if left < MIN_SESSION_SECONDS:
                reason = "the designer's time for the cycle was spent without a passing submission"
                break
            if len(ws.calls()) >= MAX_TOOL_CALLS:
                reason = "the designer's tool calls for the cycle were spent without a passing submission"
                break
            prompt = brief(ws, chain=chain, experiment=experiment, hypotheses=hypotheses, history=history,
                           feedback=feedback, submissions_left=MAX_SUBMISSIONS - k, seconds_left=left)
            (ws.root / "logs" / f"brief_{k + 1}.md").write_text(prompt)
            started = time.monotonic()
            ok = spawn(ws, prompt, left, ws.root / "logs" / f"agent_{k + 1}.jsonl")
            took = time.monotonic() - started
            spent += took
            if not ws.submission.exists():
                if took >= left - 5:
                    reason = (f"the designer's session reached its time limit ({left // 60} min) without a "
                              f"submission")
                elif not ok:
                    reason = "the designer's session failed (the agent exited with an error) without a submission"
                else:
                    reason = "the designer's session ended without a submission"
                break
            submitted = ws.root / "submissions" / f"{k + 1}.json"
            submitted.parent.mkdir(exist_ok=True)
            os.replace(ws.submission, submitted)  # the next session must submit anew
            sha = json.loads(submitted.read_text())["sha256"]
            if sha in failed_shas:
                result = SubmissionResult(sha, False, ["this exact spec already failed; it was not checked again"],
                                          ws.root / "checks" / f"{k + 1}")
            else:
                result = check_submission(ws, submitted, ws.root / "checks" / f"{k + 1}",
                                          pilot_participants=pilot_participants,
                                          participant_factory=participant_factory)
            if result.passed:
                return freeze(result, root / "frozen", chain=chain, experiment=experiment, code_commit=code_commit)
            failed_shas.add(sha)
            where = f" Its reports (and the pilot's transcripts) are in `{result.dir}`." if result.dir.exists() else ""
            feedback = (f"\n## Your last submission did not pass\n\n{result.feedback()}\n\n"
                        f"Revise and submit again.{where}\n")
            reason = f"no submission passed the harness's checks; the last: {'; '.join(result.problems)}"
        failures.append(reason or "no passing submission")
        (ws.root / "failure.txt").write_text(failures[-1])
    raise DesignerFailed(f"chain {chain}, experiment {experiment}: both designer cycles failed:\n" +
                         "\n".join(f"- cycle {i + 1}: {r}" for i, r in enumerate(failures)))


def designer_spawner(*, agent_root: Path, backend: Optional[str] = "opencode", model: Optional[str] = DESIGNER_MODEL,
                     sandbox: bool = True, network: bool = False, shell_dir: Optional[Path] = None) -> SpawnFn:
    """The real spawner: one sandboxed coding agent session per call, reading the context and
    inputs, writing only ``work/``; without network as the closed loop's agents
    (`src.rsa.loop.no_network`)."""
    from src.runtime.coding_agent import run_coding_agent, select_backend

    env = None
    if not network:
        from src.rsa.loop import no_network

        if select_backend(backend) != "opencode":
            raise ValueError("agents without network are implemented for the opencode backend only")
        if shell_dir is None:
            raise ValueError("agents without network need shell_dir for their shell wrapper")
        bash = shutil.which("bash")
        if bash is None:
            raise FileNotFoundError("no bash on PATH for the agents' shell wrapper")
        wrapper = no_network.write_shell_wrapper(Path(shell_dir), sys.executable, bash)
        no_network.verify_wrapper(wrapper, sys.executable)
        env = dict(os.environ)
        env["SHELL"] = str(wrapper)
        env["OPENCODE_PERMISSION"] = no_network.opencode_permission(env.get("OPENCODE_PERMISSION"))

    def spawn(ws: Workspace, prompt: str, timeout_sec: int, log: Path) -> bool:
        ok, _ = run_coding_agent(prompt, cwd=agent_root, log_path=log,
                                 allowed_dirs=[ws.context, ws.inputs, ws.root / "checks"],
                                 writable_dirs=[ws.work], timeout_secs=timeout_sec, backend=backend, model=model,
                                 usage_label="rsa_open:designer", stock=True, sandbox=sandbox, env=env)
        if not network:
            from src.rsa.loop.no_network import web_tool_uses

            used = web_tool_uses(log)
            if used:
                raise RuntimeError(f"{ws.root}: the designer used the web although its web tools are denied "
                                   f"(a harness bug): {used[:3]}")
        return ok

    return spawn
