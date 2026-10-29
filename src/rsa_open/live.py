"""Live collection for the open campaign's chains, both arms (docs/auto_open/PLAN.md Phase 3b, item 6).

An experiment's frozen paradigm (``design/frozen``) is deployed, recruited on Prolific and
collected; its data become the same rows and block a simulated experiment writes
(`src.rsa_open.chain.Chain.collect`). Most of the machinery is the closed campaign's and main's:

- **the page** is the open player (`src.rsa_open.page`) with the closed live hooks
  (`src.rsa.live.site.LIVE_JS`: the list from ``/assign``, the data to ``/submit``, then
  Prolific's completion URL). The deployment injects the IRB consent gate;
- **the deployment** is main's (`run_deployment`), under the closed campaign's Prolific project
  (``rsa_reference``), so the project-scoped blocklist excludes everyone from any earlier RSA
  study, closed or open, and admits the subjective-randomness studies' people
  (`PREREG_DRAFT.md` §4b). A resumed experiment never deploys or publishes twice (main's guard);
- **the study's terms are the paradigm's**: its N, its pay and its minutes
  (``recruitment_overrides``), under one neutral name and description for both arms
  (``scripts/rsa/live/open_live.yaml``), so the listing does not say which arm a study is;
- **one study recruits at a time** across the campaign (`study_slot`, a lock in the campaign
  directory that every chain's job shares): Prolific applies the blocklist when a study is
  published, so two studies open at once could share a person. Chains publish in the order
  they are ready, which interleaves the arms;
- **participant ids** are the campaign's (`campaign_ids`: ``p<k>`` from one map of Prolific ids
  in the campaign directory). A person seen in an earlier study keeps that id and is excluded
  here, on record (the blocklist should make this impossible; this is the check);
- **exclusions** (`exclusions`): a missed catch trial (the closed campaign's rule), failed
  comprehension checks on the second reading, a repeat person, or an incomplete record;
- **monitoring** (`monitor_report`): while the study fills and at the end, the share of people
  answering the same way on every test trial, for every response type (a screen position for a
  choice, the same rating, slider value, allocation, word or yes/no), comprehension, catch
  trials, exclusions and times. ``python -m src.rsa_open.monitor`` prints it for a running study.

Raw responses carry Prolific ids: they go to ``private_dir`` (outside every agents' tree, never
copied off Sherlock), as does the id map. The run directory gets only the rows, aggregates and the
deployment manifest (study ids, not participant ids).

Live recruitment is double-gated as everywhere: ``prolific_mode="live"`` needs
``confirm_live_recruitment`` (and the launcher a typed yes). An open chain's first paradigm is
deployed live only once the PI has approved it (`src.rsa_open.review`).
"""

from __future__ import annotations

import base64
import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import statistics
import time
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import yaml

PROJECT_ID = "rsa_reference"
"""The Prolific project of every RSA study, closed and open: the blocklist's scope."""
OPEN_LIVE_CONFIG = Path(__file__).resolve().parents[2] / "scripts" / "rsa" / "live" / "open_live.yaml"
_INSERT_BEFORE = "  <script>\n    // ---- Injected by `python -m src.rsa_open.page`"
DEGENERATE_SHARE = 0.9
"""A person is flagged when at least this share of their test answers are the same answer."""
DEGENERATE_MIN_TRIALS = 6
FAST_MS = 500
"""A person is flagged when their median test-trial response time is below this."""
MONITOR_EVERY_SEC = 600


class AwaitingReview(RuntimeError):
    """An open chain's first paradigm has not been approved by the PI: nothing is published."""


@dataclass(frozen=True)
class LiveConfig:
    prolific_mode: str
    """"test": deploy the page and a Prolific draft to preview, then stop; "live": publish (paid)."""
    confirm_live_recruitment: bool
    firebase_project: str
    run_label: str
    """This chain's label on its study and Hosting path (e.g. "open-c1")."""
    repo_root: Path
    """The checkout that stages and deploys the page (the job's deploy copy)."""
    private_dir: Path
    """This chain's private directory: raw responses with Prolific ids."""
    campaign_dir: Path
    """Shared by every chain of the campaign: the participant-id map and the study slots."""
    name: str
    description: str
    """May use {minutes}."""
    review_first_paradigm: bool = True
    """An open chain's experiment 1 is published only after the PI's approval (PI 2026-10-26)."""
    max_concurrent_studies: int = 1
    max_wait_sec: Optional[float] = None
    """How long a study may take to fill before it is paused (default: main's 3 h)."""
    participants: Optional[int] = None
    """Recruit this many instead of the paradigm's N: a smoke test, never modelled."""
    collection_owner: str = "auto-psych"
    firebase_region: str = "us-central1"

    def __post_init__(self):
        if self.prolific_mode not in ("test", "live"):
            raise ValueError(f"prolific_mode must be 'test' or 'live', not {self.prolific_mode!r}")
        if self.prolific_mode == "live" and not self.confirm_live_recruitment:
            raise ValueError("prolific_mode 'live' publishes a paid Prolific study: it needs "
                             "confirm_live_recruitment as well")
        if self.max_concurrent_studies < 1:
            raise ValueError("max_concurrent_studies must be at least 1")


def load_live_config(path: Path = OPEN_LIVE_CONFIG, **values) -> LiveConfig:
    """The committed study text and limits (``open_live.yaml``) with this job's values."""
    cfg = yaml.safe_load(Path(path).read_text())
    study = cfg["study"]
    kw = dict(name=study["name"], description=study["description"],
              max_concurrent_studies=int(cfg.get("max_concurrent_studies", 1)),
              review_first_paradigm=bool(cfg.get("review_first_paradigm", True)))
    if cfg.get("max_wait_hours") is not None:
        kw["max_wait_sec"] = float(cfg["max_wait_hours"]) * 3600
    kw.update(values)
    return LiveConfig(**kw)


# --------------------------------------------------------------------------- the study's terms

def recruitment_overrides(paradigm, cfg: LiveConfig) -> Dict[str, Any]:
    """The study's pay, length, name and description: the paradigm's terms, the campaign's text."""
    minutes = int(math.ceil(paradigm.design.minutes))
    cents = int(round(paradigm.design.pay_usd * 100))
    if cents <= 0 or minutes < 1:
        raise ValueError(f"the paradigm pays {cents} cents for {minutes} minutes")
    return {"reward": cents, "estimated_completion_time": minutes, "name": cfg.name,
            "description": cfg.description.format(minutes=minutes)}


# --------------------------------------------------------------------------- the page

DATA_URI = re.compile(r"data:image/(png|jpeg|gif|svg\+xml);base64,([A-Za-z0-9+/=]+)")
LIST_URL = "lists/{k}.json"


def _assets_out(text: str, assets: Path) -> str:
    """Every embedded base64 image in ``text`` as a file under ``assets`` (named by its content),
    referenced relative to the page."""
    def one(m: "re.Match") -> str:
        ext = {"png": "png", "jpeg": "jpg", "gif": "gif", "svg+xml": "svg"}[m.group(1)]
        data = base64.b64decode(m.group(2))
        name = f"{hashlib.sha256(data).hexdigest()[:16]}.{ext}"
        path = assets / name
        if not path.exists():
            path.write_bytes(data)
        return f"assets/{name}"

    return DATA_URI.sub(one, text)


def build_site(lists: Mapping[str, Any], exp_dir: Path) -> Path:
    """``<exp_dir>/experiment/``: the open page with the closed live hooks and no lists of its own,
    one file per participant's list (``lists/<k>.json``, fetched after assignment), and every embedded
    picture as a file once (``assets/``). The page has no consent screen of its own (the deployment
    injects the IRB gate). Run C: a page carrying every list was 360 MB for the fixed arm, whose
    pictures were inlined in every trial of every list."""
    from src.rsa.live.site import LIVE_JS, LIVE_MARKER
    from src.rsa_open.page import CONSENT_PLACEHOLDER_MARKER, render

    site = Path(exp_dir) / "experiment"
    if site.exists():
        shutil.rmtree(site)
    (site / "lists").mkdir(parents=True)
    (site / "assets").mkdir()
    for lst in lists["lists"]:
        text = _assets_out(json.dumps(lst, ensure_ascii=False), site / "assets")
        (site / LIST_URL.format(k=lst["list_index"])).write_text(text, encoding="utf-8")
    ks = sorted(lst["list_index"] for lst in lists["lists"])
    if ks != list(range(len(ks))):
        raise ValueError("the lists are not numbered 0..n-1")
    meta = {**lists, "lists": [], "n_lists": len(ks), "list_url": LIST_URL}
    html = _assets_out(render(meta, preview=False, consent_html=None), site / "assets")
    if html.count(_INSERT_BEFORE) != 1:
        raise ValueError("the open page has no single injection point for the live hooks")
    i = html.index(_INSERT_BEFORE)
    html = html[:i] + f'  <script id="{LIVE_MARKER}">\n{LIVE_JS}\n  </script>\n' + html[i:]
    if f'class="{CONSENT_PLACEHOLDER_MARKER}"' in html or "[CONSENT PLACEHOLDER]" in html:
        raise ValueError("a live page must not carry the consent placeholder")
    index = site / "index.html"
    index.write_text(html, encoding="utf-8")
    return index


# --------------------------------------------------------------------------- the campaign's locks

@contextmanager
def _locked(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


@contextmanager
def study_slot(campaign_dir: Path, slots: int = 1, *, poll_sec: float = 60.0, label: str = "") -> Iterator[int]:
    """Hold one of the campaign's ``slots`` recruiting slots (a file lock each, released when the
    process ends, so a killed job frees its slot); wait for one when all are held."""
    root = Path(campaign_dir) / "study_slots"
    root.mkdir(parents=True, exist_ok=True)
    waited = False
    while True:
        for k in range(slots):
            f = open(root / f"slot_{k}.lock", "a+")
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                f.close()
                continue
            try:
                f.seek(0)
                f.truncate()
                f.write(f"{label} {os.getpid()} {time.strftime('%Y-%m-%dT%H:%M:%S')}\n")
                f.flush()
                if waited:
                    print(f"[live] {label}: recruiting slot {k} free", flush=True)
                yield k
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)
                f.close()
            return
        if not waited:
            print(f"[live] {label}: every recruiting slot is held by another chain's study; waiting", flush=True)
            waited = True
        time.sleep(poll_sec)


def campaign_ids(campaign_dir: Path, study_key: str, prolific_pids: Sequence[str]) -> Tuple[Dict[str, str], List[str]]:
    """Each Prolific id's campaign participant id (``p<k>``, assigned in the order given) and the
    run ids of people first seen in another study (repeats). The map lives in the campaign
    directory, private, under a lock (every chain writes it)."""
    path = Path(campaign_dir) / "participant_ids.json"
    with _locked(path.with_suffix(".lock")):
        known = json.loads(path.read_text()) if path.exists() else {}
        ids, repeats = {}, []
        for pid in prolific_pids:
            if pid not in known:
                known[pid] = {"id": f"p{len(known)}", "first_study": study_key}
            ids[pid] = known[pid]["id"]
            if known[pid]["first_study"] != study_key:
                repeats.append(known[pid]["id"])
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(known, indent=1))
        tmp.replace(path)
    return ids, repeats


# --------------------------------------------------------------------------- one person's data

@dataclass
class Person:
    id: str
    list_index: Optional[int]
    rows: List[Dict[str, Any]]
    comprehension_final: Optional[bool]
    seconds: Optional[float]
    problems: List[str] = field(default_factory=list)


def _comprehension_final(records: Sequence[Mapping[str, Any]]) -> Optional[bool]:
    """Every check right on the person's last reading of the instructions (the page shows them at
    most twice); None for a page without checks."""
    checks = [r for r in records if r.get("task") == "comprehension"]
    if not checks:
        return None
    last = max(int(r.get("attempt") or 0) for r in checks)
    return all(bool(r.get("correct")) for r in checks if int(r.get("attempt") or 0) == last)


def person(pid: str, response: Mapping[str, Any], lists: Mapping[str, Any]) -> Person:
    """One stored response as rows (the simulated rows' fields, plus ``participant`` and ``rt``)."""
    from src.rsa.live.collect import decode_trials
    from src.rsa_open.screens import page_rows

    p = Person(pid, response.get("list_index"), [], None, None)
    try:
        records = decode_trials(response["trials"])
        rows = page_rows(records)
    except (ValueError, KeyError, TypeError) as err:
        p.problems.append(f"unreadable record: {err}")
        return p
    if records and records[0].get("paradigm_sha256") not in (None, lists["paradigm_sha256"]):
        p.problems.append("another paradigm's page")
        return p
    expected = next((lst for lst in lists["lists"] if lst["list_index"] == p.list_index), None)
    if expected is None:
        p.problems.append(f"list {p.list_index} is not one of this design's")
        return p
    if len(rows) != len(expected["trials"]):
        p.problems.append(f"incomplete: {len(rows)} of {len(expected['trials'])} trials")
    p.comprehension_final = _comprehension_final(records)
    elapsed = [r.get("time_elapsed") for r in records if isinstance(r.get("time_elapsed"), (int, float))]
    p.seconds = max(elapsed) / 1000.0 if elapsed else None
    from src.rsa_open.chain import TRIAL_DESIGN_FIELDS

    for r in rows:
        p.rows.append({"participant": pid, "list_index": p.list_index, "phase": r["phase"], "unit": r["unit"],
                       "objects": r["objects"], "heard": r["heard"],
                       **{k: r[k] for k in TRIAL_DESIGN_FIELDS if r.get(k) is not None},
                       **({"expect": r["expect"]} if r.get("expect") is not None else {}),
                       "display_order": r.get("display_order"), "response": r["response"], "rt": r.get("rt")})
    return p


def exclusions(p: Person, repeat: bool) -> List[str]:
    """Why a person's data are left out (empty: kept). A missed catch trial is the closed
    campaign's rule (``max_catch_errors: 0``)."""
    from src.rsa_open.screens import catch_passed

    why = list(p.problems)
    if repeat:
        why.append("took part in an earlier study of the campaign")
    if p.comprehension_final is False:
        why.append("failed the comprehension checks on the second reading")
    catches = [r for r in p.rows if r["phase"] == "catch"]
    if any(not catch_passed(r) for r in catches):
        why.append("missed a catch trial")
    return why


# --------------------------------------------------------------------------- monitoring

def answer_key(row: Mapping[str, Any]) -> str:
    """What "the same answer" means for a trial's response: the screen position of a choice, the
    rating, the slider value (to 5% of its range), the allocation by screen position, the word, the
    yes/no; both parts for a two-measure trial."""
    r, order = row["response"], row.get("display_order") or []

    def one(r: Mapping[str, Any]) -> str:
        if "choice" in r:
            return f"position {order.index(r['choice']) if r['choice'] in order else r['choice']}"
        if "rating" in r:
            return f"rating {r['rating']}"
        if "value" in r:
            return f"value ~{round(float(r['value']) / 5) * 5}"
        if "points" in r:
            pts = r["points"]
            return "points " + ",".join(str(pts[o]) for o in order) if order else f"points {pts}"
        if "answer" in r:
            return f"answer {r['answer']}"
        if "utterance" in r:
            return f"word {r['utterance']}"
        return json.dumps(r, sort_keys=True)

    key = one(r)
    if isinstance(r.get("second"), Mapping):
        key += " | " + one(r["second"])
    return key


def degenerate(rows: Sequence[Mapping[str, Any]]) -> Optional[str]:
    """A note when a person gave the same answer on nearly every test trial, else None."""
    tests = [r for r in rows if r["phase"] == "test"]
    if len(tests) < DEGENERATE_MIN_TRIALS:
        return None
    key, n = Counter(answer_key(r) for r in tests).most_common(1)[0]
    share = n / len(tests)
    return f"{key} on {n} of {len(tests)} test trials" if share >= DEGENERATE_SHARE else None


def monitor_report(people: Sequence[Person], excluded: Mapping[str, Sequence[str]],
                   recruited: Optional[int] = None) -> Dict[str, Any]:
    """Aggregates for the PI and the gate (`PREREG_DRAFT.md` §4, item 2); run ids only."""
    from src.rsa_open.screens import catch_passed

    readable = [p for p in people if not p.problems]
    catches = [r for p in readable for r in p.rows if r["phase"] == "catch"]
    checked = [p for p in readable if p.comprehension_final is not None]
    flags = {p.id: d for p in readable if (d := degenerate(p.rows))}
    fast = {}
    for p in readable:
        rts = [r["rt"] for r in p.rows if r["phase"] == "test" and isinstance(r.get("rt"), (int, float))]
        if rts and statistics.median(rts) < FAST_MS:
            fast[p.id] = f"median {statistics.median(rts):.0f} ms a test trial"
    reasons = Counter(w for ws in excluded.values() for w in ws)
    minutes = [p.seconds / 60 for p in readable if p.seconds]
    kept = len(people) - len(excluded)
    return {
        "recruited": recruited, "responses": len(people), "readable": len(readable), "kept": kept,
        "excluded": len(excluded), "exclusion_rate": len(excluded) / len(people) if people else None,
        "exclusion_reasons": dict(reasons),
        "comprehension_final": (sum(p.comprehension_final for p in checked) / len(checked)) if checked else None,
        "catch_pass": (sum(catch_passed(r) for r in catches) / len(catches)) if catches else None,
        "same_answer": {"n": len(flags), "share": len(flags) / len(readable) if readable else None,
                        "people": flags},
        "fast": {"n": len(fast), "people": fast},
        "minutes": ({"median": statistics.median(minutes), "p10": sorted(minutes)[int(0.1 * (len(minutes) - 1))],
                     "p90": sorted(minutes)[int(0.9 * (len(minutes) - 1))]} if minutes else None),
    }


def monitor_lines(report: Mapping[str, Any]) -> List[str]:
    def pct(x):
        return "-" if x is None else f"{x:.0%}"

    m = report.get("minutes") or {}
    lines = [f"responses {report['responses']} (recruited {report.get('recruited')}), kept {report['kept']}, "
             f"excluded {report['excluded']} ({pct(report['exclusion_rate'])}): {report['exclusion_reasons']}",
             f"comprehension (second reading) {pct(report['comprehension_final'])}, catch trials "
             f"{pct(report['catch_pass'])}, same answer on nearly every trial {report['same_answer']['n']} "
             f"({pct(report['same_answer']['share'])}), fast {report['fast']['n']}"
             + (f", minutes median {m['median']:.1f} (p10 {m['p10']:.1f}, p90 {m['p90']:.1f})" if m else "")]
    for pid, note in sorted(report["same_answer"]["people"].items()):
        lines.append(f"  same answer: {pid}: {note}")
    return lines


def read_people(responses: Sequence[Mapping[str, Any]], study_id: str, lists: Mapping[str, Any],
                ids: Mapping[str, str]) -> Tuple[List[Person], Dict[str, Any]]:
    """The study's stored responses, one person each (a reload's second submission is left out, on
    record); responses of another study (a preview, a draft) are left out, on record."""
    seen, people, other, duplicates = set(), [], 0, 0
    for resp in sorted(responses, key=lambda r: (r.get("submitted_at_client") or "", r.get("participant_id_str") or "")):
        if resp.get("prolific_study_id") != study_id:
            other += 1
            continue
        pid = resp.get("prolific_pid") or resp.get("participant_id_str")
        if pid in seen:
            duplicates += 1
            continue
        seen.add(pid)
        people.append(person(ids[pid], resp, lists))
    return people, {"other_study": other, "second_submissions": duplicates}


def _pids(responses: Sequence[Mapping[str, Any]], study_id: str) -> List[str]:
    out = []
    for resp in sorted(responses, key=lambda r: (r.get("submitted_at_client") or "", r.get("participant_id_str") or "")):
        pid = resp.get("prolific_pid") or resp.get("participant_id_str")
        if resp.get("prolific_study_id") == study_id and pid not in out:
            out.append(pid)
    return out


# --------------------------------------------------------------------------- one experiment

def study_key(chain: str, arm: str, n: int, run_label: str) -> str:
    return f"{arm}/{chain}/experiment{n}/{run_label}"


def check_review(frozen: Path, paradigm_sha256: str) -> dict:
    """The PI's approval of a frozen paradigm (`src.rsa_open.review`); raises AwaitingReview."""
    path = Path(frozen) / "review.json"
    if not path.exists():
        raise AwaitingReview(f"{frozen} has not been reviewed: preview it (PROLIFIC_MODE=test), then "
                             f"`python -m src.rsa_open.review --frozen {frozen} --by <name>`")
    review = json.loads(path.read_text())
    if review.get("paradigm_sha256") != paradigm_sha256:
        raise AwaitingReview(f"{path} approves another paradigm ({review.get('paradigm_sha256')})")
    if not review.get("approved"):
        raise AwaitingReview(f"{path}: not approved ({review.get('reason', '')})")
    return review


def _wait(manifest: Mapping[str, Any], target: int, log_dir: Path, max_wait_sec: float, lists,
          snapshot: Path, every_sec: float = MONITOR_EVERY_SEC) -> int:
    """Poll Prolific until ``target`` people have finished or the wait ends (then pause the study,
    as main's collection does), printing the monitor every ``every_sec``. The window runs from the
    first wait (``collection_started.json`` in ``log_dir``), so a resumed job continues it rather
    than starting it again."""
    from src.pipelines.outer_loop.collect import _pause_unfilled_study, _poll_prolific_until_target
    from src.rsa.live.collect import fetch_responses

    study = manifest.get("prolific_study_id")
    if not study:
        raise RuntimeError("the deployment recorded no Prolific study: nothing to collect from")
    started = Path(log_dir) / "collection_started.json"
    if not started.exists():
        started.write_text(json.dumps({"at": time.time(), "max_wait_sec": max_wait_sec}))
    start, completed = json.loads(started.read_text())["at"], 0
    while True:
        left = max_wait_sec - (time.time() - start)
        completed = _poll_prolific_until_target(study, target, log_dir, max_wait_sec=max(1.0, min(every_sec, left)))
        try:  # advisory: a failed snapshot is reported, and the final fetch must succeed
            responses = fetch_responses(manifest)
            pids = _pids(responses, study)
            people, _ = read_people(responses, study, lists, {pid: f"r{i}" for i, pid in enumerate(pids)})
            report = monitor_report(people, {p.id: w for p in people if (w := exclusions(p, False))}, completed)
            snapshot.write_text(json.dumps(report, indent=1))
            for line in monitor_lines(report):
                print(f"[monitor] {line}", flush=True)
        except Exception as exc:  # noqa: BLE001 (a snapshot must not stop the collection)
            print(f"[monitor] WARNING: the snapshot failed: {type(exc).__name__}: {exc}", flush=True)
        if completed >= target or time.time() - start >= max_wait_sec:
            break
    if completed < target:
        _pause_unfilled_study(study, completed, target, log_dir)
    return completed


def collect(*, exp_dir: Path, chain: str, arm: str, n: int, cfg: LiveConfig) -> Tuple[List[dict], Dict[str, Any], List[float]]:
    """Deploy experiment ``n``'s frozen paradigm, recruit, collect: (rows of the kept people, the
    collection record, their completion times in seconds). Raises `src.rsa.live.collect.DraftOnly`
    in test mode once the page and the draft exist."""
    from src.pipelines.outer_loop.collect import _PROLIFIC_MAX_WAIT_SEC
    from src.pipelines.outer_loop.deployment.local import run_deployment
    from src.pipelines.outer_loop.deployment.manifest import manifest_path, recorded_live_study
    from src.pipelines.outer_loop.orchestrator import require_outside_agent_trees
    from src.rsa.live.collect import DraftOnly, fetch_responses
    from src.rsa_open.paradigm import load_paradigm

    exp_dir = Path(exp_dir)
    frozen = exp_dir / "design" / "frozen"
    paradigm = load_paradigm(frozen / "paradigm.json")
    lists = json.loads((frozen / "lists.json").read_text())
    if len(lists["lists"]) != paradigm.design.n_participants:
        raise ValueError(f"{len(lists['lists'])} trial lists for the paradigm's {paradigm.design.n_participants} people")
    n_people = cfg.participants or paradigm.design.n_participants
    label = f"{chain}, experiment {n}"
    if cfg.prolific_mode == "live" and arm == "open" and n == 1 and cfg.review_first_paradigm:
        check_review(frozen, paradigm.sha256)
    raw = Path(cfg.private_dir) / "raw_collected" / f"experiment{n}.json"
    require_outside_agent_trees(raw, "the raw live responses (Prolific ids)")
    if Path(exp_dir).resolve() in raw.resolve().parents:
        raise RuntimeError(f"{raw} is inside the run directory the agents read")
    with study_slot(cfg.campaign_dir, cfg.max_concurrent_studies, label=label):
        if recorded_live_study(exp_dir) is None:
            build_site(lists, exp_dir)
            run_deployment(exp_dir=exp_dir, project_id=PROJECT_ID, run_id=n, deploy_target="firebase",
                           prolific_mode=cfg.prolific_mode, agent_backend="opencode",
                           collection_owner=cfg.collection_owner, firebase_project=cfg.firebase_project,
                           firebase_region=cfg.firebase_region, n_participants=n_people, repo_root=cfg.repo_root,
                           run_label=cfg.run_label, recruitment_overrides=recruitment_overrides(paradigm, cfg))
        manifest = json.loads(manifest_path(exp_dir).read_text())
        if cfg.prolific_mode == "test":
            raise DraftOnly(f"test mode: the page is at {manifest.get('experiment_url')} and Prolific study "
                            f"{manifest.get('prolific_study_id')} is a draft to preview; nothing to collect")
        ended = exp_dir / "deployment" / "collection_ended.json"
        if ended.exists():  # a resumed job: the wait ended (the study filled or was paused) before it stopped
            completed = json.loads(ended.read_text())["completed"]
        else:
            completed = _wait(manifest, n_people, exp_dir / "deployment", cfg.max_wait_sec or _PROLIFIC_MAX_WAIT_SEC,
                              lists, exp_dir / "deployment" / "monitor.json")
            ended.write_text(json.dumps({"completed": completed, "target": n_people,
                                         "at": time.strftime("%Y-%m-%dT%H:%M:%S")}))
    responses = fetch_responses(manifest)
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_text(json.dumps(responses))
    study = manifest["prolific_study_id"]
    ids, repeats = campaign_ids(cfg.campaign_dir, study_key(chain, arm, n, cfg.run_label), _pids(responses, study))
    people, skipped = read_people(responses, study, lists, ids)
    if not people:
        raise RuntimeError(f"{label}: no response of study {study} was stored ({len(responses)} responses in all)")
    excluded = {p.id: w for p in people if (w := exclusions(p, p.id in repeats))}
    report = monitor_report(people, excluded, completed)
    for line in monitor_lines(report):
        print(f"[live] {label}: {line}", flush=True)
    kept = [p for p in people if p.id not in excluded]
    if not kept:
        raise RuntimeError(f"{label}: every one of {len(people)} people was excluded: {report['exclusion_reasons']}")
    rows = [r for p in kept for r in p.rows]
    record = {"collection": "live", "prolific_mode": cfg.prolific_mode, "study_id": study,
              "target": n_people, "smoke_test": cfg.participants is not None, **skipped, **report,
              "excluded_people": {pid: w for pid, w in excluded.items()}}
    return rows, record, [p.seconds for p in kept if p.seconds]


# --------------------------------------------------------------------------- the launcher's checks

def read_campaign_config(path: Path = OPEN_LIVE_CONFIG) -> dict:
    """``open_live.yaml``, checked: the study terms come from each paradigm, never from here."""
    import re

    from src.pipelines.outer_loop.deployment.prolific import build_eligibility_filters

    cfg = yaml.safe_load(Path(path).read_text())
    for key in ("firebase_project", "series_label", "study", "prolific"):
        if key not in cfg:
            raise ValueError(f"{path} lacks {key}")
    own = sorted(set(cfg["prolific"]) & {"reward", "reward_per_hour", "estimated_completion_time", "name",
                                          "description", "participants", "total_available_places"})
    if own:
        raise ValueError(f"{path}: {own} are each study's (its paradigm's and the study: block), not prolific:'s")
    if not isinstance(cfg.get("confirm_live_recruitment", False), bool):
        raise ValueError("confirm_live_recruitment must be true or false")
    if "{minutes}" not in cfg["study"]["description"]:
        raise ValueError("the study description must state the paradigm's {minutes}")
    build_eligibility_filters(cfg["prolific"])  # raises on a bad approval rate
    for arm in ("open", "fixed"):
        for smoke in (False, True):
            site = hosting_site(cfg, arm, 9, smoke=smoke)
            if len(site) > 30 or not re.fullmatch(r"[a-z0-9-]+", site):
                raise ValueError(f"Hosting site id {site!r} must be <= 30 characters of [a-z0-9-]")
    return cfg


def hosting_site(cfg: Mapping[str, Any], arm: str, chain: int, smoke: bool = False) -> str:
    """Each chain deploys to a site of its own (a deploy replaces the whole site), a smoke test too."""
    return f"{cfg['firebase_project']}-{cfg['series_label']}-{arm}-c{chain}" + ("s" if smoke else "")


def render_prolific(path: Path = OPEN_LIVE_CONFIG) -> Path:
    """The ``prolific:`` block as the rsa_reference project's prolific_config.yaml (gitignored; main's
    deployment reads it, and each study's terms override it)."""
    from src.runtime.prolific import prolific_config_path

    cfg = read_campaign_config(path)
    out = prolific_config_path(PROJECT_ID)
    if not out.parent.is_dir():
        raise FileNotFoundError(f"no project directory {out.parent}")
    header = "# AUTO-GENERATED by `python -m src.rsa_open.live --render` from the open campaign's config. Do not edit.\n\n"
    out.write_text(header + yaml.safe_dump(cfg["prolific"], sort_keys=False, allow_unicode=True), encoding="utf-8")
    return out


def launch_summary(cfg: Mapping[str, Any], arm: str, chains: Sequence[int], n_experiments: int,
                   participants: Optional[int] = None) -> List[str]:
    """What a launch recruits at most: the fixed arm's terms are its spec's; an open study's are its
    designer's, within the budget."""
    from src.rsa_open.paradigm import BUDGET_USD, PROLIFIC_FEE_MULTIPLIER

    studies = len(chains) * n_experiments
    if arm == "fixed":
        from src.rsa_open.fixed import SPEC_PATH
        from src.rsa_open.paradigm import load_paradigm

        p = load_paradigm(SPEC_PATH)
        n = participants or p.design.n_participants
        per = n * p.design.pay_usd * PROLIFIC_FEE_MULTIPLIER
        terms = f"{n} people x ${p.design.pay_usd:.2f} for {p.design.minutes:g} min: ${per:,.2f} a study"
    else:
        per = BUDGET_USD if participants is None else math.nan
        terms = ("each paradigm's own N, pay and minutes, at most "
                 f"${BUDGET_USD:,.2f} a study" if participants is None else f"{participants} people a study at its paradigm's pay")
    total = "" if math.isnan(per) else f", at most ${per * studies:,.2f}"
    return [f"study name     : {cfg['study']['name']}",
            f"arm, chains    : {arm}, {list(chains)} (sites {', '.join(hosting_site(cfg, arm, k) for k in chains)})",
            f"per study      : {terms}",
            f"this launch    : up to {studies} studies{total} (finished experiments are not rerun)",
            f"one at a time  : {cfg.get('max_concurrent_studies', 1)} study recruiting at once, campaign-wide",
            f"confirmed      : {cfg.get('confirm_live_recruitment', False)}"]


@dataclass
class Args:
    config: Path = OPEN_LIVE_CONFIG
    check: bool = False
    render: bool = False
    arm: str = "open"
    chains: Tuple[int, ...] = (0,)
    n_experiments: int = 4
    participants: Optional[int] = None
    site: Optional[int] = None
    """Print chain ``site``'s Hosting site id for ``arm``."""
    smoke: bool = False


if __name__ == "__main__":
    import sys

    import tyro

    args = tyro.cli(Args)
    campaign = read_campaign_config(args.config)
    if args.site is not None:
        print(hosting_site(campaign, args.arm, args.site, smoke=args.smoke))
    if args.check:
        print("\n".join(launch_summary(campaign, args.arm, args.chains, args.n_experiments, args.participants)),
              file=sys.stderr)
    if args.render:
        print(f"[render] wrote {render_prolific(args.config)}", file=sys.stderr)
