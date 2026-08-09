"""Collection support for the active outer loop."""

from __future__ import annotations

import csv
import io
import math
import multiprocessing
import os
import random
import re
import sys
import time
import urllib.parse
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from src.pipelines.outer_loop.featurizer import Featurizer, load_featurizer
from src.pipelines.outer_loop.llm import get_llm, invoke_llm, load_prompt_for_run
from src.runtime.console import log_status
from src.models.theorist.predictions import get_model_predictions
from src.models.probability import validate_probability_distribution
from src.runtime.observability import agent_log


def _unique_batch_id() -> str:
    """Collision-resistant batch id used to build participant IDs.

    ``datetime.utcnow()`` has 1-second resolution, so two collection passes for
    the same project/run within the same second produced *identical* participant
    IDs — and the Firebase results filter keys on ``participant_id_str``, so the
    collision cross-attributes responses between runs. Add microseconds plus a
    short random token, and use a timezone-aware UTC clock (``utcnow()`` is
    deprecated and returns a naive timestamp).
    """
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    return f"{stamp}_{uuid.uuid4().hex[:6]}"

RESPONSE_OPTIONS = ["left", "right"]
MAX_PARALLEL_PARTICIPANTS = 3
_DRIVE_INTERVAL_MS = 1200
_DRIVE_TIMEOUT_MS = 180_000
_LLM_CONTEXT_MAX_SCREENS = 20
_FIXATION_WAIT_SEC = 0.25
_PROLIFIC_POLL_INTERVAL_SEC = 30
# Stop waiting on a Prolific study after this long so a stalled/under-recruited
# study (participants return or time out and the target is never met) can't hang
# the pipeline forever. On timeout we fetch whatever results exist.
_PROLIFIC_MAX_WAIT_SEC = 2 * 60 * 60  # 2 hours


def _poll_prolific_until_target(
    study_id: str,
    target_places: int,
    out_dir: Path,
    *,
    max_wait_sec: float = _PROLIFIC_MAX_WAIT_SEC,
    poll_interval_sec: float = _PROLIFIC_POLL_INTERVAL_SEC,
) -> int:
    """Poll Prolific until ``target_places`` submissions complete or time out.

    Returns the last observed COMPLETED count. Bounded by ``max_wait_sec`` so a
    study that never fills cannot block indefinitely.
    """
    from src.runtime.prolific import get_submission_counts

    start = time.monotonic()
    completed = 0
    while completed < target_places:
        counts, err = get_submission_counts(study_id)
        if err:
            agent_log(out_dir, f"Prolific poll: study_id={study_id!r} error={err!r}")
        else:
            # Prolific's submissions/counts/ has no "COMPLETED" status: a finished
            # participant lands in AWAITING REVIEW, then APPROVED (or PARTIALLY
            # APPROVED). Count those as completed. ACTIVE/RESERVED are still in
            # progress; RETURNED/TIMED-OUT/SCREENED OUT/REJECTED never yield data.
            completed = sum(
                int(counts.get(k) or 0)
                for k in ("AWAITING REVIEW", "APPROVED", "PARTIALLY APPROVED")
            )
            # Back-compat: honor an explicit COMPLETED count if the API ever adds one.
            completed = max(
                completed,
                int(counts.get("COMPLETED") or counts.get("completed") or 0),
            )
            agent_log(
                out_dir,
                f"Prolific poll: study_id={study_id!r} completed={completed} target={target_places}",
            )
            if completed >= target_places:
                break
        if time.monotonic() - start >= max_wait_sec:
            agent_log(
                out_dir,
                f"Prolific poll: timed out after {max_wait_sec}s with "
                f"completed={completed}/{target_places}; fetching partial results",
            )
            break
        time.sleep(poll_interval_sec)
    return completed


def _playwright_errors() -> tuple[type[BaseException], ...]:
    """Exception types Playwright raises for *page-level* trouble.

    ``playwright.sync_api.Error`` (whose ``TimeoutError`` is a subclass) covers
    the outcomes the steering loop legitimately has to survive: the page
    navigated mid-call so the execution context was destroyed, an element went
    stale, the page/browser was closed. Everything else — a TypeError from a
    wrong call signature, an AttributeError from a renamed helper — is a bug in
    this module and must propagate rather than be read as "the screen was
    empty".
    """
    from playwright.sync_api import Error as PlaywrightError

    return (PlaywrightError,)


def _experiment_data_present(page) -> bool:
    """Has jsPsych published ``window.__experimentData`` yet?

    Polled every drive-loop tick. A Playwright page error here means the page
    is mid-navigation, in which case the data provably is not readable yet, so
    ``False`` is the correct answer (and logging it once a tick would bury the
    real messages). Any other exception propagates.
    """
    try:
        return bool(page.evaluate("typeof window.__experimentData !== 'undefined'"))
    except _playwright_errors():
        return False


def _get_screen_content(page) -> str:
    """Return the visible experiment text, or ``""`` when it cannot be read.

    Only a Playwright page error is tolerated (see :func:`_playwright_errors`),
    and it is logged: a screen that silently reads as empty would send the
    steering LLM a blank prompt and, through it, an arbitrary choice.
    """
    try:
        return (
            page.evaluate(
                """() => {
              const sel = document.querySelector('#jspsych-content')
                || document.querySelector('.jspsych-content-wrapper')
                || document.querySelector('.jspsych-display-element')
                || document.body;
              return sel ? (sel.innerText || sel.textContent || '').trim() : '';
            }"""
            )
            or ""
        )
    except _playwright_errors() as exc:
        print(
            f"  [steering] could not read the screen ({type(exc).__name__}: {exc}); "
            "treating it as empty",
            file=sys.stderr,
            flush=True,
        )
        return ""


def _parse_steering_action(text: str) -> tuple[str, str] | None:
    if not text or not isinstance(text, str):
        return None
    text = text.strip()
    match = re.search(r"ACTION:\s*click\s+(.+)", text, re.IGNORECASE | re.DOTALL)
    if match:
        label = match.group(1).strip().split("\n")[0].strip()
        if label:
            return ("click", label)
    match = re.search(
        r"ACTION:\s*key\s+(f|j|ArrowLeft|ArrowRight)", text, re.IGNORECASE
    )
    if not match:
        return None
    key = match.group(1)
    lowered = key.lower()
    if lowered == "f":
        return ("key", "f")
    if lowered == "j":
        return ("key", "j")
    if lowered == "arrowleft":
        return ("key", "ArrowLeft")
    if lowered == "arrowright":
        return ("key", "ArrowRight")
    return ("key", key)


def check_response_variation(rows: list[dict[str, Any]]) -> tuple[bool, str]:
    """Return ``(ok, message)`` for a quality check on collected responses.

    Flags data that has no response variation — every trial chose the same side.
    Such data carries no signal for model comparison and almost always means the
    collector decided every trial (e.g. a biased fallback always clicking the
    first option) rather than the participant responding to the stimulus. Tiny
    inputs (0–1 parseable responses) are not flagged as degenerate.
    """
    if not rows:
        return False, "no response rows were collected"
    values: list[int] = []
    for row in rows:
        raw = row.get("chose_left")
        if raw is None or raw == "":
            return False, "collected row has a missing chose_left value"
        try:
            numeric = float(raw)
        except (TypeError, ValueError):
            return False, f"collected row has invalid chose_left={raw!r}; expected binary 0 or 1"
        if not math.isfinite(numeric) or numeric not in (0.0, 1.0):
            return False, f"collected row has invalid chose_left={raw!r}; expected binary 0 or 1"
        values.append(int(numeric))
    if not values:
        return False, "collected rows have no parseable chose_left values"
    if len(values) >= 2 and len(set(values)) == 1:
        return False, (
            f"all {len(values)} responses are identical (chose_left={values[0]}); "
            "the participants ignored the stimulus or the steering is biased to one "
            "side — this data has no signal for model comparison"
        )
    return True, f"ok: {len(values)} responses with variation across sides"


# Directional keys the keyboard template listens for; used to translate an LLM's
# left/right decision into the correct button on a button-based trial.
_LEFT_KEYS = {"f", "arrowleft"}
_RIGHT_KEYS = {"j", "arrowright"}


def _press_key_or_raise(page, key: str, what: str, click_error: Exception | None) -> None:
    """Press ``key``; raise if that fails too, naming both failed modalities.

    The keyboard is the last resort after the button modality was tried. If it
    also fails the trial simply did not advance — and the historical behavior
    (swallow both errors) left the drive loop spinning to its 3-minute timeout
    with nothing logged, which is how a modality mismatch produced a whole run
    of degenerate one-sided data before anyone noticed.
    """
    try:
        page.keyboard.press(key)
    except Exception as exc:
        raise RuntimeError(
            f"{what}: neither a button click nor a {key!r} key press worked "
            f"(click: {click_error!r}; key: {type(exc).__name__}: {exc}). The "
            "trial did not advance — check that the steering modality matches "
            "the modality the experiment actually renders."
        ) from exc


def _click_random_choice(page) -> None:
    """Advance a trial by an UNBIASED random choice.

    When buttons are present, click a uniformly-random one (so a declined or
    unparseable LLM action never silently biases toward the first option);
    otherwise press a random directional key. A failed click falls back to the
    keyboard — but says so, and raises if the keyboard fails as well.
    """
    click_error: Exception | None = None
    try:
        buttons = page.locator("button.jspsych-btn")
        count = buttons.count()
        if count > 0:
            buttons.nth(random.randrange(count)).click(timeout=1000)
            return
    except _playwright_errors() as exc:
        # Only a Playwright page error earns the keyboard fallback. A bug in
        # this module must NOT be papered over by a key press that "works" —
        # that is how f/j got pressed at button trials for a whole run.
        click_error = exc
        print(
            f"  [steering] random button click failed ({type(exc).__name__}: {exc}); "
            "falling back to a random key press",
            file=sys.stderr,
            flush=True,
        )
    _press_key_or_raise(
        page, random.choice(["f", "j"]), "could not advance the trial", click_error
    )


def _act_key(page, key: str) -> None:
    """Apply a directional key choice, modality-aware.

    On a multi-button trial, translate left keys (f/ArrowLeft) to the first
    button and right keys (j/ArrowRight) to the last, so the participant's
    left/right decision lands on the right option whether the experiment uses
    keyboard or button responses. A single-button screen (consent/instructions)
    is advanced by clicking it. Falls back to an actual key press — loudly, and
    raising if that fails too (see :func:`_press_key_or_raise`).
    """
    lowered = key.lower()
    click_error: Exception | None = None
    try:
        buttons = page.locator("button.jspsych-btn")
        count = buttons.count()
        if count == 1:
            buttons.first.click(timeout=1000)
            return
        if count >= 2:
            if lowered in _LEFT_KEYS:
                idx = 0
            elif lowered in _RIGHT_KEYS:
                idx = count - 1
            else:
                idx = random.randrange(count)
            buttons.nth(idx).click(timeout=1000)
            return
    except _playwright_errors() as exc:  # see _click_random_choice
        click_error = exc
        print(
            f"  [steering] button click for key {key!r} failed "
            f"({type(exc).__name__}: {exc}); falling back to a key press",
            file=sys.stderr,
            flush=True,
        )
    _press_key_or_raise(page, key, f"could not apply the {key!r} choice", click_error)


def _drive_experiment_with_llm(
    page,
    timeout_ms: int,
    project_id: str,
    run_id: int,
    logs_dir: Path,
    state: dict | None = None,
) -> tuple[bool, bool]:
    """Steer one participant through the experiment with the LLM.

    Returns ``(finished, llm_used)``. ``llm_used=False`` now means exactly one
    thing — the project ships no ``4_collect_steering`` prompt, so there is
    nothing to steer with and the caller should fall back to blind clicking.
    An LLM that cannot be constructed (missing API key, missing dependency) is
    a *configuration* error and raises: silently demoting every simulated
    participant to blind random clicking produces data that looks collected but
    carries no signal.
    """
    llm = get_llm()

    steering_prompt = load_prompt_for_run(
        project_id, run_id, "4_collect_steering", state
    )
    if not steering_prompt.strip():
        print(
            f"  [steering] no 4_collect_steering prompt for project {project_id!r} "
            f"run {run_id}; falling back to blind clicking",
            file=sys.stderr,
            flush=True,
        )
        return (False, False)

    deadline = time.monotonic() + (timeout_ms / 1000.0)
    context_parts: list[str] = []
    while time.monotonic() < deadline:
        if _experiment_data_present(page):
            return (True, True)

        screen_text = _get_screen_content(page) or "(loading or empty screen)"
        if screen_text.strip() in ("+", ""):
            time.sleep(_FIXATION_WAIT_SEC)
            continue

        context_parts.extend(
            [
                "=== CURRENT SCREEN ===",
                screen_text,
                "",
                "Reply with exactly one line: ACTION: click <button label> or ACTION: key f|j|ArrowLeft|ArrowRight",
            ]
        )
        user_msg = "\n".join(context_parts)
        try:
            response = invoke_llm(
                system=steering_prompt,
                user=user_msg,
                llm=llm,
                source="browser_steering",
            )
        except Exception as exc:
            print(f"  [LLM steering error] {exc}", file=sys.stderr, flush=True)
            (logs_dir / "llm_steering_error.txt").write_text(str(exc), encoding="utf-8")
            return (False, True)

        action = _parse_steering_action(response)
        if action is None:
            # Unparseable reply: advance without biasing toward either side.
            print(
                f"  [steering] unparseable reply {response.strip()[:80]!r}; "
                "advancing with a random choice",
                file=sys.stderr,
                flush=True,
            )
            _click_random_choice(page)
        elif action[0] == "click":
            try:
                page.get_by_role("button", name=action[1]).click(timeout=2000)
            except _playwright_errors() as exc:
                # The LLM named a button this screen does not have. Expected
                # (it is generating free text), but never silent: an unnoticed
                # stream of these means every trial was decided by the RNG.
                print(
                    f"  [steering] no clickable button named {action[1]!r} "
                    f"({type(exc).__name__}); advancing with a random choice",
                    file=sys.stderr,
                    flush=True,
                )
                _click_random_choice(page)
        else:
            # Directional key: lands on the correct button for button trials.
            _act_key(page, action[1])

        context_parts.append(f"Your action: {response.strip()[:150]}")
        if len(context_parts) > _LLM_CONTEXT_MAX_SCREENS * 4:
            context_parts = context_parts[-(_LLM_CONTEXT_MAX_SCREENS * 4) :]
        time.sleep(0.1)

    return (_experiment_data_present(page), True)


def _drive_experiment_to_finish(page, timeout_ms: int = _DRIVE_TIMEOUT_MS) -> bool:
    """Blind fallback when LLM steering is unavailable: click through to the end.

    Picks an UNBIASED random option each step rather than always the first
    button, so a blind run does not systematically favor one side (which would
    silently produce degenerate, one-sided data).
    """
    deadline = time.monotonic() + (timeout_ms / 1000.0)
    step_sec = _DRIVE_INTERVAL_MS / 1000.0
    while time.monotonic() < deadline:
        if _experiment_data_present(page):
            return True
        _click_random_choice(page)
        time.sleep(step_sec)
    return _experiment_data_present(page)


def _run_one_participant_firebase(args: tuple) -> tuple[int, bool, str | None]:
    (
        participant_index,
        participant_id_str,
        experiment_url,
        project_id,
        run_id,
        nav_timeout_ms,
        drive_timeout_ms,
        logs_dir_path,
    ) = args
    logs_dir = Path(logs_dir_path) if logs_dir_path else None
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return (participant_index, False, "playwright not installed")
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            try:
                page = browser.new_page()
                goto_url = (
                    experiment_url
                    + ("&" if "?" in experiment_url else "?")
                    + "participant_id="
                    + urllib.parse.quote(participant_id_str)
                )
                page.goto(goto_url, wait_until="load", timeout=nav_timeout_ms)
                done, _ = _drive_experiment_with_llm(
                    page, drive_timeout_ms, project_id, run_id, logs_dir
                )
                if not done:
                    done = _drive_experiment_to_finish(page, drive_timeout_ms)
                if done:
                    page.wait_for_timeout(1500)
            finally:
                browser.close()
        return (participant_index, done, None)
    except Exception as exc:
        if logs_dir:
            (logs_dir / f"p{participant_index}_error.txt").write_text(
                str(exc), encoding="utf-8"
            )
        return (participant_index, False, str(exc))


def _collect_live(
    state: dict[str, Any],
    config: dict[str, Any],
    out_dir: Path,
    logs_dir: Path,
) -> list[dict[str, Any]]:
    project_id = state["project_id"]
    run_id = state["run_id"]
    study_id = config.get("prolific_study_id")
    results_api_url = config.get("results_api_url") or config.get("experiment_url")
    target_places = (
        config.get("total_available_places")
        or config.get("simulated_n_participants")
        or 1
    )

    agent_log(out_dir, "Collect (live): waiting for Prolific study (poll every 30s)")
    # A missing study/results URL, or a failed fetch, must never return [] — an
    # empty row list is indistinguishable from "the study ran and nobody
    # responded", and downstream that becomes a silently under-powered (or
    # synthetic) analysis of a study real participants were paid for.
    if not study_id:
        msg = (
            "live collection needs `prolific_study_id` in the experiment config, "
            "but it is absent — the Prolific flow was never configured for this "
            "run. Deploy with --deploy-target firebase --prolific-mode live."
        )
        agent_log(out_dir, f"Collect (live): error - {msg}")
        raise RuntimeError(msg)
    if not results_api_url:
        msg = (
            "live collection needs `results_api_url` (or `experiment_url`) in the "
            "experiment config to fetch responses, but neither is set."
        )
        agent_log(out_dir, f"Collect (live): error - {msg}")
        raise RuntimeError(msg)

    _poll_prolific_until_target(study_id, int(target_places), out_dir)

    agent_log(out_dir, "Collect (live): fetching results from Firebase")
    url = _results_url(results_api_url, config, project_id, run_id)
    try:
        req = _results_request(url)
        with urllib.request.urlopen(req, timeout=60) as response:
            body = response.read().decode("utf-8")
    except Exception as exc:
        agent_log(out_dir, f"Collect (live): results fetch failed: {exc}")
        raise RuntimeError(
            f"live results fetch failed for {url}: {type(exc).__name__}: {exc}. "
            "The participants' data is on the server — refusing to report zero "
            "responses. Fix the endpoint/credentials and re-collect."
        ) from exc

    rows: list[dict[str, Any]] = []
    if body.strip():
        for row in csv.DictReader(io.StringIO(body)):
            rows.append(dict(row))
    agent_log(out_dir, f"Collect (live): got {len(rows)} rows from /results")
    return rows


def _collect_from_firebase(
    state: dict[str, Any],
    config: dict[str, Any],
    results_api_url: str,
    n_participants: int,
    out_dir: Path,
    logs_dir: Path,
) -> list[dict[str, Any]]:
    project_id = config.get("project_id", "")
    run_id = config.get("run_id", "")
    collection_session_id = config.get("collection_session_id")
    if not collection_session_id and not (project_id and run_id):
        # A mis-wired config must not silently read as "no participants yet".
        raise ValueError(
            "Results-API collection needs `collection_session_id` (or both "
            "`project_id` and `run_id`) in the experiment config; got neither. "
            "Was the deployment step's config merge skipped?"
        )

    experiment_url = config.get("experiment_url")
    batch_id = _unique_batch_id()
    participant_ids = [
        f"{project_id}_run{run_id}_{batch_id}_p{i}" for i in range(n_participants)
    ]
    (logs_dir / "participant_ids.txt").write_text(
        "\n".join(participant_ids), encoding="utf-8"
    )
    log_status(f"Participant IDs for this run: {logs_dir / 'participant_ids.txt'}")

    if experiment_url and n_participants > 0:
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as exc:
            # A missing dependency is a config error, not "no participants".
            msg = "playwright not installed; run: pip install playwright && playwright install chromium"
            print(msg, file=sys.stderr, flush=True)
            (logs_dir / "browser_error.txt").write_text(msg, encoding="utf-8")
            raise RuntimeError(msg) from exc

        nav_timeout_ms = 60_000
        n_parallel = (
            min(n_participants, MAX_PARALLEL_PARTICIPANTS)
            if MAX_PARALLEL_PARTICIPANTS >= 2
            else 1
        )
        if n_parallel >= 2 and n_participants >= 2:
            log_status(
                f"Running {n_participants} browser participant(s) (Firebase, {n_parallel} in parallel)..."
            )
            worker_args = [
                (
                    run_idx,
                    participant_ids[run_idx],
                    experiment_url,
                    project_id,
                    run_id,
                    nav_timeout_ms,
                    _DRIVE_TIMEOUT_MS,
                    str(logs_dir),
                )
                for run_idx in range(n_participants)
            ]
            with multiprocessing.Pool(processes=n_parallel) as pool:
                results = pool.map(_run_one_participant_firebase, worker_args)
            for idx, success, err in sorted(results, key=lambda result: result[0]):
                if err:
                    print(
                        f"  Participant {idx + 1}/{n_participants}: {err}",
                        file=sys.stderr,
                        flush=True,
                    )
                elif not success:
                    print(
                        f"  Run {idx + 1}/{n_participants}: timed out before experiment finished (no POST).",
                        file=sys.stderr,
                        flush=True,
                    )
            if n_participants:
                log_status("Steering: LLM (Gemini)")
        else:
            log_status(f"Running {n_participants} browser participant(s) (Firebase)...")
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(headless=True)
                try:
                    for run_idx in range(n_participants):
                        participant_id = participant_ids[run_idx]
                        goto_url = (
                            experiment_url
                            + ("&" if "?" in experiment_url else "?")
                            + "participant_id="
                            + urllib.parse.quote(participant_id)
                        )
                        log_status(
                            f"Participant {run_idx + 1}/{n_participants} in progress..."
                        )
                        page = browser.new_page()
                        try:
                            page.goto(
                                goto_url, wait_until="load", timeout=nav_timeout_ms
                            )
                            done, llm_used = _drive_experiment_with_llm(
                                page,
                                _DRIVE_TIMEOUT_MS,
                                project_id,
                                run_id,
                                logs_dir,
                                state,
                            )
                            log_status(
                                "Steering: LLM (Gemini)"
                                if llm_used
                                else "Steering: blind (LLM unavailable or prompt missing)"
                            )
                            if not done:
                                if llm_used:
                                    log_status(
                                        "LLM did not finish in time; falling back to blind steering."
                                    )
                                done = _drive_experiment_to_finish(
                                    page, _DRIVE_TIMEOUT_MS
                                )
                            if done:
                                page.wait_for_timeout(1500)
                            else:
                                print(
                                    f"  Run {run_idx + 1}/{n_participants}: timed out before experiment finished (no POST).",
                                    file=sys.stderr,
                                    flush=True,
                                )
                        except Exception as exc:
                            err_msg = f"Run {run_idx + 1}/{n_participants} error: {exc}"
                            print(err_msg, file=sys.stderr, flush=True)
                            (logs_dir / "browser_error.txt").write_text(
                                err_msg, encoding="utf-8"
                            )
                        finally:
                            page.close()
                finally:
                    browser.close()
        log_status("Fetching /results...")

    url = _results_url(results_api_url, config, project_id, run_id)
    try:
        req = _results_request(url)
        with urllib.request.urlopen(req, timeout=60) as response:
            body = response.read().decode("utf-8")
    except Exception as exc:
        # As in _collect_live: a failed fetch is not "no responses".
        err_msg = f"Firebase results fetch failed: {exc}"
        print(err_msg, file=sys.stderr, flush=True)
        (logs_dir / "browser_error.txt").write_text(err_msg, encoding="utf-8")
        raise RuntimeError(
            f"Firebase results fetch failed for {url}: {type(exc).__name__}: {exc}. "
            "Refusing to report zero responses for a collection that may have "
            "produced data."
        ) from exc

    rows: list[dict[str, Any]] = []
    if not body.strip():
        log_status("/results returned no data.")
        return rows

    for row in csv.DictReader(io.StringIO(body)):
        rows.append(dict(row))

    if rows and "participant_id_str" in rows[0]:
        allowed = set(participant_ids)
        filtered = [row for row in rows if row.get("participant_id_str") in allowed]
        if filtered:
            participant_index = {
                participant_id: idx
                for idx, participant_id in enumerate(participant_ids)
            }
            for row in filtered:
                row["participant_id"] = participant_index.get(
                    row.get("participant_id_str"), 0
                )
            rows = filtered
            log_status(
                f"Filtered to {len(rows)} rows from this run's {len(participant_ids)} participants."
            )
    log_status(f"Done. Got {len(rows)} response rows from Firestore.")
    return rows


def _results_url(base_url: str, config: dict[str, Any], project_id: str, run_id: int | str) -> str:
    if config.get("collection_session_id"):
        query = urllib.parse.urlencode({"collection_session_id": str(config["collection_session_id"])})
    else:
        query = urllib.parse.urlencode({"run_id": str(run_id), "project_id": str(project_id)})
    return f"{base_url.rstrip('/')}/results?{query}"


def _results_request(url: str) -> urllib.request.Request:
    """Build the (authenticated) /results request.

    The deployed /results endpoint is token-guarded — without the token anyone
    with the public config could read all participant data. A missing token in
    the collecting environment is a misconfiguration and fails loudly rather
    than surfacing as a 403 that reads like "no participants yet". Local test
    servers (plain http) are exempt.
    """
    headers = {"User-Agent": "auto-psych"}
    if url.startswith("https://"):
        token = os.environ.get("AUTO_PSYCH_RESULTS_TOKEN", "").strip()
        if not token:
            raise RuntimeError(
                "AUTO_PSYCH_RESULTS_TOKEN is not set — cannot fetch the "
                "token-guarded /results endpoint. Export the same secret the "
                "deployment provisioned into functions/.env."
            )
        headers["x-results-token"] = token
    return urllib.request.Request(url, headers=headers)


def _parse_participant_answer(text: str) -> str | None:
    """Parse a participant reply into ``"left"``/``"right"``, or ``None`` if it did
    not clearly commit to one.

    Only the explicit ``ANSWER: left|right`` form (the format the participant is
    instructed to use) or a reply that is exactly ``left``/``right`` counts. A
    loose "does the word 'left' appear anywhere?" scan is deliberately NOT used:
    chain-of-thought that merely mentions a side ("the left column looks…",
    "I'd lean left but it's close") would be miscoded as a committed choice, and
    because presentation side is fixed that miscoding would bias the data toward
    one side. An ambiguous reply is better counted as unparseable (and dropped)
    than silently mis-sided.
    """
    if not text or not isinstance(text, str):
        return None
    match = re.search(r"ANSWER\s*:\s*(left|right)\b", text, re.IGNORECASE)
    if match:
        return match.group(1).lower()
    normalized = text.strip().lower()
    if normalized in ("left", "right"):
        return normalized
    return None


def generate_llm_participant_rows(
    stimuli: list[dict[str, Any]],
    n_participants: int,
    *,
    participant_model: Any,
    prompt_text: str,
    transcripts_dir: Path | None = None,
    max_workers: int = 8,
    progress: Callable[[int, int], None] | None = None,
    seed: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Run an LLM-as-participant over ``stimuli`` and return ``(rows, stats)``.

    Model-agnostic: ``participant_model`` is any object exposing
    ``.answer(system, user) -> str`` and a ``.name`` (see
    ``participants.ParticipantModel``), so the closed (API) and open (Hugging
    Face) backends share this one loop. A backend may expose a positive integer
    ``max_concurrency`` to limit simultaneous calls on its shared instance. Each
    participant answers every stimulus;
    unparseable replies and per-trial errors are counted but never abort the run.
    If ``transcripts_dir`` is given, one Markdown transcript per participant is
    written there.

    Participants run concurrently (up to ``max_workers``) since each one's API
    calls are independent and I/O-bound; output rows are reassembled in
    participant order so the result is deterministic regardless of finish order.
    ``progress(participant_id, n_rows)`` is called as each participant completes.

    ``stats`` carries ``n_participants``, ``n_stimuli``, ``n_rows``,
    ``n_unparseable``, ``n_errors``.
    """
    model_name = getattr(participant_model, "name", "llm_participant")

    def _run_participant(participant_id: int) -> tuple[list[dict[str, Any]], int, int]:
        rows_p: list[dict[str, Any]] = []
        unparseable = 0
        errors = 0
        # Per-participant seeded RNG so the side counterbalancing is reproducible
        # regardless of the (concurrent) finish order of participants.
        prng = random.Random(seed + participant_id * 1_000_003)
        transcript = None
        if transcripts_dir is not None:
            transcripts_dir.mkdir(parents=True, exist_ok=True)
            transcript = (
                transcripts_dir / f"participant_{participant_id:03d}.md"
            ).open("w", encoding="utf-8")
            transcript.write(
                f"# Participant {participant_id} transcript ({model_name})\n\n"
            )
        try:
            for trial_index, stimulus in enumerate(stimuli):
                # A malformed stimulus must fail here, at its source: defaulting
                # a missing key to "" would show the participant a blank option
                # and record an empty sequence as if it were a real trial.
                missing = [
                    key
                    for key in ("sequence_a", "sequence_b")
                    if not stimulus.get(key)
                ]
                if missing:
                    raise ValueError(
                        f"stimulus {trial_index} is missing/empty {missing}: "
                        f"{stimulus!r}. Fix the design's stimuli.json rather than "
                        "presenting a blank option."
                    )
                # Randomize which sequence is shown on the left per trial; the
                # presented order is what we show AND what we record (a = left).
                swap = prng.random() < 0.5
                left, right = _present_sides(
                    str(stimulus["sequence_a"]),
                    str(stimulus["sequence_b"]),
                    swap,
                )
                user_msg = (
                    "Stimulus pair (left vs right):\n"
                    f"  Left:  {left}\n"
                    f"  Right: {right}\n\n"
                    "Reply with exactly one line: `ANSWER: left` or `ANSWER: right`."
                )
                try:
                    response = participant_model.answer(prompt_text, user_msg)
                except Exception as exc:
                    errors += 1
                    if transcript is not None:
                        transcript.write(
                            f"## Trial {trial_index}\n- left: `{left}`\n- right: `{right}`\n\n"
                            f"**Model error:** {exc}\n\n"
                        )
                    continue
                choice = _parse_participant_answer(response)
                if transcript is not None:
                    transcript.write(
                        f"## Trial {trial_index}\n- left: `{left}`\n- right: `{right}`\n\n"
                        f"**Reply:**\n```\n{response.strip()}\n```\n\n"
                        f"**Parsed:** {choice if choice else 'UNPARSEABLE'}\n\n"
                    )
                if choice is None:
                    unparseable += 1
                    continue
                chose_left = choice == "left"
                rows_p.append(
                    {
                        "participant_id": participant_id,
                        "trial_index": trial_index,
                        "sequence_a": left,
                        "sequence_b": right,
                        "chose_left": int(chose_left),
                        "chose_right": int(not chose_left),
                        "model": model_name,
                    }
                )
        finally:
            if transcript is not None:
                transcript.close()
        if progress is not None:
            progress(participant_id, len(rows_p))
        return rows_p, unparseable, errors

    results: list[tuple[list[dict[str, Any]], int, int] | None] = [None] * n_participants
    model_limit = getattr(participant_model, "max_concurrency", max_workers)
    if isinstance(model_limit, bool) or not isinstance(model_limit, int) or model_limit < 1:
        raise ValueError(
            f"participant model max_concurrency must be a positive integer, got "
            f"{model_limit!r}"
        )
    workers = max(1, min(max_workers, model_limit, n_participants))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_run_participant, pid): pid for pid in range(n_participants)
        }
        for future in as_completed(futures):
            results[futures[future]] = future.result()

    rows: list[dict[str, Any]] = []
    n_unparseable = 0
    n_errors = 0
    for result in results:
        assert result is not None  # every participant future populated its slot
        rows_p, unparseable, errors = result
        rows.extend(rows_p)
        n_unparseable += unparseable
        n_errors += errors

    stats = {
        "n_participants": n_participants,
        "n_stimuli": len(stimuli),
        "n_rows": len(rows),
        "n_unparseable": n_unparseable,
        "n_errors": n_errors,
    }
    return rows, stats


def _load_featurizer(featurize_path: Path | None) -> Featurizer | None:
    """Return featurize_stimulus(seq_a, seq_b) from a module path.

    None only when no featurizer was configured at all; a configured-but-broken
    one raises (see ``featurizer.load_featurizer``).
    """
    if featurize_path is None:
        return None
    return load_featurizer(featurize_path)


def _present_sides(seq_a: str, seq_b: str, swap: bool) -> tuple[str, str]:
    """Return the presented ``(left, right)`` order: canonical, or swapped.

    Side counterbalancing: each trial independently randomizes which sequence is
    shown on the LEFT, and we record the *presented* order into
    ``sequence_a`` (= left) / ``sequence_b`` (= right). The cognitive models treat
    ``sequence_a`` as the left option, so randomizing the side per trial decouples
    content from physical side — making each model's ``side_bias`` a genuine,
    identifiable physical-left preference — with no model or schema change.
    """
    return (seq_b, seq_a) if swap else (seq_a, seq_b)


def _generate_from_pymc_models(
    stimuli: list[dict[str, Any]],
    model_names: list[str],
    n_participants: int,
    *,
    models_dir: Path,
    featurize_path: Path | None = None,
    n_samples: int = 200,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Generate synthetic responses by sampling each model's prior-predictive p_left.

    Each participant is assigned a random theorist model; for every stimulus the
    model's prior-predictive mean p_left is the choice probability for a binary
    draw. The side each sequence is shown on is randomized per trial (see
    :func:`_present_sides`), and the model is evaluated on the *presented* order so
    a generative ``side_bias`` biases toward the physical left. Raw stimuli are
    featurized so the PyMC `pm.Data` columns are present. No MCMC fit — the prior
    is the generative process for synthetic participants.
    """
    from src.models.pymc_inference import prior_predict_p_left

    featurize = _load_featurizer(featurize_path)
    rng = random.Random(seed)

    # Cache prior-predictive p_left per (model, stimulus, side) — deterministic
    # given the engine seed. Each stimulus recurs across participants in both its
    # canonical and side-swapped presentation, so both orders are worth caching.
    p_left_cache: dict[tuple[str, int, bool], float] = {}

    def _feature_row(left: str, right: str) -> dict[str, Any]:
        row: dict[str, Any] = {"sequence_a": left, "sequence_b": right}
        if featurize is not None:
            row.update(featurize(left, right))
        row.setdefault("chose_left", 0)  # dummy observed value; unused for p_left
        return row

    def _p_left(model_name: str, stim_idx: int, left: str, right: str, swap: bool) -> float:
        key = (model_name, stim_idx, swap)
        if key not in p_left_cache:
            preds = prior_predict_p_left(
                [model_name], models_dir, _feature_row(left, right),
                n_samples=n_samples, seed=seed,
            )
            p_left_cache[key] = preds[model_name]
        return p_left_cache[key]

    rows: list[dict[str, Any]] = []
    for participant_id in range(n_participants):
        model_name = rng.choice(model_names)
        for trial_index, stimulus in enumerate(stimuli):
            swap = rng.random() < 0.5
            left, right = _present_sides(
                stimulus["sequence_a"], stimulus["sequence_b"], swap
            )
            p_left = _p_left(model_name, trial_index, left, right, swap)
            chose_left = rng.random() < p_left
            rows.append(
                {
                    "participant_id": participant_id,
                    "trial_index": trial_index,
                    "sequence_a": left,
                    "sequence_b": right,
                    "chose_left": int(chose_left),
                    "chose_right": int(not chose_left),
                    "model": model_name,
                }
            )
    return rows


def _generate_from_models(
    stimuli: list[dict[str, Any]],
    model_names: list[str],
    n_participants: int,
    theorist_dir: Path | None = None,
    model_registry: dict[str, Any] | None = None,
    seed: int = 0,
) -> list[dict[str, Any]]:
    # Use a seeded local RNG (not the unseeded global `random`) so synthetic
    # ground-truth data is reproducible across runs, like _generate_from_pymc_models.
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    for participant_id in range(n_participants):
        model_name = rng.choice(model_names)
        for trial_index, stimulus in enumerate(stimuli):
            # Randomize which sequence is shown on the left per trial; evaluate the
            # ground-truth model on the PRESENTED order and record it (a = left).
            swap = rng.random() < 0.5
            seq_a, seq_b = _present_sides(
                stimulus["sequence_a"], stimulus["sequence_b"], swap
            )
            stimulus_tuple = (seq_a, seq_b)
            if model_registry is not None and model_name in model_registry:
                fn = model_registry[model_name]
                preds = {model_name: fn(stimulus_tuple, RESPONSE_OPTIONS)}
            else:
                preds = get_model_predictions(
                    stimulus_tuple, RESPONSE_OPTIONS, [model_name], theorist_dir
                )
            if not preds:
                # Backstop. get_model_predictions now raises rather than dropping
                # a model, so this fires only if a registry callable returns
                # nothing. Substituting a coin flip would emit pure noise labeled
                # as this model's ground-truth data and feed it straight into model
                # comparison. Fail loudly instead of fabricating data.
                raise RuntimeError(
                    f"ground-truth model {model_name!r} produced no prediction for "
                    f"stimulus {stimulus_tuple}; refusing to substitute random "
                    "responses. Check the model loads and returns a left/right "
                    "distribution."
                )
            distribution = validate_probability_distribution(
                preds[model_name],
                RESPONSE_OPTIONS,
                context=f"ground-truth model {model_name!r}",
            )
            p_left = distribution["left"]
            chose_left = rng.random() < p_left
            rows.append(
                {
                    "participant_id": participant_id,
                    "trial_index": trial_index,
                    "sequence_a": seq_a,
                    "sequence_b": seq_b,
                    "chose_left": int(chose_left),
                    "chose_right": int(not chose_left),
                    "model": model_name,
                }
            )
    return rows
