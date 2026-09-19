"""Collection support for the active outer loop."""

from __future__ import annotations

import csv
import io
import math
import multiprocessing
import os
import sys
import time
import urllib.parse
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.pipelines.outer_loop.browser_steering import (
    _drive_experiment_to_finish,
    _drive_experiment_with_llm,
    _DRIVE_TIMEOUT_MS,
)
from src.runtime.console import log_status
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

MAX_PARALLEL_PARTICIPANTS = 3
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
