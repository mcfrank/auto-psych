"""Collection support for the active outer loop."""

from __future__ import annotations

import csv
import io
import json
import multiprocessing
import random
import re
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from src.pipelines.outer_loop.llm import get_llm, invoke_llm, load_prompt_for_run
from src.runtime.config import DEFAULT_MAX_VALIDATION_RETRIES, agent_dir_for_state
from src.runtime.console import agent_header, log_status
from src.models.project.ground_truth import get_ground_truth_models
from src.models.theorist.loader import get_model_names_from_manifest
from src.models.theorist.predictions import get_model_predictions
from src.runtime.observability import agent_log

RESPONSE_OPTIONS = ["left", "right"]
MAX_PARALLEL_PARTICIPANTS = 3
_DRIVE_INTERVAL_MS = 1200
_DRIVE_TIMEOUT_MS = 180_000
_LLM_CONTEXT_MAX_SCREENS = 20
_FIXATION_WAIT_SEC = 0.25
_PROLIFIC_POLL_INTERVAL_SEC = 30


def _get_screen_content(page) -> str:
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
    except Exception:
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


def _drive_experiment_with_llm(
    page,
    timeout_ms: int,
    project_id: str,
    run_id: int,
    logs_dir: Path,
    state: dict | None = None,
) -> tuple[bool, bool]:
    try:
        llm = get_llm()
    except Exception:
        return (False, False)

    steering_prompt = load_prompt_for_run(
        project_id, run_id, "4_collect_steering", state
    )
    if not steering_prompt.strip():
        return (False, False)

    deadline = time.monotonic() + (timeout_ms / 1000.0)
    context_parts: list[str] = []
    while time.monotonic() < deadline:
        try:
            if page.evaluate("typeof window.__experimentData !== 'undefined'"):
                return (True, True)
        except Exception:
            pass

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
            response = invoke_llm(system=steering_prompt, user=user_msg, llm=llm)
        except Exception as exc:
            print(f"  [LLM steering error] {exc}", file=sys.stderr, flush=True)
            (logs_dir / "llm_steering_error.txt").write_text(str(exc), encoding="utf-8")
            return (False, True)

        action = _parse_steering_action(response)
        if action is None:
            try:
                page.locator("button.jspsych-btn").first.click(timeout=500)
            except Exception:
                page.keyboard.press("f")
        elif action[0] == "click":
            try:
                page.get_by_role("button", name=action[1]).click(timeout=2000)
            except Exception:
                try:
                    page.locator("button.jspsych-btn").first.click(timeout=1000)
                except Exception:
                    page.keyboard.press("f")
        else:
            page.keyboard.press(action[1])

        context_parts.append(f"Your action: {response.strip()[:150]}")
        if len(context_parts) > _LLM_CONTEXT_MAX_SCREENS * 4:
            context_parts = context_parts[-(_LLM_CONTEXT_MAX_SCREENS * 4) :]
        time.sleep(0.1)

    try:
        return (
            bool(page.evaluate("typeof window.__experimentData !== 'undefined'")),
            True,
        )
    except Exception:
        return (False, True)


def _drive_experiment_to_finish(page, timeout_ms: int = _DRIVE_TIMEOUT_MS) -> bool:
    deadline = time.monotonic() + (timeout_ms / 1000.0)
    step_sec = _DRIVE_INTERVAL_MS / 1000.0
    while time.monotonic() < deadline:
        try:
            if page.evaluate("typeof window.__experimentData !== 'undefined'"):
                return True
        except Exception:
            pass
        try:
            page.locator("button.jspsych-btn").first.click(timeout=400)
        except Exception:
            pass
        try:
            page.keyboard.press("f")
        except Exception:
            pass
        time.sleep(step_sec)
    try:
        return bool(page.evaluate("typeof window.__experimentData !== 'undefined'"))
    except Exception:
        return False


def _rows_from_trial_data(
    trials: list[dict[str, Any]],
    participant_id: int,
    participant_id_str: str | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for trial_index, trial in enumerate(trials):
        if "sequence_a" not in trial or "sequence_b" not in trial:
            continue
        chose_left = trial.get("chose_left")
        if chose_left is None:
            continue
        row = {
            "participant_id": participant_id,
            "trial_index": trial_index,
            "sequence_a": str(trial["sequence_a"]),
            "sequence_b": str(trial["sequence_b"]),
            "chose_left": int(bool(chose_left)),
            "chose_right": 1 - int(bool(chose_left)),
            "model": "",
        }
        if participant_id_str is not None:
            row["participant_id_str"] = participant_id_str
        rows.append(row)
    return rows


def _run_one_participant_browser(
    args: tuple,
) -> tuple[int, list[dict[str, Any]] | None, str | None]:
    (
        participant_index,
        participant_id_str,
        experiment_url,
        project_id,
        run_id,
        timeout_ms,
        logs_dir_path,
    ) = args
    logs_dir = Path(logs_dir_path) if logs_dir_path else None
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return (participant_index, None, "playwright not installed")
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
                page.goto(goto_url, wait_until="load", timeout=timeout_ms)
                done, _ = _drive_experiment_with_llm(
                    page,
                    min(timeout_ms, _DRIVE_TIMEOUT_MS),
                    project_id,
                    run_id,
                    logs_dir,
                )
                if not done:
                    done = _drive_experiment_to_finish(
                        page, min(timeout_ms, _DRIVE_TIMEOUT_MS)
                    )
                if not done:
                    return (
                        participant_index,
                        None,
                        "timed out before experiment finished",
                    )
                data = page.evaluate("window.__experimentData")
            finally:
                browser.close()
        if data is None or not isinstance(data, list):
            return (participant_index, None, "no __experimentData")
        return (
            participant_index,
            _rows_from_trial_data(data, participant_index, participant_id_str),
            None,
        )
    except Exception as exc:
        if logs_dir:
            (logs_dir / f"p{participant_index}_error.txt").write_text(
                str(exc), encoding="utf-8"
            )
        return (participant_index, None, str(exc))


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


def run_collect(state: dict[str, Any]) -> dict[str, Any]:
    project_id = state["project_id"]
    run_id = state["run_id"]
    if state.get("last_validated_agent") != "4_collect":
        state = {**state, "validation_retry_count": 0, "validation_feedback": ""}
    if state.get("validation_retry_count", 0) == 0:
        agent_header("4_collect", run_id, state.get("total_runs"), state.get("mode"))
    elif state.get("validation_retry_count", 0) > 0:
        max_retries = state.get(
            "max_validation_retries", DEFAULT_MAX_VALIDATION_RETRIES
        )
        log_status(
            f"Repeating due to validation failure (attempt {state['validation_retry_count']}/{max_retries})"
        )

    out_dir = agent_dir_for_state(project_id, run_id, "4_collect", state)
    out_dir.mkdir(parents=True, exist_ok=True)
    attempt = (state.get("validation_retry_count") or 0) + 1
    validation_feedback = (state.get("validation_feedback") or "").strip()
    agent_log(out_dir, "=== 4_collect start ===")
    agent_log(
        out_dir,
        f"project_id={project_id!r} run_id={run_id} attempt={attempt} mode={state.get('mode')!r}",
    )
    if validation_feedback:
        agent_log(out_dir, f"Validation feedback: {validation_feedback[:500]}")

    logs_dir = out_dir / "logs"
    logs_dir.mkdir(exist_ok=True)

    stimuli_path = Path(state["stimuli_path"])
    stimuli = json.loads(stimuli_path.read_text()) if stimuli_path.exists() else []
    manifest_path = Path(state["theorist_manifest_path"])
    manifest = (
        yaml.safe_load(manifest_path.read_text()) if manifest_path.exists() else {}
    )
    theorist_dir = manifest_path.parent if manifest_path.exists() else None
    model_names = (
        get_model_names_from_manifest(manifest, theorist_dir) if theorist_dir else []
    )

    config_path = Path(state["deployment_config_path"])
    config = json.loads(config_path.read_text()) if config_path.exists() else {}

    ground_truth_model = state.get("ground_truth_model")
    if state.get("mode") in ("live", "test_prolific"):
        rows = _collect_live(state, config, out_dir, logs_dir)
    elif ground_truth_model:
        n_participants = config.get("simulated_n_participants", 5)
        registry = get_ground_truth_models(project_id)
        if ground_truth_model not in registry:
            agent_log(
                out_dir,
                f"Ground-truth model {ground_truth_model!r} not in project registry {list(registry.keys())}; skipping data generation.",
            )
            rows = []
        else:
            agent_log(
                out_dir,
                f"Ground-truth model={ground_truth_model!r}; generating data from project ground-truth only (no browser).",
            )
            rows = _generate_from_models(
                stimuli, [ground_truth_model], n_participants, model_registry=registry
            )
        model_names = [ground_truth_model]
        (logs_dir / "ground_truth_model.txt").write_text(
            ground_truth_model, encoding="utf-8"
        )
    elif state.get("mode") == "simulated_participants_nobrowser":
        agent_log(
            out_dir,
            "Mode=simulated_participants_nobrowser; using LLM-as-participant (no browser, no Firebase).",
        )
        rows = _collect_llm_participant(state, config, out_dir, logs_dir, stimuli)
        model_names = ["llm_participant"]
    else:
        rows = _collect_simulated(
            state, config, out_dir, logs_dir, stimuli, model_names, theorist_dir
        )

    batch_id = Path(state["batch_dir"]).name if state.get("batch_dir") else ""
    for row in rows:
        row["batch_id"] = batch_id

    csv_path = out_dir / "responses.csv"
    agent_log(out_dir, f"collected n_rows={len(rows) if rows else 0}")
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    n_participants = (
        config.get("simulated_n_participants", 5)
        if state.get("mode") not in ("live", "test_prolific")
        else len(rows)
    )
    (logs_dir / "n_participants.txt").write_text(str(n_participants), encoding="utf-8")
    (logs_dir / "model_names.txt").write_text("\n".join(model_names), encoding="utf-8")
    agent_log(out_dir, f"wrote {csv_path.name}")
    agent_log(out_dir, "=== 4_collect end ===")
    return {**state, "simulated_data_path": str(csv_path)}


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
    if not study_id:
        agent_log(
            out_dir,
            "Collect (live): error - no prolific_study_id in config; Prolific flow not configured",
        )
        return []
    if not results_api_url:
        agent_log(
            out_dir,
            "Collect (live): error - no results_api_url/experiment_url to fetch results",
        )
        return []

    try:
        from src.runtime.prolific import get_submission_counts
    except ImportError:
        agent_log(out_dir, "Collect (live): prolific_client not available")
        return []

    completed = 0
    while completed < target_places:
        counts, err = get_submission_counts(study_id)
        if err:
            agent_log(out_dir, f"Prolific poll: study_id={study_id!r} error={err!r}")
            time.sleep(_PROLIFIC_POLL_INTERVAL_SEC)
            continue
        completed = int(counts.get("COMPLETED") or counts.get("completed") or 0)
        agent_log(
            out_dir,
            f"Prolific poll: study_id={study_id!r} completed={completed} target={target_places}",
        )
        if completed >= target_places:
            break
        time.sleep(_PROLIFIC_POLL_INTERVAL_SEC)

    agent_log(out_dir, "Collect (live): fetching results from Firebase")
    url = _results_url(results_api_url, config, project_id, run_id)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "auto-psych"})
        with urllib.request.urlopen(req, timeout=60) as response:
            body = response.read().decode("utf-8")
    except Exception as exc:
        agent_log(out_dir, f"Collect (live): results fetch failed: {exc}")
        return []

    rows: list[dict[str, Any]] = []
    if body.strip():
        for row in csv.DictReader(io.StringIO(body)):
            rows.append(dict(row))
    agent_log(out_dir, f"Collect (live): got {len(rows)} rows from /results")
    return rows


def _collect_simulated(
    state: dict[str, Any],
    config: dict[str, Any],
    out_dir: Path,
    logs_dir: Path,
    stimuli: list[dict[str, Any]],
    model_names: list[str],
    theorist_dir: Path | None = None,
) -> list[dict[str, Any]]:
    n_participants = config.get("simulated_n_participants", 5)
    experiment_url = config.get("experiment_url")
    results_api_url = config.get("results_api_url")

    agent_log(
        out_dir,
        f"n_participants={n_participants} experiment_url={bool(experiment_url)} results_api_url={bool(results_api_url)}",
    )
    if results_api_url:
        return _collect_from_firebase(
            state, config, results_api_url, n_participants, out_dir, logs_dir
        )
    if experiment_url:
        return _collect_from_browser(
            config, experiment_url, n_participants, out_dir, logs_dir
        )
    if not model_names:
        agent_log(
            out_dir, "no theorist models loadable; cannot generate data without URL"
        )
        return []
    return _generate_from_models(
        stimuli, model_names, n_participants, theorist_dir=theorist_dir
    )


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
        return []

    experiment_url = config.get("experiment_url")
    batch_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
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
        except ImportError:
            msg = "playwright not installed; run: pip install playwright && playwright install chromium"
            print(msg, file=sys.stderr, flush=True)
            (logs_dir / "browser_error.txt").write_text(msg, encoding="utf-8")
            return []

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
        req = urllib.request.Request(url, headers={"User-Agent": "auto-psych"})
        with urllib.request.urlopen(req, timeout=60) as response:
            body = response.read().decode("utf-8")
    except Exception as exc:
        err_msg = f"Firebase results fetch failed: {exc}"
        print(err_msg, file=sys.stderr, flush=True)
        (logs_dir / "browser_error.txt").write_text(err_msg, encoding="utf-8")
        return []

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


def _server_reachable(url: str, timeout_sec: float = 2.0) -> bool:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "auto-psych"})
        urllib.request.urlopen(req, timeout=timeout_sec)
        return True
    except Exception:
        return False


def _start_experiment_server(
    experiment_path: str, port: int
) -> subprocess.Popen | None:
    exp_dir = Path(experiment_path)
    if not exp_dir.exists():
        return None
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "http.server",
            str(port),
            "--bind",
            "127.0.0.1",
            "--directory",
            str(exp_dir),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _collect_from_browser(
    config: dict[str, Any],
    experiment_url: str,
    n_participants: int,
    out_dir: Path,
    logs_dir: Path,
) -> list[dict[str, Any]]:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        msg = "playwright not installed; run: pip install playwright && playwright install chromium"
        print(msg, file=sys.stderr, flush=True)
        (logs_dir / "browser_error.txt").write_text(msg, encoding="utf-8")
        return []

    server_proc: subprocess.Popen | None = None
    if not _server_reachable(experiment_url):
        experiment_path = config.get("experiment_path", "")
        port = config.get("server_port", 8765)
        server_proc = _start_experiment_server(experiment_path, port)
        if server_proc is not None:
            for _ in range(15):
                time.sleep(0.5)
                if _server_reachable(experiment_url):
                    break
            if not _server_reachable(experiment_url):
                server_proc.terminate()
                server_proc.wait(timeout=3)
                msg = "Started server but it did not become reachable; falling back to model-based data."
                print(msg, file=sys.stderr, flush=True)
                (logs_dir / "browser_error.txt").write_text(msg, encoding="utf-8")
                return []

    project_id = config.get("project_id", "")
    run_id = config.get("run_id", 0)
    batch_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    participant_ids = [
        f"{project_id}_run{run_id}_{batch_id}_p{i}" for i in range(n_participants)
    ]
    (logs_dir / "participant_ids.txt").write_text(
        "\n".join(participant_ids), encoding="utf-8"
    )
    log_status(f"Participant IDs for this run: {logs_dir / 'participant_ids.txt'}")

    rows: list[dict[str, Any]] = []
    timeout_ms = 120_000
    n_parallel = (
        min(n_participants, MAX_PARALLEL_PARTICIPANTS)
        if MAX_PARALLEL_PARTICIPANTS >= 2
        else 1
    )
    try:
        if n_parallel >= 2 and n_participants >= 2:
            log_status(
                f"Running {n_participants} browser participant(s) (local, {n_parallel} in parallel)..."
            )
            worker_args = [
                (
                    participant_id,
                    participant_ids[participant_id],
                    experiment_url,
                    project_id,
                    run_id,
                    timeout_ms,
                    str(logs_dir),
                )
                for participant_id in range(n_participants)
            ]
            with multiprocessing.Pool(processes=n_parallel) as pool:
                results = pool.map(_run_one_participant_browser, worker_args)
            for idx, participant_rows, err in sorted(
                results, key=lambda result: result[0]
            ):
                if err:
                    print(
                        f"  Participant {idx + 1}/{n_participants}: {err}",
                        file=sys.stderr,
                        flush=True,
                    )
                elif participant_rows:
                    rows.extend(participant_rows)
            if n_participants:
                log_status("Steering: LLM (Gemini)")
            log_status(f"Done. Got {len(rows)} response rows.")
        else:
            log_status(f"Running {n_participants} browser participant(s) (local)...")
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(headless=True)
                try:
                    for participant_id in range(n_participants):
                        participant_id_str = participant_ids[participant_id]
                        goto_url = (
                            experiment_url
                            + ("&" if "?" in experiment_url else "?")
                            + "participant_id="
                            + urllib.parse.quote(participant_id_str)
                        )
                        log_status(
                            f"Participant {participant_id + 1}/{n_participants} in progress..."
                        )
                        page = browser.new_page()
                        data = None
                        try:
                            page.goto(goto_url, wait_until="load", timeout=timeout_ms)
                            done, llm_used = _drive_experiment_with_llm(
                                page,
                                min(timeout_ms, _DRIVE_TIMEOUT_MS),
                                config.get("project_id", ""),
                                config.get("run_id", 0),
                                logs_dir,
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
                                    page, min(timeout_ms, _DRIVE_TIMEOUT_MS)
                                )
                            if done:
                                data = page.evaluate("window.__experimentData")
                            else:
                                print(
                                    f"  Run {participant_id + 1}/{n_participants}: timed out before experiment finished.",
                                    file=sys.stderr,
                                    flush=True,
                                )
                        except Exception as exc:
                            print(
                                f"  Run {participant_id + 1}/{n_participants} error: {exc}",
                                file=sys.stderr,
                                flush=True,
                            )
                        finally:
                            try:
                                page.close()
                            except Exception:
                                pass
                        if data is not None and isinstance(data, list):
                            rows.extend(
                                _rows_from_trial_data(
                                    data, participant_id, participant_id_str
                                )
                            )
                finally:
                    browser.close()
            log_status(f"Done. Got {len(rows)} response rows.")
    finally:
        if server_proc is not None and server_proc.poll() is None:
            server_proc.terminate()
            try:
                server_proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                server_proc.kill()

    return rows


def _parse_participant_answer(text: str) -> str | None:
    if not text or not isinstance(text, str):
        return None
    match = re.search(r"ANSWER\s*:\s*(left|right)\b", text, re.IGNORECASE)
    if match:
        return match.group(1).lower()
    normalized = text.strip().lower()
    if normalized in ("left", "right"):
        return normalized
    has_left = bool(re.search(r"\bleft\b", normalized))
    has_right = bool(re.search(r"\bright\b", normalized))
    if has_left and not has_right:
        return "left"
    if has_right and not has_left:
        return "right"
    return None


def generate_llm_participant_rows(
    stimuli: list[dict[str, Any]],
    n_participants: int,
    *,
    participant_model: Any,
    prompt_text: str,
    transcripts_dir: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Run an LLM-as-participant over ``stimuli`` and return ``(rows, stats)``.

    Model-agnostic: ``participant_model`` is any object exposing
    ``.answer(system, user) -> str`` and a ``.name`` (see
    ``participants.ParticipantModel``), so the closed (API) and open (Hugging
    Face) backends share this one loop. Each participant answers every stimulus;
    unparseable replies and per-trial errors are counted but never abort the run.
    If ``transcripts_dir`` is given, one Markdown transcript per participant is
    written there.

    ``stats`` carries ``n_participants``, ``n_stimuli``, ``n_rows``,
    ``n_unparseable``, ``n_errors``.
    """
    model_name = getattr(participant_model, "name", "llm_participant")
    rows: list[dict[str, Any]] = []
    n_unparseable = 0
    n_errors = 0

    for participant_id in range(n_participants):
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
                seq_a = stimulus.get("sequence_a", "")
                seq_b = stimulus.get("sequence_b", "")
                user_msg = (
                    "Stimulus pair (left vs right):\n"
                    f"  Left:  {seq_a}\n"
                    f"  Right: {seq_b}\n\n"
                    "Reply with exactly one line: `ANSWER: left` or `ANSWER: right`."
                )
                try:
                    response = participant_model.answer(prompt_text, user_msg)
                except Exception as exc:
                    n_errors += 1
                    if transcript is not None:
                        transcript.write(
                            f"## Trial {trial_index}\n- left: `{seq_a}`\n- right: `{seq_b}`\n\n"
                            f"**Model error:** {exc}\n\n"
                        )
                    continue
                choice = _parse_participant_answer(response)
                if transcript is not None:
                    transcript.write(
                        f"## Trial {trial_index}\n- left: `{seq_a}`\n- right: `{seq_b}`\n\n"
                        f"**Reply:**\n```\n{response.strip()}\n```\n\n"
                        f"**Parsed:** {choice if choice else 'UNPARSEABLE'}\n\n"
                    )
                if choice is None:
                    n_unparseable += 1
                    continue
                chose_left = choice == "left"
                rows.append(
                    {
                        "participant_id": participant_id,
                        "trial_index": trial_index,
                        "sequence_a": str(seq_a),
                        "sequence_b": str(seq_b),
                        "chose_left": int(chose_left),
                        "chose_right": int(not chose_left),
                        "model": model_name,
                    }
                )
        finally:
            if transcript is not None:
                transcript.close()

    stats = {
        "n_participants": n_participants,
        "n_stimuli": len(stimuli),
        "n_rows": len(rows),
        "n_unparseable": n_unparseable,
        "n_errors": n_errors,
    }
    return rows, stats


def _collect_llm_participant(
    state: dict[str, Any],
    config: dict[str, Any],
    out_dir: Path,
    logs_dir: Path,
    stimuli: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    from src.pipelines.outer_loop.participants import get_participant_model

    project_id = state["project_id"]
    run_id = state["run_id"]
    n_participants = int(config.get("simulated_n_participants", 5))
    backend = config.get("participant_backend", "closed")
    model_name = config.get("participant_model")

    participant_prompt = load_prompt_for_run(
        project_id, run_id, "4_collect_participant", state
    )
    if not participant_prompt.strip():
        agent_log(
            out_dir,
            "LLM-participant: no 4_collect_participant.md prompt found; cannot run.",
        )
        return []

    try:
        participant_model = get_participant_model(backend, model_name)
    except Exception as exc:
        agent_log(
            out_dir,
            f"LLM-participant: failed to initialize participant model ({backend}): {exc}",
        )
        return []

    rows, stats = generate_llm_participant_rows(
        stimuli,
        n_participants,
        participant_model=participant_model,
        prompt_text=participant_prompt,
        transcripts_dir=out_dir / "transcripts",
    )

    agent_log(
        out_dir,
        f"LLM-participant ({participant_model.name}): emitted {stats['n_rows']} rows from "
        f"{stats['n_participants']} participants over {stats['n_stimuli']} stimuli "
        f"(unparseable={stats['n_unparseable']}, errors={stats['n_errors']})",
    )
    (logs_dir / "llm_participant_summary.txt").write_text(
        f"participant_model={participant_model.name}\n"
        + "\n".join(f"{k}={v}" for k, v in stats.items())
        + "\n",
        encoding="utf-8",
    )
    return rows


def _load_featurizer(featurize_path: Path | None):
    """Return featurize_stimulus(seq_a, seq_b) from a module path, or None."""
    if featurize_path is None:
        return None
    import importlib.util

    featurize_path = Path(featurize_path)
    if not featurize_path.exists():
        raise FileNotFoundError(f"featurize module not found: {featurize_path}")
    spec = importlib.util.spec_from_file_location("_collect_featurize", featurize_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load featurize module from {featurize_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, "featurize_stimulus", None)


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
    draw. Raw stimuli are featurized so the PyMC `pm.Data` columns are present.
    No MCMC fit — the prior is the generative process for synthetic participants.
    """
    from src.models.pymc_inference import prior_predict_p_left

    featurize = _load_featurizer(featurize_path)
    rng = random.Random(seed)

    # Cache prior-predictive p_left per (model, stimulus) — it is deterministic
    # given the engine seed, and stimuli repeat across participants.
    p_left_cache: dict[tuple[str, int], float] = {}

    def _p_left(model_name: str, stim_idx: int, feature_row: dict[str, Any]) -> float:
        key = (model_name, stim_idx)
        if key not in p_left_cache:
            preds = prior_predict_p_left(
                [model_name], models_dir, feature_row, n_samples=n_samples, seed=seed
            )
            p_left_cache[key] = preds[model_name]
        return p_left_cache[key]

    feature_rows: list[dict[str, Any]] = []
    for stimulus in stimuli:
        row = dict(stimulus)
        if featurize is not None:
            row.update(featurize(stimulus["sequence_a"], stimulus["sequence_b"]))
        row.setdefault("chose_left", 0)  # dummy observed value; unused for p_left
        feature_rows.append(row)

    rows: list[dict[str, Any]] = []
    for participant_id in range(n_participants):
        model_name = rng.choice(model_names)
        for trial_index, stimulus in enumerate(stimuli):
            p_left = _p_left(model_name, trial_index, feature_rows[trial_index])
            chose_left = rng.random() < p_left
            rows.append(
                {
                    "participant_id": participant_id,
                    "trial_index": trial_index,
                    "sequence_a": stimulus["sequence_a"],
                    "sequence_b": stimulus["sequence_b"],
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
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for participant_id in range(n_participants):
        model_name = random.choice(model_names)
        for trial_index, stimulus in enumerate(stimuli):
            seq_a = stimulus["sequence_a"]
            seq_b = stimulus["sequence_b"]
            stimulus_tuple = (seq_a, seq_b)
            if model_registry is not None and model_name in model_registry:
                fn = model_registry[model_name]
                preds = {model_name: fn(stimulus_tuple, RESPONSE_OPTIONS)}
            else:
                preds = get_model_predictions(
                    stimulus_tuple, RESPONSE_OPTIONS, [model_name], theorist_dir
                )
            if not preds:
                chose_left = random.choice([True, False])
            else:
                p_left = preds[model_name].get("left", 0.5)
                chose_left = random.random() < p_left
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
