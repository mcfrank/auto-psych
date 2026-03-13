"""Simulated participant: visit deployed experiment in a browser or generate from models."""

import csv
import io
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import random
import urllib.request
import urllib.parse
from datetime import datetime

from src.config import agent_dir, run_dir
from src.console_log import agent_header, log_status
from src.models.randomness import MODEL_LIBRARY, get_model_predictions

RESPONSE_OPTIONS = ["left", "right"]

# How often we try to advance the experiment (click button or press key)
_DRIVE_INTERVAL_MS = 1200
# Total time we allow one participant run to complete
_DRIVE_TIMEOUT_MS = 180_000
# Max number of screen/action pairs to keep in LLM context (to avoid token limit)
_LLM_CONTEXT_MAX_SCREENS = 20


def _get_screen_content(page) -> str:
    """Extract visible text from the current experiment screen for the LLM."""
    try:
        return page.evaluate(
            """() => {
              const sel = document.querySelector('#jspsych-content')
                || document.querySelector('.jspsych-content-wrapper')
                || document.querySelector('.jspsych-display-element')
                || document.body;
              return sel ? (sel.innerText || sel.textContent || '').trim() : '';
            }"""
        ) or ""
    except Exception:
        return ""


def _parse_steering_action(text: str) -> Optional[Tuple[str, str]]:
    """
    Parse LLM reply into (action_type, value).
    Returns ('click', 'I agree') or ('key', 'f') etc., or None if unparseable.
    """
    if not text or not isinstance(text, str):
        return None
    text = text.strip()
    # ACTION: click <label> (label can be multiple words)
    m = re.search(r"ACTION:\s*click\s+(.+)", text, re.IGNORECASE | re.DOTALL)
    if m:
        label = m.group(1).strip().split("\n")[0].strip()
        if label:
            return ("click", label)
    # ACTION: key f|j|ArrowLeft|ArrowRight
    m = re.search(r"ACTION:\s*key\s+(f|j|ArrowLeft|ArrowRight)", text, re.IGNORECASE)
    if m:
        key = m.group(1)
        if key.lower() == "f":
            return ("key", "f")
        if key.lower() == "j":
            return ("key", "j")
        if key.lower() == "arrowleft":
            return ("key", "ArrowLeft")
        if key.lower() == "arrowright":
            return ("key", "ArrowRight")
        return ("key", key)
    return None


def _drive_experiment_with_llm(
    page,
    timeout_ms: int,
    project_id: str,
    run_id: int,
    logs_dir: Path,
) -> Tuple[bool, bool]:
    """
    Advance the experiment by sending each screen to the LLM and executing its
    chosen action (click button or press key). Context grows with each screen.
    Returns (done, llm_used): done=True if __experimentData was set; llm_used=True
    if the LLM was actually used (False if LLM/prompt unavailable).
    """
    try:
        from src.agents.base import get_llm, invoke_llm, load_prompt_for_run
    except ImportError:
        return (False, False)
    try:
        llm = get_llm()
    except Exception:
        return (False, False)
    steering_prompt = load_prompt_for_run(project_id, run_id, "5simulated_participant_steering")
    if not steering_prompt.strip():
        from src.config import PROMPTS_DIR
        fallback = PROMPTS_DIR / "5simulated_participant_steering.md"
        if fallback.exists():
            steering_prompt = fallback.read_text(encoding="utf-8")
    if not steering_prompt.strip():
        return (False, False)
    deadline = time.monotonic() + (timeout_ms / 1000.0)
    context_parts: List[str] = []
    step = 0
    while time.monotonic() < deadline:
        try:
            if page.evaluate("typeof window.__experimentData !== 'undefined'"):
                return (True, True)
        except Exception:
            pass
        screen_text = _get_screen_content(page)
        if not screen_text.strip():
            screen_text = "(loading or empty screen)"
        # Fixation screen is just "+" and auto-advances after 500ms; do not ask LLM for a key
        # or that key will be applied to the next screen (the real trial) and we get all "f"
        is_fixation = screen_text.strip() in ("+", "")
        if is_fixation:
            time.sleep(0.6)  # let fixation auto-advance to the trial screen
            continue
        context_parts.append("=== CURRENT SCREEN ===")
        context_parts.append(screen_text)
        context_parts.append("")
        context_parts.append(
            "Reply with exactly one line: ACTION: click <button label> or ACTION: key f|j|ArrowLeft|ArrowRight"
        )
        user_msg = "\n".join(context_parts)
        try:
            response = invoke_llm(system=steering_prompt, user=user_msg, llm=llm)
        except Exception as e:
            print(f"  [LLM steering error] {e}", file=sys.stderr, flush=True)
            if logs_dir:
                (logs_dir / "llm_steering_error.txt").write_text(str(e), encoding="utf-8")
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
        step += 1
        time.sleep(0.2)
    try:
        return (bool(page.evaluate("typeof window.__experimentData !== 'undefined'")), True)
    except Exception:
        return (False, True)


def _drive_experiment_to_finish(page, timeout_ms: int = _DRIVE_TIMEOUT_MS) -> bool:
    """
    Advance the jsPsych experiment until window.__experimentData is set (onFinish ran).
    Clicks visible buttons (I agree, Next, End) and presses 'f' for keyboard trials.
    Returns True if __experimentData was set, False on timeout.
    """
    deadline = time.monotonic() + (timeout_ms / 1000.0)
    step_sec = _DRIVE_INTERVAL_MS / 1000.0
    while time.monotonic() < deadline:
        try:
            if page.evaluate("typeof window.__experimentData !== 'undefined'"):
                return True
        except Exception:
            pass
        try:
            # Click first visible button (consent "I agree", instructions "Next", debrief "End")
            page.locator("button.jspsych-btn").first.click(timeout=400)
        except Exception:
            pass
        try:
            # Advance keyboard trials (f = left; valid for all trial screens)
            page.keyboard.press("f")
        except Exception:
            pass
        time.sleep(step_sec)
    try:
        return bool(page.evaluate("typeof window.__experimentData !== 'undefined'"))
    except Exception:
        return False


def run_simulated_participant(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    If deployer set experiment_url: visit it N times with a headless browser, collect
    jsPsych data from each run, and write responses.csv. Otherwise: generate N
    participants by sampling from theorist models and write CSV.
    """
    project_id = state["project_id"]
    run_id = state["run_id"]
    agent_header("5simulated_participant", run_id, state.get("total_runs"), state.get("mode"))
    if state.get("validation_retry_count", 0) > 0:
        log_status(f"Repeating due to validation failure (attempt {state['validation_retry_count']}/3)")
    out_dir = agent_dir(project_id, run_id, "5simulated_participant")
    out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = out_dir / "logs"
    logs_dir.mkdir(exist_ok=True)

    stimuli_path = Path(state["stimuli_path"])
    stimuli = json.loads(stimuli_path.read_text()) if stimuli_path.exists() else []
    manifest_path = Path(state["theorist_manifest_path"])
    import yaml
    manifest = yaml.safe_load(manifest_path.read_text()) if manifest_path.exists() else {}
    model_names = [m["name"] for m in manifest.get("models", []) if m.get("name") in MODEL_LIBRARY]
    if not model_names:
        model_names = list(MODEL_LIBRARY.keys())

    config_path = Path(state["deployment_config_path"])
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    n_participants = config.get("simulated_n_participants", 5)
    experiment_url = config.get("experiment_url")
    results_api_url = config.get("results_api_url")

    if results_api_url:
        rows = _collect_from_firebase(config, results_api_url, n_participants, out_dir, logs_dir)
    elif experiment_url:
        rows = _collect_from_browser(config, experiment_url, n_participants, out_dir, logs_dir)
    else:
        rows = _generate_from_models(stimuli, model_names, n_participants)

    csv_path = out_dir / "responses.csv"
    if rows:
        fieldnames = list(rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)

    (logs_dir / "n_participants.txt").write_text(str(n_participants), encoding="utf-8")
    (logs_dir / "model_names.txt").write_text("\n".join(model_names), encoding="utf-8")

    return {
        **state,
        "simulated_data_path": str(csv_path),
    }


def _collect_from_firebase(
    config: Dict[str, Any],
    results_api_url: str,
    n_participants: int,
    out_dir: Path,
    logs_dir: Path,
) -> List[Dict[str, Any]]:
    """
    Run experiment in Playwright N times (each POSTs to /submit), then GET /results
    and parse CSV into response rows.
    """
    import csv as csv_module
    project_id = config.get("project_id", "")
    run_id = config.get("run_id", "")
    if not project_id and not run_id:
        return []
    experiment_url = config.get("experiment_url")
    # Participant IDs for this run batch (logged and passed as URL param; used to filter downloaded data)
    batch_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    participant_ids = [f"{project_id}_run{run_id}_{batch_id}_p{i}" for i in range(n_participants)]
    (logs_dir / "participant_ids.txt").write_text("\n".join(participant_ids), encoding="utf-8")
    log_status(f"Participant IDs for this run: {logs_dir / 'participant_ids.txt'}")
    # 1) Run Playwright N times so each run POSTs to /submit
    if experiment_url and n_participants > 0:
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as e:
            msg = "playwright not installed; run: pip install playwright && playwright install chromium"
            print(msg, file=sys.stderr, flush=True)
            (logs_dir / "browser_error.txt").write_text(msg, encoding="utf-8")
            return []
        log_status(f"Running {n_participants} browser participant(s) (Firebase)...")
        nav_timeout_ms = 60_000
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            try:
                for run_idx in range(n_participants):
                    pid = participant_ids[run_idx]
                    goto_url = experiment_url + ("&" if "?" in experiment_url else "?") + "participant_id=" + urllib.parse.quote(pid)
                    log_status(f"Participant {run_idx + 1}/{n_participants} in progress...")
                    page = browser.new_page()
                    try:
                        page.goto(goto_url, wait_until="networkidle", timeout=nav_timeout_ms)
                        # Prefer LLM-driven steering (model makes participant judgments); fallback to blind drive
                        done, llm_used = _drive_experiment_with_llm(
                            page, _DRIVE_TIMEOUT_MS, project_id, run_id, logs_dir
                        )
                        if llm_used:
                            log_status("Steering: LLM (Gemini)")
                        else:
                            log_status("Steering: blind (LLM unavailable or prompt missing)")
                        if not done:
                            if llm_used:
                                log_status("LLM did not finish in time; falling back to blind steering.")
                            done = _drive_experiment_to_finish(page, _DRIVE_TIMEOUT_MS)
                        if done:
                            page.wait_for_timeout(3000)  # allow POST /submit to complete
                        else:
                            print(
                                f"  Run {run_idx + 1}/{n_participants}: timed out before experiment finished (no POST).",
                                file=sys.stderr,
                                flush=True,
                            )
                    except Exception as e:
                        err_msg = f"Run {run_idx + 1}/{n_participants} error: {e}"
                        print(err_msg, file=sys.stderr, flush=True)
                        (logs_dir / "browser_error.txt").write_text(
                            err_msg,
                            encoding="utf-8",
                        )
                    finally:
                        page.close()
            finally:
                browser.close()
        log_status("Fetching /results...")
    # 2) GET /results and parse CSV
    base = results_api_url.rstrip("/")
    url = f"{base}/results?run_id={run_id}&project_id={project_id}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "auto-psych"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = resp.read().decode("utf-8")
    except Exception as e:
        err_msg = f"Firebase results fetch failed: {e}"
        print(err_msg, file=sys.stderr, flush=True)
        (logs_dir / "browser_error.txt").write_text(err_msg, encoding="utf-8")
        return []
    rows = []
    if not body.strip():
        log_status("/results returned no data.")
        return rows
    reader = csv_module.DictReader(io.StringIO(body))
    for r in reader:
        rows.append(dict(r))
    # Filter to only this run's participants (by participant_id_str) so we analyze only this batch
    if rows and "participant_id_str" in rows[0]:
        allowed = set(participant_ids)
        filtered = [r for r in rows if r.get("participant_id_str") in allowed]
        if filtered:
            # Renumber participant_id to 0..n-1 for filtered set (by participant_id_str order)
            pid_str_to_idx = {pid: i for i, pid in enumerate(participant_ids)}
            for r in filtered:
                r["participant_id"] = pid_str_to_idx.get(r.get("participant_id_str"), 0)
            rows = filtered
            log_status(f"Filtered to {len(rows)} rows from this run's {len(participant_ids)} participants.")
    log_status(f"Done. Got {len(rows)} response rows from Firestore.")
    return rows


def _server_reachable(url: str, timeout_sec: float = 2.0) -> bool:
    """Return True if we can GET the URL."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "auto-psych"})
        urllib.request.urlopen(req, timeout=timeout_sec)
        return True
    except Exception:
        return False


def _start_experiment_server(experiment_path: str, port: int) -> Optional[subprocess.Popen]:
    """Start http.server for the experiment dir. Caller must terminate the process when done."""
    exp_dir = Path(experiment_path)
    if not exp_dir.exists():
        return None
    proc = subprocess.Popen(
        [sys.executable, "-m", "http.server", str(port), "--bind", "127.0.0.1", "--directory", str(exp_dir)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return proc


def _collect_from_browser(
    config: Dict[str, Any],
    experiment_url: str,
    n_participants: int,
    out_dir: Path,
    logs_dir: Path,
) -> List[Dict[str, Any]]:
    """Run the experiment in a headless browser N times; extract trial data from window.__experimentData.
    If the server is not reachable, start it from experiment_path (agent 5 owns the process and stops it after).
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as e:
        msg = "playwright not installed; run: pip install playwright && playwright install chromium"
        print(msg, file=sys.stderr, flush=True)
        (logs_dir / "browser_error.txt").write_text(msg, encoding="utf-8")
        return []

    server_proc: Optional[subprocess.Popen] = None
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
    participant_ids = [f"{project_id}_run{run_id}_{batch_id}_p{i}" for i in range(n_participants)]
    (logs_dir / "participant_ids.txt").write_text("\n".join(participant_ids), encoding="utf-8")
    log_status(f"Participant IDs for this run: {logs_dir / 'participant_ids.txt'}")

    rows = []
    timeout_ms = 120_000  # 2 min per run
    log_status(f"Running {n_participants} browser participant(s) (local)...")

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            try:
                for participant_id in range(n_participants):
                    pid = participant_ids[participant_id]
                    goto_url = experiment_url + ("&" if "?" in experiment_url else "?") + "participant_id=" + urllib.parse.quote(pid)
                    log_status(f"Participant {participant_id + 1}/{n_participants} in progress...")
                    page = browser.new_page()
                    data = None
                    try:
                        page.goto(goto_url, wait_until="networkidle", timeout=timeout_ms)
                        done, llm_used = _drive_experiment_with_llm(
                            page,
                            min(timeout_ms, _DRIVE_TIMEOUT_MS),
                            config.get("project_id", ""),
                            config.get("run_id", 0),
                            logs_dir,
                        )
                        if llm_used:
                            log_status("Steering: LLM (Gemini)")
                        else:
                            log_status("Steering: blind (LLM unavailable or prompt missing)")
                        if not done:
                            if llm_used:
                                log_status("LLM did not finish in time; falling back to blind steering.")
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
                    except Exception as e:
                        print(f"  Run {participant_id + 1}/{n_participants} error: {e}", file=sys.stderr, flush=True)
                    finally:
                        try:
                            page.close()
                        except Exception:
                            pass
                    if data is None or not isinstance(data, list):
                        continue
                    # Filter to judgment trials (have sequence_a, sequence_b, chose_left)
                    for i, trial in enumerate(data):
                        if "sequence_a" not in trial or "sequence_b" not in trial:
                            continue
                        chose_left = trial.get("chose_left")
                        if chose_left is None:
                            continue
                        rows.append({
                            "participant_id": participant_id,
                            "participant_id_str": pid,
                            "trial_index": i,
                            "sequence_a": str(trial["sequence_a"]),
                            "sequence_b": str(trial["sequence_b"]),
                            "chose_left": int(bool(chose_left)),
                            "chose_right": 1 - int(bool(chose_left)),
                            "model": "",  # no model when from browser
                        })
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


def _generate_from_models(
    stimuli: List[Dict[str, Any]], model_names: List[str], n_participants: int
) -> List[Dict[str, Any]]:
    """Generate responses by sampling from theorist models (no browser)."""
    rows = []
    for p in range(n_participants):
        model_name = random.choice(model_names)
        for i, stim in enumerate(stimuli):
            seq_a = stim["sequence_a"]
            seq_b = stim["sequence_b"]
            preds = get_model_predictions((seq_a, seq_b), RESPONSE_OPTIONS, [model_name])
            if not preds:
                chose_left = random.choice([True, False])
            else:
                p_left = preds[model_name].get("left", 0.5)
                chose_left = random.random() < p_left
            rows.append({
                "participant_id": p,
                "trial_index": i,
                "sequence_a": seq_a,
                "sequence_b": seq_b,
                "chose_left": int(chose_left),
                "chose_right": int(not chose_left),
                "model": model_name,
            })
    return rows
