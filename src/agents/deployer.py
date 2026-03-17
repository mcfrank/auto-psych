"""Deployer agent: run simulated participants (local server or Firebase) or configure for live."""

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

from src.config import REPO_ROOT, agent_dir_for_state, DEFAULT_SIMULATED_N_PARTICIPANTS
from src.console_log import agent_header, log_status
from src.observability import agent_log

# Port for local experiment server (so you can open experiment_url in a browser and agent 5 can visit it)
LOCAL_SERVER_PORT = 8765
# Bind explicitly to 127.0.0.1 so browsers connecting to that address succeed (avoids IPv6-only bind)
BIND_ADDRESS = "127.0.0.1"

# Firebase project ID placeholder; real project required for deploy
FIREBASE_PROJECT_PLACEHOLDER = "auto-psych"


def _firebase_project_id() -> Optional[str]:
    """Return Firebase project ID from .firebaserc or env FIREBASE_PROJECT, or None."""
    import os
    pid = os.environ.get("FIREBASE_PROJECT", "").strip()
    if pid:
        return pid
    rc = REPO_ROOT / ".firebaserc"
    if not rc.exists():
        return None
    try:
        data = json.loads(rc.read_text(encoding="utf-8"))
        pid = (data.get("projects") or {}).get("default", "").strip()
        return pid if pid and pid != FIREBASE_PROJECT_PLACEHOLDER else None
    except (json.JSONDecodeError, KeyError):
        return None


def _deploy_to_firebase(
    exp_dir: Path,
    project_id: str,
    run_id: int,
    *,
    extra_config: Optional[Dict[str, Any]] = None,
) -> tuple[Optional[str], Optional[str]]:
    """
    Copy implementer output to public/, write run config for POST /submit, run firebase deploy.
    extra_config: optional dict merged into auto_psych_config.json (e.g. prolific_redirect_url).
    Returns (Hosting base URL on success, None) or (None, error_message on failure).
    """
    fb_project = _firebase_project_id()
    if not fb_project:
        msg = "Firebase skipped: no project (set .firebaserc default or FIREBASE_PROJECT)"
        print(f"Deployer: {msg}", flush=True)
        return (None, msg)
    print(f"Deployer: attempting Firebase deploy (project={fb_project})", flush=True)
    public = REPO_ROOT / "public"
    public.mkdir(parents=True, exist_ok=True)
    # Clear and copy so public/ contains only this run's experiment
    for p in public.iterdir():
        if p.name.startswith("."):
            continue
        if p.is_dir():
            shutil.rmtree(p)
        else:
            p.unlink()
    for src in exp_dir.iterdir():
        if src.name.startswith("."):
            continue
        dst = public / src.name
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
    # Config for experiment onFinish POST /submit (read by experiment via /auto_psych_config.json)
    cfg = {"project_id": project_id, "run_id": run_id}
    if extra_config:
        cfg.update(extra_config)
    (public / "auto_psych_config.json").write_text(
        json.dumps(cfg, indent=0),
        encoding="utf-8",
    )
    # Ensure index.html POSTs to /submit when run on Firebase (patch if built from older template)
    _ensure_firebase_submit_in_index(public / "index.html")
    # Firebase CLI needs hosting "site" to resolve target (must match project ID)
    firebase_json = REPO_ROOT / "firebase.json"
    original_firebase_json = firebase_json.read_text(encoding="utf-8") if firebase_json.exists() else None
    try:
        if firebase_json.exists():
            fc = json.loads(firebase_json.read_text(encoding="utf-8"))
            fc.setdefault("hosting", {})
            if isinstance(fc["hosting"], list):
                fc["hosting"] = fc["hosting"][0] if fc["hosting"] else {}
            fc["hosting"]["site"] = fb_project
            firebase_json.write_text(json.dumps(fc, indent=2), encoding="utf-8")
        last_error = "firebase deploy failed"
        for cmd in (
            ["firebase", "deploy", "--only", "hosting,functions", "--non-interactive", "--force", "--project", fb_project],
            ["npx", "-y", "firebase-tools", "deploy", "--only", "hosting,functions", "--non-interactive", "--force", "--project", fb_project],
        ):
            try:
                result = subprocess.run(
                    cmd, cwd=str(REPO_ROOT), capture_output=True, timeout=300, text=True
                )
                if result.returncode == 0:
                    print(f"Deployer: Firebase deploy OK -> https://{fb_project}.web.app", flush=True)
                    return (f"https://{fb_project}.web.app", None)
                # Combine stdout and stderr; skip Node deprecation noise so real error is visible
                out = (result.stdout or "") + "\n" + (result.stderr or "")
                out = "\n".join(
                    line for line in out.splitlines()
                    if "DEP0040" not in line and "punycode" not in line.lower()
                ).strip() or f"exit code {result.returncode}"
                last_error = out
                print(f"Deployer: firebase deploy failed (exit {result.returncode}): {last_error[:800]}", flush=True)
            except FileNotFoundError:
                last_error = f"Command not found: {cmd[0]}"
                print(f"Deployer: {last_error}", flush=True)
            except subprocess.TimeoutExpired:
                last_error = "firebase deploy timed out (300s)"
                print(f"Deployer: {last_error}", flush=True)
            except Exception as e:
                last_error = str(e)
                print(f"Deployer: firebase deploy error: {e}", flush=True)
        return (None, last_error)
    finally:
        if original_firebase_json is not None:
            firebase_json.write_text(original_firebase_json, encoding="utf-8")


def run_deploy_logic(state: Dict[str, Any], out_dir: Path) -> Dict[str, Any]:
    """
    Deterministic deploy step: write config.json to out_dir (experiment_url, etc.).
    Called from implement step (3_implement) so deploy is bundled with implement.
    """
    project_id = state["project_id"]
    run_id = state["run_id"]
    mode = state["mode"]
    experiment_path = state.get("experiment_path", "")
    config = {
        "run_mode": mode,
        "experiment_path": experiment_path,
        "project_id": project_id,
        "run_id": run_id,
    }
    if state.get("ground_truth_model"):
        config["ground_truth_model"] = state["ground_truth_model"]

    agent_log(out_dir, f"Deploy: mode={mode!r}")

    if mode == "simulated_participants":
        config["simulated_n_participants"] = state.get("simulated_n_participants", DEFAULT_SIMULATED_N_PARTICIPANTS)
        exp_dir = Path(experiment_path)

        experiment_url = None
        results_api_url = None

        # Ground-truth mode: collect uses model-only generation; skip deploy to avoid unnecessary Firebase/local server.
        if state.get("ground_truth_model"):
            agent_log(out_dir, f"Deploy: ground_truth_model={state['ground_truth_model']!r}; skipping Firebase and local server.")
            config["experiment_url"] = None
            config["results_api_url"] = None
        elif exp_dir.exists():
            agent_log(out_dir, "Deploy: attempting Firebase deploy")
            base_url, deploy_error = _deploy_to_firebase(exp_dir, project_id, run_id)
            if base_url:
                experiment_url = base_url
                results_api_url = base_url  # same origin for /submit and /results
                _ensure_experiment_data_exposed(exp_dir)
                agent_log(out_dir, f"Deploy: Firebase deploy OK -> {base_url}")
            else:
                # Local server fallback
                log_status(deploy_error or "Using local server (Firebase not configured or deploy failed).")
                agent_log(out_dir, f"Deploy: Firebase failed or skipped; using local server. {deploy_error or ''}")
                _ensure_experiment_data_exposed(exp_dir)
                experiment_url = f"http://{BIND_ADDRESS}:{LOCAL_SERVER_PORT}"
                results_api_url = None
                config["server_port"] = LOCAL_SERVER_PORT
                config["note"] = deploy_error or "Using local server (Firebase not configured or deploy failed)."
                _write_start_server_script(exp_dir, out_dir, LOCAL_SERVER_PORT)
        else:
            config["experiment_url"] = None
            agent_log(out_dir, "Deploy: experiment_path does not exist; no experiment_url")

        if experiment_url:
            config["experiment_url"] = experiment_url
        if results_api_url:
            config["results_api_url"] = results_api_url
        if not config.get("experiment_url"):
            config["experiment_url"] = None
    elif mode == "test_prolific":
        # Test Prolific: Firebase deploy + create test participant + study (1 place, allowlist) + publish
        exp_dir = Path(experiment_path)
        base_url = None
        if not exp_dir.exists():
            agent_log(out_dir, "Deploy: test_prolific; experiment_path does not exist")
            config["experiment_url"] = None
            config["results_api_url"] = None
        else:
            try:
                from src.prolific_client import (
                    load_prolific_config,
                    get_me,
                    create_test_participant,
                    create_study,
                    publish_study,
                )
                pconfig = load_prolific_config(project_id)
                completion_code = pconfig.get("completion_code", "AUTO_PSYCH_COMPLETE")
                redirect_url = f"https://app.prolific.com/submissions/complete?cc={completion_code}"
                agent_log(out_dir, "Deploy: attempting Firebase deploy (test_prolific)")
                base_url, deploy_error = _deploy_to_firebase(
                    exp_dir, project_id, run_id,
                    extra_config={"prolific_redirect_url": redirect_url},
                )
                if not base_url:
                    agent_log(out_dir, f"Deploy: test_prolific Firebase failed: {deploy_error}")
                    config["experiment_url"] = None
                    config["results_api_url"] = None
                else:
                    _ensure_experiment_data_exposed(exp_dir)
                    agent_log(out_dir, f"Deploy: Firebase OK -> {base_url}")
                    agent_log(out_dir, "Prolific: fetching researcher id")
                    user, err = get_me()
                    if err:
                        agent_log(out_dir, f"Prolific: error get_me: {err}")
                        config["experiment_url"] = base_url
                        config["results_api_url"] = base_url
                    else:
                        email = pconfig.get("test_participant_email")
                        if not email:
                            agent_log(out_dir, "Prolific: error - test_participant_email not set in prolific_config.yaml")
                            config["experiment_url"] = base_url
                            config["results_api_url"] = base_url
                        else:
                            agent_log(out_dir, f"Prolific: creating test participant (email={email!r})")
                            participant_id, err = create_test_participant(email)
                            if err:
                                agent_log(out_dir, f"Prolific: error create_test_participant: {err}")
                                config["experiment_url"] = base_url
                                config["results_api_url"] = base_url
                            else:
                                agent_log(out_dir, f"Prolific: test participant created id={participant_id!r}")
                                external_url = base_url + "/?participant_id={{%PROLIFIC_PID%}}"
                                est_min = pconfig.get("estimated_completion_time", 5)
                                payload = {
                                    "name": pconfig.get("name", "Auto-psych test"),
                                    "description": pconfig.get("description", "Test run (test participant)."),
                                    "external_study_url": external_url,
                                    "estimated_completion_time": int(est_min),
                                    "total_available_places": 1,
                                    "reward": int(pconfig.get("reward", 50)),
                                    "prolific_id_option": "url_parameters",
                                    "completion_codes": [
                                        {
                                            "code": completion_code,
                                            "code_type": "COMPLETED",
                                            "actions": [{"action": "AUTOMATICALLY_APPROVE"}],
                                        }
                                    ],
                                    "filters": [
                                        {"filter_id": "custom_allowlist", "selected_values": [participant_id]}
                                    ],
                                    "device_compatibility": ["desktop", "mobile", "tablet"],
                                }
                                agent_log(out_dir, f"Prolific: creating study (external_study_url={base_url}/..., estimated_completion_time={est_min})")
                                study_id, err = create_study(payload)
                                if err:
                                    agent_log(out_dir, f"Prolific: error create_study: {err}")
                                else:
                                    agent_log(out_dir, f"Prolific: study created id={study_id!r}")
                                    ok, err = publish_study(study_id)
                                    if err:
                                        agent_log(out_dir, f"Prolific: publish failed: {err}")
                                    else:
                                        agent_log(out_dir, "Prolific: study published")
                                config["experiment_url"] = base_url
                                config["results_api_url"] = base_url
                                config["prolific_study_id"] = study_id if study_id else None
                                config["total_available_places"] = 1
            except Exception as e:
                agent_log(out_dir, f"Deploy: test_prolific error: {type(e).__name__}: {e}")
                config["experiment_url"] = base_url if base_url else (f"https://{_firebase_project_id()}.web.app" if _firebase_project_id() else None)
                config["results_api_url"] = config.get("experiment_url")
                config["prolific_study_id"] = None
                config["total_available_places"] = 1
    else:
        # Live mode: Firebase deploy first, then Prolific (when implemented)
        agent_log(out_dir, "Deploy: mode=live; Prolific flow not yet implemented")
        config["note"] = "Live mode: configure experiment URL and Prolific when implemented."
        config["experiment_url"] = None

    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    agent_log(out_dir, "Deploy: wrote config.json")
    return {**state, "deployment_config_path": str(out_dir / "config.json")}


def run_deployer(state: Dict[str, Any]) -> Dict[str, Any]:
    """Standalone deployer (legacy); prefer run_deploy_logic from implement step."""
    project_id = state["project_id"]
    run_id = state["run_id"]
    out_dir = agent_dir_for_state(project_id, run_id, "3_implement", state)  # same dir as implement
    out_dir.mkdir(parents=True, exist_ok=True)
    return run_deploy_logic(state, out_dir)


def _ensure_firebase_submit_in_index(index_path: Path) -> None:
    """If index.html sets __experimentData but does not POST to /submit, add the fetch block."""
    if not index_path.exists():
        return
    text = index_path.read_text(encoding="utf-8")
    if "fetch('/auto_psych_config.json')" in text or "fetch(\"/auto_psych_config.json\")" in text:
        return
    if "window.__experimentData = data" not in text:
        return
    # Insert after __experimentData line: fetch block for POST /submit
    insert = (
        " fetch('/auto_psych_config.json').then(function(r){ return r.ok ? r.json() : null; }).then(function(cfg){"
        " if (cfg && cfg.project_id != null && cfg.run_id != null)"
        " fetch('/submit', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({"
        " run_id: cfg.run_id, project_id: cfg.project_id, participant_id: (typeof window !== 'undefined' && window.__participantId != null) ? window.__participantId : Date.now(), trials: data }) });"
        " }).catch(function(){});"
    )
    # Place after "window.__experimentData = data;" (same line or next)
    old = "if (typeof window !== 'undefined') window.__experimentData = data;"
    if old in text:
        new = old + insert
        index_path.write_text(text.replace(old, new, 1), encoding="utf-8")


def _ensure_experiment_data_exposed(experiment_dir: Path) -> None:
    """Patch index.html so onFinish sets window.__experimentData for Playwright (agent 5)."""
    index = experiment_dir / "index.html"
    if not index.exists() or "__experimentData" in index.read_text(encoding="utf-8"):
        return
    text = index.read_text(encoding="utf-8")
    # Replace onFinish that only console.logs with one that sets __experimentData for Playwright
    old = "function() { console.log(jsPsych.data.get().json()); }"
    new = (
        "function() { var data = jsPsych.data.get().getData(); console.log(data); "
        "if (typeof window !== 'undefined') window.__experimentData = data; }"
    )
    if old in text:
        index.write_text(text.replace(old, new), encoding="utf-8")


def _write_start_server_script(experiment_dir: Path, out_dir: Path, port: int) -> None:
    """Write a script so you can start the server for browser testing or before running agent 5."""
    script = out_dir / "run_server.sh"
    body = f"""#!/bin/sh
# Start the experiment server. Then open http://127.0.0.1:{port}/ in your browser.
# Agent 5 will start the server automatically if it's not already running.
cd "{experiment_dir}"
exec python3 -m http.server {port} --bind 127.0.0.1
"""
    script.write_text(body, encoding="utf-8")
    script.chmod(0o755)
