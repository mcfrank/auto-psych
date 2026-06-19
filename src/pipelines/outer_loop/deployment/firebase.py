"""Firebase staging and deploy helpers."""

from __future__ import annotations

from html import escape
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any

from src.runtime.config import REPO_ROOT

from .manifest import CLIENT_CONFIG_FILENAME, MANIFEST_FILENAME, DeploymentManifest

# Marker on the injected consent overlay. Used to keep injection idempotent and
# to assert at staging time that every deployed experiment gates on consent.
CONSENT_GATE_MARKER = "auto-psych-consent-gate"
CONSENT_TEXT_PATH = REPO_ROOT / "templates" / "consent.txt"


class DeploymentError(RuntimeError):
    """Raised when Firebase staging or deployment fails."""


def load_consent_html() -> str:
    """Render the verbatim IRB consent text (templates/consent.txt) as HTML.

    The first paragraph becomes a heading; the rest become paragraphs. Text is
    HTML-escaped so the participant sees exactly the approved wording.
    """
    if not CONSENT_TEXT_PATH.exists():
        raise DeploymentError(f"IRB consent text not found at {CONSENT_TEXT_PATH}")
    raw = CONSENT_TEXT_PATH.read_text(encoding="utf-8").strip()
    if not raw:
        raise DeploymentError(f"IRB consent text at {CONSENT_TEXT_PATH} is empty")
    paragraphs = [p.strip() for p in raw.split("\n\n") if p.strip()]
    head = f"<h2>{escape(paragraphs[0])}</h2>"
    body = "".join(f"<p>{escape(p)}</p>" for p in paragraphs[1:])
    return head + body


def ensure_consent_gate(index_html: str, consent_html: str) -> str:
    """Inject a full-screen consent gate that blocks the experiment until "I agree".

    Idempotent: if the gate marker is already present the HTML is returned
    unchanged. The gate is a fixed overlay (max z-index) rendered over whatever
    the experiment displays, removed only when the participant consents — so it
    works regardless of how the experiment builds its jsPsych timeline.
    """
    if CONSENT_GATE_MARKER in index_html:
        return index_html
    gate = f"""
<div id="{CONSENT_GATE_MARKER}" style="position:fixed; inset:0; z-index:2147483647; background:#ffffff; overflow:auto;">
  <div style="max-width:640px; margin:40px auto; padding:24px; font-family:sans-serif; font-size:16px; line-height:1.6; color:#222; text-align:left;">
    {consent_html}
    <div style="text-align:center; margin-top:28px;">
      <button id="{CONSENT_GATE_MARKER}-agree" type="button"
        style="padding:12px 32px; font-size:18px; border:none; border-radius:8px; cursor:pointer; background:#4a90d9; color:#fff;">
        I agree
      </button>
    </div>
  </div>
</div>
<script>
(function() {{
  var btn = document.getElementById("{CONSENT_GATE_MARKER}-agree");
  if (!btn) return;
  btn.addEventListener("click", function() {{
    var gate = document.getElementById("{CONSENT_GATE_MARKER}");
    if (gate && gate.parentNode) gate.parentNode.removeChild(gate);
  }});
}})();
</script>
""".strip()
    marker = "</body>"
    if marker.lower() in index_html.lower():
        idx = index_html.lower().rfind(marker)
        return index_html[:idx] + gate + "\n" + index_html[idx:]
    return index_html + "\n" + gate + "\n"


def firebase_project_from_rc(repo_root: Path) -> str | None:
    rc_path = repo_root / ".firebaserc"
    if not rc_path.exists():
        return None
    try:
        data = json.loads(rc_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    projects = data.get("projects") if isinstance(data, dict) else None
    default = projects.get("default") if isinstance(projects, dict) else None
    return str(default) if default else None


def load_experiment_config(exp_dir: Path) -> dict[str, Any]:
    config_path = exp_dir / "experiment" / "config.json"
    if not config_path.exists():
        return {}
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise DeploymentError(f"Invalid experiment/config.json: {exc}") from exc
    return data if isinstance(data, dict) else {}


def ensure_submit_bridge(index_html: str) -> str:
    """Inject a small submit bridge if the implementer did not add one."""
    if CLIENT_CONFIG_FILENAME in index_html and "/submit" in index_html:
        return index_html
    bridge = f"""
<script>
(function() {{
  let submitted = false;
  async function autoPsychSubmit() {{
    if (submitted || typeof window.__experimentData === "undefined") return;
    submitted = true;
    try {{
      const params = new URLSearchParams(window.location.search);
      const cfgResponse = await fetch("/{CLIENT_CONFIG_FILENAME}");
      const cfg = await cfgResponse.json();
      const participantId = params.get("participant_id") || params.get("PROLIFIC_PID") || String(Date.now());
      const payload = {{
        ...cfg,
        participant_id: participantId,
        prolific_pid: params.get("PROLIFIC_PID"),
        prolific_study_id_from_url: params.get("STUDY_ID"),
        prolific_session_id: params.get("SESSION_ID"),
        trials: window.__experimentData,
        submitted_at_client: new Date().toISOString(),
        user_agent: navigator.userAgent
      }};
      const response = await fetch("/submit", {{
        method: "POST",
        headers: {{ "Content-Type": "application/json" }},
        body: JSON.stringify(payload)
      }});
      if (response.ok && cfg.prolific_redirect_url) {{
        window.location.href = cfg.prolific_redirect_url;
      }}
    }} catch (err) {{
      console.error("auto-psych submit failed", err);
      submitted = false;
    }}
  }}
  setInterval(autoPsychSubmit, 500);
  window.addEventListener("autoPsychFinished", autoPsychSubmit);
}})();
</script>
""".strip()
    marker = "</body>"
    if marker.lower() in index_html.lower():
        idx = index_html.lower().rfind(marker)
        return index_html[:idx] + bridge + "\n" + index_html[idx:]
    return index_html + "\n" + bridge + "\n"


def stage_experiment(exp_dir: Path, manifest: DeploymentManifest, public_dir: Path) -> Path:
    source_dir = exp_dir / "experiment"
    index_path = source_dir / "index.html"
    if not index_path.exists():
        raise DeploymentError(f"Experiment index.html not found at {index_path}")

    if public_dir.exists():
        shutil.rmtree(public_dir)
    public_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_dir, public_dir)

    design_stimuli = exp_dir / "design" / "stimuli.json"
    if design_stimuli.exists() and not (public_dir / "stimuli.json").exists():
        shutil.copyfile(design_stimuli, public_dir / "stimuli.json")

    staged_index = public_dir / "index.html"
    staged_html = ensure_submit_bridge(staged_index.read_text(encoding="utf-8"))
    staged_html = ensure_consent_gate(staged_html, load_consent_html())
    if CONSENT_GATE_MARKER not in staged_html:
        raise DeploymentError(
            "Refusing to deploy: staged experiment does not gate on IRB consent "
            f"(missing {CONSENT_GATE_MARKER!r} in {staged_index})."
        )
    staged_index.write_text(staged_html, encoding="utf-8")
    (public_dir / CLIENT_CONFIG_FILENAME).write_text(
        json.dumps(manifest.to_client_config(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (public_dir / MANIFEST_FILENAME).write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return public_dir


def write_firebase_config(config_path: Path, manifest: DeploymentManifest) -> Path:
    hosting = {
        "public": "public",
        "ignore": ["firebase.json", "**/.*", "**/node_modules/**"],
        "rewrites": [
            {
                "source": "/submit",
                "function": {"functionId": "submit", "region": manifest.firebase_region},
            },
            {
                "source": "/results",
                "function": {"functionId": "results", "region": manifest.firebase_region},
            },
        ],
    }
    if manifest.firebase_project:
        hosting["site"] = manifest.firebase_project

    config = {
        "hosting": hosting,
        "functions": {
            "source": "functions",
            "runtime": "nodejs22",
        },
        # Lock the database to Admin-SDK (Cloud Functions) access only; clients
        # never read/write Firestore directly. See firestore.rules.
        "firestore": {"rules": "firestore.rules"},
    }
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return config_path


def ensure_functions_dependencies(repo_root: Path) -> None:
    functions_dir = repo_root / "functions"
    package_json = functions_dir / "package.json"
    if not package_json.exists():
        raise DeploymentError(f"Firebase Functions package.json not found at {package_json}")

    required_packages = [
        functions_dir / "node_modules" / "firebase-functions",
        functions_dir / "node_modules" / "firebase-admin",
    ]
    if all(path.exists() for path in required_packages):
        return

    npm = shutil.which("npm")
    if not npm:
        raise DeploymentError(
            "Firebase Functions dependencies are missing and npm is not installed. "
            "Install Node/npm, then run: npm --prefix functions install"
        )
    result = subprocess.run(
        [npm, "--prefix", "functions", "install"],
        cwd=repo_root,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        raise DeploymentError(
            "Failed to install Firebase Functions dependencies with npm\n"
            f"STDOUT:\n{result.stdout[-4000:]}\n"
            f"STDERR:\n{result.stderr[-4000:]}"
        )


def run_firebase_deploy(repo_root: Path, manifest: DeploymentManifest, config_path: Path) -> None:
    if not manifest.firebase_project:
        raise DeploymentError("Firebase deploy requires firebase_project")
    ensure_functions_dependencies(repo_root)
    firebase = shutil.which("firebase")
    if firebase:
        cmd = [
            firebase,
            "deploy",
            "--only",
            "hosting,functions,firestore",
            "--non-interactive",
            "--force",
            "--project",
            manifest.firebase_project,
            "--config",
            str(config_path),
        ]
    else:
        cmd = [
            "npx",
            "-y",
            "firebase-tools",
            "deploy",
            "--only",
            "hosting,functions,firestore",
            "--non-interactive",
            "--force",
            "--project",
            manifest.firebase_project,
            "--config",
            str(config_path),
        ]
    env = {**os.environ, "FUNCTION_REGION": manifest.firebase_region}
    result = subprocess.run(cmd, cwd=repo_root, text=True, capture_output=True, env=env)
    if result.returncode != 0:
        raise DeploymentError(
            "Firebase deploy failed\n"
            f"STDOUT:\n{result.stdout[-4000:]}\n"
            f"STDERR:\n{result.stderr[-4000:]}"
        )
