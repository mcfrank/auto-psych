"""Firebase staging and deploy helpers."""

from __future__ import annotations

import contextlib
from html import escape
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, Iterator

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


# JS that builds the consent overlay and keeps it present until "I agree".
# `__MARKER__` / `__CONSENT_HTML_JSON__` are substituted in Python (avoids
# brace-escaping). The overlay is attached to <html> (document.documentElement),
# NOT <body>: jsPsych's initJsPsych() clears document.body when it runs, which
# would delete an in-body overlay. A MutationObserver re-attaches it if anything
# detaches it, and keyboard input to the experiment is blocked, so the consent
# form ALWAYS shows first and is removed only when the participant consents.
_CONSENT_GATE_JS = """
(function () {
  var MARKER = "__MARKER__";
  if (window.__autoPsychConsentInstalled) return;
  window.__autoPsychConsentInstalled = true;
  window.__autoPsychConsented = false;
  var CONSENT_HTML = __CONSENT_HTML_JSON__;
  var gate = document.createElement("div");
  gate.id = MARKER;
  gate.setAttribute("role", "dialog");
  gate.setAttribute("aria-modal", "true");
  gate.style.cssText =
    "position:fixed;top:0;left:0;right:0;bottom:0;width:100%;height:100%;" +
    "z-index:2147483647;background:#ffffff;overflow:auto;margin:0;";
  gate.innerHTML =
    '<div style="max-width:640px;margin:40px auto;padding:24px;font-family:sans-serif;' +
    'font-size:16px;line-height:1.6;color:#222;text-align:left;">' + CONSENT_HTML +
    '<div style="text-align:center;margin-top:28px;">' +
    '<button id="' + MARKER + '-agree" type="button" style="padding:12px 32px;font-size:18px;' +
    'border:none;border-radius:8px;cursor:pointer;background:#4a90d9;color:#fff;">I agree</button>' +
    '</div></div>';
  var observer = null;
  function mount() {
    if (window.__autoPsychConsented) return;
    var root = document.documentElement || document.body;
    if (root && gate.parentNode !== root) root.appendChild(gate);
  }
  function blockInput(e) {
    if (window.__autoPsychConsented) return;
    if (gate.contains && gate.contains(e.target)) return;  // allow the consent dialog itself
    e.stopImmediatePropagation();
    e.preventDefault();
  }
  gate.addEventListener("click", function (e) {
    var t = e.target;
    if (t && t.id === MARKER + "-agree") {
      window.__autoPsychConsented = true;
      if (observer) observer.disconnect();
      if (gate.parentNode) gate.parentNode.removeChild(gate);
    }
  });
  mount();
  observer = new MutationObserver(mount);
  observer.observe(document.documentElement, { childList: true, subtree: true });
  document.addEventListener("DOMContentLoaded", mount);
  window.addEventListener("load", mount);
  ["keydown", "keyup", "keypress"].forEach(function (t) {
    document.addEventListener(t, blockInput, true);
  });
})();
""".strip()


def ensure_consent_gate(index_html: str, consent_html: str) -> str:
    """Inject a consent gate that ALWAYS shows first and blocks the experiment
    until the participant clicks "I agree".

    Idempotent: if the gate marker is already present the HTML is returned
    unchanged. The gate is built by JS attached to <html> (not <body>, which the
    experiment's jsPsych init clears) and re-attached via a MutationObserver, so
    it survives however the experiment builds and runs its timeline. Injected
    into <head> so it is established before the experiment's body scripts run.
    """
    if CONSENT_GATE_MARKER in index_html:
        return index_html
    js = _CONSENT_GATE_JS.replace("__MARKER__", CONSENT_GATE_MARKER).replace(
        "__CONSENT_HTML_JSON__", json.dumps(consent_html)
    )
    gate = f'<script id="{CONSENT_GATE_MARKER}-script">\n{js}\n</script>'
    for tag in ("</head>", "</body>"):
        if tag.lower() in index_html.lower():
            idx = index_html.lower().rfind(tag)
            return index_html[:idx] + gate + "\n" + index_html[idx:]
    return gate + "\n" + index_html


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
      const cfgResponse = await fetch("{CLIENT_CONFIG_FILENAME}");
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


def relativize_config_fetch(index_html: str) -> str:
    """Rewrite absolute ``/auto_psych_config.json`` fetches to a relative path.

    Each experiment is served from its own ``/e{run}/`` subpath, so the client
    config must be fetched relative to the page (``auto_psych_config.json``), not
    from the site root. ``/submit`` and ``/results`` are global Cloud Functions
    at the site root and are deliberately left absolute.
    """
    return index_html.replace(
        f'"/{CLIENT_CONFIG_FILENAME}"', f'"{CLIENT_CONFIG_FILENAME}"'
    ).replace(f"'/{CLIENT_CONFIG_FILENAME}'", f"'{CLIENT_CONFIG_FILENAME}'")


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
    staged_html = relativize_config_fetch(staged_html)
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


@contextlib.contextmanager
def _deploy_lock() -> Iterator[None]:
    """Serialize ``firebase deploy`` across parallel runs sharing one site.

    Parallel runs use isolated git worktrees, so the only shared resource is the
    remote Firebase Hosting site itself; two ``firebase deploy`` calls racing on
    the same site can clash. When ``AUTO_PSYCH_DEPLOY_LOCK`` names a lockfile we
    take a blocking advisory ``flock`` on it for the duration of the deploy, so
    only the (brief) deploy step serializes while everything else stays
    concurrent. Unset (or on a platform without ``fcntl``) this is a no-op.
    """
    lock_path = os.environ.get("AUTO_PSYCH_DEPLOY_LOCK")
    if not lock_path:
        yield
        return
    try:
        import fcntl
    except ImportError:
        yield
        return
    Path(lock_path).parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "w", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _deploy_argv(targets: str, project: str, config_path: Path) -> list[str]:
    firebase = shutil.which("firebase")
    base = [firebase] if firebase else ["npx", "-y", "firebase-tools"]
    return base + [
        "deploy",
        "--only",
        targets,
        "--non-interactive",
        "--force",
        "--project",
        project,
        "--config",
        str(config_path),
    ]


def _run_one_deploy(cmd: list[str], repo_root: Path, env: dict) -> None:
    result = subprocess.run(cmd, cwd=repo_root, text=True, capture_output=True, env=env)
    if result.returncode != 0:
        raise DeploymentError(
            f"Firebase deploy failed (--only {cmd[cmd.index('--only') + 1]})\n"
            f"STDOUT:\n{result.stdout[-4000:]}\n"
            f"STDERR:\n{result.stderr[-4000:]}"
        )


def _verify_hosting_live(url: str, *, attempts: int = 12, delay: float = 6.0) -> None:
    """Poll the deployed experiment URL until it returns 2xx, else raise.

    A combined ``--only hosting,functions,firestore`` deploy has been observed to
    return success WITHOUT publishing hosting (the page stays 404). Recruiting
    participants onto a 404 page is the worst possible failure, so after deploying
    we hard-verify the experiment page is actually live before the pipeline is
    allowed to proceed (and create a Prolific study). Allows a short window for
    CDN propagation; fails loudly if the page never comes up.
    """
    import time
    import urllib.error
    import urllib.request

    last: object = None
    for _ in range(attempts):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "auto-psych"})
            with urllib.request.urlopen(req, timeout=20) as resp:
                if 200 <= resp.status < 300:
                    return
                last = resp.status
        except urllib.error.HTTPError as exc:
            last = exc.code
        except Exception as exc:  # network / TLS / DNS
            last = repr(exc)
        time.sleep(delay)
    raise DeploymentError(
        f"Firebase deploy reported success but the experiment page is NOT live: "
        f"GET {url} -> {last} after {attempts} attempts. Refusing to proceed — "
        "the pipeline would otherwise recruit participants onto a broken (404) page."
    )


def run_firebase_deploy(repo_root: Path, manifest: DeploymentManifest, config_path: Path) -> None:
    if not manifest.firebase_project:
        raise DeploymentError("Firebase deploy requires firebase_project")
    ensure_functions_dependencies(repo_root)
    env = {**os.environ, "FUNCTION_REGION": manifest.firebase_region}
    project = manifest.firebase_project
    # Deploy functions+firestore and hosting SEPARATELY. A combined
    # `--only hosting,functions,firestore` deploy has been observed to return
    # success without actually releasing hosting (the experiment page stays 404);
    # a standalone hosting deploy releases reliably.
    with _deploy_lock():
        _run_one_deploy(_deploy_argv("functions,firestore", project, config_path), repo_root, env)
        _run_one_deploy(_deploy_argv("hosting", project, config_path), repo_root, env)
    # Hard-verify the page is actually live before anything downstream recruits.
    if manifest.experiment_url:
        _verify_hosting_live(manifest.experiment_url)
