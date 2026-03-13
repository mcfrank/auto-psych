"""Load JATOS configuration from .secrets, env, or project jatos_config.json."""

import os
from pathlib import Path
from typing import Any, Dict, Optional

from src.config import REPO_ROOT, project_dir


def _key_to_cfg(env_key: str) -> str:
    """Map JATOS_* env key to config dict key."""
    return env_key.lower()  # JATOS_BASE_URL -> jatos_base_url


def load_jatos_config(project_id: Optional[str] = None) -> Dict[str, Optional[str]]:
    """
    Return JATOS config: jatos_base_url, jatos_study_run_url, jatos_component_id, jatos_api_token.
    Sources (first wins): env vars -> .secrets file -> project jatos_config.json (urls/ids only).
    Token is only from env or .secrets, not from jatos_config.json.
    """
    out: Dict[str, Optional[str]] = {
        "jatos_base_url": None,
        "jatos_study_run_url": None,
        "jatos_component_id": None,
        "jatos_api_token": None,
    }
    # Env
    out["jatos_base_url"] = os.environ.get("JATOS_BASE_URL") or out["jatos_base_url"]
    out["jatos_study_run_url"] = os.environ.get("JATOS_STUDY_RUN_URL") or out["jatos_study_run_url"]
    out["jatos_component_id"] = os.environ.get("JATOS_COMPONENT_ID") or out["jatos_component_id"]
    out["jatos_api_token"] = os.environ.get("JATOS_API_TOKEN") or out["jatos_api_token"]
    # .secrets file
    _apply_secrets(out)
    # Project-level jatos_config.json (optional; no token in file)
    if project_id:
        _apply_project_config(project_id, out)
    return out


def _apply_secrets(out: Dict[str, Optional[str]]) -> None:
    """Read .secrets file or directory and fill missing keys."""
    secrets_path = REPO_ROOT / ".secrets"
    # .secrets as directory: one file per key (e.g. JATOS_BASE_URL, JATOS_API_TOKEN)
    if secrets_path.is_dir():
        for key in ("JATOS_BASE_URL", "JATOS_STUDY_RUN_URL", "JATOS_COMPONENT_ID", "JATOS_API_TOKEN"):
            f = secrets_path / key
            if f.exists() and out.get(_key_to_cfg(key)) in (None, ""):
                out[_key_to_cfg(key)] = f.read_text(encoding="utf-8").strip()
        return
    if not secrets_path.is_file():
        return
    for line in secrets_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        key = k.strip()
        # Strip value; remove trailing inline comment
        val = v.strip().split("#")[0].strip()
        if key == "JATOS_BASE_URL" and not out["jatos_base_url"]:
            out["jatos_base_url"] = val
        elif key == "JATOS_STUDY_RUN_URL" and not out["jatos_study_run_url"]:
            out["jatos_study_run_url"] = val
        elif key == "JATOS_COMPONENT_ID" and not out["jatos_component_id"]:
            out["jatos_component_id"] = val
        elif key in ("JATOS_API_TOKEN", "JATOS_TOKEN") and not out["jatos_api_token"]:
            out["jatos_api_token"] = val


def _apply_project_config(project_id: str, out: Dict[str, Optional[str]]) -> None:
    """Read projects/<project_id>/jatos_config.json and fill missing keys (no token)."""
    path = project_dir(project_id) / "jatos_config.json"
    if not path.exists():
        return
    try:
        import json
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return
    if not isinstance(data, dict):
        return
    if not out["jatos_base_url"] and data.get("jatos_base_url"):
        out["jatos_base_url"] = str(data["jatos_base_url"]).strip()
    if not out["jatos_study_run_url"] and data.get("jatos_study_run_url"):
        out["jatos_study_run_url"] = str(data["jatos_study_run_url"]).strip()
    if not out["jatos_component_id"] and data.get("jatos_component_id"):
        out["jatos_component_id"] = str(data["jatos_component_id"]).strip()
