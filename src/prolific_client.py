"""
Prolific API client for creating studies, test participants, and polling submissions.
Token is read from .secrets (PROLIFIC_API_TOKEN) or env PROLIFIC_API_TOKEN.
Project-level settings: projects/<project_id>/prolific_config.yaml
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import requests
import yaml

from src.config import REPO_ROOT, project_dir

# Defaults when prolific_config.yaml is missing or partial
DEFAULT_ESTIMATED_COMPLETION_MINUTES = 5
DEFAULT_COMPLETION_CODE = "AUTO_PSYCH_COMPLETE"


def load_prolific_config(project_id: str) -> Dict[str, Any]:
    """
    Load projects/<project_id>/prolific_config.yaml with defaults.
    Keys: estimated_completion_time (min), completion_code, test_participant_email (for test_prolific),
    total_available_places, reward (cents), name, description, etc.
    """
    path = project_dir(project_id) / "prolific_config.yaml"
    out = {
        "estimated_completion_time": DEFAULT_ESTIMATED_COMPLETION_MINUTES,
        "completion_code": DEFAULT_COMPLETION_CODE,
        "test_participant_email": None,
        "total_available_places": 1,
        "reward": 50,
        "name": "Auto-psych experiment",
        "description": "Psychology experiment (auto-psych pipeline).",
    }
    if path.exists():
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            for k, v in data.items():
                if v is not None:
                    out[k] = v
        except Exception:
            pass
    return out


def _get_token() -> Optional[str]:
    """Read PROLIFIC_API_TOKEN from .secrets file/dir or environment."""
    token = os.environ.get("PROLIFIC_API_TOKEN", "").strip()
    if token:
        return token
    secrets_file = REPO_ROOT / ".secrets"
    if secrets_file.is_file():
        for line in secrets_file.read_text().splitlines():
            line = line.strip()
            if line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            if k.strip() == "PROLIFIC_API_TOKEN":
                return v.strip()
    if secrets_file.is_dir():
        key_file = secrets_file / "PROLIFIC_API_TOKEN"
        if key_file.exists():
            return key_file.read_text().strip()
    return None


_BASE = "https://api.prolific.com/api/v1"


def _headers() -> Dict[str, str]:
    token = _get_token()
    if not token:
        raise ValueError("PROLIFIC_API_TOKEN not set (add to .secrets or env)")
    return {
        "Authorization": f"Token {token}",
        "Content-Type": "application/json",
    }


def get_me() -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """GET /users/me/ . Returns (user_dict, error_message). user_dict has 'id' (researcher id)."""
    try:
        r = requests.get(f"{_BASE}/users/me/", headers=_headers(), timeout=30)
        if r.status_code != 200:
            return (None, f"GET /users/me/ {r.status_code}: {r.text[:500]}")
        return (r.json(), None)
    except Exception as e:
        return (None, str(e))


def create_test_participant(email: str) -> Tuple[Optional[str], Optional[str]]:
    """
    POST /researchers/participants/ to create a test participant.
    Returns (participant_id, error_message). Email must not be already registered on Prolific.
    """
    try:
        r = requests.post(
            f"{_BASE}/researchers/participants/",
            headers=_headers(),
            json={"email": email},
            timeout=30,
        )
        if r.status_code not in (200, 201):
            return (None, f"POST /researchers/participants/ {r.status_code}: {r.text[:500]}")
        data = r.json()
        return (data.get("participant_id") or data.get("id"), None)
    except Exception as e:
        return (None, str(e))


def create_study(payload: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    """
    POST /studies/ to create a draft study. payload must include external_study_url,
    estimated_completion_time, total_available_places, etc.
    Returns (study_id, error_message).
    """
    try:
        r = requests.post(f"{_BASE}/studies/", headers=_headers(), json=payload, timeout=60)
        if r.status_code not in (200, 201):
            return (None, f"POST /studies/ {r.status_code}: {r.text[:500]}")
        data = r.json()
        return (data.get("id"), None)
    except Exception as e:
        return (None, str(e))


def publish_study(study_id: str) -> Tuple[bool, Optional[str]]:
    """POST /studies/{id}/transition/ to publish. Returns (success, error_message)."""
    try:
        r = requests.post(
            f"{_BASE}/studies/{study_id}/transition/",
            headers=_headers(),
            json={"action": "PUBLISH"},
            timeout=30,
        )
        if r.status_code != 200:
            return (False, f"POST transition PUBLISH {r.status_code}: {r.text[:500]}")
        return (True, None)
    except Exception as e:
        return (False, str(e))


def get_study(study_id: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """GET /studies/{id}/ . Returns (study_dict, error_message)."""
    try:
        r = requests.get(f"{_BASE}/studies/{study_id}/", headers=_headers(), timeout=30)
        if r.status_code != 200:
            return (None, f"GET /studies/{id}/ {r.status_code}: {r.text[:500]}")
        return (r.json(), None)
    except Exception as e:
        return (None, str(e))


def get_submission_counts(study_id: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """GET /studies/{id}/submissions/counts/ . Returns (counts_dict, error_message)."""
    try:
        r = requests.get(
            f"{_BASE}/studies/{study_id}/submissions/counts/",
            headers=_headers(),
            timeout=30,
        )
        if r.status_code != 200:
            return (None, f"GET /studies/{id}/submissions/counts/ {r.status_code}: {r.text[:500]}")
        return (r.json(), None)
    except Exception as e:
        return (None, str(e))
