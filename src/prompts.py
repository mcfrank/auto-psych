"""Prompt resolution: canonical + project overrides, and run archive."""

from pathlib import Path
from typing import Dict

from .config import PROMPTS_DIR, project_prompts_dir, prompts_used_dir


AGENT_KEYS = [
    "1_theory",
    "2_design",
    "3_implement",
    "4_collect",
    "5_analyze",
    "6_interpret",
]


def resolve_prompts(project_id: str) -> Dict[str, str]:
    """
    Resolve prompt text for each agent: use project override if present,
    otherwise canonical prompt from prompts/.
    """
    project_prompts = project_prompts_dir(project_id)
    result = {}
    for key in AGENT_KEYS:
        canonical = PROMPTS_DIR / f"{key}.md"
        override = project_prompts / f"{key}.md"
        if override.exists():
            result[key] = override.read_text(encoding="utf-8")
        elif canonical.exists():
            result[key] = canonical.read_text(encoding="utf-8")
        else:
            result[key] = ""
    return result


def archive_prompts_for_run(project_id: str, run_id: int, resolved: Dict[str, str]) -> None:
    """Copy resolved prompts to run's prompts_used/ for reproducibility."""
    archive_dir = prompts_used_dir(project_id, run_id)
    archive_dir.mkdir(parents=True, exist_ok=True)
    for key, text in resolved.items():
        (archive_dir / f"{key}.md").write_text(text, encoding="utf-8")
    # Optional manifest
    import json
    manifest = {
        key: "project" if (project_prompts_dir(project_id) / f"{key}.md").exists() else "canonical"
        for key in AGENT_KEYS
    }
    (archive_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
