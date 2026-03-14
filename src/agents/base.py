"""Base utilities for agents: LLM client, prompt loading."""

import os
from pathlib import Path
from typing import Dict

# LangChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

from src.config import PROMPTS_DIR, project_prompts_dir, prompts_used_dir


# Default request timeout (seconds); theorist and other heavy agents can override
DEFAULT_LLM_TIMEOUT = 300


def get_llm(timeout: int | None = None):
    """Return ChatGoogleGenerativeAI configured for gemini-3.1-pro-preview."""
    api_key = os.environ.get("GOOGLE_API_KEY") or _read_secret("GOOGLE_API_KEY")
    kwargs = dict(
        model="gemini-3.1-pro-preview",
        google_api_key=api_key,
        temperature=0.2,
    )
    if timeout is not None:
        kwargs["request_timeout"] = timeout
    return ChatGoogleGenerativeAI(**kwargs)


def _read_secret(key: str) -> str | None:
    """Read secret from .secrets file or directory."""
    from src.config import REPO_ROOT
    secrets_file = REPO_ROOT / ".secrets"
    if secrets_file.is_file():
        for line in secrets_file.read_text().splitlines():
            line = line.strip()
            if line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            if k.strip() == key:
                return v.strip()
    secrets_dir = REPO_ROOT / ".secrets"
    if secrets_dir.is_dir():
        key_file = secrets_dir / key
        if key_file.exists():
            return key_file.read_text().strip()
    return None


def load_prompt_for_run(project_id: str, run_id: int, agent_key: str) -> str:
    """Load the archived prompt used for this run (from prompts_used/)."""
    path = prompts_used_dir(project_id, run_id) / f"{agent_key}.md"
    if path.exists():
        return path.read_text(encoding="utf-8")
    # Fallback: resolve from canonical/project
    override = project_prompts_dir(project_id) / f"{agent_key}.md"
    canonical = PROMPTS_DIR / f"{agent_key}.md"
    if override.exists():
        return override.read_text(encoding="utf-8")
    if canonical.exists():
        return canonical.read_text(encoding="utf-8")
    return ""


def invoke_llm(system: str, user: str, llm=None, timeout: int | None = None) -> str:
    """Send system + user to LLM and return content string. timeout: seconds (default from get_llm)."""
    if llm is None:
        llm = get_llm(timeout=timeout or DEFAULT_LLM_TIMEOUT)
    messages = [
        SystemMessage(content=system),
        HumanMessage(content=user),
    ]
    invoke_kwargs = {}
    if timeout is not None:
        invoke_kwargs["timeout"] = timeout
    response = llm.invoke(messages, **invoke_kwargs)
    content = response.content if hasattr(response, "content") else response
    if isinstance(content, list):
        return " ".join(
            (getattr(part, "text", None) or str(part)) for part in content
        )
    return str(content) if content is not None else ""
