"""LLM and prompt utilities for the active outer loop."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from src.runtime.config import PROMPTS_DIR, project_prompts_dir, prompts_used_dir


DEFAULT_LLM_TIMEOUT = 300
DEFAULT_CLOSED_MODEL = "gemini-3.1-pro-preview"


def get_llm(timeout: int | None = None, model: str | None = None) -> Any:
    """Return the configured Gemini client. ``model`` overrides the default id."""
    from langchain_google_genai import ChatGoogleGenerativeAI

    api_key = os.environ.get("GOOGLE_API_KEY") or _read_secret("GOOGLE_API_KEY")
    kwargs = {
        "model": model or DEFAULT_CLOSED_MODEL,
        "google_api_key": api_key,
        "temperature": 0.2,
    }
    if timeout is not None:
        kwargs["request_timeout"] = timeout
    return ChatGoogleGenerativeAI(**kwargs)


def load_prompt_for_run(
    project_id: str,
    run_id: int,
    agent_key: str,
    state: dict | None = None,
) -> str:
    """Load the prompt archived for a run, falling back to project/canonical prompts."""
    if state and state.get("batch_dir"):
        prompts_dir = Path(state["batch_dir"]) / "prompts_used"
    else:
        prompts_dir = prompts_used_dir(project_id, run_id)
    path = prompts_dir / f"{agent_key}.md"
    if path.exists():
        return path.read_text(encoding="utf-8")

    override = project_prompts_dir(project_id) / f"{agent_key}.md"
    canonical = PROMPTS_DIR / f"{agent_key}.md"
    if override.exists():
        return override.read_text(encoding="utf-8")
    if canonical.exists():
        return canonical.read_text(encoding="utf-8")
    return ""


def invoke_llm(system: str, user: str, llm: Any = None, timeout: int | None = None) -> str:
    """Send a system + user message pair to the LLM and normalize the reply to text."""
    from langchain_core.messages import HumanMessage, SystemMessage

    if llm is None:
        llm = get_llm(timeout=timeout or DEFAULT_LLM_TIMEOUT)
    messages = [SystemMessage(content=system), HumanMessage(content=user)]
    invoke_kwargs = {}
    if timeout is not None:
        invoke_kwargs["timeout"] = timeout
    response = llm.invoke(messages, **invoke_kwargs)
    content = response.content if hasattr(response, "content") else response
    return _content_to_str(content)


def _read_secret(key: str) -> str | None:
    """Read a secret from `.secrets` as either a file or a directory."""
    from src.runtime.config import REPO_ROOT

    secrets_file = REPO_ROOT / ".secrets"
    if secrets_file.is_file():
        for line in secrets_file.read_text().splitlines():
            line = line.strip()
            if line.startswith("#") or "=" not in line:
                continue
            current_key, _, value = line.partition("=")
            if current_key.strip() == key:
                return value.strip()
    if secrets_file.is_dir():
        key_file = secrets_file / key
        if key_file.exists():
            return key_file.read_text().strip()
    return None


def _content_to_str(content: Any) -> str:
    """Normalize LangChain response content to a single string."""
    if content is None:
        return ""
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and "text" in part:
                parts.append(str(part["text"]))
            elif hasattr(part, "text") and part.text is not None:
                parts.append(str(part.text))
            else:
                parts.append(str(part))
        return "\n".join(part for part in parts if part.strip()) or " ".join(parts)
    if isinstance(content, dict) and "text" in content:
        return str(content["text"])
    return str(content)
