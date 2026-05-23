import importlib.util
import os
from pathlib import Path

from src.pipelines.inner_loop.initial_model import INITIAL_PARAMS, PARAM_BOUNDS
from src.runtime.coding_agent import run_coding_agent

_PKG_DIR = Path(__file__).resolve().parent
_THEORY_PROMPT = _PKG_DIR / "prompts" / "theory.md"


def _spawn_agent(
    candidate_dir: Path, timeout_sec: int, api_key: str | None = None
) -> None:
    prompt = _THEORY_PROMPT.read_text()
    env = {
        "PATH": os.environ["PATH"],
        "HOME": os.environ.get("HOME", ""),
        "TMPDIR": os.environ.get("TMPDIR", "/tmp"),
        "LANG": os.environ.get("LANG", "en_US.UTF-8"),
    }
    if api_key:
        env["ANTHROPIC_API_KEY"] = api_key
    agent_log = candidate_dir / "agent.jsonl"
    success, _ = run_coding_agent(
        prompt,
        cwd=candidate_dir,
        log_path=agent_log,
        allowed_dirs=[candidate_dir],
        timeout_secs=timeout_sec,
        env=env,
        on_summary=None,
    )
    if not success:
        raise RuntimeError(
            f"Coding agent failed for {candidate_dir.name}. See {agent_log}"
        )


def _validate_model(model_path: Path) -> tuple[callable, list]:
    if not model_path.exists():
        raise FileNotFoundError(
            f"Agent did not write cognitive_model.py to {model_path.parent}"
        )

    spec = importlib.util.spec_from_file_location(
        f"cognitive_model_{model_path.parent.name}", model_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "cognitive_model"):
        raise AttributeError(
            "cognitive_model.py must define a function named 'cognitive_model'"
        )

    param_names = getattr(module, "PARAM_NAMES", INITIAL_PARAMS)
    n_params = (
        len(param_names) if isinstance(param_names, list) else len(INITIAL_PARAMS)
    )
    test_params = getattr(module, "INITIAL_PARAMS", INITIAL_PARAMS[:n_params])

    result = module.cognitive_model(("HHTHTTHT", "HTHTHTHT"), ["left", "right"], test_params)
    if not isinstance(result, dict):
        raise TypeError("cognitive_model must return a dict of response probabilities")
    if set(result) != {"left", "right"}:
        raise ValueError("cognitive_model must return probabilities for 'left' and 'right'")
    total = sum(float(value) for value in result.values())
    if abs(total - 1.0) > 1e-5:
        raise ValueError(f"cognitive_model probabilities must sum to 1.0, got {total}")

    return module.cognitive_model, getattr(module, "PARAM_BOUNDS", PARAM_BOUNDS)
