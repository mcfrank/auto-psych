"""Theorist agent: select models and generate one .py file per model."""

import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

from src.config import agent_dir, run_dir
from src.agents.base import get_llm, load_prompt_for_run, invoke_llm
from src.console_log import agent_header, log_status
from src.registry import load_registry
from src.agents.llm_output_parsing import (
    ensure_str,
    extract_fenced_blocks,
    extract_yaml_from_response,
)
from src.models.randomness import MODEL_LIBRARY
from src.registry import load_registry, write_registry, DEFAULT_RESERVED_FOR_NEW


def run_theorist(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Read problem definition and optional interpreter report; use LLM to produce
    a model manifest and one Python file per model; write models_manifest.yaml,
    rationale.md, and <model_name>.py for each model.
    """
    project_id = state["project_id"]
    run_id = state["run_id"]
    agent_header("1theorist", run_id, state.get("total_runs"), state.get("mode"))
    if state.get("validation_retry_count", 0) > 0:
        log_status(f"Repeating due to validation failure (attempt {state['validation_retry_count']}/3)")
    out_dir = agent_dir(project_id, run_id, "1theorist")
    out_dir.mkdir(parents=True, exist_ok=True)

    prob_path = Path(state["problem_definition_path"])
    problem_text = prob_path.read_text(encoding="utf-8") if prob_path.exists() else ""

    interpreter_path = state.get("interpreter_report_path")
    interpreter_text = ""
    if interpreter_path:
        p = Path(interpreter_path)
        if p.exists():
            interpreter_text = p.read_text(encoding="utf-8")

    prompt = load_prompt_for_run(project_id, run_id, "1theorist")
    available_models = list(MODEL_LIBRARY.keys())

    validation_feedback = (state.get("validation_feedback") or "").strip()
    user_content = ""
    if validation_feedback:
        user_content += f"""## Validation feedback (previous attempt failed)

{validation_feedback}

Please fix the output so it passes validation. Then output your YAML and code blocks as below.

"""
    user_content += f"""## Run context

This is **Run {run_id}** of the pipeline.
"""
    if run_id >= 2:
        user_content += f"""For Run 2 and later you must **build on the previous run**. Use the interpreter report and (if provided) the previous run's theory probabilities below to decide which models to retain, drop, or add. Consider adding new theories if the evidence suggests them.

"""
    user_content += f"""## Problem definition

{problem_text}

"""
    # Previous run's theory probabilities (so theorist sees what interpreter left)
    if run_id >= 2:
        prev_registry_path = run_dir(project_id, run_id - 1) / "model_registry.yaml"
        prev_reg = load_registry(prev_registry_path)
        prev_theories = prev_reg.get("theories") or {}
        if prev_theories:
            probs_str = "\n".join(f"  {k}: {v}" for k, v in sorted(prev_theories.items()))
            user_content += f"""## Previous run's theory probabilities (Run {run_id - 1})

The interpreter from Run {run_id - 1} left these probabilities. Use them to inform your model set and to consider new theories.

```yaml
{probs_str}
```

"""
    if interpreter_text:
        user_content += f"""## Interpreter report from Run {run_id - 1}

{interpreter_text}

"""
    user_content += f"""## Available models in the library (for reference; you may implement these or your own)

{chr(10).join('- ' + m for m in available_models)}

You must output two things. The pipeline extracts your response and writes files; if you do not output the code blocks below, no .py files will be created and the run will fail.

1) A YAML block (between ---BEGIN YAML--- and ---END YAML---) with the list of models and a short rationale.
2) For EVERY model in that list, a separate fenced Python code block. Each block must start with a line ```python and the first line inside must be: # file: <model_name>.py (same name as in the YAML). Then the full Python code for that model: a function(stimulus, response_options) that returns a dict mapping each response option to a probability (sum to 1).

Example for one model named bayesian_fair_coin:

---BEGIN YAML---
models:
  - name: bayesian_fair_coin
rationale: |
  Short rationale.
---END YAML---

```python
# file: bayesian_fair_coin.py
def bayesian_fair_coin(stimulus, response_options):
    seq_a, seq_b = stimulus
    # ... compute probs ...
    return {{response_options[0]: p_a, response_options[1]: p_b}}
```

Output your YAML block first, then one ```python ... ``` block per model (same order as in the YAML). Every model in the manifest must have a corresponding code block.
"""

    try:
        response = invoke_llm(system=prompt, user=user_content)
    except Exception:
        response = ""
    response = ensure_str(response)
    # Parse YAML from response (shared helper handles literal \n and fence variants)
    manifest = extract_yaml_from_response(response) or _default_manifest(available_models, response)
    model_names = [m.get("name") for m in manifest.get("models", []) if m.get("name")]

    (out_dir / "models_manifest.yaml").write_text(yaml.dump(manifest, default_flow_style=False), encoding="utf-8")
    rationale = manifest.get("rationale", response)
    if isinstance(rationale, str):
        (out_dir / "rationale.md").write_text(rationale, encoding="utf-8")
    else:
        (out_dir / "rationale.md").write_text(str(rationale), encoding="utf-8")

    # Extract Python code blocks (shared helper normalizes literal \n) and write <model_name>.py
    extracted = _extract_python_blocks_with_names(response, model_names)
    for model_name, code in extracted:
        if code and model_name:
            (out_dir / f"{model_name}.py").write_text(code, encoding="utf-8")
    # If we expected code but got none, save raw response for debugging
    if model_names and not extracted:
        (out_dir / "_last_theorist_response.txt").write_text(
            str(response)[:50000], encoding="utf-8"
        )

    # Create or update this run's model registry (per-run; probabilities sum to 1)
    registry_path = Path(state.get("registry_path", ""))
    if not registry_path or not str(registry_path).strip() or registry_path.resolve().is_dir():
        registry_path = run_dir(project_id, run_id) / "model_registry.yaml"
    if registry_path:
        if run_id == 1:
            # Run 1: uniform prior over current models, reserve 0.25 for new
            k = len(model_names) or 1
            prob_each = (1.0 - DEFAULT_RESERVED_FOR_NEW) / k
            theories = {m: prob_each for m in model_names}
            write_registry(registry_path, theories, reserved_for_new=DEFAULT_RESERVED_FOR_NEW)
        else:
            # Run n >= 2: merge previous run's registry with current model set; renormalize
            prev_registry = run_dir(project_id, run_id - 1) / "model_registry.yaml"
            prev = load_registry(prev_registry)
            prev_theories = prev.get("theories") or {}
            reserved = prev.get("reserved_for_new", DEFAULT_RESERVED_FOR_NEW)
            existing_in_current = [m for m in model_names if m in prev_theories]
            new_in_current = [m for m in model_names if m not in prev_theories]
            existing_sum = sum(prev_theories.get(m, 0) for m in existing_in_current)
            target_existing = max(0.0, 1.0 - reserved)
            if existing_sum > 0 and existing_in_current:
                scale = target_existing / existing_sum
                theories = {m: prev_theories[m] * scale for m in existing_in_current}
            else:
                theories = {}
            if new_in_current:
                each_new = reserved / len(new_in_current)
                for m in new_in_current:
                    theories[m] = each_new
            elif theories and abs(sum(theories.values()) - 1.0) > 1e-6:
                # No new models; scale existing to sum to 1
                s = sum(theories.values())
                if s > 0:
                    theories = {m: p / s for m, p in theories.items()}
            write_registry(registry_path, theories, reserved_for_new=DEFAULT_RESERVED_FOR_NEW)

    return {
        **state,
        "theorist_manifest_path": str(out_dir / "models_manifest.yaml"),
        "theorist_rationale_path": str(out_dir / "rationale.md"),
    }


def _default_manifest(available_models: List[str], response: str) -> Dict[str, Any]:
    """Fallback when YAML cannot be parsed."""
    return {
        "models": [{"name": n} for n in available_models[:2]],
        "rationale": (response or "")[:500],
    }


def _extract_python_blocks_with_names(
    response: str, model_names: List[str]
) -> List[Tuple[str, str]]:
    """
    Extract fenced Python blocks (via shared helper) and pair each with a filename.
    If a block's first line is `# file: <name>.py`, use that name; else use model_names by order.
    Returns [(model_name, code), ...].
    """
    blocks = extract_fenced_blocks(
        response, language="python", normalize=True, min_length=20
    )
    result = []
    for i, code in enumerate(blocks):
        first_line, _, _ = code.partition("\n")
        match = re.match(
            r"#\s*file\s*:\s*([^\s.]+)(?:\.py)?\s*$",
            first_line.strip(),
            re.IGNORECASE,
        )
        if match:
            name = match.group(1).strip()
        else:
            name = model_names[i] if i < len(model_names) else ""
        if name:
            result.append((name, code))
    return result
