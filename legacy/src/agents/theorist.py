"""Theorist agent: add one theory per LLM call; iterate until 2–3 (run 1) or 1+ new (run 2+)."""

import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from src.config import agent_dir_for_state, run_dir_for_state, DEFAULT_MAX_VALIDATION_RETRIES
from src.references import load_references
from src.agents.base import load_prompt_for_run, invoke_llm
from src.console_log import agent_header, log_status
from src.observability import agent_log, write_transcript
from src.registry import load_registry
from src.agents.llm_output_parsing import (
    ensure_str,
    extract_fenced_blocks,
    extract_yaml_from_response,
)
from src.registry import load_registry, write_registry, DEFAULT_RESERVED_FOR_NEW

# Run 1: need 2–3 theories total (2–3 calls). Run 2+: need at least 1 new theory.
MIN_THEORIES_RUN1 = 2
MAX_ITERATIONS_RUN1 = 3
MIN_NEW_THEORIES_RUN2_PLUS = 1
MAX_ITERATIONS_RUN2_PLUS = 5


def run_theorist(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Iteratively call the LLM to add one theory per turn. Run 1: add 2–3 theories
    (2–3 calls). Run 2+: copy previous run's theories and add at least 1 new (1+ calls).
    Write models_manifest.yaml, rationale.md, and <model_name>.py for each model.
    """
    project_id = state["project_id"]
    run_id = state["run_id"]
    if state.get("last_validated_agent") != "1_theory":
        state = {**state, "validation_retry_count": 0, "validation_feedback": ""}
    if state.get("validation_retry_count", 0) == 0:
        agent_header("1_theory", run_id, state.get("total_runs"), state.get("mode"))
    elif state.get("validation_retry_count", 0) > 0:
        max_r = state.get("max_validation_retries", DEFAULT_MAX_VALIDATION_RETRIES)
        log_status(f"Repeating due to validation failure (attempt {state['validation_retry_count']}/{max_r})")
    out_dir = agent_dir_for_state(project_id, run_id, "1_theory", state)
    out_dir.mkdir(parents=True, exist_ok=True)
    attempt = (state.get("validation_retry_count") or 0) + 1
    validation_feedback = (state.get("validation_feedback") or "").strip()
    agent_log(out_dir, "=== 1_theory start (iterative) ===")
    agent_log(out_dir, f"project_id={project_id!r} run_id={run_id} attempt={attempt}")
    if validation_feedback:
        agent_log(out_dir, f"Validation feedback: {validation_feedback[:500]}")

    prob_path = Path(state["problem_definition_path"])
    problem_text = prob_path.read_text(encoding="utf-8") if prob_path.exists() else ""
    reference_text = load_references(project_id)
    if reference_text:
        agent_log(out_dir, f"loaded {len(reference_text)} chars of reference material")
    interpreter_path = state.get("interpreter_report_path")
    interpreter_text = ""
    if interpreter_path:
        p = Path(interpreter_path)
        if p.exists():
            interpreter_text = p.read_text(encoding="utf-8")
    prompt = load_prompt_for_run(project_id, run_id, "1_theory", state)
    # No fixed model library: theorist invents models from the problem definition.
    available_models: List[str] = []

    # Run 2+: copy previous run's 1_theory into out_dir so we retain existing theories
    current_models: List[Dict[str, Any]] = []
    new_models_this_run = 0
    if run_id >= 2:
        _copy_previous_run_theories(project_id, run_id, out_dir, state)
        current_models = _load_current_manifest(out_dir)
        agent_log(out_dir, f"loaded {len(current_models)} existing models from previous run")

    min_new = MIN_NEW_THEORIES_RUN2_PLUS if run_id >= 2 else 0
    min_total_run1 = MIN_THEORIES_RUN1
    max_iterations = MAX_ITERATIONS_RUN2_PLUS if run_id >= 2 else MAX_ITERATIONS_RUN1

    iteration = 0
    rationales: List[str] = []

    while iteration < max_iterations:
        iteration += 1
        user_content = _build_theorist_user_message(
            run_id=run_id,
            current_models=current_models,
            problem_text=problem_text,
            reference_text=reference_text,
            interpreter_text=interpreter_text,
            validation_feedback=validation_feedback if iteration == 1 else "",
            state=state,
            available_models=available_models,
            iteration=iteration,
            new_models_this_run=new_models_this_run,
        )

        agent_log(out_dir, f"invoking LLM (theory iteration {iteration})...")
        try:
            response = invoke_llm(system=prompt, user=user_content, timeout=300)
        except Exception as e:
            agent_log(out_dir, f"LLM invoke error: {type(e).__name__}: {e}")
            response = ""
        response = ensure_str(response)
        agent_log(out_dir, f"LLM response length={len(response)} chars")
        write_transcript(
            out_dir, iteration,
            system=prompt, user=user_content, response=response[:100_000],
            validation_feedback=validation_feedback if iteration == 1 else "",
        )

        model_name, rationale, code, done = _parse_single_model_response(response)
        if model_name and code:
            current_models.append({"name": model_name, "rationale": rationale or ""})
            if run_id >= 2:
                new_models_this_run += 1
            # File must match manifest name so validators find it
            (out_dir / f"{model_name}.py").write_text(code, encoding="utf-8")
            agent_log(out_dir, f"added model {model_name!r} ({model_name}.py)")
            if rationale:
                rationales.append(f"- **{model_name}**: {rationale}")
        else:
            agent_log(out_dir, "no model or code extracted; saving raw response")
            if response:
                (out_dir / f"_last_response_iteration_{iteration}.txt").write_text(
                    response[:50000], encoding="utf-8"
                )

        # Stop when agent says DONE and we have enough
        if run_id == 1:
            enough = len(current_models) >= min_total_run1
        else:
            enough = new_models_this_run >= min_new
        if done and enough:
            agent_log(out_dir, f"DONE with {len(current_models)} models (new this run: {new_models_this_run})")
            break
        if done and not enough and iteration >= max_iterations:
            break
        if not done and iteration >= max_iterations:
            agent_log(out_dir, f"reached max iterations {max_iterations}")
            break

    model_names = [m["name"] for m in current_models if m.get("name")]
    if not model_names:
        agent_log(out_dir, "no models extracted from LLM; manifest will have no models (validation may fail)")

    manifest = {
        "models": current_models,
        "rationale": "\n".join(rationales) if rationales else "Theories added iteratively.",
    }
    (out_dir / "models_manifest.yaml").write_text(
        yaml.dump(manifest, default_flow_style=False), encoding="utf-8"
    )
    (out_dir / "rationale.md").write_text(
        manifest.get("rationale", ""), encoding="utf-8"
    )
    agent_log(out_dir, f"wrote manifest (models={model_names})")
    agent_log(out_dir, "=== 1_theory end ===")

    # Registry (unchanged logic)
    registry_path = Path(state.get("registry_path", ""))
    if not registry_path or not str(registry_path).strip() or registry_path.resolve().is_dir():
        registry_path = run_dir_for_state(project_id, run_id, state) / "model_registry.yaml"
    if registry_path:
        if run_id == 1:
            k = len(model_names) or 1
            prob_each = (1.0 - DEFAULT_RESERVED_FOR_NEW) / k
            theories = {m: prob_each for m in model_names}
            write_registry(registry_path, theories, reserved_for_new=DEFAULT_RESERVED_FOR_NEW)
        else:
            prev_registry = run_dir_for_state(project_id, run_id - 1, state) / "model_registry.yaml"
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
                s = sum(theories.values())
                if s > 0:
                    theories = {m: p / s for m, p in theories.items()}
            write_registry(registry_path, theories, reserved_for_new=DEFAULT_RESERVED_FOR_NEW)

    return {
        **state,
        "theorist_manifest_path": str(out_dir / "models_manifest.yaml"),
        "theorist_rationale_path": str(out_dir / "rationale.md"),
    }


def _copy_previous_run_theories(project_id: str, run_id: int, out_dir: Path, state: Dict[str, Any]) -> None:
    """Copy previous run's 1_theory/*.py and models_manifest.yaml into out_dir."""
    prev_dir = run_dir_for_state(project_id, run_id - 1, state) / "1_theory"
    if not prev_dir.exists():
        return
    manifest_path = prev_dir / "models_manifest.yaml"
    if manifest_path.exists():
        shutil.copy2(manifest_path, out_dir / "models_manifest.yaml")
    for py_file in prev_dir.glob("*.py"):
        if not py_file.name.startswith("_"):
            shutil.copy2(py_file, out_dir / py_file.name)


def _load_current_manifest(out_dir: Path) -> List[Dict[str, Any]]:
    """Load current models list from models_manifest.yaml if it exists."""
    path = out_dir / "models_manifest.yaml"
    if not path.exists():
        return []
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "models" in data and isinstance(data["models"], list):
            return list(data["models"])
    except Exception:
        pass
    return []


def _build_theorist_user_message(
    run_id: int,
    current_models: List[Dict[str, Any]],
    problem_text: str,
    reference_text: str,
    interpreter_text: str,
    validation_feedback: str,
    state: Dict[str, Any],
    available_models: List[str],
    iteration: int,
    new_models_this_run: int,
) -> str:
    """Build the user message for one iteration."""
    parts = []
    if validation_feedback:
        parts.append(f"""## Validation feedback (previous attempt failed)

{validation_feedback}

Please fix the output so it passes validation. Output one YAML block, one Python block, then ---DONE--- or ---ADD_ANOTHER---.

""")
    parts.append(f"""## Run context

This is **Run {run_id}** of the pipeline. This is **iteration {iteration}** (one theory per turn).

""")
    if run_id >= 2:
        parts.append(f"So far this run you have added **{new_models_this_run}** new theory/theories. You must add at least one; say ---ADD_ANOTHER--- to add one more, or ---DONE--- when finished.\n\n")
    else:
        parts.append("You must add 2–3 theories total. Say ---ADD_ANOTHER--- to add one more, or ---DONE--- when you have added at least 2.\n\n")

    if current_models:
        names = [m.get("name", "") for m in current_models if m.get("name")]
        parts.append(f"""## Current manifest (already in this run)

{chr(10).join('- ' + n for n in names)}

""")
    if run_id >= 2:
        prev_registry_path = run_dir_for_state(state["project_id"], run_id - 1, state) / "model_registry.yaml"
        prev_reg = load_registry(prev_registry_path)
        prev_theories = prev_reg.get("theories") or {}
        if prev_theories:
            probs_str = "\n".join(f"  {k}: {v}" for k, v in sorted(prev_theories.items()))
            parts.append(f"""## Previous run's theory probabilities (Run {run_id - 1})

```yaml
{probs_str}
```

""")
    if interpreter_text:
        parts.append(f"""## Interpreter report from Run {run_id - 1}

{interpreter_text}

""")
    if reference_text:
        parts.append(f"""## Reference material (from references/)

{reference_text}

""")
    parts.append(f"""## Problem definition

{problem_text}

## Model implementation

Invent one probabilistic model (or implement one suggested by the problem definition). There is no fixed library; implement the function from scratch.

Output exactly: (1) one YAML block with one model (name + optional rationale), (2) one ```python block with # file: <model_name>.py and the function, (3) then ---DONE--- or ---ADD_ANOTHER---.
""")
    return "".join(parts)


def _parse_single_model_response(response: str) -> Tuple[Optional[str], str, str, bool]:
    """
    Parse one model name, rationale, one code block, and DONE vs ADD_ANOTHER.
    Returns (model_name, rationale, code, done).
    """
    data = extract_yaml_from_response(response)
    model_name = None
    rationale = ""
    if data:
        if "models" in data and isinstance(data["models"], list) and data["models"]:
            first = data["models"][0]
            if isinstance(first, dict):
                model_name = first.get("name")
                rationale = first.get("rationale", "") or ""
            elif isinstance(first, str):
                model_name = first
        elif data.get("name"):
            model_name = data["name"]
            rationale = data.get("rationale", "") or ""

    blocks = extract_fenced_blocks(
        ensure_str(response), language="python", normalize=True, min_length=20
    )
    code = blocks[0] if blocks else ""

    # Prefer filename from code block for the actual .py name
    resp_upper = (response or "").upper()
    done = "---DONE---" in resp_upper
    if "---ADD_ANOTHER---" in resp_upper:
        done = False
    return (model_name, rationale, code, done)


