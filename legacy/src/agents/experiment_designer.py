"""Experiment designer: LLM generates a design script that computes EIG and selects stimuli.

Future: When iterating over multiple runs (e.g. run 1 → interpreter → run 2), the designer
could receive paths to previous run outputs (design_script.py, stimuli.json) and be prompted
to reuse or adapt them instead of regenerating from scratch when the theorist is unchanged.
"""

import contextlib
import io
import itertools
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from src.config import agent_dir_for_state, DEFAULT_MAX_VALIDATION_RETRIES
from src.agents.base import load_prompt_for_run, invoke_llm
from src.console_log import agent_header, log_status
from src.observability import agent_log, write_transcript
from src.agents.llm_output_parsing import ensure_str, extract_fenced_blocks
from src.models.loader import get_model_names_from_manifest
from src.models.randomness import get_model_predictions, Stimulus
from src.registry import get_model_weights
from src.problem_definition import parse_problem_definition

RESPONSE_OPTIONS = ["left", "right"]


def run_experiment_designer(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ask the LLM to generate a Python script that computes EIG and selects stimuli.
    Run the script; if it fails or does not write stimuli.json, fall back to built-in EIG logic.
    """
    project_id = state["project_id"]
    run_id = state["run_id"]
    # Reset validation retry state when this agent is entered from a different agent's validation
    if state.get("last_validated_agent") != "2_design":
        state = {**state, "validation_retry_count": 0, "validation_feedback": ""}
    if state.get("validation_retry_count", 0) == 0:
        agent_header("2_design", run_id, state.get("total_runs"), state.get("mode"))
    elif state.get("validation_retry_count", 0) > 0:
        max_r = state.get("max_validation_retries", DEFAULT_MAX_VALIDATION_RETRIES)
        log_status(f"Repeating due to validation failure (attempt {state['validation_retry_count']}/{max_r})")
    out_dir = agent_dir_for_state(project_id, run_id, "2_design", state)
    out_dir.mkdir(parents=True, exist_ok=True)
    attempt = (state.get("validation_retry_count") or 0) + 1
    validation_feedback = (state.get("validation_feedback") or "").strip()
    agent_log(out_dir, "=== 2_design start ===")
    agent_log(out_dir, f"project_id={project_id!r} run_id={run_id} attempt={attempt}")
    if validation_feedback:
        agent_log(out_dir, f"Validation feedback: {validation_feedback[:500]}")

    manifest_path = Path(state["theorist_manifest_path"])
    manifest = yaml.safe_load(manifest_path.read_text()) if manifest_path.exists() else {}
    theorist_dir = manifest_path.parent
    model_names = get_model_names_from_manifest(manifest, theorist_dir)
    if not model_names:
        agent_log(out_dir, "no theorist models loadable (need 1_theory/<name>.py for each manifest name); design may fail")
    registry_path = Path(state.get("registry_path", ""))
    model_weights = get_model_weights(registry_path) if registry_path and registry_path.exists() else {}

    agent_log(out_dir, f"registry_path={registry_path!s} exists={registry_path.exists() if registry_path else False}")
    agent_log(out_dir, f"model_names={model_names}")
    agent_log(out_dir, f"model_weights={model_weights if model_weights else '(uniform prior)'}")

    prob_path = Path(state["problem_definition_path"])
    problem_text = prob_path.read_text(encoding="utf-8") if prob_path.exists() else ""
    spec = parse_problem_definition(project_id)
    total_trials = spec["total_trials"]
    allowed_lengths = spec["allowed_sequence_lengths"]
    agent_log(out_dir, f"total_trials={total_trials} allowed_sequence_lengths={allowed_lengths}")

    prompt = load_prompt_for_run(project_id, run_id, "2_design", state)
    user_content = ""
    if validation_feedback:
        user_content += f"""## Validation feedback (previous attempt failed)

{validation_feedback}

Please fix the design script so it produces valid stimuli.json with sequence_a, sequence_b, and eig for each stimulus.

"""
    run_id = state.get("run_id", 1)
    user_content += f"""## Run context

This is **Run {run_id}** of the pipeline.
"""
    if run_id >= 2:
        prev_design_dir = agent_dir_for_state(project_id, run_id - 1, "2_design", state)
        user_content += f"""The previous run's design is in: `{prev_design_dir}` (stimuli.json, design_rationale.md). You may reuse or adapt it if the theory set is unchanged or similar, so experiments stay comparable across runs.

"""
    user_content += f"""## Problem definition

{problem_text}

## Your task

Output a single Python script (one fenced ```python block) that:
1. Generates candidate stimuli per the problem definition (you may use different sequence lengths as specified; allowed lengths: {allowed_lengths}). Pairs can be same-length or mixed-length (e.g. sequence_a length 4, sequence_b length 6).
2. Scores each by expected information gain using the provided expected_information_gain(stimulus_tuple) (EIG uses the current theory probabilities).
3. Selects **exactly** total_trials stimuli and writes stimuli.json and design_rationale.md to out_dir.

Consider diversity: include stimuli that have high EIG for distinguishing different subsets of theories (e.g. some that best distinguish theory A vs B, others that best distinguish B vs C), so the experiment collectively discriminates across all theories. The total number of stimuli must be exactly total_trials.

You have access to: theorist_dir (Path), model_names (list), expected_information_gain(stimulus_tuple), get_model_predictions(...), RESPONSE_OPTIONS, out_dir (Path), total_trials (int, must be exactly {total_trials}), allowed_sequence_lengths (list, {allowed_lengths}).

Model names for this run: {model_names}
""" + (f"\nCurrent run's theory probabilities (used by expected_information_gain): {model_weights}" if model_weights else "\n(Uniform prior over models for this run.)") + """

You **must** use the provided expected_information_gain((sequence_a, sequence_b)) to score each candidate — do not implement EIG yourself. You **must** output exactly total_trials stimuli in stimuli.json. Each item must include "sequence_a", "sequence_b", and "eig". Use only the variables provided: theorist_dir, model_names, expected_information_gain, out_dir, total_trials, allowed_sequence_lengths, Path, json. Output only the script in a single fenced ```python block.
"""

    design_script_ok = False
    llm_error: Optional[str] = None
    n_blocks = 0
    script_written = False
    try:
        agent_log(out_dir, "invoking LLM...")
        response = invoke_llm(system=prompt, user=user_content)
        response = ensure_str(response)
        agent_log(out_dir, f"LLM response length={len(response)} chars")
        write_transcript(
            out_dir, attempt,
            system=prompt, user=user_content, response=response[:100_000],
            validation_feedback=validation_feedback,
        )
        blocks = extract_fenced_blocks(response, "python", normalize=True, min_length=50)
        n_blocks = len(blocks)
        agent_log(out_dir, f"extract_fenced_blocks: n_blocks={n_blocks}" + (f" first_block_len={len(blocks[0])}" if blocks else ""))
        if not blocks:
            # Save a sample of the response for debugging extraction (e.g. fence format)
            sample = (response[:8000] + "\n...[truncated]") if len(response) > 8000 else response
            (out_dir / "last_llm_response_sample.txt").write_text(sample, encoding="utf-8")
            agent_log(out_dir, "wrote last_llm_response_sample.txt for debugging (no fenced block extracted)")
        if blocks:
            script_code = blocks[0]
            (out_dir / "design_script.py").write_text(script_code, encoding="utf-8")
            script_written = True
            agent_log(out_dir, "wrote design_script.py; running script...")
            design_script_ok = _run_design_script(out_dir, theorist_dir, model_names, model_weights, total_trials, allowed_lengths)
            agent_log(out_dir, f"_run_design_script returned {design_script_ok}; stimuli.json exists={(out_dir / 'stimuli.json').exists()}")
        else:
            agent_log(out_dir, "no python block >= 50 chars; skipping script run")
    except Exception as e:
        llm_error = f"{type(e).__name__}: {e}"
        agent_log(out_dir, f"LLM or extraction failed: {llm_error}")

    if not design_script_ok or not (out_dir / "stimuli.json").exists():
        reason_parts = []
        if llm_error:
            reason_parts.append(f"llm_error={llm_error}")
        if not script_written:
            reason_parts.append("no script written (no fenced block)")
        elif not design_script_ok:
            reason_parts.append("script ran but failed or did not write stimuli.json (see design_script_log.txt)")
        elif not (out_dir / "stimuli.json").exists():
            reason_parts.append("stimuli.json missing after script run")
        agent_log(out_dir, "using FALLBACK design: " + "; ".join(reason_parts))
        _fallback_design(out_dir, theorist_dir, model_names, model_weights, total_trials, allowed_lengths)
    else:
        agent_log(out_dir, "outcome: used LLM-generated script; stimuli.json present")
        if (out_dir / "design_script_log.txt").exists():
            agent_log(out_dir, "script stdout/stderr in design_script_log.txt")
    agent_log(out_dir, "=== 2_design end ===")

    return {
        **state,
        "stimuli_path": str(out_dir / "stimuli.json"),
        "design_rationale_path": str(out_dir / "design_rationale.md"),
    }


def _run_design_script(
    out_dir: Path,
    theorist_dir: Path,
    model_names: List[str],
    model_weights: Optional[Dict[str, float]] = None,
    total_trials: int = 30,
    allowed_sequence_lengths: Optional[List[int]] = None,
) -> bool:
    """Execute design_script.py with the required namespace. Return True if it wrote stimuli.json."""
    script_path = out_dir / "design_script.py"
    if not script_path.exists():
        return False
    if allowed_sequence_lengths is None:
        allowed_sequence_lengths = [8]
    # Bind EIG so script can call expected_information_gain(stimulus) only
    def _eig(stimulus: Tuple[str, str]) -> float:
        return expected_information_gain(stimulus, model_names, theorist_dir, model_weights)
    namespace: Dict[str, Any] = {
        "theorist_dir": theorist_dir,
        "model_names": model_names,
        "get_model_predictions": get_model_predictions,
        "expected_information_gain": _eig,
        "RESPONSE_OPTIONS": RESPONSE_OPTIONS,
        "out_dir": out_dir,
        "Path": Path,
        "json": json,
        "total_trials": total_trials,
        "allowed_sequence_lengths": allowed_sequence_lengths,
    }
    out_capture = io.StringIO()
    err_capture = io.StringIO()
    try:
        code = script_path.read_text(encoding="utf-8")
        with contextlib.redirect_stdout(out_capture), contextlib.redirect_stderr(err_capture):
            exec(compile(code, str(script_path), "exec"), namespace)
        log_lines = ["=== stdout ===\n", out_capture.getvalue(), "\n=== stderr ===\n", err_capture.getvalue()]
        (out_dir / "design_script_log.txt").write_text("".join(log_lines), encoding="utf-8")
    except Exception as e:
        agent_log(out_dir, f"script exec failed: {type(e).__name__}: {e}")
        (out_dir / "design_script_log.txt").write_text(
            f"Script failed: {type(e).__name__}: {e}\n\n=== stdout ===\n{out_capture.getvalue()}\n=== stderr ===\n{err_capture.getvalue()}",
            encoding="utf-8",
        )
        return False
    if not (out_dir / "stimuli.json").exists():
        agent_log(out_dir, "script ran but did not write stimuli.json")
        return False
    # Patch missing "eig" so validation passes and we use pipeline's weighted EIG
    _patch_stimuli_eig(out_dir, model_names, theorist_dir, model_weights)
    return True


def _patch_stimuli_eig(
    out_dir: Path,
    model_names: List[str],
    theorist_dir: Path,
    model_weights: Optional[Dict[str, float]] = None,
) -> None:
    """If any stimulus in stimuli.json is missing 'eig', set it from pipeline's expected_information_gain."""
    path = out_dir / "stimuli.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return
    if not isinstance(data, list):
        return
    n_patched = 0
    for item in data:
        if isinstance(item, dict) and "sequence_a" in item and "sequence_b" in item and "eig" not in item:
            stim = (item["sequence_a"], item["sequence_b"])
            item["eig"] = round(expected_information_gain(stim, model_names, theorist_dir, model_weights), 6)
            n_patched += 1
    if n_patched:
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        agent_log(out_dir, f"patched {n_patched} stimuli with missing 'eig'")


def _fallback_design(
    out_dir: Path,
    theorist_dir: Path,
    model_names: List[str],
    model_weights: Optional[Dict[str, float]] = None,
    total_trials: int = 30,
    allowed_lengths: Optional[List[int]] = None,
) -> None:
    """Built-in EIG design when no script or script failed. Uses diversity across theory pairs when possible."""
    if allowed_lengths is None:
        allowed_lengths = [8]
    candidate_stimuli = _sample_stimulus_pairs(allowed_lengths, max_pairs=max(150, total_trials * 3))
    scored = []
    for stim in candidate_stimuli:
        eig = expected_information_gain(stim, model_names, theorist_dir, model_weights)
        scored.append((stim, eig))
    scored.sort(key=lambda x: -x[1])

    # Diversity: prefer stimuli that discriminate different theory pairs (round-robin by best theory-pair)
    selected = _select_diverse_stimuli(scored, model_names, theorist_dir, total_trials, model_weights)
    if len(selected) < total_trials:
        # Backfill from top by global EIG
        seen = {s for s, _ in selected}
        for s, eig in scored:
            if s not in seen:
                selected.append((s, eig))
                seen.add(s)
                if len(selected) >= total_trials:
                    break
    selected = selected[:total_trials]

    stimuli_payload = [
        {"sequence_a": s[0], "sequence_b": s[1], "eig": round(eig, 6)}
        for s, eig in selected
    ]
    (out_dir / "stimuli.json").write_text(
        json.dumps(stimuli_payload, indent=2), encoding="utf-8"
    )
    top_stim = selected[0][0] if selected else None
    top_eig = selected[0][1] if selected else 0.0
    rationale = f"""Fallback design: selected {len(stimuli_payload)} stimulus pairs (total_trials={total_trials}, allowed_lengths={allowed_lengths}) with EIG diversity for models {model_names}.
Top pair: {top_stim} with EIG = {top_eig:.4f}.
"""
    (out_dir / "design_rationale.md").write_text(rationale, encoding="utf-8")


def _select_diverse_stimuli(
    scored: List[Tuple[Stimulus, float]],
    model_names: List[str],
    theorist_dir: Path,
    total_trials: int,
    model_weights: Optional[Dict[str, float]],
) -> List[Tuple[Stimulus, float]]:
    """Select stimuli so that different theory pairs are discriminated (round-robin by pairwise EIG)."""
    if len(model_names) < 2 or not scored:
        return [(s, e) for s, e in scored[:total_trials]]
    # For each theory pair (i, j), rank stimuli by EIG under prior only on i and j
    pair_ranks: Dict[Tuple[str, str], List[Tuple[Stimulus, float]]] = {}
    for i, j in itertools.combinations(model_names, 2):
        pair_prior = {m: 0.0 for m in model_names}
        pair_prior[i] = 0.5
        pair_prior[j] = 0.5
        pair_scored = [(s, expected_information_gain(s, model_names, theorist_dir, pair_prior)) for s, _ in scored]
        pair_scored.sort(key=lambda x: -x[1])
        pair_ranks[(i, j)] = pair_scored
    # Round-robin: take best remaining from each pair in turn
    selected: List[Tuple[Stimulus, float]] = []
    seen_stim = set()
    indices = {k: 0 for k in pair_ranks}
    pairs_cycle = list(pair_ranks.keys())
    while len(selected) < total_trials:
        made_progress = False
        for pair in pairs_cycle:
            rank_list = pair_ranks[pair]
            while indices[pair] < len(rank_list):
                s, eig = rank_list[indices[pair]]
                indices[pair] += 1
                if s not in seen_stim:
                    seen_stim.add(s)
                    selected.append((s, expected_information_gain(s, model_names, theorist_dir, model_weights)))
                    made_progress = True
                    break
            if len(selected) >= total_trials:
                return selected
        if not made_progress:
            break
    return selected


def _sample_stimulus_pairs(
    allowed_lengths: List[int],
    max_pairs: int,
) -> List[Stimulus]:
    """Sample diverse pairs of H/T sequences; lengths can vary (same or mixed per pair)."""
    sequences_by_len: Dict[int, List[str]] = {}
    for seq_length in allowed_lengths:
        seqs = []
        for n_heads in range(0, seq_length + 1):
            base = ["H"] * n_heads + ["T"] * (seq_length - n_heads)
            random.shuffle(base)
            seqs.append("".join(base))
        for _ in range(15):
            seq = "".join(random.choice("HT") for _ in range(seq_length))
            if seq not in seqs:
                seqs.append(seq)
        sequences_by_len[seq_length] = seqs

    pairs: List[Stimulus] = []
    seen = set()
    # Same-length pairs
    for length in allowed_lengths:
        seqs = sequences_by_len[length]
        for a, b in itertools.combinations(seqs, 2):
            if a != b and (a, b) not in seen and (b, a) not in seen:
                seen.add((a, b))
                pairs.append((a, b))
                if len(pairs) >= max_pairs:
                    return pairs
    # Mixed-length pairs (unordered length pairs)
    for len_a, len_b in itertools.combinations(allowed_lengths, 2):
        for a in sequences_by_len[len_a][:20]:
            for b in sequences_by_len[len_b][:20]:
                if (a, b) not in seen and (b, a) not in seen:
                    seen.add((a, b))
                    pairs.append((a, b))
                    if len(pairs) >= max_pairs:
                        return pairs
    while len(pairs) < max_pairs:
        len_a, len_b = random.sample(allowed_lengths, 2)
        seqs_a = sequences_by_len[len_a]
        seqs_b = sequences_by_len[len_b]
        a, b = random.choice(seqs_a), random.choice(seqs_b)
        if (a, b) not in seen and (b, a) not in seen:
            seen.add((a, b))
            pairs.append((a, b))
    return pairs


def expected_information_gain(
    stimulus: Stimulus,
    model_names: List[str],
    theorist_dir: Optional[Path] = None,
    model_weights: Optional[Dict[str, float]] = None,
) -> float:
    """Expected information gain (EIG) of the stimulus about the model identity.

    Formula: EIG = H(M) - E_R[H(M|R)] = H(M) - (p_left * H(M|left) + p_right * H(M|right)).
    Entropy is base-2 (bits). Predictions are P(left) and P(right)=1-P(left) per model.
    Uses model_weights as prior over models if provided, else uniform prior.
    """
    preds = get_model_predictions(stimulus, RESPONSE_OPTIONS, model_names, theorist_dir)
    if not preds:
        return 0.0
    if model_weights:
        total_w = sum(model_weights.get(m, 0) for m in preds)
        if total_w <= 0:
            total_w = len(preds)
            p_model = {m: 1.0 / total_w for m in preds}
        else:
            p_model = {m: model_weights.get(m, 0) / total_w for m in preds}
    else:
        n_models = len(preds)
        p_model = {m: 1.0 / n_models for m in preds}
    p_left = sum(preds[m].get("left", 0.5) * p_model[m] for m in preds)
    p_right = 1.0 - p_left
    if p_left <= 0 or p_right <= 0:
        return 0.0

    def entropy_given_response(response: str):
        p_m_given_r = []
        for m in preds:
            lik = preds[m].get(response, 0.0)
            p_m_given_r.append(lik * p_model[m] / (p_left if response == "left" else p_right))
        s = sum(p_m_given_r)
        if s <= 0:
            return 0.0
        return -sum(p * math.log2(p) for p in p_m_given_r if p > 0)

    h_m = -sum(p * math.log2(p) for p in p_model.values() if p > 0)
    h_m_given_r = p_left * entropy_given_response("left") + p_right * entropy_given_response("right")
    return max(0.0, h_m - h_m_given_r)
