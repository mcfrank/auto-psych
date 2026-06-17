"""CLI: LLM-as-judge structural similarity of best model vs. ground truth.

For a finished holdout-recovery run, an LLM judge reads the then-best model's
code and the held-out ground-truth model's code at every inner-loop scoring step
and rates their mechanistic similarity on a 1-7 scale. Writes a summary JSON, a
trajectory figure (optionally overlaying the functional Pearson-r recovery), and
a judge cache so re-runs and repeated best-models cost no new API calls.

The default judge is the repo's hosted Gemini client (needs GOOGLE_API_KEY in
env or .secrets). ``--dry-run`` swaps in a deterministic code-overlap stub so the
end-to-end wiring (and the figure) can be produced with no API key — its numbers
are NOT LLM judgements and are labelled as a stub.

Usage:
    uv run python scripts/subjective_randomness/judge_model_similarity.py \\
        --result data/subjective_randomness/holdout_recovery_v2/holdout.json
"""

from __future__ import annotations

import difflib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import tyro
from pyprojroot import here

sys.path.insert(0, str(here()))

from src.subjective_randomness.config import resolve_path  # noqa: E402
from src.subjective_randomness.model_similarity_judge import (  # noqa: E402
    SIMILARITY_MAX,
    SIMILARITY_MIN,
    JudgeFn,
    load_cache,
    plot_similarity_trajectories,
    run_similarity,
    save_cache,
    similarity_summary_text,
)


def _gemini_judge_fn(model: Optional[str], timeout: int) -> JudgeFn:
    """A judge backend over the repo's hosted Gemini client.

    The repo client reads ``GOOGLE_API_KEY``; accept the Google SDK's
    ``GOOGLE_GENERATIVE_AI_API_KEY`` as a fallback so either env var works.
    """
    import os

    if not os.environ.get("GOOGLE_API_KEY"):
        alias = os.environ.get("GOOGLE_GENERATIVE_AI_API_KEY")
        if alias:
            os.environ["GOOGLE_API_KEY"] = alias

    from src.pipelines.outer_loop.llm import get_llm, invoke_llm

    llm = get_llm(timeout=timeout, model=model)

    def judge(system: str, user: str) -> str:
        return invoke_llm(system=system, user=user, llm=llm, timeout=timeout)

    return judge


def _stub_judge_fn() -> JudgeFn:
    """Deterministic, no-API stub: map code-overlap ratio onto the 1-7 scale.

    Only for exercising the pipeline without credentials. It compares surface
    code, not mechanism, so it is a wiring smoke test — not a real judgement.
    """
    span = SIMILARITY_MAX - SIMILARITY_MIN

    def judge(system: str, user: str) -> str:
        blocks = user.split("```python")
        code_a = blocks[1].split("```")[0] if len(blocks) > 1 else ""
        code_b = blocks[2].split("```")[0] if len(blocks) > 2 else ""
        ratio = difflib.SequenceMatcher(None, code_a, code_b).ratio()
        rating = SIMILARITY_MIN + round(ratio * span)
        return json.dumps({"rating": rating, "rationale": "stub: code-overlap ratio"})

    return judge


@dataclass
class Args:
    """LLM-judged structural similarity of best model vs. ground truth over time."""

    result: Path
    """Existing holdout-recovery result JSON to judge."""
    out: Optional[Path] = None
    """Summary JSON path (default: <result dir>/<stem>_similarity.json)."""
    figure: Optional[Path] = None
    """Trajectory figure path (default: <result dir>/<stem>_similarity.png)."""
    cache: Optional[Path] = None
    """Judge cache JSON (default: <result dir>/similarity_judge_cache.json)."""
    model: Optional[str] = None
    """Judge model id (default: the Gemini client's default)."""
    timeout: int = 120
    """Per-call judge timeout (seconds)."""
    no_symmetrize: bool = False
    """Judge each pair in one order only (default: judge both orders and average)."""
    overlay: bool = False
    """Also overlay the functional Pearson-r recovery on a twin axis (default: a
    clean holdout.png-style similarity-only figure)."""
    dry_run: bool = False
    """Use the no-API code-overlap stub judge instead of the LLM."""


def main(args: Args) -> None:
    result_path = resolve_path(args.result)
    result = json.loads(result_path.read_text(encoding="utf-8"))

    judge_fn = _stub_judge_fn() if args.dry_run else _gemini_judge_fn(args.model, args.timeout)

    cache_path = (
        resolve_path(args.cache)
        if args.cache is not None
        else result_path.parent / "similarity_judge_cache.json"
    )
    cache = load_cache(cache_path)

    similarity = run_similarity(
        result,
        judge_fn=judge_fn,
        cache=cache,
        symmetrize=not args.no_symmetrize,
    )
    save_cache(cache, cache_path)

    if args.dry_run:
        print("[dry-run] stub judge (code-overlap ratio) — NOT real LLM judgements\n")
    print(similarity_summary_text(similarity))

    out_path = (
        resolve_path(args.out)
        if args.out is not None
        else result_path.with_name(f"{result_path.stem}_similarity.json")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(similarity, indent=2), encoding="utf-8")
    print(f"\nWrote similarity summary to {out_path}")

    figure_path = (
        resolve_path(args.figure)
        if args.figure is not None
        else result_path.with_name(f"{result_path.stem}_similarity.png")
    )
    plot_similarity_trajectories(
        similarity,
        figure_path,
        holdout_result=result if args.overlay else None,
    )
    print(f"Wrote similarity figure to {figure_path}")


if __name__ == "__main__":
    main(tyro.cli(Args))
