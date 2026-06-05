#!/usr/bin/env python3
"""Smoke-test the open-weights LLM-as-participant path in isolation.

Loads a Hugging Face model through the open participant backend and runs it over
a handful of toy subjective-randomness stimuli — no PyMC, no API key, no
experiment directory, no coding agents. It exercises exactly the open-model
plumbing: device selection, model load, chat-templated generation, answer
parsing, and row assembly.

Prerequisite (heavy deps; install only when needed):

    uv sync --group open-models

Usage:

    # default smoke model is tiny + fast (~270 MB):
    uv run python scripts/smoke_open_participant.py

    # point at any other id (e.g. the pipeline default — large!):
    uv run python scripts/smoke_open_participant.py --hf-model Qwen/Qwen2.5-0.5B-Instruct -n 2

Exit codes: 0 = ran and produced parseable rows; 4 = ran but every reply was
unparseable (plumbing works, model just didn't follow the format); 2 = deps
missing; 3 = model failed to load.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.pipelines.outer_loop.collect import generate_llm_participant_rows
from src.pipelines.outer_loop.llm import load_prompt_for_run
from src.pipelines.outer_loop.participants import get_participant_model

# Small, fast, instruction-tuned, has a chat template — good enough to prove the
# path end to end without a multi-GB download.
SMOKE_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"

STIMULI = [
    {"sequence_a": "HHTHTHTH", "sequence_b": "HHHHHHHH"},
    {"sequence_a": "HTHTHTHT", "sequence_b": "HHTHTTHT"},
    {"sequence_a": "TTTTTTTT", "sequence_b": "HTHHTHTT"},
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--hf-model", default=SMOKE_MODEL, help=f"HF model id (default: {SMOKE_MODEL})"
    )
    ap.add_argument("-n", "--n-participants", type=int, default=2)
    args = ap.parse_args()

    prompt = load_prompt_for_run(
        "subjective_randomness", 1, "4_collect_participant", None
    )
    if not prompt.strip():
        print("FAIL: could not load the 4_collect_participant prompt", file=sys.stderr)
        sys.exit(1)

    print(f"Loading open participant model: {args.hf_model} ...", flush=True)
    try:
        model = get_participant_model("open", args.hf_model)
    except ImportError as exc:
        print(f"FAIL (deps): {str(exc).splitlines()[0]}", file=sys.stderr)
        print(exc, file=sys.stderr)
        sys.exit(2)
    except Exception as exc:  # noqa: BLE001 - smoke script surfaces any load error
        print(f"FAIL (load): {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(3)

    print(
        f"Loaded {model.name} on device '{getattr(model, '_device', '?')}'", flush=True
    )
    print(
        f"Running {args.n_participants} participant(s) x {len(STIMULI)} stimuli ...\n",
        flush=True,
    )

    rows, stats = generate_llm_participant_rows(
        STIMULI, args.n_participants, participant_model=model, prompt_text=prompt
    )

    print("stats:", stats)
    print("\nrows:")
    for row in rows:
        print(
            f"  p{row['participant_id']} t{row['trial_index']}: "
            f"chose_left={row['chose_left']}  ({row['sequence_a']} vs {row['sequence_b']})"
        )

    if stats["n_rows"] > 0 and stats["n_errors"] == 0:
        print("\nPASS: open LLM-as-participant produced parseable rows.")
        sys.exit(0)
    if stats["n_errors"] > 0:
        print("\nFAIL: the model raised during generation.", file=sys.stderr)
        sys.exit(3)
    print(
        "\nPARTIAL: model ran but produced no parseable answers "
        "(plumbing OK; try a stronger model).",
        file=sys.stderr,
    )
    sys.exit(4)


if __name__ == "__main__":
    main()
