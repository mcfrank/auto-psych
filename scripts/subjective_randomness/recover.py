"""CLI: run maximum-likelihood parameter recovery for a model family.

Usage:
    uv run python scripts/subjective_randomness/recover.py \\
        --config CONFIG.yaml --out REPORT.json [--n-repeats 20]
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import tyro
from pyprojroot import here

sys.path.insert(0, str(here()))

from src.subjective_randomness.config import load_config, resolve_path  # noqa: E402
from src.subjective_randomness.recover import run_recovery  # noqa: E402


@dataclass
class Args:
    """Run parameter recovery for a subjective-randomness model family."""

    config: Path
    """YAML config."""
    out: Path
    """Output JSON report path."""
    n_repeats: Optional[int] = None
    """Number of simulate-and-fit repeats (defaults to the config value)."""


def main(args: Args) -> None:
    config_path = resolve_path(args.config)
    result = run_recovery(load_config(config_path), config_path, args.n_repeats)
    out_path = resolve_path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        f"Wrote recovery report for {result['model']} "
        f"({result['n_repeats']} repeats, {result['n_stimuli']} stimuli) to {out_path}"
    )


if __name__ == "__main__":
    main(tyro.cli(Args))
