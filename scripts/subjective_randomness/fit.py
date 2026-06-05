"""CLI: fit subjective-randomness model-family parameters by maximum likelihood.

Usage:
    uv run python scripts/subjective_randomness/fit.py \\
        --config CONFIG.yaml --responses RESPONSES.csv [--out OUT.json]
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

from src.subjective_randomness.config import load_config, load_model, resolve_path  # noqa: E402
from src.subjective_randomness.fit import fit_rows, load_rows  # noqa: E402


@dataclass
class Args:
    """Fit a subjective-randomness model family by maximum likelihood."""

    config: Path
    """YAML config with model_module."""
    responses: Path
    """responses.csv to fit."""
    out: Optional[Path] = None
    """Optional JSON output path."""
    seed: Optional[int] = None
    """Random seed (defaults to the config value)."""


def main(args: Args) -> None:
    config_path = resolve_path(args.config)
    config = load_config(config_path)
    model = load_model(config["model_module"])
    fit_cfg = config.get("fit") or {}
    result = fit_rows(
        model,
        load_rows(resolve_path(args.responses, config_path)),
        n_starts=int(fit_cfg.get("n_starts", 24)),
        max_iters=int(fit_cfg.get("max_iters", 160)),
        seed=args.seed if args.seed is not None else int(fit_cfg.get("seed", 0)),
        fixed_params=fit_cfg.get("fixed_params") or {},
    )
    text = json.dumps(result, indent=2)
    if args.out:
        out_path = resolve_path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main(tyro.cli(Args))
