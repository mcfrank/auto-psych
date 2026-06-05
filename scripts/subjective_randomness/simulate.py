"""CLI: simulate choices from a subjective-randomness model family.

Usage:
    uv run python scripts/subjective_randomness/simulate.py \\
        --config CONFIG.yaml --out OUT.csv [--params '{"beta": 4.0}']
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
from src.subjective_randomness.simulate import generate_rows, load_stimuli, write_rows  # noqa: E402


@dataclass
class Args:
    """Simulate choices from a subjective-randomness model family."""

    config: Path
    """YAML config with model_module, stimuli_path, true_params."""
    out: Path
    """Output responses.csv path."""
    params: Optional[str] = None
    """JSON object overriding true_params."""
    n_participants: Optional[int] = None
    """Number of simulated participants (defaults to the config value)."""
    seed: Optional[int] = None
    """Random seed (defaults to the config value)."""


def main(args: Args) -> None:
    config_path = resolve_path(args.config)
    config = load_config(config_path)
    model = load_model(config["model_module"])
    stimuli = load_stimuli(resolve_path(config["stimuli_path"], config_path))
    sim_cfg = config.get("simulation") or {}

    params = dict(config.get("true_params") or {})
    if args.params:
        params.update(json.loads(args.params))
    params = {k: float(v) for k, v in params.items()}

    n_participants = args.n_participants if args.n_participants is not None else int(sim_cfg.get("n_participants", 20))
    seed = args.seed if args.seed is not None else int(sim_cfg.get("seed", 1))
    rows = generate_rows(model, stimuli, params, n_participants, seed)
    out_path = resolve_path(args.out)
    write_rows(rows, out_path)
    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main(tyro.cli(Args))
