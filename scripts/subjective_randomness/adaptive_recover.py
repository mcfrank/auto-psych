"""CLI: adaptive (sequential) EIG-driven recovery on the reference families.

The loop auto-generates a diverse candidate pool, then repeatedly selects the
highest-EIG stimulus under the current posterior, simulates the true response,
and updates — choosing the experiment's stimuli automatically instead of using a
fixed set. Two modes:

* ``--mode parameter`` — recover one model's parameters (parameter-EIG).
* ``--mode model`` — recover which model generated the data, one generating
  model at a time, into a confusion (model-identity-EIG).

This runs on the pure-Python families with an exact grid posterior (no MCMC):
fast, deterministic, the design-evaluation counterpart to the one-shot PyMC
recovery. See ``src/subjective_randomness/adaptive_recovery.py``.

Usage:
    uv run python scripts/subjective_randomness/adaptive_recover.py \\
        --config scripts/subjective_randomness/configs/adaptive_recovery.yaml \\
        --out data/subjective_randomness/adaptive/model_recovery.json
"""

from __future__ import annotations

import importlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Mapping, Optional

import tyro
from pyprojroot import here

sys.path.insert(0, str(here()))

from src.subjective_randomness.adaptive_recovery import (  # noqa: E402
    run_adaptive_model_confusion,
    run_adaptive_parameter_recovery,
)
from src.subjective_randomness.config import load_config, resolve_path  # noqa: E402
from src.subjective_randomness.stimulus_design import (  # noqa: E402
    default_model_family_names,
    generate_candidate_pool,
)


@dataclass
class Args:
    """Run adaptive EIG-driven recovery (parameter or model) on the families."""

    config: Path
    """YAML config (mode, candidate pool, rounds, true params/models)."""
    out: Path
    """Output JSON report path."""
    mode: Optional[Literal["parameter", "model"]] = None
    """Override the config's mode."""
    selected_out: Optional[Path] = None
    """Optional path to write the adaptively-chosen stimuli (parameter mode)."""
    n_rounds: Optional[int] = None
    """Override the config's number of adaptive rounds."""
    seed: Optional[int] = None
    """Override the config's RNG seed."""


def _family_default_params(name: str) -> Dict[str, float]:
    module = importlib.import_module(f"src.subjective_randomness.model_families.{name}")
    return dict(module.DEFAULT_PARAMS)


def _resolve_generating_params(
    spec: Any, model_names: List[str]
) -> Dict[str, Dict[str, float]]:
    """Resolve config's ``generating_models`` to name -> true params for families."""
    if spec is None:
        return {n: _family_default_params(n) for n in model_names}
    if isinstance(spec, list):
        return {n: _family_default_params(n) for n in spec}
    if isinstance(spec, Mapping):
        return {
            n: (dict(p) if p else _family_default_params(n)) for n, p in spec.items()
        }
    raise TypeError(
        f"generating_models must be null, a list, or a mapping; got {type(spec).__name__}."
    )


def _build_pool(config: Mapping[str, Any], seed: int) -> List[Dict[str, str]]:
    pool_cfg = dict(config.get("pool", {}))
    return generate_candidate_pool(
        n_pairs=int(pool_cfg.get("n_pairs", 200)),
        lengths=tuple(pool_cfg.get("lengths", (6, 8))),
        seed=seed,
    )


def _run_parameter_mode(config, pool, *, n_rounds, n_participants, points_per_dim, seed):
    model = config["model"]
    true_params = config.get("true_params") or _family_default_params(model)
    result = run_adaptive_parameter_recovery(
        model,
        true_params,
        pool,
        n_rounds=n_rounds,
        n_participants=n_participants,
        points_per_dim=points_per_dim,
        seed=seed,
    )
    delta = result["prior_entropy_bits"] - result["final_entropy_bits"]
    print(
        f"\nAdaptive parameter recovery — model: {model} "
        f"({n_rounds} rounds, {n_participants} participants/stimulus)"
    )
    print(
        f"  posterior entropy: {result['prior_entropy_bits']:.2f} -> "
        f"{result['final_entropy_bits']:.2f} bits (gained {delta:.2f})"
    )
    header = ["parameter", "true", "estimate", "error", "sd"]
    print("  " + "  ".join(f"{h:>12}" for h in header))
    for name in result["true_params"]:
        print(
            "  "
            + "  ".join(
                f"{v:>12}" if isinstance(v, str) else f"{v:>12.4g}"
                for v in (
                    name,
                    float(result["true_params"][name]),
                    result["posterior_mean"][name],
                    result["estimate_error"][name],
                    result["posterior_sd"][name],
                )
            )
        )
    return result


def _run_model_mode(config, pool, *, model_names, n_rounds, n_participants, points_per_dim, seed):
    generating_params = _resolve_generating_params(
        config.get("generating_models"), model_names
    )
    result = run_adaptive_model_confusion(
        pool,
        generating_params=generating_params,
        model_names=model_names,
        n_rounds=n_rounds,
        n_participants=n_participants,
        points_per_dim=points_per_dim,
        seed=seed,
    )
    entries = result["generating"]
    n_correct = sum(e["recovered_correct"] for e in entries)
    print(
        f"\nAdaptive model recovery — {len(model_names)} models, {n_rounds} rounds | "
        f"accuracy: {n_correct}/{len(entries)}"
    )
    header = ["generating_model", "recovered_model", "P(true)"]
    print("  " + "  ".join(f"{h:>22}" for h in header))
    for e in entries:
        p_true = e["model_posterior"].get(e["generating_model"], 0.0)
        flag = "" if e["recovered_correct"] else "   <- mis-recovered"
        print(
            "  "
            + "  ".join(
                f"{v:>22}"
                for v in (e["generating_model"], e["recovered_model"], f"{p_true:.3f}")
            )
            + flag
        )
    return result


def main(args: Args) -> None:
    config = load_config(resolve_path(args.config))
    mode = args.mode or config["mode"]
    seed = args.seed if args.seed is not None else int(config.get("seed", 0))
    n_rounds = args.n_rounds or int(config["n_rounds"])
    n_participants = int(config.get("n_participants", 1))
    points_per_dim = int(config.get("points_per_dim", 7))
    model_names = config.get("model_names") or default_model_family_names()
    pool = _build_pool(config, seed)

    if mode == "parameter":
        result = _run_parameter_mode(
            config, pool, n_rounds=n_rounds, n_participants=n_participants,
            points_per_dim=points_per_dim, seed=seed,
        )
    elif mode == "model":
        result = _run_model_mode(
            config, pool, model_names=model_names, n_rounds=n_rounds,
            n_participants=n_participants, points_per_dim=points_per_dim, seed=seed,
        )
    else:
        raise ValueError(f"Unknown mode {mode!r}; expected 'parameter' or 'model'.")

    out_path = resolve_path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nWrote adaptive {mode}-recovery report to {out_path}")

    if args.selected_out is not None and mode == "parameter":
        selected_path = resolve_path(args.selected_out)
        selected_path.parent.mkdir(parents=True, exist_ok=True)
        selected_path.write_text(
            json.dumps(result["selected_stimuli"], indent=2), encoding="utf-8"
        )
        print(f"Wrote adaptively-selected stimuli to {selected_path}")


if __name__ == "__main__":
    main(tyro.cli(Args))
