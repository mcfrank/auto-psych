"""Simulate subjective-randomness choices from a parametric model family."""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

import yaml

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def resolve_path(path_value: str | Path, config_path: Path | None = None) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    repo_path = REPO_ROOT / path
    if repo_path.exists() or config_path is None:
        return repo_path
    return (config_path.parent / path).resolve()


def load_config(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def load_model(module_path: str):
    return importlib.import_module(module_path)


def load_stimuli(path: Path) -> List[Dict[str, str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Stimuli file must contain a list: {path}")
    stimuli = []
    for item in data:
        if not isinstance(item, Mapping) or "sequence_a" not in item or "sequence_b" not in item:
            raise ValueError(f"Invalid stimulus item: {item!r}")
        stimuli.append({"sequence_a": str(item["sequence_a"]), "sequence_b": str(item["sequence_b"])})
    return stimuli


def generate_rows(
    model_module,
    stimuli: Iterable[Mapping[str, str]],
    params: Mapping[str, float],
    n_participants: int,
    seed: int = 0,
) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    rows: List[Dict[str, Any]] = []
    model_name = getattr(model_module, "MODEL_NAME", model_module.__name__.split(".")[-1])
    params_json = json.dumps(dict(params), sort_keys=True)
    stimuli_list = list(stimuli)
    for participant_id in range(n_participants):
        for trial_index, stim in enumerate(stimuli_list):
            p_left = float(model_module.predict_left(stim, params))
            chose_left = 1 if rng.random() < p_left else 0
            rows.append(
                {
                    "participant_id": participant_id,
                    "trial_index": trial_index,
                    "sequence_a": stim["sequence_a"],
                    "sequence_b": stim["sequence_b"],
                    "chose_left": chose_left,
                    "chose_right": 1 - chose_left,
                    "model": model_name,
                    "true_params": params_json,
                }
            )
    return rows


def write_rows(rows: List[Dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "participant_id",
        "trial_index",
        "sequence_a",
        "sequence_b",
        "chose_left",
        "chose_right",
        "model",
        "true_params",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _params_from_args(args: argparse.Namespace, config: Mapping[str, Any]) -> Dict[str, float]:
    params = dict(config.get("true_params") or {})
    if args.params:
        params.update(json.loads(args.params))
    return {k: float(v) for k, v in params.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate choices from a subjective-randomness model family")
    parser.add_argument("--config", required=True, help="YAML config with model_module, stimuli_path, true_params")
    parser.add_argument("--out", required=True, help="Output responses.csv path")
    parser.add_argument("--params", default=None, help="JSON object overriding true_params")
    parser.add_argument("--n-participants", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    config_path = resolve_path(args.config)
    config = load_config(config_path)
    model = load_model(config["model_module"])
    stimuli = load_stimuli(resolve_path(config["stimuli_path"], config_path))
    sim_cfg = config.get("simulation") or {}
    n_participants = args.n_participants if args.n_participants is not None else int(sim_cfg.get("n_participants", 20))
    seed = args.seed if args.seed is not None else int(sim_cfg.get("seed", 1))
    rows = generate_rows(model, stimuli, _params_from_args(args, config), n_participants, seed)
    write_rows(rows, resolve_path(args.out))
    print(f"Wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
