"""RSA reference-game data behind the paper's Figures 2 and 3.

Read from the campaign's aggregates in the repository; nothing is fitted.

- Figure 2: per live chain, its starting models and its final model against
  the best promoted seed, by log predictive density on the pre-registered
  experiment (OSF s48jv), which none of them was fitted on:
  ``data/rsa/prereg/run/prereg_analysis.json``. The promoted seeds are the
  best models of the campaign's existing-data loop ("run 2"), refitted; each
  chain started from its share of them (``data/rsa/live_seeds/chains/
  chains.json``), and the best of all of them is the pre-registered bar. So
  the zero is that bar and a final's difference and paired SE are the
  pre-registered test's (``primary``). A starting model's difference comes
  from the ``lpd`` totals; the analysis records no paired SE for it. The
  textbook RSA models (the campaign's "starting models", ``seed:<name>``) are
  left out; no other model of a chain was scored on the pre-registered
  experiment.
- Figure 3: each display's observed proportion choosing each object class
  (all the campaign's data pooled) beside the posterior-mean class
  probability of the best chain final (highest held-out density) and of the
  best promoted seed, both fitted on the pooled data: ``data/rsa/viz/
  panels.json`` (aggregates only). Unlike subjective randomness, the best
  final is chosen by its held-out density, not by a fit on the pooled data:
  panels.json records no pooled ELPD.

The output has the same schema as ``sr_figure_data.py``'s; its ``best_seed``
is the best promoted seed.

Usage:
    uv run python -m scripts.paper.rsa_figure_data
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import tyro
from pyprojroot import here

DEFAULT_PREREG = here() / "data" / "rsa" / "prereg" / "run" / "prereg_analysis.json"
DEFAULT_PANELS = here() / "data" / "rsa" / "viz" / "panels.json"
DEFAULT_CHAIN_SEEDS = here() / "data" / "rsa" / "live_seeds" / "chains" / "chains.json"
DEFAULT_OUT = here() / "data" / "paper" / "figure_data" / "rsa.json"

PROMOTED_PREFIX = "promoted:"
PROMOTED_PANEL_KEY = "promoted"  # panels.json's key for the best promoted seed


@dataclass
class Args:
    prereg: Path = DEFAULT_PREREG
    panels: Path = DEFAULT_PANELS
    chain_seeds: Path = DEFAULT_CHAIN_SEEDS
    """Each chain's starting models (its share of the promoted seeds)."""
    chains: List[str] = field(default_factory=lambda: ["c0", "c1", "c2"])
    """panels.json keys of the live chains' finals (``c<k>`` is chain k)."""
    out: Path = DEFAULT_OUT


def _lpds(prereg: Dict[str, Any], prefix: str) -> Dict[str, float]:
    found = {k[len(prefix):]: float(v) for k, v in prereg["lpd"].items() if k.startswith(prefix)}
    if not found:
        raise RuntimeError(f"prereg_analysis.json scores no {prefix!r} model")
    return found


def _best_promoted(prereg: Dict[str, Any]) -> str:
    promoted = _lpds(prereg, PROMOTED_PREFIX)
    recorded = prereg["best_promoted"]
    if not recorded.startswith(PROMOTED_PREFIX):
        raise RuntimeError(f"best_promoted {recorded!r} is not a promoted seed")
    recorded = recorded[len(PROMOTED_PREFIX):]
    by_density = max(promoted, key=promoted.get)
    if recorded != by_density:
        raise RuntimeError(
            f"the recorded best promoted seed {recorded!r} is not the best by density ({by_density!r})"
        )
    return recorded


def _chain_starting_models(chain_seeds: Dict[str, Any], chain: str) -> List[str]:
    match = re.fullmatch(r"c(\d+)", chain)
    if match is None:
        raise ValueError(f"chain {chain!r} is not named c<k>")
    k = int(match.group(1))
    entries = [c for c in chain_seeds["chains"] if c["chain"] == k]
    if len(entries) != 1:
        raise RuntimeError(f"chains.json has {len(entries)} entries for chain {k}")
    return list(entries[0]["models"])


def chain_record(chain: str, final: str, starting: List[str], prereg: Dict[str, Any],
                 best_promoted: str) -> Dict[str, Any]:
    if final not in prereg["primary"]:
        raise RuntimeError(f"{chain}'s final {final!r} is not in the pre-registered test")
    promoted = _lpds(prereg, PROMOTED_PREFIX)
    unscored = [m for m in starting if m not in promoted]
    if unscored:
        raise RuntimeError(f"{chain}'s starting models {unscored} were not scored on the test")
    recorded = prereg["primary"][final]
    from_totals = float(prereg["lpd"][final]) - promoted[best_promoted]
    if abs(float(recorded["diff"]) - from_totals) > 1e-6 * max(1.0, abs(from_totals)):
        raise RuntimeError(
            f"{chain}: the recorded difference {recorded['diff']} is not the density "
            f"totals' difference {from_totals}"
        )
    vs_best_seed: Dict[str, Dict[str, Any]] = {
        final: {"diff": float(recorded["diff"]), "se": float(recorded["se"])}
    }
    for model in starting:
        if model == best_promoted:
            vs_best_seed[model] = {"diff": 0.0, "se": 0.0}
        else:
            vs_best_seed[model] = {"diff": promoted[model] - promoted[best_promoted], "se": None}
    return {
        "run": chain,
        "n_trials": int(prereg["n_trials"]),
        "best_model": final,
        "best_seed": best_promoted,
        "elpd": {final: float(prereg["lpd"][final]), **{m: promoted[m] for m in starting}},
        "vs_best_seed": vs_best_seed,
        "others_at_end": {},  # no other model of a chain was scored on the test
    }


def condition_table(panels: Dict[str, Any], chain: str) -> List[Dict[str, Any]]:
    """One row per (display, object class): the observed proportion choosing
    the class and the chain final's and best promoted seed's predicted
    probability."""
    rows = []
    for display, panel in enumerate(panels["panels"]):
        n = int(panel["n"])
        if sum(int(c["count"]) for c in panel["classes"]) != n:
            raise RuntimeError(f"display {display}: class counts do not sum to n = {n}")
        for cls in panel["classes"]:
            rows.append(
                {
                    "dataset": panel["dataset"],
                    "display": display,
                    "cls": cls["cls"],
                    "n": n,
                    "observed": int(cls["count"]) / n,
                    "best_model": float(cls["preds"][chain]),
                    "best_seed": float(cls["preds"][PROMOTED_PANEL_KEY]),
                }
            )
    return rows


def main(args: Args) -> None:
    prereg = json.loads(args.prereg.read_text())
    panels = json.loads(args.panels.read_text())
    chain_seeds = json.loads(args.chain_seeds.read_text())
    best_promoted = _best_promoted(prereg)
    if panels["models"][PROMOTED_PANEL_KEY]["model"] != best_promoted:
        raise RuntimeError(
            f"panels.json's {PROMOTED_PANEL_KEY!r} model is "
            f"{panels['models'][PROMOTED_PANEL_KEY]['model']!r}, not the best promoted seed "
            f"{best_promoted!r}"
        )
    runs = [
        chain_record(c, panels["models"][c]["model"], _chain_starting_models(chain_seeds, c),
                     prereg, best_promoted)
        for c in args.chains
    ]
    best = max(runs, key=lambda r: r["elpd"][r["best_model"]])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "domain": "rsa",
                "scoring": (
                    "log predictive density on the held-out pre-registered experiment "
                    f"({int(prereg['n_trials']):,} trials), against the best promoted seed "
                    "(the pre-registered bar)"
                ),
                "runs": runs,
                "pooled": {
                    "best_model": best["best_model"],
                    "best_seed": best_promoted,
                    "winner_runs": {r["best_model"]: r["run"] for r in runs},
                    "selected_by": "highest log predictive density on the pre-registered experiment",
                },
                "conditions": condition_table(panels, best["run"]),
            },
            indent=2,
        )
    )
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main(tyro.cli(Args))
