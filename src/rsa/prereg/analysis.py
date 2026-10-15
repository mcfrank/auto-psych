"""The pre-registered analysis (docs/auto_rsa/PREREG_FINAL.md, section "Analysis").

    uv run python -m src.rsa.prereg.analysis --data <experiment>/data/modelled.csv \
        --predictions finals_pool_preds_final.json bar_pool_preds.json --out analysis.json

Each model's prediction for a trial is its frozen posterior-mean probability of
the chosen class on that display (`src.rsa.prereg.predictions`); nothing is
fitted to the new data. A model's score is its summed log predictive density
(lpd) over the modelled trials (catch trials are not modelled).

**The primary test** (one per final model): the final's lpd minus the best
promoted seed's, where the best promoted seed is the one with the highest lpd
on these data (chosen after the fact, as claim 2 did: the bar's best is the
strongest it can be, so the test is conservative). Its standard error is
clustered by display (`src.models.clustered_se.cluster_dse`: the responses to
one display share its error). z = difference / SE; one-sided p = 1 - Phi(z);
Holm's step-down procedure over the three finals at familywise alpha 0.05.

Reported beside it, not tested: each final against the best starting model and
against every bar model; each half of the design (EIG, random) on its own.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import tyro
from scipy.stats import norm

from src.models.clustered_se import cluster_dse
from src.rsa.dataset import load_forced_choice
from src.rsa.prereg.predictions import PoolPredictions, load_predictions

ALPHA = 0.05
BAR_GROUPS = ("seed", "promoted")  # bar model labels are "<group>:<name>"
HALVES = {"eig": "eig_", "random": "rand_"}  # the design's condition labels


def trial_lpd(preds: PoolPredictions, contexts, choices) -> np.ndarray:
    """(models, trials) log probability of each trial's chosen class."""
    idx = np.array([preds.display(ctx) for ctx in contexts])
    cls = np.array([ctx.choice_classes()[c] for ctx, c in zip(contexts, choices)])
    p = preds.probs[:, idx, cls]
    if np.any(p <= 0):
        bad = preds.names[int(np.where((p <= 0).any(1))[0][0])]
        raise ValueError(f"{bad} gives an observed choice probability 0")
    return np.log(p)


def holm(pvalues: Dict[str, float], alpha: float = ALPHA) -> Dict[str, bool]:
    """Holm's step-down rejections: the k-th smallest p against alpha / (m - k + 1), stopping at the first that fails."""
    out = {k: False for k in pvalues}
    m = len(pvalues)
    for k, (name, p) in enumerate(sorted(pvalues.items(), key=lambda kv: kv[1])):
        if p > alpha / (m - k):
            break
        out[name] = True
    return out


def versus(a: np.ndarray, b: np.ndarray, units: np.ndarray) -> dict:
    diff = float(a.sum() - b.sum())
    se = float(cluster_dse(a, b, units))
    z = diff / se if se > 0 else (np.inf if diff > 0 else -np.inf)
    return dict(diff=diff, se=se, z=float(z), p_one_sided=float(norm.sf(z)))


def best_of(lpd: Dict[str, np.ndarray], group: str) -> str:
    names = [k for k in lpd if k.startswith(f"{group}:")]
    if not names:
        raise ValueError(f"no {group} models among the predictions")
    return max(names, key=lambda k: lpd[k].sum())


def primary(lpd: Dict[str, np.ndarray], units: np.ndarray, finals: Sequence[str], alpha: float = ALPHA) -> dict:
    """The pre-registered test and its descriptive companions, from per-trial lpd."""
    bar = best_of(lpd, "promoted")
    tests = {f: versus(lpd[f], lpd[bar], units) for f in finals}
    reject = holm({f: t["p_one_sided"] for f, t in tests.items()}, alpha)
    for f in finals:
        tests[f]["holm_reject"] = reject[f]
    best_seed = best_of(lpd, "seed")
    bar_models = [k for k in lpd if k.split(":", 1)[0] in BAR_GROUPS]
    return dict(
        best_promoted=bar, best_starting=best_seed, alpha=alpha,
        primary=tests, all_finals_beat_the_bar=all(reject.values()),
        vs_best_starting={f: versus(lpd[f], lpd[best_seed], units) for f in finals},
        each_bar_model={f: {k: versus(lpd[f], lpd[k], units) for k in bar_models} for f in finals},
        lpd={k: float(v.sum()) for k, v in sorted(lpd.items(), key=lambda kv: -kv[1].sum())},
    )


def analyse(data: Path, preds: PoolPredictions, finals: Optional[Sequence[str]] = None, alpha: float = ALPHA) -> dict:
    trials = load_forced_choice(data)
    frame = trials.frame
    if (frame["condition"].astype(str) == "catch").any():
        raise ValueError(f"{data} has catch trials: the analysis reads the modelled trials (data/modelled.csv)")
    finals = list(finals) if finals is not None else [n for n in preds.names if ":" not in n]
    if len(finals) == 0 or any(f not in preds.names for f in finals):
        raise ValueError(f"finals {finals} are not all among the predictions {preds.names}")
    lp = trial_lpd(preds, trials.contexts, trials.choices)
    conditions = frame["condition"].astype(str).to_numpy()
    units = pd.factorize(conditions)[0]
    lpd = dict(zip(preds.names, lp))
    out = dict(
        n_trials=len(conditions), n_displays=int(units.max() + 1),
        n_participants=int(frame["participant_id"].nunique()),
        predictions_sha256=preds.sha256, finals=finals,
        **primary(lpd, units, finals, alpha),
    )
    halves = {}
    for half, prefix in HALVES.items():
        keep = np.char.startswith(conditions.astype(str), prefix)
        if keep.any():
            sub = {k: v[keep] for k, v in lpd.items()}
            u = pd.factorize(conditions[keep])[0]
            halves[half] = dict(n_trials=int(keep.sum()), n_displays=int(u.max() + 1),
                                **{k: v for k, v in primary(sub, u, finals, alpha).items()
                                   if k in ("best_promoted", "primary", "vs_best_starting")})
    out["by_half"] = halves
    return out


@dataclass
class Args:
    data: Path
    """The experiment's modelled trials (catch trials removed)."""
    predictions: List[Path]
    """The frozen predictions files (the finals' and the bar's)."""
    out: Path
    finals: List[str] = field(default_factory=list)
    """The final models (default: every unprefixed model in the predictions)."""
    alpha: float = ALPHA


def main(args: Args) -> dict:
    out = analyse(args.data, load_predictions(args.predictions), args.finals or None, args.alpha)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"best promoted seed: {out['best_promoted']}")
    for f, t in out["primary"].items():
        print(f"  {f}: {t['diff']:+.1f} ± {t['se']:.1f} (z {t['z']:.2f}, p {t['p_one_sided']:.4f}) "
              f"{'REJECTS' if t['holm_reject'] else 'does not reject'} H0 (Holm)")
    return out


if __name__ == "__main__":
    main(tyro.cli(Args))
