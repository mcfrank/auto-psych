"""Power of the pre-registered test, to choose how many displays and people.

    uv run python -m src.rsa.prereg.power --predictions finals_pool_preds_final.json \
        bar_pool_preds.json --displays 104 208 312 --participants 800 1200 1600 --out power.json

The analysis (`src.rsa.prereg.analysis`) is run on simulated experiments. Each
experiment shows ``G`` displays, each answered by ``m = floor(N x (1 -
exclusion) x trials / G)`` people (balanced trial lists, catch exclusions as in
the campaign), and its responses are drawn from a **truth** made for one final
model f at a time:

* ``calibrated:<delta>``: on each display the truth is ``lam x p_f + (1 - lam)
  x p_b``, with b the promoted seed nearest f on the design (smallest KL(p_f ||
  p_b): the strongest promoted competitor were f right) and lam chosen so the
  expected advantage of f over b is ``delta`` nats per trial on the design. The
  advantage is linear in lam (`calibrate`), so lam is exact. ``delta`` is set
  from the campaign: the committed models' advantages over the best promoted
  seed in claim 2 were 0.0058-0.0223 nats per trial (CAMPAIGN_LOG.md).
  If even lam = 1 (f is the truth) gives less than ``delta``, lam is 1 and the
  record says so (``capped``).
* ``model``: f is the truth (lam = 1), the optimistic bound.

f's power is how often Holm's procedure over the three finals rejects f's
null when the data come from f's truth; the best promoted seed is chosen on
each simulated dataset, as in the analysis.

**The designs.** Without ``--design``, each simulated experiment's displays are
drawn at random within the quotas (`random_design`), a new design every
``sims_per_design`` experiments: a conservative stand-in for the real design,
whose EIG half aims at the models' disagreements. With ``--design`` (the
pre-registered design's ``design.json``), that design.

The model-based truths make the displays differ only as the two models do; real
displays also carry both models' misfits, so the record adds the **empirical
scaling** of claim 2's own tests (`empirical_power`): z grows as sqrt(G) at
fixed responses per display.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import tyro
from scipy.stats import norm

from src.rsa.context import Context
from src.rsa.design.eig import LOG_FLOOR, _allowed
from src.rsa.design.run import parse_quotas
from src.rsa.outer.run import DESIGN_QUOTAS
from src.rsa.prereg.analysis import ALPHA, holm
from src.rsa.prereg.predictions import PoolPredictions, load_predictions

EXCLUSION_RATE = 0.078  # the campaign's catch exclusions: 1,659 kept of 1,799 (CAMPAIGN_LOG.md)
TRIALS = 13  # designed displays a person answers (+ 2 catch), as in the campaign
COST_PER_PERSON = 0.50 * 4 / 3  # $0.50 reward + Prolific's fee: $133.33 per 200 in the campaign
# Claim 2's tests of the committed model against the best promoted seed, at 52
# displays and ~47 responses a display (prospective.json, committed_vs).
CLAIM2_Z = {"c0e2": 32.2 / 23.3, "c0e3": 43.6 / 14.4, "c1e2": 32.4 / 12.0, "c1e3": 55.9 / 17.9,
            "c2e2": 13.6 / 27.2, "c2e3": 19.2 / 15.1}
CLAIM2_DISPLAYS = 52


def random_design(pool: Sequence[Context], quotas: Sequence[str], n: int, rng: np.random.Generator,
                  exclude: Sequence[int] = ()) -> List[int]:
    """``n`` pool displays at random, every quota met: each pick is uniform among
    the displays that leave every quota reachable (`src.rsa.design.eig._allowed`)."""
    qs = parse_quotas(quotas, pool, n)
    left = sorted(set(range(len(pool))) - set(exclude))
    picks: List[int] = []
    for _ in range(n):
        allowed = _allowed(qs, picks, left, n)
        if not allowed:
            raise ValueError(f"no display keeps the quotas {list(quotas)} reachable after {len(picks)} picks")
        c = allowed[int(rng.integers(len(allowed)))]
        picks.append(c)
        left.remove(c)
    return picks


def _logp(preds: PoolPredictions) -> np.ndarray:
    return np.where(preds.valid, np.log(np.maximum(preds.probs, LOG_FLOOR)), 0.0)


def calibrate(pf: np.ndarray, pb: np.ndarray, valid: np.ndarray, delta: float) -> tuple[float, bool]:
    """The lam in [0, 1] whose truth lam p_f + (1 - lam) p_b gives f an expected
    advantage of ``delta`` nats per trial over b (equal responses per display),
    and whether it was capped at 1. The advantage is A(lam) = lam A(1) + (1 - lam) A(0)."""
    gap = np.where(valid, np.log(np.maximum(pf, LOG_FLOOR)) - np.log(np.maximum(pb, LOG_FLOOR)), 0.0)
    a1 = float((pf * gap).sum(-1).mean())
    a0 = float((pb * gap).sum(-1).mean())
    if delta >= a1:
        return 1.0, delta > a1
    return float((delta - a0) / (a1 - a0)), False


def nearest_promoted(preds: PoolPredictions, final: str, design: Sequence[int]) -> str:
    pf = preds.model(final)[design]
    logp = _logp(preds)
    best, kl = "", np.inf
    for k, name in enumerate(preds.names):
        if name.startswith("promoted:"):
            d = float((pf * (logp[preds.names.index(final), design] - logp[k, design])).sum(-1).mean())
            if d < kl:
                best, kl = name, d
    if not best:
        raise ValueError("no promoted models among the predictions")
    return best


def simulate(preds: PoolPredictions, finals: Sequence[str], final: str, design: Sequence[int], m: int,
             n_sims: int, truth: str, rng: np.random.Generator, alpha: float = ALPHA) -> dict:
    """``n_sims`` experiments on ``design`` with ``m`` responses a display, from ``final``'s truth."""
    design = np.asarray(design)
    pf = preds.model(final)[design]
    valid = preds.valid[design]
    near = nearest_promoted(preds, final, design)
    if truth == "model":
        lam, capped = 1.0, False
    elif truth.startswith("calibrated:"):
        lam, capped = calibrate(pf, preds.model(near)[design], valid, float(truth.split(":", 1)[1]))
    else:
        raise ValueError(f"truth {truth!r} is not 'model' or 'calibrated:<delta>'")
    p_true = lam * pf + (1 - lam) * preds.model(near)[design]
    counts = rng.multinomial(m, p_true / p_true.sum(-1, keepdims=True), size=(n_sims, len(design)))
    sums = np.einsum("sgw,kgw->skg", counts, _logp(preds)[:, design])  # (sims, models, displays)
    promoted = [k for k, n in enumerate(preds.names) if n.startswith("promoted:")]
    totals = sums.sum(-1)
    bar = np.asarray(promoted)[np.argmax(totals[:, promoted], axis=1)]
    g = len(design)
    rejected = np.zeros(n_sims, dtype=bool)
    zs = np.zeros(n_sims)
    for s in range(n_sims):
        pvals = {}
        for f in finals:
            diff = sums[s, preds.names.index(f)] - sums[s, bar[s]]
            se = math.sqrt(g * np.var(diff))
            z = diff.sum() / se if se > 0 else np.inf
            pvals[f] = float(norm.sf(z))
            if f == final:
                zs[s] = z
        rejected[s] = holm(pvals, alpha)[final]
    return dict(power=float(rejected.mean()), z_mean=float(zs[np.isfinite(zs)].mean()), lam=lam, capped=capped,
                nearest_promoted=near, n_rejected=int(rejected.sum()), n_sims=n_sims)


def empirical_power(z0: float, g0: int, g: int, alpha: float) -> float:
    """Claim 2's test grown from ``g0`` to ``g`` displays at the same responses
    per display: z = z0 sqrt(g / g0) (a display-clustered SE falls as 1/sqrt(G))."""
    return float(norm.cdf(z0 * math.sqrt(g / g0) - norm.isf(alpha)))


def responses_per_display(participants: int, displays: int, trials: int = TRIALS,
                          exclusion: float = EXCLUSION_RATE) -> int:
    return int(participants * (1 - exclusion) * trials // displays)


@dataclass
class Args:
    predictions: List[Path]
    out: Path
    displays: List[int] = field(default_factory=lambda: [104, 208, 312])
    participants: List[int] = field(default_factory=lambda: [800, 1200, 1600])
    truths: List[str] = field(default_factory=lambda: ["calibrated:0.0058", "calibrated:0.0078", "model"])
    """'calibrated:<delta nats per trial>' or 'model' (`simulate`)."""
    finals: List[str] = field(default_factory=list)
    design: Optional[Path] = None
    """A design's design.json (default: random designs within the quotas)."""
    quotas: List[str] = field(default_factory=lambda: list(DESIGN_QUOTAS))
    n_sims: int = 2000
    sims_per_design: int = 100
    trials: int = TRIALS
    exclusion: float = EXCLUSION_RATE
    alpha: float = ALPHA
    seed: int = 0


def main(args: Args) -> dict:
    from src.rsa.experiment.design import Design

    preds = load_predictions(args.predictions)
    finals = args.finals or [n for n in preds.names if ":" not in n]
    fixed = None
    if args.design is not None:
        fixed = [preds.display(spec.context()) for spec in Design.load(args.design).specs]
    rows = []
    for g in ([len(fixed)] if fixed else args.displays):
        designs_rng = np.random.default_rng([args.seed, g])
        n_designs = 1 if fixed else max(1, args.n_sims // args.sims_per_design)
        designs = [fixed] if fixed else [random_design(preds.pool, args.quotas, g, designs_rng)
                                         for _ in range(n_designs)]
        for n in args.participants:
            m = responses_per_display(n, g, args.trials, args.exclusion)
            for truth in args.truths:
                per_final = {}
                for f in finals:
                    rng = np.random.default_rng([args.seed, g, n, finals.index(f), args.truths.index(truth)])
                    parts = [simulate(preds, finals, f, d, m, args.n_sims // n_designs, truth, rng, args.alpha)
                             for d in designs]
                    sims = sum(p["n_sims"] for p in parts)
                    per_final[f] = dict(
                        power=sum(p["n_rejected"] for p in parts) / sims, n_sims=sims,
                        z_mean=float(np.mean([p["z_mean"] for p in parts])),
                        lam=float(np.mean([p["lam"] for p in parts])),
                        capped=sum(p["capped"] for p in parts),
                        nearest_promoted=sorted({p["nearest_promoted"] for p in parts}))
                row = dict(displays=g, participants=n, responses_per_display=m, truth=truth,
                           cost_usd=round(n * COST_PER_PERSON), min_power=min(v["power"] for v in per_final.values()),
                           finals=per_final)
                rows.append(row)
                print(f"G={g} N={n} (m={m}, ${row['cost_usd']}) {truth}: "
                      + ", ".join(f"{f[:24]} {v['power']:.2f}" for f, v in per_final.items()), flush=True)
    empirical = {
        str(g): {k: dict(at_alpha=empirical_power(z, CLAIM2_DISPLAYS, g, args.alpha),
                         at_alpha_over_3=empirical_power(z, CLAIM2_DISPLAYS, g, args.alpha / 3))
                 for k, z in CLAIM2_Z.items()}
        for g in ([len(fixed)] if fixed else args.displays)}
    record = dict(predictions_sha256=preds.sha256, finals=finals, design=None if fixed is None else str(args.design),
                  quotas=args.quotas, trials=args.trials, exclusion=args.exclusion, alpha=args.alpha, seed=args.seed,
                  grid=rows, empirical_scaling=dict(
                      note=("claim 2's z (committed model vs best promoted seed, 52 displays, ~47 responses a "
                            "display) grown as sqrt(G); valid at >= ~47 responses per display"),
                      claim2_z=CLAIM2_Z, power=empirical))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(record, indent=1))
    return record


if __name__ == "__main__":
    main(tyro.cli(Args))
