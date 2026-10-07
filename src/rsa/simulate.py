"""Simulated participants: choices drawn from a ground-truth memo model.

    uv run python -m src.rsa.simulate --gt <model.py> --fit-on <train.csv> \
        --displays <combined.csv> --out <simulated.csv> --provenance <gt_dir/provenance.json>

The ground truth is fitted to real trials (``--fit-on``) so its parameters
are realistic, then every included forced-choice row of ``--displays`` gets a
new ``choice`` drawn from the ground truth's posterior predictive (a fresh
posterior draw per trial). Everything else in those rows, the displays, the
design and the people, is unchanged, so a recovery run sees the real design
with only the choices replaced. Rows the loop does not fit (production
trials, excluded participants, trials without a display) are dropped: their
real responses would otherwise sit beside the simulated ones for agents to
read.

The simulated CSV says nothing about where its choices came from: agents read
it, and the ground truth's name would give the answer away. That record (the
ground truth's file and sha, the fit, the seed) goes to ``--provenance``,
which must live outside the agents' tree.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import tyro

from src.rsa.dataset import context_from_row, load_forced_choice
from src.rsa.fit import FitSettings, class_probs, fit
from src.rsa.context import group_by_shape
from src.rsa.model_file import RSAModel
from src.rsa.split import _counted


def simulate_choices(model: RSAModel, posterior: dict, contexts, seed: int) -> np.ndarray:
    """One choice per context, each from its own posterior draw."""
    rng = np.random.default_rng(seed)
    n_draws = len(next(iter(posterior.values())))
    draws = rng.integers(0, n_draws, size=len(contexts))
    choices = np.empty(len(contexts), dtype=int)
    for group in group_by_shape(contexts).values():
        for d in np.unique(draws[group.indices]):
            rows = np.where(draws[group.indices] == d)[0]
            params = {k: v[d] for k, v in posterior.items()}
            p = np.asarray(model.group_probs(params, group), dtype=float)[rows]
            for r, probs in zip(rows, p):
                probs = np.clip(probs, 0, None)
                choices[group.indices[r]] = rng.choice(len(probs), p=probs / probs.sum())
    return choices


@dataclass
class Args:
    gt: Path
    """The ground-truth model file."""
    fit_on: Path
    """Real trials to fit the ground truth's parameters to (the training split)."""
    displays: Path
    """Rows whose choices are replaced (e.g. the combined trials; split afterwards)."""
    out: Path
    provenance: Path
    """Where the record of the ground truth goes: outside the agents' tree."""
    seed: int = 0
    num_warmup: int = 1000
    num_samples: int = 1000
    num_chains: int = 4


def main(args: Args) -> None:
    model = RSAModel(args.gt, name=Path(args.gt).stem)
    real = load_forced_choice(args.fit_on)
    settings = FitSettings(num_warmup=args.num_warmup, num_samples=args.num_samples,
                           num_chains=args.num_chains, seed=args.seed)
    fitted = fit(model, real.contexts, real.choices, settings)
    if not fitted.converged:
        raise RuntimeError(f"the ground truth's fit did not converge: {fitted.convergence_problems}")
    post = {k: np.asarray(fitted.idata.posterior[k]).reshape(-1) for k in fitted.param_names}
    frame = pd.read_csv(args.displays)
    mask = _counted(frame).to_numpy()
    contexts = [context_from_row(r) for _, r in frame[mask].iterrows()]
    frame.loc[mask, "choice"] = simulate_choices(model, post, contexts, args.seed)
    frame = frame[mask]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    Path(args.provenance).parent.mkdir(parents=True, exist_ok=True)
    Path(args.provenance).write_text(json.dumps(dict(
        ground_truth=str(args.gt), ground_truth_sha256=hashlib.sha256(Path(args.gt).read_bytes()).hexdigest(),
        fit_on=str(args.fit_on), displays=str(args.displays), out=str(out), seed=args.seed,
        settings=vars(settings), n_simulated=int(mask.sum()),
        posterior_means={k: float(v.mean()) for k, v in post.items()},
    ), indent=1))
    print(f"simulated {int(mask.sum())} choices -> {out}")


if __name__ == "__main__":
    main(tyro.cli(Args))
