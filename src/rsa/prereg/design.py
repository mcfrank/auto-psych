"""The pre-registered experiment's design: half chosen by the campaign's own rule, half at random.

    python -m src.rsa.prereg.design --finals data/rsa/live/finals \
        --data $WORK_ROOT/finals_pool/pooled.csv --cache $WORK_ROOT/finals_pool/.fit_cache \
        --displays 208 --participants 1200 --out <design dir>

* **The EIG half** (G/2 displays): `src.rsa.design.run`, unchanged, with the
  three finals as the models going in and the starting models and promoted
  seeds as the bar (prior mass 0.5 on the finals, as every campaign design),
  the campaign's quotas, fitted to the pooled data the frozen predictions come
  from. It is chosen at the responses each display gets in the whole design:
  N/2 people over G/2 displays is N x trials / G a display.
* **The random half** (G/2 displays): drawn from the rest of the pool at random
  within the same quotas (`src.rsa.prereg.power.random_design`), so the test
  does not rest only on displays chosen to favour a disagreement.

Writes ``design.json`` (the G displays, labelled ``eig_XX`` and ``rand_XX``),
``trial_lists.json`` (N balanced lists of ``trials`` displays plus catch
trials, `src.rsa.experiment.design.trial_lists`) and ``prereg_design.json``
(both halves' picks and kinds, the quota counts, the hashes); the EIG half's
own record is in ``eig_half/``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np
import tyro

from src.rsa.design import run as design_run
from src.rsa.design.eig import composition
from src.rsa.experiment.design import Design, TrialSpec, trial_lists
from src.rsa.loop.fitting import NO_TIME_LIMIT
from src.rsa.loop.novelty import pool_digest, pool_record
from src.rsa.outer.run import DESIGN_QUOTAS, STARTING_MODELS
from src.rsa.prereg.power import TRIALS, random_design
from src.runtime.config import REPO_ROOT

PROMOTED = REPO_ROOT / "data" / "rsa" / "live_seeds" / "models"
N_CATCH = 2


@dataclass
class Args:
    finals: Path
    """The final models (a manifest and their files): data/rsa/live/finals."""
    data: Path
    """The pooled trials the frozen predictions were fitted to."""
    cache: Path
    """Their fit cache (the pooled fits are reused, not repeated)."""
    out: Path
    displays: int = 208
    participants: int = 1200
    starting: Path = STARTING_MODELS
    promoted: Path = PROMOTED
    trials: int = TRIALS
    n_catch: int = N_CATCH
    quotas: List[str] = field(default_factory=lambda: list(DESIGN_QUOTAS))
    carried_share: float = 0.5
    n_draws: int = 200
    n_scenarios: int = 2000
    seed: int = 2026
    """The selection's and the random half's seed (fixed in the pre-registration)."""
    num_warmup: int = 1000
    num_samples: int = 1000
    num_chains: int = 4
    """The pooled fits' settings (as the frozen predictions': the cached fits are reused)."""


def main(args: Args) -> dict:
    g, n = args.displays, args.participants
    if g % 2 or n % 2:
        raise ValueError(f"displays ({g}) and participants ({n}) must be even: the design is two halves")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    half = g // 2
    record = design_run.main(design_run.Args(
        models_dir=args.finals, data=args.data, cache=args.cache, out=out / "eig_half",
        bar_models_dirs=[args.starting, args.promoted], carried_share=args.carried_share, quotas=list(args.quotas),
        trials_per_participant=args.trials, participants=n // 2, displays=[half], power_participants=[n // 2],
        n_draws=args.n_draws, n_scenarios=args.n_scenarios, n_power_scenarios=args.n_scenarios, seed=args.seed,
        fit_seed=0, dense_mass=True, time_limit_sec=NO_TIME_LIMIT, num_warmup=args.num_warmup,
        num_samples=args.num_samples, num_chains=args.num_chains))
    pool = design_run.design_pool()
    index = {json.dumps(pool_record(ctx), sort_keys=True): i for i, ctx in enumerate(pool)}
    eig = Design.load(out / "eig_half" / f"design_d{half}.json")
    eig_idx = [index[json.dumps(pool_record(s.context()), sort_keys=True)] for s in eig.specs]
    rand_idx = random_design(pool, args.quotas, half, np.random.default_rng([args.seed, g]), exclude=eig_idx)
    specs = tuple(eig.specs) + tuple(TrialSpec.from_context(pool[i], label=f"rand_{k:02d}")
                                     for k, i in enumerate(rand_idx))
    design = Design(name=f"prereg_final_d{g}_n{n}", specs=specs)
    (out / "design.json").write_text(json.dumps(design.to_json(), indent=1))
    lists = trial_lists(design, seed=args.seed, n_lists=n, n_catch=args.n_catch, n_trials=args.trials)
    (out / "trial_lists.json").write_text(json.dumps(lists, indent=1))
    quotas = design_run.parse_quotas(args.quotas, pool, half)
    summary = dict(
        displays=g, participants=n, trials=args.trials, n_catch=args.n_catch, seed=args.seed,
        quotas=list(args.quotas), pool_size=len(pool), pool_digest=pool_digest(pool),
        design_sha256=design.sha256,
        trial_lists_sha256=hashlib.sha256((out / "trial_lists.json").read_bytes()).hexdigest(),
        data_sha256=record["data_sha256"], models=record["models"], merged=record["merged"],
        screened_out=record["screened_out"],
        eig=dict(kinds=design_run.kinds(pool, eig_idx),
                 quota_counts=composition(quotas, eig_idx) if quotas else {},
                 n_eig_picks=record["designs"][0]["n_eig_picks"], power=record["designs"][0]["power"]),
        random=dict(kinds=design_run.kinds(pool, rand_idx),
                    quota_counts=composition(quotas, rand_idx) if quotas else {}),
    )
    (out / "prereg_design.json").write_text(json.dumps(summary, indent=1))
    print(f"design {design.sha256[:12]}: {half} EIG + {half} random displays, {n} lists of {args.trials} "
          f"(+{args.n_catch} catch); EIG kinds {summary['eig']['kinds']}; random kinds {summary['random']['kinds']}",
          flush=True)
    return summary


if __name__ == "__main__":
    from src.rsa.cpus import pin_main_thread

    pin_main_thread()  # one core per process (src.rsa.cpus)
    main(tyro.cli(Args))
