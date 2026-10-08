"""Design a live experiment from the current models, and say how much power it has.

    uv run python -m src.rsa.design.run --models-dir data/rsa/live_seeds/models \
        --data <the trials the models are fitted to> --cache <fit cache> --out <design dir>

1. Each model is fitted to ``--data`` (the loop's fit rule and settings; a
   cached fit is reused, so after `src.rsa.promote` nothing is refitted).
2. Its posterior draws give per-draw choice-class probabilities on the design
   pool: every game up to 4 objects x 4 features, every word and the prior
   query (`src.rsa.design_space.context_pool`; the displays the experiment
   page can show). A model undefined on some pool display is left out of the
   design, on record (``screened_out``), as main's design does.
3. `src.rsa.design.eig.select` picks ``n_select`` displays, each answered by
   every participant (``n_responses`` = participants per experiment).
4. `src.rsa.design.eig.power` scores the design at each of ``power_ns`` on
   fresh scenarios: how often the model that generated the data ends with the
   highest posterior.

Writes ``<out>/design.json`` (a `src.rsa.experiment.design.Design`: build the
page with ``python -m src.rsa.experiment.build --design``) and
``<out>/eig.json`` (the picks, their joint EIG, the power table, the models,
the settings).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Sequence

import numpy as np
import tyro

from src.models.model_manifest import read_manifest_names
from src.rsa.context import Context, group_by_shape
from src.rsa.design.eig import power, select
from src.rsa.design_space import context_pool
from src.rsa.experiment.design import Design, TrialSpec
from src.rsa.fit import FitSettings, class_probs, draw_batches, posterior_flat
from src.rsa.loop.fitting import FIT_TIME_LIMIT_SEC, loop_fit
from src.rsa.loop.novelty import POOL_SIZES, pool_digest, pool_record
from src.rsa.model_file import RSAModel

WIDTH = 4  # the most objects a pool display has


def design_pool() -> List[Context]:
    return context_pool(POOL_SIZES, include_prior_queries=True)


def class_layout(pool: Sequence[Context]) -> np.ndarray:
    """(displays, WIDTH): the slots that hold a choice class (its first object)."""
    valid = np.zeros((len(pool), WIDTH), dtype=bool)
    for i, ctx in enumerate(pool):
        for j, cls in enumerate(ctx.choice_classes()):
            valid[i, j] = cls == j
    return valid


def class_prob_draws(model: RSAModel, flat, pool: Sequence[Context], n_draws: int) -> np.ndarray:
    """(draws, displays, WIDTH) class probabilities at ``n_draws`` evenly spaced posterior draws."""
    total = len(next(iter(flat.values())))
    take = np.linspace(0, total - 1, min(n_draws, total)).astype(int)
    out = np.zeros((len(take), len(pool), WIDTH))
    for group in group_by_shape(pool).values():
        start = 0
        for n, params in draw_batches(flat, take):
            p = np.asarray(class_probs(model.draws_probs(params, group), group.classes), dtype=np.float64)[:n]
            out[start:start + n, group.indices, : group.shape[0]] = p
            start += n
    return out


@dataclass
class Args:
    models_dir: Path
    """The models to design for (a manifest and their files), e.g. data/rsa/live_seeds/models."""
    data: Path
    """The trials they are fitted to (all data so far)."""
    cache: Path
    """Fit cache (promote's, to reuse its fits)."""
    out: Path
    n_select: int = 60
    """Designed displays per participant (64 trials with 2 catch trials and the practice trial)."""
    n_responses: int = 100
    """Participants per experiment: every one answers every designed display."""
    power_ns: List[int] = field(default_factory=lambda: [40, 60, 100, 150])
    n_draws: int = 200
    n_scenarios: int = 2000
    n_power_scenarios: int = 4000
    seed: int = 0
    num_warmup: int = 1000
    num_samples: int = 1000
    num_chains: int = 4
    fit_seed: int = 0
    time_limit_sec: float = FIT_TIME_LIMIT_SEC


def main(args: Args) -> dict:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    settings = FitSettings(num_warmup=args.num_warmup, num_samples=args.num_samples, num_chains=args.num_chains,
                           seed=args.fit_seed)
    pool = design_pool()
    valid = class_layout(pool)
    names, probs, screened = [], [], []
    for name in read_manifest_names(args.models_dir):
        path = Path(args.models_dir) / f"{name}.py"
        fitted = loop_fit(path, name, args.data, settings, args.cache, time_limit_sec=args.time_limit_sec)
        p = class_prob_draws(RSAModel(path, name=name), posterior_flat(fitted), pool, args.n_draws)
        bad = ~np.isfinite(p).all(axis=(0, 2)) | ~np.isclose(p.sum(-1), 1.0, atol=1e-4).all(0)
        if bad.any():
            screened.append(dict(model=name, reason="choice probabilities undefined on some pool displays",
                                 n_displays=int(bad.sum()), examples=[pool_record(pool[i]) for i in np.where(bad)[0][:3]]))
            print(f"  {name}: screened out ({int(bad.sum())} pool displays undefined)", flush=True)
            continue
        names.append(name)
        probs.append(p)
        print(f"  {name}: {p.shape[0]} draws on {len(pool)} displays", flush=True)
    if len(names) < 2:
        raise ValueError(f"{len(names)} usable models: a design needs at least two to tell apart")

    sel = select(probs, valid, args.n_select, n_responses=args.n_responses, n_scenarios=args.n_scenarios, seed=args.seed)
    table = power(probs, valid, sel.indices, args.power_ns, n_scenarios=args.n_power_scenarios, seed=args.seed + 101,
                  names=names)
    specs = tuple(TrialSpec.from_context(pool[i], label=f"eig_{k:02d}") for k, i in enumerate(sel.indices))
    design = Design(name=f"eig_{len(names)}models_n{args.n_responses}", specs=specs)
    (out / "design.json").write_text(json.dumps(design.to_json(), indent=1))
    record = dict(
        models=names, screened_out=screened, n_select=args.n_select, n_responses=args.n_responses,
        pool_size=len(pool), pool_digest=pool_digest(pool), n_draws=args.n_draws, n_scenarios=args.n_scenarios,
        seed=args.seed, fit_settings=vars(settings),
        data_sha256=hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
        design_sha256=design.sha256,
        picks=[dict(display=pool_record(pool[i]), joint_eig_bits=b, source=s)
               for i, b, s in zip(sel.indices, sel.joint_eig_bits, sel.sources)],
        n_eig_picks=sum(s == "eig" for s in sel.sources),
        power=table,
    )
    (out / "eig.json").write_text(json.dumps(record, indent=1))
    print(f"design: {sum(s == 'eig' for s in sel.sources)} EIG picks + "
          f"{sum(s != 'eig' for s in sel.sources)} single-response fill; joint EIG "
          f"{sel.joint_eig_bits[-1]:.2f} of {np.log2(len(names)):.2f} bits")
    for row in table:
        print(f"  N={row['n_responses']}: P(generating model wins) {row['p_correct']:.3f} ± {row['p_correct_se']:.3f}, "
              f"joint EIG {row['joint_eig_bits']:.2f} bits")
    return record


if __name__ == "__main__":
    main(tyro.cli(Args))
