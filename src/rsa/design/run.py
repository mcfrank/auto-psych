"""Design a live experiment from the current models, and say how much power it has.

    uv run python -m src.rsa.design.run --models-dir data/rsa/live_seeds/models \
        --data <the trials the models are fitted to> --cache <fit cache> --out <design dir>

1. Each model is fitted to ``--data`` (the loop's fit rule and settings; a
   cached fit is reused, so after `src.rsa.promote` nothing is refitted).
2. Its posterior draws give per-draw choice-class probabilities on the design
   pool: every game up to 4 objects x 4 features, redundant features included,
   every word and the prior query (`src.rsa.loop.novelty.plain_pool`; the
   displays the experiment page can show). A model undefined on some pool display is left out of the
   design, on record (``screened_out``), as main's design does.
3. A participant answers ``trials_per_participant`` designed displays (plus
   catch trials), a balanced subset of the design's D (`trial_lists` with
   ``n_trials``), so each display gets ``participants x trials / D``
   responses. For each D in ``displays``, `src.rsa.design.eig.select` picks D
   displays at that many responses each. Fewer displays means more responses
   per display; more displays, more kinds of display: the power table says
   which tells the models apart better.
4. `src.rsa.design.eig.power` scores each design at each of
   ``power_participants`` on fresh scenarios: how often the model that
   generated the data ends with the highest posterior.
5. With ``quotas`` (minimum shares of kinds of display; PI 2026-10-10), EIG
   chooses within them (`src.rsa.design.eig.Quota`), and the design without
   them is selected too, in a second process, and recorded beside it
   (``free``: its picks, kinds and power), so what the quotas cost is on
   record for every design.

Writes ``<out>/design_d<D>.json`` per D (a `src.rsa.experiment.design.Design`:
build the page with ``python -m src.rsa.experiment.build --design ... --n-trials
<trials_per_participant>``) and ``<out>/eig.json`` (per D the picks, their joint
EIG and the power table; the models; the settings).
"""

from __future__ import annotations

import hashlib
import json
import math
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from multiprocessing import get_context
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import tyro

from src.models.model_manifest import read_manifest_names
from src.rsa.context import Context, group_by_shape
from src.rsa.design.distinct import SAME_ON_POOL_RMSE, rmse
from src.rsa.cpus import CORES, pinned_thread
from src.rsa.design.eig import Quota, composition, power, select
from src.rsa.experiment.design import Design, TrialSpec
from src.rsa.fit import FitSettings, class_probs, draw_batches, posterior_flat
from src.rsa.loop.fitting import FIT_TIME_LIMIT_SEC, loop_fit
from src.rsa.loop.novelty import plain_pool, pool_digest, pool_record
from src.rsa.model_file import RSAModel

WIDTH = 4  # the most objects a pool display has


def design_pool() -> List[Context]:
    return plain_pool()


# The kinds of display a quota can ask for: "objects=<2|3|4>" and "query=<word|prior>".
KIND_DIMENSIONS = {
    "objects": lambda ctx: str(len(ctx.objects)),
    "query": lambda ctx: "prior" if ctx.utterance is None else "word",
}


def parse_quotas(specs: Sequence[str], pool: Sequence[Context], n_displays: int) -> List[Quota]:
    """``"<dimension>=<kind>:<share>"`` -> a `Quota` of at least round(share x D)
    displays of that kind (half up), e.g. "objects=2:0.15" is 6 of 40."""
    quotas = []
    for spec in specs:
        try:
            kind, share = spec.rsplit(":", 1)
            dim, value = kind.split("=", 1)
            share = float(share)
        except ValueError:
            raise ValueError(f"quota {spec!r} is not <dimension>=<kind>:<share>") from None
        if dim not in KIND_DIMENSIONS:
            raise ValueError(f"quota {spec!r}: no dimension {dim!r} (one of {sorted(KIND_DIMENSIONS)})")
        if not 0 < share <= 1:
            raise ValueError(f"quota {spec!r}: the share must be in (0, 1]")
        members = np.array([KIND_DIMENSIONS[dim](ctx) == value for ctx in pool])
        if not members.any():
            raise ValueError(f"quota {spec!r}: no pool display is {kind}")
        quotas.append(Quota(name=kind, dimension=dim, members=members,
                            minimum=int(math.floor(share * n_displays + 0.5))))
    return quotas


def kinds(pool: Sequence[Context], picks: Sequence[int]) -> Dict[str, int]:
    """The picks by size and query, e.g. {"4x4 word": 12, "2x3 prior": 1}."""
    out: Dict[str, int] = {}
    for i in picks:
        ctx = pool[i]
        key = f"{len(ctx.objects)}x{len(ctx.feature_names)} {KIND_DIMENSIONS['query'](ctx)}"
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items()))


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
    bar_models_dirs: List[Path] = field(default_factory=list)
    """Further models the design must also tell apart: the outer loop's bar
    (the starting models and every promoted seed), so a design tests the claim
    it is scored on (PI 2026-10-10: the rehearsal's displays aimed only at the
    carried models, and the bar models were never exposed). Each is labelled
    ``bar:<name>``; one that is the same file as, or the same hypothesis on
    this pool (`src.rsa.design.distinct`) as, a model already in is merged."""
    carried_share: float = 0.5
    withhold: List[Path] = field(default_factory=list)
    """Model files left out of the bar, unnamed (a simulated run's ground truth:
    the design must not be handed it, and its record is readable by agents)."""
    """With bar models, the prior mass on ``models_dir``'s models (spread
    evenly); the bar models share the rest."""
    trials_per_participant: int = 10
    """Designed displays a participant answers (12 test trials with 2 catch trials; PI 2026-10-08)."""
    participants: int = 200
    """Participants per experiment the designs are chosen for."""
    displays: List[int] = field(default_factory=lambda: [10, 20, 30])
    """Design sizes D to compare (each at least trials_per_participant)."""
    power_participants: List[int] = field(default_factory=lambda: [100, 200, 300])
    quotas: List[str] = field(default_factory=list)
    """Minimum shares of kinds of display, ``<dimension>=<kind>:<share>`` with
    dimensions ``objects`` (2, 3, 4) and ``query`` (word, prior), e.g.
    ``objects=2:0.15 query=prior:0.2`` (`parse_quotas`). Empty: free EIG."""
    n_draws: int = 200
    n_scenarios: int = 2000
    n_power_scenarios: int = 4000
    seed: int = 0
    num_warmup: int = 1000
    num_samples: int = 1000
    num_chains: int = 4
    fit_seed: int = 0
    dense_mass: bool = False
    """NUTS with a dense mass matrix (`src.rsa.fit.FitSettings.dense_mass`)."""
    time_limit_sec: float = FIT_TIME_LIMIT_SEC


def main(args: Args) -> dict:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    settings = FitSettings(num_warmup=args.num_warmup, num_samples=args.num_samples, num_chains=args.num_chains,
                           seed=args.fit_seed, dense_mass=args.dense_mass)
    pool = design_pool()
    valid = class_layout(pool)
    names, probs, screened, merged, groups = [], [], [], [], []
    shas = {}
    entries = [(Path(args.models_dir), name, name, "carried") for name in read_manifest_names(args.models_dir)]
    for folder in args.bar_models_dirs:
        entries += [(Path(folder), name, f"bar:{name}", "bar") for name in read_manifest_names(folder)]
    withheld = {hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in args.withhold}
    n_withheld = 0
    for folder, name, label, group in entries:
        path = folder / f"{name}.py"
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        if group == "bar" and sha in withheld:
            n_withheld += 1
            continue
        if label in names or sha in shas:
            merged.append(dict(model=label, same_as=shas.get(sha, label), reason="the same model file"))
            continue
        fitted = loop_fit(path, name, args.data, settings, args.cache, time_limit_sec=args.time_limit_sec)
        p = class_prob_draws(RSAModel(path, name=name), posterior_flat(fitted), pool, args.n_draws)
        bad = ~np.isfinite(p).all(axis=(0, 2)) | ~np.isclose(p.sum(-1), 1.0, atol=1e-4).all(0)
        if bad.any():
            screened.append(dict(model=label, reason="choice probabilities undefined on some pool displays",
                                 n_displays=int(bad.sum()), examples=[pool_record(pool[i]) for i in np.where(bad)[0][:3]]))
            print(f"  {label}: screened out ({int(bad.sum())} pool displays undefined)", flush=True)
            continue
        if group == "bar" and names:
            dist = {n: rmse(q.mean(0), p.mean(0)) for n, q in zip(names, probs)}
            twin = min(dist, key=dist.get)
            if dist[twin] < SAME_ON_POOL_RMSE:
                merged.append(dict(model=label, same_as=twin, rmse=dist[twin],
                                   reason=f"the same hypothesis on the design pool (RMSE < {SAME_ON_POOL_RMSE})"))
                continue
        names.append(label)
        probs.append(p)
        groups.append(group)
        shas[sha] = label
        print(f"  {label}: {p.shape[0]} draws on {len(pool)} displays", flush=True)
    if len(names) < 2:
        raise ValueError(f"{len(names)} usable models: a design needs at least two to tell apart")

    n_carried = groups.count("carried")
    if n_carried < len(groups):
        share = args.carried_share if n_carried else 0.0
        prior = np.array([share / n_carried if g == "carried" else (1 - share) / (len(groups) - n_carried)
                          for g in groups])
    else:
        prior = None
    t = args.trials_per_participant
    designs = []
    for d in args.displays:
        if d < t:
            raise ValueError(f"a design of {d} displays cannot fill {t} trials per participant")
        per_display = responses_per_display(args.participants, t, d)
        quotas = parse_quotas(args.quotas, pool, d)
        ns = [responses_per_display(n, t, d) for n in args.power_participants]
        selection = dict(n_responses=per_display, prior=prior, n_scenarios=args.n_scenarios, seed=args.seed)
        scoring = dict(prior=prior, n_scenarios=args.n_power_scenarios, seed=args.seed + 101, names=names)
        free = None
        if quotas:
            # The free design in a second process, on a core of its own, while this one selects.
            with ProcessPoolExecutor(1, mp_context=get_context("spawn")) as ex:
                core = CORES.take()
                try:
                    with pinned_thread(core):  # the worker starts here and keeps this core
                        pending = ex.submit(_free_design, probs, valid, d, selection, ns, scoring)
                    sel = select(probs, valid, d, quotas=quotas, **selection)
                    free_sel, free_table = pending.result()
                finally:
                    CORES.give_back(core)
            for row, n in zip(free_table, args.power_participants):
                row["participants"] = n
            free = dict(
                n_eig_picks=sum(s == "eig" for s in free_sel.sources), kinds=kinds(pool, free_sel.indices),
                quota_counts=composition(quotas, free_sel.indices),
                picks=[dict(display=pool_record(pool[i]), joint_eig_bits=b, source=s)
                       for i, b, s in zip(free_sel.indices, free_sel.joint_eig_bits, free_sel.sources)],
                power=free_table,
            )
        else:
            sel = select(probs, valid, d, **selection)
        table = power(probs, valid, sel.indices, ns, **scoring)
        for row, n in zip(table, args.power_participants):
            row["participants"] = n
        specs = tuple(TrialSpec.from_context(pool[i], label=f"eig_{k:02d}") for k, i in enumerate(sel.indices))
        design = Design(name=f"eig_{len(names)}models_d{d}_n{args.participants}", specs=specs)
        (out / f"design_d{d}.json").write_text(json.dumps(design.to_json(), indent=1))
        designs.append(dict(
            displays=d, responses_per_display=per_display, design_sha256=design.sha256,
            n_eig_picks=sum(s == "eig" for s in sel.sources),
            kinds=kinds(pool, sel.indices),
            quotas=[dict(kind=q.name, minimum=q.minimum, count=c)
                    for q, c in zip(quotas, composition(quotas, sel.indices).values())],
            picks=[dict(display=pool_record(pool[i]), joint_eig_bits=b, source=s)
                   for i, b, s in zip(sel.indices, sel.joint_eig_bits, sel.sources)],
            power=table,
            free=free,
        ))
        print(f"D={d} ({per_display} responses per display at N={args.participants}): "
              f"{designs[-1]['n_eig_picks']} EIG picks, joint EIG {sel.joint_eig_bits[-1]:.2f} of "
              f"{np.log2(len(names)):.2f} bits; {designs[-1]['kinds']}", flush=True)
        for label, rows in [("", table)] + ([("free: ", free["power"])] if free else []):
            for row in rows:
                print(f"  {label}N={row['participants']}: P(generating model wins) {row['p_correct']:.3f} ± "
                      f"{row['p_correct_se']:.3f}, joint EIG {row['joint_eig_bits']:.2f} bits", flush=True)
    record = dict(
        models=names, model_groups=groups, prior=None if prior is None else prior.tolist(), merged=merged,
        n_withheld=n_withheld,
        screened_out=screened, trials_per_participant=t, participants=args.participants,
        quotas=list(args.quotas), pool_size=len(pool), pool_digest=pool_digest(pool), n_draws=args.n_draws, n_scenarios=args.n_scenarios,
        seed=args.seed, fit_settings=vars(settings),
        data_sha256=hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
        designs=designs,
    )
    (out / "eig.json").write_text(json.dumps(record, indent=1))
    return record


def _free_design(probs, valid, n_select, selection, ns, scoring):
    """The design without quotas and its power (`main`'s comparison, run in a worker)."""
    sel = select(probs, valid, n_select, **selection)
    return sel, power(probs, valid, sel.indices, ns, **scoring)


def responses_per_display(participants: int, trials: int, displays: int) -> int:
    """Each display's responses when every participant answers ``trials`` of
    ``displays`` in balanced subsets (rounded down: the conservative count)."""
    return max(1, participants * trials // displays)


if __name__ == "__main__":
    from src.rsa.cpus import pin_main_thread

    pin_main_thread()  # one core per process (src.rsa.cpus)
    main(tyro.cli(Args))
