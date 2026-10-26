"""The designer's EIG tool (docs/auto_open/PLAN.md §4): how much a paradigm would tell the live
theories apart, per dollar, and whether that clears the floor.

    uv run python -m src.rsa_open.design --paradigm paradigm.json --context <design context> --out report.json

**The design context** (`write_context`, `load_context`) is written by the harness and only read
by the designer: the live theories (``theories/<name>.py``), each theory's posterior draws on the
data so far (``draws/<name>.npz``, one array per parameter), and a uniform prior over the
theories (``context.json``). A context whose draws do not match its theories' parameters is
refused.

**Trial units** (`trial_units`) are the smallest things a design picks, for the paradigm's link:
a scene (choice); a scene and a probed object (rating); a scene and a target (production); a
scene, a target and a real message (binary judgment); a scene and a probed object (slider:
the ordered-beta link, its responses binned into 12 outcomes, so the score is a lower bound). A probe or target is taken once per class
of identical objects, since probing either gives the same predictions. A speaker's trial shows
no heard message, so its units are taken from the first scene of each game only. Bets (the allocation
link) are not categorical, and the tool refuses them (`NotImplementedError`) until it has an
estimator for them.

**Scoring** (`score_paradigm`): a new paradigm's link has no data yet, so its parameters are
drawn from the link's prior: ``LINK_SETTINGS`` settings, each paired with the same posterior
draws of every theory (``n_draws`` in all per theory). The outcome probabilities go to the
closed campaign's estimator (`src.rsa.design.eig`) with the draws grouped by link setting: a
scenario (a theory, one of its draws and simulated responses) compares the theories at its own
link setting. So the score is the information about the theories **given** the response
process, averaged over plausible response processes. Averaging over the link's prior inside
each theory's evidence instead (what the estimator does without groups) is a draw lottery:
with diffuse nuisance parameters and ~1,000 responses, a theory's evidence is whichever single
draw happens to fit, and a test on a rating paradigm picked the generating theory 72% of the
time at 32 draws, 91% at 256 and 100% with the link fixed, while reporting near-certainty
throughout. Links with more nuisance parameters would be scored worse, arbitrarily. The cost of
conditioning: a theory difference the link can mimic (a sharper listener vs a higher
temperature) counts as information here, though the joint fit may not separate it.

The design is a greedy joint-EIG selection of the paradigm's ``distinct_trials`` units, each
answered ``n_participants x trials_per_participant / distinct_trials`` times; its joint EIG and
power are then computed on fresh scenarios. The report gives the design, its bits, bits per
dollar, and the floor: ``FLOOR_SHARE`` of the bits per dollar of the closed campaign's
paradigm (`REFERENCE`) under the same theories (PI 2026-10-21). The reference is computed once
per context and settings and cached in the context when it is writable.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import tyro

from src.rsa.context import Context, group_by_shape
from src.rsa.design import eig
from src.rsa_open.fit import TheoryModel
from src.rsa_open.links import LinkLikelihood, likelihood

FLOOR_SHARE = 0.5
LINK_SETTINGS = 8
MAX_UNITS = 20_000
DEFAULT_DRAWS = 64
DEFAULT_SCENARIOS = 1000
# Lazy greedy (`src.rsa.design.eig.select`): a full pass every 16 picks, as main's design.
LAZY_FULL_EVERY = 16
# The closed campaign's paradigm at the per-experiment budget: forced choice over the 1,597-display
# plain pool, 13 of 52 designed displays each, for 400 people (twice the closed campaign's 200 at
# its $0.50, as the budget is twice its: 2026-10-26).
REFERENCE = {"sizes": ((2, 2), (2, 3), (2, 4), (3, 2), (3, 3), (3, 4), (4, 2), (4, 3), (4, 4)),
             "prior_queries": True, "synonyms": True, "link": "choice",
             "n_participants": 400, "trials_per_participant": 13, "distinct_trials": 52}


class ContextInvalid(ValueError):
    """A design context that cannot be used."""


@dataclass
class DesignContext:
    root: Path
    theories: Dict[str, TheoryModel]
    draws: Dict[str, Dict[str, np.ndarray]]
    prior: np.ndarray
    sha256: str

    @property
    def names(self) -> List[str]:
        return list(self.theories)


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_context(root: Path, theories: Mapping[str, Tuple[Path, Mapping[str, np.ndarray]]]) -> Path:
    """Write a design context: each theory's file and its posterior draws (param -> (draws,))."""
    root = Path(root)
    (root / "theories").mkdir(parents=True, exist_ok=False)
    (root / "draws").mkdir()
    record = {"theories": {}}
    for name, (path, draws) in theories.items():
        shutil.copyfile(path, root / "theories" / f"{name}.py")
        np.savez(root / "draws" / f"{name}.npz", **{k: np.asarray(v) for k, v in draws.items()})
        record["theories"][name] = {"theory_sha256": _sha(root / "theories" / f"{name}.py"),
                                    "draws_sha256": _sha(root / "draws" / f"{name}.npz")}
    (root / "context.json").write_text(json.dumps(record, indent=2))
    return root


def load_context(root: Path) -> DesignContext:
    root = Path(root)
    record_path = root / "context.json"
    if not record_path.exists():
        raise ContextInvalid(f"{root}: no context.json")
    record = json.loads(record_path.read_text())
    theories, draws, problems = {}, {}, []
    for name, shas in record["theories"].items():
        tpath, dpath = root / "theories" / f"{name}.py", root / "draws" / f"{name}.npz"
        if not tpath.exists() or not dpath.exists():
            problems.append(f"{name}: missing {tpath.name if not tpath.exists() else dpath.name}")
            continue
        if _sha(tpath) != shas["theory_sha256"] or _sha(dpath) != shas["draws_sha256"]:
            problems.append(f"{name}: its files changed since the context was written")
        theory = TheoryModel(tpath, name=name)
        with np.load(dpath) as d:
            arrays = {k: np.asarray(d[k]) for k in d.files}
        missing, extra = sorted(set(theory.params) - set(arrays)), sorted(set(arrays) - set(theory.params))
        if missing or extra:
            problems.append(f"{name}: draws lack {missing} / have extra {extra} (the theory's parameters are "
                            f"{list(theory.params)})")
            continue
        lengths = {len(v) for v in arrays.values()}
        if not theory.params:
            pass  # no parameters: nothing to draw
        elif len(lengths) != 1 or next(iter(lengths)) < 2:
            problems.append(f"{name}: every parameter needs the same number (>= 2) of draws; got {lengths}")
        elif not all(np.all(np.isfinite(v)) for v in arrays.values()):
            problems.append(f"{name}: draws are not all finite")
        theories[name], draws[name] = theory, arrays
    if problems:
        raise ContextInvalid(f"{root}:\n" + "\n".join(f"- {p}" for p in problems))
    if not theories:
        raise ContextInvalid(f"{root}: no theories")
    return DesignContext(root, theories, draws, np.full(len(theories), 1.0 / len(theories)), _sha(record_path))


# --------------------------------------------------------------------------- trial units

@dataclass(frozen=True)
class Unit:
    scene: int
    fields: Tuple[Tuple[str, int], ...]  # probe / target / utterance

    def as_dict(self) -> Dict[str, int]:
        return dict(self.fields)


def _scene_fields(ctx: Context, link: LinkLikelihood) -> List[Tuple[Tuple[str, int], ...]]:
    """The design fields of one scene's units under a single link."""
    reps = sorted(set(ctx.choice_classes()))
    n_real = ctx.shape[1] - (1 if ctx.needs_sink else 0)
    if link.name in ("choice", "allocation"):  # a bet's whole split is the response: one unit per scene
        return [()]
    if link.name in ("rating", "ordered_beta"):
        return [(("probe", r),) for r in reps]
    if link.name == "production":
        return [(("target", r),) for r in reps]
    return [(("target", r), ("utterance", u)) for r in reps for u in range(n_real)]


def trial_units(scenes: Sequence[Context], link: LinkLikelihood) -> List[Unit]:
    if link.name == "pair":  # every pairing of the two measures' units of a scene
        from src.rsa_open.links import SECOND_FIELDS

        for part in (link.first, link.second):
            if part.name not in LinkLikelihood.CATEGORICAL:
                raise NotImplementedError(f"the design tool cannot score {part.name!r} measures yet")
        units: List[Unit] = []
        games = set()
        for i, ctx in enumerate(scenes):
            if link.side == "speaker":
                if ctx.objects in games:
                    continue
                games.add(ctx.objects)
            for f1 in _scene_fields(ctx, link.first):
                for f2 in _scene_fields(ctx, link.second):
                    units.append(Unit(i, f1 + tuple((SECOND_FIELDS[k], v) for k, v in f2)))
        if len(units) > MAX_UNITS:
            raise ValueError(f"{len(units)} trial units, over the {MAX_UNITS} the EIG tool scores; narrow the "
                             f"scene_space (fewer or smaller sizes)")
        return units
    if link.name not in LinkLikelihood.CATEGORICAL:
        raise NotImplementedError(
            f"the design tool cannot score {link.name!r} ({link.response_type}) paradigms yet: its "
            f"responses are not categorical. Use choice, rating, production or binary_judgment.")
    units: List[Unit] = []
    games = set()
    for i, ctx in enumerate(scenes):
        if link.side == "speaker":  # a speaker's trial has no heard message: one unit set per game
            if ctx.objects in games:
                continue
            games.add(ctx.objects)
        units += [Unit(i, f) for f in _scene_fields(ctx, link)]
    if len(units) > MAX_UNITS:
        raise ValueError(f"{len(units)} trial units, over the {MAX_UNITS} the EIG tool scores; narrow the "
                         f"scene_space (fewer or smaller sizes)")
    return units


def describe_unit(scenes: Sequence[Context], unit: Unit) -> Dict[str, Any]:
    ctx = scenes[unit.scene]
    return {"objects": [list(o) for o in ctx.objects], "heard": ctx.utterance, **unit.as_dict()}


# --------------------------------------------------------------------------- outcome probabilities

def outcome_prob_draws(theory: TheoryModel, link: LinkLikelihood, theory_draws: Mapping[str, np.ndarray],
                       link_draws: Mapping[str, np.ndarray], scenes: Sequence[Context],
                       units: Sequence[Unit]) -> Tuple[np.ndarray, np.ndarray]:
    """(draws, units, W) outcome probabilities and (units, W) valid outcome slots."""
    n_draws = len(next(iter(link_draws.values())))
    if theory_draws and len(next(iter(theory_draws.values()))) != n_draws:
        raise ValueError("theory and link draws must pair one to one")
    groups = group_by_shape(scenes)
    position = {}
    for group in groups.values():
        for pos, i in enumerate(group.indices):
            position[int(i)] = (group.shape, pos)
    by_group: Dict[Tuple[int, int], List[int]] = {}
    for u_idx, unit in enumerate(units):
        by_group.setdefault(position[unit.scene][0], []).append(u_idx)
    width = max(max(groups[s].shape) for s in by_group)  # outcome slots never exceed objects, messages, or points
    if link.name == "rating":
        width = max(width, link.options["scale_points"])
    if link.name == "ordered_beta":
        from src.rsa_open.links import slider_bin_edges

        width = max(width, len(slider_bin_edges(link.options["total_points"])) + 1)
    if link.name == "allocation":
        from src.rsa_open.links import bet_compositions, design_bet_total

        width = max(len(bet_compositions(n, design_bet_total(n, link.options["total_points"])))
                    for n in {s[0] for s in by_group})
    if link.name == "pair":  # the product of the parts' slots, per scene shape
        width = 0
        for shape, unit_ids in by_group.items():
            group = groups[shape]
            pos = np.asarray([position[units[u].scene][1] for u in unit_ids])
            arrays = jax.tree_util.tree_map(lambda a: a[pos], group.arrays)
            keys = [k for k, _ in units[unit_ids[0]].fields]
            enc = {k: jnp.asarray([units[u].as_dict()[k] for u in unit_ids]) for k in keys}
            width = max(width, int(link.outcome_valid(arrays, jnp.asarray(group.classes[pos]), enc).shape[1]))
    width = max(width, 2)
    probs = np.zeros((n_draws, len(units), width), dtype=np.float64)
    valid = np.zeros((len(units), width), dtype=bool)
    tp = {k: jnp.asarray(v) for k, v in theory_draws.items()}
    lp = {k: jnp.asarray(v) for k, v in link_draws.items()}
    for shape, unit_ids in by_group.items():
        group = groups[shape]
        latent = theory.latent_draws(tp, group, link.side == "speaker", n_draws)
        pos = np.asarray([position[units[u].scene][1] for u in unit_ids])
        arrays = jax.tree_util.tree_map(lambda a: a[pos], group.arrays)
        classes = jnp.asarray(group.classes[pos])
        keys = [k for k, _ in units[unit_ids[0]].fields]
        enc = {k: jnp.asarray([units[u].as_dict()[k] for u in unit_ids]) for k in keys}
        per_draw = jax.vmap(lambda lat, p: link.outcome_probs(lat, p, enc, arrays, classes))
        p = np.asarray(per_draw(latent[:, pos], lp), dtype=np.float64)
        v = np.asarray(link.outcome_valid(arrays, classes, enc))
        probs[:, unit_ids, :p.shape[2]] = p
        valid[unit_ids, :v.shape[1]] = v
    probs = np.where(valid[None], probs, 0.0)
    probs /= probs.sum(-1, keepdims=True)
    return probs, valid


def _link_draws(link: LinkLikelihood, n: int, seed: int) -> Dict[str, np.ndarray]:
    keys = jax.random.split(jax.random.PRNGKey(seed), len(link.params))
    return {k: np.asarray(d.sample(key, (n,))) for (k, d), key in zip(link.params.items(), keys)}


def _theory_draws(draws: Mapping[str, np.ndarray], n: int) -> Dict[str, np.ndarray]:
    if not draws:  # a theory without parameters
        return {}
    total = len(next(iter(draws.values())))
    take = np.unique(np.linspace(0, total - 1, min(n, total)).round().astype(int))
    return {k: np.asarray(v)[take] for k, v in draws.items()}


def _score(ctx: DesignContext, scenes: Sequence[Context], link: LinkLikelihood, *, n_participants: int,
           trials_per_participant: int, distinct_trials: int, cost_usd: float, n_draws: int, n_scenarios: int,
           seed: int, lazy_full_every: Optional[int] = LAZY_FULL_EVERY) -> Dict[str, Any]:
    units = trial_units(scenes, link)
    if distinct_trials > len(units):
        raise ValueError(f"the design picks {distinct_trials} trial units but the scene space gives {len(units)}")
    responses = n_participants * trials_per_participant // distinct_trials
    per_setting = max(2, n_draws // LINK_SETTINGS)
    settings = _link_draws(link, LINK_SETTINGS, seed)  # one link setting per group, shared by every theory
    groups = np.repeat(np.arange(LINK_SETTINGS), per_setting)
    ld = {k: np.repeat(v, per_setting) for k, v in settings.items()}
    probs, valid = [], None
    for name in ctx.names:
        td = {k: np.tile(v, LINK_SETTINGS) for k, v in _theory_draws(ctx.draws[name], per_setting).items()}
        if td and len(next(iter(td.values()))) != len(groups):
            raise ContextInvalid(f"{name}: fewer than {per_setting} posterior draws")
        p, valid = outcome_prob_draws(ctx.theories[name], link, td, ld, scenes, units)
        probs.append(p)
    draw_groups = [groups] * len(probs)
    selection = eig.select(probs, valid, distinct_trials, n_responses=responses, prior=ctx.prior,
                           n_scenarios=n_scenarios, seed=seed, draw_groups=draw_groups,
                           lazy_full_every=lazy_full_every)
    (row,) = eig.power(probs, valid, selection.indices, [responses], prior=ctx.prior,
                       n_scenarios=n_scenarios, seed=seed + 1, names=ctx.names, draw_groups=draw_groups)
    return {
        "link": link.name, "n_units": len(units), "design_size": distinct_trials, "responses_per_unit": responses,
        "design": [{**describe_unit(scenes, units[i]), "source": src}
                   for i, src in zip(selection.indices, selection.sources)],
        "joint_eig_bits": row["joint_eig_bits"], "max_bits": row["max_bits"], "p_correct": row["p_correct"],
        "by_theory": row["by_model"], "cost_usd": cost_usd, "bits_per_dollar": row["joint_eig_bits"] / cost_usd,
    }


def reference_report(ctx: DesignContext, *, n_draws: int, n_scenarios: int, seed: int) -> Dict[str, Any]:
    """The closed campaign's paradigm scored under the context's theories (cached in the context)."""
    from src.rsa.design_space import context_pool
    from src.rsa_open.paradigm import BUDGET_USD

    key = f"reference_{ctx.sha256[:12]}_{n_draws}_{n_scenarios}_{seed}_lazy{LAZY_FULL_EVERY}.json"
    cached = ctx.root / key
    if cached.exists():
        return json.loads(cached.read_text())
    scenes = context_pool(REFERENCE["sizes"], include_prior_queries=REFERENCE["prior_queries"],
                          synonyms=REFERENCE["synonyms"])
    report = _score(ctx, scenes, likelihood(REFERENCE["link"]), n_participants=REFERENCE["n_participants"],
                    trials_per_participant=REFERENCE["trials_per_participant"],
                    distinct_trials=REFERENCE["distinct_trials"], cost_usd=BUDGET_USD, n_draws=n_draws,
                    n_scenarios=n_scenarios, seed=seed)
    report.pop("design")
    try:
        cached.write_text(json.dumps(report))
    except OSError:
        pass  # a read-only context (the designer's sandbox): the reference is recomputed, not lost
    return report


def score_paradigm(paradigm, ctx: DesignContext, *, n_draws: int = DEFAULT_DRAWS,
                   n_scenarios: int = DEFAULT_SCENARIOS, seed: int = 0, reference: bool = True,
                   reference_seed: int = 0) -> Dict[str, Any]:
    """The paradigm's design under the context's theories, its bits per dollar and the floor.

    The reference has a seed of its own (``reference_seed``), so every paradigm scored in one
    context meets the same floor, computed once (`reference_report`)."""
    from src.rsa_open.paradigm import paradigm_likelihood, scene_pool

    d = paradigm.design
    report = {"paradigm_sha256": paradigm.sha256, "context_sha256": ctx.sha256, "theories": ctx.names,
              **_score(ctx, scene_pool(paradigm), paradigm_likelihood(paradigm),
                       n_participants=d.n_participants, trials_per_participant=d.trials_per_participant,
                       distinct_trials=d.distinct_trials, cost_usd=paradigm.cost_usd, n_draws=n_draws,
                       n_scenarios=n_scenarios, seed=seed),
              "settings": {"n_draws": n_draws, "n_scenarios": n_scenarios, "seed": seed}}
    if reference:
        ref = reference_report(ctx, n_draws=n_draws, n_scenarios=n_scenarios, seed=reference_seed)
        report["reference"] = ref
        report["floor_bits_per_dollar"] = FLOOR_SHARE * ref["bits_per_dollar"]
        report["meets_floor"] = bool(report["bits_per_dollar"] >= report["floor_bits_per_dollar"])
    return report


@dataclass
class Args:
    paradigm: Path
    context: Path
    out: Path
    n_draws: int = DEFAULT_DRAWS
    n_scenarios: int = DEFAULT_SCENARIOS
    seed: int = 0


def main(args: Args) -> Dict[str, Any]:
    from src.rsa_open.paradigm import load_paradigm

    report = score_paradigm(load_paradigm(args.paradigm), load_context(args.context), n_draws=args.n_draws,
                            n_scenarios=args.n_scenarios, seed=args.seed)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"{args.paradigm}: {report['joint_eig_bits']:.3f} of {report['max_bits']:.3f} bits, "
          f"${report['cost_usd']:.2f}, {report['bits_per_dollar']:.4f} bits/$ "
          f"(floor {report['floor_bits_per_dollar']:.4f}: {'meets' if report['meets_floor'] else 'BELOW'})")
    return report


if __name__ == "__main__":
    main(tyro.cli(Args))
