"""Reference games: every other model of each live chain scored on the
pre-registered experiment, for Figure 2's "other models live at the end" and
"pruned during the run" (``rsa_figure_data.py --chain-models``).

Descriptive only. The pre-registered experiment tested the three finals
against the bar; its data are spent (``docs/auto_rsa/CONSOLIDATION.md`` §5),
and these scores are exploratory. Every model scored here was written before
those data existed.

Each model is scored exactly as the finals and the bar were (``src.rsa.prereg``):
fitted once on the campaign's pooled data (the pre-registration's
``pool/pooled.csv``, whose sha256 the bar's predictions record) with the bar's
settings (``src.rsa.loop.fitting.loop_fit``), its posterior-mean choice-class
probabilities on the design pool frozen in ``pool_fit``'s format, and its
log predictive density summed over the pre-registered experiment's modelled
trials (``src.rsa.prereg.analysis.trial_lpd``). The score step first checks
that this reproduces ``prereg_analysis.json``'s lpd of the finals and the
bar from their frozen predictions.

The models, per chain (``data/rsa/live/live_rsa_c<k>/outputs``):

- other models live at the end: experiment 3's ``model_loop/export.json``
  ``live``, but its ``best_model`` (the final, already scored);
- removed models: every model its ledger records as ``admitted`` that is not
  live at the end and not one of the chain's starting models (``chains.json``,
  already scored as ``promoted:``/``seed:``): pruned (its last ledger entry),
  or retired between experiments, which the ledger does not record: then an
  experiment's ``carry.json`` must list it as ``over_cap`` or ``merged``.

About 220 models, so they are fitted in shards (``--mode fit --shard k
--shards n``, a Slurm array: ``scripts/paper/submit_rsa_chain_models.sh``),
one record per model in ``<work-dir>/records/<chain>/<name>.json`` (its
frozen probabilities, or why it has none: a fit failure of the model's own,
or no convergence after the loop's one refit). A shard skips models whose
record is current. ``--mode score`` then needs every record; it writes sums
only (``--out``), never trial rows.

``--model-set campaign`` scores, the same way, the incumbents of the promotion
campaign's existing-data loop instead (Sherlock run 2's three real-data
replicates, ``data/rsa/sherlock_run2/real_rep<k>``, whose models were promoted
to seed the live chains), for Figure 2's trajectories over the promoted seeds:
every model a replicate's ``history.json`` names as ``best_model`` at a step.
One whose source is byte for byte a model the pre-registered analysis scored
(the bar records each source's sha256) is not refitted: it takes that score
(``rsa_l2``, every replicate's start, and two of real_rep1's incumbents, which
were promoted unchanged). A model with the same name but other code would be
refitted. Its output (``data/rsa/prereg/campaign_incumbents_lpd.json``) lists
each incumbent's lpd and how it was scored, and each replicate's path: its
incumbent at every step. Equally exploratory and descriptive.
"""

from __future__ import annotations

import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import tyro
from pyprojroot import here

from scripts.paper import removed_models as rm
from scripts.paper import sr_figure_data as sr

LIVE_ROOT = here() / "data" / "rsa" / "live"
DEFAULT_CHAIN_SEEDS = here() / "data" / "rsa" / "live_seeds" / "chains" / "chains.json"
DEFAULT_PREREG = here() / "data" / "rsa" / "prereg" / "run" / "prereg_analysis.json"
DEFAULT_OUT = here() / "data" / "rsa" / "prereg" / "chain_models_lpd.json"
DEFAULT_WORK_DIR = here() / "data" / "paper" / "work" / "rsa_chain_models"
CHAINS = ["c0", "c1", "c2"]
DEFAULT_SWEEP = here() / "data" / "rsa" / "sherlock_run2"
CAMPAIGN_CELLS = ["real_rep1", "real_rep2", "real_rep3"]
DEFAULT_CAMPAIGN_OUT = here() / "data" / "rsa" / "prereg" / "campaign_incumbents_lpd.json"
SCORE_NOTE = (
    "Descriptive and exploratory: the pre-registered experiment's data are spent "
    "(docs/auto_rsa/CONSOLIDATION.md section 5). Each model fitted once on the campaign's "
    "pooled data with the bar's settings and scored like the finals and the bar "
    "(src.rsa.prereg.analysis.trial_lpd: summed log posterior-mean probability of each "
    "modelled trial's chosen class).")


@dataclass
class Args:
    mode: Literal["fit", "score"]
    model_set: Literal["chains", "campaign"] = "chains"
    """``chains``: the live chains' other models; ``campaign``: the existing-data loop's incumbents."""
    live_root: Path = LIVE_ROOT
    """``data/rsa/live``: the chains' outputs and the finals' and bar's frozen predictions."""
    chain_seeds: Path = DEFAULT_CHAIN_SEEDS
    prereg_analysis: Path = DEFAULT_PREREG
    work_dir: Path = DEFAULT_WORK_DIR
    pooled: Optional[Path] = None
    """``--mode fit``: the campaign's pooled trials (on Sherlock only)."""
    test: Optional[Path] = None
    """``--mode score``: the pre-registered experiment's modelled trials (on Sherlock only)."""
    out: Optional[Path] = None
    """The sums (default ``chain_models_lpd.json``, or ``campaign_incumbents_lpd.json`` for the campaign)."""
    chains: List[str] = field(default_factory=lambda: list(CHAINS))
    sweep: Path = DEFAULT_SWEEP
    """``--model-set campaign``: the existing-data sweep (its cells' ``history.json`` and ``models/``)."""
    cells: List[str] = field(default_factory=lambda: list(CAMPAIGN_CELLS))
    shard: int = 0
    shards: int = 1
    workers: int = 8
    """Models fitted at once (each in a child pinned to one core)."""


# --- the models -----------------------------------------------------------------


def _experiments(outputs: Path) -> List[Path]:
    found = sorted(outputs.glob("experiment*"), key=lambda p: int(re.fullmatch(r"experiment(\d+)", p.name).group(1)))
    if not found:
        raise FileNotFoundError(f"no experiment directories in {outputs}")
    return found


def chain_models(live_root: Path, chain: str, starting: List[str]) -> Dict[str, Dict[str, Any]]:
    """The chain's models to score: name -> ``file`` and ``role``
    (``live`` or ``removed``) and, for a removed one, ``outcome``."""
    from src.pipelines.inner_loop.hypothesis_ledger import HypothesisLedger

    outputs = live_root / f"live_rsa_{chain}" / "outputs"
    experiments = _experiments(outputs)
    final_loop = experiments[-1] / "model_loop"
    export = json.loads((final_loop / "export.json").read_text())
    live = list(export["live"])
    if export["best_model"] not in live:
        raise RuntimeError(f"{chain}: its final {export['best_model']!r} is not in its live set")
    carried_off: Dict[str, str] = {}
    for experiment in experiments:
        carry = experiment / "carry.json"
        if carry.exists():
            record = json.loads(carry.read_text())
            for kind in ("over_cap", "merged"):
                for item in record.get(kind, []):
                    carried_off[item["name"] if isinstance(item, dict) else item] = (
                        f"retired at {experiment.name}'s carry-over ({kind})")
    entries = HypothesisLedger(final_loop / "attempted_hypotheses.jsonl").entries()
    last = {e.name: e for e in entries}
    models: Dict[str, Dict[str, Any]] = {
        name: {"file": final_loop / "models" / f"{name}.py", "role": "live"}
        for name in live if name != export["best_model"]
    }
    for name in dict.fromkeys(e.name for e in entries if e.outcome == "admitted"):
        if name in live or name in starting:
            continue
        if last[name].outcome == "pruned":
            outcome = "pruned"
        elif name in carried_off:
            outcome = carried_off[name]
        else:
            raise RuntimeError(f"{chain}: {name!r} was admitted, is not live at the end and was "
                               f"neither pruned nor retired at a carry-over (last: {last[name].outcome})")
        models[name] = {"file": rm.source_file(outputs, name), "role": "removed", "outcome": outcome}
    return models


def campaign_paths(args: Args) -> Dict[str, List[Dict[str, Any]]]:
    """Each replicate's incumbent at every step of its ``history.json``."""
    return {cell: [{"step": st["step"], "round": st["round"], "best_model": st["best_model"]}
                   for st in json.loads((args.sweep / cell / "history.json").read_text())]
            for cell in args.cells}


def campaign_incumbents(args: Args) -> Dict[str, Dict[str, Any]]:
    """Each replicate's distinct incumbents, keyed ``<cell>:<name>``: its ``file`` and ``reuse``, the
    pre-registered analysis's models with this source byte for byte (the bar's ``model_sha256``, the
    file's sha256 truncated), whose score it takes; empty if none (then it is fitted)."""
    bar = json.loads((args.live_root / "bar_pool_preds.json").read_text())["models"]
    out: Dict[str, Dict[str, Any]] = {}
    for cell, path in campaign_paths(args).items():
        for name in dict.fromkeys(st["best_model"] for st in path):
            copies = [p for p in (args.sweep / cell / "models" / f"{name}.py",
                                  args.sweep / cell / "models" / "pruned" / f"{name}.py") if p.exists()]
            if len(copies) != 1:
                raise RuntimeError(f"{cell}: {len(copies)} source files of its incumbent {name!r}")
            sha = sr._sha256(copies[0])
            reuse = sorted(label for label, m in bar.items() if sha.startswith(m["model_sha256"]))
            out[f"{cell}:{name}"] = {"file": copies[0], "chain": cell, "name": name, "reuse": reuse}
    return out


def all_models(args: Args) -> Dict[str, Dict[str, Any]]:
    """The models to fit, keyed ``<chain>:<name>`` (names repeat across chains): every chain's, or
    (``--model-set campaign``) every campaign incumbent the pre-registered analysis did not score."""
    if args.model_set == "campaign":
        return {label: info for label, info in campaign_incumbents(args).items() if not info["reuse"]}
    seeds = json.loads(args.chain_seeds.read_text())["chains"]
    out: Dict[str, Dict[str, Any]] = {}
    for chain in args.chains:
        k = int(chain[1:])
        entries = [c for c in seeds if c["chain"] == k]
        if len(entries) != 1:
            raise RuntimeError(f"chains.json has {len(entries)} entries for chain {k}")
        for name, info in chain_models(args.live_root, chain, list(entries[0]["models"])).items():
            out[f"{chain}:{name}"] = {**info, "chain": chain, "name": name}
    return out


# --- fitting ---------------------------------------------------------------------


BAR_REFIT = "loop_fit: one refit at target_accept 0.95 if the first fails the convergence gate"


def fit_settings_record() -> Dict[str, Any]:
    """The settings as ``pool_fit`` records them. ``init`` is left out when it is ``uniform``, its
    default and the only start there was when the bar was fitted (as the loop's fit cache keys do,
    ``src.rsa.loop.fitting.fingerprint``)."""
    settings = vars(bar_settings()).copy()
    if settings["init"] == "uniform":
        del settings["init"]
    return dict(settings, refit=BAR_REFIT)


def bar_settings():
    """The bar's fit settings (``pool_fit``'s defaults, as ``bar_pool_preds.json`` records)."""
    from src.rsa.fit import FitSettings

    return FitSettings(num_warmup=1000, num_samples=1000, num_chains=4, seed=0, dense_mass=True)


def _inputs(info: Dict[str, Any], pooled_sha: str, pool_digest: str) -> Dict[str, Any]:
    return {"model_sha256": sr._sha256(info["file"]), "pooled_sha256": pooled_sha,
            "pool_digest": pool_digest, "fit_settings": fit_settings_record()}


def _record_path(work_dir: Path, label: str) -> Path:
    chain, name = label.split(":", 1)
    return work_dir / "records" / chain / f"{name}.json"


def check_bar_files(args: Args) -> Dict[str, Any]:
    """The bar's predictions file, checked to record this pooled data and these settings."""
    bar = json.loads((args.live_root / "bar_pool_preds.json").read_text())
    if bar["fit_settings"] != fit_settings_record():
        raise RuntimeError(f"the bar was fitted with {bar['fit_settings']}, this script fits with "
                           f"{fit_settings_record()}")
    return bar


def fit_shard(args: Args) -> None:
    from src.rsa.loop.fitting import NO_TIME_LIMIT, ModelFailure, describe, loop_fit
    from src.rsa.loop.novelty import plain_pool, pool_digest, posterior_mean_class_probs
    from src.rsa.model_file import RSAModel

    if args.pooled is None:
        raise ValueError("--mode fit needs --pooled")
    bar = check_bar_files(args)
    pooled_sha = hashlib.sha256(args.pooled.read_bytes()).hexdigest()
    if pooled_sha != bar["data"]["sha256"]:
        raise RuntimeError(f"{args.pooled} is not the pooled data the bar was fitted on "
                           f"(sha256 {pooled_sha}, the bar's {bar['data']['sha256']})")
    pool = plain_pool()
    digest = pool_digest(pool)
    models = all_models(args)
    mine = rm.shard_of(sorted(models), args.shard, args.shards)
    todo = [label for label in mine if sr.load_scored(
        _record_path(args.work_dir, label), _inputs(models[label], pooled_sha, digest)) is None]
    print(f"[rsa] shard {args.shard} of {args.shards}: {len(mine) - len(todo)} of {len(mine)} "
          "already scored", flush=True)
    cache = args.work_dir / "fit_cache"

    def run(label: str) -> None:
        info = models[label]
        try:
            fitted = loop_fit(info["file"], info["name"], args.pooled, bar_settings(), cache,
                              time_limit_sec=NO_TIME_LIMIT)
        except ModelFailure as exc:  # the model's own failure; anything else raises
            result: Dict[str, Any] = {"unscored": f"fit failed ({exc})"}
        else:
            if not fitted.converged:
                result = {"unscored": f"not converged ({'; '.join(fitted.convergence_problems)})"}
            else:
                probs = posterior_mean_class_probs(RSAModel(info["file"], name=info["name"]), fitted, pool)
                result = {"fit": describe(fitted), "probs": [round(float(p), 6) for p in probs]}
        sr.save_scored(_record_path(args.work_dir, label), _inputs(info, pooled_sha, digest), result)
        print(f"[rsa] {label}: {'scored' if 'probs' in result else result['unscored']}", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(run, todo))


# --- scoring ---------------------------------------------------------------------


def _noun(args: Args) -> str:
    return {"chains": "chain", "campaign": "campaign"}[args.model_set]


def score(args: Args) -> None:
    import pandas as pd

    from src.rsa.dataset import load_forced_choice
    from src.rsa.loop.novelty import plain_pool, pool_digest, pool_record
    from src.rsa.prereg.analysis import trial_lpd, versus
    from src.rsa.prereg.predictions import PoolPredictions, load_predictions

    if args.test is None:
        raise ValueError("--mode score needs --test")
    bar = check_bar_files(args)
    pool = plain_pool()
    digest = pool_digest(pool)
    models = all_models(args)
    records = {label: sr.load_scored(_record_path(args.work_dir, label),
                                     _inputs(info, bar["data"]["sha256"], digest))
               for label, info in models.items()}
    missing = sorted(label for label, r in records.items() if r is None)
    if missing:
        raise RuntimeError(f"{len(missing)} of {len(models)} {_noun(args)} models have no current record in "
                           f"{args.work_dir / 'records'} (fit them first with --mode fit): {missing}")

    trials = load_forced_choice(args.test)
    if (trials.frame["condition"].astype(str) == "catch").any():
        raise ValueError(f"{args.test} has catch trials: score the modelled trials (data/modelled.csv)")
    prereg = json.loads(args.prereg_analysis.read_text())
    if len(trials.choices) != prereg["n_trials"]:
        raise RuntimeError(f"{args.test} has {len(trials.choices)} trials, the pre-registered "
                           f"analysis {prereg['n_trials']}")

    # This scoring reproduces the pre-registered lpd of the finals and the bar.
    frozen = load_predictions([args.live_root / "finals_pool_preds_final.json",
                               args.live_root / "bar_pool_preds.json"])
    frozen_lp = dict(zip(frozen.names, trial_lpd(frozen, trials.contexts, trials.choices)))
    reproduced = {n: lp.sum() for n, lp in frozen_lp.items()}
    off = {n: (float(v), prereg["lpd"][n]) for n, v in reproduced.items()
           if abs(float(v) - prereg["lpd"][n]) > 1e-6}
    if off:
        raise RuntimeError(f"scoring does not reproduce the pre-registered lpd (got, recorded): {off}")

    # The chain models' records in pool_fit's format, loaded like the frozen files.
    scored = {label: r["record"] for label, r in records.items() if "probs" in r["record"]}
    combined = args.work_dir / f"{_noun(args)}_models_pool_preds.json"
    combined.write_text(json.dumps(dict(
        pool=dict(size=len(pool), digest=digest, n_entries=[len(c.objects) for c in pool],
                  choice_classes=[list(c.choice_classes()) for c in pool],
                  contexts=[pool_record(c) for c in pool]),
        models={label: dict(source=str(models[label]["file"]), fit=r["fit"], probs=r["probs"])
                for label, r in scored.items()},
    )))
    preds = load_predictions([combined])
    lpd: Dict[str, float] = {}
    unscored = {label: r["record"]["unscored"] for label, r in records.items()
                if "unscored" in r["record"]}
    for i, label in enumerate(preds.names):
        one = PoolPredictions(names=[label], probs=preds.probs[i:i + 1], valid=preds.valid,
                              pool=preds.pool, sha256=preds.sha256)
        try:
            lpd[label] = float(trial_lpd(one, trials.contexts, trials.choices).sum())
        except ValueError as exc:  # an observed choice the model gives probability 0
            unscored[label] = str(exc)

    best_promoted = prereg["best_promoted"]
    if args.model_set == "campaign":
        return write_campaign(args, lpd, unscored, prereg, bar, trials,
                              max(abs(float(v) - prereg["lpd"][n]) for n, v in reproduced.items()))
    # Every bar model and final against the best promoted seed, as the
    # pre-registered analysis compares (clustered by display): Figure 2's CIs.
    units = pd.factorize(trials.frame["condition"].astype(str).to_numpy())[0]
    bar_vs_best_promoted = {}
    for name, lp in frozen_lp.items():
        if name != best_promoted:
            row = versus(lp, frozen_lp[best_promoted], units)
            bar_vs_best_promoted[name] = {"diff": row["diff"], "se": row["se"]}
    chains: Dict[str, Any] = {}
    for chain in args.chains:
        mine = {label: info for label, info in models.items() if info["chain"] == chain}

        def by_role(role: str) -> Dict[str, float]:
            return {info["name"]: lpd[label] for label, info in mine.items()
                    if info["role"] == role and label in lpd}

        chains[chain] = {
            "others_at_end": by_role("live"),
            "pruned_during_run": by_role("removed"),
            "removed_outcome": {info["name"]: info["outcome"] for info in mine.values()
                                if info["role"] == "removed"},
            "unscored": {models[label]["name"]: reason for label, reason in unscored.items()
                         if models[label]["chain"] == chain},
        }
    out = args.out or DEFAULT_OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "note": SCORE_NOTE,
        "test": {"n_trials": len(trials.choices),
                 "sha256": hashlib.sha256(args.test.read_bytes()).hexdigest()},
        "pooled_sha256": bar["data"]["sha256"],
        "fit_settings": fit_settings_record(),
        "best_promoted": best_promoted,
        "best_promoted_lpd": prereg["lpd"][best_promoted],
        "bar_vs_best_promoted": bar_vs_best_promoted,
        "chains": chains,
    }, indent=1))
    print(f"wrote {out}: {len(lpd)} scored, {len(unscored)} unscored", flush=True)


def write_campaign(args: Args, lpd: Dict[str, float], unscored: Dict[str, str], prereg: Dict[str, Any],
                   bar: Dict[str, Any], trials: Any, reproduced_within: float) -> None:
    """The campaign's sums: each incumbent's lpd (fitted here, or the pre-registered analysis's for an
    identical source) and each replicate's path, its incumbent's lpd at every step."""
    incumbents = campaign_incumbents(args)
    models: Dict[str, Dict[str, Any]] = {}
    for label, info in incumbents.items():
        record: Dict[str, Any] = {"source": str(info["file"].relative_to(args.sweep)),
                                  "sha256": sr._sha256(info["file"])}
        if info["reuse"]:
            scores = {prereg["lpd"][r] for r in info["reuse"]}
            if len(scores) != 1:
                raise RuntimeError(f"{label}: its identical models {info['reuse']} score differently: {scores}")
            record.update(lpd=scores.pop(), scored=(
                f"not refitted: its source is byte for byte that of {', '.join(info['reuse'])} (the bar's "
                "model_sha256), so it takes that model's lpd in prereg_analysis.json"))
        elif label in lpd:
            record.update(lpd=lpd[label], scored="fitted here on the pooled data with the bar's settings")
        else:
            record.update(lpd=None, scored=f"unscored: {unscored[label]}")
        models[label] = record
    replicates = {cell: [{**st, "lpd": models[f"{cell}:{st['best_model']}"]["lpd"]} for st in path]
                  for cell, path in campaign_paths(args).items()}
    out = args.out or DEFAULT_CAMPAIGN_OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "note": SCORE_NOTE + (" The incumbents (each step's best_model) of the promotion campaign's "
                              "existing-data loop, whose models were promoted to seed the live chains."),
        "test": {"n_trials": len(trials.choices),
                 "sha256": hashlib.sha256(args.test.read_bytes()).hexdigest()},
        "pooled_sha256": bar["data"]["sha256"],
        "fit_settings": fit_settings_record(),
        "reproduction_check": (f"the scoring reproduces prereg_analysis.json's lpd of the finals and the "
                               f"bar from their frozen predictions within {reproduced_within:.2e} nats"),
        "best_promoted": prereg["best_promoted"],
        "best_promoted_lpd": prereg["lpd"][prereg["best_promoted"]],
        "sweep": args.sweep.name,
        "models": models,
        "replicates": replicates,
    }, indent=1))
    print(f"wrote {out}: {sum(m['lpd'] is not None for m in models.values())} of {len(models)} scored",
          flush=True)


def main(args: Args) -> None:
    if args.mode == "fit":
        fit_shard(args)
    else:
        score(args)


if __name__ == "__main__":
    from src.rsa.cpus import pin_main_thread

    pin_main_thread()  # before JAX starts its thread pools (src.rsa.cpus)
    main(tyro.cli(Args))
