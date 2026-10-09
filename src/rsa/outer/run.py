"""The RSA outer loop: design, collect, score prospectively, run the inner loop; repeat.

    uv run python -m src.rsa.outer.run --run-dir <out>/run1 --seeds data/rsa/live_seeds/models \
        --existing-data <all existing trials> --collection simulated --ground-truth <model.py>

One run is ``n_experiments`` experiments (the live campaign: three independent
runs of three; PLAN.md, "Decisions (2026-10-08, the live campaign)"). The
stages pass state through files, as main's outer loop does, and each is
skipped when its outputs exist, so an interrupted run resumes where it stopped
(the inner loop resumes from its last scored step).

For experiment N (``<run-dir>/experiment<N>/``):

1. ``models_input/``: the live set going in: the promoted seeds for
   experiment 1, the previous experiment's live set after.
2. ``data/prior.csv``: every trial so far (the existing data and experiments
   1..N-1).
3. ``design/``: the models fitted to ``prior.csv`` (cached), ``D`` displays
   picked by joint EIG for ``participants`` people answering ``trials`` each
   (`src.rsa.design.run`), and the trial lists (balanced subsets, one list per
   participant).
4. ``data/responses.csv``: this experiment's rows. ``simulated``: people
   answering from a ground-truth model through the page's records and
   `convert` (`src.rsa.outer.simulate`); ``live``: the deployed page's data
   (not wired yet: it raises). Participants who miss more than
   ``max_catch_errors`` catch trials are excluded (``data/participants.json``).
   Participant ids continue across the run's experiments.
5. ``prospective.json`` (in the private directory): claim 2's measure. Each model going in, fitted only to
   ``prior.csv``, scores this experiment's new data before any model is
   refitted. The bar is the best of the five starting models fitted to the
   same data (``seed:<name>``; PI 2026-10-09: beat all of them, not only
   rsa_l2); from experiment 2 on also the promoted seeds the live phase began
   with (``promoted:<name>``): beating them is the live loop's own progress.
   Held-out lpd, differences and SEs clustered by designed display.
6. ``model_loop/``: the inner loop on ``data/cumulative.csv`` (prior +
   this experiment), starting from ``models_input/``: ``max_iterations``
   rounds of ``candidate_count`` agents, the two-stale-round stop, grouped CV.
   Its live set is the next experiment's ``models_input``.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Literal, Optional

import numpy as np
import pandas as pd
import tyro
import yaml

from src.models.clustered_se import cluster_dse
from src.models.model_manifest import read_manifest_entries, read_manifest_names
from src.rsa.dataset import load_forced_choice
from src.rsa.design import run as design_run
from src.rsa.evaluate_heldout import heldout_lpd
from src.rsa.experiment.design import Design, trial_lists
from src.rsa.fit import FitSettings
from src.rsa.loop.fitting import FIT_TIME_LIMIT_SEC, loop_fit
from src.rsa.loop.orchestrator import LoopConfig, RSALoop, SpawnFn
from src.rsa.model_file import RSAModel
from src.rsa.outer.simulate import simulate_participants
from src.runtime.config import PROJECT_ASSETS_DIR

STARTING_MODELS = PROJECT_ASSETS_DIR / "rsa_reference" / "seed_models"
LIVE_SOURCE = "auto_psych"  # src.rsa.experiment.convert's source label


@dataclass
class OuterConfig:
    run_dir: Path
    seeds: Path
    """The promoted seed set (data/rsa/live_seeds/models)."""
    existing_data: Path
    """Every existing trial (train + test; promote's all_trials.csv)."""
    private_dir: Optional[Path] = None
    """Where everything that names the ground truth goes (the run's configuration,
    the recovery records, the fit cache): outside the agents' tree on the
    cluster. Default: <run_dir>/.private (tests)."""
    collection: Literal["simulated", "live"] = "simulated"
    ground_truth: Optional[Path] = None
    """simulated: the model file people answer from (fitted to the existing data)."""
    n_experiments: int = 3
    participants: int = 200
    trials: int = 10
    """Designed displays per participant (plus n_catch catch trials: 12 test trials; PI 2026-10-08)."""
    displays: int = 20
    n_catch: int = 2
    max_catch_errors: int = 0
    max_iterations: int = 5
    candidate_count: int = 6
    stop_after_stale_rounds: int = 2
    cv_folds: int = 5
    selection_scope: Literal["all", "live"] = "all"
    """What the inner loop selects on: grouped CV over all data so far, or over
    the live trials only (every fit still uses all data). Open (PI 2026-10-09):
    the simulated rehearsal compares the two."""
    num_warmup: int = 1000
    num_samples: int = 1000
    num_chains: int = 4
    n_draws: int = 200
    n_scenarios: int = 2000
    seed: int = 0
    fit_time_limit_sec: Optional[float] = FIT_TIME_LIMIT_SEC
    starting_models: Path = STARTING_MODELS
    """The five starting models every claim is measured against (literal, rsa_l1, rsa_l2, salience, shared prior)."""


def _copy_models(src_dir: Path, names: List[str], dest: Path) -> None:
    tmp = dest.with_name(dest.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    entries = {e["name"]: e for e in read_manifest_entries(src_dir)} if (src_dir / "models_manifest.yaml").exists() else {}
    for n in names:
        shutil.copyfile(src_dir / f"{n}.py", tmp / f"{n}.py")
    (tmp / "models_manifest.yaml").write_text(
        yaml.safe_dump({"models": [entries.get(n, {"name": n}) for n in names]}, sort_keys=False))
    tmp.rename(dest)


class OuterRun:
    def __init__(self, cfg: OuterConfig, spawner: Callable[[int, Path, Path], SpawnFn]):
        """``spawner(n, models_dir, responses_path)`` gives experiment n's agent spawner."""
        self.cfg = cfg
        self.dir = Path(cfg.run_dir)
        self.spawner = spawner
        self.settings = FitSettings(num_warmup=cfg.num_warmup, num_samples=cfg.num_samples,
                                    num_chains=cfg.num_chains, seed=cfg.seed)
        # The agents can read the run directory: nothing in it may name the
        # ground truth, so its fit (and every outer fit) is cached privately.
        self.private = Path(cfg.private_dir) if cfg.private_dir else self.dir / ".private"
        self.cache = self.private / ".fit_cache"
        if cfg.collection == "simulated" and cfg.ground_truth is None:
            raise ValueError("simulated collection needs --ground-truth")

    def exp(self, n: int) -> Path:
        return self.dir / f"experiment{n}"

    def label(self, n: int) -> str:
        return f"{self.dir.name}_e{n}"

    # ---------- stages ----------
    def models_input(self, n: int) -> Path:
        dest = self.exp(n) / "models_input"
        if not dest.exists():
            if n == 1:
                _copy_models(Path(self.cfg.seeds), read_manifest_names(self.cfg.seeds), dest)
            else:
                prev = self.exp(n - 1) / "model_loop"
                export = json.loads((prev / "export.json").read_text())
                _copy_models(prev / "models", export["live"], dest)
        return dest

    def prior_data(self, n: int) -> Path:
        path = self.exp(n) / "data" / "prior.csv"
        if not path.exists():
            existing = pd.read_csv(self.cfg.existing_data)
            if "source" not in existing.columns or existing["source"].isna().any():
                raise ValueError(f"{self.cfg.existing_data} needs a source on every row (live rows are 'auto_psych')")
            frames = [existing] + [pd.read_csv(self.exp(k) / "data" / "responses.csv") for k in range(1, n)]
            _write_csv(pd.concat(frames, ignore_index=True), path)
        return path

    def design(self, n: int) -> Path:
        out = self.exp(n) / "design"
        lists_path = out / "trial_lists.json"
        if lists_path.exists():
            return lists_path
        c = self.cfg
        design_run.main(design_run.Args(
            models_dir=self.models_input(n), data=self.prior_data(n), cache=self.cache, out=out,
            trials_per_participant=c.trials, participants=c.participants, displays=[c.displays],
            power_participants=[c.participants], n_draws=c.n_draws, n_scenarios=c.n_scenarios,
            n_power_scenarios=c.n_scenarios, seed=c.seed + n, num_warmup=c.num_warmup,
            num_samples=c.num_samples, num_chains=c.num_chains, fit_seed=c.seed,
            time_limit_sec=c.fit_time_limit_sec or FIT_TIME_LIMIT_SEC))
        design = Design.load(out / f"design_d{c.displays}.json")
        shutil.copyfile(out / f"design_d{c.displays}.json", out / "design.json")
        doc = trial_lists(design, seed=c.seed * 1000 + n, n_lists=c.participants, n_catch=c.n_catch, n_trials=c.trials)
        _write_json(doc, lists_path)
        return lists_path

    def collect(self, n: int) -> Path:
        path = self.exp(n) / "data" / "responses.csv"
        if path.exists():
            return path
        doc = json.loads(self.design(n).read_text())
        if self.cfg.collection == "live":
            raise NotImplementedError("live collection: deploying the page and collecting its data is not wired yet")
        gt = Path(self.cfg.ground_truth)
        fitted = loop_fit(gt, gt.stem, self.cfg.existing_data, self.settings, self.cache,
                          time_limit_sec=self.cfg.fit_time_limit_sec)
        first = sum(json.loads((self.exp(k) / "data" / "participants.json").read_text())["n_recruited"]
                    for k in range(1, n))
        rows = simulate_participants(doc, gt, fitted, self.cfg.participants, experiment=self.label(n),
                                     seed=self.cfg.seed * 1000 + n, first_id=first)
        kept, record = exclude_on_catch(rows, self.cfg.max_catch_errors)
        _write_json(dict(record, n_recruited=self.cfg.participants, collection=self.cfg.collection,
                         ), self.exp(n) / "data" / "participants.json")
        _write_csv(kept, path)
        return path

    def prospective(self, n: int) -> Path:
        """Claim 2: the models going in, fitted to the data before this
        experiment, scored on its new data before anything is refitted."""
        # Private: in a simulated run the best starting model is often the ground truth.
        out = self.private / f"experiment{n}" / "prospective.json"
        if out.exists():
            return out
        new = load_forced_choice(self.collect(n))
        units = pd.factorize(new.frame["condition"].astype(str))[0]
        groups = {"live": self.models_input(n), "seed": Path(self.cfg.starting_models)}
        if n > 1:
            groups["promoted"] = Path(self.cfg.seeds)
        lpd: Dict[str, np.ndarray] = {}
        for group, folder in groups.items():
            for name in read_manifest_names(folder):
                path = folder / f"{name}.py"
                fitted = loop_fit(path, name, self.prior_data(n), self.settings, self.cache,
                                  time_limit_sec=self.cfg.fit_time_limit_sec)
                label = name if group == "live" else f"{group}:{name}"
                lpd[label] = heldout_lpd(RSAModel(path, name=name), fitted, new.contexts, new.choices)

        def best_of(prefix):
            names = [k for k in lpd if (k.startswith(prefix) if prefix else ":" not in k)]
            return max(names, key=lambda k: lpd[k].sum()) if names else None

        bars = {"best_seed": best_of("seed:"), "best_promoted": best_of("promoted:")}
        live_best = best_of("")

        def versus(k, ref):
            return dict(diff=float(lpd[k].sum() - lpd[ref].sum()),
                        se=0.0 if k == ref else float(cluster_dse(lpd[k], lpd[ref], units)))

        _write_json(dict(
            experiment=n, n_trials=len(new.contexts), n_displays=int(units.max() + 1),
            best_live=live_best, **bars,
            live_vs={bar: versus(live_best, ref) for bar, ref in bars.items() if ref},
            models={k: dict(lpd=float(v.sum()), **{f"vs_{bar}": versus(k, ref) for bar, ref in bars.items() if ref})
                    for k, v in sorted(lpd.items(), key=lambda kv: -kv[1].sum())},
        ), out)
        return out

    def model_loop(self, n: int) -> Path:
        out = self.exp(n) / "model_loop"
        if (out / "export.json").exists():
            return out
        cumulative = self.exp(n) / "data" / "cumulative.csv"
        if not cumulative.exists():
            frames = [pd.read_csv(self.prior_data(n)), pd.read_csv(self.collect(n))]
            _write_csv(pd.concat(frames, ignore_index=True), cumulative)
        c = self.cfg
        cfg = LoopConfig(
            responses_path=cumulative, seed_models_dir=self.models_input(n), results_dir=out,
            max_iterations=c.max_iterations, candidate_count=c.candidate_count, settings=self.settings,
            fit_time_limit_sec=c.fit_time_limit_sec, stop_after_stale_rounds=c.stop_after_stale_rounds,
            cv_folds=c.cv_folds, selection_source=LIVE_SOURCE if c.selection_scope == "live" else None,
            report_title=f"RSA inner loop · {self.label(n)}",
        )
        RSALoop(cfg, self.spawner(n, out / "models", cumulative)).run(resume=(out / "history.json").exists())
        return out

    def recovery(self, n: int) -> Optional[Path]:
        """Simulated runs: how far each model is from the hidden ground truth
        on the design pool (RMSE of choice-class probabilities), going into
        experiment n and after its inner loop."""
        if self.cfg.collection != "simulated":
            return None
        out = self.private / f"experiment{n}" / "recovery.json"
        if out.exists():
            return out
        from src.rsa.loop.novelty import posterior_mean_class_probs

        pool = design_run.design_pool()
        gt = Path(self.cfg.ground_truth)
        gt_fit = loop_fit(gt, gt.stem, self.cfg.existing_data, self.settings, self.cache,
                          time_limit_sec=self.cfg.fit_time_limit_sec)
        truth = posterior_mean_class_probs(RSAModel(gt, name=gt.stem), gt_fit, pool)

        def distances(folder: Path, data: Path, cache: Path) -> Dict[str, float]:
            d = {}
            for name in read_manifest_names(folder):
                path = folder / f"{name}.py"
                fitted = loop_fit(path, name, data, self.settings, cache, time_limit_sec=self.cfg.fit_time_limit_sec)
                p = posterior_mean_class_probs(RSAModel(path, name=name), fitted, pool)
                d[name] = float(np.sqrt(np.mean((p - truth) ** 2)))
            return dict(sorted(d.items(), key=lambda kv: kv[1]))

        loop = self.model_loop(n)
        export = json.loads((loop / "export.json").read_text())
        before = distances(self.models_input(n), self.prior_data(n), self.cache)
        after = distances(loop / "models", self.exp(n) / "data" / "cumulative.csv", loop / ".fit_cache")
        _write_json(dict(
            experiment=n, ground_truth=gt.stem, selection_scope=self.cfg.selection_scope,
            exported=export["best_model"], exported_rmse=after[export["best_model"]],
            closest_after=next(iter(after)), closest_after_rmse=next(iter(after.values())),
            closest_before=next(iter(before)), closest_before_rmse=next(iter(before.values())),
            before=before, after=after,
        ), out)
        return out

    def run(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        record = self.private / "outer_config.json"
        cfg = {k: str(v) if isinstance(v, Path) else v for k, v in asdict(self.cfg).items()}
        if record.exists() and json.loads(record.read_text()) != cfg:
            raise ValueError(f"{record} records another configuration; a run is resumed with its own")
        _write_json(cfg, record)
        for n in range(1, self.cfg.n_experiments + 1):
            print(f"[outer] {self.label(n)}: design", flush=True)
            self.design(n)
            print(f"[outer] {self.label(n)}: collect ({self.cfg.collection})", flush=True)
            self.collect(n)
            print(f"[outer] {self.label(n)}: prospective score", flush=True)
            self.prospective(n)
            print(f"[outer] {self.label(n)}: inner loop", flush=True)
            self.model_loop(n)
            self.recovery(n)


def exclude_on_catch(rows: pd.DataFrame, max_errors: int) -> tuple[pd.DataFrame, dict]:
    """Drop the participants who missed more than ``max_errors`` catch trials."""
    cov = rows["covariates"].map(json.loads)
    missed = cov.map(lambda c: bool(c.get("is_catch")) and not c.get("catch_correct"))
    errors = missed.groupby(rows["participant_id"]).sum()
    out = sorted(errors[errors > max_errors].index)
    kept = rows[~rows["participant_id"].isin(out)].reset_index(drop=True)
    return kept, dict(n_included=int(kept["participant_id"].nunique()), excluded=out, max_catch_errors=max_errors)


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.csv")
    frame.to_csv(tmp, index=False, lineterminator="\n")
    tmp.replace(path)


def _write_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.json")
    tmp.write_text(json.dumps(obj, indent=1))
    tmp.replace(path)


@dataclass
class Args(OuterConfig):
    agent_model: Optional[str] = None
    agent_timeout_sec: int = 2400
    agent_root: Optional[Path] = None
    no_sandbox: bool = False
    coding_agent: Optional[str] = None


def main(args: Args) -> None:
    from src.rsa.loop.orchestrator import coding_agent_spawner
    from src.rsa.loop.run import DEFAULT_AGENT_MODEL

    def spawner(n: int, models_dir: Path, responses: Path) -> SpawnFn:
        return coding_agent_spawner(
            models_dir=models_dir, responses_path=responses, timeout_sec=args.agent_timeout_sec,
            backend=args.coding_agent, model=args.agent_model or DEFAULT_AGENT_MODEL, agent_root=args.agent_root,
            sandbox=not args.no_sandbox, network=False, shell_dir=models_dir.parent / ".agent_shell")

    cfg = OuterConfig(**{f: getattr(args, f) for f in OuterConfig.__dataclass_fields__})
    OuterRun(cfg, spawner).run()


if __name__ == "__main__":
    main(tyro.cli(Args))
