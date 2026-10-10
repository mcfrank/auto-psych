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
   rsa_l2); from experiment 2 on also every promoted seed (``promoted:<name>``,
   ``promoted``: all of them even when this chain began with a share, PI
   2026-10-09; from experiment 1 on when it did): beating them is the live
   loop's own progress.
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
from src.rsa.dataset import load_forced_choice, write_plain_trials
from src.rsa.design import run as design_run
from src.rsa.design.distinct import SAME_ON_POOL_RMSE, keep_distinct
from src.rsa.evaluate_heldout import heldout_lpd
from src.rsa.experiment.design import Design, trial_lists
from src.rsa.fit import FitSettings
from src.rsa.loop.fitting import FIT_TIME_LIMIT_SEC, NO_TIME_LIMIT, loop_fit
from src.rsa.loop.brief import DEFAULT_RSA_LENSES, PLAIN_RSA_LENSES, PLAIN_SCOPE_NOTE
from src.rsa.loop.novelty import posterior_mean_class_probs
from src.rsa.loop.orchestrator import LoopConfig, RSALoop, SpawnFn
from src.rsa.model_file import RSAModel
from src.rsa.outer.simulate import simulate_participants
from src.runtime.token_usage import read_usage_log, start_usage_log, summarize, write_usage_report
from src.runtime.config import PROJECT_ASSETS_DIR

STARTING_MODELS = PROJECT_ASSETS_DIR / "rsa_reference" / "seed_models"
LIVE_SOURCE = "auto_psych"  # src.rsa.experiment.convert's source label
# Every design's minimum shares of kinds of display (PI 2026-10-10): of 40, at
# least 6 two-object and 10 three-object displays, and 8-20 mumble trials.
DESIGN_QUOTAS = ("objects=2:0.15", "objects=3:0.25", "query=prior:0.2", "query=word:0.5")


@dataclass
class OuterConfig:
    run_dir: Path
    seeds: Path
    """The seeds this run starts from: the promoted set (data/rsa/live_seeds/models)
    or one chain's share of it (data/rsa/live_seeds/chains/chain_<k>)."""
    existing_data: Path
    """Every existing trial (train + test; promote's all_trials.csv)."""
    private_dir: Optional[Path] = None
    """Where everything that names the ground truth goes (the run's configuration,
    the recovery records, the fit cache): outside the agents' tree on the
    cluster. Default: <run_dir>/.private (tests)."""
    promoted: Optional[Path] = None
    """Every promoted seed, the prospective bar (default: ``seeds``)."""
    collection: Literal["simulated", "live"] = "simulated"
    prolific_mode: Literal["test", "live"] = "test"
    """Live collection: "test" deploys the page and makes a Prolific draft to
    preview (nothing is collected); "live" publishes the study (real money)."""
    confirm_live_recruitment: bool = False
    """Required with prolific_mode "live" (main's double gate; the launcher asks for a typed yes)."""
    firebase_project: Optional[str] = None
    run_label: str = ""
    """This run's label: its pages are served at <site>/e<N>-<label>/."""
    collection_owner: str = "auto-psych"
    repo_root: Optional[Path] = None
    """The checkout that stages and deploys the page (default: this code's)."""
    max_wait_sec: float = 3 * 60 * 60
    stop_after_collect: bool = False
    """End the run once the first experiment's data are in (a pilot: the page,
    the timing, the catch-trial rate, the data path), before any modelling."""
    """How long a live experiment waits for its participants before pausing the study (main's 3 h)."""
    ground_truth: Optional[Path] = None
    """simulated: the model file people answer from (fitted to the existing data)."""
    n_experiments: int = 3
    participants: int = 200  # per experiment: 50 responses for each of 40 displays (PI 2026-10-09)
    trials: int = 10
    """Designed displays per participant (plus n_catch catch trials: 12 test trials; PI 2026-10-08)."""
    displays: int = 40
    n_catch: int = 2
    max_catch_errors: int = 0
    max_iterations: int = 5
    candidate_count: int = 6
    stop_after_stale_rounds: int = 2
    cv_folds: int = 5
    selection_scope: Literal["all", "live", "guarded"] = "guarded"
    """What the inner loop selects on (every fit uses all data): grouped CV over
    all data so far ("all"), over the live trials only ("live"), or over the
    live trials among the models within ``selection_guard_dse`` clustered SEs
    of the best on the existing data ("guarded"; PI 2026-10-10, after the
    rehearsal: "all" never moved off the literature, "live" collapsed onto one
    family of near-copies that predicted worse than plain RSA)."""
    selection_guard_dse: float = 4.0
    scope: Literal["plain", "all"] = "plain"
    """The live phase's scope (PI 2026-10-10): "plain" leaves the existing trials
    on valence, familiarization and greyscale displays out of every fit
    (pragmods E5, E6, E7 and the colour-prior rerun; the live displays vary
    none of them), compares models for novelty on plain displays only, drops
    the framing lens and tells the agents. "all": everything, as run 2."""
    max_carried: int = 8
    """Models carried into the next experiment, one per distinct hypothesis (`carry`)."""
    carried_share: float = 0.5
    """The design's prior mass on the models going in; the bar models (starting
    models and promoted seeds, so the displays test claim 2) share the rest."""
    quotas: List[str] = field(default_factory=lambda: list(DESIGN_QUOTAS))
    """Minimum shares of kinds of display in every design, EIG choosing within
    them (`src.rsa.design.run.parse_quotas`; PI 2026-10-10). The default: of
    40 displays at least 6 with two objects, 10 with three, 8 mumble trials and
    20 with a word. Each design also records the free design's power beside it."""
    num_warmup: int = 1000
    num_samples: int = 1000
    num_chains: int = 4
    dense_mass: bool = True
    """NUTS with a dense mass matrix (`src.rsa.fit.FitSettings.dense_mass`).
    Measured 2026-10-09 on 28k existing trials, 4 chains of 1000 + 1000: the
    same posterior means, higher ESS, and 2.5-9x fewer leapfrog steps on the
    promoted models with 5-7 parameters (the slowest fit, 494 s -> 149 s);
    no change on 2-parameter rsa_l2."""
    n_draws: int = 200
    n_scenarios: int = 2000
    seed: int = 0
    fit_time_limit_sec: Optional[float] = FIT_TIME_LIMIT_SEC
    starting_models: Path = STARTING_MODELS
    """The five starting models every claim is measured against (literal, rsa_l1, rsa_l2, salience, shared prior)."""


def mean_kl(p, q, n_displays: int) -> float:
    """Mean KL(p || q) per display, in nats. The class probabilities come
    flattened over the displays (`posterior_mean_class_probs`), so the total is
    divided by the display count (rehearsal 2's interim report, 2026-10-10,
    found the recovery record holding the sum)."""
    p, q = np.asarray(p, float), np.clip(np.asarray(q, float), 1e-12, None)
    total = np.sum(np.where(p > 0, p * (np.log(np.clip(p, 1e-12, None)) - np.log(q)), 0.0))
    return float(total / n_displays)


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
        # The outer loop fits only models already in play (the models going in,
        # the bar models, the ground truth): never time-limited (fitting.NO_TIME_LIMIT).
        self.ref_limit = None if cfg.fit_time_limit_sec is None else NO_TIME_LIMIT
        self.settings = FitSettings(num_warmup=cfg.num_warmup, num_samples=cfg.num_samples,
                                    num_chains=cfg.num_chains, seed=cfg.seed, dense_mass=cfg.dense_mass)
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
                _copy_models(prev / "models", self.carry(n, prev, export["live"]), dest)
        return dest

    def carry(self, n: int, prev: Path, live: List[str]) -> List[str]:
        """The previous experiment's live set, one model per hypothesis the live
        displays can tell apart (`src.rsa.design.distinct`), best first by the
        loop's final standing (trusted, then its selection criterion), at most
        ``max_carried``. Recorded in ``experiment<n>/carry.json``. The
        2026-10-09 rehearsal carried 12 near-copies into experiment 3."""
        out = self.exp(n) / "carry.json"
        if out.exists():
            return json.loads(out.read_text())["kept"]
        standing = json.loads((prev / "history.json").read_text())[-1]["standing"]
        order = sorted(live, key=lambda m: (not standing[m]["trusted"], -standing[m]["sel_elpd"], m))
        pool = design_run.design_pool()
        preds = {}
        for name in order:
            path = prev / "models" / f"{name}.py"
            fitted = loop_fit(path, name, prev / "responses.csv", self.settings, prev / ".fit_cache",
                              time_limit_sec=self.ref_limit)
            preds[name] = posterior_mean_class_probs(RSAModel(path, name=name), fitted, pool)
        chosen = keep_distinct(order, preds, cap=self.cfg.max_carried)
        _write_json(dict(rule=f"best first by the loop's final standing; a model within {SAME_ON_POOL_RMSE} "
                              f"(RMSE of posterior-mean choice-class probabilities over the design pool) of one "
                              f"kept is the same hypothesis for the live phase; at most {self.cfg.max_carried}",
                         order=order, **chosen), out)
        return chosen["kept"]

    def existing(self) -> Path:
        """The existing data the run fits: all of it, or (scope "plain") its
        plain-display trials, with what was left out in ``existing_scope.json``."""
        if self.cfg.scope == "all":
            return Path(self.cfg.existing_data)
        out = self.private / "existing_plain.csv"
        if not out.exists():
            record = write_plain_trials(Path(self.cfg.existing_data), out)
            _write_json(dict(record, scope="plain", source=str(self.cfg.existing_data)),
                        self.dir / "existing_scope.json")
        return out

    def prior_data(self, n: int) -> Path:
        path = self.exp(n) / "data" / "prior.csv"
        if not path.exists():
            existing = pd.read_csv(self.existing())
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
            bar_models_dirs=[Path(c.starting_models), Path(c.promoted or c.seeds)], carried_share=c.carried_share,
            quotas=list(c.quotas),
            withhold=[Path(c.ground_truth)] if c.ground_truth else [],
            trials_per_participant=c.trials, participants=c.participants, displays=[c.displays],
            power_participants=[c.participants], n_draws=c.n_draws, n_scenarios=c.n_scenarios,
            n_power_scenarios=c.n_scenarios, seed=c.seed + n, num_warmup=c.num_warmup,
            num_samples=c.num_samples, num_chains=c.num_chains, fit_seed=c.seed, dense_mass=c.dense_mass,
            time_limit_sec=self.ref_limit if self.ref_limit is not None else FIT_TIME_LIMIT_SEC))
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
            rows, extra = self.collect_live(n, doc)
            n_recruited = extra["n_responses"]
        else:
            gt = Path(self.cfg.ground_truth)
            fitted = loop_fit(gt, gt.stem, self.existing(), self.settings, self.cache,
                              time_limit_sec=self.ref_limit)
            first = sum(json.loads((self.exp(k) / "data" / "participants.json").read_text())["n_recruited"]
                        for k in range(1, n))
            rows = simulate_participants(doc, gt, fitted, self.cfg.participants, experiment=self.label(n),
                                         seed=self.cfg.seed * 1000 + n, first_id=first)
            n_recruited, extra = self.cfg.participants, {}
        kept, record = exclude_on_catch(rows, self.cfg.max_catch_errors)
        _write_json(dict(record, n_recruited=n_recruited, n_target=self.cfg.participants,
                         collection=self.cfg.collection, **extra), self.exp(n) / "data" / "participants.json")
        _write_csv(kept, path)
        return path

    def collect_live(self, n: int, doc: dict) -> tuple[pd.DataFrame, dict]:
        """Deploy experiment n's page, recruit on Prolific, fetch and convert its data.

        Double-gated like main's live runs: ``prolific_mode="live"`` publishes a
        paid study, so it also needs ``confirm_live_recruitment``. The raw
        responses (with Prolific ids) and the map from Prolific ids to the run's
        participant ids are written to the private directory only."""
        from src.pipelines.outer_loop.orchestrator import require_outside_agent_trees
        from src.rsa.live import collect as live
        from src.runtime.config import REPO_ROOT

        c = self.cfg
        if c.prolific_mode == "live" and not c.confirm_live_recruitment:
            raise RuntimeError("prolific_mode 'live' publishes a paid Prolific study: it needs "
                               "--confirm-live-recruitment as well")
        if not c.run_label:
            raise ValueError("live collection needs --run-label (this run's Hosting path and study label)")
        settings = live.LiveSettings(prolific_mode=c.prolific_mode, firebase_project=c.firebase_project,
                                     run_label=c.run_label, collection_owner=c.collection_owner,
                                     repo_root=Path(c.repo_root) if c.repo_root else REPO_ROOT,
                                     max_wait_sec=c.max_wait_sec)
        manifest = live.deploy(self.exp(n), doc, c.participants, n, settings)
        live.wait_for_participants(manifest, c.participants, self.exp(n), settings.max_wait_sec)
        responses = live.fetch_responses(manifest)
        raw = self.private / "raw_collected" / f"experiment{n}.json"
        require_outside_agent_trees(raw, "the raw live responses (Prolific ids)")
        if self.dir.resolve() in raw.resolve().parents:
            raise RuntimeError(f"{raw} is inside the run directory the agents read")
        _write_json(responses, raw)
        id_map = self.private / "participant_ids.json"
        ids = json.loads(id_map.read_text()) if id_map.exists() else {}
        rows, ids, record = live.responses_to_rows(responses, study_id=manifest["prolific_study_id"],
                                                   experiment=self.label(n), ids=ids)
        _write_json(ids, id_map)
        return rows, record

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
        promoted = Path(self.cfg.promoted or self.cfg.seeds)
        if n > 1 or promoted.resolve() != Path(self.cfg.seeds).resolve():
            groups["promoted"] = promoted
        lpd: Dict[str, np.ndarray] = {}
        for group, folder in groups.items():
            for name in read_manifest_names(folder):
                path = folder / f"{name}.py"
                fitted = loop_fit(path, name, self.prior_data(n), self.settings, self.cache,
                                  time_limit_sec=self.ref_limit)
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

        # Claim 2's test (PI 2026-10-10): the model the loop committed to before
        # these data existed, the previous inner loop's export, against the bar.
        # The bar's best is picked after the fact, so the comparison is
        # conservative. Experiment 1 has no committed model: the loop has not
        # chosen one yet (its models going in are the chain's seeds, part of the bar).
        committed = None
        if n > 1:
            committed = json.loads((self.exp(n - 1) / "model_loop" / "export.json").read_text())["best_model"]
            if committed not in lpd:
                raise RuntimeError(f"the committed model {committed} is not among the models going into "
                                   f"experiment {n}: carry-forward must keep the export")
        bar_models = [k for k in lpd if k.startswith(("seed:", "promoted:"))]
        _write_json(dict(
            experiment=n, n_trials=len(new.contexts), n_displays=int(units.max() + 1),
            committed=committed, **bars,
            committed_vs=None if committed is None else dict(
                {bar: versus(committed, ref) for bar, ref in bars.items() if ref},
                each_bar_model={k: versus(committed, k) for k in bar_models}),
            best_live=live_best,
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
        plain = c.scope == "plain"
        prev_ledger = self.exp(n - 1) / "model_loop" / "attempted_hypotheses.jsonl" if n > 1 else None
        cfg = LoopConfig(
            responses_path=cumulative, seed_models_dir=self.models_input(n), results_dir=out,
            max_iterations=c.max_iterations, candidate_count=c.candidate_count, settings=self.settings,
            fit_time_limit_sec=c.fit_time_limit_sec, stop_after_stale_rounds=c.stop_after_stale_rounds,
            cv_folds=c.cv_folds, selection_source=None if c.selection_scope == "all" else LIVE_SOURCE,
            selection_guard_dse=c.selection_guard_dse if c.selection_scope == "guarded" else 0.0,
            report_title=f"RSA inner loop · {self.label(n)}",
            novelty_pool="plain" if plain else "full", brief_note=PLAIN_SCOPE_NOTE if plain else "",
            lenses=list(PLAIN_RSA_LENSES if plain else DEFAULT_RSA_LENSES), inherit_ledger=prev_ledger,
        )
        # The agents' spend, per experiment (appended across resumes; the summary
        # covers the whole log), written even when the loop fails. The rehearsal
        # of 2026-10-09 recorded none.
        usage_log = out / "token_usage.jsonl"
        marker = start_usage_log(usage_log)
        try:
            RSALoop(cfg, self.spawner(n, out / "models", cumulative)).run(resume=(out / "history.json").exists())
        finally:
            write_usage_report(out, marker, heading=f"RSA inner loop, {self.label(n)}",
                               records=read_usage_log(usage_log) if usage_log.exists() else [])
            self.write_usage_summary()
        return out

    def write_usage_summary(self) -> dict:
        """The run's agent spend, per experiment and in total: ``<run>/token_usage_summary.json``."""
        per, every = {}, []
        for n in range(1, self.cfg.n_experiments + 1):
            log = self.exp(n) / "model_loop" / "token_usage.jsonl"
            if log.exists():
                records = read_usage_log(log)
                per[f"experiment{n}"] = summarize(records)
                every += records
        summary = dict(total=summarize(every), experiments=per)
        _write_json(summary, self.dir / "token_usage_summary.json")
        return summary

    def recovery(self, n: int) -> Optional[Path]:
        """Simulated runs: how far each model is from the hidden ground truth,
        going into experiment n and after its inner loop. The measure is the
        mean KL divergence from the ground truth's choice-class probabilities
        (nats per display), on the design pool and on this experiment's
        designed displays; RMSE on the pool is kept beside it. The 2026-10-09
        rehearsal's pool RMSE said its models were getting closer while they
        predicted the designed displays worse: an average over 794 displays,
        most of which tell little apart, hides the few that matter."""
        if self.cfg.collection != "simulated":
            return None
        out = self.private / f"experiment{n}" / "recovery.json"
        if out.exists():
            return out
        pool = design_run.design_pool()
        designed = [spec.context() for spec in Design.load(self.exp(n) / "design" / "design.json").specs]
        gt = Path(self.cfg.ground_truth)
        gt_fit = loop_fit(gt, gt.stem, self.existing(), self.settings, self.cache,
                          time_limit_sec=self.ref_limit)
        gt_model = RSAModel(gt, name=gt.stem)
        truth_pool = posterior_mean_class_probs(gt_model, gt_fit, pool)
        truth_design = posterior_mean_class_probs(gt_model, gt_fit, designed)

        def distances(folder: Path, data: Path, cache: Path) -> Dict[str, dict]:
            d = {}
            for name in read_manifest_names(folder):
                path = folder / f"{name}.py"
                fitted = loop_fit(path, name, data, self.settings, cache, time_limit_sec=self.ref_limit)
                model = RSAModel(path, name=name)
                p_pool = posterior_mean_class_probs(model, fitted, pool)
                p_design = posterior_mean_class_probs(model, fitted, designed)
                d[name] = dict(kl_pool=mean_kl(truth_pool, p_pool, len(pool)),
                               kl_design=mean_kl(truth_design, p_design, len(designed)),
                               rmse_pool=float(np.sqrt(np.mean((p_pool - truth_pool) ** 2))))
            return dict(sorted(d.items(), key=lambda kv: kv[1]["kl_pool"]))

        loop = self.model_loop(n)
        export = json.loads((loop / "export.json").read_text())
        before = distances(self.models_input(n), self.prior_data(n), self.cache)
        after = distances(loop / "models", self.exp(n) / "data" / "cumulative.csv", loop / ".fit_cache")
        _write_json(dict(
            experiment=n, ground_truth=gt.stem, selection_scope=self.cfg.selection_scope,
            measure="kl_pool: mean KL(ground truth || model) over the design pool's displays (nats); "
                    "kl_design: the same over this experiment's designed displays; rmse_pool: RMSE on the pool",
            exported=export["best_model"], exported_distance=after[export["best_model"]],
            closest_after=next(iter(after)), closest_after_distance=next(iter(after.values())),
            closest_before=next(iter(before)), closest_before_distance=next(iter(before.values())),
            before=before, after=after,
        ), out)
        return out

    def run(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        record = self.private / "outer_config.json"
        cfg = {k: str(v) if isinstance(v, Path) else v for k, v in asdict(self.cfg).items()}
        # How a live run recruits may change between its test deployment and its
        # live one, and on a resume; what it studies may not.
        operational = ("prolific_mode", "confirm_live_recruitment", "max_wait_sec", "repo_root")
        same = lambda a, b: {k: v for k, v in a.items() if k not in operational} == {  # noqa: E731
            k: v for k, v in b.items() if k not in operational}
        if record.exists() and not same(json.loads(record.read_text()), cfg):
            raise ValueError(f"{record} records another configuration; a run is resumed with its own")
        _write_json(cfg, record)
        from src.rsa.live.collect import DraftOnly

        for n in range(1, self.cfg.n_experiments + 1):
            print(f"[outer] {self.label(n)}: design", flush=True)
            self.design(n)
            print(f"[outer] {self.label(n)}: collect ({self.cfg.collection})", flush=True)
            try:
                self.collect(n)
            except DraftOnly as stop:
                print(f"[outer] {self.label(n)}: stopped: {stop}", flush=True)
                return
            if self.cfg.stop_after_collect:
                print(f"[outer] {self.label(n)}: stopped after collection (stop_after_collect)", flush=True)
                return
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
    from src.rsa.cpus import pin_main_thread

    pin_main_thread()  # one core per process (src.rsa.cpus)
    main(tyro.cli(Args))
