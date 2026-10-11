"""The RSA inner loop: agents propose memo models; the data decide.

A parallel of the PyMC domain's `src.pipelines.inner_loop.pymc_orchestrator`
that reuses its domain-neutral parts unchanged (slot roles and the lens walk,
the hypothesis ledger, the agent launcher, the LOO reliability verdict and
convergence gate, the clustered SE of an ELPD difference) and has its own
memo admission (`src.rsa.loop.gates`), fitting and novelty pool.

Layout under ``results_dir``::

    responses.csv               the trials the loop fits (copied in once)
    models/                     the live set: <name>.py + models_manifest.yaml
    models/pruned/              models pruned or retired at the end
    attempted_hypotheses.jsonl  the ledger (every slot, admitted or not)
    novelty_pool.json           the novelty gate's displays
    .fit_cache/                 content-addressed fits
    round_<k>/candidate_<i>[_retry_1|_repair_1]/   agent working directories
    steps/step_<n>/             comparison.csv, params.csv, cells.csv, summary.json
    history.json                one entry per scoring step
    report.html                 the latest report page (src.rsa.report)
    best_model.py, export.json  the exported winner

Each round: spawn every slot's agent concurrently; re-spawn once a slot that
wrote no candidate.py; admit sequentially; re-spawn once, with the reason, a
candidate that was refused; admit the repairs; score. At the end: prune
models distinguishable from the best trusted model, cap the live set, export.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

from src.models.clustered_se import cluster_dse
from src.models.model_manifest import read_manifest_entries
from src.pipelines.inner_loop.hypothesis_ledger import (
    LEDGER_FILENAME,
    HypothesisLedger,
    LedgerEntry,
    collapse_whitespace,
)
from src.pipelines.inner_loop.model_zoo import (
    SLOT_EXPLORE,
    SLOT_REFINE_INCUMBENT,
    _lens_index,
    exploratory_slots_per_round,
    slot_roles,
)
from src.rsa.dataset import load_forced_choice
from src.rsa.fit import FitSettings, RSAFit, compare
from src.rsa.loop import brief as briefs
from src.rsa.loop import critique as critique_mod
from src.rsa.loop.fitting import FIT_TIME_LIMIT_SEC, NO_TIME_LIMIT, ModelFailure
from src.rsa.loop.code_gate import code_problems
from src.rsa.cpus import job_cpus, one_core
from src.rsa.loop.cv import CVResult, compare_cv, cv_pointwise, make_folds, source_labels
from src.rsa.loop.gates import GateConfig, admit, fit_with_refit, read_candidate
from src.rsa.loop.novelty import (
    DEFAULT_NOVELTY_RMSE_THRESHOLD,
    novelty_pool,
    plain_pool,
    pool_digest,
    pool_record,
    posterior_mean_class_probs,
)
from src.rsa.model_file import RSAModel

DEFAULT_PRUNE_DSE_MULTIPLIER = 2.0
# The live set's cap after the end-of-run prune (PI decision 2026-10-08, fixed
# before Sherlock run 2): up to 12 models not distinguishable from the best
# under grouped CV, ranked by total ELPD-CV, with no source-specific rule. At 8
# the cap, not the prune, decided what survived; at 12, run 1's real_rep1 would
# have kept 4 of the 5 per-source best models.
MAX_LIVE_MODELS = 12
NAME_RE = re.compile(r"^[a-z][a-z0-9_]{2,39}$")
DISPLAY_COLUMNS = ("objects", "query", "utterance", "familiarization", "grayscale", "framing")
# The unit the end-of-run prune treats as one observation (clustered SE of an
# ELPD difference): a display within an experimental condition. Clustering by
# display alone (PI decision 2026-10-07 to change it) made the pragmods
# simple display, shared by many experiments, one cluster of thousands of
# trials whose sum dominated the variance: nothing was pruned, not even the
# literal listener 554 nats behind (SE 324). By experiment x condition x
# display it is 4.4 SEs behind; in a multi-trial design the unit is the item.
CLUSTER_COLUMNS = ("source", "experiment", "condition") + DISPLAY_COLUMNS + ("messages",)


def cluster_ids(frame: pd.DataFrame) -> np.ndarray:
    """One id per trial; trials of one display in one experimental condition share it.

    Columns a dataset lacks (pragmods has no source or messages) are left out.
    """
    cols = [c for c in CLUSTER_COLUMNS if c in frame.columns]
    missing = {"experiment", "condition"} - set(cols)
    if missing:
        raise ValueError(f"responses lack the cluster columns {sorted(missing)}")
    return pd.factorize(frame[cols].astype(str).agg("|".join, axis=1))[0]

SpawnFn = Callable[[Path, str], bool]


@dataclass
class LoopConfig:
    responses_path: Path
    seed_models_dir: Path
    results_dir: Path
    max_iterations: int = 1
    candidate_count: int = 3
    settings: FitSettings = field(default_factory=FitSettings)
    novelty_threshold: float = DEFAULT_NOVELTY_RMSE_THRESHOLD
    prune_dse_multiplier: float = DEFAULT_PRUNE_DSE_MULTIPLIER
    max_live_models: int = MAX_LIVE_MODELS
    fit_time_limit_sec: Optional[float] = FIT_TIME_LIMIT_SEC
    lenses: List[str] = field(default_factory=lambda: list(briefs.DEFAULT_RSA_LENSES))
    report_title: str = "RSA inner loop"
    selection: str = "cv"
    """What the loop selects, prunes and exports on: "cv", grouped
    cross-validation over training conditions (src.rsa.loop.cv; PI decision
    2026-10-08), or "loo", trial-level PSIS-LOO (Sherlock run 1)."""
    cv_folds: int = 5
    selection_source: Optional[str] = None
    """With grouped CV, select (standing, incumbent, prune, cap, export) on
    the out-of-fold lpd of this source's trials only, while every fit uses
    all trials: the live loop's "fit on everything, select on the live
    data" (``auto_psych``). None: all trials (Sherlock run 2). The overall
    ELPD-CV stays in the standing beside it."""
    selection_guard_dse: float = 0.0
    """With ``selection_source``: a model may be selected (incumbent, export)
    only while its ELPD-CV on the *other* trials is within this many clustered
    SEs of the best there; one further behind is pruned at the end. The live
    loop's guarded rule (PI 2026-10-10): rank on the new experiment, but stay
    answerable to the literature. 0: no guard. Without it the 2026-10-09
    rehearsal's live-only selection collapsed onto one family of near-copies
    that predicted the next experiment worse than plain RSA."""
    novelty_pool: str = "full"
    """The displays the novelty gate compares predictions on: "full" (with
    valence, familiarization and greyscale displays) or "plain" (the live
    phase's displays, `src.rsa.loop.novelty.plain_pool`; PI 2026-10-10)."""
    brief_note: str = ""
    """Text added to every agent's context (the live phase's scope note)."""
    inherit_ledger: Optional[Path] = None
    """A previous inner loop's ledger to continue (the live phase: what earlier
    experiments tried and pruned reaches this experiment's agents; PI 2026-10-10)."""
    agent_network: bool = False
    """Whether agents have internet access (see `coding_agent_spawner`); the
    brief tells them which."""
    fit_workers: int = 0
    """Fits run at once (each its own process): a round's candidates, and the
    seeds, are fitted concurrently before their sequential admission. 0: as
    many as the CPUs this process may use (`default_fit_workers`)."""
    critique: bool = False
    """Run the critique step before every round (`src.rsa.loop.critique`, main's
    CriticAL): a critique agent's test statistics, scored against the
    incumbent's posterior-predictive replicates, and the significant
    discrepancies in every candidate's brief (PI 2026-10-10: on for the live
    phase; runs 1-2 had none)."""
    n_critique_proposals: int = critique_mod.CRITIQUE_N_PROPOSALS
    critique_alpha: float = critique_mod.CRITIQUE_SIGNIFICANCE_ALPHA
    n_critique_replicates: int = critique_mod.CRITIQUE_PPC_REPLICATES
    stop_after_stale_rounds: int = 2
    """End the run early once this many rounds in a row have left the best
    model unchanged (`stale_rounds`); 0: always run ``max_iterations`` rounds.
    PI decision 2026-10-08. Not 1: in Sherlock run 2 a round without a new
    best model was often followed by the run's biggest gain (real_rep1: rsa_l2
    stayed best after round 1; unchanged at round 6, +40 lpd held out at round 7)."""


@dataclass
class Live:
    name: str
    hypothesis: str
    model: RSAModel
    fit: RSAFit
    pool_preds: np.ndarray
    source: str  # "seed" or the ledger context of the slot that proposed it


def default_fit_workers() -> int:
    """One fit per CPU the job was given (each fit runs single-threaded,
    `src.rsa.loop.fitting.FIT_THREADS_ENV`)."""
    return max(1, len(job_cpus()))  # the job's cores, not the (pinned) calling thread's


class RSALoop:
    def __init__(self, cfg: LoopConfig, spawn: SpawnFn) -> None:
        self.cfg = cfg
        self.spawn = spawn
        self.dir = Path(cfg.results_dir)
        self.models_dir = self.dir / "models"
        self.responses = self.dir / "responses.csv"
        self.live: Dict[str, Live] = {}
        self.history: List[dict] = []
        self.step = 0

    # ---------- setup ----------
    def _prepare(self) -> None:
        """What a fresh start and a resume share: data, clusters, pool, gates."""
        self.dir.mkdir(parents=True, exist_ok=True)
        if not self.responses.exists():
            df = pd.read_csv(self.cfg.responses_path)
            df.to_csv(self.responses, index=False)
        self.trials = load_forced_choice(self.responses)
        if not self.trials.contexts:
            raise ValueError(f"{self.cfg.responses_path} has no included forced-choice trials")
        self.clusters = cluster_ids(self.trials.frame)
        if self.cfg.novelty_pool not in ("full", "plain"):
            raise ValueError(f"novelty_pool must be 'full' or 'plain', not {self.cfg.novelty_pool!r}")
        self.pool = plain_pool() if self.cfg.novelty_pool == "plain" else novelty_pool()
        if self.cfg.selection not in ("cv", "loo"):
            raise ValueError(f"selection must be 'cv' or 'loo', not {self.cfg.selection!r}")
        self.cv: Dict[str, Optional[CVResult]] = {}
        if self.cfg.selection == "cv":
            self.folds = make_folds(self.responses, self.dir / ".cv", self.cfg.cv_folds, seed=self.cfg.settings.seed)
        self.gate_cfg = GateConfig(
            responses_path=self.responses, cache_dir=self.dir / ".fit_cache",
            settings=self.cfg.settings, novelty_threshold=self.cfg.novelty_threshold,
            time_limit_sec=self.cfg.fit_time_limit_sec,
        )
        # Models already in play (seeds, carried, admitted) are never time-limited:
        # only a candidate's admission fit is (`src.rsa.loop.fitting.NO_TIME_LIMIT`).
        self.live_cfg = replace(
            self.gate_cfg, time_limit_sec=None if self.cfg.fit_time_limit_sec is None else NO_TIME_LIMIT)

    def setup(self) -> None:
        self._prepare()
        (self.dir / "novelty_pool.json").write_text(json.dumps(
            dict(digest=pool_digest(self.pool), contexts=[pool_record(c) for c in self.pool])))
        if self.cfg.inherit_ledger is not None and not Path(self.cfg.inherit_ledger).exists():
            raise FileNotFoundError(f"inherit_ledger {self.cfg.inherit_ledger} does not exist")
        self.ledger = HypothesisLedger.create(self.dir / LEDGER_FILENAME, inherit_from=self.cfg.inherit_ledger)
        self.models_dir.mkdir(exist_ok=True)
        entries = read_manifest_entries(self.cfg.seed_models_dir)
        for entry in entries:
            shutil.copyfile(Path(self.cfg.seed_models_dir) / f"{entry['name']}.py",
                            self.models_dir / f"{entry['name']}.py")
        self._prefit([(self.models_dir / f"{e['name']}.py", e["name"]) for e in entries], self.live_cfg)
        for entry in entries:
            name, rationale = entry["name"], entry.get("rationale", "")
            dst = self.models_dir / f"{name}.py"
            model = RSAModel(dst, name=name)
            try:
                fitted = fit_with_refit(dst, name, self.live_cfg)
            except ModelFailure as exc:
                dst.unlink()
                self._ledger(name, "dropped", f"seed failed to fit: {exc}", rationale, "seed")
                continue
            preds = posterior_mean_class_probs(model, fitted, self.pool)
            self.live[name] = Live(name, rationale, model, fitted, preds, "seed")
        if not self.live:
            raise RuntimeError("no seed model could be fitted; nothing to compare")
        self._write_manifest()

    # ---------- bookkeeping ----------
    def _ledger(self, name: str, outcome: str, detail: str, hypothesis: str, context: str) -> None:
        self.ledger.append(LedgerEntry(name=name, outcome=outcome, detail=detail,
                                       hypothesis=collapse_whitespace(hypothesis), context=context))

    def _write_manifest(self) -> None:
        entries = [dict(name=m.name, rationale=collapse_whitespace(m.hypothesis)) for m in self.live.values()]
        (self.models_dir / "models_manifest.yaml").write_text(yaml.safe_dump({"models": entries}, sort_keys=False))

    def _reserved(self) -> set:
        return set(self.live) | {e.name for e in self.ledger.entries()}

    def _name_for(self, candidate_dir: Path, fallback: str, taken: frozenset = frozenset()) -> str:
        f = candidate_dir / "model_name.txt"
        raw = f.read_text(encoding="utf-8").strip().splitlines()[0].strip() if f.exists() and f.read_text().strip() else ""
        name = raw if NAME_RE.match(raw) else fallback
        base, k = name, 2
        while name in self._reserved() | taken:
            name = f"{base}_{k}"
            k += 1
        return name

    # ---------- scoring ----------
    def _cv_for(self, names: List[str]) -> None:
        """Compute (once per model) the grouped-CV lpd of the named models,
        several at a time. A model whose fold fit fails has no CV (None) and
        is untrusted for selection."""
        todo = [n for n in names if n not in self.cv]
        if not todo:
            return
        workers = self.cfg.fit_workers or default_fit_workers()

        def one(name: str):
            try:
                return name, cv_pointwise(self.models_dir / f"{name}.py", name, self.responses, self.folds,
                                          self.cfg.settings, self.dir / ".fit_cache",
                                          time_limit_sec=self.live_cfg.time_limit_sec,
                                          workers=self.cfg.cv_folds)
            except ModelFailure as exc:
                print(f"  {name}: no grouped CV ({exc})")
                return name, None

        parallel = 1 if self.live_cfg.time_limit_sec is None else max(1, workers // self.cfg.cv_folds)
        with ThreadPoolExecutor(max_workers=parallel) as pool:
            for name, result in pool.map(one, todo):
                self.cv[name] = result

    def standing(self) -> Dict[str, dict]:
        """Every live model's PSIS-LOO and, with grouped CV, its ELPD-CV. The
        ``sel_*`` fields are the selection criterion's (cfg.selection) and
        ``trusted`` says whether the model may be selected or prune others."""
        table = compare({n: m.fit for n, m in self.live.items()})
        out = {}
        for name, row in table.iterrows():
            loo = self.live[name].fit.loo()
            out[name] = dict(rank=int(row["rank"]), elpd_loo=float(row["elpd_loo"]), se=float(row["se"]),
                             elpd_diff=float(row["elpd_diff"]), dse=float(row["dse"]), p_loo=float(row["p_loo"]),
                             loo_reliable=not loo.unreliable, converged=self.live[name].fit.converged,
                             convergence_problems="; ".join(self.live[name].fit.convergence_problems))
            out[name]["trusted"] = out[name]["loo_reliable"] and out[name]["converged"]
        if self.cfg.selection == "cv":
            self._cv_for(list(out))
            have = {n: self.cv[n] for n in out if self.cv[n] is not None}
            cv = compare_cv(have, self.folds.units, source_labels(self.trials.frame)) if have else {}
            mask, clusters = self._selection_rows()
            sel = compare_cv({n: CVResult(r.pointwise[mask], r.converged) for n, r in have.items()}, clusters) \
                if have else {}
            for name, s in out.items():
                c = cv.get(name, dict(elpd_cv=float("-inf"), cv_diff=float("inf"), cv_dse=0.0, cv_converged=False))
                s.update(c)
                s["trusted"] = s["trusted"] and c["cv_converged"]
                z = sel.get(name, dict(elpd_cv=float("-inf"), cv_diff=float("inf"), cv_dse=0.0))
                s.update(sel_elpd=z["elpd_cv"], sel_diff=z["cv_diff"], sel_dse=z["cv_dse"])
            if self._guarded() and have:
                rest = compare_cv({n: CVResult(r.pointwise[~mask], r.converged) for n, r in have.items()},
                                  self.folds.units[~mask])
                for name, s in out.items():
                    g = rest.get(name)
                    s["guard_diff"] = float("inf") if g is None else g["cv_diff"]
                    s["guard_dse"] = 0.0 if g is None else g["cv_dse"]
                    s["eligible"] = g is not None and g["cv_diff"] <= self.cfg.selection_guard_dse * g["cv_dse"]
                    s["trusted"] = s["trusted"] and s["eligible"]
        else:
            for s in out.values():
                s.update(sel_elpd=s["elpd_loo"], sel_diff=s["elpd_diff"], sel_dse=s["dse"])
        return out

    def _guarded(self) -> bool:
        return self.cfg.selection == "cv" and bool(self.cfg.selection_source) and self.cfg.selection_guard_dse > 0

    def _selection_rows(self) -> tuple[np.ndarray, np.ndarray]:
        """The trials selection sums over (all, or cfg.selection_source's) and their CV clusters."""
        if self.cfg.selection_source is None:
            return np.ones(len(self.folds.units), dtype=bool), self.folds.units
        mask = source_labels(self.trials.frame) == self.cfg.selection_source
        if not mask.any():
            raise ValueError(f"selection_source {self.cfg.selection_source!r} has no trials in {self.responses}")
        return mask, self.folds.units[mask]

    def incumbent(self, standing: Dict[str, dict]) -> str:
        trusted = [n for n, s in standing.items() if s["trusted"]]
        pool = trusted or list(standing)
        return max(pool, key=lambda n: standing[n]["sel_elpd"])

    def score(self, round_index: int, events: List[dict], critique: Optional[dict] = None) -> dict:
        from src.rsa.compare_seeds import cell_table
        from src.rsa.fit import posterior_mean_probs

        standing = self.standing()
        best = self.incumbent(standing)
        step_dir = self.dir / "steps" / f"step_{self.step}"
        step_dir.mkdir(parents=True, exist_ok=True)
        comp = pd.DataFrame.from_dict(standing, orient="index").sort_values("rank")
        comp.to_csv(step_dir / "comparison.csv")
        rows = []
        for name, m in self.live.items():
            for p in m.fit.param_names:
                d = np.asarray(m.fit.idata.posterior[p]).ravel()
                rows.append(dict(model=name, param=p, mean=d.mean(), lo=np.quantile(d, 0.03), hi=np.quantile(d, 0.97)))
        pd.DataFrame(rows).to_csv(step_dir / "params.csv", index=False)
        preds = {n: posterior_mean_probs(m.model, m.fit, self.trials.contexts) for n, m in self.live.items()}
        cell_table(self.trials, preds).to_csv(step_dir / "cells.csv", index=False)
        (step_dir / "summary.json").write_text(json.dumps(dict(
            n_trials=len(self.trials.contexts), settings=vars(self.cfg.settings), step=self.step)))
        # The live set and the ledger's length at this step: what --resume
        # restores (`resume`).
        entry = dict(step=self.step, round=round_index, best_model=best, standing=standing, events=events,
                     critique=critique,
                     live=[dict(name=m.name, hypothesis=m.hypothesis, source=m.source) for m in self.live.values()],
                     ledger_entries=len(self.ledger.entries()))
        self.history.append(entry)
        _write_atomic(self.dir / "history.json", json.dumps(self.history, indent=1))
        self._write_report(step_dir)
        self.step += 1
        return entry

    def _write_report(self, step_dir: Path) -> None:
        from src.rsa.report import build_bundle, render

        bundle = build_bundle(step_dir, self.models_dir, title=self.cfg.report_title,
                              dataset_label=f"{self.responses.name} · step {self.step}", min_cell_n=10)
        bundle["timeline"] = dict(history=[
            dict(step=h["step"], round=h["round"], best_model=h["best_model"], events=h["events"])
            for h in self.history
        ])
        (self.dir / "report.html").write_text(render(bundle), encoding="utf-8")
        (self.dir / "report.bundle.json").write_text(json.dumps(bundle, indent=1))

    # ---------- a round ----------
    def _docs_for(self, cdir: Path, role: str, lens: Optional[str], round_index: int,
                  standing: Dict[str, dict], incumbent: str, note: Optional[str],
                  critiques: Optional[str] = None) -> str:
        def desc(n):
            if n not in standing:
                return "admitted this round, not yet scored"
            s = standing[n]
            what = "on held-out training conditions (grouped CV)" if self.cfg.selection == "cv" else "(PSIS-LOO)"
            if self.cfg.selection == "cv" and self.cfg.selection_source:
                what = (f"on held-out {self.cfg.selection_source} conditions (grouped CV over the "
                        f"{self.cfg.selection_source} trials, the ones the loop selects on)")
            if self._guarded() and "guard_diff" in s and not s["eligible"]:
                what += (f"; NOT eligible for selection: {s['guard_diff']:.1f} ± {s['guard_dse']:.1f} nats behind "
                         f"the best on the existing (non-{self.cfg.selection_source}) data, more than "
                         f"{self.cfg.selection_guard_dse:g} SEs")
            text = (f"best {what}" if s["sel_diff"] == 0
                    else f"{s['sel_diff']:.1f} ± {s['sel_dse']:.1f} nats behind the best {what}")
            by = s.get("cv_behind_by_source")
            if by and len(by) > 1:
                # Models specialise by source: say where this one leads and lags.
                parts = [("best" if v == 0 else f"{v:.1f} ± {s['cv_dse_by_source'][k]:.1f} behind") + f" on {k}"
                         for k, v in by.items()]
                text += "; by source: " + ", ".join(parts)
            return text
        # Ranked as the refinement menu shows them: live models best first
        # (unscored last), pruned ones by how far behind the best they were.
        def behind(n):
            return standing[n]["sel_diff"] if n in standing else float("inf")
        live = [briefs.ZooModel(n, self.live[n].hypothesis, self.models_dir / f"{n}.py", desc(n))
                for n in sorted(self.live, key=behind)]
        pruned_dir = self.models_dir / "pruned"
        pruned = [briefs.ZooModel(e.name, e.hypothesis, pruned_dir / f"{e.name}.py", f"pruned: {e.detail}")
                  for e in sorted(self.ledger.pruned(self.live), key=lambda e: prune_margin(e.detail))
                  if (pruned_dir / f"{e.name}.py").exists()]
        context = briefs.context_md(
            candidate_dir=cdir, responses_path=self.responses, round_index=round_index,
            n_rounds=self.cfg.max_iterations, n_trials=len(self.trials.contexts),
            experiments=sorted(self.trials.frame["experiment"].unique()), network=self.cfg.agent_network,
            scope_note=self.cfg.brief_note)
        docs = briefs.write_docs(cdir, role=role, lens=lens, context=context, live=live, pruned=pruned,
                                 incumbent=incumbent, ledger=self.ledger, attempt_note=note, critiques=critiques)
        return briefs.build_prompt(cdir, docs)

    def _spawn_all(self, jobs: List[tuple[Path, str]]) -> List[bool]:
        if not jobs:
            return []
        with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
            return list(pool.map(lambda j: self.spawn(*j), jobs))

    def _prefit(self, items: List[tuple[Path, str]], cfg: Optional[GateConfig] = None) -> None:
        """Fit models concurrently (cached; a model's failure is remembered) so
        that their sequential admission reads finished fits: the same
        verdicts as fitting at admission, in the time of the slowest fit
        rather than the sum. Files the cheap gates refuse are not fitted."""
        def one(item: tuple[Path, str]) -> None:
            path, name = item
            if path.name == "candidate.py" and read_candidate(path.parent) is not None:
                return
            if code_problems(path.read_text(encoding="utf-8")):
                return
            try:
                fit_with_refit(path, name, cfg or self.gate_cfg)
            except ModelFailure:
                pass  # remembered; admission reports it

        if not items:
            return
        workers = min(len(items), self.cfg.fit_workers or default_fit_workers())
        if self.gate_cfg.time_limit_sec is None:
            workers = 1  # fits run in this process then, and numpyro's handler stack is not thread-safe
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(one, items))

    def _try_admit(self, cdir: Path, name: str, context: str) -> tuple[bool, str]:
        hyp_file = cdir / "hypothesis.md"
        hypothesis = hyp_file.read_text(encoding="utf-8") if hyp_file.exists() else ""
        same = self._same_hypothesis(hypothesis)
        if same is not None:
            reason = (f"hypothesis.md is word for word {same}'s hypothesis. State this candidate's own "
                      "mechanism, including what it changes from the model it refines: the ledger, the "
                      "briefs and the critique describe every model by its hypothesis.")
            self._ledger(name, "rejected", reason, hypothesis, context)
            return False, reason
        out = admit(cdir, name, cfg=self.gate_cfg, training=self.trials.contexts, pool=self.pool,
                    admitted_preds={n: m.pool_preds for n, m in self.live.items()})
        if out.admitted:
            dst = self.models_dir / f"{name}.py"
            shutil.copyfile(cdir / "candidate.py", dst)
            self.live[name] = Live(name, hypothesis, RSAModel(dst, name=name), out.fit, out.pool_preds, context)
            self._write_manifest()
            self._ledger(name, "admitted", f"ELPD-LOO {out.elpd_loo:.1f}", hypothesis, context)
        else:
            self._ledger(name, "rejected", out.reason, hypothesis, context)
        return out.admitted, out.reason

    def _same_hypothesis(self, hypothesis: str) -> Optional[str]:
        """The model (live, or admitted earlier in the run's ledger) whose hypothesis
        this one repeats verbatim, ignoring whitespace and case; None if it is new.
        Rehearsal 3's round-1 winner copied its parent's hypothesis, so the ledger,
        every brief and the next critique described it without its new mechanism."""
        key = collapse_whitespace(hypothesis).strip().lower()
        if not key:
            return None
        texts = [(m.name, m.hypothesis) for m in self.live.values()]
        texts += [(e.name, e.hypothesis) for e in self.ledger.entries() if e.outcome == "admitted"]
        for name, text in texts:
            if collapse_whitespace(text).strip().lower() == key:
                return name
        return None

    def run_round(self, round_index: int) -> dict:
        standing = self.standing()
        incumbent = self.incumbent(standing)
        roles = slot_roles(self.cfg.candidate_count)
        per_round = exploratory_slots_per_round(self.cfg.candidate_count)
        rdir = self.dir / f"round_{round_index + 1}"
        crit = self._critique(rdir, incumbent)
        critiques = crit.critiques_md
        slots, explore_i = [], 0
        for i, role in enumerate(roles):
            lens = None
            if role == SLOT_EXPLORE:
                lens = self.cfg.lenses[_lens_index(0, round_index, per_round, explore_i, len(self.cfg.lenses))]
                explore_i += 1
            # Only the incumbent slot's target is known; which model a
            # refine-chosen agent picked is in its hypothesis, never parsed.
            context = f"round {round_index + 1} candidate {i + 1} {role}" + (
                f" {incumbent}" if role == SLOT_REFINE_INCUMBENT else "")
            slots.append(dict(i=i, role=role, lens=lens, dir=rdir / f"candidate_{i + 1}", context=context))
        jobs = [(s["dir"], self._docs_for(s["dir"], s["role"], s["lens"], round_index, standing, incumbent, None,
                                          critiques))
                for s in slots]
        self._spawn_all(jobs)
        # Retry once a slot that wrote no candidate.py.
        empty = [s for s in slots if not (s["dir"] / "candidate.py").exists()]
        retry_jobs = []
        for s in empty:
            s["dir"] = s["dir"].with_name(s["dir"].name + "_retry_1")
            s["context"] += " retry 1"
            retry_jobs.append((s["dir"], self._docs_for(s["dir"], s["role"], s["lens"], round_index, standing,
                                                        incumbent, briefs.retry_note(), critiques)))
        self._spawn_all(retry_jobs)
        events, repairs = [], []
        # Names first (as sequential admission would give them), then every
        # candidate's fit at once, then admission one by one.
        taken: set = set()
        for s in slots:
            if (s["dir"] / "candidate.py").exists():
                s["name"] = self._name_for(s["dir"], f"r{round_index + 1}_c{s['i'] + 1}", frozenset(taken))
                taken.add(s["name"])
        self._prefit([(s["dir"] / "candidate.py", s["name"]) for s in slots if "name" in s])
        for s in slots:
            if not (s["dir"] / "candidate.py").exists():
                self._ledger(f"slot_{s['i'] + 1}", "rejected", "the agent wrote no candidate.py", "", s["context"])
                events.append(dict(slot=s["i"] + 1, role=s["role"], name=None, outcome="no file", reason=""))
                continue
            name = s["name"]
            ok, reason = self._try_admit(s["dir"], name, s["context"])
            events.append(dict(slot=s["i"] + 1, role=s["role"], name=name, outcome="admitted" if ok else "rejected",
                               reason=reason, hypothesis=_read(s["dir"] / "hypothesis.md")))
            if not ok:
                repairs.append((s, reason))
        # Repair once, with the reason, every refused candidate.
        repair_jobs = []
        for s, reason in repairs:
            rd = s["dir"].with_name(s["dir"].name + "_repair_1")
            rd.mkdir(parents=True, exist_ok=True)
            for f in ("candidate.py", "hypothesis.md", "model_name.txt"):
                if (s["dir"] / f).exists():
                    shutil.copyfile(s["dir"] / f, rd / f)
            s["repair_dir"] = rd
            repair_jobs.append((rd, self._docs_for(rd, s["role"], s["lens"], round_index, standing, incumbent,
                                                   briefs.repair_note(reason), critiques)))
        self._spawn_all(repair_jobs)
        taken = set()
        for s, _ in repairs:
            s["repair_name"] = self._name_for(s["repair_dir"], f"r{round_index + 1}_c{s['i'] + 1}", frozenset(taken))
            taken.add(s["repair_name"])
        self._prefit([(s["repair_dir"] / "candidate.py", s["repair_name"]) for s, _ in repairs
                      if (s["repair_dir"] / "candidate.py").exists()])
        for s, _ in repairs:
            rd = s["repair_dir"]
            name = s["repair_name"]
            ok, reason = self._try_admit(rd, name, s["context"] + " repair 1")
            events.append(dict(slot=s["i"] + 1, role=s["role"], name=name, outcome="admitted" if ok else "rejected",
                               reason=reason, hypothesis=_read(rd / "hypothesis.md"), repair=True))
        return self.score(round_index, events, crit.status)

    def _critique(self, rdir: Path, incumbent: str) -> "critique_mod.CritiqueOutcome":
        """The round's critique of the incumbent (`src.rsa.loop.critique`), or a
        disabled record. A critique that fails for any reason but the agents'
        own infrastructure is recorded and the round runs without one (main's
        rule: a critique failure must not end a long run)."""
        if not self.cfg.critique:
            return critique_mod.CritiqueOutcome(None, critique_mod.disabled_status())
        from src.runtime.coding_agent import AgentPermissionDenied
        from src.runtime.usage_limits import AgentInfrastructureError

        m = self.live[incumbent]
        try:
            return critique_mod.run_critique(
                rdir, spawn=self.spawn, incumbent=incumbent, model=m.model, fitted=m.fit,
                hypothesis=m.hypothesis, incumbent_file=self.models_dir / f"{incumbent}.py",
                frame=self.trials.frame, contexts=self.trials.contexts, choices=self.trials.choices,
                responses_path=self.responses, n_proposals=self.cfg.n_critique_proposals,
                alpha=self.cfg.critique_alpha, n_replicates=self.cfg.n_critique_replicates,
                seed=self.cfg.settings.seed + self.step, scope_note=self.cfg.brief_note,
                source=self.cfg.selection_source)
        except (AgentPermissionDenied, AgentInfrastructureError):
            raise
        except Exception as exc:  # recorded, never swallowed silently
            reason = f"{type(exc).__name__}: {exc}"
            print(f"  [critique] NO CRITIQUE this round: {reason}", flush=True)
            return critique_mod.CritiqueOutcome(None, dict(
                status=critique_mod.CRITIQUE_STATUS_NONE, incumbent=incumbent, reason=reason))

    # ---------- end of the run ----------
    def end(self, stopped: Optional[str] = None) -> dict:
        standing = self.standing()
        trusted = [n for n, s in standing.items() if s["trusted"]]
        events = []
        criterion = "grouped CV" if self.cfg.selection == "cv" else "PSIS-LOO"
        if self.cfg.selection == "cv" and self.cfg.selection_source:
            criterion += f" on {self.cfg.selection_source} trials"
        if self._guarded():
            for n, s in standing.items():
                if not s["eligible"] and s["guard_diff"] != float("inf") and n in self.live:
                    reason = (f"{s['guard_diff']:.1f} nats behind the best on the other trials (grouped CV), "
                              f"> {self.cfg.selection_guard_dse} x clustered dse {s['guard_dse']:.1f}")
                    self._retire(n, "pruned", f"ineligible: {reason}")
                    events.append(dict(name=n, outcome="pruned", reason=reason))
            standing = {n: s for n, s in standing.items() if n in self.live}
            trusted = [n for n in trusted if n in self.live]
        if trusted:
            best = max(trusted, key=lambda n: standing[n]["sel_elpd"])
            if self.cfg.selection == "cv":
                mask, clusters = self._selection_rows()
                pw = {n: self.cv[n].pointwise[mask] for n in trusted}
            else:
                pw = {n: np.asarray(self.live[n].fit.loo().loo.loo_i).ravel() for n in trusted}
                clusters = self.clusters
            for n in trusted:
                if n == best:
                    continue
                diff = float(pw[best].sum() - pw[n].sum())
                dse = cluster_dse(pw[best], pw[n], clusters)
                if diff > self.cfg.prune_dse_multiplier * dse:
                    self._retire(n, "pruned", f"elpd_diff {diff:.1f} > {self.cfg.prune_dse_multiplier} x clustered dse {dse:.1f} vs {best} ({criterion})")
                    events.append(dict(name=n, outcome="pruned", reason=f"{diff:.1f} nats behind {best} on {criterion} (clustered dse {dse:.1f})"))
        while len(self.live) > self.cfg.max_live_models:
            standing = self.standing()
            keep = self.incumbent(standing)
            order = sorted((n for n in standing if n != keep),
                           key=lambda n: (standing[n]["trusted"], standing[n]["sel_elpd"]))
            self._retire(order[0], "pruned", f"retired by the cap of {self.cfg.max_live_models} live models")
            events.append(dict(name=order[0], outcome="retired", reason="live-set cap"))
        if stopped:
            events.append(dict(name="loop", outcome="stopped", reason=stopped))
        entry = self.score(self.cfg.max_iterations, events)
        best = entry["best_model"]
        shutil.copyfile(self.models_dir / f"{best}.py", self.dir / "best_model.py")
        export = dict(best_model=best, live=sorted(self.live))
        if stopped:
            export["stopped_early"] = stopped
        (self.dir / "export.json").write_text(json.dumps(export))
        return entry

    def _retire(self, name: str, outcome: str, detail: str) -> None:
        m = self.live.pop(name)
        pruned = self.models_dir / "pruned"
        pruned.mkdir(exist_ok=True)
        shutil.move(str(self.models_dir / f"{name}.py"), pruned / f"{name}.py")
        self._ledger(name, outcome, detail, m.hypothesis, m.source)
        self._write_manifest()

    def resume(self) -> int:
        """Restore an interrupted run to its last scored step; returns the
        first round still to run.

        The step's history entry holds the live set and the ledger's length.
        Whatever a half-finished round did after it is undone: ledger lines
        past that length are dropped, models it admitted are removed from the
        set (their files and fits stay in the cache and the round's
        directory), a model the end step had pruned is put back, and the
        round's directory is renamed ``round_<k>_abandoned_<n>`` so the round
        runs again from its agents. Fits come back from the cache.
        """
        hist_path = self.dir / "history.json"
        if not hist_path.exists():
            raise FileNotFoundError(
                f"{hist_path} not found: the run stopped before its seeds were scored; start it afresh")
        if (self.dir / "export.json").exists():
            raise RuntimeError(f"{self.dir} finished (export.json exists); there is nothing to resume")
        self.history = json.loads(hist_path.read_text())
        last = self.history[-1]
        if "live" not in last or "ledger_entries" not in last:
            raise ValueError(f"{hist_path} predates resume support (no live set recorded); start afresh")
        self._prepare()
        recorded = json.loads((self.dir / "novelty_pool.json").read_text())["digest"]
        if recorded != pool_digest(self.pool):
            raise ValueError("the novelty pool changed since the run started (other code?); start afresh")
        ledger_path = self.dir / LEDGER_FILENAME
        lines = [ln for ln in ledger_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if len(lines) < last["ledger_entries"]:
            raise ValueError(f"{ledger_path} has {len(lines)} entries, fewer than the "
                             f"{last['ledger_entries']} recorded at step {last['step']}")
        _write_atomic(ledger_path, "".join(ln + "\n" for ln in lines[:last["ledger_entries"]]))
        self.ledger = HypothesisLedger(ledger_path)
        self.ledger.entries()  # parses, or raises
        live = {e["name"]: e for e in last["live"]}
        pruned_dir = self.models_dir / "pruned"
        for name in live:
            if not (self.models_dir / f"{name}.py").exists() and (pruned_dir / f"{name}.py").exists():
                shutil.move(str(pruned_dir / f"{name}.py"), self.models_dir / f"{name}.py")
            if not (self.models_dir / f"{name}.py").exists():
                raise FileNotFoundError(f"{name} was live at step {last['step']} but its file is gone")
        for f in self.models_dir.glob("*.py"):
            if f.stem not in live:
                f.unlink()  # admitted by the abandoned round; its candidate dir keeps the source
        next_round = last["round"] + 1
        partial = self.dir / f"round_{next_round + 1}"
        if partial.exists():
            k = 1
            while (self.dir / f"round_{next_round + 1}_abandoned_{k}").exists():
                k += 1
            partial.rename(self.dir / f"round_{next_round + 1}_abandoned_{k}")
        self._prefit([(self.models_dir / f"{n}.py", n) for n in live], self.live_cfg)
        for name, e in live.items():
            path = self.models_dir / f"{name}.py"
            model = RSAModel(path, name=name)
            fitted = fit_with_refit(path, name, self.live_cfg)
            self.live[name] = Live(name, e["hypothesis"], model, fitted,
                                   posterior_mean_class_probs(model, fitted, self.pool), e["source"])
        self._write_manifest()
        self.step = last["step"] + 1
        return next_round

    def run(self, resume: bool = False) -> dict:
        if resume:
            first = self.resume()
        else:
            self.setup()
            self.score(-1, [dict(outcome="seeded", name=n) for n in self.live])
            first = 0
        patience = self.cfg.stop_after_stale_rounds
        for r in range(first, self.cfg.max_iterations):
            self.run_round(r)
            if patience and r + 1 < self.cfg.max_iterations and stale_rounds(self.history) >= patience:
                why = (f"stopped after round {r + 1} of {self.cfg.max_iterations}: {patience} rounds in a row "
                       f"left the best model ({self.history[-1]['best_model']}) unchanged")
                print(f"[rsa loop] {why}", flush=True)
                return self.end(stopped=why)
        return self.end()


def stale_rounds(history: List[dict]) -> int:
    """How many of the latest rounds in a row left the best model unchanged
    (each history step against the one before it; the seeds' step is never stale)."""
    n = 0
    for prev, cur in zip(history[-2::-1], history[::-1]):
        if cur["round"] < 0 or cur["best_model"] != prev["best_model"]:
            break
        n += 1
    return n


def _write_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


_MARGIN_RE = re.compile(r"elpd_diff (-?[0-9.]+)")


def prune_margin(detail: str) -> float:
    """How far behind the best a pruned model was (the prune's own record);
    models retired by the cap, without a margin, sort last."""
    m = _MARGIN_RE.search(detail)
    return float(m.group(1)) if m else float("inf")


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip() if path.exists() else ""


def coding_agent_spawner(*, models_dir: Path, responses_path: Path, timeout_sec: int, backend: Optional[str],
                         model: Optional[str], agent_root: Optional[Path], sandbox: bool,
                         network: bool = False, shell_dir: Optional[Path] = None) -> SpawnFn:
    """The real spawner: one coding agent per candidate directory.

    Without ``network`` (the default; PI decision 2026-10-08) the agent's shell
    commands run under the no-internet filter and opencode's web tools are
    denied (`src.rsa.loop.no_network`); the wrapper is written to
    ``shell_dir``, which the agent must be able to read. An agent whose log
    shows a completed web tool call stops the run.
    """
    from src.runtime.coding_agent import run_coding_agent, select_backend
    from src.runtime.config import REPO_ROOT

    env = None
    if not network:
        from src.rsa.loop import no_network

        if select_backend(backend) != "opencode":
            raise ValueError("agents without network are implemented for the opencode backend only")
        if shell_dir is None:
            raise ValueError("agents without network need shell_dir for their shell wrapper")
        bash = shutil.which("bash")
        if bash is None:
            raise FileNotFoundError("no bash on PATH for the agents' shell wrapper")
        wrapper = no_network.write_shell_wrapper(Path(shell_dir), sys.executable, bash)
        no_network.verify_wrapper(wrapper, sys.executable)  # before any agent runs
        env = dict(os.environ)
        env["SHELL"] = str(wrapper)
        env["OPENCODE_PERMISSION"] = no_network.opencode_permission(env.get("OPENCODE_PERMISSION"))

    def spawn(candidate_dir: Path, prompt: str) -> bool:
        log = candidate_dir / "agent.jsonl"
        # On a core of its own (src.rsa.cpus): the agent's shell, and the JAX
        # fit of its self-check, see one core and size their pools to it.
        with one_core():
            ok, _ = run_coding_agent(
                prompt, cwd=agent_root or REPO_ROOT, log_path=log,
                allowed_dirs=[candidate_dir, models_dir, responses_path.parent], writable_dirs=[candidate_dir],
                timeout_secs=timeout_sec, backend=backend, model=model,
                usage_label="rsa:critique" if candidate_dir.name.startswith("critique") else "rsa:candidate",
                stock=True, sandbox=sandbox, env=env,
            )
        if not network:
            from src.rsa.loop.no_network import web_tool_uses

            used = web_tool_uses(log)
            if used:
                raise RuntimeError(f"{candidate_dir}: the agent used the web although its web tools are denied "
                                   f"(a harness bug): {used[:3]}")
        return ok

    return spawn
