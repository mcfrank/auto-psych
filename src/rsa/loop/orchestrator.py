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
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
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
from src.rsa.loop.fitting import FIT_TIME_LIMIT_SEC, ModelFailure
from src.rsa.loop.code_gate import code_problems
from src.rsa.loop.gates import GateConfig, admit, fit_with_refit, read_candidate
from src.rsa.loop.novelty import (
    DEFAULT_NOVELTY_RMSE_THRESHOLD,
    novelty_pool,
    pool_digest,
    pool_record,
    posterior_mean_class_probs,
)
from src.rsa.model_file import RSAModel

DEFAULT_PRUNE_DSE_MULTIPLIER = 2.0
MAX_LIVE_MODELS = 8
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
    fit_workers: int = 0
    """Fits run at once (each its own process): a round's candidates, and the
    seeds, are fitted concurrently before their sequential admission. 0: as
    many as the CPUs this process may use (`default_fit_workers`)."""


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
    return max(1, len(os.sched_getaffinity(0)))


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
        self.pool = novelty_pool()
        self.gate_cfg = GateConfig(
            responses_path=self.responses, cache_dir=self.dir / ".fit_cache",
            settings=self.cfg.settings, novelty_threshold=self.cfg.novelty_threshold,
            time_limit_sec=self.cfg.fit_time_limit_sec,
        )

    def setup(self) -> None:
        self._prepare()
        (self.dir / "novelty_pool.json").write_text(json.dumps(
            dict(digest=pool_digest(self.pool), contexts=[pool_record(c) for c in self.pool])))
        self.ledger = HypothesisLedger.create(self.dir / LEDGER_FILENAME, inherit_from=None)
        self.models_dir.mkdir(exist_ok=True)
        entries = read_manifest_entries(self.cfg.seed_models_dir)
        for entry in entries:
            shutil.copyfile(Path(self.cfg.seed_models_dir) / f"{entry['name']}.py",
                            self.models_dir / f"{entry['name']}.py")
        self._prefit([(self.models_dir / f"{e['name']}.py", e["name"]) for e in entries])
        for entry in entries:
            name, rationale = entry["name"], entry.get("rationale", "")
            dst = self.models_dir / f"{name}.py"
            model = RSAModel(dst, name=name)
            try:
                fitted = fit_with_refit(dst, name, self.gate_cfg)
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
    def standing(self) -> Dict[str, dict]:
        table = compare({n: m.fit for n, m in self.live.items()})
        out = {}
        for name, row in table.iterrows():
            loo = self.live[name].fit.loo()
            out[name] = dict(rank=int(row["rank"]), elpd_loo=float(row["elpd_loo"]), se=float(row["se"]),
                             elpd_diff=float(row["elpd_diff"]), dse=float(row["dse"]), p_loo=float(row["p_loo"]),
                             loo_reliable=not loo.unreliable, converged=self.live[name].fit.converged,
                             convergence_problems="; ".join(self.live[name].fit.convergence_problems))
        return out

    def incumbent(self, standing: Dict[str, dict]) -> str:
        trusted = [n for n, s in standing.items() if s["loo_reliable"] and s["converged"]]
        pool = trusted or list(standing)
        return max(pool, key=lambda n: standing[n]["elpd_loo"])

    def score(self, round_index: int, events: List[dict]) -> dict:
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
                  standing: Dict[str, dict], incumbent: str, note: Optional[str]) -> str:
        def desc(n):
            if n not in standing:
                return "admitted this round, not yet scored"
            s = standing[n]
            return "best" if s["elpd_diff"] == 0 else f"{s['elpd_diff']:.1f} ± {s['dse']:.1f} nats behind the best"
        # Ranked as the refinement menu shows them: live models best first
        # (unscored last), pruned ones by how far behind the best they were.
        def behind(n):
            return standing[n]["elpd_diff"] if n in standing else float("inf")
        live = [briefs.ZooModel(n, self.live[n].hypothesis, self.models_dir / f"{n}.py", desc(n))
                for n in sorted(self.live, key=behind)]
        pruned_dir = self.models_dir / "pruned"
        pruned = [briefs.ZooModel(e.name, e.hypothesis, pruned_dir / f"{e.name}.py", f"pruned: {e.detail}")
                  for e in sorted(self.ledger.pruned(self.live), key=lambda e: prune_margin(e.detail))
                  if (pruned_dir / f"{e.name}.py").exists()]
        context = briefs.context_md(
            candidate_dir=cdir, responses_path=self.responses, round_index=round_index,
            n_rounds=self.cfg.max_iterations, n_trials=len(self.trials.contexts),
            experiments=sorted(self.trials.frame["experiment"].unique()))
        docs = briefs.write_docs(cdir, role=role, lens=lens, context=context, live=live, pruned=pruned,
                                 incumbent=incumbent, ledger=self.ledger, attempt_note=note)
        return briefs.build_prompt(cdir, docs)

    def _spawn_all(self, jobs: List[tuple[Path, str]]) -> List[bool]:
        if not jobs:
            return []
        with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
            return list(pool.map(lambda j: self.spawn(*j), jobs))

    def _prefit(self, items: List[tuple[Path, str]]) -> None:
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
                fit_with_refit(path, name, self.gate_cfg)
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

    def run_round(self, round_index: int) -> dict:
        standing = self.standing()
        incumbent = self.incumbent(standing)
        roles = slot_roles(self.cfg.candidate_count)
        per_round = exploratory_slots_per_round(self.cfg.candidate_count)
        rdir = self.dir / f"round_{round_index + 1}"
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
        jobs = [(s["dir"], self._docs_for(s["dir"], s["role"], s["lens"], round_index, standing, incumbent, None))
                for s in slots]
        self._spawn_all(jobs)
        # Retry once a slot that wrote no candidate.py.
        empty = [s for s in slots if not (s["dir"] / "candidate.py").exists()]
        retry_jobs = []
        for s in empty:
            s["dir"] = s["dir"].with_name(s["dir"].name + "_retry_1")
            s["context"] += " retry 1"
            retry_jobs.append((s["dir"], self._docs_for(s["dir"], s["role"], s["lens"], round_index, standing,
                                                        incumbent, briefs.retry_note())))
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
                                                   briefs.repair_note(reason))))
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
        return self.score(round_index, events)

    # ---------- end of the run ----------
    def end(self) -> dict:
        standing = self.standing()
        trusted = [n for n, s in standing.items() if s["loo_reliable"] and s["converged"]]
        events = []
        if trusted:
            best = max(trusted, key=lambda n: standing[n]["elpd_loo"])
            pw = {n: np.asarray(self.live[n].fit.loo().loo.loo_i).ravel() for n in trusted}
            for n in trusted:
                if n == best:
                    continue
                diff = float(pw[best].sum() - pw[n].sum())
                dse = cluster_dse(pw[best], pw[n], self.clusters)
                if diff > self.cfg.prune_dse_multiplier * dse:
                    self._retire(n, "pruned", f"elpd_diff {diff:.1f} > {self.cfg.prune_dse_multiplier} x clustered dse {dse:.1f} vs {best}")
                    events.append(dict(name=n, outcome="pruned", reason=f"{diff:.1f} nats behind {best} (clustered dse {dse:.1f})"))
        while len(self.live) > self.cfg.max_live_models:
            standing = self.standing()
            keep = self.incumbent(standing)
            order = sorted((n for n in standing if n != keep),
                           key=lambda n: (standing[n]["loo_reliable"] and standing[n]["converged"], standing[n]["elpd_loo"]))
            self._retire(order[0], "pruned", f"retired by the cap of {self.cfg.max_live_models} live models")
            events.append(dict(name=order[0], outcome="retired", reason="live-set cap"))
        entry = self.score(self.cfg.max_iterations, events)
        best = entry["best_model"]
        shutil.copyfile(self.models_dir / f"{best}.py", self.dir / "best_model.py")
        (self.dir / "export.json").write_text(json.dumps(dict(best_model=best, live=sorted(self.live))))
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
        self._prefit([(self.models_dir / f"{n}.py", n) for n in live])
        for name, e in live.items():
            path = self.models_dir / f"{name}.py"
            model = RSAModel(path, name=name)
            fitted = fit_with_refit(path, name, self.gate_cfg)
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
        for r in range(first, self.cfg.max_iterations):
            self.run_round(r)
        return self.end()


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
                         model: Optional[str], agent_root: Optional[Path], sandbox: bool) -> SpawnFn:
    """The real spawner: one coding agent per candidate directory."""
    from src.runtime.coding_agent import run_coding_agent
    from src.runtime.config import REPO_ROOT

    def spawn(candidate_dir: Path, prompt: str) -> bool:
        ok, _ = run_coding_agent(
            prompt, cwd=agent_root or REPO_ROOT, log_path=candidate_dir / "agent.jsonl",
            allowed_dirs=[candidate_dir, models_dir, responses_path.parent], writable_dirs=[candidate_dir],
            timeout_secs=timeout_sec, backend=backend, model=model, usage_label="rsa:candidate",
            stock=True, sandbox=sandbox,
        )
        return ok

    return spawn
