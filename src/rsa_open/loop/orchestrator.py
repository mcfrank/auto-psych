"""The open inner loop: theorists propose theories; every paradigm of the chain decides
(docs/auto_open/PLAN.md §5; the closed loop's `src.rsa.loop.orchestrator`, for theories).

Layout under ``results_dir``::

    corpus.json                 the blocks the loop fits (`src.rsa_open.loop.fitting.write_corpus`)
    data/<block>.jsonl          the same trials, for the theorists to analyse
    models/                     the live set: <name>.py + models_manifest.yaml
    models/pruned/              theories pruned or retired at the end
    attempted_hypotheses.jsonl  the ledger (every slot, admitted or not)
    .cv/fold_<f>.json           each fold's training corpus (every block without its fold-f trials)
    .fit_cache/                 content-addressed corpus fits
    round_<k>/candidate_<i>[_retry_1|_repair_1]/   agent working directories
    history.json                one entry per scoring step
    best_model.py, export.json  the exported incumbent and the live set

Each round, as the closed loop's: every slot's agent concurrently (exploratory slots walk the
lenses; two refine the incumbent; one refines a theory of its choice); a slot that wrote no
candidate.py is re-spawned once; names, then every candidate's fit at once, then admission one
by one (`src.rsa_open.loop.gates`); a refused candidate is re-spawned once with the reason;
then scoring. The run stops early after ``stop_after_stale_rounds`` rounds that leave the
incumbent unchanged.

**Selection** is grouped CV within every block (`src.rsa_open.corpus`): a theory is ranked by
its out-of-fold lpd on the newest block, and is **eligible** only while, on every earlier
block (E included), it is within ``guard_k`` clustered SEs of that block's best. The
incumbent is the best eligible theory whose fit and CV converged. At the end, ineligible
theories are pruned, then every theory more than ``prune_dse_multiplier`` clustered SEs behind
the incumbent on the newest block, and the live set is capped at ``max_live_models``.

Not yet here (the closed loop has them): the critique step (it needs each link's simulated
replicates) and resuming an interrupted run.
"""

from __future__ import annotations

import json
import math
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional

import numpy as np
import yaml

from src.models.model_manifest import read_manifest_entries
from src.pipelines.inner_loop.hypothesis_ledger import LEDGER_FILENAME, HypothesisLedger, LedgerEntry, collapse_whitespace
from src.pipelines.inner_loop.model_zoo import (SLOT_EXPLORE, SLOT_REFINE_INCUMBENT, _lens_index,
                                                exploratory_slots_per_round, slot_roles)
from src.rsa.fit import FitSettings
from src.rsa.loop import brief as closed_brief
from src.rsa.loop.code_gate import code_problems
from src.rsa.loop.fitting import FIT_TIME_LIMIT_SEC, ModelFailure
from src.rsa.loop.gates import read_candidate
from src.rsa.loop.novelty import DEFAULT_NOVELTY_RMSE_THRESHOLD, plain_pool
from src.rsa.loop.orchestrator import NAME_RE, _read, default_fit_workers, prune_margin, stale_rounds
from src.rsa_open.corpus import DEFAULT_FOLDS, GUARD_K, CorpusCV, CorpusFit, assign_folds, heldout_lpd, standing
from src.rsa_open.fit import TheoryModel
from src.rsa_open.loop import brief as briefs
from src.rsa_open.loop import critique as critique_mod
from src.rsa_open.loop.critique import CRITIQUE_N_PROPOSALS, CRITIQUE_PPC_REPLICATES, CRITIQUE_SIGNIFICANCE_ALPHA
from src.rsa_open.loop.fitting import loop_fit, read_corpus, write_corpus
from src.rsa_open.loop.gates import GateConfig, admit, pool_predictions

SpawnFn = Callable[[Path, str], bool]
DEFAULT_PRUNE_DSE_MULTIPLIER = 2.0
MAX_LIVE_MODELS = 8  # the closed campaign's carry-forward cap
ADMISSION_FACTOR = 2.0
"""A candidate's admission fit is limited at ``ADMISSION_FACTOR`` times the median full-fit time of
the theories the loop starts from (its seeds or carried theories, fitted at setup), and never below
``fit_time_limit_sec`` (PI 2026-10-24). A fixed 30 minutes was the closed campaign's, set for its
smaller corpora: on all of E, 12 of the 15 bar theories take longer than that themselves (the
temperature audit: `rsa_l1` 26 min, `rsa_l2` 46, the starting incumbent 83), and run B rejected 43
of 61 candidates as too slow, refinements of the incumbent above all. The limit is fixed for the
loop (it does not ratchet up as slow candidates are admitted) and is recorded in history.json, the
theorists' brief and their self-check command."""
IN_PLAY_FIT_LIMIT_SEC = 4 * 3600
"""The ceiling on every fit the loop runs (PI 2026-10-28): a theory in play (a seed, a carried or an
admitted theory) whose fit does not finish within it is dropped on record, as a seed that fails to
fit is; no candidate's admission limit and no fold's limit exceeds it. Run C's starting fits took
8-10 h on single blocks and cost a 24-h job; the slowest healthy fit seen is 2.7 h, on all of E
(run B′)."""
FOLD_TIME_FACTOR = 3.0
FOLD_MIN_SEC = 10 * 60
"""Each of a theory's cross-validation folds is time-limited: at
``FOLD_TIME_FACTOR`` times its own full fit's time (at least ``FOLD_MIN_SEC``; the admission limit
stands in for a full fit recorded without its time). A warm-started fold runs 800 iterations to
the full fit's 2,000, so a fold past the bound is stuck, not slow: run B's fifth fold of one
theory sampled 5 h 50 min to R-hat 1.55 while the round waited (PI 2026-10-23). A fold that fails
leaves the theory unscored and untrusted, with the reason in its standing (``cv_problem``)."""


@dataclass
class OpenLoopConfig:
    corpus_path: Path
    seed_models_dir: Path
    results_dir: Path
    max_iterations: int = 5
    candidate_count: int = 6
    settings: FitSettings = field(default_factory=lambda: FitSettings(dense_mass=True))
    novelty_threshold: float = DEFAULT_NOVELTY_RMSE_THRESHOLD
    cv_folds: int = DEFAULT_FOLDS
    guard_k: float = GUARD_K
    prune_dse_multiplier: float = DEFAULT_PRUNE_DSE_MULTIPLIER
    max_live_models: int = MAX_LIVE_MODELS
    fit_time_limit_sec: Optional[float] = FIT_TIME_LIMIT_SEC
    in_play_fit_limit_sec: float = IN_PLAY_FIT_LIMIT_SEC
    """`IN_PLAY_FIT_LIMIT_SEC` (with ``fit_time_limit_sec`` None nothing is limited)."""
    theorist_timeout_sec: Optional[float] = None
    """The theorists' session limit, stated in their brief and their self-check (None: not stated)."""
    lenses: List[str] = field(default_factory=lambda: list(briefs.LENSES))
    paradigm_notes: Dict[str, str] = field(default_factory=dict)
    """Per block, a line on what people did (an experiment's paradigm, in its own words)."""
    inherit_ledger: Optional[Path] = None
    agent_network: bool = False
    fit_workers: int = 0
    stop_after_stale_rounds: int = 2
    cv_warmup: int = 300
    cv_samples: int = 500
    """A cross-validation fold starts at its theory's full fit (posterior medians) and runs
    ``cv_warmup`` + ``cv_samples`` iterations per chain instead of the full fit's 1,000 + 1,000: the
    folds are 5/6 of the loop's fitting, and a held-out score needs predictions, not a fresh
    posterior exploration (PI 2026-10-22)."""
    critique: bool = True
    """Critique the incumbent on the newest block before every round (`src.rsa_open.loop.critique`,
    the closed loop's critic); its critiques.md goes into every theorist's brief."""
    n_critique_proposals: int = CRITIQUE_N_PROPOSALS
    critique_alpha: float = CRITIQUE_SIGNIFICANCE_ALPHA
    n_critique_replicates: int = CRITIQUE_PPC_REPLICATES


@dataclass
class Live:
    name: str
    hypothesis: str
    theory: TheoryModel
    fit: CorpusFit
    pool_preds: np.ndarray
    source: str


class OpenLoop:
    def __init__(self, cfg: OpenLoopConfig, spawn: SpawnFn) -> None:
        self.cfg = cfg
        self.spawn = spawn
        self.dir = Path(cfg.results_dir)
        self.models_dir = self.dir / "models"
        self.corpus = self.dir / "corpus.json"
        self.live: Dict[str, Live] = {}
        self.cv: Dict[str, Optional[CorpusCV]] = {}
        self.cv_problems: Dict[str, str] = {}
        self.admission_limit_sec: Optional[float] = None
        self.history: List[dict] = []
        self.step = 0

    # ---------- setup ----------
    def _prepare(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        self.blocks = read_corpus(self.cfg.corpus_path)
        write_corpus(self.corpus, self.blocks)
        self.data_paths = briefs.write_block_data(self.dir / "data", self.blocks)
        # The theorists' copy, for their self-check: they read data/ and models/, never the loop's own files.
        self.agent_corpus = write_corpus(self.dir / "data" / "corpus.json", self.blocks)
        # The self-check's corpus: large blocks on a fixed fraction of their displays (brief.self_check_blocks).
        self.check_corpus = write_corpus(self.dir / "data" / "check_corpus.json", briefs.self_check_blocks(self.blocks))
        self.pool = plain_pool()
        self.folds = {b.name: assign_folds(b, self.cfg.cv_folds, seed=self.cfg.settings.seed) for b in self.blocks}
        self.fold_paths = [
            write_corpus(self.dir / ".cv" / f"fold_{f}.json",
                         [b.subset(np.where(self.folds[b.name] != f)[0]) for b in self.blocks])
            for f in range(self.cfg.cv_folds)]
        self.gate_cfg = GateConfig(self.corpus, self.dir / ".fit_cache", self.cfg.settings, self.cfg.novelty_threshold,
                                   self.cfg.fit_time_limit_sec)
        # Theories in play (seeds, carried, admitted) are limited by the loop's ceiling only.
        self.live_cfg = replace(self.gate_cfg, time_limit_sec=None if self.cfg.fit_time_limit_sec is None
                                else self.cfg.in_play_fit_limit_sec)

    def setup(self) -> None:
        self._prepare()
        if self.cfg.inherit_ledger is not None and not Path(self.cfg.inherit_ledger).exists():
            raise FileNotFoundError(f"inherit_ledger {self.cfg.inherit_ledger} does not exist")
        self.ledger = HypothesisLedger.create(self.dir / LEDGER_FILENAME, inherit_from=self.cfg.inherit_ledger)
        self.models_dir.mkdir(exist_ok=True)
        entries = read_manifest_entries(self.cfg.seed_models_dir)
        for e in entries:
            shutil.copyfile(Path(self.cfg.seed_models_dir) / f"{e['name']}.py", self.models_dir / f"{e['name']}.py")
        self._prefit([(self.models_dir / f"{e['name']}.py", e["name"]) for e in entries], self.live_cfg)
        for e in entries:
            name, hypothesis = e["name"], e.get("rationale", "")
            path = self.models_dir / f"{name}.py"
            theory = TheoryModel(path, name=name)
            try:
                fitted = self._fit(path, name, self.live_cfg)
            except ModelFailure as exc:
                path.unlink()
                print(f"[open loop] dropped {name}: its fit failed: {exc}", flush=True)
                self._ledger(name, "dropped", f"seed failed to fit: {exc}", hypothesis, "seed")
                continue
            self.live[name] = Live(name, hypothesis, theory, fitted, pool_predictions(theory, fitted, self.pool), "seed")
        if not self.live:
            raise RuntimeError("no seed theory could be fitted; nothing to compare")
        self._set_admission_limit()
        self._write_manifest()

    def _set_admission_limit(self) -> None:
        """`ADMISSION_FACTOR` x the median full-fit time of the starting theories (None: unlimited)."""
        if self.cfg.fit_time_limit_sec is None:
            return
        timed = {n: m.fit.timing["warmup_sec"] + m.fit.timing["sampling_sec"] for n, m in self.live.items()
                 if m.fit.timing}
        if not timed:
            raise RuntimeError("no starting theory's fit records its time (fits cached by code before "
                               "2026-10-23?), so the admission limit cannot be set; start in a fresh work root")
        self.admission_limit_sec = min(self.cfg.in_play_fit_limit_sec, max(
            self.cfg.fit_time_limit_sec, ADMISSION_FACTOR * float(np.median(list(timed.values())))))
        self.gate_cfg = replace(self.gate_cfg, time_limit_sec=self.admission_limit_sec)
        print(f"[open loop] admission limit {self.admission_limit_sec / 60:.0f} min: {ADMISSION_FACTOR:g} x the median "
              f"of the starting fits ({', '.join(f'{n} {t / 60:.0f}' for n, t in sorted(timed.items()))} min)", flush=True)

    def _fit(self, path: Path, name: str, cfg: GateConfig) -> CorpusFit:
        return loop_fit(path, name, cfg.corpus_path, cfg.settings, cfg.cache_dir, time_limit_sec=cfg.time_limit_sec)

    def _prefit(self, items, cfg: Optional[GateConfig] = None) -> None:
        """Fit concurrently (cached, failures remembered) so sequential admission reads finished fits."""
        cfg = cfg or self.gate_cfg

        def one(item) -> None:
            path, name = item
            if path.name == "candidate.py" and read_candidate(path.parent) is not None:
                return
            if code_problems(path.read_text(encoding="utf-8")):
                return
            try:
                self._fit(path, name, cfg)
            except ModelFailure:
                pass  # remembered; admission reports it
            except Exception:
                if path.name == "candidate.py":
                    return  # a file the gates will refuse (not a theory, a memo error): admission says why
                raise

        if not items:
            return
        workers = 1 if cfg.time_limit_sec is None else min(len(items), self.cfg.fit_workers or default_fit_workers())
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(one, items))

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
    def fold_time_limit(self, name: str) -> Optional[float]:
        """The bound on each of ``name``'s fold fits (`FOLD_TIME_FACTOR`); None when nothing is limited."""
        if self.live_cfg.time_limit_sec is None:
            return None
        t = self.live[name].fit.timing
        full = t["warmup_sec"] + t["sampling_sec"] if t else self.cfg.fit_time_limit_sec
        return min(self.cfg.in_play_fit_limit_sec, max(FOLD_MIN_SEC, FOLD_TIME_FACTOR * full))

    def _cv_one(self, name: str) -> Optional[CorpusCV]:
        path = self.models_dir / f"{name}.py"
        theory = self.live[name].theory
        limit = self.fold_time_limit(name)

        full = self.live[name].fit.flat()
        start = {k: float(np.median(v)) for k, v in full.items()}
        fold_settings = replace(self.cfg.settings, num_warmup=self.cfg.cv_warmup, num_samples=self.cfg.cv_samples)

        def fold_fit(f: int) -> CorpusFit:
            return loop_fit(path, name, self.fold_paths[f], fold_settings, self.dir / ".fit_cache",
                            time_limit_sec=limit, init=start)

        workers = self.cfg.cv_folds if self.live_cfg.time_limit_sec is not None else 1
        try:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                fits = list(pool.map(fold_fit, range(self.cfg.cv_folds)))
        except ModelFailure as exc:
            self.cv_problems[name] = f"a cross-validation fold failed, so it is unscored: {exc}"
            return None
        bad = [f"fold {f + 1}: " + "; ".join(fit.convergence_problems) for f, fit in enumerate(fits) if not fit.converged]
        if bad:
            self.cv_problems[name] = "cross-validation folds did not converge, so it is untrusted: " + " | ".join(bad)
        out = {b.name: np.empty(b.n) for b in self.blocks}
        for f, fitted in enumerate(fits):
            for b in self.blocks:
                rows = np.where(self.folds[b.name] == f)[0]
                if len(rows):
                    out[b.name][rows] = heldout_lpd(theory, fitted, b.subset(rows))
        return CorpusCV(out, all(fit.converged for fit in fits))

    def standing(self) -> Dict[str, dict]:
        todo = [n for n in self.live if n not in self.cv]
        workers = min(len(todo), max(1, (self.cfg.fit_workers or default_fit_workers()) // self.cfg.cv_folds)) or 1
        if self.live_cfg.time_limit_sec is None:
            workers = 1
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for name, result in zip(todo, pool.map(self._cv_one, todo)):
                self.cv[name] = result
        scored = {n: self.cv[n] for n in self.live if self.cv.get(n) is not None}
        out = standing(scored, self.blocks, guard_k=self.cfg.guard_k) if scored else {}
        for n in self.live:
            if n not in out:  # a fold fit failed: no CV, never selected
                out[n] = dict(sel_elpd=-math.inf, sel_diff=math.inf, sel_dse=0.0, eligible=False, guard_failures=[],
                              by_block={}, cv_converged=False)
            out[n]["trusted"] = bool(self.live[n].fit.converged and out[n]["cv_converged"])
            if n in self.cv_problems:
                out[n]["cv_problem"] = self.cv_problems[n]
        return out

    def incumbent(self, s: Mapping[str, dict]) -> str:
        for pool in ([n for n in s if s[n]["trusted"] and s[n]["eligible"]], [n for n in s if s[n]["trusted"]], list(s)):
            if pool:
                return max(pool, key=lambda n: s[n]["sel_elpd"])
        raise RuntimeError("no theories")

    def score(self, round_index: int, events: List[dict], critique: Optional[dict] = None) -> dict:
        s = self.standing()
        best = self.incumbent(s)
        entry = dict(step=self.step, round=round_index, best_model=best, events=events, critique=critique,
                     admission_limit_sec=self.admission_limit_sec,
                     standing={n: {k: v for k, v in d.items()} for n, d in s.items()},
                     live=[dict(name=m.name, hypothesis=m.hypothesis, source=m.source) for m in self.live.values()],
                     ledger_entries=len(self.ledger.entries()))
        self.history.append(entry)
        (self.dir / "history.json").write_text(json.dumps(self.history, indent=1, default=float))
        self.step += 1
        return entry

    # ---------- a round ----------
    def _describe(self, name: str, s: Mapping[str, dict]) -> str:
        if name not in s:
            return "admitted this round, not yet scored"
        d, newest = s[name], self.blocks[-1].name
        if math.isinf(d["sel_elpd"]):
            return d.get("cv_problem", "not scored")
        text = ("best" if d["sel_diff"] == 0 else f"{d['sel_diff']:.1f} ± {d['sel_dse']:.1f} nats behind the best") + \
            f" on {newest} (held out, grouped CV)"
        parts = [("best" if v["behind"] == 0 else f"{v['behind']:.1f} ± {v['dse']:.1f} behind") + f" on {b}"
                 for b, v in d["by_block"].items() if b != newest]
        if parts:
            text += "; " + ", ".join(parts)
        if d.get("cv_problem"):
            text += f"; {d['cv_problem']}"
        if d["guard_failures"]:
            text += (f"; NOT eligible to lead: more than {self.cfg.guard_k:g} SEs behind the best on "
                     + ", ".join(d["guard_failures"]))
        return text

    def _docs_for(self, cdir: Path, role: str, lens: Optional[str], round_index: int, s: Mapping[str, dict],
                  incumbent: str, note: Optional[str], critiques: Optional[str] = None) -> str:
        def behind(n):
            return s[n]["sel_diff"] if n in s else float("inf")
        live = [closed_brief.ZooModel(n, self.live[n].hypothesis, self.models_dir / f"{n}.py", self._describe(n, s))
                for n in sorted(self.live, key=behind)]
        pruned_dir = self.models_dir / "pruned"
        pruned = [closed_brief.ZooModel(e.name, e.hypothesis, pruned_dir / f"{e.name}.py", f"pruned: {e.detail}")
                  for e in sorted(self.ledger.pruned(self.live), key=lambda e: prune_margin(e.detail))
                  if (pruned_dir / f"{e.name}.py").exists()]
        context = briefs.context_md(candidate_dir=cdir, corpus_path=self.check_corpus,
                                    full_corpus_path=self.agent_corpus, admission_limit_sec=self.admission_limit_sec,
                                    blocks=self.blocks,
                                    data_paths=self.data_paths, paradigm_notes=self.cfg.paradigm_notes,
                                    round_index=round_index, n_rounds=self.cfg.max_iterations,
                                    network=self.cfg.agent_network,
                                    session_minutes=(None if self.cfg.theorist_timeout_sec is None
                                                     else self.cfg.theorist_timeout_sec / 60))
        if self.cfg.theorist_timeout_sec is not None:
            briefs.write_deadline(cdir, self.cfg.theorist_timeout_sec)
        docs = briefs.write_docs(cdir, role=role, lens=lens, context=context, live=live, pruned=pruned,
                                 incumbent=incumbent, ledger=self.ledger, attempt_note=note, critiques=critiques)
        return briefs.build_prompt(cdir, docs)

    def _critique(self, rdir: Path, incumbent: str) -> "critique_mod.CritiqueOutcome":
        """The round's critique of the incumbent on the newest block, or a disabled record. A
        critique that fails for any reason but the agents' own infrastructure is recorded and the
        round runs without one (the closed loop's rule: a critique failure must not end a run)."""
        if not self.cfg.critique:
            return critique_mod.CritiqueOutcome(None, critique_mod.disabled_status())
        from src.runtime.coding_agent import AgentPermissionDenied
        from src.runtime.usage_limits import AgentInfrastructureError

        m, newest = self.live[incumbent], self.blocks[-1]
        try:
            return critique_mod.run_critique(
                rdir, spawn=self.spawn, incumbent=incumbent, theory=m.theory, fitted=m.fit, hypothesis=m.hypothesis,
                incumbent_file=self.models_dir / f"{incumbent}.py", block=newest,
                paradigm_note=self.cfg.paradigm_notes.get(newest.name, ""), data_path=self.data_paths[newest.name],
                n_proposals=self.cfg.n_critique_proposals, alpha=self.cfg.critique_alpha,
                n_replicates=self.cfg.n_critique_replicates, seed=self.cfg.settings.seed + self.step)
        except (AgentPermissionDenied, AgentInfrastructureError):
            raise
        except Exception as exc:  # recorded, never swallowed silently
            reason = f"{type(exc).__name__}: {exc}"
            print(f"  [critique] NO CRITIQUE this round: {reason}", flush=True)
            return critique_mod.CritiqueOutcome(None, dict(
                status=critique_mod.CRITIQUE_STATUS_NONE, incumbent=incumbent, reason=reason))

    def _spawn_all(self, jobs) -> List[bool]:
        if not jobs:
            return []
        with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
            return list(pool.map(lambda j: self.spawn(*j), jobs))

    def _same_hypothesis(self, hypothesis: str) -> Optional[str]:
        key = collapse_whitespace(hypothesis).strip().lower()
        if not key:
            return None
        texts = [(m.name, m.hypothesis) for m in self.live.values()]
        texts += [(e.name, e.hypothesis) for e in self.ledger.entries() if e.outcome == "admitted"]
        return next((n for n, t in texts if key in collapse_whitespace(t).strip().lower()), None)

    def _try_admit(self, cdir: Path, name: str, context: str) -> tuple[bool, str]:
        hypothesis = _read(cdir / "hypothesis.md")
        same = self._same_hypothesis(hypothesis)
        if same is not None:
            reason = (f"hypothesis.md is word for word {same}'s hypothesis, or part of it. State this candidate's own "
                      "mechanism, including what it changes from the theory it refines.")
            self._ledger(name, "rejected", reason, hypothesis, context)
            return False, reason
        out = admit(cdir, name, cfg=self.gate_cfg, blocks=self.blocks, pool=self.pool,
                    admitted_preds={n: m.pool_preds for n, m in self.live.items()})
        if out.admitted:
            dst = self.models_dir / f"{name}.py"
            shutil.copyfile(cdir / "candidate.py", dst)
            self.live[name] = Live(name, hypothesis, TheoryModel(dst, name=name), out.fit, out.pool_preds, context)
            self._write_manifest()
            self._ledger(name, "admitted", "admitted", hypothesis, context)
        else:
            self._ledger(name, "rejected", out.reason, hypothesis, context)
        return out.admitted, out.reason

    def _admit_all(self, slots, key: str, round_index: int, context_suffix: str = "") -> List[tuple]:
        """Name every slot with a candidate in ``slots[key]``, fit them at once, admit one by one."""
        taken: set = set()
        for s in slots:
            if (s[key] / "candidate.py").exists():
                s[key + "_name"] = self._name_for(s[key], f"r{round_index + 1}_c{s['i'] + 1}", frozenset(taken))
                taken.add(s[key + "_name"])
        self._prefit([(s[key] / "candidate.py", s[key + "_name"]) for s in slots if key + "_name" in s])
        results = []
        for s in slots:
            if key + "_name" not in s:
                continue
            ok, reason = self._try_admit(s[key], s[key + "_name"], s["context"] + context_suffix)
            results.append((s, ok, reason))
        return results

    def run_round(self, round_index: int) -> dict:
        s = self.standing()
        incumbent = self.incumbent(s)
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
            context = f"round {round_index + 1} candidate {i + 1} {role}" + (
                f" {incumbent}" if role == SLOT_REFINE_INCUMBENT else "")
            slots.append(dict(i=i, role=role, lens=lens, dir=rdir / f"candidate_{i + 1}", context=context))
        self._spawn_all([(sl["dir"], self._docs_for(sl["dir"], sl["role"], sl["lens"], round_index, s, incumbent, None,
                                                    critiques)) for sl in slots])
        retry = []
        for sl in slots:
            if not (sl["dir"] / "candidate.py").exists():
                sl["dir"] = sl["dir"].with_name(sl["dir"].name + "_retry_1")
                sl["context"] += " retry 1"
                retry.append((sl["dir"], self._docs_for(sl["dir"], sl["role"], sl["lens"], round_index, s, incumbent,
                                                        briefs.retry_note(), critiques)))
        self._spawn_all(retry)
        events = []
        for sl in slots:
            if not (sl["dir"] / "candidate.py").exists():
                self._ledger(f"slot_{sl['i'] + 1}", "rejected", "the agent wrote no candidate.py", "", sl["context"])
                events.append(dict(slot=sl["i"] + 1, role=sl["role"], name=None, outcome="no file", reason=""))
        repairs = []
        for sl, ok, reason in self._admit_all(slots, "dir", round_index):
            events.append(dict(slot=sl["i"] + 1, role=sl["role"], name=sl["dir_name"],
                               outcome="admitted" if ok else "rejected", reason=reason,
                               hypothesis=_read(sl["dir"] / "hypothesis.md")))
            if not ok:
                repairs.append((sl, reason))
        jobs = []
        for sl, reason in repairs:
            rd = sl["dir"].with_name(sl["dir"].name + "_repair_1")
            rd.mkdir(parents=True, exist_ok=True)
            for f in ("candidate.py", "hypothesis.md", "model_name.txt"):
                if (sl["dir"] / f).exists():
                    shutil.copyfile(sl["dir"] / f, rd / f)
            sl["repair"] = rd
            jobs.append((rd, self._docs_for(rd, sl["role"], sl["lens"], round_index, s, incumbent,
                                            briefs.repair_note(reason), critiques)))
        self._spawn_all(jobs)
        for sl, ok, reason in self._admit_all([sl for sl, _ in repairs], "repair", round_index, " repair 1"):
            events.append(dict(slot=sl["i"] + 1, role=sl["role"], name=sl["repair_name"],
                               outcome="admitted" if ok else "rejected", reason=reason,
                               hypothesis=_read(sl["repair"] / "hypothesis.md"), repair=True))
        return self.score(round_index, events, crit.status)

    # ---------- the end ----------
    def _retire(self, name: str, outcome: str, detail: str) -> None:
        m = self.live.pop(name)
        pruned = self.models_dir / "pruned"
        pruned.mkdir(exist_ok=True)
        shutil.move(str(self.models_dir / f"{name}.py"), pruned / f"{name}.py")
        self._ledger(name, outcome, detail, m.hypothesis, m.source)
        self._write_manifest()

    def references(self, s: Mapping[str, dict]) -> Dict[str, str]:
        """Each earlier block's best theory (block -> name): kept as the 4-SE guard's reference.
        Never pruned, nor retired by the cap, while it is best there (PI 2026-10-26: run B′ pruned
        E's best theory, about 110 nats ahead on E, on the newest block alone, and the guard then
        compared only survivors). It may still be ineligible to lead."""
        out = {}
        for b in self.blocks[:-1]:
            best = [n for n, d in s.items() if d["by_block"].get(b.name, {}).get("behind") == 0]
            if best:
                out[b.name] = best[0]
        return out

    def end(self, stopped: Optional[str] = None) -> dict:
        from src.models.clustered_se import cluster_dse

        s = self.standing()
        events = []
        newest = self.blocks[-1]
        # An untrusted theory (its fit or its folds did not converge, or ran out of time) is not carried:
        # it cannot be judged, and its refit on more data only gets slower (PI 2026-10-28; run C carried
        # three and refitted them for 8 h and more). The incumbent stays even so, when no theory is trusted.
        lead = self.incumbent(s)
        for n in [n for n in s if not s[n]["trusted"] and n != lead]:
            why = s[n].get("cv_problem") or ("its fit did not converge: "
                                             + "; ".join(self.live[n].fit.convergence_problems or ["untrusted"]))
            self._retire(n, "pruned", f"untrusted, not carried: {why}")
            events.append(dict(name=n, outcome="pruned", reason="untrusted, not carried"))
        s = self.standing()  # re-ranked among the rest: a dropped theory is no block's reference
        kept = self.references(s)
        protected = set(kept.values())
        for b, n in kept.items():
            events.append(dict(name=n, outcome="kept", reason=f"the best theory on {b}, the guard's reference there"))
        for n, d in s.items():
            if d["guard_failures"] and n not in protected:
                why = ", ".join(f"{b}: {d['by_block'][b]['behind']:.1f} ± {d['by_block'][b]['dse']:.1f}"
                                for b in d["guard_failures"])
                self._retire(n, "pruned", f"ineligible: more than {self.cfg.guard_k:g} SEs behind the best on {why}")
                events.append(dict(name=n, outcome="pruned", reason=f"ineligible ({why})"))
        s = {n: d for n, d in s.items() if n in self.live}
        best = self.incumbent(s)
        units = np.unique(np.asarray(newest.units), return_inverse=True)[1]
        for n in [n for n in s if n != best and s[n]["trusted"] and n not in protected]:
            a, b = self.cv[best].pointwise[newest.name], self.cv[n].pointwise[newest.name]
            diff, dse = float(a.sum() - b.sum()), float(cluster_dse(a, b, units))
            if diff > self.cfg.prune_dse_multiplier * dse:
                self._retire(n, "pruned", f"elpd_diff {diff:.1f} > {self.cfg.prune_dse_multiplier} x clustered dse "
                                          f"{dse:.1f} vs {best} on {newest.name} (grouped CV)")
                events.append(dict(name=n, outcome="pruned", reason=f"{diff:.1f} nats behind {best}"))
        while len(self.live) > self.cfg.max_live_models:
            s = self.standing()
            keep = self.incumbent(s)
            droppable = sorted((n for n in s if n != keep and n not in protected),
                               key=lambda n: (s[n]["trusted"], s[n]["sel_elpd"]))
            if not droppable:  # the incumbent and the references alone exceed the cap
                events.append(dict(name="loop", outcome="over cap",
                                   reason=f"{len(self.live)} live theories: the incumbent and the references"))
                break
            drop = droppable[0]
            self._retire(drop, "pruned", f"retired by the cap of {self.cfg.max_live_models} live theories")
            events.append(dict(name=drop, outcome="retired", reason="live-set cap"))
        if stopped:
            events.append(dict(name="loop", outcome="stopped", reason=stopped))
        entry = self.score(self.cfg.max_iterations, events)
        shutil.copyfile(self.models_dir / f"{entry['best_model']}.py", self.dir / "best_model.py")
        export = dict(best_model=entry["best_model"], live=sorted(self.live))
        if stopped:
            export["stopped_early"] = stopped
        (self.dir / "export.json").write_text(json.dumps(export))
        return entry

    def run(self) -> dict:
        self.setup()
        self.score(-1, [dict(outcome="seeded", name=n) for n in self.live])
        patience = self.cfg.stop_after_stale_rounds
        for r in range(self.cfg.max_iterations):
            self.run_round(r)
            if patience and r + 1 < self.cfg.max_iterations and stale_rounds(self.history) >= patience:
                why = (f"stopped after round {r + 1} of {self.cfg.max_iterations}: {patience} rounds in a row left "
                       f"the incumbent ({self.history[-1]['best_model']}) unchanged")
                return self.end(stopped=why)
        return self.end()
