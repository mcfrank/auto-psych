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
    newest_block_cv.json        every scored theory's out-of-fold lpd on the newest block, and the
                                chance theory's (`CHANCE_THEORY`): the feasibility gate's item 3
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

**Scheduling** (PI 2026-11-02, after open chain 1's experiment 2 lost its loop twice to the
job's memory): every fit child waits for room in the job's memory (`src.rsa_open.loop.fitting`,
``FitScheduler``). The candidates that have files are fitted while the slots that wrote none are
re-spawned, and an admitted theory's CV folds start as soon as it is admitted, while repairs
run; admission is still one by one in slot order, on finished fits, so the order of decisions is
the sequential loop's.

**Resuming** (`resume`, the closed loop's): a loop interrupted after a scored step (a crash, a
lost node, the job's end) restarts from that step: the ledger is cut back to the step's length,
theories the unfinished round admitted leave the set, and that round runs again from its agents
(its directory kept as ``round_<k>_abandoned_<i>``). Fits come back from the cache. With
``deadline`` a round is started only with ``round_min_hours`` of the job left (`LoopOutOfTime`).
"""

from __future__ import annotations

import json
import math
import shutil
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
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
from src.rsa.loop.orchestrator import NAME_RE, _read, _write_atomic, default_fit_workers, prune_margin, stale_rounds
from src.rsa_open.corpus import DEFAULT_FOLDS, GUARD_K, CorpusCV, CorpusFit, assign_folds, heldout_lpd, standing
from src.rsa_open.fit import TheoryModel
from src.rsa_open.loop import brief as briefs
from src.rsa_open.loop import critique as critique_mod
from src.rsa_open.loop.critique import CRITIQUE_N_PROPOSALS, CRITIQUE_PPC_REPLICATES, CRITIQUE_SIGNIFICANCE_ALPHA
from src.rsa_open.loop import slurm as step
from src.rsa_open.loop.fit_array import FitPending, FitsPending, FoldAfter, collect_pending
from src.rsa_open.loop.fitting import FitSummary, fit_backend, loop_fit, loop_fit_summary, read_corpus, write_corpus
from src.rsa_open.loop.gates import MAX_POOL_DRAWS, GateConfig, admit, magic_number_problems, pool_predictions
from src.rsa_open.loop.scores import ScoreStore

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
IMPOSSIBLE_LPD = 1e-3
"""A held-out log probability above this (float32 rounding aside) is impossible for any link."""
FOLD_TIME_FACTOR = 3.0
FOLD_MIN_SEC = 10 * 60
"""Each of a theory's cross-validation folds is time-limited: at
``FOLD_TIME_FACTOR`` times its own full fit's time (at least ``FOLD_MIN_SEC``; the admission limit
stands in for a full fit recorded without its time). A warm-started fold runs 800 iterations to
the full fit's 2,000, so a fold past the bound is stuck, not slow: run B's fifth fold of one
theory sampled 5 h 50 min to R-hat 1.55 while the round waited (PI 2026-10-23). A fold that fails
leaves the theory unscored and untrusted, with the reason in its standing (``cv_problem``)."""
CV_FOLD_FAILED = "a cross-validation fold failed, so it is unscored"
ROUND_MIN_HOURS = 7.0
"""A round is started only with this much of the job left (``deadline``): chain 1's rounds took
5-6.5 h; one killed part way is run again from its agents by the next job."""
CV_THEORIES_AT_ONCE = 8
"""Theories whose CV folds are in flight at once (each runs its folds together); the scheduler
decides how many fit children actually run."""


class LoopOutOfTime(RuntimeError):
    """Too little of the job is left for another round: the next job resumes the loop."""


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
    deadline: Optional[float] = None
    """When the job ends (epoch seconds): a round starts only with ``round_min_hours`` left."""
    round_min_hours: float = ROUND_MIN_HOURS


@dataclass
class Live:
    """A theory in play. Its fit stays in the cache: `fit` loads it when a step needs its draws (its
    folds' starting point, the critique), and `summary` answers the rest. A live set used to hold
    every fit, about 0.9 GB each (19.3 GB at 21 theories after a resume on chain 1; A1)."""
    name: str
    hypothesis: str
    theory: TheoryModel
    summary: FitSummary
    pool_preds: np.ndarray
    source: str
    loader: Callable[[], CorpusFit]

    @property
    def fit(self) -> CorpusFit:
        return self.loader()


CHANCE_NAME = "chance"
CHANCE_THEORY = '''"""Chance: a listener who ignores the display, uniform over its objects (its speaker, derived,
is uniform over the real messages). Under a fitted link it is the responses' distribution with
no sensitivity to the scene: the feasibility gate's baseline (PREREG_DRAFT.md §4, item 3). Never
a candidate, never carried, never shown to the theorists."""

import jax.numpy as jnp

PARAMS = {}


def listener(params, ctx):
    return jnp.ones(ctx.lex.shape[1]) / ctx.lex.shape[1]
'''


def _journal_default(o):
    """numpy scalars in a journal entry, as Python values."""
    return o.item() if hasattr(o, "item") else float(o)


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
        self.chance_cv: Optional[CorpusCV] = None
        self.admission_limit_sec: Optional[float] = None
        self.history: List[dict] = []
        self.step = 0
        self._cv_jobs: Dict[str, Future] = {}
        self._cv_pool: Optional[ThreadPoolExecutor] = None
        self._parent_jax = threading.Lock()
        # Step mode's round journal (`_memo`): None in a local run, which keeps no journal.
        self._jpath: Optional[Path] = None
        self._jmemo: Dict[str, object] = {}
        self._jlock = threading.Lock()

    # ---------- step mode ----------
    def _step_mode(self) -> bool:
        """Whether this process is a step (`src.rsa_open.loop.slurm`): fits are pending, not fitted here."""
        if step.active() is None:
            return False
        if fit_backend().name != "slurm":
            raise RuntimeError("step mode needs RSA_OPEN_FIT_BACKEND=slurm: a step never fits in its own job")
        return True

    def _open_journal(self, rdir: Path) -> None:
        """Step mode: the round's journal, which a continuation replays from (`_memo`)."""
        if not self._step_mode():
            self._jpath, self._jmemo = None, {}
            return
        self._jpath = rdir / "journal.json"
        self._jmemo = json.loads(self._jpath.read_text()) if self._jpath.exists() else {}

    def _memo(self, key: str, fn: Callable[[], object], dump=lambda x: x, load=lambda x: x):
        """``fn()`` once per round in step mode: its result is journaled the moment it is known, and a
        continuation replaying the round reads it back instead of calling ``fn`` again (an agent
        session, a name, an admission, a ledger line). A local run calls ``fn`` as before."""
        if self._jpath is None:
            return fn()
        with self._jlock:
            if key in self._jmemo:
                return load(self._jmemo[key])
        out = fn()
        with self._jlock:
            self._jmemo[key] = dump(out)
            self._jpath.parent.mkdir(parents=True, exist_ok=True)
            _write_atomic(self._jpath, json.dumps(self._jmemo, default=_journal_default))
        return out

    def _journal_put(self, key: str, value) -> None:
        """Record ``value`` under ``key`` in the round's journal now (step mode)."""
        with self._jlock:
            self._jmemo[key] = value
            self._jpath.parent.mkdir(parents=True, exist_ok=True)
            _write_atomic(self._jpath, json.dumps(self._jmemo, default=_journal_default))

    def _fit_early(self, items, reason: str) -> None:
        """Step mode: submit fits a later phase needs now, so they run while agents work."""
        stepper = step.active()
        if stepper is None or not items:
            return
        try:
            self._prefit(items)
        except FitsPending as ps:
            stepper.submit_early(ps, reason)

    def _cv_early(self, names, reason: str) -> None:
        """Step mode: submit admitted theories' fold fits now (local runs start them in threads)."""
        stepper = step.active()
        if stepper is None:
            return
        todo = [n for n in names if n in self.live and n not in self.cv]
        try:
            collect_pending([lambda n=n: self._cv_one(n) for n in todo])
        except FitsPending as ps:
            stepper.submit_early(ps, reason)

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
        # Out-of-fold scores and pool predictions, saved as they are computed: a resume reads them back
        # instead of recomputing them (74 min of a 27-theory resume on chain 1, 2026-11-03).
        self.scores = ScoreStore(self.dir / ".scores", self.corpus, self.folds, self.cfg.settings,
                                 self._fold_settings(), len(self.pool), MAX_POOL_DRAWS)

    def _fold_settings(self) -> FitSettings:
        return replace(self.cfg.settings, num_warmup=self.cfg.cv_warmup, num_samples=self.cfg.cv_samples)

    def _pool_preds(self, name: str, path: Path, theory: TheoryModel, fitted: CorpusFit) -> np.ndarray:
        """``name``'s pool predictions: saved ones, or computed and saved."""
        saved = self.scores.load_preds(name, path)
        if saved is not None:
            return saved
        preds = pool_predictions(theory, fitted, self.pool)
        self.scores.save_preds(name, path, preds)
        return preds

    def setup(self) -> None:
        self._prepare()
        if self.cfg.inherit_ledger is not None and not Path(self.cfg.inherit_ledger).exists():
            raise FileNotFoundError(f"inherit_ledger {self.cfg.inherit_ledger} does not exist")
        self.ledger = HypothesisLedger.create(self.dir / LEDGER_FILENAME, inherit_from=self.cfg.inherit_ledger)
        self.models_dir.mkdir(exist_ok=True)
        entries = read_manifest_entries(self.cfg.seed_models_dir)
        for e in entries:
            shutil.copyfile(Path(self.cfg.seed_models_dir) / f"{e['name']}.py", self.models_dir / f"{e['name']}.py")
        seeds = [(self.models_dir / f"{e['name']}.py", e["name"]) for e in entries]
        if self._step_mode():  # one array: the seeds' fits and the chance theory's
            def chance() -> None:
                try:
                    self._chance_fit()
                except ModelFailure:
                    pass  # score_chance records it
                except FitPending as p:
                    rule = {"fixed": None if self.live_cfg.time_limit_sec is None else self.cfg.in_play_fit_limit_sec}
                    raise FitsPending([p], self._folds_after(self._chance_path(), CHANCE_NAME, p.mem_gb, rule)) from None

            collect_pending([lambda: self._prefit(seeds, self.live_cfg, folds_after=True), chance])
        self._prefit(seeds, self.live_cfg)
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
            self.live[name] = Live(name, hypothesis, theory, FitSummary.of(fitted),
                                   self._pool_preds(name, path, theory, fitted), "seed", self._loader(path, name))
        if not self.live:
            raise RuntimeError("no seed theory could be fitted; nothing to compare")
        self._set_admission_limit()
        self._write_manifest()

    def _set_admission_limit(self) -> None:
        """`ADMISSION_FACTOR` x the median full-fit time of the starting theories (None: unlimited)."""
        if self.cfg.fit_time_limit_sec is None:
            return
        timed = {n: m.summary.timing["warmup_sec"] + m.summary.timing["sampling_sec"] for n, m in self.live.items()
                 if m.summary.timing}
        if not timed:
            raise RuntimeError("no starting theory's fit records its time (fits cached by code before "
                               "2026-10-23?), so the admission limit cannot be set; start in a fresh work root")
        self.admission_limit_sec = min(self.cfg.in_play_fit_limit_sec, max(
            self.cfg.fit_time_limit_sec, ADMISSION_FACTOR * float(np.median(list(timed.values())))))
        self.gate_cfg = replace(self.gate_cfg, time_limit_sec=self.admission_limit_sec)
        print(f"[open loop] admission limit {self.admission_limit_sec / 60:.0f} min: {ADMISSION_FACTOR:g} x the median "
              f"of the starting fits ({', '.join(f'{n} {t / 60:.0f}' for n, t in sorted(timed.items()))} min)", flush=True)

    def _loader(self, path: Path, name: str) -> Callable[[], CorpusFit]:
        """Loads a theory in play's fit from the cache when a step needs it."""
        return lambda: self._fit(path, name, self.live_cfg)

    def _fit(self, path: Path, name: str, cfg: GateConfig) -> CorpusFit:
        return loop_fit(path, name, cfg.corpus_path, cfg.settings, cfg.cache_dir, time_limit_sec=cfg.time_limit_sec,
                        log_likelihood=False)

    def _prefit(self, items, cfg: Optional[GateConfig] = None, folds_after: bool = False,
                only_if_converged: bool = False) -> None:
        """Fit concurrently (cached, failures remembered) so sequential admission reads finished fits.
        Step mode, ``folds_after``: a theory whose full fit is pending has its folds submitted to follow
        it (`fit_array.FoldAfter`), so no step is spent only submitting them."""
        cfg = cfg or self.gate_cfg

        def one(item) -> None:
            path, name = item
            if path.name == "candidate.py" and read_candidate(path.parent) is not None:
                return
            if code_problems(path.read_text(encoding="utf-8")) or (
                    path.name == "candidate.py" and magic_number_problems(path.read_text(encoding="utf-8"))):
                return
            try:
                self._fit(path, name, cfg)
            except ModelFailure:
                pass  # remembered; admission reports it
            except (FitPending, FitsPending):
                raise  # step mode: not done yet, which is nobody's failure
            except Exception:
                if path.name == "candidate.py":
                    return  # a file the gates will refuse (not a theory, a memo error): admission says why
                raise

        if not items:
            return
        if self._step_mode():  # every pending fit at once, for one array
            try:
                collect_pending([lambda item=item: one(item) for item in items])
            except FitsPending as ps:
                if not folds_after:
                    raise
                gb = {p.job.name: p.mem_gb for p in ps.pending}
                after = [a for path, name in items if name in gb
                         for a in self._folds_after(path, name, gb[name], only_if_converged=only_if_converged)]
                raise FitsPending(ps.pending, list(ps.after) + after) from None
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
        from src.rsa_open.loop.fit_array import fold_limit

        return fold_limit(self._fold_rule(), self.live[name].summary.timing)

    def _fold_rule(self) -> dict:
        """`fit_array.fold_limit`'s rule for a theory in play (shared with the folds-after tasks)."""
        if self.live_cfg.time_limit_sec is None:
            return {"fixed": None}
        return {"in_play": self.cfg.in_play_fit_limit_sec, "fallback": self.cfg.fit_time_limit_sec,
                "min": FOLD_MIN_SEC, "factor": FOLD_TIME_FACTOR}

    def _folds_after(self, path: Path, name: str, mem_gb: int, rule: Optional[dict] = None,
                     only_if_converged: bool = False) -> List[FoldAfter]:
        """``name``'s folds, to fit as soon as its full fit (at ``live_cfg``) is in the cache."""
        cfg = self.live_cfg
        return [FoldAfter(Path(path), name, Path(cfg.corpus_path), cfg.settings, Path(cfg.cache_dir), Path(fp),
                          self._fold_settings(), rule or self._fold_rule(), only_if_converged, mem_gb)
                for fp in self.fold_paths]

    def _cv_one(self, name: str) -> Optional[CorpusCV]:
        m = self.live[name]
        return self._cv_file(name, self.models_dir / f"{name}.py", m.theory, m.loader, self.fold_time_limit(name))

    def _cv_file(self, name: str, path: Path, theory: TheoryModel, fit: Callable[[], CorpusFit],
                 limit: Optional[float]) -> Optional[CorpusCV]:
        """``name``'s out-of-fold scores (None: a fold failed), saved ones or computed and saved."""
        saved = self.scores.load_cv(name, path, limit)
        if saved is not None:
            cv, problem = saved
            if problem:
                self.cv_problems[name] = problem
            return cv
        self.cv_problems.pop(name, None)
        cv = self._compute_cv(name, path, theory, fit(), limit)  # the fit's draws only when computing
        self.scores.save_cv(name, path, limit, cv, self.cv_problems.get(name))
        return cv

    def _compute_cv(self, name: str, path: Path, theory: TheoryModel, fit: CorpusFit,
                    limit: Optional[float]) -> Optional[CorpusCV]:
        full = fit.flat()
        start = {k: float(np.median(v)) for k, v in full.items()}
        fold_settings = self._fold_settings()

        def fold_fit(f: int) -> CorpusFit:
            return loop_fit(path, name, self.fold_paths[f], fold_settings, self.dir / ".fit_cache",
                            time_limit_sec=limit, init=start, log_likelihood=False)

        workers = self.cfg.cv_folds if self.live_cfg.time_limit_sec is not None else 1
        try:
            if self._step_mode():
                fits = collect_pending([lambda f=f: fold_fit(f) for f in range(self.cfg.cv_folds)])
            else:
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    fits = list(pool.map(fold_fit, range(self.cfg.cv_folds)))
        except ModelFailure as exc:
            self.cv_problems[name] = f"{CV_FOLD_FAILED}: {exc}"
            return None
        bad = [f"fold {f + 1}: " + "; ".join(fit.convergence_problems) for f, fit in enumerate(fits) if not fit.converged]
        if bad:
            self.cv_problems[name] = "cross-validation folds did not converge, so it is untrusted: " + " | ".join(bad)
        out = {b.name: np.empty(b.n) for b in self.blocks}
        with self._parent_jax:  # this process runs on one core: one theory's held-out scores at a time
            for f, fitted in enumerate(fits):
                for b in self.blocks:
                    rows = np.where(self.folds[b.name] == f)[0]
                    if len(rows):
                        out[b.name][rows] = heldout_lpd(theory, fitted, b.subset(rows))
        impossible = [b for b, v in out.items() if np.nanmax(v) > IMPOSSIBLE_LPD]
        if impossible:  # every link is a probability mass function: a held-out log probability is <= 0
            self.cv_problems[name] = ("an impossible held-out score (a log probability above 0 on "
                                      f"{', '.join(impossible)}: a numerical failure), so it is untrusted")
            return CorpusCV(out, False)
        return CorpusCV(out, all(fit.converged for fit in fits))

    def score_chance(self) -> Optional[CorpusCV]:
        """The chance theory's out-of-fold lpd, as every theory's is computed (``.chance/``, outside
        what the theorists read). A failure is recorded, never raised: the gate reports it."""
        path = self._chance_path()
        try:
            fitted = self._chance_fit()
        except ModelFailure as exc:
            self.cv_problems[CHANCE_NAME] = f"its fit failed: {exc}"
            return None
        limit = None if self.live_cfg.time_limit_sec is None else self.cfg.in_play_fit_limit_sec
        self.chance_cv = self._cv_file(CHANCE_NAME, path, TheoryModel(path, name=CHANCE_NAME), lambda: fitted, limit)
        return self.chance_cv

    def _chance_path(self) -> Path:
        path = self.dir / ".chance" / f"{CHANCE_NAME}.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists() or path.read_text() != CHANCE_THEORY:
            path.write_text(CHANCE_THEORY)
        return path

    def _chance_fit(self):
        return loop_fit(self._chance_path(), CHANCE_NAME, self.corpus, self.cfg.settings, self.dir / ".fit_cache",
                        time_limit_sec=self.live_cfg.time_limit_sec, log_likelihood=False)

    def write_newest_cv(self) -> Path:
        """``newest_block_cv.json``: the newest block's units and every scored theory's out-of-fold lpd
        there (the chance theory's too), for the feasibility gate (`src.rsa_open.gate`)."""
        newest = self.blocks[-1]
        units = np.unique(np.asarray(newest.units), return_inverse=True)[1]
        out = {"block": newest.name, "units": units.tolist(),
               "theories": {n: r.pointwise[newest.name].tolist() for n, r in self.cv.items() if r is not None},
               "converged": {n: bool(r.converged) for n, r in self.cv.items() if r is not None},
               "chance": None if self.chance_cv is None else self.chance_cv.pointwise[newest.name].tolist(),
               "chance_converged": None if self.chance_cv is None else bool(self.chance_cv.converged),
               "chance_problem": self.cv_problems.get(CHANCE_NAME)}
        path = self.dir / "newest_block_cv.json"
        path.write_text(json.dumps(out))
        return path

    def _start_cv(self, name: str) -> None:
        """Start ``name``'s CV folds in the background (time-limited fits only: in-process fits run
        one at a time, in `standing`)."""
        if self.live_cfg.time_limit_sec is None or name in self.cv or name in self._cv_jobs or self._step_mode():
            return
        if self._cv_pool is None:
            self._cv_pool = ThreadPoolExecutor(max_workers=CV_THEORIES_AT_ONCE, thread_name_prefix="cv")
        self._cv_jobs[name] = self._cv_pool.submit(self._cv_one, name)

    def _stop_background(self) -> None:
        if self._cv_pool is not None:
            self._cv_pool.shutdown(wait=False, cancel_futures=True)
            self._cv_pool = None

    def standing(self) -> Dict[str, dict]:
        todo = [n for n in self.live if n not in self.cv]
        if self._step_mode():
            for name, cv in zip(todo, collect_pending([lambda n=n: self._cv_one(n) for n in todo])):
                self.cv[name] = cv
        for name in todo:
            self._start_cv(name)
        for name in [n for n in todo if n not in self.cv]:
            job = self._cv_jobs.pop(name, None)
            self.cv[name] = job.result() if job is not None else self._cv_one(name)
        # Only converged cross-validation is compared (PI 2026-10-28): run C′'s one unconverged fold
        # scored +121,409 nats, and every other theory was then 138,000 nats "behind" it on that block.
        scored = {n: self.cv[n] for n in self.live if self.cv.get(n) is not None and self.cv[n].converged}
        out = standing(scored, self.blocks, guard_k=self.cfg.guard_k) if scored else {}
        for n in self.live:
            if n not in out:  # a fold failed, or did not converge: unscored, never selected
                out[n] = dict(sel_elpd=-math.inf, sel_diff=math.inf, sel_dse=0.0, eligible=False, guard_failures=[],
                              by_block={}, cv_converged=False)
            out[n]["trusted"] = bool(self.live[n].summary.converged and out[n]["cv_converged"])
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
        _write_atomic(self.dir / "history.json", json.dumps(self.history, indent=1, default=float))
        self.write_newest_cv()
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
                rdir, spawn=self._spawn_once, incumbent=incumbent, theory=m.theory, fitted=m.fit, hypothesis=m.hypothesis,
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
            return list(pool.map(lambda j: self._spawn_once(*j), jobs))

    def _spawn_once(self, cdir: Path, prompt: str) -> bool:
        """An agent session, once per round in step mode (`_memo`). The files in ``cdir`` when it starts
        (its brief, a repair's copies) are journaled first; a session whose step was killed before it
        ended (review V3 §11) is set aside as ``.killed_<dir>_<i>`` and run again from those files
        alone, not from the killed session's partial ones."""
        cdir = Path(cdir)
        if self._jpath is None:
            return bool(self.spawn(cdir, prompt))
        key, started = f"spawn:{cdir}", f"started:{cdir}"
        with self._jlock:
            if key in self._jmemo:
                return self._jmemo[key]
            brief = self._jmemo.get(started)
        if brief is not None and cdir.exists():
            k = 1
            while (cdir.parent / f".killed_{cdir.name}_{k}").exists():
                k += 1
            aside = cdir.parent / f".killed_{cdir.name}_{k}"
            cdir.rename(aside)
            cdir.mkdir()
            for rel in brief:
                if (aside / rel).is_file():
                    (cdir / rel).parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(aside / rel, cdir / rel)
            print(f"[open loop] {cdir.name}: its session was killed with its step; set aside as {aside.name}, "
                  "run again", flush=True)
        files = sorted(str(f.relative_to(cdir)) for f in cdir.rglob("*") if f.is_file()) if cdir.exists() else []
        self._journal_put(started, files)
        return self._memo(key, lambda: bool(self.spawn(cdir, prompt)))

    def _same_hypothesis(self, hypothesis: str) -> Optional[str]:
        key = collapse_whitespace(hypothesis).strip().lower()
        if not key:
            return None
        texts = [(m.name, m.hypothesis) for m in self.live.values()]
        texts += [(e.name, e.hypothesis) for e in self.ledger.entries() if e.outcome == "admitted"]
        return next((n for n, t in texts if key in collapse_whitespace(t).strip().lower()), None)

    def _try_admit(self, cdir: Path, name: str, context: str) -> tuple[bool, str]:
        r = self._memo(f"admit:{cdir}", lambda: dict(zip(("ok", "reason"), self._admit_now(cdir, name, context)),
                                                       name=name, context=context))
        return r["ok"], r["reason"]

    def _admit_now(self, cdir: Path, name: str, context: str) -> tuple[bool, str]:
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
            self.live[name] = Live(name, hypothesis, TheoryModel(dst, name=name), FitSummary.of(out.fit), out.pool_preds,
                                   context, self._loader(dst, name))
            self.scores.save_preds(name, dst, out.pool_preds)
            self._write_manifest()
            self._ledger(name, "admitted", "admitted", hypothesis, context)
            self._start_cv(name)  # its folds run while the round goes on
        else:
            self._ledger(name, "rejected", out.reason, hypothesis, context)
        return out.admitted, out.reason

    def _name_slots(self, slots, key: str, round_index: int, taken: set) -> list:
        """Name every slot with a candidate in ``slots[key]``, avoiding ``taken`` (which grows)."""
        named = []
        for s in slots:
            if (s[key] / "candidate.py").exists():
                s[key + "_name"] = self._memo(f"name:{s[key]}", lambda s=s: self._name_for(
                    s[key], f"r{round_index + 1}_c{s['i'] + 1}", frozenset(taken)))
                taken.add(s[key + "_name"])
                named.append(s)
        return named

    @staticmethod
    def _fit_items(slots, key: str) -> list:
        return [(s[key] / "candidate.py", s[key + "_name"]) for s in slots]

    def _admit_named(self, slots, key: str, context_suffix: str = "") -> List[tuple]:
        """Admit the named slots one by one, in slot order (their fits finished or remembered)."""
        results = []
        for s in slots:
            if key + "_name" not in s:
                continue
            ok, reason = self._try_admit(s[key], s[key + "_name"], s["context"] + context_suffix)
            results.append((s, ok, reason))
        return results

    def _admit_all(self, slots, key: str, round_index: int, context_suffix: str = "") -> List[tuple]:
        """Name every slot with a candidate in ``slots[key]``, fit them at once, admit one by one. In
        step mode a converged repair's folds follow its fit, so no step only submits them."""
        self._prefit(self._fit_items(self._name_slots(slots, key, round_index, set()), key),
                     folds_after=True, only_if_converged=True)
        return self._admit_named(slots, key, context_suffix)

    def run_round(self, round_index: int) -> dict:
        rdir = self.dir / f"round_{round_index + 1}"
        self._open_journal(rdir)
        # The round's starting point: a continuation replays from these, not from a standing that
        # already holds the round's own admissions (step mode).
        start = self._memo("start", lambda: {"standing": self.standing()})
        s = start["standing"]
        incumbent = self.incumbent(s)
        roles = slot_roles(self.cfg.candidate_count)
        per_round = exploratory_slots_per_round(self.cfg.candidate_count)
        crit = self._memo("critique", lambda: self._critique(rdir, incumbent),
                          dump=lambda c: {"md": c.critiques_md, "status": c.status},
                          load=lambda d: critique_mod.CritiqueOutcome(d["md"], d["status"]))
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
        taken: set = set()
        first = self._name_slots(slots, "dir", round_index, taken)
        retry, retried = [], []
        for sl in slots:
            if not (sl["dir"] / "candidate.py").exists():
                sl["dir"] = sl["dir"].with_name(sl["dir"].name + "_retry_1")
                sl["context"] += " retry 1"
                retried.append(sl)
                retry.append((sl["dir"], self._docs_for(sl["dir"], sl["role"], sl["lens"], round_index, s, incumbent,
                                                        briefs.retry_note(), critiques)))
        # The first attempts are fitted while the retries' agents work (in-process fits, untimed, run
        # one at a time); admission waits for both. A step submits them early, and yields for both.
        if self._step_mode():
            self._fit_early(self._fit_items(first, "dir"), f"round {round_index + 1}: first candidates")
            self._spawn_all(retry)
            self._prefit(self._fit_items(first, "dir") + self._fit_items(
                self._name_slots(retried, "dir", round_index, taken), "dir"))
        else:
          with ThreadPoolExecutor(max_workers=1, thread_name_prefix="prefit") as background:
            fitting = (background.submit(self._prefit, self._fit_items(first, "dir"))
                       if self.gate_cfg.time_limit_sec is not None else None)
            if fitting is None:
                self._prefit(self._fit_items(first, "dir"))
            self._spawn_all(retry)
            self._prefit(self._fit_items(self._name_slots(retried, "dir", round_index, taken), "dir"))
            if fitting is not None:
                fitting.result()
        events = []
        for sl in slots:
            if not (sl["dir"] / "candidate.py").exists():
                self._memo(f"ledger:no file:{sl['dir']}", lambda sl=sl: self._ledger(
                    f"slot_{sl['i'] + 1}", "rejected", "the agent wrote no candidate.py", "", sl["context"]))
                events.append(dict(slot=sl["i"] + 1, role=sl["role"], name=None, outcome="no file", reason=""))
        repairs = []
        for sl, ok, reason in self._admit_named(slots, "dir"):
            events.append(dict(slot=sl["i"] + 1, role=sl["role"], name=sl["dir_name"],
                               outcome="admitted" if ok else "rejected", reason=reason,
                               hypothesis=_read(sl["dir"] / "hypothesis.md")))
            if not ok:
                repairs.append((sl, reason))
        # A step submits the admitted theories' folds now; they run while the repair agents work.
        self._cv_early([sl["dir_name"] for sl in slots if sl.get("dir_name") in self.live],
                       f"round {round_index + 1}: admitted theories' folds")
        jobs = []
        for sl, reason in repairs:
            rd = sl["dir"].with_name(sl["dir"].name + "_repair_1")
            sl["repair"] = rd
            if self._jpath is not None and f"spawn:{rd}" in self._jmemo:
                continue  # a continuation: the repair session ran; its files are the agent's, not copies
            rd.mkdir(parents=True, exist_ok=True)
            for f in ("candidate.py", "hypothesis.md", "model_name.txt"):
                if (sl["dir"] / f).exists():
                    shutil.copyfile(sl["dir"] / f, rd / f)
            jobs.append((rd, self._docs_for(rd, sl["role"], sl["lens"], round_index, s, incumbent,
                                            briefs.repair_note(reason), critiques)))
        self._spawn_all(jobs)
        changed = []
        for sl, reason in repairs:
            before, after = sl["dir"] / "candidate.py", sl["repair"] / "candidate.py"
            if after.exists() and before.exists() and after.read_bytes() == before.read_bytes():
                # Run C′: a repair session spent its hour on fits of its own and left the file as it was;
                # refitting the same file can only fail the same way.
                why = "the repair left candidate.py unchanged, so the rejection stands: " + reason
                self._memo(f"ledger:unchanged:{sl['repair']}", lambda sl=sl, why=why: self._ledger(
                    sl["dir_name"] + "_repair", "rejected", why, _read(sl["repair"] / "hypothesis.md"),
                    sl["context"] + " repair 1"))
                events.append(dict(slot=sl["i"] + 1, role=sl["role"], name=None, outcome="rejected", reason=why,
                                   repair=True))
            else:
                changed.append(sl)
        for sl, ok, reason in self._admit_all(changed, "repair", round_index, " repair 1"):
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
                                             + "; ".join(self.live[n].summary.convergence_problems or ["untrusted"]))
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
        return self._export(entry, stopped)

    def _export(self, entry: dict, stopped: Optional[str]) -> dict:
        shutil.copyfile(self.models_dir / f"{entry['best_model']}.py", self.dir / "best_model.py")
        export = dict(best_model=entry["best_model"], live=sorted(e["name"] for e in entry["live"]))
        if stopped:
            export["stopped_early"] = stopped
        _write_atomic(self.dir / "export.json", json.dumps(export))
        return entry

    # ---------- resuming ----------
    def resume(self) -> int:
        """Restore an interrupted loop to its last scored step; returns the first round still to run
        (``max_iterations + 1`` when the end step was scored and only the export is missing).

        The step's history entry holds the live set and the ledger's length (the closed loop's
        `src.rsa.loop.orchestrator.RSALoop.resume`). What an unfinished round or end did after it
        is undone: ledger lines past that length are dropped (kept beside the round's directory as
        ``ledger_tail.jsonl``), theories it admitted leave the set (their files stay in the round's
        directory), a theory the end had pruned is put back, and the round's directory becomes
        ``round_<k>_abandoned_<i>``, so the round runs again from its agents. Fits come back from
        the cache; the admission limit is the one recorded.

        A step's continuation is the exception: a round with a journal (``round_<k>/journal.json``,
        step mode only) is continued, not abandoned. Its ledger lines stay, the theories it admitted
        stay live, and `run_round` replays it from the journal (`_memo`)."""
        hist_path = self.dir / "history.json"
        if not hist_path.exists():
            raise FileNotFoundError(f"{hist_path} not found: the loop stopped before its seeds were scored")
        if (self.dir / "export.json").exists():
            raise RuntimeError(f"{self.dir} finished (export.json exists); there is nothing to resume")
        self.history = json.loads(hist_path.read_text())
        last = self.history[-1]
        missing = {"step", "round", "live", "ledger_entries", "admission_limit_sec"} - set(last)
        if missing:
            raise ValueError(f"{hist_path}'s last step lacks {sorted(missing)}; it cannot be resumed")
        self._prepare()
        next_round = last["round"] + 1
        partial = self.dir / f"round_{next_round + 1}"
        aside = None
        journal = partial / "journal.json"
        cont = json.loads(journal.read_text()) if self._step_mode() and journal.exists() else None
        if partial.exists() and cont is None:
            k = 1
            while (self.dir / f"round_{next_round + 1}_abandoned_{k}").exists():
                k += 1
            aside = self.dir / f"round_{next_round + 1}_abandoned_{k}"
            partial.rename(aside)
        ledger_path = self.dir / LEDGER_FILENAME
        lines = [ln for ln in ledger_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if len(lines) < last["ledger_entries"]:
            raise ValueError(f"{ledger_path} has {len(lines)} entries, fewer than the "
                             f"{last['ledger_entries']} recorded at step {last['step']}")
        tail = [] if cont is not None else lines[last["ledger_entries"]:]
        if tail:
            where = aside or self.dir
            _write_atomic(where / ("ledger_tail.jsonl" if aside else f"ledger_tail_step_{last['step']}.jsonl"),
                          "".join(ln + "\n" for ln in tail))
        if cont is None:
            _write_atomic(ledger_path, "".join(ln + "\n" for ln in lines[:last["ledger_entries"]]))
        self.ledger = HypothesisLedger(ledger_path)
        self.ledger.entries()  # parses, or raises
        live = {e["name"]: e for e in last["live"]}
        for key, r in (cont or {}).items():  # the continued round's admissions, in the order it made them
            if key.startswith("admit:") and r["ok"]:
                live[r["name"]] = {"name": r["name"], "source": r["context"],
                                   "hypothesis": _read(Path(key[len("admit:"):]) / "hypothesis.md")}
        pruned_dir = self.models_dir / "pruned"
        for name in live:
            if not (self.models_dir / f"{name}.py").exists() and (pruned_dir / f"{name}.py").exists():
                shutil.move(str(pruned_dir / f"{name}.py"), self.models_dir / f"{name}.py")
            if not (self.models_dir / f"{name}.py").exists():
                raise FileNotFoundError(f"{name} was live at step {last['step']} but its file is gone")
        for f in self.models_dir.glob("*.py"):
            if f.stem not in live:
                f.unlink()  # admitted by the unfinished round; its candidate directory keeps the source
        self.admission_limit_sec = last["admission_limit_sec"]
        if self.admission_limit_sec is not None:
            self.gate_cfg = replace(self.gate_cfg, time_limit_sec=self.admission_limit_sec)
        def summary(name: str) -> Optional[FitSummary]:
            return loop_fit_summary(self.models_dir / f"{name}.py", name, self.corpus, self.cfg.settings,
                                    self.dir / ".fit_cache")

        # A theory in play was fitted before: its summary comes from the cache's sidecars, and its draws
        # are loaded only if its pool predictions weren't saved (A1). One whose fit is gone is fitted.
        self._prefit([(self.models_dir / f"{n}.py", n) for n in live if summary(n) is None], self.live_cfg)
        for name, e in live.items():
            path = self.models_dir / f"{name}.py"
            theory = TheoryModel(path, name=name)
            preds = self.scores.load_preds(name, path)
            if preds is None:
                preds = self._pool_preds(name, path, theory, self._fit(path, name, self.live_cfg))
            self.live[name] = Live(name, e["hypothesis"], theory, summary(name), preds, e["source"],
                                   self._loader(path, name))
        for name, d in last.get("standing", {}).items():
            # A failed fold is remembered only in its process: keep the step's verdict rather than
            # refitting the fold (for up to FOLD_TIME_FACTOR x its full fit) to perhaps another one.
            if name in self.live and str(d.get("cv_problem", "")).startswith(CV_FOLD_FAILED):
                self.cv[name], self.cv_problems[name] = None, d["cv_problem"]
        self._write_manifest()
        self.step = last["step"] + 1
        print(f"[open loop] resumed after step {last['step']} (round {last['round'] + 1} of {self.cfg.max_iterations}"
              f" scored; {len(self.live)} live theories)" + (f"; the unfinished round is kept as {aside.name}"
                                                             if aside else "")
              + (f"; continuing round {next_round + 1} from its journal" if cont is not None else ""), flush=True)
        return next_round

    def _time_for_round(self, r: int) -> None:
        if self.cfg.deadline is None:
            return
        left = (self.cfg.deadline - time.time()) / 3600
        if left < self.cfg.round_min_hours:
            raise LoopOutOfTime(f"round {r + 1} of {self.cfg.max_iterations}: {left:.1f} h of the job left, under the "
                                f"{self.cfg.round_min_hours:g} h a round needs; the next job resumes the loop")

    def _score_chance_and_standing(self) -> None:
        """The chance theory's folds; in step mode together with the live theories' (one array)."""
        if self._step_mode():
            collect_pending([self.score_chance, self.standing])
        else:
            self.score_chance()

    def run(self, resume: bool = False) -> dict:
        try:
            return self._run(resume)
        finally:
            self._stop_background()

    def _run(self, resume: bool) -> dict:
        if resume:
            first = self.resume()
            if first > self.cfg.max_iterations:  # the end was scored; only its export is missing
                last = self.history[-1]
                stopped = next((e["reason"] for e in last["events"] if e.get("outcome") == "stopped"), None)
                return self._export(last, stopped)
            self._score_chance_and_standing()
        else:
            self.setup()
            self._score_chance_and_standing()
            self.score(-1, [dict(outcome="seeded", name=n) for n in self.live])
            first = 0
        patience = self.cfg.stop_after_stale_rounds
        for r in range(first, self.cfg.max_iterations):
            if r > 0 and patience and stale_rounds(self.history) >= patience:
                why = (f"stopped after round {r} of {self.cfg.max_iterations}: {patience} rounds in a row left "
                       f"the incumbent ({self.history[-1]['best_model']}) unchanged")
                return self.end(stopped=why)
            self._time_for_round(r)
            self.run_round(r)
        return self.end()
