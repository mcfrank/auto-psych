"""One chain of the open campaign, experiment by experiment (docs/auto_open/PLAN.md §4-§5, Phase 2b).

    experiment<n>/
      models_input/        the theories this experiment starts from (the seeds, then the previous
                           experiment's live set) with their manifest
      prior_corpus.json    the data so far: E and every earlier experiment's block
      design/              the designer's cycles and frozen/ (`src.rsa_open.designer.run_designer`)
      data/rows.jsonl      every participant's trials and responses
      data/timing.json     a live experiment's completion times (`record_timing`; none when simulated)
      data/block.json      the experiment's test trials as a corpus block
      corpus.json          E, every earlier block and this one
      loop/                the inner loop (`src.rsa_open.loop.orchestrator.OpenLoop`) and its export

Each stage is skipped when its output exists, so a chain resumes where it stopped. A design or
loop that stopped part way starts again from scratch (as main's model-loop stage does); the
abandoned attempt is kept beside it as ``design.abandoned_<i>`` / ``loop.abandoned_<i>`` (no agent
reads it), and a loop's fit cache moves to the new attempt, so no fit is repeated:

1. **design**: the carried theories and **the bar** (the chain's starting theories, `bar_dir`, kept
   in every design although the loop may prune them, as the closed campaign's design kept its
   seeds) are fitted to the data so far; their posterior draws (theory parameters only) and
   hypotheses, the chain's earlier paradigms with each theory's standing on them, and the timing
   records go to the designer, whose frozen paradigm runs. Without the bar a loop that pruned all
   but one theory would leave the next design nothing to tell apart (zero bits, no paradigm
   above the floor: the first two-experiment dry run);
2. **collect**: simulated participants (`GroundTruth`: a theory at fixed parameters, each link at
   its reference values) answer every list of the frozen paradigm; the live path (Prolific) is
   Phase 4's and is not here;
3. **loop**: the inner loop on E plus every block so far, starting from ``models_input`` and
   continuing the previous experiment's ledger; its live set is the next experiment's input.
"""

from __future__ import annotations

import json
import math
import shutil
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import jax
import numpy as np
import yaml

from src.models.model_manifest import read_manifest_entries
from src.rsa.fit import FitSettings
from src.rsa_open.corpus import Block
from src.rsa_open.designer import EarlierParadigm, SpawnFn as DesignerSpawn, run_designer
from src.rsa_open.fit import TheoryModel, simulate
from src.rsa_open.loop.fitting import cached_loop_fit, fit_cached, read_corpus, write_corpus
from src.rsa_open.loop.orchestrator import OpenLoop, OpenLoopConfig
from src.rsa_open.timing import CLOSED_CAMPAIGN, TimingRecord, read_records, record_from_sessions, write_records

DESIGN_DRAWS = 200
# The design's own fits (a theory the previous loop did not fit on the same data): the design needs
# DESIGN_DRAWS posterior draws, not the loop's 4 x 1,000, and no convergence gate (main's design
# twin until 2026-10-08). A fit not back within DESIGN_FIT_LIMIT_SEC, or failing, leaves its
# theory out of that design, on record (`design_screened_out.json`): run A's design waited 6 h+
# on one bar theory's fit, which had no time limit.
DESIGN_SETTINGS = FitSettings(num_warmup=500, num_samples=250, num_chains=2, dense_mass=True)
DESIGN_FIT_LIMIT_SEC = 2 * 3600
SCREENED_OUT = "design_screened_out.json"


@dataclass(frozen=True)
class GroundTruth:
    """Simulated participants: a theory at fixed parameters; each link at ``link_params`` (by link
    name), else its reference values."""

    theory_path: Path
    theory_params: Mapping[str, float]
    link_params: Mapping[str, Mapping[str, float]] = field(default_factory=dict)


@dataclass
class ChainConfig:
    chain: str
    seeds_dir: Path
    literature: Block
    out_dir: Path
    ground_truth: GroundTruth
    n_experiments: int = 4
    loop: Dict[str, Any] = field(default_factory=dict)
    """OpenLoopConfig overrides (max_iterations, candidate_count, settings, ...)."""
    settings: FitSettings = field(default_factory=lambda: FitSettings(dense_mass=True))
    pilot_participants: int = 12
    timing: Sequence[TimingRecord] = CLOSED_CAMPAIGN
    code_commit: Optional[str] = None
    seed: int = 0
    bar_dir: Optional[Path] = None
    """The bar: theories every design scores beside the carried ones (default: the seeds)."""
    design_settings: FitSettings = DESIGN_SETTINGS
    design_fit_limit_sec: float = DESIGN_FIT_LIMIT_SEC
    other_chains: Sequence[Path] = ()
    """The campaign's other chains' run directories, whose finished experiments' timing every
    design reads too (PLAN.md, "Timing digest")."""


def _hypotheses(models_dir: Path) -> Dict[str, str]:
    return {e["name"]: e.get("rationale", "") for e in read_manifest_entries(models_dir)}


def set_aside(stage_dir: Path, keep: Sequence[str] = ()) -> Optional[Path]:
    """Move an unfinished stage's directory to ``<name>.abandoned_<i>``, carrying the entries
    named in ``keep`` (e.g. its fit cache) into a fresh ``stage_dir``; None if there was none."""
    stage_dir = Path(stage_dir)
    if not stage_dir.exists():
        return None
    i = 1
    while stage_dir.with_name(f"{stage_dir.name}.abandoned_{i}").exists():
        i += 1
    aside = stage_dir.with_name(f"{stage_dir.name}.abandoned_{i}")
    stage_dir.rename(aside)
    stage_dir.mkdir()
    for name in keep:
        if (aside / name).exists():
            (aside / name).rename(stage_dir / name)
    return aside


def fresh_corpus(path: Path, blocks: Sequence[Block]) -> Path:
    """Write an unfinished stage's corpus; one left by an earlier attempt on other data (a restart
    after the data or their convention changed) is set aside as ``<stem>.abandoned_<i>.json``
    rather than refused (run A's restart, 2026-10-22, had to move them by hand)."""
    path = Path(path)
    try:
        return write_corpus(path, blocks)
    except FileExistsError:
        i = 1
        while path.with_name(f"{path.stem}.abandoned_{i}.json").exists():
            i += 1
        path.rename(path.with_name(f"{path.stem}.abandoned_{i}.json"))
        return write_corpus(path, blocks)


def _copy_models(src: Path, dst: Path) -> None:
    """A models directory (theories + manifest), copied once."""
    if (dst / "models_manifest.yaml").exists():
        return
    tmp = dst.with_name(dst.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    for e in read_manifest_entries(src):
        shutil.copyfile(src / f"{e['name']}.py", tmp / f"{e['name']}.py")
    shutil.copyfile(src / "models_manifest.yaml", tmp / "models_manifest.yaml")
    tmp.rename(dst)


class Chain:
    def __init__(self, cfg: ChainConfig, *, designer_spawn: DesignerSpawn,
                 theorist_spawn: Callable[[Path], Callable[[Path, str], bool]], participant_factory=None) -> None:
        """``theorist_spawn(loop_dir)`` is an experiment's theorist spawner (theorists read that
        loop's ``models/`` and ``data/``)."""
        self.cfg = cfg
        self.designer_spawn = designer_spawn
        self.theorist_spawn = theorist_spawn
        self.participant_factory = participant_factory
        self.out = Path(cfg.out_dir)

    def exp(self, n: int) -> Path:
        return self.out / f"experiment{n}"

    # ---------- the data so far ----------
    def blocks_before(self, n: int) -> List[Block]:
        blocks = [self.cfg.literature]
        for k in range(1, n):
            blocks += read_corpus(self.exp(k) / "data" / "block.json")
        return blocks

    def models_input(self, n: int) -> Path:
        dst = self.exp(n) / "models_input"
        src = Path(self.cfg.seeds_dir) if n == 1 else self.exp(n - 1) / "loop" / "models"
        if n > 1 and not (self.exp(n - 1) / "loop" / "export.json").exists():
            raise FileNotFoundError(f"experiment {n - 1}'s inner loop has not finished (no export.json)")
        _copy_models(src, dst)
        return dst

    # ---------- timing ----------
    def record_timing(self, n: int, seconds: Sequence[float]) -> TimingRecord:
        """A live experiment's completion times as its timing record (written once, after
        collection; simulated people take no time, so a simulated experiment has none)."""
        path = self.exp(n) / "data" / "timing.json"
        if path.exists():
            raise FileExistsError(f"{path} exists: an experiment's timing is recorded once")
        lists = json.loads((self.exp(n) / "design" / "frozen" / "lists.json").read_text())
        record = record_from_sessions(f"{self.cfg.chain}, experiment {n}", lists, seconds)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_records(path, [record])
        return record

    def timing(self, n: int) -> List[TimingRecord]:
        """What experiment n's designer reads: the configured records (the closed campaign's), this
        chain's earlier experiments' and every finished experiment of the other chains."""
        records = list(self.cfg.timing)
        for k in range(1, n):
            path = self.exp(k) / "data" / "timing.json"
            if path.exists():
                records += read_records(path)
        for other in self.cfg.other_chains:
            for path in sorted(Path(other).glob("experiment*/data/timing.json")):
                records += read_records(path)
        return records

    # ---------- 1. design ----------
    def design_theories(self, n: int) -> Dict[str, tuple]:
        """name -> (theory file, hypothesis) for the design: the carried theories, then every bar
        theory not carried (labelled as the bar in its hypothesis)."""
        models = self.models_input(n)
        out = {name: (models / f"{name}.py", h) for name, h in _hypotheses(models).items()}
        bar = Path(self.cfg.bar_dir or self.cfg.seeds_dir)
        for name, h in _hypotheses(bar).items():
            if name not in out:
                out[name] = (bar / f"{name}.py", "(Not in the live set: one of the theories the chain started from, "
                                                 f"kept in every design as the bar.) {h}")
        return out

    def design_fit(self, n: int, path: Path, name: str, corpus: Path) -> Tuple[Optional[Any], str]:
        """(fit, where it came from): the previous loop's fit on the same data if it has one (its
        corpus is this design's data so far, byte for byte), else a design fit at
        ``design_settings`` within ``design_fit_limit_sec``. Raises ModelFailure (the theory's
        failure, or the time limit)."""
        if n > 1:
            cached = cached_loop_fit(path, name, corpus, self.cfg.settings, self.exp(n - 1) / "loop" / ".fit_cache")
            if cached is not None:
                return cached, "the previous loop's fit"
        settings = replace(self.cfg.design_settings, seed=self.cfg.settings.seed)
        fitted = fit_cached(path, name, corpus, settings, self.exp(n) / ".fit_cache",
                            time_limit_sec=self.cfg.design_fit_limit_sec)
        return fitted, "a design fit" + ("" if fitted.converged else
                                          f" (not converged: {'; '.join(fitted.convergence_problems)})")

    def design_context(self, n: int) -> Dict[str, tuple]:
        """Each design theory (`design_theories`) with posterior draws of its parameters on the data
        so far, fitted concurrently (each in a child of its own, as the loop's). A theory whose fit
        fails or runs out of time is left out of this design and recorded in
        ``design_screened_out.json`` (an empty list: none was); fewer than two left raises."""
        from concurrent.futures import ThreadPoolExecutor

        from src.rsa.loop.fitting import ModelFailure
        from src.rsa.loop.orchestrator import default_fit_workers

        corpus = fresh_corpus(self.exp(n) / "prior_corpus.json", self.blocks_before(n))
        items = list(self.design_theories(n).items())

        def one(item):
            name, (path, _) = item
            try:
                fitted, source = self.design_fit(n, path, name, corpus)
            except ModelFailure as exc:
                return name, None, f"{type(exc).__name__}: {exc}"
            flat = fitted.flat()
            total = len(next(iter(flat.values())))
            take = np.unique(np.linspace(0, total - 1, min(total, DESIGN_DRAWS)).round().astype(int))
            return name, (path, {k: flat[k][take] for k in TheoryModel(path, name=name).params}), source

        with ThreadPoolExecutor(max_workers=max(1, min(len(items), default_fit_workers()))) as pool:
            results = list(pool.map(one, items))
        screened = [{"theory": name, "reason": why} for name, ctx, why in results if ctx is None]
        sources = {name: why for name, ctx, why in results if ctx is not None}
        (self.exp(n)).mkdir(parents=True, exist_ok=True)
        (self.exp(n) / SCREENED_OUT).write_text(json.dumps({"screened_out": screened, "fits": sources}, indent=1))
        for s in screened:
            print(f"[chain] experiment {n}'s design leaves out {s['theory']}: {s['reason']}", flush=True)
        context = {name: ctx for name, ctx, _ in results if ctx is not None}
        if len(context) < 2:
            raise RuntimeError(f"experiment {n}: {len(context)} theory left for the design after screening "
                               f"({screened}); a design needs two to tell apart")
        return context

    def history(self, n: int) -> List[EarlierParadigm]:
        """The chain's earlier paradigms, each with the theories' standing on its block at the end of
        the previous experiment's inner loop (no participant data)."""
        if n == 1:
            return []
        last = json.loads((self.exp(n - 1) / "loop" / "history.json").read_text())[-1]["standing"]
        out = []
        for k in range(1, n):
            block = f"exp{k}"
            rows = ["| theory | behind the best on this paradigm (held-out lpd, nats) |", "|---|---|"]
            for name, s in sorted(last.items(), key=lambda kv: kv[1]["by_block"].get(block, {}).get("behind", math.inf)):
                b = s["by_block"].get(block)
                rows.append(f"| {name} | " + ("not scored" if b is None else
                                               ("best" if b["behind"] == 0 else f"{b['behind']:.1f} ± {b['dse']:.1f}"))
                            + " |")
            spec = self.exp(k) / "design" / "frozen" / "paradigm.json"
            out.append(EarlierParadigm(json.loads(spec.read_text())["name"], spec, "\n".join(rows) + "\n"))
        return out

    def design(self, n: int) -> Path:
        frozen = self.exp(n) / "design" / "frozen"
        if (frozen / "record.json").exists():
            return frozen
        if set_aside(self.exp(n) / "design") is not None:
            shutil.rmtree(self.exp(n) / "design")  # run_designer lays its directory out itself
        context = self.design_context(n)
        run_designer(self.exp(n) / "design", chain=self.cfg.chain, experiment=n, theories=context,
                     hypotheses={k: h for k, (_, h) in self.design_theories(n).items() if k in context},
                     history=self.history(n),
                     spawn=self.designer_spawn,
                     pilot_participants=self.cfg.pilot_participants, participant_factory=self.participant_factory,
                     code_commit=self.cfg.code_commit, timing=self.timing(n))
        return frozen

    # ---------- 2. collect ----------
    def collect(self, n: int) -> Path:
        from src.rsa_open.corpus import paradigm_block
        from src.rsa_open.paradigm import load_paradigm, paradigm_likelihood
        from src.rsa_open.screens import scene_of

        out = self.exp(n) / "data" / "block.json"
        if out.exists():
            return out
        frozen = self.exp(n) / "design" / "frozen"
        paradigm = load_paradigm(frozen / "paradigm.json")
        lists = json.loads((frozen / "lists.json").read_text())
        link = paradigm_likelihood(paradigm)
        gt = self.cfg.ground_truth
        theory = TheoryModel(Path(gt.theory_path), name="ground_truth")
        params = {**gt.theory_params, **link.reference_params, **gt.link_params.get(link.name, {})}
        trials = [t for lst in lists["lists"] for t in lst["trials"]]
        probe_key = "probe" if link.side == "listener" else "target"
        rng = np.random.default_rng([self.cfg.seed, n])
        probes = [t.get(probe_key, int(rng.integers(len(t["objects"])))) for t in trials]
        utterances = [t.get("utterance", 0) for t in trials] if link.name == "binary_judgment" else None
        responses = simulate(theory, link, [scene_of(t) for t in trials], params,
                             jax.random.PRNGKey(self.cfg.seed * 1000 + n), probes=probes, utterances=utterances)
        rows = []
        for lst in lists["lists"]:
            for t in lst["trials"]:
                rows.append({"list_index": lst["list_index"], "phase": t["phase"], "unit": t["unit"],
                             "objects": t["objects"], "heard": t["heard"],
                             **{k: t[k] for k in ("probe", "target", "utterance") if k in t},
                             "response": responses[len(rows)]})
        data = out.parent
        data.mkdir(parents=True, exist_ok=True)
        with (data / "rows.jsonl").open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        write_corpus(out, [paradigm_block(f"exp{n}", paradigm, rows)])
        return out

    # ---------- 3. the inner loop ----------
    def model_loop(self, n: int) -> dict:
        from src.rsa_open.paradigm import load_paradigm

        loop_dir = self.exp(n) / "loop"
        if (loop_dir / "export.json").exists():
            return json.loads((loop_dir / "export.json").read_text())
        set_aside(loop_dir, keep=(".fit_cache",))
        corpus = fresh_corpus(self.exp(n) / "corpus.json", self.blocks_before(n + 1))
        notes = {}
        for k in range(1, n + 1):
            p = load_paradigm(self.exp(k) / "design" / "frozen" / "paradigm.json")
            notes[f"exp{k}"] = f"{p.summary} The prompt: \"{p.response.prompt}\""
        prev_ledger = self.exp(n - 1) / "loop" / "attempted_hypotheses.jsonl" if n > 1 else None
        cfg = OpenLoopConfig(corpus_path=corpus, seed_models_dir=self.models_input(n), results_dir=loop_dir,
                             settings=self.cfg.settings, paradigm_notes=notes, inherit_ledger=prev_ledger,
                             **self.cfg.loop)
        OpenLoop(cfg, self.theorist_spawn(loop_dir)).run()
        return json.loads((loop_dir / "export.json").read_text())

    def run(self) -> List[dict]:
        exports = []
        for n in range(1, self.cfg.n_experiments + 1):
            self.design(n)
            self.collect(n)
            exports.append(self.model_loop(n))
        return exports
