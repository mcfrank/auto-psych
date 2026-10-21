"""One chain of the open campaign, experiment by experiment (docs/auto_open/PLAN.md §4-§5, Phase 2b).

    experiment<n>/
      models_input/        the theories this experiment starts from (the seeds, then the previous
                           experiment's live set) with their manifest
      prior_corpus.json    the data so far: E and every earlier experiment's block
      design/              the designer's cycles and frozen/ (`src.rsa_open.designer.run_designer`)
      data/rows.jsonl      every participant's trials and responses
      data/block.json      the experiment's test trials as a corpus block
      corpus.json          E, every earlier block and this one
      loop/                the inner loop (`src.rsa_open.loop.orchestrator.OpenLoop`) and its export

Each stage is skipped when its output exists, so a chain resumes where it stopped:

1. **design**: the carried theories are fitted to the data so far; their posterior draws (theory
   parameters only) and hypotheses, the chain's earlier paradigms with each theory's standing on
   them, and the timing records go to the designer, whose frozen paradigm runs;
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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import jax
import numpy as np
import yaml

from src.models.model_manifest import read_manifest_entries
from src.rsa.fit import FitSettings
from src.rsa_open.corpus import Block
from src.rsa_open.designer import EarlierParadigm, SpawnFn as DesignerSpawn, run_designer
from src.rsa_open.fit import TheoryModel, simulate
from src.rsa_open.loop.fitting import loop_fit, read_corpus, write_corpus
from src.rsa_open.loop.orchestrator import OpenLoop, OpenLoopConfig
from src.rsa_open.timing import CLOSED_CAMPAIGN, TimingRecord

DESIGN_DRAWS = 200


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
    n_experiments: int = 3
    loop: Dict[str, Any] = field(default_factory=dict)
    """OpenLoopConfig overrides (max_iterations, candidate_count, settings, ...)."""
    settings: FitSettings = field(default_factory=lambda: FitSettings(dense_mass=True))
    pilot_participants: int = 12
    timing: Sequence[TimingRecord] = CLOSED_CAMPAIGN
    code_commit: Optional[str] = None
    seed: int = 0


def _hypotheses(models_dir: Path) -> Dict[str, str]:
    return {e["name"]: e.get("rationale", "") for e in read_manifest_entries(models_dir)}


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
    def __init__(self, cfg: ChainConfig, *, designer_spawn: DesignerSpawn, theorist_spawn: Callable[[Path, str], bool],
                 participant_factory=None) -> None:
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

    # ---------- 1. design ----------
    def design_context(self, n: int) -> Dict[str, tuple]:
        """Each carried theory with posterior draws of its parameters on the data so far."""
        models = self.models_input(n)
        corpus = write_corpus(self.exp(n) / "prior_corpus.json", self.blocks_before(n))
        theories = {}
        for name in _hypotheses(models):
            path = models / f"{name}.py"
            fitted = loop_fit(path, name, corpus, self.cfg.settings, self.exp(n) / ".fit_cache", time_limit_sec=math.inf)
            flat = fitted.flat()
            theory = TheoryModel(path, name=name)
            total = len(next(iter(flat.values())))
            take = np.unique(np.linspace(0, total - 1, min(total, DESIGN_DRAWS)).round().astype(int))
            theories[name] = (path, {k: flat[k][take] for k in theory.params})
        return theories

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
        models = self.models_input(n)
        run_designer(self.exp(n) / "design", chain=self.cfg.chain, experiment=n, theories=self.design_context(n),
                     hypotheses=_hypotheses(models), history=self.history(n), spawn=self.designer_spawn,
                     pilot_participants=self.cfg.pilot_participants, participant_factory=self.participant_factory,
                     code_commit=self.cfg.code_commit, timing=self.cfg.timing)
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
        corpus = write_corpus(self.exp(n) / "corpus.json", self.blocks_before(n + 1))
        notes = {}
        for k in range(1, n + 1):
            p = load_paradigm(self.exp(k) / "design" / "frozen" / "paradigm.json")
            notes[f"exp{k}"] = f"{p.summary} The prompt: \"{p.response.prompt}\""
        prev_ledger = self.exp(n - 1) / "loop" / "attempted_hypotheses.jsonl" if n > 1 else None
        cfg = OpenLoopConfig(corpus_path=corpus, seed_models_dir=self.models_input(n), results_dir=loop_dir,
                             settings=self.cfg.settings, paradigm_notes=notes, inherit_ledger=prev_ledger,
                             **self.cfg.loop)
        OpenLoop(cfg, self.theorist_spawn).run()
        return json.loads((loop_dir / "export.json").read_text())

    def run(self) -> List[dict]:
        exports = []
        for n in range(1, self.cfg.n_experiments + 1):
            self.design(n)
            self.collect(n)
            exports.append(self.model_loop(n))
        return exports
