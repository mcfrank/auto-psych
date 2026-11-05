"""One step of a step-mode rehearsal with no agents: `HANDOFF_stepmode.md` §2's machinery (real arrays,
continuations, dependencies, journals) on chain 1's experiment 2, where each "agent session" copies what
that session's agent wrote there. No model is called, so the jobs start no agent session (rule 10 of
/etc/agents/AGENTS.md). It goes through `run_chain.main`, so a step's exit codes are the real ones."""

from __future__ import annotations

import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from types import SimpleNamespace

import yaml

from src.rsa.cpus import pin_main_thread
from src.rsa.fit import FitSettings
from src.rsa_open import run_chain
from src.rsa_open.chain import clear_stage
from src.rsa_open.loop.orchestrator import OpenLoop, OpenLoopConfig


class ReplayTheorist:
    """Writes into a session's directory the files that session's agent wrote in the source loop."""

    FILES = ("candidate.py", "hypothesis.md", "model_name.txt")

    def __init__(self, source_loop: Path, log: Path) -> None:
        self.source, self.log = source_loop, log

    def __call__(self, cdir, prompt) -> bool:
        cdir = Path(cdir)
        src = self.source / cdir.parent.name / cdir.name
        cdir.mkdir(parents=True, exist_ok=True)
        copied = []
        for name in self.FILES:
            if (src / name).exists():
                shutil.copyfile(src / name, cdir / name)
                copied.append(name)
        if (src / "test_stats").is_dir():
            shutil.copytree(src / "test_stats", cdir / "test_stats", dirs_exist_ok=True)
            copied.append("test_stats")
        with self.log.open("a") as f:  # a session replayed twice shows here (the journal should prevent it)
            f.write(json.dumps({"session": f"{cdir.parent.name}/{cdir.name}", "copied": copied}) + "\n")
        return True


@dataclass
class ReplayChainConfig:
    """What `run_chain.main` reads of a chain's config in step mode (its launch: one experiment)."""
    chain: str = "replay"
    n_experiments: int = 1
    stop_after_experiment: Optional[int] = None


class ReplayChain:
    """`Chain.model_loop`'s one loop, on the source's corpus, starting set and inherited ledger."""

    def __init__(self, a) -> None:
        self.a = a
        self.cfg = ReplayChainConfig()

    def first_unfinished(self) -> int:
        return 2 if (Path(self.a.run_dir) / "loop" / "export.json").exists() else 1

    def run(self) -> list:
        loop = Path(self.a.run_dir) / "loop"
        resume = (loop / "history.json").exists()
        if not resume:
            clear_stage(loop, keep=(".fit_cache", ".scores"))  # as chain.model_loop does in a step
        cfg = OpenLoopConfig(corpus_path=Path(self.a.corpus), seed_models_dir=Path(self.a.seeds), results_dir=loop,
                             max_iterations=self.a.rounds, candidate_count=self.a.candidates, cv_folds=5,
                             settings=FitSettings(dense_mass=True), inherit_ledger=Path(self.a.ledger))
        theorist = ReplayTheorist(Path(self.a.source_loop), Path(self.a.run_dir) / "replayed_sessions.jsonl")
        OpenLoop(cfg, theorist).run(resume=resume)
        return [json.loads((loop / "export.json").read_text())]


def subset_seeds(source: Path, out: Path, keep: list) -> Path:
    """The source's starting set, or only ``keep`` of it (a smaller rehearsal)."""
    if not keep:
        return source
    entries = [e for e in yaml.safe_load((source / "models_manifest.yaml").read_text())["models"] if e["name"] in keep]
    if {e["name"] for e in entries} != set(keep):
        raise ValueError(f"not all of {keep} are in {source}")
    out.mkdir(parents=True, exist_ok=True)
    for e in entries:
        shutil.copyfile(source / f"{e['name']}.py", out / f"{e['name']}.py")
    (out / "models_manifest.yaml").write_text(yaml.safe_dump({"models": entries}))
    return out


if __name__ == "__main__":
    pin_main_thread()
    run_dir, input_dir, source_loop, rounds, candidates = sys.argv[1:6]
    keep = sys.argv[6].split(",") if len(sys.argv) > 6 and sys.argv[6] else []
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    seeds = subset_seeds(Path(input_dir) / "models_input", Path(run_dir) / "seeds", keep)
    args = SimpleNamespace(run_dir=run_dir, corpus=str(Path(input_dir) / "corpus.json"), seeds=str(seeds),
                           ledger=str(Path(input_dir) / "attempted_hypotheses.jsonl"), source_loop=source_loop,
                           rounds=int(rounds), candidates=int(candidates))
    run_chain.build = ReplayChain  # the only change to run_chain.main: the chain it runs
    run_chain.main(args)
