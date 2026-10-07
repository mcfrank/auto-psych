"""Resume the OOM-killed smoke-test-2 loop after round 2's agents had finished.

Rebuilds RSALoop's state from disk (the live set in models/ with fits from the
content-addressed cache, the ledger, history.json), then runs round 2 with a
spawner that reuses the three finished round-2 agents' outputs (their
candidate.py was written before the kill) and spawns real sandboxed agents
for any retry or repair, then the end-of-run prune and export.
"""
import json, sys
from pathlib import Path
from src.pipelines.inner_loop.hypothesis_ledger import HypothesisLedger, LEDGER_FILENAME
from src.models.model_manifest import read_manifest_entries
from src.rsa.dataset import load_forced_choice
from src.rsa.fit import FitSettings
from src.rsa.loop.gates import GateConfig, fit_with_refit
from src.rsa.loop.novelty import novelty_pool, pool_digest, posterior_mean_class_probs
from src.rsa.loop.orchestrator import LoopConfig, RSALoop, Live, cluster_ids, coding_agent_spawner
from src.rsa.model_file import RSAModel
from src.runtime.token_usage import start_usage_log, write_usage_report  # noqa

R = Path("data/rsa/loop_smoke2")
cfg = LoopConfig(responses_path=Path("data/rsa/combined_trials.csv"),
                 seed_models_dir=Path("src/pipelines/outer_loop/projects/rsa_reference/seed_models"),
                 results_dir=R, max_iterations=2, candidate_count=3,
                 settings=FitSettings(num_warmup=500, num_samples=500, num_chains=2))
real = coding_agent_spawner(models_dir=R / "models", responses_path=R / "responses.csv", timeout_sec=2400,
                            backend=None, model="google/gemini-3.8-flash", agent_root=None, sandbox=True)
FINISHED = {(R / "round_2" / f"candidate_{i}").resolve() for i in (1, 2, 3)}

def spawn(cdir: Path, prompt: str) -> bool:
    if cdir.resolve() in FINISHED:
        if not (cdir / "candidate.py").exists():
            raise RuntimeError(f"{cdir} has no candidate.py; cannot reuse")
        print(f"[resume] reusing the finished agent's output in {cdir}", flush=True)
        return True
    return real(cdir, prompt)


def main():
    global loop
    loop = RSALoop(cfg, spawn)
    loop.trials = load_forced_choice(loop.responses)
    loop.clusters = cluster_ids(loop.trials.frame)
    loop.pool = novelty_pool()
    saved = json.loads((R / "novelty_pool.json").read_text())["digest"]
    if pool_digest(loop.pool) != saved:
        raise RuntimeError("novelty pool differs from the run's")
    loop.ledger = HypothesisLedger(R / LEDGER_FILENAME)
    loop.gate_cfg = GateConfig(responses_path=loop.responses, cache_dir=R / ".fit_cache", settings=cfg.settings,
                               novelty_threshold=cfg.novelty_threshold, time_limit_sec=cfg.fit_time_limit_sec)
    contexts = {e.name: e.context for e in loop.ledger.entries() if e.outcome == "admitted"}
    for entry in read_manifest_entries(loop.models_dir):
        name = entry["name"]
        path = loop.models_dir / f"{name}.py"
        fitted = fit_with_refit(path, name, loop.gate_cfg)
        model = RSAModel(path, name=name)
        loop.live[name] = Live(name, entry.get("rationale", ""), model, fitted,
                               posterior_mean_class_probs(model, fitted, loop.pool), contexts.get(name, "seed"))
        print(f"[resume] loaded {name} from the cache", flush=True)
    loop.history = json.loads((R / "history.json").read_text())
    loop.step = len(loop.history)
    print(f"[resume] step {loop.step}, live {list(loop.live)}", flush=True)
    marker = start_usage_log(R / "token_usage.jsonl")
    try:
        loop.run_round(1)
        final = loop.end()
    finally:
        write_usage_report(R, marker, heading="RSA inner loop")
    print(f"best model: {final['best_model']}; live: {sorted(final['standing'])}")


if __name__ == "__main__":
    main()
