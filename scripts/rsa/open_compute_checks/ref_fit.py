"""One reference fit in this array task's own cache: the same theory, corpus and settings as a fit chain 1 ran on the lab
node, through the same child path. Writes one JSON of aggregates: where it ran, how long, its peak."""
import json, os, socket, sys, time
from pathlib import Path

from src.rsa.fit import FitSettings
from src.rsa_open.loop.fitting import cache_paths, fingerprint, fit_cached


def main():
    base = Path(sys.argv[1])
    name = "structural_alignment_pragmatic_tom"
    theory, corpus = base / "input" / f"{name}.py", base / "input" / "corpus.json"
    settings = FitSettings(dense_mass=True)
    fp = fingerprint(theory, corpus, settings)
    assert fp == "087f70e474a27c6a2d37", f"not the lab node's fit: {fp}"
    env = os.environ
    tag = f"{env['SLURM_ARRAY_JOB_ID']}_{env['SLURM_ARRAY_TASK_ID']}"
    cache = base / "runs" / tag
    t0 = time.monotonic()
    fit_cached(theory, name, corpus, settings, cache, time_limit_sec=3 * 3600, log_likelihood=False)
    meta = json.loads(cache_paths(cache, name, fp)[1].read_text())
    cpu = next(l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name"))
    (base / "results").mkdir(exist_ok=True)
    (base / "results" / f"{tag}.json").write_text(json.dumps(dict(
        task=tag, partition=env.get("SLURM_JOB_PARTITION"), host=socket.gethostname(), cpu=cpu,
        restarts=int(env.get("SLURM_RESTART_COUNT", 0)), wall_min=round((time.monotonic() - t0) / 60, 1),
        timing=meta["timing"], peak_hwm_mb=meta["peak_hwm_mb"], problems=meta["convergence_problems"]), indent=1))
    print("done", tag, flush=True)


if __name__ == "__main__":
    main()
