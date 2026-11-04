"""Where a resume's time goes: resume(), score_chance() and the first standing(), on a copy of a loop directory.
No agent starts and nothing is fitted (a cache miss raises). Prints aggregates only."""
import json, sys, threading, time
from pathlib import Path

import src.rsa_open.loop.fitting as ft
import src.rsa_open.loop.orchestrator as o
from src.rsa.fit import FitSettings

work = Path(sys.argv[1])
lock, spent = threading.Lock(), {}


def timed(mod, attr):
    orig = getattr(mod, attr)

    def wrapper(*a, **k):
        t = time.monotonic()
        try:
            return orig(*a, **k)
        finally:
            with lock:
                n, s = spent.get(attr, (0, 0.0))
                spent[attr] = (n + 1, s + time.monotonic() - t)
    setattr(mod, attr, wrapper)


def refuse(*a, **k):
    raise RuntimeError(f"cache miss for {a[1] if len(a) > 1 else a}: this profile never fits")


def no_agents(*a, **k):
    raise RuntimeError("this profile starts no agent")


ft._fit_in_child = refuse
for attr in ("loop_fit", "pool_predictions", "heldout_lpd", "read_corpus", "plain_pool", "write_corpus"):
    timed(o, attr)
cfg = o.OpenLoopConfig(corpus_path=work / "corpus.json", seed_models_dir=work / "unused", results_dir=work / "loop",
                       settings=FitSettings(dense_mass=True))
loop = o.OpenLoop(cfg, no_agents)
out, marks = {}, []


def phase(name, fn):
    t = time.monotonic()
    fn()
    marks.append(dict(phase=name, minutes=round((time.monotonic() - t) / 60, 1), rss_gb=round(ft.own_rss() / 2**30, 1),
                      spent={k: (v[0], round(v[1] / 60, 1)) for k, v in spent.items()}))
    print(json.dumps(marks[-1]), flush=True)


phase("resume", loop.resume)
out["live"] = len(loop.live)
phase("score_chance", loop.score_chance)
phase("standing", loop.standing)
out["phases"] = marks
(work / "profile.json").write_text(json.dumps(out, indent=1))
print(json.dumps(dict(live=out["live"], total_minutes=round(sum(m["minutes"] for m in marks), 1))), flush=True)
