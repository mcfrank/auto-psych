"""Fit each chain's latest exported best model on one pooled dataset and save its
posterior-mean choice-class probabilities on novelty.plain_pool().

Pooled data: the existing plain trials (identical in every chain) + every live
experiment's data/modelled.csv so far (catch trials already removed), with live
participant ids prefixed by chain (they are numbered per chain). Fits go through
the loop's own loop_fit (campaign fit settings), each in a spawned child pinned to
one core (NO_TIME_LIMIT), three at once. Trial data stay in $WORK_ROOT/finals_pool/.

argv: <work_root> <out_json>
"""
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from src.rsa.cpus import pin_main_thread



# The fits run in spawned children, which import this module: everything runs under main().
def main():
    pin_main_thread()  # before JAX starts its thread pools (src.rsa.cpus)
    from src.rsa.fit import FitSettings
    from src.rsa.loop.fitting import NO_TIME_LIMIT, describe, loop_fit
    from src.rsa.loop.novelty import plain_pool, pool_digest, pool_record, posterior_mean_class_probs
    from src.rsa.model_file import RSAModel

    W = Path(sys.argv[1])
    OUT = Path(sys.argv[2])
    TREES = W.parent / "agent_trees"
    WORK = W / "finals_pool"
    (WORK / "models").mkdir(parents=True, exist_ok=True)

    CHAINS = {}
    for c in (0, 1, 2):
        tree = (W / "cells" / f"live_rsa_c{c}" / "agent_tree_id").read_text().strip()
        CHAINS[c] = TREES / tree / "repo" / "_runs" / "outer"

    # --- the pooled dataset ---
    existing = W / "cells" / "live_rsa_c0" / "private" / "existing_plain.csv"
    frames = [pd.read_csv(existing, low_memory=False)]
    sources = [dict(part="existing_plain", rows=len(frames[0]), sha256=hashlib.sha256(existing.read_bytes()).hexdigest()[:16])]
    for c, run in CHAINS.items():
        for exp in sorted(run.glob("experiment*")):
            f = exp / "data" / "modelled.csv"
            if not f.exists():
                continue
            df = pd.read_csv(f, low_memory=False)
            assert (df["condition"].astype(str) != "catch").all(), f"catch rows in {f}"
            df["participant_id"] = df["participant_id"].astype(str).str.replace("auto_psych:", f"auto_psych:c{c}:", n=1, regex=False)
            frames.append(df)
            sources.append(dict(part=f"chain {c} {exp.name}", rows=len(df), participants=int(df["participant_id"].nunique())))
    pooled = pd.concat(frames, ignore_index=True)
    pooled_path = WORK / "pooled.csv"
    pooled.to_csv(pooled_path, index=False)
    print("pooled rows", len(pooled), "| parts", [(s["part"], s["rows"]) for s in sources], flush=True)

    # --- the models: each chain's latest export's best ---
    models = {}
    for c, run in CHAINS.items():
        exports = sorted(run.glob("experiment*/model_loop/export.json"), key=lambda p: int(p.parent.parent.name[len("experiment"):]))
        export = exports[-1]
        best = json.loads(export.read_text())["best_model"]
        src = export.parent / "models" / f"{best}.py"
        dst = WORK / "models" / f"{best}.py"
        dst.write_text(src.read_text())
        models[best] = dict(chain=c, experiment=export.parent.parent.name, path=dst,
                            sha256=hashlib.sha256(src.read_bytes()).hexdigest()[:16])
    print("models", {k: (v["chain"], v["experiment"]) for k, v in models.items()}, flush=True)

    settings = FitSettings(num_warmup=1000, num_samples=1000, num_chains=4, seed=0, dense_mass=True)
    pool = plain_pool()


    def run(name):
        m = models[name]
        fitted = loop_fit(m["path"], name, pooled_path, settings, WORK / ".fit_cache", time_limit_sec=NO_TIME_LIMIT)
        probs = posterior_mean_class_probs(RSAModel(m["path"], name=name), fitted, pool)
        print("done", name, describe(fitted), flush=True)
        return name, fitted, probs


    with ThreadPoolExecutor(max_workers=len(models)) as ex:
        results = list(ex.map(run, models))

    choice_classes = [list(ctx.choice_classes()) for ctx in pool]  # each object's class: its first identical object
    n_entries = [len(cc) for cc in choice_classes]
    out = dict(
        note=("Posterior-mean choice-class probabilities on novelty.plain_pool(), flattened by context in pool "
              "order: context i occupies n_entries[i] entries (one per object); a class's probability sits at its "
              "first object's index and the class's other objects get 0 (src.rsa.fit.class_probs; choice_classes[i] "
              "gives each object's class). Each model fitted once on the pooled data below "
              "(src.rsa.loop.novelty.posterior_mean_class_probs, max 100 draws)."),
        pool=dict(size=len(pool), digest=pool_digest(pool), n_entries=n_entries, choice_classes=choice_classes,
                  contexts=[pool_record(c) for c in pool]),
        data=dict(rows=len(pooled), parts=sources,
                  participant_ids="live ids prefixed by chain (auto_psych:c<k>:...)"),
        fit_settings=dict(num_warmup=1000, num_samples=1000, num_chains=4, seed=0, dense_mass=True,
                          refit="loop_fit: one refit at target_accept 0.95 if the first fails the convergence gate"),
        models={},
    )
    for name, fitted, probs in results:
        assert len(probs) == sum(n_entries), (name, len(probs), sum(n_entries))
        m = models[name]
        out["models"][name] = dict(chain=m["chain"], exported_in=m["experiment"], model_sha256=m["sha256"],
                                   fit=describe(fitted), probs=[round(float(p), 6) for p in probs])
    OUT.write_text(json.dumps(out))
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
