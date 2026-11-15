"""Grouped 5-fold CV on the committed literature (E: pragmods, Sikos 2021, Mayn & Demberg 2026) for one theory.
argv: name path out_dir. Writes <out>/<name>.npz (out-of-fold lpd per trial) and <name>.json (timing, convergence)."""
import json, sys, time
from pathlib import Path
import numpy as np
from src.rsa.fit import FitSettings
from src.rsa_open.corpus import Block, literature_block, assign_folds, cv_corpus
from src.rsa_open.fit import TheoryModel
D = Path("src/pipelines/outer_loop/projects/rsa_reference/data")
name, path, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
parts = [literature_block(D / f) for f in ("pragmods_trials.csv", "sikos_2021_trials.csv", "mayn_demberg_2026_trials.csv")]
E = Block("E", parts[0].link, sum((b.contexts for b in parts), ()), sum((b.responses for b in parts), ()),
          sum((b.units for b in parts), ()), sum((b.sources for b in parts), ()), parts[0].fixed)
folds = {"E": assign_folds(E, 5, seed=0)}
t = time.time()
cv = cv_corpus(TheoryModel(path, name=name), [E], folds, 5,
               settings=FitSettings(num_warmup=500, num_samples=500, num_chains=4, dense_mass=True, seed=0))
np.savez(out / f"{name}.npz", lpd=cv.pointwise["E"], units=np.asarray(E.units), sources=np.asarray(E.sources))
json.dump({"name": name, "converged": bool(cv.converged), "sec": time.time() - t, "elpd": float(cv.pointwise["E"].sum())},
          open(out / f"{name}.json", "w"))
print(name, "elpd", round(float(cv.pointwise["E"].sum()), 1), "converged", cv.converged, f"{time.time()-t:.0f}s", flush=True)
