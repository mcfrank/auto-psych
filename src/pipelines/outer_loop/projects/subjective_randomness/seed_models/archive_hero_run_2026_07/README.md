# Hero-run seed pool (promoted 2026-07, retired 2026-08)

These four PyMC models are **promoted experiment outputs**, not hand-written
project assets: they are the best-fitting models discovered by the three human
replicate runs of `experiment3`, promoted in 2026-07 to become the live seed
pool for the hero run. `models_manifest.yaml` here is the pool's manifest as it
stood, including the per-model provenance comments (which run each model won,
and its posterior probability).

They were retired from the live pool in the 2026-08 seed reconciliation. The
pool had diverged from the recovery registry
(`src/subjective_randomness/pymc_model_families/`), which the 2026-08 fidelity
review had meanwhile consolidated onto four literature-faithful models. Two
disjoint "seed sets" were live at once, and because these four have **no
pure-Python twin** in `src/subjective_randomness/model_families/`, every helper
that resolves a seed name to its twin — `model_recovery.default_generating_params`
above all — raised `ModuleNotFoundError` for all of them. The registry is now
the single source of truth and the live pool mirrors it.

Nothing loads this directory automatically. The models are kept so that

* the promoted run artifacts stay reproducible (old runs seeded from here), and
* the pool can be resurrected deliberately, by pointing a harness's
  `seed_models_dir` at this directory or copying files back one at a time.

Resurrecting one for recovery work also means writing its pure-Python twin in
`src/subjective_randomness/model_families/`, or `tests/test_model_manifest.py`
will fail — by design.
