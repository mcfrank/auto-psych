# Subjective Randomness PyMC Model Families

These modules are PyMC adapters for the pure-Python model families in
`../model_families/`. They use precomputed numeric columns from the featurizer in
`src.subjective_randomness.features`, expose a deterministic `p_left`, and define
the Bernoulli response likelihood expected by `src.models.pymc_inference`.

This directory is the **recovery registry**, and its `models_manifest.yaml` is
the single source of truth for which models are active. The recovery harnesses
(`model_recovery.py`, `holdout_recovery.py`, and the impossible variant) point
their `seed_models_dir` here, because ground-truth generation and
fixed-parameter baselines need the pure-Python family twins. The outer loop's
live seed pool
(`src/pipelines/outer_loop/projects/subjective_randomness/seed_models/`) is a
**mirror** of this manifest, not an independent set: to change the seed set,
edit the manifest here and copy it plus the model files across.
`tests/test_model_manifest.py` and
`tests/test_subjective_randomness_seed_recovery.py` fail if the two diverge, or
if a manifest name has no pure-Python twin. (They diverged once, between the
hero-run promotion of 2026-07 and the reconciliation of 2026-08, which left
`model_recovery.default_generating_params` raising `ModuleNotFoundError` for
every pool model; the retired winners are archived under the seed pool's
`archive_hero_run_2026_07/`.)

The 2026-08 seed-model fidelity review **replaced** the original four models
with closer paper-anchored versions (each with a pure-Python twin and
paper-derived test vectors in `tests/test_literature_model_families.py`):

| Active model | Replaces | Faithful to |
| --- | --- | --- |
| `falk_konold_dp` | `encoding_compressibility` | Falk & Konold (1997) Difficulty Predictor, minimal parse |
| `motif_stack` | `bayesian_diagnosticity` | Griffiths et al. (2018) four-motif stack automaton |
| `finite_experience_occurrence` | `window_typicality` | Hahn & Warren (2009) occurrence probability in finite experience |
| `local_representativeness` | `prototype_similarity` | Explicit quantitative operationalization of Kahneman & Tversky (1972) |

Only manifest-listed models are active — for recovery, the fitted and
no-learning baselines, the outer loop's seed pool, and the EIG design defaults
(`stimulus_design.default_model_family_names` reads this manifest directly).
The superseded originals' `.py` files and twins remain on disk solely so
pre-consolidation run artifacts can be refit; do not add them back to the
manifest. A newly discovered model earns its place here only through the
standard recovery/holdout comparison — and needs a pure-Python twin in
`../model_families/` before it can be added.

Regenerate feature columns before fitting:

```bash
uv run python scripts/subjective_randomness/preprocess.py \
  --input-csv data/subjective_randomness/experiment1/responses.csv \
  --output-csv data/subjective_randomness/responses.csv
```

Run model comparison:

```bash
uv run python -m src.model_comparison.posterior \
  --responses data/subjective_randomness/responses.csv \
  --models-dir src/subjective_randomness/pymc_model_families
```
