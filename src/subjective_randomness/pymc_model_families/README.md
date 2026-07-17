# Subjective Randomness PyMC Model Families

These modules are PyMC adapters for the canonical pure-Python model families in
`../model_families/`. They use precomputed numeric columns from the featurizer in
`src.subjective_randomness.features`, expose a deterministic `p_left`, and define
the Bernoulli response likelihood expected by `src.models.pymc_inference`.

Since the hero-run seed promotion (2026-07) this directory is also the frozen
**recovery registry**: the recovery harnesses (`model_recovery.py`,
`holdout_recovery.py`, and the impossible variant) point their
`seed_models_dir` here, because ground-truth generation and fixed-parameter
baselines need the pure-Python family twins that only these original models
have. The *live* seed pool
(`src/pipelines/outer_loop/projects/subjective_randomness/seed_models/`) is
separate — it holds the promoted replicate winners (which have no twins) and
evolves independently of this registry.

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

