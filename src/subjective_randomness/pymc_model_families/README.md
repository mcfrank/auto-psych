# Subjective Randomness PyMC Model Families

These modules are PyMC adapters for the canonical pure-Python model families in
`../model_families/`. They use precomputed numeric columns from the featurizer in
`src.subjective_randomness.features`, expose a deterministic `p_left`, and define
the Bernoulli response likelihood expected by `src.models.pymc_inference`.

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

