# Subjective Randomness PyMC Model Families

These modules are PyMC adapters for the canonical pure-Python model families in
`../model_families/`. They use precomputed numeric columns from
`../preprocess_data.py`, expose a deterministic `p_left`, and define the
Bernoulli response likelihood expected by `src.models.pymc_inference`.

Regenerate feature columns before fitting:

```bash
uv run python cc_pipeline/projects/subjective_randomness/preprocess_data.py \
  --input-csv cc_pipeline/projects/subjective_randomness/experiment1/data/responses.csv \
  --output-csv cc_pipeline/projects/subjective_randomness/data/responses.csv
```

Run model comparison:

```bash
uv run python -m src.model_comparison.posterior \
  --responses cc_pipeline/projects/subjective_randomness/data/responses.csv \
  --models-dir cc_pipeline/projects/subjective_randomness/pymc_model_families
```

