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

The 2026-08 seed-model fidelity review **replaced** the original four models
with literature-faithful versions (each with a pure-Python twin and
paper-derived test vectors in `tests/test_literature_model_families.py`):

| Active model | Replaces | Faithful to |
| --- | --- | --- |
| `falk_konold_dp` | `encoding_compressibility` | Falk & Konold (1997) Difficulty Predictor, minimal parse |
| `motif_hmm` | `bayesian_diagnosticity` | Griffiths et al. (2018) motif HMM, row-normalised, parse-marginalised |
| `finite_experience_occurrence` | `window_typicality` | Hahn & Warren (2009) occurrence probability in finite experience |
| `local_representativeness` | `prototype_similarity` | Kahneman & Tversky (1972) local representativeness |

Only manifest-listed models are active — for recovery, the fitted and
no-learning baselines, and the EIG design defaults
(`stimulus_design.default_model_family_names` reads this manifest as the
single source of truth). The superseded originals' `.py` files and twins
remain on disk solely so pre-consolidation run artifacts can be refit; do not
add them back to the manifest. New models enter the live hero-run pool only
via the standard recovery/holdout comparison.

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

