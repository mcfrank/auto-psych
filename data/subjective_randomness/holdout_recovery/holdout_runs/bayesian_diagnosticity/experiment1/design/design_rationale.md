# Experiment Design Rationale

- **Number of stimuli**: 30 candidate pairs
- **EIG Range**: 0.0281 – 0.3170

## Discrimination Strategy

The candidate pairs were selected to maximize Expected Information Gain (EIG) in order to confidently distinguish between two cognitive models of subjective randomness: `prototype_similarity` and `encoding_compressibility`.

- **`prototype_similarity`** assumes human judges look for sequences that represent a "balanced" prototype with an expected alternation rate (e.g., matching the true generative process of a fair coin).
- **`encoding_compressibility`** assumes judges are sensitive to cognitive compressibility, penalizing sequences with easily-described patterns like long runs or periodic templates.

By scoring our candidate space via prior-predictive EIG, the selected design specifically targets regions of the stimulus space where these two theories diverge—for instance, sequences that might have balanced H/T proportions (favored by similarity) but contain highly compressible periodic structures or long consecutive runs (penalized by compressibility).
