# Inner Model Loop Report

Each model below is ONE distinct cognitive hypothesis. The posterior mass shows which single hypothesis best explains the data — it is **not** a recipe to combine the top models into a blend.

- Best model: **smoothed_prototype_distance** (posterior=0.547, elpd_loo=-1714.40)
- Trials: 2560
- Models compared: 11

## Posterior over models (ELPD-LOO)

| model | posterior | elpd_loo |
| --- | --- | --- |
| smoothed_prototype_distance | 0.5470 | -1714.40 |
| iter1_candidate0 | 0.3256 | -1714.92 |
| iter0_candidate0 | 0.1247 | -1715.18 |
| prototype_similarity | 0.0027 | -1720.11 |
| bayesian_diagnosticity | 0.0000 | -1723.64 |
| inner_loop_model | 0.0000 | -1726.02 |
| encoding_compressibility | 0.0000 | -1765.14 |
| window_typicality | 0.0000 | -1769.42 |
| falk_konold_complexity | 0.0000 | -1774.59 |
| iter0_candidate2 | 0.0000 | -1769.01 |
| iter1_candidate2 | 0.0000 | -1758.61 |

## Hypotheses

- **smoothed_prototype_distance**: People judge the randomness of a sequence by comparing its feature proportions to a subjective ideal, but they estimate these proportions using Bayesian smoothing with subjective pseudo-counts, naturally tolerating extreme imbalances in short sequences because their prior pulls the estimates toward the ideal.
- **iter1_candidate0**: People judge the randomness of a sequence by comparing its feature proportions to a subjective ideal using Bayesian smoothing, but their sensitivity to these deviations follows a Gaussian-like generalization gradient. Instead of penalizing deviations linearly, they disproportionately penalize sequences that exhibit extreme, glaring deviations on any single feature dimension, while easily tolerating small differences from the prototype.
- **iter0_candidate0**: People judge the randomness of a sequence by its smoothed distance from a subjective prototype, but their ideal template includes an expectation for higher-order alternating motifs (like HTHT) in addition to basic heads/tails balance and simple bigram alternations.
- **prototype_similarity**: Random-looking sequences are close to a prototype with balanced H/T counts and an ideal alternation rate.
- **bayesian_diagnosticity**: Randomness is the log-likelihood ratio of a fair coin versus a regular process: a mixture of a complexity-penalized motif process (Griffiths et al. 2018) and a biased coin. Merges the former diagnosticity and statistical-inference accounts.
- **inner_loop_model**: People judge the randomness of a sequence by comparing its features (head proportion and alternation rate) to a subjective ideal, but their psychological penalty for deviations is scaled by the square root of the sequence length, reflecting an intuitive sensitivity to the standard error of small samples.
- **encoding_compressibility**: Random-looking sequences are those with low simple-description penalties: long runs, periodic templates, and imbalance.
- **window_typicality**: Random-looking sequences have a longest run typical of a fair coin seen through a limited memory window; over-long streaks look non-random (Hahn & Warren 2009).
- **falk_konold_complexity**: People judge the randomness of a sequence by its structural complexity when parsed into continuous alternating and repeating sub-sequences (Falk & Konold's difficulty of encoding), perceiving sequences with a lower rate of sub-sequences as simpler and therefore less random.
- **iter0_candidate2**: People judge the randomness of a sequence by its smoothed distance from a subjective prototype, but this prototype tracks the proportion of 4-item alternating motifs (like HTHT) instead of simple bigram alternations, viewing these longer, more complex alternations as the primary structural signature of local randomness.
- **iter1_candidate2**: People judge the randomness of a sequence by the probabilistic surprise of its macroscopic features (such as heads and alternations), rather than the likelihood of the specific sequence. They intuitively evaluate the exact Binomial probability of observing those specific feature counts under a subjective ideal model, naturally perceiving sequences with highly probable feature counts as more random while appropriately penalizing deviations more strictly in longer sequences.

## Distinguishability (arviz.compare, PSIS-LOO)

`elpd_diff` and `dse` are relative to the best model. A model is only clearly worse than the best when `elpd_diff > 2 * dse`; models within ~2·dse of the top are statistically indistinguishable. `LOO reliable` is False when PSIS-LOO flagged this model's estimate as untrustworthy (many high Pareto-k points) — its row should be read with caution.

| model | elpd_diff | dse | distinguishable from best | weight | LOO reliable |
| --- | --- | --- | --- | --- | --- |
| smoothed_prototype_distance ←selected | 0.00 | 0.00 | — (best) | 0.383 | yes |
| iter1_candidate0 | 0.52 | 3.87 | no (within ~2·dse) | 0.313 | yes |
| iter0_candidate0 | 0.78 | 0.36 | yes | 0.000 | yes |
| prototype_similarity | 5.70 | 3.88 | no (within ~2·dse) | 0.000 | yes |
| bayesian_diagnosticity | 9.24 | 7.73 | no (within ~2·dse) | 0.304 | yes |
| inner_loop_model | 11.62 | 4.49 | yes | 0.000 | yes |
| iter1_candidate2 | 44.21 | 9.35 | yes | 0.000 | yes |
| encoding_compressibility | 50.74 | 9.68 | yes | 0.000 | yes |
| iter0_candidate2 | 54.60 | 9.93 | yes | 0.000 | yes |
| window_typicality | 55.02 | 9.87 | yes | 0.000 | yes |
| falk_konold_complexity | 60.19 | 10.44 | yes | 0.000 | yes |
