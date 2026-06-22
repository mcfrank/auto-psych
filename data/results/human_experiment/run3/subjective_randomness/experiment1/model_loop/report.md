# Inner Model Loop Report

Each model below is ONE distinct cognitive hypothesis. The posterior mass shows which single hypothesis best explains the data — it is **not** a recipe to combine the top models into a blend.

- Best model: **iter1_candidate2** (posterior=0.932, elpd_loo=-859.70)
- Trials: 1280
- Models compared: 10

## Posterior over models (ELPD-LOO)

| model | posterior | elpd_loo |
| --- | --- | --- |
| iter1_candidate2 | 0.9323 | -859.70 |
| iter1_candidate0 | 0.0521 | -862.99 |
| prototype_similarity | 0.0154 | -864.21 |
| iter1_candidate1 | 0.0002 | -868.47 |
| bayesian_diagnosticity | 0.0000 | -867.87 |
| iter0_candidate0 | 0.0000 | -871.43 |
| encoding_compressibility | 0.0000 | -889.63 |
| window_typicality | 0.0000 | -887.51 |
| iter0_candidate1 | 0.0000 | -877.19 |
| iter0_candidate2 | 0.0000 | -886.39 |

## Hypotheses

- **iter1_candidate2**: People judge the randomness of a sequence by comparing its features (head proportion and alternation rate) to a subjective ideal, but their psychological penalty for deviations is scaled by the square root of the sequence length, reflecting an intuitive sensitivity to the standard error of small samples. This statistical-evidence weighting means that while people still penalize non-ideal proportions, they are far more tolerant of extreme imbalance in very short sequences where the small sample size provides weak evidence of non-randomness.
- **iter1_candidate0**: People judge the randomness of a sequence by evaluating its distance from an ideal subjective prototype, but their sensitivity to deviations follows a psychophysical power law with a compressive exponent. This sub-linear scaling means that while initial departures from the prototype are penalized, further deviations toward extremes (like perfect imbalance) result in diminishing additional penalties.
- **prototype_similarity**: Random-looking sequences are close to a prototype with balanced H/T counts
and an ideal alternation rate.
- **iter1_candidate1**: People judge the randomness of a sequence by comparing it to an ideal prototype, but they evaluate its departure from this ideal using the absolute difference in the raw counts of heads and alternations, rather than their proportions. By using unnormalized counts, people naturally exhibit length-dependent tolerance, applying much smaller penalties to perfectly imbalanced short sequences than to perfectly imbalanced long sequences.
- **bayesian_diagnosticity**: Randomness is the log-likelihood ratio of a fair coin versus a regular
process: a mixture of a complexity-penalized motif process (Griffiths et
al. 2018) and a biased coin. Merges the former diagnosticity and
statistical-inference accounts.
- **iter0_candidate0**: People judge the randomness of a sequence based on its Gaussian similarity to an ideal subjective prototype, meaning the sequence's perceived randomness decays exponentially with its squared distance in feature space (proportion of heads and alternations) from the expected prototype. This formulation penalizes extreme deviations more severely than a linear distance metric.
- **encoding_compressibility**: Random-looking sequences are those with low simple-description penalties:
long runs, periodic templates, and imbalance.
- **window_typicality**: Random-looking sequences have a longest run typical of a fair coin seen
through a limited memory window; over-long streaks look non-random (Hahn &
Warren 2009).
- **iter0_candidate1**: People judge the randomness of a sequence by evaluating the statistical typicality of its macroscopic features. They compute the joint Binomial probability of observing the sequence's specific number of heads and alternations under their subjective expectations for a random process, perceiving sequences with highly improbable feature counts as less random.
- **iter0_candidate2**: People judge the randomness of a sequence by the diversity of its run lengths, perceiving sequences as more random when they contain an unpredictable mix of short and long streaks, which is evaluated as the Shannon entropy of the sequence's run-length distribution.

## Distinguishability (arviz.compare, PSIS-LOO)

`elpd_diff` and `dse` are relative to the best model. A model is only clearly worse than the best when `elpd_diff > 2 * dse`; models within ~2·dse of the top are statistically indistinguishable. `LOO reliable` is False when PSIS-LOO flagged this model's estimate as untrustworthy (many high Pareto-k points) — its row should be read with caution.

| model | elpd_diff | dse | distinguishable from best | weight | LOO reliable |
| --- | --- | --- | --- | --- | --- |
| iter1_candidate2 ←selected | 0.00 | 0.00 | — (best) | 0.652 | yes |
| iter1_candidate0 | 3.28 | 4.21 | no (within ~2·dse) | 0.026 | yes |
| prototype_similarity | 4.51 | 4.72 | no (within ~2·dse) | 0.035 | yes |
| bayesian_diagnosticity | 8.17 | 6.59 | no (within ~2·dse) | 0.288 | yes |
| iter1_candidate1 | 8.77 | 3.22 | yes | 0.000 | yes |
| iter0_candidate0 | 11.73 | 6.11 | no (within ~2·dse) | 0.000 | yes |
| iter0_candidate1 | 17.49 | 5.81 | yes | 0.000 | yes |
| iter0_candidate2 | 26.69 | 7.59 | yes | 0.000 | yes |
| window_typicality | 27.81 | 6.86 | yes | 0.000 | yes |
| encoding_compressibility | 29.93 | 7.56 | yes | 0.000 | yes |
