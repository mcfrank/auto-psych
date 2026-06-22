# Inner Model Loop Report

Each model below is ONE distinct cognitive hypothesis. The posterior mass shows which single hypothesis best explains the data — it is **not** a recipe to combine the top models into a blend.

- Best model: **iter1_candidate0** (posterior=1.000, elpd_loo=-771.49)
- Trials: 1280
- Models compared: 10

## Posterior over models (ELPD-LOO)

| model | posterior | elpd_loo |
| --- | --- | --- |
| iter1_candidate0 | 1.0000 | -771.49 |
| iter0_candidate0 | 0.0000 | -784.88 |
| prototype_similarity | 0.0000 | -793.68 |
| encoding_compressibility | 0.0000 | -841.68 |
| bayesian_diagnosticity | 0.0000 | -794.24 |
| window_typicality | 0.0000 | -855.68 |
| iter0_candidate1 | 0.0000 | -887.75 |
| iter0_candidate2 | 0.0000 | -886.66 |
| iter1_candidate1 | 0.0000 | -788.75 |
| iter1_candidate2 | 0.0000 | -871.51 |

## Hypotheses

- **iter1_candidate0**: People evaluate the randomness of a sequence by accumulating a subjective sense of typicality over its length, rather than merely evaluating its average properties. They evaluate how well a sequence conforms to a mental prototype—represented by an ideal proportion of heads and alternations—and integrate this fit across all events. Because this typicality is accumulated, longer sequences that match the prototype accrue a higher total randomness score, while longer sequences that deviate strongly accumulate a heavier penalty, explaining why humans prefer longer sequences when average rates are equal.
- **iter0_candidate0**: People judge randomness by comparing a sequence to a mental prototype, but this prototype is subjectively biased: it possesses an ideal proportion of heads and an ideal alternation rate that may deviate from objective fairness. Sequences are perceived as more random when their proportion of heads and alternations have a smaller squared deviation from these subjective ideals.
- **prototype_similarity**: Random-looking sequences are close to a prototype with balanced H/T counts
and an ideal alternation rate.
- **encoding_compressibility**: Random-looking sequences are those with low simple-description penalties:
long runs, periodic templates, and imbalance.
- **bayesian_diagnosticity**: Randomness is the log-likelihood ratio of a fair coin versus a regular
process: a mixture of a complexity-penalized motif process (Griffiths et
al. 2018) and a biased coin. Merges the former diagnosticity and
statistical-inference accounts.
- **window_typicality**: Random-looking sequences have a longest run typical of a fair coin seen
through a limited memory window; over-long streaks look non-random (Hahn &
Warren 2009).
- **iter0_candidate1**: People judge randomness by searching for local "clumps" of identical outcomes, specifically associating moderate-length runs (pairs and triplets) with the natural clumpiness of a stochastic process. Sequences are perceived as more random when a higher proportion of their outcomes belong to these moderate-length clusters, as this simultaneously avoids the artificial regularity of strict alternation and the perceived non-randomness of overly long streaks.
- **iter0_candidate2**: People evaluate the randomness of a sequence based on its proportion of tails, exhibiting a cognitive bias where tails are perceived as inherently more random than heads. Thus, when comparing two sequences, they are more likely to judge the sequence with a higher proportion of tails as the more random one.
- **iter1_candidate1**: People judge the randomness of a sequence by comparing it to a mental prototype that embodies the expected natural variability of a stochastic process, actively distrusting sequences that appear artificially "too perfect." Rather than expecting an exactly balanced sequence, they recognize that flawless balance is statistically rare and thus expect a moderate, typical degree of outcome imbalance alongside a specific alternation rate. Sequences are perceived as more random when their degree of imbalance and alternation rate have a smaller squared deviation from these subjective, non-zero ideal expectations.
- **iter1_candidate2**: People judge the randomness of a sequence by comparing it to a mental prototype, but instead of focusing on global alternation rates, they evaluate the local 'clumpiness' of the sequence via its density of repeated motifs. Sequences are perceived as more random when their proportion of heads and their proportion of repeated motifs have a smaller squared deviation from a subjectively biased ideal.

## Distinguishability (arviz.compare, PSIS-LOO)

`elpd_diff` and `dse` are relative to the best model. A model is only clearly worse than the best when `elpd_diff > 2 * dse`; models within ~2·dse of the top are statistically indistinguishable. `LOO reliable` is False when PSIS-LOO flagged this model's estimate as untrustworthy (many high Pareto-k points) — its row should be read with caution.

| model | elpd_diff | dse | distinguishable from best | weight | LOO reliable |
| --- | --- | --- | --- | --- | --- |
| iter1_candidate0 ←selected | 0.00 | 0.00 | — (best) | 0.762 | yes |
| iter0_candidate0 | 13.39 | 7.50 | no (within ~2·dse) | 0.170 | yes |
| iter1_candidate1 | 17.26 | 7.43 | yes | 0.000 | no ⚠ |
| prototype_similarity | 22.19 | 7.54 | yes | 0.000 | yes |
| bayesian_diagnosticity | 22.75 | 6.56 | yes | 0.000 | yes |
| encoding_compressibility | 70.19 | 13.55 | yes | 0.068 | yes |
| window_typicality | 84.19 | 13.27 | yes | 0.000 | yes |
| iter1_candidate2 | 100.03 | 13.69 | yes | 0.000 | yes |
| iter0_candidate2 | 115.18 | 14.53 | yes | 0.000 | yes |
| iter0_candidate1 | 116.26 | 14.61 | yes | 0.000 | no ⚠ |
