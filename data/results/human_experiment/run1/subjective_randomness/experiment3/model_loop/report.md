# Inner Model Loop Report

Each model below is ONE distinct cognitive hypothesis. The posterior mass shows which single hypothesis best explains the data — it is **not** a recipe to combine the top models into a blend.

- Best model: **iter1_candidate0** (posterior=0.969, elpd_loo=-2286.26)
- Trials: 3840
- Models compared: 16

## Posterior over models (ELPD-LOO)

| model | posterior | elpd_loo |
| --- | --- | --- |
| iter1_candidate0 | 0.9688 | -2286.26 |
| linear_accumulated_typicality | 0.0192 | -2290.63 |
| asymmetric_alternation_typicality | 0.0075 | -2291.12 |
| inner_loop_model | 0.0023 | -2292.76 |
| power_law_accumulated_typicality | 0.0011 | -2293.12 |
| leaky_accumulated_typicality | 0.0009 | -2293.22 |
| iter1_candidate1 | 0.0001 | -2295.39 |
| prototype_similarity | 0.0000 | -2330.85 |
| encoding_compressibility | 0.0000 | -2411.18 |
| bayesian_diagnosticity | 0.0000 | -2335.55 |
| window_typicality | 0.0000 | -2460.92 |
| accumulated_alternation_typicality | 0.0000 | -2317.52 |
| iter0_candidate0 | 0.0000 | -2321.04 |
| iter0_candidate1 | 0.0000 | -2410.39 |
| iter0_candidate2 | 0.0000 | -2533.62 |
| iter1_candidate2 | 0.0000 | -2347.36 |

## Hypotheses

- **iter1_candidate0**: People evaluate the randomness of a sequence by accumulating a subjective sense of typicality over its length, computing each event's typicality as a penalty based on its distance from a mental prototype. Rather than assuming a strictly linear (City Block) or quadratic (Euclidean) penalty, this distance is computed using a freely inferred exponent (a Minkowski-like metric), allowing the model to naturally capture how human observers disproportionately scale the punishment for extreme feature deviations.
- **linear_accumulated_typicality**: People evaluate the randomness of a sequence by accumulating a subjective sense of typicality over its length, but their penalty for deviating from the ideal head proportion and alternation rate is linear (absolute difference) rather than quadratic, treating extreme deviations proportionally to small ones.
- **asymmetric_alternation_typicality**: People evaluate the randomness of a sequence by accumulating a subjective sense of typicality over its length, but their penalty for deviating from the ideal alternation rate is asymmetric: sequences that under-alternate (cluster) are heavily penalized, while those that over-alternate are penalized much less severely.
- **inner_loop_model**: People evaluate the randomness of a sequence by accumulating a subjective sense of typicality over its length, evaluating how well a sequence conforms to a mental prototype for heads and alternations.
- **power_law_accumulated_typicality**: People evaluate the randomness of a sequence by accumulating a subjective sense of typicality over its length, but this evidence integration exhibits diminishing returns scaling according to a power-law function of sequence length, which prevents extremely long sequences from disproportionately dominating judgments.
- **leaky_accumulated_typicality**: People evaluate the randomness of a sequence by accumulating a subjective sense of typicality over its events, but working memory for past events is leaky with exponential decay, meaning the total accumulated randomness score saturates for longer sequences.
- **iter1_candidate1**: People evaluate the randomness of a sequence by accumulating the Kullback-Leibler (KL) divergence (relative entropy) between its empirical features—specifically its proportion of heads and alternation rate—and a mental prototype. By acting as intuitive statisticians measuring information-theoretic divergence rather than geometric distance, they inherently apply a progressively steeper penalty for extreme deviations (such as highly imbalanced proportions), while naturally accumulating this evidence over the length of the sequence.
- **prototype_similarity**: Random-looking sequences are close to a prototype with balanced H/T counts and an ideal alternation rate.
- **encoding_compressibility**: Random-looking sequences are those with low simple-description penalties: long runs, periodic templates, and imbalance.
- **bayesian_diagnosticity**: Randomness is the log-likelihood ratio of a fair coin versus a regular process: a mixture of a complexity-penalized motif process (Griffiths et al. 2018) and a biased coin. Merges the former diagnosticity and statistical-inference accounts.
- **window_typicality**: Random-looking sequences have a longest run typical of a fair coin seen through a limited memory window; over-long streaks look non-random (Hahn & Warren 2009).
- **accumulated_alternation_typicality**: People evaluate the randomness of a sequence by accumulating a subjective sense of typicality over its length, but this typicality is based entirely on how closely the sequence's alternation rate matches a mental prototype, completely ignoring the overall proportion of heads and tails.
- **iter0_candidate0**: People evaluate the randomness of a sequence by accumulating a subjective sense of typicality over its length, but we refine how alternation typicality is evaluated. Rather than penalizing the sequence's global average alternation rate, people evaluate typicality at the level of individual runs (streaks): they accumulate a penalty for each run that grows quadratically with how much its length deviates from an ideal run length. This non-linear run-level penalty explains why sequences with atypically long streaks are judged as significantly less random even when their overall alternation rates match.
- **iter0_candidate1**: People evaluate the randomness of a sequence based solely on the proportion of the sequence occupied by its longest contiguous streak of identical outcomes, penalizing sequences where a single streak dominates the sequence length.
- **iter0_candidate2**: People evaluate the randomness of a sequence solely by assessing whether its longest streak of identical outcomes is typical for a random sequence of that length. They linearly penalize sequences based on the absolute deviation of their observed maximum run length from the mathematically expected maximum run length of a fair coin (approximately log2 of the sequence length).
- **iter1_candidate2**: People evaluate the randomness of a sequence based on its global, length-normalized properties rather than by accumulating evidence over time. They compute a penalty based on the squared deviation of the sequence's overall proportion of heads and alternation rate from an ideal mental prototype, penalizing extreme deviations more heavily, but evaluating this average typicality without multiplying it by the sequence length.

## Distinguishability (arviz.compare, PSIS-LOO)

`elpd_diff` and `dse` are relative to the best model. A model is only clearly worse than the best when `elpd_diff > 2 * dse`; models within ~2·dse of the top are statistically indistinguishable. `LOO reliable` is False when PSIS-LOO flagged this model's estimate as untrustworthy (many high Pareto-k points) — its row should be read with caution.

| model | elpd_diff | dse | distinguishable from best | weight | LOO reliable |
| --- | --- | --- | --- | --- | --- |
| iter1_candidate0 ←selected | 0.00 | 0.00 | — (best) | 0.054 | yes |
| linear_accumulated_typicality | 4.37 | 3.46 | no (within ~2·dse) | 0.391 | yes |
| asymmetric_alternation_typicality | 4.86 | 3.11 | no (within ~2·dse) | 0.000 | yes |
| inner_loop_model | 6.50 | 3.71 | no (within ~2·dse) | 0.000 | yes |
| power_law_accumulated_typicality | 6.86 | 4.89 | no (within ~2·dse) | 0.114 | yes |
| leaky_accumulated_typicality | 6.96 | 4.54 | no (within ~2·dse) | 0.000 | yes |
| iter1_candidate1 | 9.13 | 4.61 | no (within ~2·dse) | 0.000 | yes |
| accumulated_alternation_typicality | 31.25 | 8.09 | yes | 0.000 | no ⚠ |
| iter0_candidate0 | 34.78 | 10.81 | yes | 0.195 | yes |
| prototype_similarity | 44.59 | 11.64 | yes | 0.012 | yes |
| bayesian_diagnosticity | 49.29 | 11.60 | yes | 0.067 | yes |
| iter1_candidate2 | 61.10 | 13.52 | yes | 0.137 | yes |
| iter0_candidate1 | 124.13 | 19.16 | yes | 0.031 | yes |
| encoding_compressibility | 124.92 | 19.13 | yes | 0.000 | yes |
| window_typicality | 174.66 | 20.08 | yes | 0.000 | yes |
| iter0_candidate2 | 247.36 | 23.01 | yes | 0.000 | no ⚠ |
