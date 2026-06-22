# Inner Model Loop Report

Each model below is ONE distinct cognitive hypothesis. The posterior mass shows which single hypothesis best explains the data — it is **not** a recipe to combine the top models into a blend.

- Best model: **inner_loop_model** (posterior=0.550, elpd_loo=-1491.38)
- Trials: 2560
- Models compared: 13

## Posterior over models (ELPD-LOO)

| model | posterior | elpd_loo |
| --- | --- | --- |
| inner_loop_model | 0.5504 | -1491.38 |
| iter1_candidate1 | 0.2460 | -1491.83 |
| iter0_candidate0 | 0.1221 | -1492.83 |
| iter0_candidate1 | 0.0808 | -1493.10 |
| linear_accumulated_typicality | 0.0008 | -1497.97 |
| prototype_similarity | 0.0000 | -1535.68 |
| encoding_compressibility | 0.0000 | -1614.88 |
| bayesian_diagnosticity | 0.0000 | -1539.38 |
| window_typicality | 0.0000 | -1654.87 |
| accumulated_alternation_typicality | 0.0000 | -1512.62 |
| iter0_candidate2 | 0.0000 | -1506.17 |
| iter1_candidate0 | 0.0000 | -1512.34 |
| iter1_candidate2 | 0.0000 | -1521.97 |

## Hypotheses

- **inner_loop_model**: People evaluate the randomness of a sequence by accumulating a subjective sense of typicality over its length, rather than merely evaluating its average properties. They evaluate how well a sequence conforms to a mental prototype—represented by an ideal proportion of heads and alternations—and integrate this fit across all events.
- **iter1_candidate1**: People evaluate the randomness of a sequence by accumulating a subjective sense of typicality over its length, but their penalty for deviating from the ideal alternation rate is asymmetric. Because human intuition strongly associates randomness with frequent switching, sequences that under-alternate (cluster) are heavily penalized, while those that over-alternate are penalized much less severely.
- **iter0_candidate0**: People evaluate the randomness of a sequence by accumulating a subjective sense of typicality over its length, but this evidence integration exhibits diminishing returns. Rather than growing linearly, the accumulated typicality scales according to a power-law function of sequence length, which prevents extremely long sequences from disproportionately dominating judgments and explains why the preference for longer sequences saturates.
- **iter0_candidate1**: People evaluate the randomness of a sequence by accumulating a subjective sense of typicality over its events, but working memory for past events is leaky. The evidence contributed by each event decays exponentially over time, meaning the total accumulated randomness score saturates for longer sequences, which prevents large differences in sequence length from having an oversized effect on choice.
- **linear_accumulated_typicality**: People evaluate the randomness of a sequence by accumulating a subjective sense of typicality over its length, but their penalty for deviating from the ideal head proportion and alternation rate is linear (absolute difference) rather than quadratic, treating extreme deviations proportionally to small ones.
- **prototype_similarity**: Random-looking sequences are close to a prototype with balanced H/T counts and an ideal alternation rate.
- **encoding_compressibility**: Random-looking sequences are those with low simple-description penalties: long runs, periodic templates, and imbalance.
- **bayesian_diagnosticity**: Randomness is the log-likelihood ratio of a fair coin versus a regular process: a mixture of a complexity-penalized motif process (Griffiths et al. 2018) and a biased coin. Merges the former diagnosticity and statistical-inference accounts.
- **window_typicality**: Random-looking sequences have a longest run typical of a fair coin seen through a limited memory window; over-long streaks look non-random (Hahn & Warren 2009).
- **accumulated_alternation_typicality**: People evaluate the randomness of a sequence by accumulating a subjective sense of typicality over its length, but this typicality is based entirely on how closely the sequence's alternation rate matches a mental prototype, completely ignoring the overall proportion of heads and tails.
- **iter0_candidate2**: People evaluate the randomness of a sequence by accumulating a subjective sense of typicality over its events, but their choice between two sequences is governed by Weber's law. Instead of comparing the absolute difference in accumulated typicality, they evaluate this difference relative to the total length of both sequences. This relative comparison explains why the preference for a longer sequence saturates, as a given absolute difference in sequence length has a much weaker perceptual effect when the sequences being compared are already long.
- **iter1_candidate0**: People evaluate the randomness of a sequence by accumulating a subjective sense of typicality over its length, but the typicality of each event is strictly positive and bounded. Instead of an unbounded penalty that can drive the per-event score negative, typicality consists of a positive baseline plus an exponentially decaying similarity to a mental prototype. This guarantees that even highly unnatural sequences accumulate positive randomness as they grow, explaining why people still prefer longer sequences when both options are heavily clustered.
- **iter1_candidate2**: People evaluate the randomness of a sequence by accumulating a subjective sense of typicality over its length, but rather than penalizing deviations from an "ideal" alternation rate, they treat alternations as inherently random and thus monotonically rewarding. Because their per-event typicality includes a linear reward for alternation rate rather than a quadratic penalty, sequences with more total alternations naturally accumulate higher randomness scores, explaining both why people prefer highly over-alternating sequences and why they still prefer longer sequences even when both options are heavily clustered.

## Distinguishability (arviz.compare, PSIS-LOO)

`elpd_diff` and `dse` are relative to the best model. A model is only clearly worse than the best when `elpd_diff > 2 * dse`; models within ~2·dse of the top are statistically indistinguishable. `LOO reliable` is False when PSIS-LOO flagged this model's estimate as untrustworthy (many high Pareto-k points) — its row should be read with caution.

| model | elpd_diff | dse | distinguishable from best | weight | LOO reliable |
| --- | --- | --- | --- | --- | --- |
| inner_loop_model ←selected | 0.00 | 0.00 | — (best) | 0.651 | yes |
| iter1_candidate1 | 0.46 | 0.10 | yes | 0.000 | yes |
| iter0_candidate0 | 1.46 | 2.08 | no (within ~2·dse) | 0.000 | yes |
| iter0_candidate1 | 1.72 | 1.66 | no (within ~2·dse) | 0.000 | yes |
| linear_accumulated_typicality | 6.59 | 5.73 | no (within ~2·dse) | 0.245 | yes |
| iter0_candidate2 | 14.79 | 3.20 | yes | 0.000 | yes |
| iter1_candidate0 | 20.96 | 7.43 | yes | 0.000 | yes |
| accumulated_alternation_typicality | 21.24 | 6.77 | yes | 0.000 | no ⚠ |
| iter1_candidate2 | 30.59 | 8.17 | yes | 0.000 | yes |
| prototype_similarity | 44.30 | 10.96 | yes | 0.000 | yes |
| bayesian_diagnosticity | 48.00 | 9.42 | yes | 0.000 | yes |
| encoding_compressibility | 123.50 | 17.96 | yes | 0.104 | yes |
| window_typicality | 163.50 | 18.92 | yes | 0.000 | yes |
