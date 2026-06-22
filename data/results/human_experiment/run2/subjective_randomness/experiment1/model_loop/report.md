# Inner Model Loop Report

Each model below is ONE distinct cognitive hypothesis. The posterior mass shows which single hypothesis best explains the data — it is **not** a recipe to combine the top models into a blend.

- Best model: **iter1_candidate0** (posterior=0.999, elpd_loo=-788.83)
- Trials: 1280
- Models compared: 9

## Posterior over models (ELPD-LOO)

| model | posterior | elpd_loo |
| --- | --- | --- |
| iter1_candidate0 | 0.9992 | -788.83 |
| iter0_candidate0 | 0.0008 | -796.31 |
| prototype_similarity | 0.0000 | -809.56 |
| encoding_compressibility | 0.0000 | -842.50 |
| bayesian_diagnosticity | 0.0000 | -813.69 |
| window_typicality | 0.0000 | -838.52 |
| iter0_candidate1 | 0.0000 | -816.16 |
| iter0_candidate2 | 0.0000 | -819.85 |
| iter1_candidate2 | 0.0000 | -808.77 |

## Hypotheses

- **iter1_candidate0**: Random-looking sequences are judged by their Euclidean distance to a "messy" prototype: rather than expecting perfectly balanced counts, people apply a "too-perfect" heuristic and judge sequences against a prototype with a strictly positive ideal imbalance. The cognitive penalty accelerates quadratically with deviations from this ideal imbalance and an ideal alternation rate, creating a wide tolerance for near-ideal sequences while remaining asymmetric to penalize under-alternating sequences more harshly.
- **iter0_candidate0**: Random-looking sequences are judged by their similarity to a prototype with balanced counts and an ideal alternation rate, but the penalty for deviating from the ideal alternation rate is asymmetric: sequences that alternate less than the ideal are penalized more harshly than those that alternate more.
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
- **iter0_candidate1**: People judge a sequence's randomness by evaluating the joint typicality of its macroscopic features — specifically, the binomial probability of its head count given a fair coin, and the binomial probability of its alternation count given an ideal alternation rate. Rather than using length-invariant proportional heuristics, this mechanism naturally scales deviation penalties with sequence length and provides a statistically grounded, asymmetric tolerance for over-alternating (highly periodic) sequences.
- **iter0_candidate2**: People judge randomness by multi-scale representativeness: they expect the proportion of heads to closely match the 50/50 fair-coin ideal not just globally, but across all possible contiguous sub-sequences. Rather than using separate heuristics for global imbalance and alternations, people penalize a sequence based on the mean squared deviation of the head proportion from 0.5 across all sub-windows of length 2 or greater, which naturally favors evenly spaced, periodic sequences because those exhibit the least local variance.
- **iter1_candidate2**: People judge a sequence's randomness by comparing its local pattern frequencies to a subjective expectation: they compute the Kullback-Leibler divergence between the sequence's empirical bigram distribution (HH, HT, TH, TT) and an idealized prototype distribution that heavily favors alternating pairs, perceiving sequences with lower divergence from this prototype as more random.

## Distinguishability (arviz.compare, PSIS-LOO)

`elpd_diff` and `dse` are relative to the best model. A model is only clearly worse than the best when `elpd_diff > 2 * dse`; models within ~2·dse of the top are statistically indistinguishable. `LOO reliable` is False when PSIS-LOO flagged this model's estimate as untrustworthy (many high Pareto-k points) — its row should be read with caution.

| model | elpd_diff | dse | distinguishable from best | weight | LOO reliable |
| --- | --- | --- | --- | --- | --- |
| iter1_candidate0 ←selected | 0.00 | 0.00 | — (best) | 0.725 | yes |
| iter0_candidate0 | 7.49 | 4.01 | no (within ~2·dse) | 0.000 | yes |
| iter1_candidate2 | 19.94 | 6.96 | yes | 0.000 | yes |
| prototype_similarity | 20.73 | 5.59 | yes | 0.000 | yes |
| bayesian_diagnosticity | 24.86 | 8.73 | yes | 0.154 | yes |
| iter0_candidate1 | 27.34 | 6.77 | yes | 0.000 | yes |
| iter0_candidate2 | 31.02 | 9.40 | yes | 0.121 | yes |
| window_typicality | 49.69 | 11.31 | yes | 0.000 | yes |
| encoding_compressibility | 53.67 | 11.42 | yes | 0.000 | yes |
