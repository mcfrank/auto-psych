# Inner Model Loop Report

Each model below is ONE distinct cognitive hypothesis. The posterior mass shows which single hypothesis best explains the data — it is **not** a recipe to combine the top models into a blend.

- Best model: **iter1_candidate0** (posterior=0.593, elpd_loo=-302.80)
- Trials: 600
- Models compared: 8

## Posterior over models (ELPD-LOO)

| model | posterior | elpd_loo |
| --- | --- | --- |
| iter1_candidate0 | 0.5932 | -302.80 |
| bayesian_diagnosticity | 0.3023 | -303.77 |
| encoding_compressibility | 0.0891 | -306.29 |
| iter0_candidate2 | 0.0154 | -308.84 |
| iter0_candidate1 | 0.0000 | -317.79 |
| iter0_candidate0 | 0.0000 | -416.85 |
| iter1_candidate1 | 0.0000 | -416.83 |
| iter1_candidate2 | 0.0000 | -341.96 |

## Hypotheses

- **iter1_candidate0**: People judge a sequence as more random when it is more diagnostic of a fair-coin generator relative to salient non-random alternatives (alternating, biased, or streaky generators). This Bayesian diagnostic comparison is the same mechanism as the leading model, but the switch probability that defines the "alternating" prototype is inferred from participants' choices rather than fixed — the model learns how strongly alternating the non-random alternative must be to explain the data, rather than assuming a perfectly regular alternating generator.
- **bayesian_diagnosticity**: Random-looking sequences are diagnostic of a fair coin over salient
non-random alternatives: alternating, biased, and streaky generators.
- **encoding_compressibility**: Random-looking sequences are those with low simple-description penalties:
long runs, periodic templates, and imbalance.
- **iter0_candidate2**: People judge a sequence as more random when its proportion of heads is closer to 50%. When comparing two sequences, they choose whichever has better head/tail balance — the smaller absolute deviation from equal proportions — as the more random one. A single inverse-temperature parameter governs how sensitively this balance difference drives the choice.
- **iter0_candidate1**: People judge a sequence as less random the longer its longest unbroken streak of identical outcomes. The maximum run length is the single most salient cue of non-randomness: when comparing two sequences, people choose the one whose longest run is shorter as the more random one. Sensitivity to this cue varies across individuals but is captured by a single inverse-temperature parameter.
- **iter0_candidate0**: People judge a sequence as more random when its alternation rate is closest to the prototype value of 0.5 — the expected alternation rate for a fair coin. Randomness perception follows a Gaussian similarity function: the closer a sequence's p_alts is to 0.5, the more random it looks, with similarity falling off symmetrically as a function of squared deviation from the prototype. On each trial, people choose whichever of the two sequences is nearer to this internal prototype.
- **iter1_candidate1**: People judge a sequence as non-random when it exhibits rhythmic, periodic structure — a repeating temporal pattern that stands out as too regular to be the product of a fair coin. When choosing which of two sequences looks more random, people pick the one with lower periodicity, regardless of its overall balance or run lengths. A single sensitivity parameter captures how strongly this detected periodicity difference drives the choice.
- **iter1_candidate2**: People judge a sequence as more random when its head count is more probable under a fair coin: they implicitly evaluate the binomial likelihood B(h; n, 0.5), preferring whichever sequence has a head count closer to the most probable outcome for its length. This differs from simple linear imbalance — it encodes the full combinatorial structure of how likely a given count is, not just its distance from 50%.

## Distinguishability (arviz.compare, PSIS-LOO)

`elpd_diff` and `dse` are relative to the best model. A model is only clearly worse than the best when `elpd_diff > 2 * dse`; models within ~2·dse of the top are statistically indistinguishable.

| model | elpd_diff | dse | distinguishable from best | weight |
| --- | --- | --- | --- | --- |
| iter1_candidate0 | 0.00 | 0.00 | — (best) | 0.905 |
| bayesian_diagnosticity | 0.97 | 0.48 | yes | 0.000 |
| encoding_compressibility | 3.50 | 2.23 | no (within ~2·dse) | 0.000 |
| iter0_candidate2 | 6.05 | 3.79 | no (within ~2·dse) | 0.095 |
| iter0_candidate1 | 15.00 | 5.25 | yes | 0.000 |
| iter1_candidate2 | 39.17 | 6.96 | yes | 0.000 |
| iter1_candidate1 | 114.04 | 12.52 | yes | 0.000 |
| iter0_candidate0 | 114.05 | 12.52 | yes | 0.000 |
