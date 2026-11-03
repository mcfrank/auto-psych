# Inner Model Loop Report

Each model below is ONE distinct cognitive hypothesis. The posterior mass shows which single hypothesis best explains the data — it is **not** a recipe to combine the top models into a blend.

- Best model: **parity_restricted_range_conjunctions** (posterior=0.771, elpd_loo=-796.71)
- Trials: 2250
- Models compared: 8

## Posterior over models (ELPD-LOO)

| model | posterior | elpd_loo |
| --- | --- | --- |
| parity_restricted_range_conjunctions | 0.7711 | -796.71 |
| parity_restricted_compressed_intervals | 0.2107 | -798.01 |
| log_compressed_interval_sampling | 0.0079 | -801.30 |
| weber_scaled_profile_neighborhoods | 0.0040 | -801.98 |
| graded_profile_with_neighborhoods | 0.0027 | -802.37 |
| round_boundary_interval_prior | 0.0026 | -802.39 |
| log_compressed_number_line_intervals | 0.0006 | -803.88 |
| erlang_width_compressed_intervals | 0.0004 | -804.25 |

## Hypotheses

- **parity_restricted_range_conjunctions**: People can build compound concepts that join a numerical range with the examples' parity ("even numbers between 20 and 60"), but considering such a two-part concept carries a cognitive cost, so these conjunctions get only a fitted, reduced share of prior belief alongside the simple rules and plain ranges of exception-tolerant Bayesian concept learning on a compressed number line. Once considered, the size principle strongly favours these small compound concepts, so people fill in the range around the examples but reject immediate neighbours of the wrong parity (and odd-one-out neighbours of an outlier example) far more than a learner who only has plain ranges.
- **parity_restricted_compressed_intervals**: Refinement of `log_compressed_interval_sampling`: people do the same exception-tolerant, tempered Bayesian concept learning over the same 45 rules and all ranges of 1-100, with range members pictured as sampled along a compressed mental number line, but their vocabulary of magnitude concepts also contains parity-restricted ranges ("even numbers between 20 and 40", "odd numbers between 51 and 67"), which share a fitted portion of the prior given to ranges. The one change is these conjunctive parity-and-magnitude concepts: when the examples share a parity, people generalise to nearby numbers of the same parity but reject immediate neighbours (which flip parity) and near numbers around an off-pattern example more than a learner with only plain ranges would.
- **log_compressed_interval_sampling**: Refinement of `exception_tolerant_tempered_mixture`: people do the same exception-tolerant, tempered Bayesian concept learning over the same 45 rules and all intervals of 1-100, but they picture the examples of an interval concept as sampled along a compressed (roughly logarithmic) mental number line, so within a range larger numbers are individually less likely to be drawn and the size principle charges less for extending a range upward than downward. The one change is this compressed-number-line sampling of interval members (degree of compression fitted; no compression gives back essentially the original model), which predicts that people generalise past the largest example more than below the smallest, closing the critique that humans extend upward more than the symmetric models predict.
- **weber_scaled_profile_neighborhoods**: Refinement of `graded_profile_with_neighborhoods`: people still judge membership by matching the target against a frequency-weighted, distinctiveness-weighted profile of the examples' arithmetic and surface properties plus each example's own soft numerical neighbourhood, but that neighbourhood lives on a logarithmically compressed (Weber-like) mental number line rather than the linear 1-100 scale. The one change is this compressed neighbourhood (grafted from the compressed-number-line models): "close to 8" covers only a unit or two while "close to 80" covers many, the neighbourhood reaches further above an example than below it, and a broader neighbourhood counts as a less distinctive property — so people extend upward beyond the examples more than downward, as the critique of the incumbent reports.
- **graded_profile_with_neighborhoods**: Refinement of `graded_feature_profile_matching`: people still judge membership by matching the target against a frequency-weighted profile of the examples' properties (parity, multiples, powers, squares, primes, final digit, contained digits, decade), rarer properties counting more, but the profile also contains each example's own numerical neighbourhood ("numbers close to 27") as one more graded property, with a fitted width, weighted by the same distinctiveness rule and shared in proportion to how many examples fall near it. The one change is this soft neighbourhood property: targets just one to three away from an example gain strong support that decays steeply within a few units, closing the critique that human generalisation falls off over short numerical distances far more sharply than the current models predict.
- **round_boundary_interval_prior**: People infer numerical-range concepts anchored on round numbers: when they consider that the program produces "a range of numbers", they think of ranges whose edges fall on decade boundaries ("from 20 to 50", "the 80s", "up to 100") as far more plausible than ranges with arbitrary edges such as 27-92 (the strength of this round-boundary preference fitted), while otherwise doing the same exception-tolerant, tempered Bayesian learning over arithmetic rules and ranges sampled along a compressed number line. Because the preferred range is snapped outward to the nearest decade edges, people fill in the span between the examples, generalise within the examples' decades but not across a decade boundary just beside an example, and extend above an example ending in a low digit more than below it.
- **log_compressed_number_line_intervals**: People represent numerical-magnitude concepts ("numbers between 40 and 60") as regions on a logarithmically compressed mental number line, not on the linear 1-100 scale: when they infer which stretch of numbers the program produces, a stretch's size, and how likely the examples are to fall where they do within it, are judged in compressed (Weber-like) units, while arithmetic and digit rules are judged as before. Because the same numerical step is subjectively smaller among large numbers, people extend a magnitude region upward beyond the largest example more readily than downward below the smallest, and generalise across the span between the examples.
- **erlang_width_compressed_intervals**: Refinement of `log_compressed_interval_sampling`: people do the same exception-tolerant, tempered Bayesian concept learning over the same 45 rules and all intervals of 1-100, with interval members pictured as sampled along a compressed number line, but they do not consider every numerical range equally plausible beforehand — they expect a program that produces "a range of numbers" to produce a range of moderate width, so tiny ranges hugging one or two examples are a priori unlikely (an Erlang-shaped prior on range width, its typical width fitted). The one change is this width prior on intervals, replacing the flat prior: people fill in the whole span between the examples more, do not build a tight range around a lone exceptional example, and lean on parity or digit rules rather than a tiny range for a target just one step from an example.

## Distinguishability (arviz.compare, PSIS-LOO)

`elpd_diff` and `dse` are relative to the best model. A model is only clearly worse than the best when `elpd_diff > 2 * dse`; models within ~2·dse of the top are statistically indistinguishable. `LOO reliable` is False when PSIS-LOO flagged this model's estimate as untrustworthy (many high Pareto-k points) — its row should be read with caution.

| model | elpd_diff | dse | distinguishable from best | weight | LOO reliable |
| --- | --- | --- | --- | --- | --- |
| parity_restricted_range_conjunctions ←selected | 0.00 | 0.00 | — (best) | 0.771 | yes |
| parity_restricted_compressed_intervals | 1.30 | 1.43 | no (within ~2·dse) | 0.211 | yes |
| log_compressed_interval_sampling | 4.59 | 4.14 | no (within ~2·dse) | 0.008 | yes |
| weber_scaled_profile_neighborhoods | 5.26 | 10.30 | no (within ~2·dse) | 0.004 | yes |
| graded_profile_with_neighborhoods | 5.65 | 10.01 | no (within ~2·dse) | 0.003 | yes |
| round_boundary_interval_prior | 5.68 | 4.24 | no (within ~2·dse) | 0.003 | yes |
| log_compressed_number_line_intervals | 7.17 | 4.28 | no (within ~2·dse) | 0.001 | yes |
| erlang_width_compressed_intervals | 7.54 | 4.09 | no (within ~2·dse) | 0.000 | yes |
