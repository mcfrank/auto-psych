# Inner Model Loop Report

Each model below is ONE distinct cognitive hypothesis. The posterior mass shows which single hypothesis best explains the data — it is **not** a recipe to combine the top models into a blend.

- Best model: **sticky_guess_power_law_ranges** (posterior=0.444, elpd_loo=-866.07)
- Trials: 2250
- Models compared: 8

## Posterior over models (ELPD-LOO)

| model | posterior | elpd_loo |
| --- | --- | --- |
| sticky_guess_power_law_ranges | 0.4442 | -866.07 |
| first_guess_win_stay_concepts | 0.2390 | -866.69 |
| digit_rule_discount_sticky_guess | 0.1438 | -867.19 |
| fuzzy_trace_example_memory | 0.0775 | -867.81 |
| power_law_endpoints_erlang_width | 0.0371 | -868.55 |
| sticky_guess_last_digit_rule_prior | 0.0273 | -868.85 |
| round_endpoint_interval_concepts | 0.0173 | -869.31 |
| magnitude_compressed_sampling_concepts | 0.0138 | -869.54 |

## Hypotheses

- **sticky_guess_power_law_ranges**: Refinement of first_guess_win_stay_concepts: people hold a single working guess about the concept (a rule or a numerical range), formed from the first example by Bayesian concept learning and kept, with a fitted stickiness, as later examples arrive as long as each new example fits it, reconsidering from all examples so far only when it is contradicted (or they do not stick). The one change, the component taken from power_law_interval_endpoints: when they consider a numerical range they do not treat all locations on the number line alike — the range's endpoints have a prior plausibility that follows a fitted power law over 1-100 (on top of the incumbent's Erlang preference for a typical width), so whether a range guess is formed and kept depends on whether the examples are small or large numbers.
- **first_guess_win_stay_concepts**: People do not weigh all examples together at once; they hold a single working guess about the concept (a rule or a numerical range) that they form from the first example by Bayesian concept learning, and as each further example arrives they keep that guess, with a fitted stickiness, as long as the new example fits it, reconsidering the concept from all examples seen so far only when the new example contradicts the guess (or when they do not stick). Because the guess is anchored on what fitted the early examples, a broad concept chosen after the first example survives later examples that would have favoured a narrower one, so the same examples in a different order produce different generalisations (a primacy effect).
- **digit_rule_discount_sticky_guess**: Refinement of sticky_guess_power_law_ranges: people hold a single working guess about the concept (a rule or a numerical range), formed from the first example by Bayesian concept learning and kept with a fitted stickiness while later examples fit it, reconsidering from all examples only when contradicted, with range guesses shaped by a power-law endpoint and Erlang width prior. The one change: rules defined only by how a number is written ("ends in 7", "contains the digit 3") are a priori less (or more) plausible than arithmetic rules (odd, multiples, powers, squares, primes), by a single fitted weight on the digit-rule family — so people latch onto and extend "same last digit" guesses to far-away numbers less readily than the incumbent predicts.
- **fuzzy_trace_example_memory**: People do Bayesian concept learning over mathematical rules and numerical ranges (with the size principle), but their memory of the examples is imprecise: each example is kept verbatim only with a fitted probability, and otherwise only its rough magnitude (within about five of its true value) is retained. Concepts are evaluated against these fuzzy memory traces, so concepts that depend on an example's exact identity (last digit, multiples, powers) lose support with every imperfectly remembered example while numerical ranges, which only need the magnitude, do not — making people generalise less along digit-based rules to far-away targets, especially from longer sets.
- **power_law_endpoints_erlang_width**: Refinement of power_law_interval_endpoints: people do the same Bayesian concept learning over the 45 rules and all intervals of 1-100 (size principle, 10% noisy examples, a prior weight per rule) with interval endpoints whose prior plausibility follows a fitted power law over the number line, but the one change is that they also expect a numerical range to have a characteristic width: the interval prior is additionally weighted by an Erlang density on the interval's size with a fitted typical size (the component taken from erlang_interval_prior_concepts). So where a range lies on the number line and how broad it is both shape the prior, which jointly sets how far people generalise beyond the examples' span for small- versus large-number sets.
- **sticky_guess_last_digit_rule_prior**: Refinement of sticky_guess_power_law_ranges: people hold a single working guess about the concept (a rule or a numerical range, with power-law and Erlang priors on ranges), formed from the first example by Bayesian concept learning and kept with a fitted stickiness while later examples fit it, reconsidering only when contradicted. The one change: rules are not all equally natural guesses — the ten "ends in digit d" rules share a single fitted prior weight relative to the other rules, so people latch onto and keep last-digit rules less (or more) readily than arithmetic rules like multiples or powers, which governs how much they endorse far-away numbers merely because they share an example's final digit.
- **round_endpoint_interval_concepts**: Refinement of erlang_interval_prior_concepts: people do the same Bayesian concept learning over the 45 rules and all intervals of 1-100 (size principle, 10% noisy examples, a free prior weight per rule, an Erlang prior on interval size), but when they imagine the program picking a numerical range they favour ranges with round-number boundaries (starting at 1 or a multiple of ten, ending at a multiple of ten, e.g. "20 to 40" or "1 to 50") over arbitrary ones like "17 to 43". The one change is a fitted prior boost for each round endpoint, so generalisation from clustered examples extends to, and then drops off sharply at, the nearest decade boundaries rather than fading smoothly with width alone.
- **magnitude_compressed_sampling_concepts**: Refinement of power_law_endpoints_erlang_width (same 45 rules and all intervals of 1-100, size principle, 10% noisy examples, a prior weight per rule, power-law endpoint prior and Erlang width prior). The one change is in how people think the examples were sampled: they imagine the program drawing numbers uniformly on a compressed, logarithmic-like mental number line rather than uniformly in raw value, with the degree of compression fitted, so within a range or rule smaller numbers are expected to be drawn more often than larger ones. Because a span among large numbers is then psychologically narrower, the size principle penalises extending a range less for large-number examples than for small-number ones, so generalisation beyond the examples widens in proportion to their magnitude (Weber-like scaling).

## Distinguishability (arviz.compare, PSIS-LOO)

`elpd_diff` and `dse` are relative to the best model. A model is only clearly worse than the best when `elpd_diff > 2 * dse`; models within ~2·dse of the top are statistically indistinguishable. `LOO reliable` is False when PSIS-LOO flagged this model's estimate as untrustworthy (many high Pareto-k points) — its row should be read with caution.

| model | elpd_diff | dse | distinguishable from best | weight | LOO reliable |
| --- | --- | --- | --- | --- | --- |
| sticky_guess_power_law_ranges ←selected | 0.00 | 0.00 | — (best) | 0.444 | yes |
| first_guess_win_stay_concepts | 0.62 | 1.85 | no (within ~2·dse) | 0.239 | yes |
| digit_rule_discount_sticky_guess | 1.13 | 1.19 | no (within ~2·dse) | 0.144 | yes |
| fuzzy_trace_example_memory | 1.75 | 2.14 | no (within ~2·dse) | 0.078 | yes |
| power_law_endpoints_erlang_width | 2.48 | 2.91 | no (within ~2·dse) | 0.037 | yes |
| sticky_guess_last_digit_rule_prior | 2.79 | 2.10 | no (within ~2·dse) | 0.027 | yes |
| round_endpoint_interval_concepts | 3.25 | 3.27 | no (within ~2·dse) | 0.017 | yes |
| magnitude_compressed_sampling_concepts | 3.47 | 2.91 | no (within ~2·dse) | 0.014 | yes |
