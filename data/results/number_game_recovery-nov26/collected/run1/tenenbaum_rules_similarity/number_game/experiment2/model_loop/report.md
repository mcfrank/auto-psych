# Inner Model Loop Report

Each model below is ONE distinct cognitive hypothesis. The posterior mass shows which single hypothesis best explains the data — it is **not** a recipe to combine the top models into a blend.

- Best model: **first_example_evidence_weight** (posterior=0.846, elpd_loo=-2254.05)
- Trials: 4500
- Models compared: 8

## Posterior over models (ELPD-LOO)

| model | posterior | elpd_loo |
| --- | --- | --- |
| first_example_evidence_weight | 0.8462 | -2254.05 |
| round_number_range_boundaries | 0.0588 | -2256.71 |
| without_replacement_size_principle | 0.0536 | -2256.81 |
| round_number_ranges_primes_family | 0.0139 | -2258.15 |
| without_replacement_primes_family | 0.0117 | -2258.33 |
| centre_weighted_ranges_with_thresholds | 0.0084 | -2258.66 |
| centre_weighted_range_sampling | 0.0049 | -2259.19 |
| threshold_concepts_primes_family | 0.0023 | -2259.94 |

## Hypotheses

- **first_example_evidence_weight**: People infer the hidden program as an ideal Bayesian concept learner would (number rules, bounded ranges with a round-number bias on their edges, and one-sided thresholds, weighed by family plausibility, range width and the size principle, with examples drawn without repetition), but they revise their belief example by example in the order shown and the first example carries its own evidential weight: met before they have any guess about the program, the opening example counts as a looser (or, if the data say so, stronger) piece of evidence than each later example, which counts normally. So the "suspicious coincidence" that the first example points to a tight concept is discounted relative to later examples, which matters most for single-example sets and for which example of a set is shown first.
- **round_number_range_boundaries**: People infer the hidden program as an ideal Bayesian concept learner would (number rules, bounded ranges and one-sided thresholds, weighed by family plausibility, range width and the size principle, with examples drawn without repetition), but their sense of how plausible a bounded range is is biased by round numbers: multiples of ten act as cognitive reference points, so a range whose lower or upper boundary falls exactly on one (e.g. 20-37 or 40-60) is judged more, or less, likely a priori than an otherwise identical range, with the direction and strength of this round-number bias inferred from the data. This shifts where people stop saying "yes" near multiples of ten — for example whether 40 is included after examples in the low thirties — while leaving rule-based generalisation unchanged.
- **without_replacement_size_principle**: Refinement of `threshold_concepts_erlang_interval`: people do the same Bayesian concept learning over number rules, bounded ranges (with an Erlang prior on width) and one-sided thresholds, with a prior mass per concept family, but they believe the program never repeats itself — each example is a member of the concept not already shown, drawn without replacement. The one change is this without-replacement size principle, which makes the "suspicious coincidence" grow faster than an ideal with-replacement learner's whenever the examples use up a large share of a small concept: sets that nearly exhaust a tiny rule or a narrow range (e.g. 4 of the 7 powers of 2, or 11, 15 filling a five-number band) favour that tight concept even more strongly, so people reject targets just outside it more often than the best model predicts, while sets that are few relative to every candidate concept behave as before.
- **round_number_ranges_primes_family**: Refinement of `round_number_range_boundaries`, adding one component from `without_replacement_primes_family`: people do the same Bayesian concept learning over number rules, bounded ranges (Erlang width prior, with a learned round-number bias on ranges whose boundaries fall on multiples of ten) and one-sided thresholds, with a prior mass per concept family and examples drawn without replacement. The one change is that "prime numbers" is no longer lumped with squares, cubes and powers in a shared special-sets family but forms its own family with a separately inferred prior mass, so prime-like sets (e.g. 3, 7 or 53, 2) are read as "primes" — people endorse other primes and reject nearby composites — independently of how plausible squares or powers are, while the round-number bias still shapes where magnitude-based "yes" answers stop.
- **without_replacement_primes_family**: Refinement of `without_replacement_size_principle`, adding one component from `threshold_concepts_primes_family`: people do the same Bayesian concept learning over number rules, bounded ranges (with an Erlang prior on width) and one-sided thresholds, with a prior mass per concept family, and they believe the program never repeats itself, so examples are drawn from the concept without replacement. The one change is that "prime numbers" is no longer lumped with squares, cubes and powers in a shared special-sets family but forms its own family with a separately inferred prior mass, so prime-like sets (e.g. 3, 7 or 53, 2) are read as "primes" — people endorse other primes and reject nearby composites — independently of how plausible squares or powers are, while the without-replacement coincidence still sharpens generalisation for sets that use up much of a small concept.
- **centre_weighted_ranges_with_thresholds**: Refinement of `threshold_concepts_erlang_interval`, adding one component from `centre_weighted_range_sampling`. People do the same Bayesian concept learning over number rules, bounded ranges (with an Erlang prior on width) and one-sided thresholds (their own family), with the size principle and a prior mass per concept family. The one change is that people expect a bounded-range program ("numbers between 30 and 50") to produce numbers clustered toward the middle of its range, while a threshold program ("numbers above 60") still produces its members evenly. How strongly ranges cluster is inferred from the data, and zero recovers the incumbent. An example at a range's edge is then a suspicious coincidence, so ranges centred on the examples that reach past them gain belief relative to thresholds and rules.
- **centre_weighted_range_sampling**: Refinement of `erlang_interval_size_family_prior`: people do the same Bayesian concept learning over number rules and numeric ranges, with the size principle, a prior mass per concept family (exception rules in their own family) and an Erlang prior over range width, but they do not expect a range-based program to produce every number in its range equally often — they expect its outputs to cluster toward the middle of the range, with how strongly they do so inferred from the data (none recovers the parent model). The one change is this centre-weighted sampling from ranges: examples sitting at a range's edge are a "suspicious coincidence" for it, so ranges that are centred on the examples and extend beyond the lowest and highest example gain support, and people say "yes" to targets somewhat past the examples' extremes more often than a uniform-sampling learner predicts.
- **threshold_concepts_primes_family**: Refinement of `threshold_concepts_erlang_interval`: people do the same Bayesian concept learning over number rules, bounded numeric ranges (with an Erlang prior on width) and one-sided magnitude thresholds (their own family), with the size principle and a prior mass per concept family, but they do not treat "prime numbers" as just one more member of a shared "special sets" family with squares, cubes and powers. The one change is that primes form their own concept family with a separately inferred prior mass, so examples such as 3, 7 or 53, 2 can be read as "primes" strongly enough that people endorse other primes (13, 29, 5, 11) and reject nearby composites, independently of how readily they read 81, 25 as "squares" or 8, 64 as "powers".

## Distinguishability (arviz.compare, PSIS-LOO)

`elpd_diff` and `dse` are relative to the best model. A model is only clearly worse than the best when `elpd_diff > 2 * dse`; models within ~2·dse of the top are statistically indistinguishable. `LOO reliable` is False when PSIS-LOO flagged this model's estimate as untrustworthy (many high Pareto-k points) — its row should be read with caution.

| model | elpd_diff | dse | distinguishable from best | weight | LOO reliable |
| --- | --- | --- | --- | --- | --- |
| first_example_evidence_weight ←selected | 0.00 | 0.00 | — (best) | 0.846 | yes |
| round_number_range_boundaries | 2.67 | 1.64 | no (within ~2·dse) | 0.059 | yes |
| without_replacement_size_principle | 2.76 | 1.60 | no (within ~2·dse) | 0.054 | yes |
| round_number_ranges_primes_family | 4.11 | 2.61 | no (within ~2·dse) | 0.014 | yes |
| without_replacement_primes_family | 4.28 | 2.58 | no (within ~2·dse) | 0.012 | yes |
| centre_weighted_ranges_with_thresholds | 4.61 | 2.68 | no (within ~2·dse) | 0.008 | yes |
| centre_weighted_range_sampling | 5.14 | 2.94 | no (within ~2·dse) | 0.005 | yes |
| threshold_concepts_primes_family | 5.89 | 3.01 | no (within ~2·dse) | 0.002 | yes |
