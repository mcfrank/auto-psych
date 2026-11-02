# Inner Model Loop Report

Each model below is ONE distinct cognitive hypothesis. The posterior mass shows which single hypothesis best explains the data — it is **not** a recipe to combine the top models into a blend.

- Best model: **powers_own_family_erlang_interval** (posterior=0.229, elpd_loo=-928.01)
- Trials: 2250
- Models compared: 8

## Posterior over models (ELPD-LOO)

| model | posterior | elpd_loo |
| --- | --- | --- |
| powers_own_family_erlang_interval | 0.2290 | -928.01 |
| power_base_family_prior | 0.2187 | -928.06 |
| exception_rule_family_prior_2 | 0.1332 | -928.56 |
| erlang_interval_size_family_prior | 0.1250 | -928.62 |
| parity_bounded_range_concepts | 0.0958 | -928.89 |
| erlang_interval_size_family_prior_2 | 0.0931 | -928.91 |
| exception_rule_family_prior | 0.0687 | -929.22 |
| erlang_interval_size_exception_prior | 0.0363 | -929.86 |

## Hypotheses

- **powers_own_family_erlang_interval**: Refinement of `erlang_interval_size_family_prior`: people do the same Bayesian concept learning over rules and numeric intervals, with the size principle, a prior mass per concept family (exception rules in their own family) and an Erlang prior over interval width, but they do not lump "powers of m" (2, 4, 8, 16, …; 3, 9, 27, 81; 8, 64; …) together with primes, squares and cubes. The one change is that repeated-multiplication rules form their own concept family with a separately inferred prior mass, so if people find powers an especially natural kind of program, examples like 8 and 64 are read as "powers of 8" (rejecting 2, 4, 16 and 32), and sets of powers of 2 crowd out "even" more strongly than the current model predicts.
- **power_base_family_prior**: Refinement of `exception_rule_family_prior_2`: people do the same Bayesian concept learning over rules and intervals with the size principle, a separate prior mass per concept family and the exception rules in their own family, but they do not lump "powers of a base" (powers of 2, 3, ... 10) together with primes, squares and cubes as one "special sets" family. The one change is to give powers-of-a-base rules their own family with its own inferred prior mass: people readily think of a set as successive powers (or repeated multiplication) of a number, while primes, squares and cubes each have a separate, independently inferred plausibility, so generalisation after power-like sets (e.g. 8, 64) and after square- or prime-like sets (e.g. 81, 9 or 3, 31) can differ in strength instead of being tied to a shared per-rule prior.
- **exception_rule_family_prior_2**: Refinement of `family_weighted_concept_prior`: people do the same Bayesian concept learning over rules and intervals with the size principle and a separate prior mass per concept family, but the two gerrymandered exception rules ("powers of 2 except 32" and "powers of 2 plus 37") are not natural mathematical sets — people give them their own, separately inferred (and expectedly tiny) prior mass rather than an equal share of the "special sets" family alongside primes, squares and powers. The one change is this separate family for exception rules, so for sets of powers of 2 people should reject the lure 37 nearly always while still endorsing the other powers of 2.
- **erlang_interval_size_family_prior**: Refinement of `exception_rule_family_prior_2`: people do the same Bayesian concept learning over rules and intervals, with the size principle and a separate prior mass per concept family (exception rules in their own family), but they do not regard all numeric intervals as equally plausible a priori. The one change is an Erlang-shaped prior over interval width with a learned characteristic size (as in Tenenbaum's number game): very narrow and very wide ranges are a priori unlikely, so after a single example or tightly clustered examples people generalise to a band of moderate width around them rather than collapsing onto the tightest interval, and after widely spread examples broad ranges are penalised relative to natural rules.
- **parity_bounded_range_concepts**: Refinement of `exception_rule_family_prior_2`: people do the same Bayesian concept learning over rules and numeric intervals, with the size principle and a prior mass per concept family (the exception rules in their own family), but their hypothesis space also contains conjunctive "parity within a range" concepts — "even numbers between 20 and 60", "odd numbers from 9 to 69" — as a new family with its own inferred prior mass. The one change is this composed rule-plus-magnitude family: when examples are all even (or all odd) and clustered in magnitude, people generalise to same-parity numbers near the examples while rejecting both nearby numbers of the other parity and same-parity numbers far outside the examples' range, which neither a pure parity rule nor a pure interval predicts.
- **erlang_interval_size_family_prior_2**: Refinement of `exception_rule_family_prior_2`: people do the same Bayesian concept learning over rules (with a prior mass per concept family, the two exception rules in their own family) and numeric intervals with the size principle, but they do not treat every interval of 1-100 as equally likely a priori — they expect a magnitude-based concept to span a characteristic, moderate range, so their prior over intervals follows an Erlang distribution over interval width with an inferred typical size (Tenenbaum's interval prior). The one change is this size-dependent interval prior: it penalises both the hairline intervals that hug the examples and the very broad ones, which changes how far people generalise by magnitude beyond the examples (e.g. to targets 10-20 away from a single example) relative to the incumbent's flat interval prior.
- **exception_rule_family_prior**: Refinement of `size_principle_strength_family_prior`: people do the same Bayesian concept learning over rules and intervals, with a prior weight per concept family and a learned strength of the size principle, but they do not treat ad-hoc "exception" rules (powers of 2 except 32, powers of 2 plus 37) as being as natural as clean mathematical sets. The one change is to give these exception rules their own concept family with its own inferred prior mass, so people can find "powers of 2, or 37" implausible and reject the 37 lure for sets of powers of 2 while still generalising to the other powers of 2.
- **erlang_interval_size_exception_prior**: Refinement of `exception_rule_family_prior`: people do the same Bayesian concept learning over rules and intervals, with a prior mass per concept family (exception rules as their own family) and a learned strength of the size principle, but they do not consider every numeric interval equally likely a priori. The one change is an Erlang prior on interval size with a learned typical size, as in Tenenbaum's number game: people expect a magnitude-based program to cover a range of moderate width, so very narrow intervals hugging the examples and very broad ones are both discounted, which reshapes how far "yes" answers extend beyond the examples in magnitude.

## Distinguishability (arviz.compare, PSIS-LOO)

`elpd_diff` and `dse` are relative to the best model. A model is only clearly worse than the best when `elpd_diff > 2 * dse`; models within ~2·dse of the top are statistically indistinguishable. `LOO reliable` is False when PSIS-LOO flagged this model's estimate as untrustworthy (many high Pareto-k points) — its row should be read with caution.

| model | elpd_diff | dse | distinguishable from best | weight | LOO reliable |
| --- | --- | --- | --- | --- | --- |
| powers_own_family_erlang_interval ←selected | 0.00 | 0.00 | — (best) | 0.229 | yes |
| power_base_family_prior | 0.05 | 1.16 | no (within ~2·dse) | 0.219 | yes |
| exception_rule_family_prior_2 | 0.54 | 1.88 | no (within ~2·dse) | 0.133 | yes |
| erlang_interval_size_family_prior | 0.61 | 1.49 | no (within ~2·dse) | 0.125 | yes |
| parity_bounded_range_concepts | 0.87 | 2.56 | no (within ~2·dse) | 0.096 | yes |
| erlang_interval_size_family_prior_2 | 0.90 | 1.53 | no (within ~2·dse) | 0.093 | yes |
| exception_rule_family_prior | 1.20 | 1.94 | no (within ~2·dse) | 0.069 | yes |
| erlang_interval_size_exception_prior | 1.84 | 1.56 | no (within ~2·dse) | 0.036 | yes |
