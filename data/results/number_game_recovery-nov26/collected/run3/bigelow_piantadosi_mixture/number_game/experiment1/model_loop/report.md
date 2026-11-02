# Inner Model Loop Report

Each model below is ONE distinct cognitive hypothesis. The posterior mass shows which single hypothesis best explains the data — it is **not** a recipe to combine the top models into a blend.

- Best model: **iter4_candidate5** (posterior=0.203, elpd_loo=-835.66)
- Trials: 2250
- Models compared: 8

## Posterior over models (ELPD-LOO)

| model | posterior | elpd_loo |
| --- | --- | --- |
| iter4_candidate5 | 0.2025 | -835.66 |
| uncertain_reliability_noise_concepts | 0.1980 | -835.68 |
| digit_sum_concepts | 0.1873 | -835.74 |
| family_pooled_uncertain_reliability | 0.1482 | -835.97 |
| complement_stray_reliability_concepts | 0.1423 | -836.01 |
| family_pooled_rule_prior | 0.0663 | -836.78 |
| family_pooled_win_stay_lose_sample | 0.0375 | -837.34 |
| compressed_number_line_ranges | 0.0179 | -838.09 |

## Hypotheses

- **iter4_candidate5**: Refinement of `win_stay_lose_sample_concepts`: people learn the concept one example at a time, holding a single working guess (a rule or magnitude range) that they keep with some probability while it still contains each new example and otherwise re-pick from the Bayesian posterior given the examples so far, so early examples shape generalisation more than later ones. The one change, taken from `uncertain_reliability_noise_concepts`, is the posterior people re-pick from: they are unsure how reliable the program is and average over that uncertainty, so strays are correlated within a set. One example that breaks a tight rule makes the remaining ones suspect too and pushes re-picked guesses toward broader concepts, while a fully consistent prefix strongly favours the tight rule. This predicts order effects (from the sticky guess) that interact with how consistent the early examples are.
- **uncertain_reliability_noise_concepts**: Refinement of `inferred_example_noise_concepts`: people do the same Bayesian concept learning over the 45 rules (each with its own inferred prior weight) and all intervals of 1-100 with the size principle, allowing that some examples are stray draws from 1-100 — but they do not know the program's reliability, so instead of assuming one fixed stray rate they hold a belief over how reliable the program is and average over it for each example set. The one change is this uncertainty about the noise rate, which makes strays exchangeable but correlated within a set: one example that breaks a tight rule makes people suspect the program is unreliable on that set, so they discount the remaining examples too and generalise more broadly, while a set fully consistent with a tight rule is taken as strong evidence for it.
- **digit_sum_concepts**: Refinement of `uncertain_reliability_noise_concepts`: people do the same Bayesian concept learning over the 45 rules and all intervals of 1-100 (size principle, uncertain belief in how often examples are stray draws), but they also entertain digit-arithmetic concepts — "numbers whose digits add up to s", one concept for each digit sum from 1 to 18 — which together hold a small a-priori share of the rule prior (about 4%, inferred around that value) split equally among them. The one change is this enlarged concept space, which predicts that sets whose examples share a digit sum (e.g. 23, 41, 50, 14) draw "yes" for other numbers with that digit sum (32, 5, 68) even though no arithmetic rule, final digit or magnitude range explains them.
- **family_pooled_uncertain_reliability**: Refinement of `uncertain_reliability_noise_concepts`: people do the same Bayesian concept learning over the 45 rules and all intervals of 1-100 with the size principle, holding an uncertain belief about how reliable the program is (how often an example is a stray draw from 1-100) and averaging over it for each example set — but their prior over rules is organised by kind of rule rather than set rule by rule. The one change, taken from `family_pooled_rule_prior`, is this family-structured, partially pooled rule prior: each rule's plausibility is drawn around a shared level for its family (parity, primes, squares and cubes, powers, multiples, final digit, contained digit) with an inferred spread, so rules of the same kind (e.g. "contains a 3" and "contains a 7") are about equally plausible unless the evidence says otherwise, which predicts more uniform generalisation across rules within a family than the incumbent's independent rule weights.
- **complement_stray_reliability_concepts**: Refinement of `uncertain_reliability_noise_concepts`: people do the same Bayesian concept learning over the 45 rules (each with its own inferred prior weight) and all intervals of 1-100 with the size principle, and they are just as unsure how reliable the program is on each set — but they think of a stray example as a mistake, a number that does *not* belong to the concept, rather than a draw that could land anywhere in 1-100. The one change is this stray distribution, drawn from the concept's complement: an example that a broad concept (e.g. "even", "contains a 3") misses is better explained as a stray than under a uniform stray generator, whereas a tight concept gains nothing. So people keep broad rules that miss one example more readily, and examples that a concept already contains are always taken as genuine evidence for it.
- **family_pooled_rule_prior**: Refinement of `inferred_example_noise_concepts`: people do the same Bayesian concept learning over the 45 rules and all intervals of 1-100 (size principle, an inferred rate at which examples are stray draws), but their prior over rules is organised by kind of rule rather than set rule by rule: each rule's plausibility is drawn around a shared level for its family (parity, primes, squares and cubes, powers, multiples, final digit, contained digit), with an inferred spread, so rules of the same kind (e.g. "contains a 3" and "contains a 7") are about equally plausible unless the evidence says otherwise. The one change is this family-structured, partially pooled rule prior, which replaces the incumbent's independent weight for every rule and predicts more uniform generalisation across rules within a family.
- **family_pooled_win_stay_lose_sample**: Refinement of `win_stay_lose_sample_concepts`: people learn the concept one example at a time, holding a single working guess (a rule or magnitude range picked by size-principle Bayesian concept learning) that they keep with some probability while it still contains each new example and otherwise re-pick from the posterior given all examples so far — so early examples shape generalisation more than later ones. The one change, taken from `family_pooled_rule_prior`, is the prior from which guesses are picked: rules are a priori plausible by kind (parity, primes, squares and cubes, powers, multiples, final digit, contained digit), each rule's weight drawn around a shared family level with an inferred spread, rather than an independently learned weight for every rule, so rules of the same kind (e.g. "contains a 3" and "contains a 7") are about equally likely to be adopted as the working guess.
- **compressed_number_line_ranges**: People do Bayesian concept learning over mathematical rules and magnitude ranges of 1-100 (size principle, a prior weight per rule, an uncertain belief in how often an example is a stray draw), but they represent magnitude ranges on a compressed mental number line, on which small numbers take up more room than large ones, and they treat a range's examples as spread evenly along that line, not evenly over the integers. So ranges among small numbers feel mentally wide and give weak evidence, which makes generalisation broad, while equally wide ranges among large numbers feel narrow and tighten quickly. Within a range, its lower members are more typical than its upper ones, so people extend a range further upward than downward. How compressed the line is gets inferred, from logarithmic to linear.

## Distinguishability (arviz.compare, PSIS-LOO)

`elpd_diff` and `dse` are relative to the best model. A model is only clearly worse than the best when `elpd_diff > 2 * dse`; models within ~2·dse of the top are statistically indistinguishable. `LOO reliable` is False when PSIS-LOO flagged this model's estimate as untrustworthy (many high Pareto-k points) — its row should be read with caution.

| model | elpd_diff | dse | distinguishable from best | weight | LOO reliable |
| --- | --- | --- | --- | --- | --- |
| iter4_candidate5 ←selected | 0.00 | 0.00 | — (best) | 0.203 | yes |
| uncertain_reliability_noise_concepts | 0.02 | 1.35 | no (within ~2·dse) | 0.198 | yes |
| digit_sum_concepts | 0.08 | 1.33 | no (within ~2·dse) | 0.187 | yes |
| family_pooled_uncertain_reliability | 0.31 | 3.33 | no (within ~2·dse) | 0.148 | yes |
| complement_stray_reliability_concepts | 0.35 | 3.85 | no (within ~2·dse) | 0.142 | yes |
| family_pooled_rule_prior | 1.12 | 3.46 | no (within ~2·dse) | 0.066 | yes |
| family_pooled_win_stay_lose_sample | 1.69 | 3.31 | no (within ~2·dse) | 0.038 | yes |
| compressed_number_line_ranges | 2.43 | 1.54 | no (within ~2·dse) | 0.018 | yes |
