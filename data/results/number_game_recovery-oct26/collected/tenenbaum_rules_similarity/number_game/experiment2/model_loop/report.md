# Inner Model Loop Report

Each model below is ONE distinct cognitive hypothesis. The posterior mass shows which single hypothesis best explains the data — it is **not** a recipe to combine the top models into a blend.

- Best model: **last_example_marks_range_top** (posterior=0.386, elpd_loo=-2058.17)
- Trials: 4500
- Models compared: 8

## Posterior over models (ELPD-LOO)

| model | posterior | elpd_loo |
| --- | --- | --- |
| last_example_marks_range_top | 0.3857 | -2058.17 |
| top_marking_uncertain_sampling | 0.1927 | -2058.86 |
| uncertain_sampling_top_marking | 0.1654 | -2059.01 |
| top_marking_range_sampling | 0.0976 | -2059.54 |
| upper_edge_marking_ranges | 0.0467 | -2060.28 |
| uncertain_sampling_assumption | 0.0427 | -2060.37 |
| upper_edge_discounted_range_size | 0.0367 | -2060.52 |
| prime_concept_excludes_one | 0.0325 | -2060.64 |

## Hypotheses

- **last_example_marks_range_top**: Refinement of `top_marking_range_sampling`: people do the same Bayesian concept learning over arithmetic rules (primes excluding 1), every range of 1-100 and range-restricted rules, and believe that, with some inferred odds, the program deliberately shows its range's top number and draws the other examples at random — but they believe the program ends its demonstration on that top number. The one change: the marked top is read from the order of presentation, so a range ending at the largest example gets the generative top-marking credit (larger for wider ranges) only when the largest example is the last one shown, and then at full strength whatever the set size, while sets whose largest example comes earlier get no edge credit — after 20,35,60 people say "yes" to 45 but not 62, whereas after 60,35,20 they extend above 60 as readily as below 20.
- **top_marking_uncertain_sampling**: Refinement of `top_marking_range_sampling`: people do the same Bayesian concept learning over arithmetic rules (primes excluding 1), every range of 1-100 and range-restricted rules, and believe that with inferred odds the program deliberately shows its range's top number and draws the other examples at random — but, the one change grafted from `uncertain_sampling_assumption`, they are unsure whether the remaining examples were drawn from the program's concept (strong sampling, size principle) or are merely numbers it happens to accept (weak sampling), and average over both readings with an inferred probability. So the marked top still blocks upward generalization, while broad concepts ("even numbers", wide ranges ending at the largest example) keep a floor of support however many examples fit a narrower one, and "yes" tightens with more examples less than the incumbent predicts.
- **uncertain_sampling_top_marking**: Refinement of `uncertain_sampling_assumption`. People still do Bayesian concept learning over arithmetic rules (primes excluding 1) and ranges of 1-100, and they are still unsure whether the examples were drawn from the program's concept (strong sampling) or are just numbers it happens to accept (weak sampling). The one change, taken from `top_marking_range_sampling`, is how the largest example marks the top of a range: instead of a fixed bonus, people think that with some inferred odds the program deliberately showed the top of its range and drew the other examples at random. A range ending at the largest example is then credited by how unlikely a chance draw would have been to hit that number under each sampling reading. Under strong sampling this credit is large for wide ranges and shrinks as more examples are shown; under weak sampling it is flat. So downward generalization from sparse sets is strong, and upward generalization is blocked, under both readings.
- **top_marking_range_sampling**: Refinement of `upper_edge_marking_ranges`: people do the same Bayesian concept learning over arithmetic rules (primes excluding 1), every range of 1-100 and range-restricted rules, and believe the program shows where its range stops at the top — but they reason about this generatively rather than granting a fixed bonus. The one change: people assume that, with some inferred odds, the program deliberately shows its range's top number and draws the other examples at random, so the largest example is explained by marking instead of by chance; a range ending at the largest example is therefore credited in proportion to how unlikely a random draw would have been to hit that number — far more for wide ranges than narrow ones, and less as more examples are shown — so single examples and sparse sets license especially strong downward generalization (after 60, "yes" to 40 but not 62), while dense sets fall back toward ordinary size-principle ranges.
- **upper_edge_marking_ranges**: Refinement of `range_restricted_rule_concepts` (with its prime concept corrected to exclude 1, as in the incumbent): people do the same Bayesian concept learning over arithmetic rules, every range of 1-100 and range-restricted rules, but they assume the program's examples mark where its range stops at the top — the largest example shown is taken as evidence that the range ends there. The one change, grafted in one-sided form from `endpoint_marking_interval_sampling`: a range whose upper end coincides with the largest example gets an inferred evidence bonus, while its lower end is unconstrained, so people generalize downward from the examples more readily than upward (after 16 they accept 15 or 13 more than 17 or 19; after 83,77 they reject 84).
- **uncertain_sampling_assumption**: People do Bayesian concept learning over arithmetic rules (primes excluding 1) and numeric ranges of 1-100 (moderate, decade-bounded ranges preferred, the largest example marking a range's top), but they are unsure how the examples were chosen: with an inferred probability they assume the examples were drawn from the program's concept (strong sampling, so a small concept that fits is strongly favoured), and otherwise that the examples are merely numbers the program happens to accept (weak sampling, so every concept containing them stays as plausible as it was a priori). Their "yes" averages over both readings, so broad concepts such as "even numbers" or wide ranges keep a floor of support however many examples fit a narrower one, and generalization tightens with more examples less than an ideal strong-sampling learner predicts — for rules and ranges alike.
- **upper_edge_discounted_range_size**: Refinement of `upper_edge_marking_ranges`: people do the same Bayesian concept learning over arithmetic rules (primes excluding 1), every range of 1-100 and range-restricted rules, and they take the largest example as marking where the program's range stops at the top. The one change, grafted from `range_discounted_size_principle`: people apply the size principle at full strength only to arithmetic rules and only partly to numeric ranges, judging a range's fit to the examples as if it were wider than it is (by an inferred degree), so it is the marked top edge — not the size principle — that stops range generalization going upward, while downward from the examples "yes" spreads further across their neighbourhood than strong sampling predicts.
- **prime_concept_excludes_one**: Refinement of `range_restricted_rule_concepts`: people do the same Bayesian concept learning over arithmetic rules, every range of 1-100 and range-restricted rules (kind-level rule prior, Erlang width prior with a decade-boundary preference, size principle with 10% random examples, "yes" = posterior probability the concept contains the target), but their concept of "prime numbers" is the mathematical one, which excludes 1. The one change is removing 1 from the prime concept: when 1 is among the examples (e.g. 31,3,1) people cannot explain the set as primes and fall back on "odd numbers" (yes to 9 or 25, no special yes to 2), and after a set of primes (e.g. 67,31,17,61) they answer "no" to 1 rather than accepting it as a prime.

## Distinguishability (arviz.compare, PSIS-LOO)

`elpd_diff` and `dse` are relative to the best model. A model is only clearly worse than the best when `elpd_diff > 2 * dse`; models within ~2·dse of the top are statistically indistinguishable. `LOO reliable` is False when PSIS-LOO flagged this model's estimate as untrustworthy (many high Pareto-k points) — its row should be read with caution.

| model | elpd_diff | dse | distinguishable from best | weight | LOO reliable |
| --- | --- | --- | --- | --- | --- |
| last_example_marks_range_top ←selected | 0.00 | 0.00 | — (best) | 0.386 | yes |
| top_marking_uncertain_sampling | 0.69 | 3.16 | no (within ~2·dse) | 0.193 | yes |
| uncertain_sampling_top_marking | 0.85 | 3.67 | no (within ~2·dse) | 0.165 | yes |
| top_marking_range_sampling | 1.37 | 2.82 | no (within ~2·dse) | 0.098 | yes |
| upper_edge_marking_ranges | 2.11 | 2.37 | no (within ~2·dse) | 0.047 | yes |
| uncertain_sampling_assumption | 2.20 | 3.24 | no (within ~2·dse) | 0.043 | yes |
| upper_edge_discounted_range_size | 2.35 | 3.33 | no (within ~2·dse) | 0.037 | yes |
| prime_concept_excludes_one | 2.47 | 2.53 | no (within ~2·dse) | 0.032 | yes |
