# Existing hypotheses

Each model below is ONE cognitive hypothesis, with how it stands on the current data by ELPD-LOO (best first). `elpd_diff ± dse` is a model's deficit against the best and the standard error of that difference (with the responses to each stimulus pair counted together, as they are correlated): within about 2·dse the two are statistically tied on this data; beyond it the model has lost. "PSIS-LOO unreliable" means the estimate itself is untrustworthy (too many high-Pareto-k trials), not that the model is bad. Propose a hypothesis that is genuinely different from these, or a refinement of a single one of them — never a combination of several.

## erlang_interval_family_prior_concepts  — rank 0, the best model on this data, ELPD-LOO -1083.6

Refinement of `erlang_interval_size_concept_learning` (the narrowest-losing live model): people do Bayesian concept learning over the same 45 rules and magnitude ranges, with the size principle and an Erlang prior that favours ranges of a characteristic, inferred width over very wide or single-number ones. The one change is how prior belief is shared among the rules: instead of an independent weight for each of the 45 rules, people first favour a kind of rule (parity, primes, squares/cubes, powers, multiples, last digit, contained digit), each kind with its own inferred plausibility, and treat rules of the same kind as equally likely. So whole kinds of rule (e.g. "contains the digit d") can be implausible a priori, while magnitude generalisation follows ranges of a preferred size.

## rule_family_prior_concepts  — rank 1, 1.6 ± 2.5 nats behind the best (0.6× dse: statistically tied with the best), ELPD-LOO -1085.1

Refinement of `bigelow_piantadosi_mixture`: people do Bayesian concept learning over the same 45 rules and all intervals (size principle, 10% of examples random), but their prior over rules is organised by kind of rule rather than flat: they first favour a family of rule (parity, primes, squares/cubes, powers, multiples, last digit, contained digit), each with its own inferred plausibility, and rules within a family are equally likely. The one change is replacing the mixture's uniform rule prior with this inferred family-level prior, so whole kinds of rule (e.g. "contains the digit d") can be a priori implausible while others (multiples, last digit) dominate generalization.

## serial_position_weighted_evidence  — rank 2, 3.2 ± 2.6 nats behind the best (1.2× dse: statistically tied with the best), ELPD-LOO -1086.8

People do Bayesian concept learning over the same rules (with a family-organised rule prior) and magnitude ranges as the best model, but they update sequentially as the examples are read and do not weigh every example equally: each example's evidence is weighted by its position in the sequence, decaying (or growing) geometrically with how far it lies before the last example shown, at an inferred rate. A recency rate means earlier examples are partly forgotten, so longer sets narrow generalisation less than the size principle predicts; a primacy rate means early examples are overcounted, so longer sets narrow it more.

## evidence_overweighting_concept_learning  — rank 3, 3.9 ± 2.6 nats behind the best (1.5× dse: statistically tied with the best), ELPD-LOO -1087.4

People do Bayesian concept learning over the same rules (with a family-organised rule prior) and magnitude intervals as the best model, using the size principle, but they mis-weight the evidence: each example's likelihood is raised to an inferred power, so the examples count for more (or less) than a normative learner would let them. If people over-weight evidence, their generalisation collapses onto the smallest fitting concept faster as examples accumulate than the size principle predicts, which would explain why their yes-rate drops more from two- to four-example sets than the best model expects.

## log_number_line_range_concepts  — rank 4, 12.7 ± 4.9 nats behind the best (2.6× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1096.3

Refinement of `rule_family_prior_concepts`: people do the same Bayesian concept learning over the 45 rules (with the family-organised rule prior) and magnitude ranges of 1-100, with size-principle sampling, but they picture the ranges on a compressed, roughly logarithmic mental number line rather than a linear one. The one change is the prior over ranges: a range's endpoints are equally likely to fall anywhere on that mental line, with the degree of compression inferred, so ranges among small numbers (where numbers are spread out on the mental line) are a priori more plausible than equally long ranges among large numbers (where they are crowded together). As a result, examples among large numbers are explained by a range less readily and by a rule more often than the incumbent predicts.

## membership_criterion_threshold  — rank 5, 23.2 ± 6.5 nats behind the best (3.6× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1106.8

People infer how likely the target is to belong to the hidden concept exactly as the current best model's Bayesian concept learner does (same rules and magnitude ranges, family-organised rule prior, size principle), but they do not report that probability directly: they answer "yes" only when the inferred membership probability exceeds an internal decision criterion, and this criterion varies noisily from judgment to judgment. So targets with middling membership probability are mostly rejected (or mostly accepted, depending on where the criterion sits), and answers are more all-or-none than probability matching predicts.

## erlang_interval_size_concept_learning  — rank 6, 81.2 ± 12.0 nats behind the best (6.8× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1164.7

A refinement of bigelow_piantadosi_independent: people do Bayesian concept learning over the same 45 rules and all intervals of 1-100 (prior weight per rule, share lambda_0 for rules, size-principle sampling with 10% random examples, "yes" = posterior probability the concept contains the target). The one change is the prior over intervals: instead of every interval being equally likely (which makes wide ranges a priori far more numerous), people expect a range of a characteristic width, with an Erlang prior over interval size whose scale is inferred from the data (as in Tenenbaum's number game), so moderately narrow ranges are preferred a priori and very wide or single-number ranges are penalised.

## bigelow_piantadosi_independent  — rank 7, 84.2 ± 12.0 nats behind the best (7.0× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1167.8

Bayesian concept learning over 45 rules (odd, even, primes, squares,
cubes, powers of 2-10, powers of 2 without 32 or with 37, multiples of
3-9, 11 and 12, ending in 0-9, containing the digit 0-9) and every
interval of 1-100, with a prior weight inferred for each rule separately
and a share lambda_0 for the rules as a group; each interval has prior
(1 - lambda_0) / 100^2. Examples are sampled from the concept (the size
principle), 10% of them uniformly from 1-100, and "yes" is the posterior
probability that the concept contains the target.

## weak_sampling_concept_learning  — rank 8, 202.0 ± 21.1 nats behind the best (9.6× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1285.5

People entertain the same kinds of number concepts (arithmetic rules and magnitude ranges) but treat the examples only as consistent with the hidden concept rather than as random samples drawn from it (weak sampling, no size principle), so every concept that contains the examples stays plausible in proportion to its prior regardless of how many other numbers it includes. The sharpest disagreement with the size-principle best model lives in sets that fit a small rule nested in a larger one (e.g. 16,32,2,8 or 20,32,92): this model predicts broad "yes" to targets that only the larger concept contains (other even numbers, numbers anywhere in a wide range), whereas the best model confines "yes" to the smallest fitting concept.

## bigelow_piantadosi_mixture  — rank 9, 253.6 ± 28.1 nats behind the best (9.0× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1337.1

Bayesian concept learning over 45 rules (odd, even, primes, squares,
cubes, powers of 2-10, powers of 2 without 32 or with 37, multiples of
3-9, 11 and 12, ending in 0-9, containing the digit 0-9) and every
interval of 1-100. A rule is chosen with probability lambda_0, and the
rules are equally likely; each interval has prior (1 - lambda_0) / 100^2.
Examples are sampled from the concept (the size principle), 10% of them
uniformly from 1-100, and "yes" is the posterior probability that the
concept contains the target.

## exemplar_featural_similarity_gcm  — rank 10, 389.3 ± 28.0 nats behind the best (13.9× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1472.8

People do not infer a rule at all: they judge the target by its exemplar similarity to the examples shown, as in a generalized context model. Each example is compared with the target on a few perceptual dimensions of the number — its magnitude, its final digit, its tens digit and its parity — with attention shared among these dimensions, similarity decays exponentially with attention-weighted mismatch, and the more similar the target is on average to the examples, the more likely they say yes.

## residue_class_progression  — rank 11, 446.7 ± 38.3 nats behind the best (11.7× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1530.2

People assume the hidden program produces an arithmetic progression: all numbers in 1-100 that leave the same remainder when divided by some step size (step 1 = every number, step 2 = all odds or all evens, step 10 = a shared final digit, step 12 = every twelfth number, and so on). They consider every step size whose residue class contains all the examples, favour smaller steps a priori but larger (tighter) steps when several examples all fit them (the size principle), and say "yes" with the posterior probability that the target lies in the same progression as the examples.
