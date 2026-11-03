# Existing hypotheses

Each model below is ONE cognitive hypothesis, with how it stands on the current data by ELPD-LOO (best first). `elpd_diff ± dse` is a model's deficit against the best and the standard error of that difference (with the responses to each stimulus pair counted together, as they are correlated): within about 2·dse the two are statistically tied on this data; beyond it the model has lost. "PSIS-LOO unreliable" means the estimate itself is untrustworthy (too many high-Pareto-k trials), not that the model is bad. Propose a hypothesis that is genuinely different from these, or a refinement of a single one of them — never a combination of several.

## rule_family_prior_concepts  — rank 0, the best model on this data, ELPD-LOO -1085.1

Refinement of `bigelow_piantadosi_mixture`: people do Bayesian concept learning over the same 45 rules and all intervals (size principle, 10% of examples random), but their prior over rules is organised by kind of rule rather than flat: they first favour a family of rule (parity, primes, squares/cubes, powers, multiples, last digit, contained digit), each with its own inferred plausibility, and rules within a family are equally likely. The one change is replacing the mixture's uniform rule prior with this inferred family-level prior, so whole kinds of rule (e.g. "contains the digit d") can be a priori implausible while others (multiples, last digit) dominate generalization.

## erlang_interval_size_concept_learning  — rank 1, 79.6 ± 12.5 nats behind the best (6.4× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1164.7

A refinement of bigelow_piantadosi_independent: people do Bayesian concept learning over the same 45 rules and all intervals of 1-100 (prior weight per rule, share lambda_0 for rules, size-principle sampling with 10% random examples, "yes" = posterior probability the concept contains the target). The one change is the prior over intervals: instead of every interval being equally likely (which makes wide ranges a priori far more numerous), people expect a range of a characteristic width, with an Erlang prior over interval size whose scale is inferred from the data (as in Tenenbaum's number game), so moderately narrow ranges are preferred a priori and very wide or single-number ranges are penalised.

## bigelow_piantadosi_independent  — rank 2, 82.7 ± 12.2 nats behind the best (6.8× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1167.8

Bayesian concept learning over 45 rules (odd, even, primes, squares,
cubes, powers of 2-10, powers of 2 without 32 or with 37, multiples of
3-9, 11 and 12, ending in 0-9, containing the digit 0-9) and every
interval of 1-100, with a prior weight inferred for each rule separately
and a share lambda_0 for the rules as a group; each interval has prior
(1 - lambda_0) / 100^2. Examples are sampled from the concept (the size
principle), 10% of them uniformly from 1-100, and "yes" is the posterior
probability that the concept contains the target.

## weak_sampling_concept_learning  — rank 3, 200.4 ± 21.6 nats behind the best (9.3× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1285.5

People entertain the same kinds of number concepts (arithmetic rules and magnitude ranges) but treat the examples only as consistent with the hidden concept rather than as random samples drawn from it (weak sampling, no size principle), so every concept that contains the examples stays plausible in proportion to its prior regardless of how many other numbers it includes. The sharpest disagreement with the size-principle best model lives in sets that fit a small rule nested in a larger one (e.g. 16,32,2,8 or 20,32,92): this model predicts broad "yes" to targets that only the larger concept contains (other even numbers, numbers anywhere in a wide range), whereas the best model confines "yes" to the smallest fitting concept.

## bigelow_piantadosi_mixture  — rank 4, 252.0 ± 28.2 nats behind the best (8.9× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1337.1

Bayesian concept learning over 45 rules (odd, even, primes, squares,
cubes, powers of 2-10, powers of 2 without 32 or with 37, multiples of
3-9, 11 and 12, ending in 0-9, containing the digit 0-9) and every
interval of 1-100. A rule is chosen with probability lambda_0, and the
rules are equally likely; each interval has prior (1 - lambda_0) / 100^2.
Examples are sampled from the concept (the size principle), 10% of them
uniformly from 1-100, and "yes" is the posterior probability that the
concept contains the target.

## exemplar_featural_similarity_gcm  — rank 5, 387.7 ± 27.7 nats behind the best (14.0× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1472.8

People do not infer a rule at all: they judge the target by its exemplar similarity to the examples shown, as in a generalized context model. Each example is compared with the target on a few perceptual dimensions of the number — its magnitude, its final digit, its tens digit and its parity — with attention shared among these dimensions, similarity decays exponentially with attention-weighted mismatch, and the more similar the target is on average to the examples, the more likely they say yes.

## residue_class_progression  — rank 6, 445.1 ± 38.2 nats behind the best (11.7× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1530.2

People assume the hidden program produces an arithmetic progression: all numbers in 1-100 that leave the same remainder when divided by some step size (step 1 = every number, step 2 = all odds or all evens, step 10 = a shared final digit, step 12 = every twelfth number, and so on). They consider every step size whose residue class contains all the examples, favour smaller steps a priori but larger (tighter) steps when several examples all fit them (the size principle), and say "yes" with the posterior probability that the target lies in the same progression as the examples.
