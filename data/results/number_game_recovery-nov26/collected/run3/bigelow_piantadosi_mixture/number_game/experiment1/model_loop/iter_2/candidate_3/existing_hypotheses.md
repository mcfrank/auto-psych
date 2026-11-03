# Existing hypotheses

Each model below is ONE cognitive hypothesis, with how it stands on the current data by ELPD-LOO (best first). `elpd_diff ± dse` is a model's deficit against the best and the standard error of that difference (with the responses to each stimulus pair counted together, as they are correlated): within about 2·dse the two are statistically tied on this data; beyond it the model has lost. "PSIS-LOO unreliable" means the estimate itself is untrustworthy (too many high-Pareto-k trials), not that the model is bad. Propose a hypothesis that is genuinely different from these, or a refinement of a single one of them — never a combination of several.

## inferred_example_noise_concepts  — rank 0, the best model on this data, ELPD-LOO -839.6

Refinement of `bigelow_piantadosi_independent`: people do the same Bayesian concept learning over its 45 rules (each with its own inferred prior weight) and all intervals of 1-100 with the size principle, but how much they trust the examples to come from the concept is a free parameter rather than fixed at 90%. The one change is this inferred example-noise rate: the more people believe some examples are stray draws from 1-100, the more readily they keep a tight rule or range that misses one example, rather than retreating to a broad concept that covers all of them.

## bigelow_piantadosi_independent  — rank 1, 0.7 ± 0.7 nats behind the best (1.0× dse: statistically tied with the best), ELPD-LOO -840.3

Bayesian concept learning over 45 rules (odd, even, primes, squares,
cubes, powers of 2-10, powers of 2 without 32 or with 37, multiples of
3-9, 11 and 12, ending in 0-9, containing the digit 0-9) and every
interval of 1-100, with a prior weight inferred for each rule separately
and a share lambda_0 for the rules as a group; each interval has prior
(1 - lambda_0) / 100^2. Examples are sampled from the concept (the size
principle), 10% of them uniformly from 1-100, and "yes" is the posterior
probability that the concept contains the target.

## interval_width_prior_concepts  — rank 2, 1.0 ± 0.9 nats behind the best (1.0× dse: statistically tied with the best), ELPD-LOO -840.6

Refinement of `bigelow_piantadosi_independent`: the same Bayesian concept learner over its 45 rules (each with its own inferred prior weight) and all intervals of 1-100, with the size principle, but people's prior over magnitude intervals depends on the interval's width — each interval's prior is proportional to its width raised to an inferred exponent (0 recovers the incumbent's uniform interval prior; negative values mean narrow ranges like "numbers in the 60s" are a priori more plausible than wide ones). The single change is this width-dependent interval prior, which governs how readily people generalise to nearby versus distant magnitudes when no rule explains the examples.

## log_odds_compressed_concept_posterior  — rank 3, 1.6 ± 1.0 nats behind the best (1.7× dse: statistically tied with the best), ELPD-LOO -841.2

Refinement of `bigelow_piantadosi_independent`: people do the same Bayesian concept learning over the 45 rules and all intervals of 1-100 (size principle, a prior weight inferred per rule), but they do not report the posterior probability that the target belongs faithfully — their subjective sense of that probability is distorted linearly in log-odds, so confident beliefs are pulled toward (or pushed away from) even odds, as in classic probability-weighting. The one change is this single probability-distortion slope, which predicts that near-certain "yes" and near-certain "no" cases are both answered less extremely, with the compression strongest for the most confident judgments, unlike a fixed random-response lapse that adds the same floor everywhere.

## bayesian_rules_with_response_lapse  — rank 4, 1.6 ± 1.0 nats behind the best (1.6× dse: statistically tied with the best), ELPD-LOO -841.2

Refinement of bigelow_piantadosi_independent: people do the same Bayesian concept learning over the 45 rules and all intervals (size principle, a prior weight inferred per rule), but on a fixed fraction of trials they do not consult that posterior at all and answer yes or no at random. The one change is this response-lapse rate, which lets the model explain "no" answers to targets that are themselves examples and stray "yes" answers to targets that no plausible concept contains.

## leaky_example_memory_concepts  — rank 5, 1.7 ± 1.5 nats behind the best (1.2× dse: statistically tied with the best), ELPD-LOO -841.3

People generalize by Bayesian concept learning over mathematical rules and magnitude intervals (the size principle, a prior weight per rule), but their working memory is leaky: each example shown is independently retained with some probability and the forgotten ones play no part in the inference. So a judgment is the rule-and-interval posterior computed from whichever examples survive, which predicts weaker rule-based tightening and broader generalization for larger example sets than full Bayesian learning, and occasional "no" answers to targets that were themselves examples.

## tenenbaum_digit_rule_inventory  — rank 6, 2.6 ± 5.4 nats behind the best (0.5× dse: statistically tied with the best), ELPD-LOO -842.2

Refinement of `tenenbaum_rules_similarity`: people do Tenenbaum's Bayesian concept learning (rules share one prior fraction equally, intervals are weighted by an Erlang density on their size, examples are sampled from the concept), but the rules they entertain are the richer inventory of the Bigelow-Piantadosi grammar — notably "contains the digit d" and "ends in 0", plus the two near-miss powers-of-2 sets — instead of Tenenbaum's 33. The one change is the hypothesis space; the claim is that what the people consider, not a separately learned weight for every rule, is what separates the two models, so equal rule weights with two free parameters should suffice.

## inferred_outlier_rate_rule_inventory  — rank 7, 3.6 ± 5.5 nats behind the best (0.7× dse: statistically tied with the best), ELPD-LOO -843.2

Refinement of `tenenbaum_digit_rule_inventory`: people do the same Bayesian concept learning over the 45 Bigelow-Piantadosi rules (one equal, shared prior weight) and all magnitude intervals (Erlang prior on size), with examples sampled from the concept, but how strongly they believe an example may be a random stray rather than a draw from the concept is a property of the learner, inferred from the data instead of fixed at 10%. The one change is this inferred outlier rate: a high rate means people tolerate examples that break a rule and weaken the size principle (generalising more broadly), while a low rate means a single inconsistent example rules a concept out and tight concepts dominate.

## map_single_concept_commitment  — rank 8, 240.7 ± 26.0 nats behind the best (9.3× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1080.3

People do not average over all the concepts consistent with the examples; they commit to the single concept that best explains them — the most probable rule or magnitude interval under a size-principle Bayesian evaluation with a flat prior (rules as a group as likely as intervals as a group) — and answer "yes" exactly when the target belongs to that one concept, apart from occasional response errors. This predicts all-or-none generalization: no graded falloff near the examples' range and no partial endorsement of targets that only some plausible concepts contain (e.g. 32 after "16,8,64" is rejected if the smaller "powers of 2 without 32" wins).

## tenenbaum_rules_similarity  — rank 9, 312.3 ± 47.5 nats behind the best (6.6× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1151.9

Bayesian concept learning over 33 mathematical rules (odd, even, squares,
cubes, primes, multiples of 3-12, powers of 2-10, ending in 1-9) and every
interval of 1-100. The rules share a prior fraction lambda equally; the
intervals share the rest by an Erlang density on their size (scale sigma).
Examples are sampled from the concept (the size principle), 10% of them
uniformly from 1-100, and "yes" is the posterior probability that the
concept contains the target.

## digit_exemplar_similarity  — rank 10, 349.6 ± 33.3 nats behind the best (10.5× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1189.2

People do not induce a rule at all; they judge a target by its summed similarity to the remembered examples, where numbers are represented by their two decimal digits: a target feels like it "belongs" to the extent that its tens digit is close to an example's tens digit and its ones digit matches an example's ones digit. Attention is split between the tens (magnitude) and ones (final-digit) dimensions, similarity decays exponentially with this digit-wise distance, and the probability of saying yes rises with the average similarity of the target to the examples.

## digit_overlap_exemplar_similarity  — rank 11, 385.7 ± 33.0 nats behind the best (11.7× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1225.2

People do not induce mathematical rules or magnitude intervals at all; they judge a target by its surface resemblance to the examples, treating each number as the set of decimal digits it is written with, and say "yes" in proportion to the average digit-overlap similarity between the target and the examples (an exemplar-similarity process over written digits). This disagrees most sharply with the rule-based Bayesian model on targets that share a digit with some examples but fit no rule or interval consistent with all of them (e.g. 12 after "31,19,49,100"), and on rule-consistent targets written with entirely different digits (e.g. 2 after "16,8,64"), where it predicts low endorsement.

## spread_scaled_range_generalization  — rank 12, 417.3 ± 33.4 nats behind the best (12.5× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1256.9

People do not search over mathematical rules; they treat the examples as a sample from a band of magnitudes and generalize by how far the target falls outside the span the examples cover, with a tolerance that grows with how spread out the examples are. A target inside the span of the examples is endorsed at a constant high rate, and endorsement decays smoothly with the target's distance beyond the nearest edge, more slowly for widely spaced examples than for tightly clustered ones (or a single example).
