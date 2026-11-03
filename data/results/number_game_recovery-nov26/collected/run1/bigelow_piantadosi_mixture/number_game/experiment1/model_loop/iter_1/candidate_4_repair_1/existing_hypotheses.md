# Existing hypotheses

Each model below is ONE cognitive hypothesis, with how it stands on the current data by ELPD-LOO (best first). `elpd_diff ± dse` is a model's deficit against the best and the standard error of that difference (with the responses to each stimulus pair counted together, as they are correlated): within about 2·dse the two are statistically tied on this data; beyond it the model has lost. "PSIS-LOO unreliable" means the estimate itself is untrustworthy (too many high-Pareto-k trials), not that the model is bad. Propose a hypothesis that is genuinely different from these, or a refinement of a single one of them — never a combination of several.

## bigelow_piantadosi_independent  — rank 0, the best model on this data, ELPD-LOO -872.8

Bayesian concept learning over 45 rules (odd, even, primes, squares,
cubes, powers of 2-10, powers of 2 without 32 or with 37, multiples of
3-9, 11 and 12, ending in 0-9, containing the digit 0-9) and every
interval of 1-100, with a prior weight inferred for each rule separately
and a share lambda_0 for the rules as a group; each interval has prior
(1 - lambda_0) / 100^2. Examples are sampled from the concept (the size
principle), 10% of them uniformly from 1-100, and "yes" is the posterior
probability that the concept contains the target.

## bayesian_concepts_logodds_readout  — rank 1, 0.4 ± 2.3 nats behind the best (0.2× dse: statistically tied with the best), ELPD-LOO -873.2

Refinement of bigelow_piantadosi_independent: people do the same Bayesian concept learning over the 45 rules and all intervals (size principle, 10% noisy examples), but they do not report the posterior probability of membership directly; instead they answer "yes" through a graded, possibly biased decision readout in which the log-odds of membership is scaled by a free confidence factor and shifted by a free yes/no criterion. This lets near-certain posteriors (e.g. for the examples themselves, or for numbers far outside the concept) map onto less extreme, empirically observed yes rates, and lets judges be more or less liberal than the posterior.

## inferred_sampling_reliability_concepts  — rank 2, 0.7 ± 2.9 nats behind the best (0.2× dse: statistically tied with the best), ELPD-LOO -873.5

Refinement of bigelow_piantadosi_independent: people do Bayesian concept learning over the same 45 rules and all intervals of 1-100, with the size principle, but how strongly they believe the examples were drawn from the concept (rather than at random from 1-100) is a free parameter inferred from the data instead of being fixed at 90%. This one change sets how sharply the size principle favours small, specific concepts over broad ones, and so how tightly people generalise from a few examples.

## partial_strong_sampling_rules_intervals  — rank 3, 88.1 ± 21.9 nats behind the best (4.0× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -960.9

Refinement of the dropped `tenenbaum_rules_similarity` model (Bayesian concept learning over 33 mathematical rules and all intervals of 1-100, rules sharing a prior fraction equally and intervals weighted by an Erlang density on their size). The one change: learners do not assume every example was deliberately sampled from the concept (pure strong sampling, the size principle); they treat each example as drawn from the concept with probability theta (likelihood 1/|concept|) and otherwise as an arbitrary number that merely happens to fall in it (weak sampling, likelihood independent of concept size), with theta fitted — so the size principle is only partly applied, and generalisation is broader, e.g. beyond the examples' range and to larger rules, than strong sampling predicts. (The original's crash on cross-validation folds, a data-shape dependence on the fold, is also fixed.)

## spread_scaled_exemplar_similarity  — rank 4, 258.5 ± 29.4 nats behind the best (8.8× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1131.3

People do not infer any mathematical rule; they generalize by numerical similarity alone, judging a target more likely to be produced the closer it lies to the individual examples on the number line. The breadth of that similarity gradient is set by how spread out the examples are: tightly clustered examples yield a narrow gradient, widely scattered examples a broad one, so the same distance counts for less when the examples cover a wide range.

## exemplar_digit_magnitude_gcm  — rank 5, 263.1 ± 25.7 nats behind the best (10.2× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1135.9

People do not infer a hidden rule; they judge a target by its summed similarity to the remembered examples (an exemplar, generalized-context-model process). Similarity to each example decays exponentially with a city-block psychological distance built from how far apart the two numbers are in magnitude and whether they differ in their tens digit and in their ones digit, and the probability of "yes" grows with the target's total similarity to the examples relative to a response criterion.

## exemplar_similarity_number_line  — rank 6, 292.8 ± 27.2 nats behind the best (10.7× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1165.6

People do not induce any rule or interval: they judge a target by its summed perceptual similarity on the number line to the examples shown, with similarity decaying exponentially with numeric distance (an exemplar / GCM account), and say "yes" in proportion to that summed similarity relative to a fixed response criterion. This disagrees most sharply with the rule-and-interval learner in the gaps between widely spaced examples (e.g. 50 for the set 92,14,20,5, where an interval spanning the examples predicts "yes" but similarity predicts "no") and for targets just outside the examples' range (where the interval predicts "no" but similarity predicts "yes").

## leaky_memory_concept_learning  — no comparison row

People do Bayesian concept learning over mathematical rules and intervals (with the size principle), but they read the examples one at a time and their memory of earlier examples fades: each example's evidence is discounted by a free forgetting factor for every example shown after it, so the most recent example counts fully and the first one of a long set counts least. As a result, concepts that fit only the later examples (and rules or intervals violated by an early example) remain plausible, and generalisation from long sets is broader and recency-biased.

## serial_position_weighted_concepts  — no comparison row

People do Bayesian concept learning over the same 45 mathematical rules and all intervals of 1-100 (size principle, 10% noisy examples), but they do not weigh the examples equally: because the examples are seen in sequence, each example's evidence is weighted by its serial position, with a single fitted order-bias parameter that shifts weight toward the most recent examples (recency) or toward the first ones (primacy) while keeping the total amount of evidence fixed. So the same examples shown in different orders lead to different generalisations, favouring concepts that fit the weighted-up examples.

## map_single_concept_selection  — no comparison row

People do not average over all the concepts consistent with the examples: they commit to the single most probable concept (a rule such as "powers of 2" or "multiples of 10", or a numeric interval, chosen by a fixed prior and the size principle) and answer "yes" exactly when the target belongs to that one concept, apart from occasional response errors. Generalisation is therefore all-or-none relative to the winning concept, with no graded uncertainty from runner-up concepts.

## erlang_interval_prior_concepts  — no comparison row

Refinement of bigelow_piantadosi_independent: people do the same Bayesian concept learning over the 45 rules and all intervals of 1-100 (size principle, 10% noisy examples, a free prior weight per rule), but they do not regard every interval as equally plausible a priori. Instead, their prior over intervals follows an Erlang density on the interval's size with a fitted characteristic size, so very narrow ranges hugging the examples are a priori implausible and ranges of a typical intermediate width are favoured; this one change controls how far beyond the examples' span people generalise when they think in terms of a numerical range.
