# Existing hypotheses

Each model below is ONE cognitive hypothesis, with how it stands on the current data by ELPD-LOO (best first). `elpd_diff ± dse` is a model's deficit against the best and the standard error of that difference (with the responses to each stimulus pair counted together, as they are correlated): within about 2·dse the two are statistically tied on this data; beyond it the model has lost. "PSIS-LOO unreliable" means the estimate itself is untrustworthy (too many high-Pareto-k trials), not that the model is bad. Propose a hypothesis that is genuinely different from these, or a refinement of a single one of them — never a combination of several.

## recency_weighted_even_odd_family_prior  — rank 0, the best model on this data, ELPD-LOO -871.3

Refinement of `even_odd_split_family_prior`: people do the same Bayesian concept learning over the same 45 rules (with separate prior plausibilities for even, odd and the other kinds of rule) and all intervals of 1-100 with a tempered size principle, but they do not weigh every example equally — they read the examples in order and each example's evidence counts more the later it was shown (a fitted order gradient, with the total amount of evidence held fixed). The single change is grafting this recency weighting of the examples (from `recency_weighted_concept_learning`) onto the incumbent, motivated by the critique that targets near the last-shown example get more yeses than targets near the first-shown one, which an order-blind learner cannot produce.

## even_odd_split_family_prior  — rank 1, 1.3 ± 1.3 nats behind the best (1.0× dse: statistically tied with the best), ELPD-LOO -872.7

Refinement of `rule_family_prior_tempered_mixture`: people do the same Bayesian concept learning over the same 45 rules and all intervals of 1-100, with a tempered size principle and a prior over kinds of rule, but they do not treat "even" and "odd" as one kind of rule with a single shared plausibility — "even numbers" and "odd numbers" are separate kinds, each with its own prior plausibility inferred from the data. The single change is splitting the parity family in two, predicting that an all-even set of examples triggers parity generalization (yeses to distant same-parity targets) more readily than an all-odd set does.

## recency_decay_even_odd_family_prior  — rank 2, 2.0 ± 2.0 nats behind the best (1.0× dse: statistically tied with the best), ELPD-LOO -873.3

Refinement of `even_odd_split_family_prior`: people do the same Bayesian concept learning over the same 45 rules (with separate prior plausibilities for even, odd and the other kinds of rule) and all intervals of 1-100 with a tempered size principle, but they read the examples one at a time and their memory of earlier examples fades, so each example's evidence counts less the more examples were shown after it. The single change is grafting this recency-weighted evidence (from `recency_weighted_concept_learning`) onto the incumbent, motivated by the critique that people say yes more to targets near the last-shown example than to targets near the first-shown one, which the order-blind incumbent cannot produce.

## stochastic_example_recall_sampling  — rank 3, 2.8 ± 1.7 nats behind the best (1.6× dse: statistically tied with the best), ELPD-LOO -874.1

People do not reason from all the examples they were shown, but from the examples they can recall when judging the target: the most recent example is always remembered, while each earlier example is recalled only by chance, less often the further back it was shown, and the learner infers the hidden concept (by Bayesian rule-and-range inference) from whichever examples were recalled. Because the answer averages over these all-or-none recall outcomes, targets near the last example are endorsed more than targets near the first, and sets shown earlier in a sequence partly behave like smaller, broader-generalizing sets.

## arithmetic_progression_concepts  — rank 4, 3.1 ± 1.7 nats behind the best (1.9× dse: statistically tied with the best), ELPD-LOO -874.5

Refinement of `even_odd_split_family_prior`: people do the same Bayesian concept learning (tempered size principle, separate even and odd kinds, a prior over kinds of rule), but their space of concepts also includes regularly spaced sequences that do not start at zero — "every m-th number starting from r" (numbers leaving the same remainder r when divided by m, for spacings 3–12, e.g. 1, 5, 9, 13, …), a new kind of rule with its own prior plausibility inferred from the data. The single change is adding this arithmetic-progression kind, predicting that sets whose examples share a common spacing (such as 5, 9, 17, 33) earn yeses for distant targets continuing that spacing, which no multiple, parity or final-digit rule covers.

## rule_family_prior_tempered_mixture  — rank 5, 16.8 ± 4.6 nats behind the best (3.7× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -888.2

Refinement of `tempered_size_principle_mixture`: people do the same Bayesian concept learning over the same 45 rules and all intervals of 1-100 with a partially applied (tempered) size principle, but they do not treat every rule as equally plausible a priori — they hold a prior over *kinds* of rule (parity, primes, squares/cubes, powers, multiples, final digit, contained digit), each kind having its own plausibility, inferred from the data, and the rules within a kind sharing it equally. The single change is this family-structured prior in place of the uniform rule prior: it predicts that whole kinds of rule (e.g. "contains the digit 2" or "multiples of 4") are rarely entertained even when examples fit them, while parity and final-digit rules compete strongly with numerical intervals.

## erlang_interval_family_prior  — rank 6, 17.8 ± 5.1 nats behind the best (3.5× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -889.1

Refinement of `rule_family_prior_tempered_mixture`: people do the same Bayesian concept learning over the same 45 rules (with a prior over kinds of rule) and all intervals of 1-100 with a tempered size principle, but they do not consider every numerical range equally plausible a priori — they expect ranges of a typical, intermediate width (an Erlang prior on interval size with a fitted scale, as in Tenenbaum's number game), so very narrow and very broad ranges are discounted before any examples are seen. The single change is grafting this size-graded interval prior (from `learned_rule_prior_erlang_intervals`) onto the incumbent in place of its uniform interval prior, predicting that magnitude-based generalization from one or two examples extends to a characteristic distance around them instead of collapsing onto the tightest covering range or spreading over the whole 1-100 line.

## recency_weighted_rule_family_prior  — rank 7, 18.3 ± 5.0 nats behind the best (3.6× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -889.6

Refinement of `rule_family_prior_tempered_mixture`: people do the same Bayesian concept learning over the same 45 rules (with a prior over kinds of rule) and all intervals of 1-100 with a tempered size principle, but they read the examples in the order shown and their memory of earlier examples fades, so each example's evidence counts less the further back it was shown and the most recent examples dominate which concept is inferred. The single change is grafting this recency-weighted evidence (from `recency_weighted_concept_learning`) onto the family-prior model, motivated by the critique that targets near the last-shown example are endorsed more than targets near the first-shown one, which no order-blind learner can produce.

## response_sharpened_tempered_rule_prior  — rank 8, 30.2 ± 18.2 nats behind the best (1.7× dse: statistically tied with the best), ELPD-LOO -901.5

Refinement of `tempered_learned_rule_prior`: people do the same Bayesian concept learning over 45 rules (each with its own learned prior plausibility) and all intervals of 1-100 with a tempered size principle, but they turn their belief that the target is in the concept into a yes/no answer more decisively than probability matching — their answer follows the belief's log-odds scaled by a free decisiveness exponent, so targets the learner thinks are probably in (or probably out of) the concept are answered yes (or no) more consistently than the belief itself. The single change is this sharpened belief-to-response mapping (a one-parameter log-odds scaling, from `llo_distorted_concept_posterior`) grafted onto `tempered_learned_rule_prior`, motivated by the critique that people's repeated answers to the same set and target are more unanimous than the graded posterior predicts.

## tempered_learned_rule_prior  — rank 9, 31.7 ± 18.4 nats behind the best (1.7× dse: statistically tied with the best), ELPD-LOO -903.0

Refinement of `learned_rule_prior_mixture`: people learn concepts by the same Bayesian inference over 45 mathematical rules (each with its own learned prior plausibility) and all intervals of 1-100, but they apply the size principle only partially — they discount the evidence that the examples were sampled from a small concept, so the likelihood of the examples is tempered by a free sampling-strength exponent (1 = full strong sampling, 0 = weak sampling). The single change is grafting this tempered size principle (from `tempered_size_principle_mixture`) onto the learned-rule-prior model, predicting that broad favoured rules and wide intervals keep more posterior weight after a few examples than an ideal strong-sampling learner allows.

## recency_weighted_concept_learning  — rank 10, 44.4 ± 22.1 nats behind the best (2.0× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -915.7

People infer the hidden concept by Bayesian inference over mathematical rules (each with its own prior plausibility) and numerical intervals, but they read the examples one at a time and their memory of earlier examples fades: each example's evidence counts less the further back in the sequence it was shown, so the most recent examples dominate which concept is inferred. This predicts that the same set shown in a different order yields different generalization, with targets that fit the last examples (but not the first ones) endorsed more than an ideal, order-blind learner would.

## first_example_win_stay_lose_sample  — rank 11, 47.8 ± 8.1 nats behind the best (5.9× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -919.1

People do not weigh all examples together; they form a guess about the concept from the first example they see and then test it against the later examples ("win-stay, lose-sample"). If every later example fits the guess formed from the first example, they keep it; only when an example contradicts it do they abandon it and re-infer the concept from all the examples. Because the first example's guess survives whenever it is consistent, broad concepts suggested by the first example (e.g. "even", a wide range around it) persist far more than an ideal learner allows, and the same set in a different order can yield different generalization.

## learned_rule_prior_mixture  — rank 12, 51.0 ± 27.9 nats behind the best (1.8× dse: statistically tied with the best), ELPD-LOO -922.3

Refinement of bigelow_piantadosi_mixture: people learn concepts by the same Bayesian inference over 45 mathematical rules and all intervals of 1-100 with the size principle, but they do not hold the rules as equally likely a priori — each rule (e.g. odd, even, ends in a digit) carries its own prior plausibility, inferred from the data. The single change is replacing the uniform prior over rules with a learned, non-uniform one (the core idea of Bigelow & Piantadosi's prior inference), so favoured rules such as parity and shared final digits can dominate interval-based similarity when the examples support them.

## learned_rule_prior_erlang_intervals  — rank 13, 55.6 ± 27.1 nats behind the best (2.1× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -926.9

Refinement of `learned_rule_prior_mixture`: people learn concepts by the same Bayesian inference over 45 mathematical rules (each with its own learned prior plausibility) and all intervals of 1-100 with the full size principle, but they do not consider every numerical range equally plausible a priori — they expect ranges of a typical, intermediate width (an Erlang prior on interval size with a fitted scale, as in Tenenbaum's number game), so very narrow and very broad ranges are discounted before any examples are seen. The single change is replacing the incumbent's uniform prior over intervals with this size-graded interval prior (taken from `tenenbaum_rules_similarity`), predicting that magnitude-based generalization from one or two examples extends to a characteristic distance around them rather than collapsing onto the tightest range that covers them.

## tempered_size_principle_mixture  — rank 14, 157.2 ± 22.4 nats behind the best (7.0× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1028.5

Refinement of `bigelow_piantadosi_mixture`: people do the same Bayesian concept learning over the same 45 rules and all intervals of 1-100, but they apply the size principle only partially — they discount the evidence that the examples were sampled from a small concept, so the likelihood of the examples is tempered by a free "sampling strength" exponent (1 = the incumbent's full strong sampling, 0 = weak sampling). With weaker sampling, broad rules such as odd/even and digit rules keep more posterior weight against tight intervals, predicting stronger parity generalization, more yeses to far targets and less pure numerical-proximity generalization.

## llo_distorted_concept_posterior  — rank 15, 168.2 ± 21.7 nats behind the best (7.7× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1039.6

People infer the hidden concept as an ideal Bayesian learner would — over the same 45 mathematical rules (equally likely a priori) and all intervals of 1-100, with the size principle — but they misread their own belief when turning it into a yes/no answer: the posterior probability that the target is in the concept is distorted linearly in log-odds (Zhang & Maloney-style probability weighting), compressed toward a reference point. As a result, near-certain members get fewer yeses and near-certain non-members more yeses than the ideal learner predicts, while the ordering of targets is left intact.

## lapse_guess_concept_learning  — rank 16, 198.0 ± 20.8 nats behind the best (9.5× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1069.3

People infer the hidden concept as an ideal Bayesian learner would — over the same 45 mathematical rules (equally likely a priori) and all intervals of 1-100, with the size principle — and normally answer by probability matching to their belief that the target is in the concept, but on a fixed fraction of trials they do not consult that belief at all and instead guess, with their own overall bias toward "yes" or "no". This lapse-and-guess decision rule predicts a floor of yeses for clear non-members and a ceiling below one for clear members (including the examples themselves), both set by the same lapse rate and guessing bias and independent of how certain the inference is.

## tempered_rules_similarity  — rank 17, 204.9 ± 26.7 nats behind the best (7.7× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1076.2

Refinement of `tenenbaum_rules_similarity`: people do Bayesian concept learning over the same mathematical rules and numerical intervals, but update conservatively — they under-weight the evidence of the examples relative to their prior (the example likelihood is raised to a fitted power below one, a weakened size principle). As a result broad concepts like "even" or "odd" and wide intervals survive a few examples better than for an ideal learner, producing more yeses to distant parity-matching targets and less sharply peaked generalization to numbers right next to the examples.

## featural_exemplar_similarity  — rank 18, 208.3 ± 25.0 nats behind the best (8.3× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1079.6

People do not induce a hidden rule; they judge a target by its featural similarity to the remembered examples (exemplar generalization over symbolic number features). A target is accepted in proportion to its average similarity to the examples, where similarity decays with the number of salient surface features it fails to share with each example — its parity, its final digit, and its decade — rather than with its numerical distance.

## digit_parity_exemplar_similarity  — rank 19, 217.1 ± 24.6 nats behind the best (8.8× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1088.4

People do not induce a hidden rule at all; they judge a target by its exemplar similarity to the examples shown, where two numbers are similar to the extent they share surface features (same parity, same final digit) and are close in magnitude, and "yes" grows with the target's average similarity to the examples. This disagrees most sharply with the rule-based Bayesian best model on targets that match only some examples' final digit or share the examples' parity without fitting any whole rule (graded yeses where the rule model says no), and on near-in-magnitude targets that break the examples' parity/digit pattern (fewer yeses than interval hypotheses predict).

## ones_digit_exemplar_similarity  — rank 20, 241.2 ± 26.1 nats behind the best (9.2× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1112.6

People judge a target by comparing its final (ones) digit with the final digit of each example, ignoring magnitude and mathematical rules: a target is endorsed in proportion to how many examples it resembles in surface form, where an example with the identical final digit counts fully and an example whose final digit merely has the same parity (odd/even) counts partially. This graded, example-by-example digit matching predicts yeses for targets that share a final digit with only some of the examples and for parity-matching targets far from every example.

## recency_weighted_category_exemplar  — rank 21, 244.0 ± 26.2 nats behind the best (9.3× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1115.3

People do not infer a hidden concept; they judge a target by its summed similarity to the remembered examples (an exemplar account), where two numbers are similar to the extent they are close in magnitude and share membership in stored number categories (even, odd, primes, squares, powers, multiples, final digit, contained digit), with membership in small, distinctive categories counting more than in broad ones. The examples are held in memory with strengths that fade with how far back in the sequence they were shown, so the most recently seen examples dominate the comparison — predicting more yeses to targets resembling the last example than to targets resembling only the first.

## divisor_overlap_similarity  — rank 22, 347.2 ± 32.9 nats behind the best (10.6× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1218.5

People judge a target by its arithmetic similarity to the examples rather than by magnitude: two numbers are similar to the extent that they share divisors (the overlap of their sets of divisors greater than one), and the willingness to say "yes" rises smoothly with the target's average divisor-overlap similarity to the examples. Because even numbers all share the divisor 2 while odd numbers share no common divisor, this predicts strong parity-like generalization from even sets, much weaker generalization from odd sets, and graded yeses for targets sharing factors (e.g. multiples of 3 or 5) with some of the examples.

## bigelow_piantadosi_mixture  — rank 23, 390.9 ± 46.6 nats behind the best (8.4× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1262.2

Bayesian concept learning over 45 rules (odd, even, primes, squares,
cubes, powers of 2-10, powers of 2 without 32 or with 37, multiples of
3-9, 11 and 12, ending in 0-9, containing the digit 0-9) and every
interval of 1-100. A rule is chosen with probability lambda_0, and the
rules are equally likely; each interval has prior (1 - lambda_0) / 100^2.
Examples are sampled from the concept (the size principle), 10% of them
uniformly from 1-100, and "yes" is the posterior probability that the
concept contains the target.

## tenenbaum_rules_similarity  — rank 24, 533.2 ± 59.8 nats behind the best (8.9× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1404.5

Bayesian concept learning over 33 mathematical rules (odd, even, squares,
cubes, primes, multiples of 3-12, powers of 2-10, ending in 1-9) and every
interval of 1-100. The rules share a prior fraction lambda equally; the
intervals share the rest by an Erlang density on their size (scale sigma).
Examples are sampled from the concept (the size principle), 10% of them
uniformly from 1-100, and "yes" is the posterior probability that the
concept contains the target.

## interval_only_size_principle  — rank 25, 657.0 ± 37.8 nats behind the best (17.4× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1528.4

People generalize from the examples purely by numerical magnitude, ignoring every arithmetic or digit rule: they infer which contiguous range of 1-100 the program draws from, favouring tighter ranges that still cover all the examples (the size principle) under a prior preference for ranges of a typical width, and say "yes" with the probability that the target lies inside the inferred range. This predicts near-certain yeses between the smallest and largest example, a smooth fall-off just outside them that sharpens as more examples are shown, and no special boost for targets that share the examples' parity, final digit or multiplicative structure.

## previous_set_intrusion_concept_learning  — no comparison row

People infer the hidden concept by Bayesian rule-and-range inference (tempered size principle, a prior over kinds of rule with separate even and odd kinds), but the examples of the previously shown, different set linger in memory and intrude as extra, down-weighted evidence about the current set's concept (proactive interference between trials). This disagrees most sharply with the order-blind, trial-independent best model whenever a set follows an unrelated set: the pooled, partly conflicting examples push belief toward broad ranges and broad rules (more yeses to distant targets), or, when the previous set shares the current set's pattern (e.g. both even), toward that shared rule — so the same set and target get different answers depending on what was shown just before.

## familiarity_relaxed_size_principle  — no comparison row

People do the same recency-weighted Bayesian concept learning over rules and numerical ranges as the best model, but how strongly they apply the size principle depends on how familiar the example set is: the first time a participant meets a set they read its examples as tightly sampled from a small concept, and each further time they are asked about that same set (having already been probed with other targets for it) they discount that sampling evidence more, so their generalization broadens. This predicts fewer yeses on a set's first encounter than on its repeats, concentrated on targets outside the tightest range or rule covering the examples.
