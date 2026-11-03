# Existing hypotheses

Each model below is ONE cognitive hypothesis, with how it stands on the current data by ELPD-LOO (best first). `elpd_diff ± dse` is a model's deficit against the best and the standard error of that difference (with the responses to each stimulus pair counted together, as they are correlated): within about 2·dse the two are statistically tied on this data; beyond it the model has lost. "PSIS-LOO unreliable" means the estimate itself is untrustworthy (too many high-Pareto-k trials), not that the model is bad. Propose a hypothesis that is genuinely different from these, or a refinement of a single one of them — never a combination of several.

## erlang_interval_simplicity_search  — rank 0, the best model on this data, ELPD-LOO -890.4

Refinement of `simplicity_cost_concept_search`: people still build concepts from mental primitives, paying a fitted cost per extra primitive for rules, weighing examples by the size principle and committing nearly all-or-none to the best-explaining concept. The one change (taken from `tenenbaum_rules_similarity`) is the prior over numerical intervals: instead of treating every stretch of the number line as equally natural, people favour intervals of a typical width (an Erlang prior on interval size with a fitted scale), so tight ranges hugging the examples are a priori unlikely and rules win more often — predicting less "same decade / nearby number" generalisation and stricter respect for shared properties such as parity than the incumbent.

## simplicity_cost_concept_search  — rank 1, 8.8 ± 3.9 nats behind the best (2.3× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -899.2

People search for a concept by building it from simple mental primitives, and every extra primitive a concept needs costs them: the simplest concepts (odd, even) come to mind readily, while compound concepts (multiples of 7, ending in 2, powers of 3, a rule with an exception) are penalised in proportion to how many primitives describe them, on top of the size principle and the near-all-or-none commitment to the best-explaining concept. This complexity cost makes people fall back on broad parity rules much more than the size principle alone predicts — e.g. saying yes to any even number after 22, 62, 26, 82 rather than only to numbers ending in 2.

## no_default_lapse_simplicity_search  — rank 2, 10.5 ± 3.7 nats behind the best (2.8× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -900.8

Refinement of `simplicity_cost_concept_search`: people still search for a concept built from simple primitives (a fitted cost per extra primitive, the size principle, near-all-or-none commitment to the best concept), and on a fraction of trials they lapse and answer without consulting the concept. The one change is what a lapse produces: instead of a coin-flip yes/no, a lapsing person falls back on a learned default answer — a fitted, typically low, base rate of saying yes (a "when in doubt, say no" bias) — so parity-violating and merely nearby targets get fewer stray yeses and repeated answers to the same item are more consistent than the coin-flip lapse predicts.

## parity_salient_concept_learning  — rank 3, 24.4 ± 8.2 nats behind the best (3.0× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -914.8

Refinement of `concept_posterior_with_lapse_guessing`: people do Bayesian concept learning over the same 45 rules and all intervals of 1-100 (size principle, lapse-and-guess response stage), but parity — odd versus even — is a privileged, highly accessible concept whose prior is boosted above the other rules by a fitted salience factor. The one change is this non-uniform rule prior favouring odd/even, which lets parity survive the size principle against tighter rules and intervals, so people say yes to parity-matching targets of an all-odd or all-even set far more than the uniform-prior learner predicts.

## interval_breadth_biased_commitment  — rank 4, 25.8 ± 11.7 nats behind the best (2.2× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -916.2

Refinement of `breadth_biased_rule_commitment`: people commit nearly all-or-none to the concept (a mathematical rule or a numerical interval) that best explains the examples under the size principle, with broad rules a priori more natural in proportion to their size. The one change is that the same size-graded naturalness applies to numerical ranges: instead of all intervals being equally likely a priori, an interval's prior grows (or shrinks) with its width by a fitted breadth exponent, so if people favour broad ranges, tight "nearby number" intervals rarely win and people say yes to non-example numbers in the examples' own decade less than a flat interval prior predicts.

## breadth_biased_rule_commitment  — rank 5, 29.1 ± 11.9 nats behind the best (2.4× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -919.5

Refinement of `sharpened_map_concept_selection`: people still commit, nearly all-or-none, to the one or few concepts (a mathematical rule or a numerical interval) that best explain the examples under the size principle, and answer yes only if the target is in that concept. The one change is the prior over rules: rather than treating all rules as equally likely, people find broad, familiar rules (odd, even, and other large categories) a priori more natural in proportion to how many numbers they cover, with the strength of this breadth bias fitted from data — so a broad rule like parity can win over tighter rules and intervals when all examples share it, predicting stronger parity-based generalization than the incumbent, as the critique reports.

## serial_position_weighted_concept_search  — rank 6, 33.6 ± 13.2 nats behind the best (2.5× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -924.0

People search for the simplest concept that explains the examples (complexity-costed rules and intervals, size principle, near-all-or-none commitment), but they do not weigh the examples equally: the evidence each example contributes depends on its position in the order shown, growing or shrinking steadily from the first example to the last at a fitted rate. If the first example anchors the guess (primacy), a concept fitting the early examples survives a later example that breaks it; if the latest example dominates (recency), the concept is pulled toward what the last examples share.

## weakened_size_principle_sampling  — rank 7, 49.6 ± 11.4 nats behind the best (4.4× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -939.9

People do Bayesian concept learning over mathematical rules and numerical intervals, but they apply the size principle only partially: they discount the evidence that a small concept explains the examples better than a large one, as if the examples were only partly sampled from the concept rather than just happening to fall in it. This weakened size penalty keeps broad concepts such as "odd numbers", "even numbers" or a wide stretch of the number line alive, so people generalise along parity and across a region of numbers more than an ideal strong-sampling learner does.

## confidence_criterion_decision  — rank 8, 64.1 ± 14.3 nats behind the best (4.5× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -954.5

People infer how likely the target is to belong to the hidden concept exactly as a simple Bayesian concept learner would (rules and intervals under the size principle), but they do not report that belief by probability matching: they compare it against a personal criterion of how sure they must be to say "yes", answering yes when their confidence exceeds the criterion and no otherwise, with only a little noise around the cut-off. This soft-threshold decision rule makes answers far more all-or-none than the belief itself, and shifts borderline targets toward "no" (or "yes") depending on whether the criterion sits above (or below) even odds.

## sharpened_map_concept_selection  — rank 9, 71.6 ± 15.4 nats behind the best (4.7× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -961.9

People do not average over all concepts consistent with the examples; they commit, nearly all-or-none, to the one or few concepts (a mathematical rule or a tight numerical interval) that best explain the examples under the size principle, and answer yes only if the target falls in the concept they committed to. This sharpened, winner-take-most choice of concept makes answers more deterministic and leaves far less graded "nearby number" generalization than full Bayesian averaging predicts.

## parity_restricted_range_commitment  — rank 10, 76.0 ± 15.0 nats behind the best (5.1× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -966.4

Refinement of `sharpened_map_concept_selection`: people still commit, nearly all-or-none, to the concept that best explains the examples under the size principle, but the concepts they can commit to include parity-restricted ranges ("even numbers between 16 and 64", "odd numbers from 31 to 47") alongside plain mathematical rules and plain intervals. The one change is adding these parity-by-interval conjunctions to the hypothesis space (with a fitted share of the interval prior), so that when all examples share parity people say yes to parity-matching targets inside the examples' range and no to opposite-parity near neighbours and to numbers outside the range.

## concept_posterior_with_lapse_guessing  — rank 11, 85.0 ± 17.2 nats behind the best (5.0× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -975.3

Refinement of `bigelow_piantadosi_mixture`: people do Bayesian concept learning over the same 45 rules and all intervals of 1-100 (size principle, rule prior lambda_0), but they do not answer every trial from that posterior — on a fixed fraction of trials (a lapse) they ignore the examples and guess "yes" with a fixed base rate. The one change is this lapse-and-guess response stage, which makes answers to the same (set, target) less deterministic than the concept posterior alone implies and pulls confident predictions toward the guess rate.

## tempered_posterior_commitment  — rank 12, 94.0 ± 18.3 nats behind the best (5.1× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -984.3

People hold a belief over candidate concepts (mathematical rules and numeric intervals, with the size principle) whose commitment differs from Bayesian averaging by a single tempering exponent: the posterior over concepts is raised to a free power and renormalised, so people either commit almost entirely to the best concept (exponent above 1) or hedge conservatively across many weaker ones (exponent below 1). It disagrees most sharply with the averaging model on sets where a rule and an interval compete, such as targets just outside the examples' range or numeric neighbours that fit no rule, because there the weight on the runner-up concepts (set by the exponent) decides the answer.

## overweighted_evidence_rules_intervals  — rank 13, 124.7 ± 21.8 nats behind the best (5.7× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1015.0

Refinement of `tenenbaum_rules_similarity`: people do Bayesian concept learning over the same mathematical rules and Erlang-weighted intervals, but they treat the examples as stronger evidence than ideal random sampling warrants — the size-principle likelihood is raised to a fitted power (an "evidence-weight" exponent), so the posterior concentrates on the tightest rules and intervals consistent with the examples. The one change is this fitted exponent (fixed at 1 in the original), which predicts less generalization to numerical neighbours and to numbers just outside the examples' range than the ideal learner, as the critique of the incumbent reports.

## range_bounded_rule_learning  — rank 14, 153.6 ± 27.7 nats behind the best (5.5× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1043.9

People never entertain an unbounded mathematical rule on its own: every concept they consider is a rule confined to a numerical range (e.g. "even numbers between 20 and 60", with "any number between 20 and 60" as the plain-interval case), and they infer which range-bounded rule produced the examples under the size principle, answering yes by how much of that inference contains the target. Because a rule bounded tightly around the examples is always a smaller, better explanation than the same rule across all of 1-100, this predicts strong use of shared properties like parity inside the examples' span but sharply less generalization of those properties to targets outside it.

## tversky_prototype_feature_contrast  — rank 15, 178.5 ± 29.6 nats behind the best (6.0× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1068.8

People judge membership by feature matching rather than by Bayesian inference over concepts: they abstract a prototype made of every property all the examples share (parity, divisibility, being a square/cube/prime/power, last digit, digits contained, decade, lying within the examples' range), and say "yes" the more of those shared properties the target has, penalised for each shared property it lacks and, more weakly, for each property it has that the prototype lacks (Tversky's contrast model of similarity). There is no size principle: a broad shared property counts as much as a narrow one.

## stray_tolerant_concept_learning  — rank 16, 188.3 ± 30.1 nats behind the best (6.3× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1078.7

Refinement of bigelow_piantadosi_mixture: people do Bayesian concept learning over the same 45 rules and all intervals of 1-100 with the same grammar prior, and say yes with the posterior probability that the concept contains the target. The one change is that how often people think an example is a stray number unrelated to the concept is learned from the data rather than fixed at 10%; tolerating more strays lets a scattered set be read as a tight rule plus an outlier, so people generalise less to the examples' numerical neighbours than the incumbent predicts.

## bigelow_piantadosi_mixture  — rank 17, 222.3 ± 34.2 nats behind the best (6.5× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1112.7

Bayesian concept learning over 45 rules (odd, even, primes, squares,
cubes, powers of 2-10, powers of 2 without 32 or with 37, multiples of
3-9, 11 and 12, ending in 0-9, containing the digit 0-9) and every
interval of 1-100. A rule is chosen with probability lambda_0, and the
rules are equally likely; each interval has prior (1 - lambda_0) / 100^2.
Examples are sampled from the concept (the size principle), 10% of them
uniformly from 1-100, and "yes" is the posterior probability that the
concept contains the target.

## tenenbaum_rules_similarity  — rank 18, 278.1 ± 41.7 nats behind the best (6.7× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1168.5

Bayesian concept learning over 33 mathematical rules (odd, even, squares,
cubes, primes, multiples of 3-12, powers of 2-10, ending in 1-9) and every
interval of 1-100. The rules share a prior fraction lambda equally; the
intervals share the rest by an Erlang density on their size (scale sigma).
Examples are sampled from the concept (the size principle), 10% of them
uniformly from 1-100, and "yes" is the posterior probability that the
concept contains the target.

## digit_space_exemplar_similarity  — rank 19, 279.7 ± 35.3 nats behind the best (7.9× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1170.0

People judge a target by its exemplar similarity to the examples in a digit-based representation of numbers: each number is coded by its tens digit and its ones digit, the psychological distance between the target and an example is a weighted city-block distance (a graded penalty for each step apart in tens digit plus a penalty for a differing ones digit, with fitted attention weights on the two dimensions), and similarity falls off exponentially with that distance. The probability of saying yes rises with the target's summed similarity to all the examples, so targets sharing the examples' final digit or lying in their decades are accepted in graded fashion, with no rule inference or size principle.

## multiplicative_not_exponential_concepts  — no comparison row

People's concept space for numbers is built on multiplication, not exponentiation: they readily entertain parity, "multiples of m", final digits, digits contained and numerical ranges, but concepts that require repeated multiplication — powers of a base, perfect squares, perfect cubes — are a priori much less accessible, carrying a fitted extra cost on top of the ordinary per-primitive simplicity cost, size principle and near-all-or-none commitment. So after 32, 4, 16 or 4, 64 people generalise to other multiples of 4 (12, 72) rather than restricting themselves to powers of 2 or 4, extending shared-divisor rules more than a learner who finds powers as natural as multiples.

## additive_cluster_exemplar_similarity  — no comparison row

People do not infer a hidden concept; they compare the target to each stored example, one at a time, and judge how similar the two numbers are by the stored number categories they have in common (odd/even, multiples of m, squares, powers, last digit, digits contained, and stretches of the number line), with a shared small category counting far more than a shared broad one. They say yes the more the target resembles every example (a geometric-mean, all-examples-must-match comparison), so a target sharing a divisor with all the examples, or lying close to all of them, is accepted in graded fashion.

## stepwise_progression_induction  — no comparison row

People assume the hidden program is a counting procedure: it starts at some number and steps upward by a fixed stride until it stops (e.g. "from 22 to 82 in steps of 4", "from 8 to 64 in steps of 8", or "every number from 40 to 60" for a stride of 1). They infer which start, stride and stop produced the examples — preferring small strides and, by the size principle, short progressions — and say yes to a target if the inferred progressions land on it. This predicts generalisation along shared step sizes (multiples of m, numbers spaced by the examples' common difference) more than fixed rule lists do, and less generalisation to immediate neighbours (1-3 away) that fall between the steps than to numbers a full stride away.
