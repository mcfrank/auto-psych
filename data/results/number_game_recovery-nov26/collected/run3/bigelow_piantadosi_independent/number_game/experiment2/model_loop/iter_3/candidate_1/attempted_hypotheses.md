# Tried before

44 hypotheses proposed earlier in this project are no longer in the model set. Read them before you propose:

- A *pruned* entry entered the set and later lost to the best model by the stated margin, on the data available then. Its mechanism may be partly right: a model that changes it substantively is welcome, but do not re-propose it unchanged or merely re-parameterised, under any name. Pruned models stay readable under `models/pruned/`.
- A *rejected* entry never entered the set. If it was a near-duplicate of a model still in the set, that region is already covered: do not re-propose it. If it failed on its code or its fit (see its outcome), the idea itself was never tested and a correct implementation may be worth trying.

### tversky_prototype_feature_contrast — pruned (experiment1 end of experiment)

**Outcome:** 258.2 nats behind costly_squares_final_digit_concepts (9.2× dse)

**Hypothesis:** People judge membership by feature matching rather than by Bayesian inference over concepts: they abstract a prototype made of every property all the examples share (parity, divisibility, being a square/cube/prime/power, last digit, digits contained, decade, lying within the examples' range), and say "yes" the more of those shared properties the target has, penalised for each shared property it lacks and, more weakly, for each property it has that the prototype lacks (Tversky's contrast model of similarity). There is no size principle: a broad shared property counts as much as a narrow one.

### tempered_posterior_commitment — pruned (experiment1 end of experiment)

**Outcome:** 173.7 nats behind costly_squares_final_digit_concepts (6.6× dse)

**Hypothesis:** People hold a belief over candidate concepts (mathematical rules and numeric intervals, with the size principle) whose commitment differs from Bayesian averaging by a single tempering exponent: the posterior over concepts is raised to a free power and renormalised, so people either commit almost entirely to the best concept (exponent above 1) or hedge conservatively across many weaker ones (exponent below 1). It disagrees most sharply with the averaging model on sets where a rule and an interval compete, such as targets just outside the examples' range or numeric neighbours that fit no rule, because there the weight on the runner-up concepts (set by the exponent) decides the answer.

### sharpened_map_concept_selection — pruned (experiment1 end of experiment)

**Outcome:** 151.3 nats behind costly_squares_final_digit_concepts (6.4× dse)

**Hypothesis:** People do not average over all concepts consistent with the examples; they commit, nearly all-or-none, to the one or few concepts (a mathematical rule or a tight numerical interval) that best explain the examples under the size principle, and answer yes only if the target falls in the concept they committed to. This sharpened, winner-take-most choice of concept makes answers more deterministic and leaves far less graded "nearby number" generalization than full Bayesian averaging predicts.

### stray_tolerant_concept_learning — pruned (experiment1 end of experiment)

**Outcome:** 268.1 nats behind costly_squares_final_digit_concepts (7.1× dse)

**Hypothesis:** Refinement of bigelow_piantadosi_mixture: people do Bayesian concept learning over the same 45 rules and all intervals of 1-100 with the same grammar prior, and say yes with the posterior probability that the concept contains the target. The one change is that how often people think an example is a stray number unrelated to the concept is learned from the data rather than fixed at 10%; tolerating more strays lets a scattered set be read as a tight rule plus an outlier, so people generalise less to the examples' numerical neighbours than the incumbent predicts.

### concept_posterior_with_lapse_guessing — pruned (experiment1 end of experiment)

**Outcome:** 164.7 nats behind costly_squares_final_digit_concepts (6.5× dse)

**Hypothesis:** Refinement of `bigelow_piantadosi_mixture`: people do Bayesian concept learning over the same 45 rules and all intervals of 1-100 (size principle, rule prior lambda_0), but they do not answer every trial from that posterior — on a fixed fraction of trials (a lapse) they ignore the examples and guess "yes" with a fixed base rate. The one change is this lapse-and-guess response stage, which makes answers to the same (set, target) less deterministic than the concept posterior alone implies and pulls confident predictions toward the guess rate.

### overweighted_evidence_rules_intervals — pruned (experiment1 end of experiment)

**Outcome:** 204.4 nats behind costly_squares_final_digit_concepts (7.8× dse)

**Hypothesis:** Refinement of `tenenbaum_rules_similarity`: people do Bayesian concept learning over the same mathematical rules and Erlang-weighted intervals, but they treat the examples as stronger evidence than ideal random sampling warrants — the size-principle likelihood is raised to a fitted power (an "evidence-weight" exponent), so the posterior concentrates on the tightest rules and intervals consistent with the examples. The one change is this fitted exponent (fixed at 1 in the original), which predicts less generalization to numerical neighbours and to numbers just outside the examples' range than the ideal learner, as the critique of the incumbent reports.

### simplicity_cost_concept_search — pruned (experiment1 end of experiment)

**Outcome:** 88.6 nats behind costly_squares_final_digit_concepts (5.0× dse)

**Hypothesis:** People search for a concept by building it from simple mental primitives, and every extra primitive a concept needs costs them: the simplest concepts (odd, even) come to mind readily, while compound concepts (multiples of 7, ending in 2, powers of 3, a rule with an exception) are penalised in proportion to how many primitives describe them, on top of the size principle and the near-all-or-none commitment to the best-explaining concept. This complexity cost makes people fall back on broad parity rules much more than the size principle alone predicts — e.g. saying yes to any even number after 22, 62, 26, 82 rather than only to numbers ending in 2.

### weakened_size_principle_sampling — pruned (experiment1 end of experiment)

**Outcome:** 129.3 nats behind costly_squares_final_digit_concepts (6.2× dse)

**Hypothesis:** People do Bayesian concept learning over mathematical rules and numerical intervals, but they apply the size principle only partially: they discount the evidence that a small concept explains the examples better than a large one, as if the examples were only partly sampled from the concept rather than just happening to fall in it. This weakened size penalty keeps broad concepts such as "odd numbers", "even numbers" or a wide stretch of the number line alive, so people generalise along parity and across a region of numbers more than an ideal strong-sampling learner does.

### range_bounded_rule_learning — pruned (experiment1 end of experiment)

**Outcome:** 233.3 nats behind costly_squares_final_digit_concepts (8.1× dse)

**Hypothesis:** People never entertain an unbounded mathematical rule on its own: every concept they consider is a rule confined to a numerical range (e.g. "even numbers between 20 and 60", with "any number between 20 and 60" as the plain-interval case), and they infer which range-bounded rule produced the examples under the size principle, answering yes by how much of that inference contains the target. Because a rule bounded tightly around the examples is always a smaller, better explanation than the same rule across all of 1-100, this predicts strong use of shared properties like parity inside the examples' span but sharply less generalization of those properties to targets outside it.

### breadth_biased_rule_commitment — pruned (experiment1 end of experiment)

**Outcome:** 108.9 nats behind costly_squares_final_digit_concepts (5.6× dse)

**Hypothesis:** Refinement of `sharpened_map_concept_selection`: people still commit, nearly all-or-none, to the one or few concepts (a mathematical rule or a numerical interval) that best explain the examples under the size principle, and answer yes only if the target is in that concept. The one change is the prior over rules: rather than treating all rules as equally likely, people find broad, familiar rules (odd, even, and other large categories) a priori more natural in proportion to how many numbers they cover, with the strength of this breadth bias fitted from data — so a broad rule like parity can win over tighter rules and intervals when all examples share it, predicting stronger parity-based generalization than the incumbent, as the critique reports.

### parity_restricted_range_commitment — pruned (experiment1 end of experiment)

**Outcome:** 155.8 nats behind costly_squares_final_digit_concepts (6.7× dse)

**Hypothesis:** Refinement of `sharpened_map_concept_selection`: people still commit, nearly all-or-none, to the concept that best explains the examples under the size principle, but the concepts they can commit to include parity-restricted ranges ("even numbers between 16 and 64", "odd numbers from 31 to 47") alongside plain mathematical rules and plain intervals. The one change is adding these parity-by-interval conjunctions to the hypothesis space (with a fitted share of the interval prior), so that when all examples share parity people say yes to parity-matching targets inside the examples' range and no to opposite-parity near neighbours and to numbers outside the range.

### parity_salient_concept_learning — pruned (experiment1 end of experiment)

**Outcome:** 104.1 nats behind costly_squares_final_digit_concepts (5.1× dse)

**Hypothesis:** Refinement of `concept_posterior_with_lapse_guessing`: people do Bayesian concept learning over the same 45 rules and all intervals of 1-100 (size principle, lapse-and-guess response stage), but parity — odd versus even — is a privileged, highly accessible concept whose prior is boosted above the other rules by a fitted salience factor. The one change is this non-uniform rule prior favouring odd/even, which lets parity survive the size principle against tighter rules and intervals, so people say yes to parity-matching targets of an all-odd or all-even set far more than the uniform-prior learner predicts.

### serial_position_weighted_concept_search — pruned (experiment1 end of experiment)

**Outcome:** 113.3 nats behind costly_squares_final_digit_concepts (5.3× dse)

**Hypothesis:** People search for the simplest concept that explains the examples (complexity-costed rules and intervals, size principle, near-all-or-none commitment), but they do not weigh the examples equally: the evidence each example contributes depends on its position in the order shown, growing or shrinking steadily from the first example to the last at a fitted rate. If the first example anchors the guess (primacy), a concept fitting the early examples survives a later example that breaks it; if the latest example dominates (recency), the concept is pulled toward what the last examples share.

### digit_space_exemplar_similarity — pruned (experiment1 end of experiment)

**Outcome:** 359.4 nats behind costly_squares_final_digit_concepts (11.1× dse)

**Hypothesis:** People judge a target by its exemplar similarity to the examples in a digit-based representation of numbers: each number is coded by its tens digit and its ones digit, the psychological distance between the target and an example is a weighted city-block distance (a graded penalty for each step apart in tens digit plus a penalty for a differing ones digit, with fitted attention weights on the two dimensions), and similarity falls off exponentially with that distance. The probability of saying yes rises with the target's summed similarity to all the examples, so targets sharing the examples' final digit or lying in their decades are accepted in graded fashion, with no rule inference or size principle.

### confidence_criterion_decision — pruned (experiment1 end of experiment)

**Outcome:** 143.9 nats behind costly_squares_final_digit_concepts (6.5× dse)

**Hypothesis:** People infer how likely the target is to belong to the hidden concept exactly as a simple Bayesian concept learner would (rules and intervals under the size principle), but they do not report that belief by probability matching: they compare it against a personal criterion of how sure they must be to say "yes", answering yes when their confidence exceeds the criterion and no otherwise, with only a little noise around the cut-off. This soft-threshold decision rule makes answers far more all-or-none than the belief itself, and shifts borderline targets toward "no" (or "yes") depending on whether the criterion sits above (or below) even odds.

### erlang_interval_simplicity_search — pruned (experiment1 end of experiment)

**Outcome:** 79.7 nats behind costly_squares_final_digit_concepts (4.9× dse)

**Hypothesis:** Refinement of `simplicity_cost_concept_search`: people still build concepts from mental primitives, paying a fitted cost per extra primitive for rules, weighing examples by the size principle and committing nearly all-or-none to the best-explaining concept. The one change (taken from `tenenbaum_rules_similarity`) is the prior over numerical intervals: instead of treating every stretch of the number line as equally natural, people favour intervals of a typical width (an Erlang prior on interval size with a fitted scale), so tight ranges hugging the examples are a priori unlikely and rules win more often — predicting less "same decade / nearby number" generalisation and stricter respect for shared properties such as parity than the incumbent.

### no_default_lapse_simplicity_search — pruned (experiment1 end of experiment)

**Outcome:** 90.2 nats behind costly_squares_final_digit_concepts (5.1× dse)

**Hypothesis:** Refinement of `simplicity_cost_concept_search`: people still search for a concept built from simple primitives (a fitted cost per extra primitive, the size principle, near-all-or-none commitment to the best concept), and on a fraction of trials they lapse and answer without consulting the concept. The one change is what a lapse produces: instead of a coin-flip yes/no, a lapsing person falls back on a learned default answer — a fitted, typically low, base rate of saying yes (a "when in doubt, say no" bias) — so parity-violating and merely nearby targets get fewer stray yeses and repeated answers to the same item are more consistent than the coin-flip lapse predicts.

### interval_breadth_biased_commitment — pruned (experiment1 end of experiment)

**Outcome:** 105.5 nats behind costly_squares_final_digit_concepts (5.6× dse)

**Hypothesis:** Refinement of `breadth_biased_rule_commitment`: people commit nearly all-or-none to the concept (a mathematical rule or a numerical interval) that best explains the examples under the size principle, with broad rules a priori more natural in proportion to their size. The one change is that the same size-graded naturalness applies to numerical ranges: instead of all intervals being equally likely a priori, an interval's prior grows (or shrinks) with its width by a fitted breadth exponent, so if people favour broad ranges, tight "nearby number" intervals rarely win and people say yes to non-example numbers in the examples' own decade less than a flat interval prior predicts.

### multiplicative_not_exponential_concepts — pruned (experiment1 end of experiment)

**Outcome:** 29.4 nats behind costly_squares_final_digit_concepts (3.2× dse)

**Hypothesis:** People's concept space for numbers is built on multiplication, not exponentiation: they readily entertain parity, "multiples of m", final digits, digits contained and numerical ranges, but concepts that require repeated multiplication — powers of a base, perfect squares, perfect cubes — are a priori much less accessible, carrying a fitted extra cost on top of the ordinary per-primitive simplicity cost, size principle and near-all-or-none commitment. So after 32, 4, 16 or 4, 64 people generalise to other multiples of 4 (12, 72) rather than restricting themselves to powers of 2 or 4, extending shared-divisor rules more than a learner who finds powers as natural as multiples.

### additive_cluster_exemplar_similarity — pruned (experiment1 end of experiment)

**Outcome:** 263.6 nats behind costly_squares_final_digit_concepts (9.5× dse)

**Hypothesis:** People do not infer a hidden concept; they compare the target to each stored example, one at a time, and judge how similar the two numbers are by the stored number categories they have in common (odd/even, multiples of m, squares, powers, last digit, digits contained, and stretches of the number line), with a shared small category counting far more than a shared broad one. They say yes the more the target resembles every example (a geometric-mean, all-examples-must-match comparison), so a target sharing a divisor with all the examples, or lying close to all of them, is accepted in graded fashion.

### stepwise_progression_induction — pruned (experiment1 end of experiment)

**Outcome:** 331.2 nats behind costly_squares_final_digit_concepts (9.8× dse)

**Hypothesis:** People assume the hidden program is a counting procedure: it starts at some number and steps upward by a fixed stride until it stops (e.g. "from 22 to 82 in steps of 4", "from 8 to 64 in steps of 8", or "every number from 40 to 60" for a stride of 1). They infer which start, stride and stop produced the examples — preferring small strides and, by the size principle, short progressions — and say yes to a target if the inferred progressions land on it. This predicts generalisation along shared step sizes (multiples of m, numbers spaced by the examples' common difference) more than fixed rule lists do, and less generalisation to immediate neighbours (1-3 away) that fall between the steps than to numbers a full stride away.

### divisibility_salient_simplicity_search — rejected (experiment1 round 3 candidate 3 refine incumbent erlang_interval_simplicity_search)

**Outcome:** predicts like existing model 'erlang_interval_simplicity_search' (p_yes RMSE 0.00086 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of erlang_interval_simplicity_search, not a new hypothesis.

**Hypothesis:** Refinement of `erlang_interval_simplicity_search`: people still build concepts from mental primitives (a fitted cost per extra primitive, the size principle, an Erlang size prior on intervals, near-all-or-none commitment to the best-explaining concept). The one change is that divisibility — "multiples of m" — is a privileged, highly accessible kind of arithmetic rule whose prior is shifted by its own fitted salience bonus, rather than costing exactly what every other two-primitive rule (powers, final digits, containing a digit, squares) costs. If divisibility is salient, a set whose examples all share a factor (e.g. 18, 45, 72 or 39, 60, 12, 45) is read as "multiples of m" more often than as a power, a final digit or a range, so people say yes to other multiples of m more than the incumbent predicts, as the critique reports.

### divisibility_salient_concept_search — rejected (experiment1 round 3 candidate 4 refine incumbent erlang_interval_simplicity_search)

**Outcome:** predicts like existing model 'erlang_interval_simplicity_search' (p_yes RMSE 0.00093 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of erlang_interval_simplicity_search, not a new hypothesis.

**Hypothesis:** Refinement of `erlang_interval_simplicity_search`: people still build concepts from mental primitives (a fitted cost per extra primitive, Erlang-weighted intervals, the size principle and near-all-or-none commitment to the best-explaining concept). The one change is that divisibility is a privileged, readily accessible kind of rule: "multiples of m" concepts receive a fitted prior bonus over other two-primitive rules (powers, ending in a digit, containing a digit), so when all the examples share a divisor people more often commit to a multiple-of-m rule and say yes to other multiples of it — closing the critique that people extend arithmetic rules more than the incumbent predicts.

### costly_power_rule_simplicity_search — pruned (experiment1 end of experiment)

**Outcome:** 32.1 nats behind costly_squares_final_digit_concepts (3.2× dse)

**Hypothesis:** Refinement of `simplicity_cost_concept_search`: people still build concepts from simple mental primitives, paying a fitted cost per extra primitive, weighing examples by the size principle and committing nearly all-or-none to the best-explaining concept. The one change is that rules built on repeated multiplication — squares, cubes, powers of a number and their exception variants — are harder to conceive than other rules of the same length and pay a fitted extra prior cost, so when examples such as 32, 4, 16 fit both "powers of 2" and "multiples of 4", people favour the multiplicative rule and extend it to new multiples (12, 72) far more than the size principle predicts — addressing the critique that people extend shared-divisor rules more than the best model does.

### notation_cheap_simplicity_search — pruned (experiment1 end of experiment)

**Outcome:** 59.4 nats behind costly_squares_final_digit_concepts (4.2× dse)

**Hypothesis:** Refinement of `erlang_interval_simplicity_search`: people still build concepts from mental primitives (a fitted cost per extra primitive, the size principle, an Erlang size prior on intervals, near-all-or-none commitment to the best-explaining concept). The one change is that concepts read straight off the written numeral — "ends in 2", "contains a 7" — cost a different, fitted amount from arithmetic concepts that must be computed (primes, squares, cubes, powers, multiples of m), instead of every two-primitive rule costing the same. If digit-notation concepts are cheaper, people more readily read a set like 92, 82, 27, 42 as "ends in 2" with one stray, and reject numbers that only share a range with the examples or are merely odd/even neighbours of them.

### final_digit_salient_concept_search — pruned (experiment1 end of experiment)

**Outcome:** 40.7 nats behind costly_squares_final_digit_concepts (3.2× dse)

**Hypothesis:** Refinement of `erlang_interval_simplicity_search`: people still build concepts from mental primitives (a fitted cost per extra primitive, Erlang-weighted intervals, the size principle and near-all-or-none commitment to the best-explaining concept). The one change is that the final digit of a written number is a privileged, perceptually salient feature: rules of the form "ends in d" get a fitted prior discount compared with other compound rules (multiples, powers, primes, contains-a-digit). So after 62, 32, 22 or 92, 82, 27, 42 people commit to "ends in 2" instead of "even numbers" or "contains a 2", and say no to other even numbers and to the rest of the 20s.

### contains_digit_costly_concepts — pruned (experiment1 end of experiment)

**Outcome:** 15.7 nats behind costly_squares_final_digit_concepts (2.2× dse)

**Hypothesis:** Refinement of `multiplicative_not_exponential_concepts`: people still build number concepts from simple primitives (a fitted cost per extra primitive, the size principle, Erlang-weighted intervals, near-all-or-none commitment to the best-explaining concept), with an extra cost on powers, squares and cubes. The one change is that "contains the digit d anywhere" is a hard, rarely entertained concept, because it ignores where the digit is written. It pays its own fitted prior cost, so people seldom commit to it. The sharpest disagreement with the best model is on sets whose examples all contain a digit d (e.g. 92, 82, 27, 42). There, people say no to targets that contain d without sharing an example's last digit (e.g. 23, 29, 12), while the best model predicts yes.

### hidden_digit_costly_concepts — rejected (experiment1 round 4 candidate 1 lens 2)

**Outcome:** predicts like existing model 'contains_digit_costly_concepts' (p_yes RMSE 0.00000 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of contains_digit_costly_concepts, not a new hypothesis.

**Hypothesis:** People's number concepts are built from multiplication (parity, multiples of m, final digits, ranges, with powers/squares/cubes costing extra), but a digit buried anywhere inside a numeral is hard to notice as the basis of a rule: concepts of the form "contains the digit d" carry a fitted extra prior cost over other rules of the same length. So when all examples happen to contain a 2 (e.g. 92, 82, 27, 42), people read them as "ends in 2" with a stray, or as a range, rather than "contains a 2", and say no to targets such as 23 or 25 that share a contained digit but not a final digit.

### place_value_digit_search_cost — rejected (experiment1 round 4 candidate 2 lens 3)

**Outcome:** predicts like existing model 'contains_digit_costly_concepts' (p_yes RMSE 0.00000 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of contains_digit_costly_concepts, not a new hypothesis.

**Hypothesis:** People read a written numeral by its places (a tens digit and a ones digit), so a concept that requires scanning for a digit regardless of its position — "contains a 2", "has an 8 in it" — is harder to conceive than other rules of the same length and pays a fitted extra prior cost, on top of the per-primitive simplicity cost, the extra cost of power/square/cube concepts, the size principle and near-all-or-none commitment. So when every example happens to contain some digit d, people extend that "contains d" rule to new numbers (e.g. 27 after 92, 82, 42) less than a learner who finds digit search as natural as any other rule, falling back on "ends in d", ranges or arithmetic rules instead.

### final_digit_multiplicative_concepts — pruned (experiment1 end of experiment)

**Outcome:** 8.0 nats behind costly_squares_final_digit_concepts (2.3× dse)

**Hypothesis:** Refinement of `multiplicative_not_exponential_concepts`: people still build number concepts from mental primitives. They pay a fitted cost per extra primitive and a fitted extra cost for concepts built on repeated multiplication (powers, squares, cubes), they weigh examples by the size principle, and they commit nearly all-or-none to the concept that best explains the examples. The one change, taken from `final_digit_salient_concept_search`, is that a written number's final digit is perceptually privileged: "ends in d" rules get a fitted prior discount over other compound rules, including "contains the digit d". So when every example contains some digit, people read the set as "ends in d" where they can, and say no to targets that only contain that digit somewhere else. This addresses the critique that people use "contains digit d" less than the incumbent predicts.

### salient_ending_multiplicative_concepts — rejected (experiment1 round 4 candidate 4 refine incumbent multiplicative_not_exponential_concepts)

**Outcome:** predicts like existing model 'final_digit_multiplicative_concepts' (p_yes RMSE 0.00000 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of final_digit_multiplicative_concepts, not a new hypothesis.

**Hypothesis:** Refinement of `multiplicative_not_exponential_concepts`: people still build number concepts from mental primitives on a multiplicative, not exponential, basis (a fitted cost per extra primitive, a fitted extra cost on powers, squares and cubes, Erlang-weighted intervals, the size principle and near-all-or-none commitment to the best-explaining concept). The one change, taken from `final_digit_salient_concept_search`, is that a number's final digit is a privileged, perceptually salient feature: "ends in d" rules get a fitted prior discount relative to other compound rules, including "contains the digit d". So when the examples share a digit, people read it as a shared ending (allowing a stray) rather than "contains d", and they say no to targets that contain that digit somewhere other than the final position, fixing the incumbent's over-prediction of yes for contained-digit targets with other endings.

### final_digit_salient_no_exponent_rules — rejected (experiment1 round 4 candidate 5 refine chosen)

**Outcome:** predicts like existing model 'final_digit_multiplicative_concepts' (p_yes RMSE 0.00042 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of final_digit_multiplicative_concepts, not a new hypothesis.

**Hypothesis:** Refinement of `final_digit_salient_concept_search`: people still build concepts from mental primitives (a fitted cost per extra primitive, Erlang-weighted intervals, the size principle, near-all-or-none commitment) and treat the final digit as a privileged feature, so "ends in d" rules get a fitted prior discount. The one change, grafted from `multiplicative_not_exponential_concepts`, is that rules built on repeated multiplication (squares, cubes, powers and their exception variants) pay a fitted extra prior cost. The claim is that people find final-digit rules easy and exponent rules hard. After 32, 4, 16 they extend "multiples of 4" to 12 and 72. After 62, 32, 22 they commit to "ends in 2" and do not fall back on "contains a 2", which is the critique's over-predicted contains-digit generalisation.

### ascending_range_sampling_concepts — pruned (experiment1 end of experiment)

**Outcome:** 28.3 nats behind costly_squares_final_digit_concepts (3.0× dse)

**Hypothesis:** People think of a range-type program as counting upward from where its range starts, so they expect the examples they have seen to be bunched toward the bottom of the range, with the rest of the range lying above them. A range that starts just below the examples and runs well above them therefore explains the examples better than one centred on them, and people generalise more to numbers a little above the examples than to numbers equally far below. Rule-type concepts (parity, multiples, final digits, with powers, squares and cubes costing extra) are judged as before.

### forward_counting_range_sampling — pruned (experiment1 end of experiment)

**Outcome:** 28.8 nats behind costly_squares_final_digit_concepts (3.0× dse)

**Hypothesis:** People assume the hidden program produces a numerical range by counting upward from its starting point, so the examples they see are taken to be early, low members of the range rather than evenly spread across it; a range that starts near the examples and extends above them therefore explains the examples better than one extending below them, with the strength (and direction) of this counting-forward assumption fitted from data. Everything else follows the best model (primitive-built rules with a per-primitive cost and an extra cost on powers/squares/cubes, the size principle, Erlang-weighted ranges, near-all-or-none commitment), so people generalise from a set to numbers just above the examples more than to numbers just below them.

### tenenbaum_rules_similarity — pruned (experiment1 end of experiment)

**Outcome:** 357.8 nats behind costly_squares_final_digit_concepts (7.4× dse)

**Hypothesis:** Bayesian concept learning over 33 mathematical rules (odd, even, squares, cubes, primes, multiples of 3-12, powers of 2-10, ending in 1-9) and every interval of 1-100. The rules share a prior fraction lambda equally; the intervals share the rest by an Erlang density on their size (scale sigma). Examples are sampled from the concept (the size principle), 10% of them uniformly from 1-100, and "yes" is the posterior probability that the concept contains the target.

### bigelow_piantadosi_mixture — pruned (experiment1 end of experiment)

**Outcome:** 302.0 nats behind costly_squares_final_digit_concepts (7.0× dse)

**Hypothesis:** Bayesian concept learning over 45 rules (odd, even, primes, squares, cubes, powers of 2-10, powers of 2 without 32 or with 37, multiples of 3-9, 11 and 12, ending in 0-9, containing the digit 0-9) and every interval of 1-100. A rule is chosen with probability lambda_0, and the rules are equally likely; each interval has prior (1 - lambda_0) / 100^2. Examples are sampled from the concept (the size principle), 10% of them uniformly from 1-100, and "yes" is the posterior probability that the concept contains the target.

### session_drifting_yes_bias — rejected (experiment2 round 0 candidate 1 lens 5)

**Outcome:** model binds ['trial_index'] — response-row bookkeeping a bare stimulus never carries — so it cannot be evaluated on a stimulus pool (the novelty gate now, the held-out evaluation later); compute everything from number_set and target.

**Hypothesis:** Refinement of `costly_squares_final_digit_concepts`: people infer the hidden concept exactly as in that model (primitive-built rules with a per-primitive cost, a final-digit discount, extra costs on powers/cubes and on squares, Erlang-weighted intervals, the size principle), but their willingness to turn that belief into a "yes" is not constant: they become steadily more reluctant to say yes as the session goes on, as if growing skeptical, so their response bias shifts against "yes" at a fitted rate per trial. Early in the session people answer yes more readily than their belief alone implies, and late in the session less readily, even for the same set and target.

### session_fatigue_default_no_concepts — rejected (experiment2 round 0 candidate 3 refine incumbent costly_squares_final_digit_concepts)

**Outcome:** model binds ['trial_index'] — response-row bookkeeping a bare stimulus never carries — so it cannot be evaluated on a stimulus pool (the novelty gate now, the held-out evaluation later); compute everything from number_set and target.

**Hypothesis:** Refinement of `costly_squares_final_digit_concepts`: people still build number concepts from mental primitives (a fitted cost per extra primitive, a discount for "ends in d" rules, separate extra costs for powers/cubes and for squares, Erlang-weighted intervals, the size principle and near-all-or-none commitment to the best-explaining concept). The one change is a fatigue-driven "when in doubt, say no" default that grows over the session: on a fraction of trials that rises (at a fitted rate) with how far into the session the trial is, people stop consulting their concept and simply answer no. So late in the session people reject even targets their concept contains (including the examples themselves), and the yes rate falls with trial position beyond what the stimuli explain.

### session_drifting_default_concepts — rejected (experiment2 round 0 candidate 4 refine incumbent costly_squares_final_digit_concepts)

**Outcome:** model binds ['trial_index'] — response-row bookkeeping a bare stimulus never carries — so it cannot be evaluated on a stimulus pool (the novelty gate now, the held-out evaluation later); compute everything from number_set and target.

**Hypothesis:** Refinement of `costly_squares_final_digit_concepts`: people still build number concepts from mental primitives (per-primitive cost, "ends in d" discount, separate costs for powers/cubes and for squares, Erlang-weighted intervals, size principle, near-all-or-none commitment) and on a fraction of trials answer without consulting the concept. The one change is what those inattentive answers are: instead of a fixed coin flip, the default answer a disengaged person gives drifts with their position in the session, at a fitted rate, so that as fatigue sets in their fallback becomes "no". This predicts that the same (set, target) is answered yes less often late in the session than early, beyond what the stimulus order implies, most visibly for targets the concept leaves uncertain or rejects only weakly.

### drifting_default_answer_concepts — rejected (experiment2 round 0 candidate 5 refine chosen)

**Outcome:** model binds ['trial_index'] — response-row bookkeeping a bare stimulus never carries — so it cannot be evaluated on a stimulus pool (the novelty gate now, the held-out evaluation later); compute everything from number_set and target.

**Hypothesis:** Refinement of the pruned `final_digit_multiplicative_concepts` (concepts built from primitives with a per-primitive cost, an extra cost on powers/squares/cubes, a prior discount on "ends in d" rules, Erlang-weighted ranges, the size principle and near-all-or-none commitment). The one change is in the response stage. On a fitted fraction of trials people answer without consulting the concept they inferred, and the default answer they give then is not a coin flip: it drifts steadily from leaning "yes" early in the session toward leaning "no" late in it, at a fitted rate. This addresses the critique that people say yes less and less over the session, beyond what the order of stimuli implies. It predicts that targets the concept leaves uncertain or rejects get more stray yeses early and fewer late.

### default_no_lapse_costly_squares — rejected (experiment2 round 0 candidate 3 refine incumbent costly_squares_final_digit_concepts repair 1)

**Outcome:** predicts like existing model 'costly_squares_final_digit_concepts' (p_yes RMSE 0.00062 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of costly_squares_final_digit_concepts, not a new hypothesis.

**Hypothesis:** Refinement of `costly_squares_final_digit_concepts`, grafting the response stage of the pruned `no_default_lapse_simplicity_search`: people still build number concepts from mental primitives (a per-primitive cost, an "ends in d" discount, separate extra costs for powers/cubes and for squares, Erlang-weighted intervals, the size principle and near-all-or-none commitment to the best-explaining concept), and on a fitted fraction of trials they answer without consulting that concept. The one change is what those unconsidered answers are: not a coin flip but a learned default with a fitted base rate of "yes", expected to lean toward "no" ("when in doubt, say no"). So people sometimes reject even the examples themselves, while targets the concept excludes get fewer stray yeses than a coin-flip lapse predicts.

### uncertainty_triggered_guessing — dropped (experiment2 round 1)

**Outcome:** fold 3 too slow to fit: the fit of 'uncertainty_triggered_guessing' (target_accept 0.99) was still sampling after the 30-minute limit and was stopped. Every sampling run of a candidate's admission fit has a 30-minute limit. Make the model cheaper to evaluate and easier to sample: vectorise the likelihood over trials (no Python loops, pytensor scan or per-trial subgraphs), compute features once per unique sequence (in compute_features or prepare_observed, not in the graph), and drop or merge parameters the data barely constrain — a weakly identified posterior makes NUTS take maximal-length trajectories.

**Hypothesis:** People infer the hidden concept as the simple concept-search learner does, but they turn that belief into an answer by a two-route decision rule: when their belief makes the target clearly in or clearly out they answer from it, and the closer the belief is to a toss-up, the more likely they are to give up deliberating and simply guess. So answers to targets the concept clearly decides are near-deterministic (rare stray answers even to the examples themselves), while targets the learner is torn about are answered close to a coin flip rather than in proportion to the belief.

### prime_salient_hundred_chart_2 — rejected (experiment2 round 2 candidate 4 refine incumbent hundred_chart_row_ranges)

**Outcome:** predicts like existing model 'prime_salient_hundred_chart' (p_yes RMSE 0.00080 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of prime_salient_hundred_chart, not a new hypothesis.

**Hypothesis:** Refinement of `hundred_chart_row_ranges` (the incumbent). Everything in it is kept: primitive-built rules with a cost per extra primitive, the final-digit discount, the costs on powers and squares, the times-table-graded multiples costs, Erlang ranges with a bonus for whole hundred-chart rows, the size principle and near-all-or-none commitment. The one change, taken from `prime_salient_concepts`, is that "prime numbers" is a familiar school-math category kept ready to hand, so it gets a fitted prior bonus over other compound rules. After all-prime sets (e.g. 2, 47; 31, 17, 61) people therefore commit to "primes" and say yes to other primes more than to odd composite numbers, which the incumbent fails to reproduce.

### prime_salient_chart_rows — rejected (experiment2 round 2 candidate 5 refine chosen)

**Outcome:** predicts like existing model 'prime_salient_hundred_chart' (p_yes RMSE 0.00000 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of prime_salient_hundred_chart, not a new hypothesis.

**Hypothesis:** Refinement of `prime_salient_concepts`: people build number concepts from mental primitives (a fitted cost per extra primitive, a final-digit discount, extra costs for powers/cubes and for squares, times-table-graded multiples costs, Erlang-weighted ranges, the size principle, near-all-or-none commitment), and "prime numbers" is a familiar school-math category with a fitted prior bonus, so after all-prime sets such as 2, 47 or 67, 31, 17, 61 people say yes to other primes and no to odd composites. The one change, taken from `hundred_chart_row_ranges`, is how people picture numerical ranges: they lay 1-100 out as a hundred chart of ten rows, so ranges made of whole rows (1-50, 21-40, 81-90) get a fitted prior bonus, and when no rule (primes included) explains the examples they fall back on a broad round block of the chart rather than a tight range hugging the examples.
