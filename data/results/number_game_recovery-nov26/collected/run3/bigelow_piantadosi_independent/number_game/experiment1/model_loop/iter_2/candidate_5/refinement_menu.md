# Refinement menu

The models you may refine, other than the incumbent `simplicity_cost_concept_search`. Each entry is one mechanism: its hypothesis in full as its author stated it, how it stands, and where its source is. This is a menu, not a list of what is ruled out: a pruned model lost to the best model on the data it was scored on, by the stated margin, but its mechanism may be partly right — and a model that lost narrowly is the most promising target here. Read the source of the model you pick before you write anything.

## Live models other than the incumbent (in the set; best first)

### parity_salient_concept_learning — rank 1, 15.6 ± 6.7 nats behind the best (2.3× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -914.8

**Hypothesis:** Refinement of `concept_posterior_with_lapse_guessing`: people do Bayesian concept learning over the same 45 rules and all intervals of 1-100 (size principle, lapse-and-guess response stage), but parity — odd versus even — is a privileged, highly accessible concept whose prior is boosted above the other rules by a fitted salience factor. The one change is this non-uniform rule prior favouring odd/even, which lets parity survive the size principle against tighter rules and intervals, so people say yes to parity-matching targets of an all-odd or all-even set far more than the uniform-prior learner predicts.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/d6b6238b22d937a5/repo/_runs/cell_1/experiment1/model_loop/models/parity_salient_concept_learning.py`

### breadth_biased_rule_commitment — rank 2, 20.3 ± 11.8 nats behind the best (1.7× dse: statistically tied with the best), ELPD-LOO -919.5

**Hypothesis:** Refinement of `sharpened_map_concept_selection`: people still commit, nearly all-or-none, to the one or few concepts (a mathematical rule or a numerical interval) that best explain the examples under the size principle, and answer yes only if the target is in that concept. The one change is the prior over rules: rather than treating all rules as equally likely, people find broad, familiar rules (odd, even, and other large categories) a priori more natural in proportion to how many numbers they cover, with the strength of this breadth bias fitted from data — so a broad rule like parity can win over tighter rules and intervals when all examples share it, predicting stronger parity-based generalization than the incumbent, as the critique reports.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/d6b6238b22d937a5/repo/_runs/cell_1/experiment1/model_loop/models/breadth_biased_rule_commitment.py`

### weakened_size_principle_sampling — rank 3, 40.7 ± 10.5 nats behind the best (3.9× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -939.9

**Hypothesis:** People do Bayesian concept learning over mathematical rules and numerical intervals, but they apply the size principle only partially: they discount the evidence that a small concept explains the examples better than a large one, as if the examples were only partly sampled from the concept rather than just happening to fall in it. This weakened size penalty keeps broad concepts such as "odd numbers", "even numbers" or a wide stretch of the number line alive, so people generalise along parity and across a region of numbers more than an ideal strong-sampling learner does.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/d6b6238b22d937a5/repo/_runs/cell_1/experiment1/model_loop/models/weakened_size_principle_sampling.py`

### sharpened_map_concept_selection — rank 4, 62.7 ± 15.0 nats behind the best (4.2× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -961.9

**Hypothesis:** People do not average over all concepts consistent with the examples; they commit, nearly all-or-none, to the one or few concepts (a mathematical rule or a tight numerical interval) that best explain the examples under the size principle, and answer yes only if the target falls in the concept they committed to. This sharpened, winner-take-most choice of concept makes answers more deterministic and leaves far less graded "nearby number" generalization than full Bayesian averaging predicts.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/d6b6238b22d937a5/repo/_runs/cell_1/experiment1/model_loop/models/sharpened_map_concept_selection.py`

### parity_restricted_range_commitment — rank 5, 67.2 ± 14.7 nats behind the best (4.6× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -966.4

**Hypothesis:** Refinement of `sharpened_map_concept_selection`: people still commit, nearly all-or-none, to the concept that best explains the examples under the size principle, but the concepts they can commit to include parity-restricted ranges ("even numbers between 16 and 64", "odd numbers from 31 to 47") alongside plain mathematical rules and plain intervals. The one change is adding these parity-by-interval conjunctions to the hypothesis space (with a fitted share of the interval prior), so that when all examples share parity people say yes to parity-matching targets inside the examples' range and no to opposite-parity near neighbours and to numbers outside the range.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/d6b6238b22d937a5/repo/_runs/cell_1/experiment1/model_loop/models/parity_restricted_range_commitment.py`

### concept_posterior_with_lapse_guessing — rank 6, 76.1 ± 16.6 nats behind the best (4.6× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -975.3

**Hypothesis:** Refinement of `bigelow_piantadosi_mixture`: people do Bayesian concept learning over the same 45 rules and all intervals of 1-100 (size principle, rule prior lambda_0), but they do not answer every trial from that posterior — on a fixed fraction of trials (a lapse) they ignore the examples and guess "yes" with a fixed base rate. The one change is this lapse-and-guess response stage, which makes answers to the same (set, target) less deterministic than the concept posterior alone implies and pulls confident predictions toward the guess rate.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/d6b6238b22d937a5/repo/_runs/cell_1/experiment1/model_loop/models/concept_posterior_with_lapse_guessing.py`

### tempered_posterior_commitment — rank 7, 85.1 ± 18.0 nats behind the best (4.7× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -984.3

**Hypothesis:** People hold a belief over candidate concepts (mathematical rules and numeric intervals, with the size principle) whose commitment differs from Bayesian averaging by a single tempering exponent: the posterior over concepts is raised to a free power and renormalised, so people either commit almost entirely to the best concept (exponent above 1) or hedge conservatively across many weaker ones (exponent below 1). It disagrees most sharply with the averaging model on sets where a rule and an interval compete, such as targets just outside the examples' range or numeric neighbours that fit no rule, because there the weight on the runner-up concepts (set by the exponent) decides the answer.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/d6b6238b22d937a5/repo/_runs/cell_1/experiment1/model_loop/models/tempered_posterior_commitment.py`

### overweighted_evidence_rules_intervals — rank 8, 115.8 ± 22.0 nats behind the best (5.3× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1015.0

**Hypothesis:** Refinement of `tenenbaum_rules_similarity`: people do Bayesian concept learning over the same mathematical rules and Erlang-weighted intervals, but they treat the examples as stronger evidence than ideal random sampling warrants — the size-principle likelihood is raised to a fitted power (an "evidence-weight" exponent), so the posterior concentrates on the tightest rules and intervals consistent with the examples. The one change is this fitted exponent (fixed at 1 in the original), which predicts less generalization to numerical neighbours and to numbers just outside the examples' range than the ideal learner, as the critique of the incumbent reports.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/d6b6238b22d937a5/repo/_runs/cell_1/experiment1/model_loop/models/overweighted_evidence_rules_intervals.py`

### range_bounded_rule_learning — rank 9, 144.7 ± 27.2 nats behind the best (5.3× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1043.9

**Hypothesis:** People never entertain an unbounded mathematical rule on its own: every concept they consider is a rule confined to a numerical range (e.g. "even numbers between 20 and 60", with "any number between 20 and 60" as the plain-interval case), and they infer which range-bounded rule produced the examples under the size principle, answering yes by how much of that inference contains the target. Because a rule bounded tightly around the examples is always a smaller, better explanation than the same rule across all of 1-100, this predicts strong use of shared properties like parity inside the examples' span but sharply less generalization of those properties to targets outside it.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/d6b6238b22d937a5/repo/_runs/cell_1/experiment1/model_loop/models/range_bounded_rule_learning.py`

### tversky_prototype_feature_contrast — rank 10, 169.6 ± 30.8 nats behind the best (5.5× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1068.8

**Hypothesis:** People judge membership by feature matching rather than by Bayesian inference over concepts: they abstract a prototype made of every property all the examples share (parity, divisibility, being a square/cube/prime/power, last digit, digits contained, decade, lying within the examples' range), and say "yes" the more of those shared properties the target has, penalised for each shared property it lacks and, more weakly, for each property it has that the prototype lacks (Tversky's contrast model of similarity). There is no size principle: a broad shared property counts as much as a narrow one.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/d6b6238b22d937a5/repo/_runs/cell_1/experiment1/model_loop/models/tversky_prototype_feature_contrast.py`

### stray_tolerant_concept_learning — rank 11, 179.5 ± 29.7 nats behind the best (6.0× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1078.7

**Hypothesis:** Refinement of bigelow_piantadosi_mixture: people do Bayesian concept learning over the same 45 rules and all intervals of 1-100 with the same grammar prior, and say yes with the posterior probability that the concept contains the target. The one change is that how often people think an example is a stray number unrelated to the concept is learned from the data rather than fixed at 10%; tolerating more strays lets a scattered set be read as a tight rule plus an outlier, so people generalise less to the examples' numerical neighbours than the incumbent predicts.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/d6b6238b22d937a5/repo/_runs/cell_1/experiment1/model_loop/models/stray_tolerant_concept_learning.py`

### bigelow_piantadosi_mixture — rank 12, 213.5 ± 33.1 nats behind the best (6.4× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1112.7

**Hypothesis:** Bayesian concept learning over 45 rules (odd, even, primes, squares, cubes, powers of 2-10, powers of 2 without 32 or with 37, multiples of 3-9, 11 and 12, ending in 0-9, containing the digit 0-9) and every interval of 1-100. A rule is chosen with probability lambda_0, and the rules are equally likely; each interval has prior (1 - lambda_0) / 100^2. Examples are sampled from the concept (the size principle), 10% of them uniformly from 1-100, and "yes" is the posterior probability that the concept contains the target.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/d6b6238b22d937a5/repo/_runs/cell_1/experiment1/model_loop/models/bigelow_piantadosi_mixture.py`

### tenenbaum_rules_similarity — rank 13, 269.3 ± 40.9 nats behind the best (6.6× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1168.5

**Hypothesis:** Bayesian concept learning over 33 mathematical rules (odd, even, squares, cubes, primes, multiples of 3-12, powers of 2-10, ending in 1-9) and every interval of 1-100. The rules share a prior fraction lambda equally; the intervals share the rest by an Erlang density on their size (scale sigma). Examples are sampled from the concept (the size principle), 10% of them uniformly from 1-100, and "yes" is the posterior probability that the concept contains the target.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/d6b6238b22d937a5/repo/_runs/cell_1/experiment1/model_loop/models/tenenbaum_rules_similarity.py`

## Pruned models (out of the set; narrowest margin first)

No model has been pruned yet in this project.
