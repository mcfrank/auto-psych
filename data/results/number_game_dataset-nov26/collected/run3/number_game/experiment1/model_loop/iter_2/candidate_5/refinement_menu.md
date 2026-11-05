# Refinement menu

The models you may refine, other than the incumbent `tempered_size_principle_independent`. Each entry is one mechanism: its hypothesis in full as its author stated it, how it stands, and where its source is. This is a menu, not a list of what is ruled out: a pruned model lost to the best model on the data it was scored on, by the stated margin, but its mechanism may be partly right — and a model that lost narrowly is the most promising target here. Read the source of the model you pick before you write anything.

## Live models other than the incumbent (in the set; best first)

### sampled_hypothesis_majority_vote — rank 1, 3.0 ± 4.9 nats behind the best (0.6× dse: statistically tied with the best), ELPD-LOO -1602.7

**Hypothesis:** Refinement of tempered_size_principle_independent: people do the same tempered Bayesian concept learning over the 45 rules and all intervals, but they cannot hold the whole posterior in mind — they bring a small number of candidate concepts to mind by sampling them from their posterior and answer "yes" if the majority of those sampled concepts contain the target. The one change is this limited-sample majority vote (with an inferred number of samples; one sample recovers the original model), which makes answers more all-or-none than the posterior itself: likely members are endorsed more consistently and weakly supported targets, such as numbers beyond the range of the examples, are endorsed less.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/8a52ff3d6e8e6ef7/repo/_runs/cell_1/experiment1/model_loop/models/sampled_hypothesis_majority_vote.py`

### compact_interval_tempered_bayes — rank 2, 9.7 ± 4.8 nats behind the best (2.0× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1609.4

**Hypothesis:** Refinement of `tempered_size_principle_independent`: people do the same tempered-evidence Bayesian concept learning over the 45 rules (each with its own prior weight) and all intervals of 1-100, but they do not consider all intervals equally plausible a priori — they prefer compact magnitude ranges, with an Erlang prior over interval size whose scale is inferred (the one change, taken from `tenenbaum_rules_similarity`). Because the tempered evidence no longer spreads interval belief onto wide ranges, people confine magnitude-based generalization close to the [min, max] range of the examples and endorse targets outside it less than the incumbent predicts.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/8a52ff3d6e8e6ef7/repo/_runs/cell_1/experiment1/model_loop/models/compact_interval_tempered_bayes.py`

### size_penalized_interval_prior — rank 3, 15.9 ± 4.9 nats behind the best (3.3× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1615.6

**Hypothesis:** Refinement of `tempered_size_principle_independent`: people do the same tempered Bayesian concept learning over 45 individually weighted rules and all intervals of 1-100, but a priori they expect a magnitude-based concept to be a narrow range rather than treating all intervals as equally likely — their prior on an interval falls off as a power of its width, with the steepness inferred. The one change is this size-penalizing interval prior (steepness 0 recovers a uniform interval prior); it predicts that people confine generalization more tightly to the range spanned by the examples, and endorse targets outside that range less, than the incumbent allows.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/8a52ff3d6e8e6ef7/repo/_runs/cell_1/experiment1/model_loop/models/size_penalized_interval_prior.py`

### conservative_evidence_concept_learning — rank 4, 36.9 ± 26.6 nats behind the best (1.4× dse: statistically tied with the best), ELPD-LOO -1636.6

**Hypothesis:** Refinement of `bigelow_piantadosi_mixture`: people do the same Bayesian concept learning over the 45 rules and all intervals of 1-100 with the size principle, but they under-weight the evidence of the examples, as if each example counted for only a fraction of a full observation (a conservative, tempered likelihood). The one change is a free evidence-weight exponent on the likelihood of the examples, so confidence in the smallest consistent rule grows more slowly with examples and more posterior stays on broader rules and intervals — making learners less sure of the winning rule than the ideal learner.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/8a52ff3d6e8e6ef7/repo/_runs/cell_1/experiment1/model_loop/models/conservative_evidence_concept_learning.py`

### domain_specific_evidence_weight — rank 5, 43.3 ± 26.8 nats behind the best (1.6× dse: statistically tied with the best), ELPD-LOO -1642.9

**Hypothesis:** Refinement of `conservative_evidence_concept_learning`: people do the same Bayesian concept learning over the 45 equally weighted rules and all intervals of 1-100 with the size principle, but how strongly they weigh the examples as evidence depends on the kind of concept — they are conservative about mathematical rules yet apply the size principle sharply to magnitude ranges, so once they think "a range of numbers" they confine it tightly to the span of the examples. The one change is splitting the single evidence-weight exponent into one for rule hypotheses and one for interval hypotheses; it predicts less endorsement of targets just outside the examples' [min, max] range (relative to inside it) than a single shared tempering allows, while keeping graded endorsement of rule members.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/8a52ff3d6e8e6ef7/repo/_runs/cell_1/experiment1/model_loop/models/domain_specific_evidence_weight.py`

### log_odds_belief_readout — rank 6, 49.6 ± 25.9 nats behind the best (1.9× dse: statistically tied with the best), ELPD-LOO -1649.3

**Hypothesis:** Refinement of bigelow_piantadosi_mixture: people do the same Bayesian concept learning over the 45 rules and all intervals of 1-100 with the size principle, but they do not answer "yes" with probability equal to their belief that the concept contains the target; instead they read out that belief through a distorted, linear-in-log-odds probability weighting (as in judgments of probability and decisions under risk), which compresses confident beliefs toward the middle and shifts them by a response bias. The one change is this readout: confident beliefs that a minimal-rule member is in the concept are endorsed less than certainly, and confident beliefs that a number is out are rejected less than certainly.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/8a52ff3d6e8e6ef7/repo/_runs/cell_1/experiment1/model_loop/models/log_odds_belief_readout.py`

### feature_overlap_similarity — rank 7, 64.0 ± 30.2 nats behind the best (2.1× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1663.7

**Hypothesis:** People do not infer a single hidden rule; instead they judge the target by its featural similarity to the examples they were shown. Each number is represented by its properties (parity, primality, being a square/cube/power, divisibility, final digit, digits it contains, and which decade it lies in), and the more properties the target shares with each example on average (feature overlap relative to the properties either has), the more likely people say yes — so endorsement is graded even among numbers that fit a rule, and partial matches such as a shared final digit or nearby magnitude still earn some yeses.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/8a52ff3d6e8e6ef7/repo/_runs/cell_1/experiment1/model_loop/models/feature_overlap_similarity.py`

### featural_exemplar_similarity — rank 8, 67.1 ± 29.9 nats behind the best (2.2× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1666.8

**Hypothesis:** People do not pick a single best-fitting rule; they judge a target by its graded exemplar similarity to the examples they saw, where two numbers are similar to the extent that they are close on the number line and share mathematical/digit features (parity, last digit, multiples, powers, primality, digits contained), and "yes" rises with the target's average similarity to the examples. This disagrees most sharply with the rule-based Bayesian model on members of the winning rule that are far from or featurally unlike the examples (which it endorses fully and this model only partly, producing graded typicality), and on non-members that are numerically close to the examples or share their last digit (which it rejects and this model partially endorses).

**Source:** `/scr/wurgaft/auto-psych/agent_trees/8a52ff3d6e8e6ef7/repo/_runs/cell_1/experiment1/model_loop/models/featural_exemplar_similarity.py`

### bounded_range_rule_learning — rank 9, 69.0 ± 38.0 nats behind the best (1.8× dse: statistically tied with the best), ELPD-LOO -1668.7

**Hypothesis:** People believe the program produces numbers that follow a mathematical rule but only within a bounded range spanned by the examples: they do Bayesian concept learning over "rule restricted to a range" concepts (any of the 45 rules, or no rule at all, intersected with an interval that covers every example), weighting each by the size principle with an inferred, tempered evidence strength. Because a rule restricted to the examples' range is a smaller, better-fitting concept than the unbounded rule, people generalize mainly to rule members inside the [min, max] span of the examples and endorse rule members far outside that span much less.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/8a52ff3d6e8e6ef7/repo/_runs/cell_1/experiment1/model_loop/models/bounded_range_rule_learning.py`

### featural_contrast_similarity — rank 10, 80.7 ± 30.6 nats behind the best (2.6× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1680.4

**Hypothesis:** People do not pick a single rule; they judge a target by featural similarity to the examples, in the manner of Tversky's contrast model. Each number is represented by its properties (parity, primality, being a square/cube/power, divisibility, final digit, digits it contains, decade), and the more of the examples' shared properties the target also has (weighted by how distinctive each property is), and the fewer extra properties the examples lack, the more likely people are to say yes — so members of a rule are graded by how many other properties they share with the examples, and non-members that share e.g. a last digit still get partial endorsement.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/8a52ff3d6e8e6ef7/repo/_runs/cell_1/experiment1/model_loop/models/featural_contrast_similarity.py`

### outlier_tolerant_concept_learning — rank 11, 124.6 ± 16.9 nats behind the best (7.4× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1724.3

**Hypothesis:** People do normative Bayesian concept learning over the 45 mathematical rules (each with its own prior weight) and all intervals of 1-100, using the size principle, but they mis-estimate how the examples were generated: they believe a much larger share of the examples are random strays from 1-100 than the ideal learner assumes (an inferred noise belief instead of a fixed 10%). Because a stray-tolerant learner keeps believing in a tight concept (a small rule or a narrow interval around the examples) even when one example does not fit it, belief stays on tight hypotheses rather than spreading to broad rules and wide intervals. So members of the tight hypotheses keep graded endorsement, while far-away non-members, which only broad hypotheses cover, are rejected more firmly.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/8a52ff3d6e8e6ef7/repo/_runs/cell_1/experiment1/model_loop/models/outlier_tolerant_concept_learning.py`

### bigelow_piantadosi_mixture — rank 12, 203.6 ± 42.3 nats behind the best (4.8× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1803.3

**Hypothesis:** Bayesian concept learning over 45 rules (odd, even, primes, squares, cubes, powers of 2-10, powers of 2 without 32 or with 37, multiples of 3-9, 11 and 12, ending in 0-9, containing the digit 0-9) and every interval of 1-100. A rule is chosen with probability lambda_0, and the rules are equally likely; each interval has prior (1 - lambda_0) / 100^2. Examples are sampled from the concept (the size principle), 10% of them uniformly from 1-100, and "yes" is the posterior probability that the concept contains the target.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/8a52ff3d6e8e6ef7/repo/_runs/cell_1/experiment1/model_loop/models/bigelow_piantadosi_mixture.py`

### bigelow_piantadosi_independent — rank 13, 223.4 ± 35.9 nats behind the best (6.2× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1823.1

**Hypothesis:** Bayesian concept learning over 45 rules (odd, even, primes, squares, cubes, powers of 2-10, powers of 2 without 32 or with 37, multiples of 3-9, 11 and 12, ending in 0-9, containing the digit 0-9) and every interval of 1-100, with a prior weight inferred for each rule separately and a share lambda_0 for the rules as a group; each interval has prior (1 - lambda_0) / 100^2. Examples are sampled from the concept (the size principle), 10% of them uniformly from 1-100, and "yes" is the posterior probability that the concept contains the target.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/8a52ff3d6e8e6ef7/repo/_runs/cell_1/experiment1/model_loop/models/bigelow_piantadosi_independent.py`

### tenenbaum_rules_similarity — rank 14, 514.8 ± 85.7 nats behind the best (6.0× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -2114.5

**Hypothesis:** Bayesian concept learning over 33 mathematical rules (odd, even, squares, cubes, primes, multiples of 3-12, powers of 2-10, ending in 1-9) and every interval of 1-100. The rules share a prior fraction lambda equally; the intervals share the rest by an Erlang density on their size (scale sigma). Examples are sampled from the concept (the size principle), 10% of them uniformly from 1-100, and "yes" is the posterior probability that the concept contains the target.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/8a52ff3d6e8e6ef7/repo/_runs/cell_1/experiment1/model_loop/models/tenenbaum_rules_similarity.py`

## Pruned models (out of the set; narrowest margin first)

No model has been pruned yet in this project.
