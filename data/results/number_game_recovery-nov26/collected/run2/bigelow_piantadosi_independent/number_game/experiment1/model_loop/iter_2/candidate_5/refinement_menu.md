# Refinement menu

The models you may refine, other than the incumbent `rule_family_prior_tempered_mixture`. Each entry is one mechanism: its hypothesis in full as its author stated it, how it stands, and where its source is. This is a menu, not a list of what is ruled out: a pruned model lost to the best model on the data it was scored on, by the stated margin, but its mechanism may be partly right — and a model that lost narrowly is the most promising target here. Read the source of the model you pick before you write anything.

## Live models other than the incumbent (in the set; best first)

### tempered_learned_rule_prior — rank 1, 14.9 ± 19.1 nats behind the best (0.8× dse: statistically tied with the best), ELPD-LOO -903.0

**Hypothesis:** Refinement of `learned_rule_prior_mixture`: people learn concepts by the same Bayesian inference over 45 mathematical rules (each with its own learned prior plausibility) and all intervals of 1-100, but they apply the size principle only partially — they discount the evidence that the examples were sampled from a small concept, so the likelihood of the examples is tempered by a free sampling-strength exponent (1 = full strong sampling, 0 = weak sampling). The single change is grafting this tempered size principle (from `tempered_size_principle_mixture`) onto the learned-rule-prior model, predicting that broad favoured rules and wide intervals keep more posterior weight after a few examples than an ideal strong-sampling learner allows.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/fc3d1e4e3afb1c37/repo/_runs/cell_1/experiment1/model_loop/models/tempered_learned_rule_prior.py`

### recency_weighted_concept_learning — rank 2, 27.6 ± 22.4 nats behind the best (1.2× dse: statistically tied with the best), ELPD-LOO -915.7

**Hypothesis:** People infer the hidden concept by Bayesian inference over mathematical rules (each with its own prior plausibility) and numerical intervals, but they read the examples one at a time and their memory of earlier examples fades: each example's evidence counts less the further back in the sequence it was shown, so the most recent examples dominate which concept is inferred. This predicts that the same set shown in a different order yields different generalization, with targets that fit the last examples (but not the first ones) endorsed more than an ideal, order-blind learner would.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/fc3d1e4e3afb1c37/repo/_runs/cell_1/experiment1/model_loop/models/recency_weighted_concept_learning.py`

### learned_rule_prior_mixture — rank 3, 34.2 ± 28.5 nats behind the best (1.2× dse: statistically tied with the best), ELPD-LOO -922.3

**Hypothesis:** Refinement of bigelow_piantadosi_mixture: people learn concepts by the same Bayesian inference over 45 mathematical rules and all intervals of 1-100 with the size principle, but they do not hold the rules as equally likely a priori — each rule (e.g. odd, even, ends in a digit) carries its own prior plausibility, inferred from the data. The single change is replacing the uniform prior over rules with a learned, non-uniform one (the core idea of Bigelow & Piantadosi's prior inference), so favoured rules such as parity and shared final digits can dominate interval-based similarity when the examples support them.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/fc3d1e4e3afb1c37/repo/_runs/cell_1/experiment1/model_loop/models/learned_rule_prior_mixture.py`

### learned_rule_prior_erlang_intervals — rank 4, 38.8 ± 27.7 nats behind the best (1.4× dse: statistically tied with the best), ELPD-LOO -926.9

**Hypothesis:** Refinement of `learned_rule_prior_mixture`: people learn concepts by the same Bayesian inference over 45 mathematical rules (each with its own learned prior plausibility) and all intervals of 1-100 with the full size principle, but they do not consider every numerical range equally plausible a priori — they expect ranges of a typical, intermediate width (an Erlang prior on interval size with a fitted scale, as in Tenenbaum's number game), so very narrow and very broad ranges are discounted before any examples are seen. The single change is replacing the incumbent's uniform prior over intervals with this size-graded interval prior (taken from `tenenbaum_rules_similarity`), predicting that magnitude-based generalization from one or two examples extends to a characteristic distance around them rather than collapsing onto the tightest range that covers them.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/fc3d1e4e3afb1c37/repo/_runs/cell_1/experiment1/model_loop/models/learned_rule_prior_erlang_intervals.py`

### tempered_size_principle_mixture — rank 5, 140.3 ± 22.2 nats behind the best (6.3× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1028.5

**Hypothesis:** Refinement of `bigelow_piantadosi_mixture`: people do the same Bayesian concept learning over the same 45 rules and all intervals of 1-100, but they apply the size principle only partially — they discount the evidence that the examples were sampled from a small concept, so the likelihood of the examples is tempered by a free "sampling strength" exponent (1 = the incumbent's full strong sampling, 0 = weak sampling). With weaker sampling, broad rules such as odd/even and digit rules keep more posterior weight against tight intervals, predicting stronger parity generalization, more yeses to far targets and less pure numerical-proximity generalization.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/fc3d1e4e3afb1c37/repo/_runs/cell_1/experiment1/model_loop/models/tempered_size_principle_mixture.py`

### llo_distorted_concept_posterior — rank 6, 151.4 ± 21.6 nats behind the best (7.0× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1039.6

**Hypothesis:** People infer the hidden concept as an ideal Bayesian learner would — over the same 45 mathematical rules (equally likely a priori) and all intervals of 1-100, with the size principle — but they misread their own belief when turning it into a yes/no answer: the posterior probability that the target is in the concept is distorted linearly in log-odds (Zhang & Maloney-style probability weighting), compressed toward a reference point. As a result, near-certain members get fewer yeses and near-certain non-members more yeses than the ideal learner predicts, while the ordering of targets is left intact.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/fc3d1e4e3afb1c37/repo/_runs/cell_1/experiment1/model_loop/models/llo_distorted_concept_posterior.py`

### tempered_rules_similarity — rank 7, 188.1 ± 26.7 nats behind the best (7.0× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1076.2

**Hypothesis:** Refinement of `tenenbaum_rules_similarity`: people do Bayesian concept learning over the same mathematical rules and numerical intervals, but update conservatively — they under-weight the evidence of the examples relative to their prior (the example likelihood is raised to a fitted power below one, a weakened size principle). As a result broad concepts like "even" or "odd" and wide intervals survive a few examples better than for an ideal learner, producing more yeses to distant parity-matching targets and less sharply peaked generalization to numbers right next to the examples.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/fc3d1e4e3afb1c37/repo/_runs/cell_1/experiment1/model_loop/models/tempered_rules_similarity.py`

### featural_exemplar_similarity — rank 8, 191.5 ± 25.1 nats behind the best (7.6× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1079.6

**Hypothesis:** People do not induce a hidden rule; they judge a target by its featural similarity to the remembered examples (exemplar generalization over symbolic number features). A target is accepted in proportion to its average similarity to the examples, where similarity decays with the number of salient surface features it fails to share with each example — its parity, its final digit, and its decade — rather than with its numerical distance.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/fc3d1e4e3afb1c37/repo/_runs/cell_1/experiment1/model_loop/models/featural_exemplar_similarity.py`

### digit_parity_exemplar_similarity — rank 9, 200.3 ± 24.3 nats behind the best (8.2× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1088.4

**Hypothesis:** People do not induce a hidden rule at all; they judge a target by its exemplar similarity to the examples shown, where two numbers are similar to the extent they share surface features (same parity, same final digit) and are close in magnitude, and "yes" grows with the target's average similarity to the examples. This disagrees most sharply with the rule-based Bayesian best model on targets that match only some examples' final digit or share the examples' parity without fitting any whole rule (graded yeses where the rule model says no), and on near-in-magnitude targets that break the examples' parity/digit pattern (fewer yeses than interval hypotheses predict).

**Source:** `/scr/wurgaft/auto-psych/agent_trees/fc3d1e4e3afb1c37/repo/_runs/cell_1/experiment1/model_loop/models/digit_parity_exemplar_similarity.py`

### ones_digit_exemplar_similarity — rank 10, 224.4 ± 26.2 nats behind the best (8.6× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1112.6

**Hypothesis:** People judge a target by comparing its final (ones) digit with the final digit of each example, ignoring magnitude and mathematical rules: a target is endorsed in proportion to how many examples it resembles in surface form, where an example with the identical final digit counts fully and an example whose final digit merely has the same parity (odd/even) counts partially. This graded, example-by-example digit matching predicts yeses for targets that share a final digit with only some of the examples and for parity-matching targets far from every example.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/fc3d1e4e3afb1c37/repo/_runs/cell_1/experiment1/model_loop/models/ones_digit_exemplar_similarity.py`

### bigelow_piantadosi_mixture — rank 11, 374.1 ± 46.2 nats behind the best (8.1× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1262.2

**Hypothesis:** Bayesian concept learning over 45 rules (odd, even, primes, squares, cubes, powers of 2-10, powers of 2 without 32 or with 37, multiples of 3-9, 11 and 12, ending in 0-9, containing the digit 0-9) and every interval of 1-100. A rule is chosen with probability lambda_0, and the rules are equally likely; each interval has prior (1 - lambda_0) / 100^2. Examples are sampled from the concept (the size principle), 10% of them uniformly from 1-100, and "yes" is the posterior probability that the concept contains the target.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/fc3d1e4e3afb1c37/repo/_runs/cell_1/experiment1/model_loop/models/bigelow_piantadosi_mixture.py`

### tenenbaum_rules_similarity — rank 12, 516.3 ± 59.2 nats behind the best (8.7× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1404.5

**Hypothesis:** Bayesian concept learning over 33 mathematical rules (odd, even, squares, cubes, primes, multiples of 3-12, powers of 2-10, ending in 1-9) and every interval of 1-100. The rules share a prior fraction lambda equally; the intervals share the rest by an Erlang density on their size (scale sigma). Examples are sampled from the concept (the size principle), 10% of them uniformly from 1-100, and "yes" is the posterior probability that the concept contains the target.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/fc3d1e4e3afb1c37/repo/_runs/cell_1/experiment1/model_loop/models/tenenbaum_rules_similarity.py`

### interval_only_size_principle — rank 13, 640.2 ± 38.2 nats behind the best (16.8× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1528.4

**Hypothesis:** People generalize from the examples purely by numerical magnitude, ignoring every arithmetic or digit rule: they infer which contiguous range of 1-100 the program draws from, favouring tighter ranges that still cover all the examples (the size principle) under a prior preference for ranges of a typical width, and say "yes" with the probability that the target lies inside the inferred range. This predicts near-certain yeses between the smallest and largest example, a smooth fall-off just outside them that sharpens as more examples are shown, and no special boost for targets that share the examples' parity, final digit or multiplicative structure.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/fc3d1e4e3afb1c37/repo/_runs/cell_1/experiment1/model_loop/models/interval_only_size_principle.py`

## Pruned models (out of the set; narrowest margin first)

No model has been pruned yet in this project.
