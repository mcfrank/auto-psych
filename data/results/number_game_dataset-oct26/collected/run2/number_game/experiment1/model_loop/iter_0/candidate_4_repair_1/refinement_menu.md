# Refinement menu

The models you may refine, other than the incumbent `bigelow_piantadosi_independent`. Each entry is one mechanism: its hypothesis in full as its author stated it, how it stands, and where its source is. This is a menu, not a list of what is ruled out: a pruned model lost to the best model on the data it was scored on, by the stated margin, but its mechanism may be partly right — and a model that lost narrowly is the most promising target here. Read the source of the model you pick before you write anything.

## Live models other than the incumbent (in the set; best first)

### bigelow_piantadosi_interval — rank 1, 10.0 ± 19.4 nats behind the best (0.5× dse: statistically tied with the best), ELPD-LOO -1802.1

**Hypothesis:** Bayesian concept learning over 42 rules (powers of 2-10, multiples of 3-12, final digit 1-9, contained digit 1-9, primes, even, odd, squares, cubes) and every interval of 1-100, with a prior weight inferred for each rule separately and a share p_mix for the rules as a group. An interval's prior follows a Beta density over its midpoint, with both shapes inferred, so concepts in some parts of 1-100 are favoured. Examples are sampled from the concept (the size principle), 10% of them uniformly from 1-100, and "yes" is the posterior probability that the concept contains the target.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/7e8e4116c4d40231/repo/_runs/cell_1/experiment1/model_loop/models/bigelow_piantadosi_interval.py`

### tenenbaum_rules_similarity — rank 2, 262.6 ± 63.9 nats behind the best (4.1× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -2054.7

**Hypothesis:** Bayesian concept learning over 33 mathematical rules (odd, even, squares, cubes, primes, multiples of 3-12, powers of 2-10, ending in 1-9) and every interval of 1-100. The rules share a prior fraction lambda equally; the intervals share the rest by an Erlang density on their size (scale sigma). Examples are sampled from the concept (the size principle), 10% of them uniformly from 1-100, and "yes" is the posterior probability that the concept contains the target.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/7e8e4116c4d40231/repo/_runs/cell_1/experiment1/model_loop/models/tenenbaum_rules_similarity.py`

### recency_weighted_exemplar_similarity — no comparison row

**Hypothesis:** People do not infer a hidden rule; they judge a target by its similarity to the remembered examples, as in an exemplar model. Similarity to each example decays exponentially with the psychological distance between the two numbers, a weighted sum of mismatches on how far apart they are, whether they share a final digit, whether they share parity, and whether one divides the other. Memory for the examples fades with how long ago each was shown, so the last-shown example counts most, and the probability of "yes" rises with the target's recency-weighted summed similarity to the examples.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/7e8e4116c4d40231/repo/_runs/cell_1/experiment1/model_loop/models/recency_weighted_exemplar_similarity.py`

### recency_forgetting_concept_learning — no comparison row

**Hypothesis:** People are Bayesian concept learners (rules and intervals, size principle) who reason only from the examples they still hold in memory: the last-shown example is always remembered, and each earlier example is forgotten with a probability that grows the further back it was shown, so the concept is inferred from a recency-biased subset of the set. This disagrees most with the order-invariant best model on multi-example sets whose early examples point to a different concept than the last one: for targets close to the last example but outside the concept implied by the whole set it predicts many more "yes" answers, and for targets near only the first example, fewer.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/7e8e4116c4d40231/repo/_runs/cell_1/experiment1/model_loop/models/recency_forgetting_concept_learning.py`

### recency_decayed_concept_learning — no comparison row

**Hypothesis:** People infer the hidden program by Bayesian concept learning over mathematical rules and numeric intervals, but their memory of the examples decays with recency: the most recently shown example counts as full evidence while each earlier example is discounted by a constant factor per position back. Concepts consistent with the later examples therefore dominate, so targets near or sharing structure with the last-shown example get more "yes" answers than those tied to the first-shown one, and evidence from long sets is weaker (broader generalization).

**Source:** `/scr/wurgaft/auto-psych/agent_trees/7e8e4116c4d40231/repo/_runs/cell_1/experiment1/model_loop/models/recency_decayed_concept_learning.py`

## Pruned models (out of the set; narrowest margin first)

No model has been pruned yet in this project.
