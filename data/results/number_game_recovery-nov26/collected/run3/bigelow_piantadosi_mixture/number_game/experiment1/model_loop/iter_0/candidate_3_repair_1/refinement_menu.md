# Refinement menu

The models you may refine, other than the incumbent `bigelow_piantadosi_independent`. Each entry is one mechanism: its hypothesis in full as its author stated it, how it stands, and where its source is. This is a menu, not a list of what is ruled out: a pruned model lost to the best model on the data it was scored on, by the stated margin, but its mechanism may be partly right — and a model that lost narrowly is the most promising target here. Read the source of the model you pick before you write anything.

## Live models other than the incumbent (in the set; best first)

### tenenbaum_rules_similarity — rank 1, 311.6 ± 47.5 nats behind the best (6.6× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1151.9

**Hypothesis:** Bayesian concept learning over 33 mathematical rules (odd, even, squares, cubes, primes, multiples of 3-12, powers of 2-10, ending in 1-9) and every interval of 1-100. The rules share a prior fraction lambda equally; the intervals share the rest by an Erlang density on their size (scale sigma). Examples are sampled from the concept (the size principle), 10% of them uniformly from 1-100, and "yes" is the posterior probability that the concept contains the target.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/f8fb9bba6e6fa0a9/repo/_runs/cell_1/experiment1/model_loop/models/tenenbaum_rules_similarity.py`

### digit_exemplar_similarity — no comparison row

**Hypothesis:** People do not induce a rule at all; they judge a target by its summed similarity to the remembered examples, where numbers are represented by their two decimal digits: a target feels like it "belongs" to the extent that its tens digit is close to an example's tens digit and its ones digit matches an example's ones digit. Attention is split between the tens (magnitude) and ones (final-digit) dimensions, similarity decays exponentially with this digit-wise distance, and the probability of saying yes rises with the average similarity of the target to the examples.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/f8fb9bba6e6fa0a9/repo/_runs/cell_1/experiment1/model_loop/models/digit_exemplar_similarity.py`

### digit_overlap_exemplar_similarity — no comparison row

**Hypothesis:** People do not induce mathematical rules or magnitude intervals at all; they judge a target by its surface resemblance to the examples, treating each number as the set of decimal digits it is written with, and say "yes" in proportion to the average digit-overlap similarity between the target and the examples (an exemplar-similarity process over written digits). This disagrees most sharply with the rule-based Bayesian model on targets that share a digit with some examples but fit no rule or interval consistent with all of them (e.g. 12 after "31,19,49,100"), and on rule-consistent targets written with entirely different digits (e.g. 2 after "16,8,64"), where it predicts low endorsement.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/f8fb9bba6e6fa0a9/repo/_runs/cell_1/experiment1/model_loop/models/digit_overlap_exemplar_similarity.py`

## Pruned models (out of the set; narrowest margin first)

No model has been pruned yet in this project.
