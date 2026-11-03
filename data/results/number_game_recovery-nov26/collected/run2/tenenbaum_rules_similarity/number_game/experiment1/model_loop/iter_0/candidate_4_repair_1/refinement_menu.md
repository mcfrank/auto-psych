# Refinement menu

The models you may refine, other than the incumbent `bigelow_piantadosi_independent`. Each entry is one mechanism: its hypothesis in full as its author stated it, how it stands, and where its source is. This is a menu, not a list of what is ruled out: a pruned model lost to the best model on the data it was scored on, by the stated margin, but its mechanism may be partly right — and a model that lost narrowly is the most promising target here. Read the source of the model you pick before you write anything.

## Live models other than the incumbent (in the set; best first)

### bigelow_piantadosi_mixture — rank 1, 112.8 ± 16.5 nats behind the best (6.8× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1327.0

**Hypothesis:** Bayesian concept learning over 45 rules (odd, even, primes, squares, cubes, powers of 2-10, powers of 2 without 32 or with 37, multiples of 3-9, 11 and 12, ending in 0-9, containing the digit 0-9) and every interval of 1-100. A rule is chosen with probability lambda_0, and the rules are equally likely; each interval has prior (1 - lambda_0) / 100^2. Examples are sampled from the concept (the size principle), 10% of them uniformly from 1-100, and "yes" is the posterior probability that the concept contains the target.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/962b9e6d929c436d/repo/_runs/cell_1/experiment1/model_loop/models/bigelow_piantadosi_mixture.py`

### exemplar_attention_similarity — no comparison row

**Hypothesis:** People do not infer a rule; they judge the target by its summed similarity to the stored examples (exemplar-based generalization, as in the Generalized Context Model). Each number is represented on a few psychological dimensions — magnitude, parity, final digit and tens digit — each with its own attention weight, similarity to an example decays exponentially with attention-weighted mismatch, and the probability of "yes" rises with the target's average similarity to the examples.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/962b9e6d929c436d/repo/_runs/cell_1/experiment1/model_loop/models/exemplar_attention_similarity.py`

### non_accumulating_size_principle — no comparison row

**Hypothesis:** People do Bayesian concept learning over the same rules and intervals, but they do not accumulate the size-principle evidence across examples: they judge how well a concept fits the examples by the average fit of one example, sharpened by a fixed amount, so their generalisation is no tighter after four examples than after one. This disagrees most with the current best model on sets with one or two examples, where it predicts sharper, more rule-like generalisation, and on four-example sets, where it predicts broader, more graded generalisation that keeps giving partial "yes" to targets that fit only a looser concept (for example, odd numbers that are not in a narrow rule the examples happen to share).

**Source:** `/scr/wurgaft/auto-psych/agent_trees/962b9e6d929c436d/repo/_runs/cell_1/experiment1/model_loop/models/non_accumulating_size_principle.py`

### exemplar_feature_similarity — no comparison row

**Hypothesis:** People do not infer a rule at all; they judge a target by its summed similarity to the remembered examples (exemplar generalization), where similarity between two numbers falls off exponentially with how many simple arithmetic properties they disagree on (parity, divisibility by 3-10, last digit, tens digit, being prime, square or a power of two) plus how far apart they are in magnitude. A target that closely resembles even one example is accepted, with no size-principle penalty and no requirement that it fit all examples jointly.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/962b9e6d929c436d/repo/_runs/cell_1/experiment1/model_loop/models/exemplar_feature_similarity.py`

### bayesian_concepts_with_guessing_lapse — no comparison row

**Hypothesis:** A refinement of bigelow_piantadosi_independent: people still judge by Bayesian concept learning over the same 45 rules and all intervals of 1-100 (independently weighted rule prior, size principle), but on a fraction of trials they do not consult that posterior at all and instead give a guess, saying "yes" with their own fixed guessing rate. The one change is this response-lapse stage, which explains why even targets that are among the examples are often rejected and why clearly excluded targets are sometimes accepted.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/962b9e6d929c436d/repo/_runs/cell_1/experiment1/model_loop/models/bayesian_concepts_with_guessing_lapse.py`

## Pruned models (out of the set; narrowest margin first)

No model has been pruned yet in this project.
