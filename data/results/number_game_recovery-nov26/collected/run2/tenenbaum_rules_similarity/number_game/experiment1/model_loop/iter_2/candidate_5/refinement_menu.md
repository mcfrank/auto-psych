# Refinement menu

The models you may refine, other than the incumbent `inferred_example_noise_concepts`. Each entry is one mechanism: its hypothesis in full as its author stated it, how it stands, and where its source is. This is a menu, not a list of what is ruled out: a pruned model lost to the best model on the data it was scored on, by the stated margin, but its mechanism may be partly right — and a model that lost narrowly is the most promising target here. Read the source of the model you pick before you write anything.

## Live models other than the incumbent (in the set; best first)

### erlang_interval_size_prior — rank 1, 0.4 ± 2.3 nats behind the best (0.2× dse: statistically tied with the best), ELPD-LOO -1212.5

**Hypothesis:** A refinement of bigelow_piantadosi_independent: people do the same Bayesian concept learning over the 45 independently weighted rules and all intervals of 1-100 (size principle, 10% of examples random), but they do not treat every interval as equally likely a priori. Their prior over intervals favours ranges of a typical width (an Erlang prior on interval size, as in Tenenbaum's number game), with that preferred width inferred from the data, so they neither readily entertain tiny ranges hugging the examples nor ranges spanning most of 1-100; the one change is this size-dependent interval prior in place of the uniform one.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/962b9e6d929c436d/repo/_runs/cell_1/experiment1/model_loop/models/erlang_interval_size_prior.py`

### bigelow_piantadosi_independent — rank 2, 2.1 ± 1.4 nats behind the best (1.5× dse: statistically tied with the best), ELPD-LOO -1214.2

**Hypothesis:** Bayesian concept learning over 45 rules (odd, even, primes, squares, cubes, powers of 2-10, powers of 2 without 32 or with 37, multiples of 3-9, 11 and 12, ending in 0-9, containing the digit 0-9) and every interval of 1-100, with a prior weight inferred for each rule separately and a share lambda_0 for the rules as a group; each interval has prior (1 - lambda_0) / 100^2. Examples are sampled from the concept (the size principle), 10% of them uniformly from 1-100, and "yes" is the posterior probability that the concept contains the target.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/962b9e6d929c436d/repo/_runs/cell_1/experiment1/model_loop/models/bigelow_piantadosi_independent.py`

### erlang_interval_width_prior — rank 3, 2.8 ± 2.0 nats behind the best (1.4× dse: statistically tied with the best), ELPD-LOO -1214.9

**Hypothesis:** Refinement of `bayesian_concepts_with_guessing_lapse`: people judge by Bayesian concept learning over the same 45 independently weighted rules and the intervals of 1-100 (size principle, with a guessing lapse), but they do not consider every range of numbers equally plausible a priori — they expect a "range" concept to have a typical width, favouring intervals of moderate size over both tiny and nearly-all-numbers ranges (an Erlang-shaped prior over interval size, as in Tenenbaum's number game). The one change is this size prior on intervals, with its preferred width inferred from the data, which governs how far people extend "yes" around examples that fit no rule.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/962b9e6d929c436d/repo/_runs/cell_1/experiment1/model_loop/models/erlang_interval_width_prior.py`

### recency_decay_concept_learning — rank 4, 3.1 ± 2.0 nats behind the best (1.5× dse: statistically tied with the best), ELPD-LOO -1215.2

**Hypothesis:** People do Bayesian concept learning over the same rules and intervals (size principle, independently weighted rule prior), but their memory of the examples fades as they read the set: the most recently shown example counts in full, and each earlier example's evidence is discounted by a constant factor per later example. Because early examples are partly forgotten, concepts that fit the last examples but not the first ones keep substantial posterior weight, so generalisation is driven by the end of the list more than its beginning.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/962b9e6d929c436d/repo/_runs/cell_1/experiment1/model_loop/models/recency_decay_concept_learning.py`

### bayesian_concepts_with_guessing_lapse — rank 5, 3.4 ± 1.1 nats behind the best (3.2× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1215.5

**Hypothesis:** A refinement of bigelow_piantadosi_independent: people still judge by Bayesian concept learning over the same 45 rules and all intervals of 1-100 (independently weighted rule prior, size principle), but on a fraction of trials they do not consult that posterior at all and instead give a guess, saying "yes" with their own fixed guessing rate. The one change is this response-lapse stage, which explains why even targets that are among the examples are often rejected and why clearly excluded targets are sometimes accepted.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/962b9e6d929c436d/repo/_runs/cell_1/experiment1/model_loop/models/bayesian_concepts_with_guessing_lapse.py`

### posterior_confidence_miscalibration — rank 6, 5.8 ± 2.7 nats behind the best (2.2× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1217.9

**Hypothesis:** People do the normative Bayesian concept learning of the incumbent model (45 rules with individually weighted priors and all intervals of 1-100, size principle, 10% random examples), but they mis-weight their own posterior confidence when turning it into a yes/no answer: the log-odds of saying "yes" are the posterior log-odds that the concept contains the target, multiplied by a single confidence-scaling factor. A factor below one means people under-trust their inference (strongly supported generalisations are answered less consistently and strongly excluded targets get more yeses, in a graded way rather than a flat lapse), and a factor above one means they over-commit to the more probable answer.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/962b9e6d929c436d/repo/_runs/cell_1/experiment1/model_loop/models/posterior_confidence_miscalibration.py`

### non_accumulating_size_principle — rank 7, 82.7 ± 15.0 nats behind the best (5.5× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1294.7

**Hypothesis:** People do Bayesian concept learning over the same rules and intervals, but they do not accumulate the size-principle evidence across examples: they judge how well a concept fits the examples by the average fit of one example, sharpened by a fixed amount, so their generalisation is no tighter after four examples than after one. This disagrees most with the current best model on sets with one or two examples, where it predicts sharper, more rule-like generalisation, and on four-example sets, where it predicts broader, more graded generalisation that keeps giving partial "yes" to targets that fit only a looser concept (for example, odd numbers that are not in a narrow rule the examples happen to share).

**Source:** `/scr/wurgaft/auto-psych/agent_trees/962b9e6d929c436d/repo/_runs/cell_1/experiment1/model_loop/models/non_accumulating_size_principle.py`

### arithmetic_vs_digit_rule_grammar — rank 8, 102.8 ± 17.5 nats behind the best (5.9× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1314.9

**Hypothesis:** Refinement of `bigelow_piantadosi_mixture`: people do Bayesian concept learning over the same 45 rules and all intervals of 1-100 (size principle, 10% of examples random), but their prior over rules is not flat. It first chooses between two kinds of rule, arithmetic properties of the number (odd, even, primes, squares, cubes, powers, multiples) or patterns in how it is written in decimal (ends in a digit, contains a digit), and then picks uniformly within that kind. The one change is this two-kind grammar step, with the share of prior given to written-digit rules inferred from the data rather than fixed by how many such rules there are.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/962b9e6d929c436d/repo/_runs/cell_1/experiment1/model_loop/models/arithmetic_vs_digit_rule_grammar.py`

### bigelow_piantadosi_mixture — rank 9, 114.9 ± 16.7 nats behind the best (6.9× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1327.0

**Hypothesis:** Bayesian concept learning over 45 rules (odd, even, primes, squares, cubes, powers of 2-10, powers of 2 without 32 or with 37, multiples of 3-9, 11 and 12, ending in 0-9, containing the digit 0-9) and every interval of 1-100. A rule is chosen with probability lambda_0, and the rules are equally likely; each interval has prior (1 - lambda_0) / 100^2. Examples are sampled from the concept (the size principle), 10% of them uniformly from 1-100, and "yes" is the posterior probability that the concept contains the target.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/962b9e6d929c436d/repo/_runs/cell_1/experiment1/model_loop/models/bigelow_piantadosi_mixture.py`

### weak_sampling_concept_learning — rank 10, 174.1 ± 19.1 nats behind the best (9.1× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1386.2

**Hypothesis:** People infer which concept (one of the 45 arithmetic or digit rules, or an interval of 1-100) produced the examples, but they use weak sampling rather than the size principle: a concept is judged only by whether it is consistent with the examples, not by how small it is, so a broad concept like "even numbers" that contains the examples is exactly as well supported as a narrow one like "powers of 2". Their answer is the posterior probability that the concept contains the target, which predicts generalisation that stays broad and does not tighten as more consistent examples are seen.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/962b9e6d929c436d/repo/_runs/cell_1/experiment1/model_loop/models/weak_sampling_concept_learning.py`

### exemplar_feature_similarity — rank 11, 319.5 ± 26.9 nats behind the best (11.9× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1531.6

**Hypothesis:** People do not infer a rule at all; they judge a target by its summed similarity to the remembered examples (exemplar generalization), where similarity between two numbers falls off exponentially with how many simple arithmetic properties they disagree on (parity, divisibility by 3-10, last digit, tens digit, being prime, square or a power of two) plus how far apart they are in magnitude. A target that closely resembles even one example is accepted, with no size-principle penalty and no requirement that it fit all examples jointly.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/962b9e6d929c436d/repo/_runs/cell_1/experiment1/model_loop/models/exemplar_feature_similarity.py`

### exemplar_attention_similarity — rank 12, 336.0 ± 27.4 nats behind the best (12.3× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1548.1

**Hypothesis:** People do not infer a rule; they judge the target by its summed similarity to the stored examples (exemplar-based generalization, as in the Generalized Context Model). Each number is represented on a few psychological dimensions — magnitude, parity, final digit and tens digit — each with its own attention weight, similarity to an example decays exponentially with attention-weighted mismatch, and the probability of "yes" rises with the target's average similarity to the examples.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/962b9e6d929c436d/repo/_runs/cell_1/experiment1/model_loop/models/exemplar_attention_similarity.py`

## Pruned models (out of the set; narrowest margin first)

No model has been pruned yet in this project.
