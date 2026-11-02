# Refinement menu

The models you may refine, other than the incumbent `noisy_examples_erlang_intervals`. Each entry is one mechanism: its hypothesis in full as its author stated it, how it stands, and where its source is. This is a menu, not a list of what is ruled out: a pruned model lost to the best model on the data it was scored on, by the stated margin, but its mechanism may be partly right — and a model that lost narrowly is the most promising target here. Read the source of the model you pick before you write anything.

## Live models other than the incumbent (in the set; best first)

### flexible_shape_interval_width_prior — rank 1, 1.2 ± 0.4 nats behind the best (2.7× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1211.5

**Hypothesis:** A refinement of noisy_examples_erlang_intervals: people do the same Bayesian concept learning over the 45 independently weighted rules and all intervals of 1-100, with the size principle and an inferred rate of random examples, and a prior over how wide a "range" concept is, but that width prior is not fixed to the Erlang shape. The one change is that the shape of the width prior is inferred along with its typical width (a gamma-shaped prior whose shape can fall below the Erlang value), so people can give tight ranges hugging the examples high prior weight while still allowing occasional wide ranges. This predicts that people extend "yes" beyond tightly clustered examples less than the incumbent does, while still generalising across widely spread ones.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/962b9e6d929c436d/repo/_runs/cell_1/experiment1/model_loop/models/flexible_shape_interval_width_prior.py`

### inferred_example_noise_concepts — rank 2, 1.8 ± 1.5 nats behind the best (1.2× dse: statistically tied with the best), ELPD-LOO -1212.1

**Hypothesis:** A refinement of bigelow_piantadosi_independent: people do the same Bayesian concept learning over the 45 rules (each with its own inferred prior weight) and all intervals of 1-100 with the size principle, but they do not assume a fixed 10% of examples are random noise; how often they believe the program emits an example from outside its concept is a free quantity inferred from the data. The one change is this inferred example-noise rate, which governs both how sharply the size principle accumulates over examples and how readily a rule is kept despite an example it does not cover.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/962b9e6d929c436d/repo/_runs/cell_1/experiment1/model_loop/models/inferred_example_noise_concepts.py`

### log_scale_interval_width_prior — rank 3, 1.8 ± 1.8 nats behind the best (1.0× dse: statistically tied with the best), ELPD-LOO -1212.1

**Hypothesis:** A refinement of erlang_interval_size_prior: people do the same Bayesian concept learning over the 45 independently weighted rules and all intervals of 1-100 (size principle, 10% of examples random) with a prior favouring ranges of a typical width, but they judge a range's width on a compressed, logarithmic mental number line rather than in raw counts. The one change is that the Erlang width prior applies to the ratio of a range's upper to lower end (its log-width), so a "typical" range is narrow among small numbers (e.g. 3-6) and wide among large ones (e.g. 60-100), making people extend "yes" further around large examples than around small ones that fit no rule.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/962b9e6d929c436d/repo/_runs/cell_1/experiment1/model_loop/models/log_scale_interval_width_prior.py`

### erlang_interval_size_prior — rank 4, 2.3 ± 1.7 nats behind the best (1.4× dse: statistically tied with the best), ELPD-LOO -1212.5

**Hypothesis:** A refinement of bigelow_piantadosi_independent: people do the same Bayesian concept learning over the 45 independently weighted rules and all intervals of 1-100 (size principle, 10% of examples random), but they do not treat every interval as equally likely a priori. Their prior over intervals favours ranges of a typical width (an Erlang prior on interval size, as in Tenenbaum's number game), with that preferred width inferred from the data, so they neither readily entertain tiny ranges hugging the examples nor ranges spanning most of 1-100; the one change is this size-dependent interval prior in place of the uniform one.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/962b9e6d929c436d/repo/_runs/cell_1/experiment1/model_loop/models/erlang_interval_size_prior.py`

### new_set_cautious_erlang_concepts — rank 5, 2.3 ± 1.4 nats behind the best (1.6× dse: statistically tied with the best), ELPD-LOO -1212.6

**Hypothesis:** A refinement of noisy_examples_erlang_intervals: people do the same Bayesian concept learning (45 independently weighted rules, all intervals of 1-100 with an Erlang prior favouring a typical range width, size principle, inferred rate of random examples), but on the first question after a new set of examples appears they are initially cautious and withhold some of the "yes" answers that inference supports, only fully committing to their generalisation once they have been asked about the same set again. The one change is this new-set caution: on the first query after a set change the probability of "yes" is the posterior probability that the concept contains the target scaled down by an inferred withholding fraction, while repeated queries about the same set follow the posterior unchanged.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/962b9e6d929c436d/repo/_runs/cell_1/experiment1/model_loop/models/new_set_cautious_erlang_concepts.py`

### log_number_line_interval_prior — rank 6, 2.7 ± 1.5 nats behind the best (1.7× dse: statistically tied with the best), ELPD-LOO -1213.0

**Hypothesis:** A refinement of inferred_example_noise_concepts: people do the same Bayesian concept learning over the 45 independently weighted rules and all intervals of 1-100, with the size principle and an inferred rate of random examples, but they picture ranges of numbers on a compressed, logarithmic-like mental number line, so a priori they find ranges whose endpoints are small numbers more plausible than equally wide ranges among large numbers. The one change is this magnitude-dependent prior over intervals (the strength of the compression inferred from the data, with zero recovering the incumbent's uniform prior), which predicts that "yes" extends further below the examples than above them.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/962b9e6d929c436d/repo/_runs/cell_1/experiment1/model_loop/models/log_number_line_interval_prior.py`

### bigelow_piantadosi_independent — rank 7, 3.9 ± 1.9 nats behind the best (2.1× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1214.2

**Hypothesis:** Bayesian concept learning over 45 rules (odd, even, primes, squares, cubes, powers of 2-10, powers of 2 without 32 or with 37, multiples of 3-9, 11 and 12, ending in 0-9, containing the digit 0-9) and every interval of 1-100, with a prior weight inferred for each rule separately and a share lambda_0 for the rules as a group; each interval has prior (1 - lambda_0) / 100^2. Examples are sampled from the concept (the size principle), 10% of them uniformly from 1-100, and "yes" is the posterior probability that the concept contains the target.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/962b9e6d929c436d/repo/_runs/cell_1/experiment1/model_loop/models/bigelow_piantadosi_independent.py`

### erlang_interval_width_prior — rank 8, 4.6 ± 1.4 nats behind the best (3.3× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1214.9

**Hypothesis:** Refinement of `bayesian_concepts_with_guessing_lapse`: people judge by Bayesian concept learning over the same 45 independently weighted rules and the intervals of 1-100 (size principle, with a guessing lapse), but they do not consider every range of numbers equally plausible a priori — they expect a "range" concept to have a typical width, favouring intervals of moderate size over both tiny and nearly-all-numbers ranges (an Erlang-shaped prior over interval size, as in Tenenbaum's number game). The one change is this size prior on intervals, with its preferred width inferred from the data, which governs how far people extend "yes" around examples that fit no rule.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/962b9e6d929c436d/repo/_runs/cell_1/experiment1/model_loop/models/erlang_interval_width_prior.py`

### recency_decay_concept_learning — rank 9, 4.9 ± 2.4 nats behind the best (2.1× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1215.2

**Hypothesis:** People do Bayesian concept learning over the same rules and intervals (size principle, independently weighted rule prior), but their memory of the examples fades as they read the set: the most recently shown example counts in full, and each earlier example's evidence is discounted by a constant factor per later example. Because early examples are partly forgotten, concepts that fit the last examples but not the first ones keep substantial posterior weight, so generalisation is driven by the end of the list more than its beginning.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/962b9e6d929c436d/repo/_runs/cell_1/experiment1/model_loop/models/recency_decay_concept_learning.py`

### bayesian_concepts_with_guessing_lapse — rank 10, 5.3 ± 1.6 nats behind the best (3.2× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1215.5

**Hypothesis:** A refinement of bigelow_piantadosi_independent: people still judge by Bayesian concept learning over the same 45 rules and all intervals of 1-100 (independently weighted rule prior, size principle), but on a fraction of trials they do not consult that posterior at all and instead give a guess, saying "yes" with their own fixed guessing rate. The one change is this response-lapse stage, which explains why even targets that are among the examples are often rejected and why clearly excluded targets are sometimes accepted.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/962b9e6d929c436d/repo/_runs/cell_1/experiment1/model_loop/models/bayesian_concepts_with_guessing_lapse.py`

### posterior_confidence_miscalibration — rank 11, 7.7 ± 3.1 nats behind the best (2.5× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1217.9

**Hypothesis:** People do the normative Bayesian concept learning of the incumbent model (45 rules with individually weighted priors and all intervals of 1-100, size principle, 10% random examples), but they mis-weight their own posterior confidence when turning it into a yes/no answer: the log-odds of saying "yes" are the posterior log-odds that the concept contains the target, multiplied by a single confidence-scaling factor. A factor below one means people under-trust their inference (strongly supported generalisations are answered less consistently and strongly excluded targets get more yeses, in a graded way rather than a flat lapse), and a factor above one means they over-commit to the more probable answer.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/962b9e6d929c436d/repo/_runs/cell_1/experiment1/model_loop/models/posterior_confidence_miscalibration.py`

### offset_arithmetic_progressions — rank 12, 18.8 ± 5.4 nats behind the best (3.5× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1229.1

**Hypothesis:** A refinement of noisy_examples_erlang_intervals: people do the same Bayesian concept learning (size principle, inferred share of random examples, Erlang prior on interval width), but their space of rules also includes arithmetic progressions that do not start at zero — every number that leaves a given remainder when divided by 3 to 12 (e.g. 2, 5, 8, 11, ... or 5, 11, 17, 23, ...), each such "step" rule equally likely a priori and the group as a whole given a prior share inferred from the data. The one change is this added family of offset arithmetic-progression concepts, which predicts that examples sharing a step-and-offset pattern (e.g. 23, 35, 83, 95, all 5 more than a multiple of 6) are generalised to that narrow progression rather than to a broad rule such as odd numbers.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/962b9e6d929c436d/repo/_runs/cell_1/experiment1/model_loop/models/offset_arithmetic_progressions.py`

### noisy_posterior_criterion — rank 13, 33.8 ± 6.3 nats behind the best (5.4× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1244.0

**Hypothesis:** People do the same Bayesian concept learning as bigelow_piantadosi_independent (45 individually weighted rules and all intervals of 1-100, size principle, 10% random examples), but they answer "yes" by comparing their posterior belief that the target belongs to the concept against an internal criterion of how sure they must be, with noise in that comparison. The criterion (which need not be 50%) and the comparison noise are inferred; because the comparison is made on the probability itself, confident beliefs at either extreme give similar, non-certain answers, and beliefs below the criterion are mostly refused.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/962b9e6d929c436d/repo/_runs/cell_1/experiment1/model_loop/models/noisy_posterior_criterion.py`

### win_stay_lose_shift_concepts — rank 14, 61.1 ± 9.4 nats behind the best (6.5× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1271.4

**Hypothesis:** People do not hold a full posterior over concepts; they keep a single running guess and revise it example by example (win-stay, lose-shift). After the first example they pick a concept (rule or range) in proportion to its Bayesian posterior given that example; each later example that fits the current guess leaves it unchanged, and only an example that does not fit makes them abandon it and re-pick a concept from the posterior given all examples seen so far. "Yes" is the probability that the guess they end with contains the target, so the size principle tightens generalisation only when a later example forces a switch, and the order of the examples changes which guess they end up holding.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/962b9e6d929c436d/repo/_runs/cell_1/experiment1/model_loop/models/win_stay_lose_shift_concepts.py`

### non_accumulating_size_principle — rank 15, 84.5 ± 14.9 nats behind the best (5.7× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1294.7

**Hypothesis:** People do Bayesian concept learning over the same rules and intervals, but they do not accumulate the size-principle evidence across examples: they judge how well a concept fits the examples by the average fit of one example, sharpened by a fixed amount, so their generalisation is no tighter after four examples than after one. This disagrees most with the current best model on sets with one or two examples, where it predicts sharper, more rule-like generalisation, and on four-example sets, where it predicts broader, more graded generalisation that keeps giving partial "yes" to targets that fit only a looser concept (for example, odd numbers that are not in a narrow rule the examples happen to share).

**Source:** `/scr/wurgaft/auto-psych/agent_trees/962b9e6d929c436d/repo/_runs/cell_1/experiment1/model_loop/models/non_accumulating_size_principle.py`

### arithmetic_vs_digit_rule_grammar — rank 16, 104.7 ± 17.5 nats behind the best (6.0× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1314.9

**Hypothesis:** Refinement of `bigelow_piantadosi_mixture`: people do Bayesian concept learning over the same 45 rules and all intervals of 1-100 (size principle, 10% of examples random), but their prior over rules is not flat. It first chooses between two kinds of rule, arithmetic properties of the number (odd, even, primes, squares, cubes, powers, multiples) or patterns in how it is written in decimal (ends in a digit, contains a digit), and then picks uniformly within that kind. The one change is this two-kind grammar step, with the share of prior given to written-digit rules inferred from the data rather than fixed by how many such rules there are.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/962b9e6d929c436d/repo/_runs/cell_1/experiment1/model_loop/models/arithmetic_vs_digit_rule_grammar.py`

### bigelow_piantadosi_mixture — rank 17, 116.7 ± 16.6 nats behind the best (7.1× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1327.0

**Hypothesis:** Bayesian concept learning over 45 rules (odd, even, primes, squares, cubes, powers of 2-10, powers of 2 without 32 or with 37, multiples of 3-9, 11 and 12, ending in 0-9, containing the digit 0-9) and every interval of 1-100. A rule is chosen with probability lambda_0, and the rules are equally likely; each interval has prior (1 - lambda_0) / 100^2. Examples are sampled from the concept (the size principle), 10% of them uniformly from 1-100, and "yes" is the posterior probability that the concept contains the target.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/962b9e6d929c436d/repo/_runs/cell_1/experiment1/model_loop/models/bigelow_piantadosi_mixture.py`

### weak_sampling_concept_learning — rank 18, 175.9 ± 19.5 nats behind the best (9.0× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1386.2

**Hypothesis:** People infer which concept (one of the 45 arithmetic or digit rules, or an interval of 1-100) produced the examples, but they use weak sampling rather than the size principle: a concept is judged only by whether it is consistent with the examples, not by how small it is, so a broad concept like "even numbers" that contains the examples is exactly as well supported as a narrow one like "powers of 2". Their answer is the posterior probability that the concept contains the target, which predicts generalisation that stays broad and does not tighten as more consistent examples are seen.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/962b9e6d929c436d/repo/_runs/cell_1/experiment1/model_loop/models/weak_sampling_concept_learning.py`

### exemplar_feature_similarity — rank 19, 321.4 ± 26.9 nats behind the best (12.0× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1531.6

**Hypothesis:** People do not infer a rule at all; they judge a target by its summed similarity to the remembered examples (exemplar generalization), where similarity between two numbers falls off exponentially with how many simple arithmetic properties they disagree on (parity, divisibility by 3-10, last digit, tens digit, being prime, square or a power of two) plus how far apart they are in magnitude. A target that closely resembles even one example is accepted, with no size-principle penalty and no requirement that it fit all examples jointly.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/962b9e6d929c436d/repo/_runs/cell_1/experiment1/model_loop/models/exemplar_feature_similarity.py`

### exemplar_attention_similarity — rank 20, 337.8 ± 27.4 nats behind the best (12.3× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1548.1

**Hypothesis:** People do not infer a rule; they judge the target by its summed similarity to the stored examples (exemplar-based generalization, as in the Generalized Context Model). Each number is represented on a few psychological dimensions — magnitude, parity, final digit and tens digit — each with its own attention weight, similarity to an example decays exponentially with attention-weighted mismatch, and the probability of "yes" rises with the target's average similarity to the examples.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/962b9e6d929c436d/repo/_runs/cell_1/experiment1/model_loop/models/exemplar_attention_similarity.py`

### spacing_scaled_magnitude_similarity — rank 21, 344.9 ± 27.5 nats behind the best (12.6× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1555.2

**Hypothesis:** People judge a target by its numerical closeness to the nearest example on the 1-100 number line, with no rule inference: the chance of "yes" falls off smoothly with the distance from the target to the closest example. The width of this generalization gradient is set by how spread out the examples are — the mean gap between neighbouring examples (plus a baseline width for single examples) — so widely spaced examples license "yes" to far-away targets, while tightly clustered examples restrict "yes" to targets right beside them.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/962b9e6d929c436d/repo/_runs/cell_1/experiment1/model_loop/models/spacing_scaled_magnitude_similarity.py`

### exemplar_consequential_region_similarity — rank 22, 374.5 ± 34.6 nats behind the best (10.8× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1584.8

**Hypothesis:** People do not infer one concept that covers all the examples together; they store each example as a separate exemplar and judge the target by its average similarity to the stored exemplars. The similarity of the target to one exemplar is Shepard-style generalization through consequential regions: how likely a number concept (an arithmetic or digit rule, or a range of a typical width) that contains that single exemplar also contains the target, with smaller concepts counting more. Because each exemplar is generalized from on its own and the results are summed, a target resembling only some of the examples gets partial "yes", and evidence never tightens jointly across examples as it would under concept inference.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/962b9e6d929c436d/repo/_runs/cell_1/experiment1/model_loop/models/exemplar_consequential_region_similarity.py`

## Pruned models (out of the set; narrowest margin first)

No model has been pruned yet in this project.
