# Refinement menu

The models you may refine, other than the incumbent `extended_multiples_concepts`. Each entry is one mechanism: its hypothesis in full as its author stated it, how it stands, and where its source is. This is a menu, not a list of what is ruled out: a pruned model lost to the best model on the data it was scored on, by the stated margin, but its mechanism may be partly right — and a model that lost narrowly is the most promising target here. Read the source of the model you pick before you write anything.

## Live models other than the incumbent (in the set; best first)

### without_replacement_example_sampling — rank 1, 0.6 ± 2.4 nats behind the best (0.3× dse: statistically tied with the best), ELPD-LOO -1699.3

**Hypothesis:** People do the same Bayesian concept learning as the current best model (45 independently weighted rules, ranges of 1-100 with a preferred typical width, an inferred share of random examples), but they assume the program never produces the same number twice, so they treat the examples as drawn from the concept without replacement: each further example that fits a concept is taken as one of that concept's members not yet shown. This makes several examples far stronger evidence for a concept that has barely more members than were shown (cubes, powers of 3 or 4, squares, a tight range) than the best model's with-replacement size principle allows, so the sharpest disagreement lives on three- and four-example sets that nearly exhaust a small rule or range, where this model predicts fewer "yes" answers to targets outside that small concept.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/962b9e6d929c436d/repo/_runs/cell_1/experiment2/model_loop/models/without_replacement_example_sampling.py`

### large_number_biased_sampling — rank 2, 1.4 ± 1.4 nats behind the best (1.0× dse: statistically tied with the best), ELPD-LOO -1700.1

**Hypothesis:** People do the same Bayesian concept learning as the current best model (45 independently weighted rules plus multiples of 13-50, ranges of 1-100 with a preferred typical width, an inferred share of random examples, examples taken without replacement), but they believe the program does not pick its outputs evenly from its concept: it produces larger members more often than smaller ones (output chance growing as a power of the number, the power inferred, zero recovering the best model). Because small numbers then add little to how many outputs a concept "spends" elsewhere, a range stretched downward toward small numbers costs little under the size principle while one stretched upward costs much, so the sharpest disagreement with the best model lives just outside sets of examples that fit no rule, where this model predicts "yes" to targets 1-15 below the smallest example far more than to targets 1-15 above the largest — through the sampling assumption alone, with no new concepts added.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/962b9e6d929c436d/repo/_runs/cell_1/experiment2/model_loop/models/large_number_biased_sampling.py`

### range_specific_evidence_weighting — rank 3, 1.8 ± 2.7 nats behind the best (0.7× dse: statistically tied with the best), ELPD-LOO -1700.5

**Hypothesis:** A refinement of without_replacement_example_sampling: people do the same Bayesian concept learning (45 independently weighted rules, ranges of 1-100 with a preferred typical width, an inferred share of random examples, examples taken without replacement), but they weigh the examples' evidence differently for the two kinds of concept — the evidence for an arithmetic or digit rule is used at full strength, while the evidence for a range of magnitudes is weighted by its own inferred power (below one: range generalisation is a looser, similarity-like judgement that tightens slowly as examples accumulate; above one: ranges are pinned down faster). The one change is this range-specific evidence weight, which predicts that for examples fitting no rule "yes" spreads around the examples more (or less) than calibrated inference implies, while generalisation within rule-like sets is unchanged.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/962b9e6d929c436d/repo/_runs/cell_1/experiment2/model_loop/models/range_specific_evidence_weighting.py`

### conservative_evidence_weighting — rank 4, 1.9 ± 3.0 nats behind the best (0.6× dse: statistically tied with the best), ELPD-LOO -1700.6

**Hypothesis:** People do the same Bayesian concept learning as the current best model (45 independently weighted rules, ranges of 1-100 with a preferred typical width, an inferred share of random examples, examples drawn from the concept without replacement), but they are conservative updaters: they under-weight (or over-weight) the evidence of the examples relative to their prior beliefs about which concepts are plausible, as if the examples' likelihood were raised to a single power below (or above) one. This predicts that generalisation tightens toward the smallest consistent concept more slowly (or faster) as examples accumulate than calibrated inference implies, while the relative plausibility of concepts before any evidence is unchanged.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/962b9e6d929c436d/repo/_runs/cell_1/experiment2/model_loop/models/conservative_evidence_weighting.py`

### strong_weak_sampling_mixture — rank 5, 2.5 ± 2.7 nats behind the best (0.9× dse: statistically tied with the best), ELPD-LOO -1701.2

**Hypothesis:** A refinement of without_replacement_example_sampling: people do the same Bayesian concept learning (45 independently weighted rules, ranges of 1-100 with a preferred typical width, an inferred share of random examples, examples taken from the concept without replacement), but they are unsure how the program chose its examples — whether it picked them from the concept itself (strong sampling, which makes small concepts much more likely) or picked numbers at random and showed the ones that happened to fit (weak sampling, where a concept's size does not matter). The one change is this inferred mixture of strong and weak sampling, which weakens the size principle — so people keep generalising to broader concepts such as "even numbers" after several examples of a narrow one — without making them any more willing to keep a concept that excludes one of the examples.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/962b9e6d929c436d/repo/_runs/cell_1/experiment2/model_loop/models/strong_weak_sampling_mixture.py`

### guessing_lapse_without_replacement — rank 6, 2.7 ± 2.4 nats behind the best (1.1× dse: statistically tied with the best), ELPD-LOO -1701.4

**Hypothesis:** A refinement of without_replacement_example_sampling: people do the same Bayesian concept learning (45 independently weighted rules, ranges of 1-100 with a preferred typical width, an inferred share of random examples, examples taken as drawn from the concept without replacement), but on some share of trials they do not consult their inference at all and simply guess, saying "yes" at their own fixed rate. The one change is this response-stage guessing lapse (grafted from bayesian_concepts_with_guessing_lapse), which claims that "no" answers to the shown examples and "yes" answers to far-off targets come partly from inattentive guessing rather than only from believing that some examples were random, so the inferred share of random examples need not carry both and generalisation from the examples can be sharper.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/962b9e6d929c436d/repo/_runs/cell_1/experiment2/model_loop/models/guessing_lapse_without_replacement.py`

### first_example_trust_concepts — rank 7, 2.7 ± 3.3 nats behind the best (0.8× dse: statistically tied with the best), ELPD-LOO -1701.4

**Hypothesis:** People do the same Bayesian concept learning as the current best model (45 independently weighted rules, ranges of 1-100 with a preferred typical width, examples drawn without replacement, an inferred share of random examples), but they trust the first example more than the later ones: the first number shown is taken as the one most surely produced by the concept, while later examples are more readily dismissed as random outputs. The one change is that the chance an example is random is smaller for the first example (by an inferred factor) than for later ones, so concepts that exclude the first example are penalised more than concepts that exclude a later example, and the same set in a different order generalises differently.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/962b9e6d929c436d/repo/_runs/cell_1/experiment2/model_loop/models/first_example_trust_concepts.py`

### count_from_one_ceiling_concepts — rank 8, 2.7 ± 2.2 nats behind the best (1.2× dse: statistically tied with the best), ELPD-LOO -1701.4

**Hypothesis:** People picture the program as a counter that produces numbers from 1 up to some ceiling, so besides the arithmetic and digit rules and ranges of a typical width, they entertain "every number from 1 up to k" (anchored at the start of the number line, any ceiling equally likely a priori, the family taking an inferred share of the prior) and infer the ceiling from the examples by the size principle, as in estimating a maximum from samples. Because such concepts always contain every number below the examples but none above the largest, this predicts that people extend "yes" downward toward small numbers more readily than upward beyond the largest example, an asymmetry that symmetric ranges and rules cannot produce.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/962b9e6d929c436d/repo/_runs/cell_1/experiment2/model_loop/models/count_from_one_ceiling_concepts.py`

### log_width_without_replacement — rank 9, 7.6 ± 4.2 nats behind the best (1.8× dse: statistically tied with the best), ELPD-LOO -1706.3

**Hypothesis:** I chose log_scale_interval_width_prior and refined it. People judge how wide a "range" concept is on a compressed, logarithmic mental number line, so their prior expects a typical range to be narrow among small numbers (e.g. 3-6) and wide among large ones (e.g. 60-100); the one change is the concept learner underneath, which is replaced by the current best one (45 independently weighted rules, an inferred share of random examples, and examples taken as drawn from the concept without replacement) in place of the old fixed-10%-noise, with-replacement learner. This predicts that, for examples that fit no rule, "yes" extends much further around large examples than around small ones, and further below a cluster of large examples than the linear-width prior of the best model allows.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/962b9e6d929c436d/repo/_runs/cell_1/experiment2/model_loop/models/log_width_without_replacement.py`

### stray_count_per_set_noise — rank 10, 18.6 ± 6.9 nats behind the best (2.7× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1717.3

**Hypothesis:** I chose the incumbent, without_replacement_example_sampling, and refined how it believes random examples arise: people do the same Bayesian concept learning (45 independently weighted rules, ranges of 1-100 with a preferred typical width, in-concept examples drawn without replacement), but they think of a stray example as an occasional slip of the program per set of examples, not as an independent per-example coin flip — the chance that a set contains one, two or more random numbers falls off geometrically with their count (rate inferred) and does not grow with how many examples were shown, and any given count of strays is equally likely to be any of the examples. This predicts that with one or two examples a concept excluding an example is kept about as readily as with four, so the odd example out of a long, otherwise regular set (such as 25 in 91, 25, 61, 31) is judged a stray less readily than independent per-example noise implies, while short sets are discounted more readily.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/962b9e6d929c436d/repo/_runs/cell_1/experiment2/model_loop/models/stray_count_per_set_noise.py`

### single_outlier_concept_learning — rank 11, 69.6 ± 27.4 nats behind the best (2.5× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1768.2

**Hypothesis:** A refinement of without_replacement_example_sampling: people do the same Bayesian concept learning (45 independently weighted rules, ranges of 1-100 with a preferred typical width, examples taken from the concept without replacement), but they do not think each example has its own independent chance of being random. They think a set holds at most one stray example: with some inferred probability one of the examples, any of them equally, is a random number, and otherwise every example came from the concept. The one change is this at-most-one-outlier view of noise, which predicts that people forgive a single example that breaks an otherwise clean pattern even in a four-example set, but completely reject any concept that leaves out two or more of the examples, where the incumbent still gives such concepts some weight.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/962b9e6d929c436d/repo/_runs/cell_1/experiment2/model_loop/models/single_outlier_concept_learning.py`

### predicted_output_threshold — rank 12, 144.8 ± 18.8 nats behind the best (7.7× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1843.5

**Hypothesis:** People do the same Bayesian concept learning as the current best model (45 independently weighted rules, ranges of 1-100 with a preferred typical width, an inferred share of random examples, examples taken as drawn without replacement), but their decision rule is not "is the target in the concept?": they ask how probable it is that the program would output this very number, and answer "yes" when that predicted output probability clears an internal threshold (with noise in the comparison). Because a member of a small concept is a far more probable output than a member of a big one, this predicts that, at the same belief that the target belongs to the concept, targets supported by narrow concepts (powers, cubes, tight ranges) get more "yes" than targets supported only by broad ones (even numbers, wide ranges).

**Source:** `/scr/wurgaft/auto-psych/agent_trees/962b9e6d929c436d/repo/_runs/cell_1/experiment2/model_loop/models/predicted_output_threshold.py`

### most_specific_concept_only — rank 13, 573.2 ± 37.3 nats behind the best (15.3× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -2271.9

**Hypothesis:** People do not weigh many candidate concepts; they adopt the single most specific concept that covers all the examples — the smallest of the arithmetic or digit rules (odd, primes, squares, powers, multiples, final digit, contained digit) or the tightest range spanning the examples — and answer "yes" when the target belongs to that concept and "no" otherwise, apart from a constant rate of response noise. This minimal-generalization strategy predicts no graded size-principle averaging: a single example generalizes to almost nothing, and "yes" falls off sharply at the edge of the one concept chosen.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/962b9e6d929c436d/repo/_runs/cell_1/experiment2/model_loop/models/most_specific_concept_only.py`

### distinctive_common_feature_similarity — rank 14, 821.8 ± 42.3 nats behind the best (19.4× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -2520.5

**Hypothesis:** People judge the target by its similarity to the example set as a whole, using a feature-based (Tversky-style common-features) metric: each arithmetic or digit property the target shares with the examples (parity, multiples, powers, squares, final digit, contained digit, and so on) adds to similarity in proportion to how consistently the examples share it and how distinctive (rare among 1-100) the property is, and closeness on the number line to the span of the examples adds a magnitude-band match that decays with distance. The probability of "yes" rises smoothly with this summed similarity; there is no inference over which single concept produced the examples, so a target that shares a distinctive property with most but not all examples still gets substantial "yes".

**Source:** `/scr/wurgaft/auto-psych/agent_trees/962b9e6d929c436d/repo/_runs/cell_1/experiment2/model_loop/models/distinctive_common_feature_similarity.py`

## Pruned models (out of the set; narrowest margin first)

### noisy_examples_erlang_intervals — pruned (experiment1 end of experiment)

**Margin:** 3.8 nats behind without_replacement_example_sampling (3.0× dse)

**Hypothesis:** A refinement of inferred_example_noise_concepts: people do the same Bayesian concept learning over the 45 independently weighted rules and all intervals of 1-100, with the size principle and an inferred rate at which they believe examples are random noise, but they do not treat every interval as equally likely a priori. The one change is grafting the Erlang interval-size prior from erlang_interval_size_prior: people expect a "range" concept to have a typical width (inferred from the data), so they neither readily entertain tiny ranges hugging the examples nor ranges spanning most of 1-100, which governs how far they extend "yes" around examples that fit no rule.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/962b9e6d929c436d/repo/_runs/cell_1/experiment1/model_loop/models/pruned/noisy_examples_erlang_intervals.py`

### decade_aligned_range_concepts — pruned (experiment1 end of experiment)

**Margin:** 3.8 nats behind without_replacement_example_sampling (2.8× dse)

**Hypothesis:** A refinement of noisy_examples_erlang_intervals: people do the same Bayesian concept learning over the 45 independently weighted rules and all intervals of 1-100, with the size principle, an inferred rate of random examples and an Erlang prior favouring a typical range width, but they do not find every range equally natural: ranges whose ends fall on decade boundaries (starting at 1, 11, 21, ... or ending at 10, 20, ..., 100, such as "the 30s" or "1 to 20") are a priori more plausible than equally wide ranges with arbitrary ends. The one change is this decade-alignment boost to the interval prior (its strength inferred, zero recovering the incumbent), which predicts that "yes" around examples that fit no rule stops sharply at decade boundaries, e.g. examples 32 and 37 generalise to 30-39 more than to 28 or 41.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/962b9e6d929c436d/repo/_runs/cell_1/experiment1/model_loop/models/pruned/decade_aligned_range_concepts.py`

### flexible_shape_interval_width_prior — pruned (experiment1 end of experiment)

**Margin:** 4.9 nats behind without_replacement_example_sampling (3.8× dse)

**Hypothesis:** A refinement of noisy_examples_erlang_intervals: people do the same Bayesian concept learning over the 45 independently weighted rules and all intervals of 1-100, with the size principle and an inferred rate of random examples, and a prior over how wide a "range" concept is, but that width prior is not fixed to the Erlang shape. The one change is that the shape of the width prior is inferred along with its typical width (a gamma-shaped prior whose shape can fall below the Erlang value), so people can give tight ranges hugging the examples high prior weight while still allowing occasional wide ranges. This predicts that people extend "yes" beyond tightly clustered examples less than the incumbent does, while still generalising across widely spread ones.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/962b9e6d929c436d/repo/_runs/cell_1/experiment1/model_loop/models/pruned/flexible_shape_interval_width_prior.py`

### overcommitted_noisy_erlang_concepts — pruned (experiment1 end of experiment)

**Margin:** 5.1 nats behind without_replacement_example_sampling (2.3× dse)

**Hypothesis:** I chose posterior_confidence_miscalibration and refined it. People turn their posterior belief that the target belongs to the concept into a yes/no answer by scaling its log-odds with a single confidence factor. The factor is inferred; above one means they commit more firmly than their belief warrants. The one change is the concept learner underneath: the old one assumed a fixed 10% of random examples and treated all intervals as equally likely. It is replaced by one that infers how often examples are random and expects ranges of a typical width (Erlang). This claims that people's beliefs about the concept are broad, because they discount examples as possibly random, but that they answer more decisively than those broad beliefs imply, giving firmer yes and no answers than calibrated Bayesian inference would.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/962b9e6d929c436d/repo/_runs/cell_1/experiment1/model_loop/models/pruned/overcommitted_noisy_erlang_concepts.py`

### inferred_example_noise_concepts — pruned (experiment1 end of experiment)

**Margin:** 5.6 nats behind without_replacement_example_sampling (2.7× dse)

**Hypothesis:** A refinement of bigelow_piantadosi_independent: people do the same Bayesian concept learning over the 45 rules (each with its own inferred prior weight) and all intervals of 1-100 with the size principle, but they do not assume a fixed 10% of examples are random noise; how often they believe the program emits an example from outside its concept is a free quantity inferred from the data. The one change is this inferred example-noise rate, which governs both how sharply the size principle accumulates over examples and how readily a rule is kept despite an example it does not cover.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/962b9e6d929c436d/repo/_runs/cell_1/experiment1/model_loop/models/pruned/inferred_example_noise_concepts.py`

### log_scale_interval_width_prior — pruned (experiment1 end of experiment)

**Margin:** 5.6 nats behind without_replacement_example_sampling (2.5× dse)

**Hypothesis:** A refinement of erlang_interval_size_prior: people do the same Bayesian concept learning over the 45 independently weighted rules and all intervals of 1-100 (size principle, 10% of examples random) with a prior favouring ranges of a typical width, but they judge a range's width on a compressed, logarithmic mental number line rather than in raw counts. The one change is that the Erlang width prior applies to the ratio of a range's upper to lower end (its log-width), so a "typical" range is narrow among small numbers (e.g. 3-6) and wide among large ones (e.g. 60-100), making people extend "yes" further around large examples than around small ones that fit no rule.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/962b9e6d929c436d/repo/_runs/cell_1/experiment1/model_loop/models/pruned/log_scale_interval_width_prior.py`

### erlang_interval_size_prior — pruned (experiment1 end of experiment)

**Margin:** 6.0 nats behind without_replacement_example_sampling (3.0× dse)

**Hypothesis:** A refinement of bigelow_piantadosi_independent: people do the same Bayesian concept learning over the 45 independently weighted rules and all intervals of 1-100 (size principle, 10% of examples random), but they do not treat every interval as equally likely a priori. Their prior over intervals favours ranges of a typical width (an Erlang prior on interval size, as in Tenenbaum's number game), with that preferred width inferred from the data, so they neither readily entertain tiny ranges hugging the examples nor ranges spanning most of 1-100; the one change is this size-dependent interval prior in place of the uniform one.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/962b9e6d929c436d/repo/_runs/cell_1/experiment1/model_loop/models/pruned/erlang_interval_size_prior.py`

### new_set_cautious_erlang_concepts — pruned (experiment1 end of experiment)

**Margin:** 6.1 nats behind without_replacement_example_sampling (2.7× dse)

**Hypothesis:** A refinement of noisy_examples_erlang_intervals: people do the same Bayesian concept learning (45 independently weighted rules, all intervals of 1-100 with an Erlang prior favouring a typical range width, size principle, inferred rate of random examples), but on the first question after a new set of examples appears they are initially cautious and withhold some of the "yes" answers that inference supports, only fully committing to their generalisation once they have been asked about the same set again. The one change is this new-set caution: on the first query after a set change the probability of "yes" is the posterior probability that the concept contains the target scaled down by an inferred withholding fraction, while repeated queries about the same set follow the posterior unchanged.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/962b9e6d929c436d/repo/_runs/cell_1/experiment1/model_loop/models/pruned/new_set_cautious_erlang_concepts.py`

### log_number_line_interval_prior — pruned (experiment1 end of experiment)

**Margin:** 6.5 nats behind without_replacement_example_sampling (3.2× dse)

**Hypothesis:** A refinement of inferred_example_noise_concepts: people do the same Bayesian concept learning over the 45 independently weighted rules and all intervals of 1-100, with the size principle and an inferred rate of random examples, but they picture ranges of numbers on a compressed, logarithmic-like mental number line, so a priori they find ranges whose endpoints are small numbers more plausible than equally wide ranges among large numbers. The one change is this magnitude-dependent prior over intervals (the strength of the compression inferred from the data, with zero recovering the incumbent's uniform prior), which predicts that "yes" extends further below the examples than above them.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/962b9e6d929c436d/repo/_runs/cell_1/experiment1/model_loop/models/pruned/log_number_line_interval_prior.py`

### decade_aligned_interval_prior — pruned (experiment1 end of experiment)

**Margin:** 6.9 nats behind without_replacement_example_sampling (2.7× dse)

**Hypothesis:** A refinement of noisy_examples_erlang_intervals: people do the same Bayesian concept learning over the 45 independently weighted rules and all intervals of 1-100 (size principle, an inferred share of random examples, a prior favouring a typical range width), but they think of ranges in round, decade-aligned terms ("20 to 40", "the 60s", "1 to 10"), so a range whose endpoints fall on decade boundaries is a priori more plausible than an equally wide range with arbitrary endpoints such as 23-47. The one change is this preference for decade-aligned range endpoints, with its strength (and sign) inferred from the data, which predicts that "yes" answers to examples that fit no rule extend up to, and stop at, decade lines.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/962b9e6d929c436d/repo/_runs/cell_1/experiment1/model_loop/models/pruned/decade_aligned_interval_prior.py`

### bigelow_piantadosi_independent — pruned (experiment1 end of experiment)

**Margin:** 7.7 nats behind without_replacement_example_sampling (3.2× dse)

**Hypothesis:** Bayesian concept learning over 45 rules (odd, even, primes, squares, cubes, powers of 2-10, powers of 2 without 32 or with 37, multiples of 3-9, 11 and 12, ending in 0-9, containing the digit 0-9) and every interval of 1-100, with a prior weight inferred for each rule separately and a share lambda_0 for the rules as a group; each interval has prior (1 - lambda_0) / 100^2. Examples are sampled from the concept (the size principle), 10% of them uniformly from 1-100, and "yes" is the posterior probability that the concept contains the target.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/962b9e6d929c436d/repo/_runs/cell_1/experiment1/model_loop/models/pruned/bigelow_piantadosi_independent.py`

### erlang_interval_width_prior — pruned (experiment1 end of experiment)

**Margin:** 8.3 nats behind without_replacement_example_sampling (4.0× dse)

**Hypothesis:** Refinement of `bayesian_concepts_with_guessing_lapse`: people judge by Bayesian concept learning over the same 45 independently weighted rules and the intervals of 1-100 (size principle, with a guessing lapse), but they do not consider every range of numbers equally plausible a priori — they expect a "range" concept to have a typical width, favouring intervals of moderate size over both tiny and nearly-all-numbers ranges (an Erlang-shaped prior over interval size, as in Tenenbaum's number game). The one change is this size prior on intervals, with its preferred width inferred from the data, which governs how far people extend "yes" around examples that fit no rule.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/962b9e6d929c436d/repo/_runs/cell_1/experiment1/model_loop/models/pruned/erlang_interval_width_prior.py`

### recency_decay_concept_learning — pruned (experiment1 end of experiment)

**Margin:** 8.6 nats behind without_replacement_example_sampling (2.6× dse)

**Hypothesis:** People do Bayesian concept learning over the same rules and intervals (size principle, independently weighted rule prior), but their memory of the examples fades as they read the set: the most recently shown example counts in full, and each earlier example's evidence is discounted by a constant factor per later example. Because early examples are partly forgotten, concepts that fit the last examples but not the first ones keep substantial posterior weight, so generalisation is driven by the end of the list more than its beginning.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/962b9e6d929c436d/repo/_runs/cell_1/experiment1/model_loop/models/pruned/recency_decay_concept_learning.py`

### bayesian_concepts_with_guessing_lapse — pruned (experiment1 end of experiment)

**Margin:** 9.0 nats behind without_replacement_example_sampling (3.9× dse)

**Hypothesis:** A refinement of bigelow_piantadosi_independent: people still judge by Bayesian concept learning over the same 45 rules and all intervals of 1-100 (independently weighted rule prior, size principle), but on a fraction of trials they do not consult that posterior at all and instead give a guess, saying "yes" with their own fixed guessing rate. The one change is this response-lapse stage, which explains why even targets that are among the examples are often rejected and why clearly excluded targets are sometimes accepted.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/962b9e6d929c436d/repo/_runs/cell_1/experiment1/model_loop/models/pruned/bayesian_concepts_with_guessing_lapse.py`

### posterior_confidence_miscalibration — pruned (experiment1 end of experiment)

**Margin:** 11.4 nats behind without_replacement_example_sampling (3.3× dse)

**Hypothesis:** People do the normative Bayesian concept learning of the incumbent model (45 rules with individually weighted priors and all intervals of 1-100, size principle, 10% random examples), but they mis-weight their own posterior confidence when turning it into a yes/no answer: the log-odds of saying "yes" are the posterior log-odds that the concept contains the target, multiplied by a single confidence-scaling factor. A factor below one means people under-trust their inference (strongly supported generalisations are answered less consistently and strongly excluded targets get more yeses, in a graded way rather than a flat lapse), and a factor above one means they over-commit to the more probable answer.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/962b9e6d929c436d/repo/_runs/cell_1/experiment1/model_loop/models/pruned/posterior_confidence_miscalibration.py`

### offset_arithmetic_progressions — pruned (experiment1 end of experiment)

**Margin:** 22.6 nats behind without_replacement_example_sampling (4.0× dse)

**Hypothesis:** A refinement of noisy_examples_erlang_intervals: people do the same Bayesian concept learning (size principle, inferred share of random examples, Erlang prior on interval width), but their space of rules also includes arithmetic progressions that do not start at zero — every number that leaves a given remainder when divided by 3 to 12 (e.g. 2, 5, 8, 11, ... or 5, 11, 17, 23, ...), each such "step" rule equally likely a priori and the group as a whole given a prior share inferred from the data. The one change is this added family of offset arithmetic-progression concepts, which predicts that examples sharing a step-and-offset pattern (e.g. 23, 35, 83, 95, all 5 more than a multiple of 6) are generalised to that narrow progression rather than to a broad rule such as odd numbers.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/962b9e6d929c436d/repo/_runs/cell_1/experiment1/model_loop/models/pruned/offset_arithmetic_progressions.py`

### noise_coupled_yes_floor — pruned (experiment1 end of experiment)

**Margin:** 33.4 nats behind without_replacement_example_sampling (5.4× dse)

**Hypothesis:** A refinement of noisy_examples_erlang_intervals: people do the same Bayesian concept learning (45 independently weighted rules, all intervals of 1-100 with an Erlang prior on range width, size principle, inferred share of random examples), but they answer "would the program produce this number?" with the same generative model they used to explain the examples. Since they believe the program emits a random number from 1-100 on some share of its outputs, they say "yes" whenever the target is in the inferred concept or when they think the program might be emitting a random number, so the one change is that the inferred noise rate also sets a floor of "yes" for targets outside every plausible concept (no new free parameter, unlike a free guessing lapse).

**Source:** `/scr/wurgaft/auto-psych/agent_trees/962b9e6d929c436d/repo/_runs/cell_1/experiment1/model_loop/models/pruned/noise_coupled_yes_floor.py`

### noisy_posterior_criterion — pruned (experiment1 end of experiment)

**Margin:** 37.5 nats behind without_replacement_example_sampling (5.7× dse)

**Hypothesis:** People do the same Bayesian concept learning as bigelow_piantadosi_independent (45 individually weighted rules and all intervals of 1-100, size principle, 10% random examples), but they answer "yes" by comparing their posterior belief that the target belongs to the concept against an internal criterion of how sure they must be, with noise in that comparison. The criterion (which need not be 50%) and the comparison noise are inferred; because the comparison is made on the probability itself, confident beliefs at either extreme give similar, non-certain answers, and beliefs below the criterion are mostly refused.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/962b9e6d929c436d/repo/_runs/cell_1/experiment1/model_loop/models/pruned/noisy_posterior_criterion.py`

### win_stay_lose_shift_concepts — pruned (experiment1 end of experiment)

**Margin:** 64.9 nats behind without_replacement_example_sampling (6.6× dse)

**Hypothesis:** People do not hold a full posterior over concepts; they keep a single running guess and revise it example by example (win-stay, lose-shift). After the first example they pick a concept (rule or range) in proportion to its Bayesian posterior given that example; each later example that fits the current guess leaves it unchanged, and only an example that does not fit makes them abandon it and re-pick a concept from the posterior given all examples seen so far. "Yes" is the probability that the guess they end with contains the target, so the size principle tightens generalisation only when a later example forces a switch, and the order of the examples changes which guess they end up holding.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/962b9e6d929c436d/repo/_runs/cell_1/experiment1/model_loop/models/pruned/win_stay_lose_shift_concepts.py`

### non_accumulating_size_principle — pruned (experiment1 end of experiment)

**Margin:** 88.2 nats behind without_replacement_example_sampling (5.7× dse)

**Hypothesis:** People do Bayesian concept learning over the same rules and intervals, but they do not accumulate the size-principle evidence across examples: they judge how well a concept fits the examples by the average fit of one example, sharpened by a fixed amount, so their generalisation is no tighter after four examples than after one. This disagrees most with the current best model on sets with one or two examples, where it predicts sharper, more rule-like generalisation, and on four-example sets, where it predicts broader, more graded generalisation that keeps giving partial "yes" to targets that fit only a looser concept (for example, odd numbers that are not in a narrow rule the examples happen to share).

**Source:** `/scr/wurgaft/auto-psych/agent_trees/962b9e6d929c436d/repo/_runs/cell_1/experiment1/model_loop/models/pruned/non_accumulating_size_principle.py`

### arithmetic_vs_digit_rule_grammar — pruned (experiment1 end of experiment)

**Margin:** 108.4 nats behind without_replacement_example_sampling (6.0× dse)

**Hypothesis:** Refinement of `bigelow_piantadosi_mixture`: people do Bayesian concept learning over the same 45 rules and all intervals of 1-100 (size principle, 10% of examples random), but their prior over rules is not flat. It first chooses between two kinds of rule, arithmetic properties of the number (odd, even, primes, squares, cubes, powers, multiples) or patterns in how it is written in decimal (ends in a digit, contains a digit), and then picks uniformly within that kind. The one change is this two-kind grammar step, with the share of prior given to written-digit rules inferred from the data rather than fixed by how many such rules there are.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/962b9e6d929c436d/repo/_runs/cell_1/experiment1/model_loop/models/pruned/arithmetic_vs_digit_rule_grammar.py`

### bigelow_piantadosi_mixture — pruned (experiment1 end of experiment)

**Margin:** 120.5 nats behind without_replacement_example_sampling (7.3× dse)

**Hypothesis:** Bayesian concept learning over 45 rules (odd, even, primes, squares, cubes, powers of 2-10, powers of 2 without 32 or with 37, multiples of 3-9, 11 and 12, ending in 0-9, containing the digit 0-9) and every interval of 1-100. A rule is chosen with probability lambda_0, and the rules are equally likely; each interval has prior (1 - lambda_0) / 100^2. Examples are sampled from the concept (the size principle), 10% of them uniformly from 1-100, and "yes" is the posterior probability that the concept contains the target.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/962b9e6d929c436d/repo/_runs/cell_1/experiment1/model_loop/models/pruned/bigelow_piantadosi_mixture.py`

### weak_sampling_concept_learning — pruned (experiment1 end of experiment)

**Margin:** 179.7 nats behind without_replacement_example_sampling (8.8× dse)

**Hypothesis:** People infer which concept (one of the 45 arithmetic or digit rules, or an interval of 1-100) produced the examples, but they use weak sampling rather than the size principle: a concept is judged only by whether it is consistent with the examples, not by how small it is, so a broad concept like "even numbers" that contains the examples is exactly as well supported as a narrow one like "powers of 2". Their answer is the posterior probability that the concept contains the target, which predicts generalisation that stays broad and does not tighten as more consistent examples are seen.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/962b9e6d929c436d/repo/_runs/cell_1/experiment1/model_loop/models/pruned/weak_sampling_concept_learning.py`

### exemplar_feature_similarity — pruned (experiment1 end of experiment)

**Margin:** 325.1 nats behind without_replacement_example_sampling (12.0× dse)

**Hypothesis:** People do not infer a rule at all; they judge a target by its summed similarity to the remembered examples (exemplar generalization), where similarity between two numbers falls off exponentially with how many simple arithmetic properties they disagree on (parity, divisibility by 3-10, last digit, tens digit, being prime, square or a power of two) plus how far apart they are in magnitude. A target that closely resembles even one example is accepted, with no size-principle penalty and no requirement that it fit all examples jointly.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/962b9e6d929c436d/repo/_runs/cell_1/experiment1/model_loop/models/pruned/exemplar_feature_similarity.py`

### exemplar_attention_similarity — pruned (experiment1 end of experiment)

**Margin:** 341.6 nats behind without_replacement_example_sampling (12.4× dse)

**Hypothesis:** People do not infer a rule; they judge the target by its summed similarity to the stored examples (exemplar-based generalization, as in the Generalized Context Model). Each number is represented on a few psychological dimensions — magnitude, parity, final digit and tens digit — each with its own attention weight, similarity to an example decays exponentially with attention-weighted mismatch, and the probability of "yes" rises with the target's average similarity to the examples.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/962b9e6d929c436d/repo/_runs/cell_1/experiment1/model_loop/models/pruned/exemplar_attention_similarity.py`

### spacing_scaled_magnitude_similarity — pruned (experiment1 end of experiment)

**Margin:** 348.7 nats behind without_replacement_example_sampling (12.6× dse)

**Hypothesis:** People judge a target by its numerical closeness to the nearest example on the 1-100 number line, with no rule inference: the chance of "yes" falls off smoothly with the distance from the target to the closest example. The width of this generalization gradient is set by how spread out the examples are — the mean gap between neighbouring examples (plus a baseline width for single examples) — so widely spaced examples license "yes" to far-away targets, while tightly clustered examples restrict "yes" to targets right beside them.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/962b9e6d929c436d/repo/_runs/cell_1/experiment1/model_loop/models/pruned/spacing_scaled_magnitude_similarity.py`

### exemplar_consequential_region_similarity — pruned (experiment1 end of experiment)

**Margin:** 378.3 nats behind without_replacement_example_sampling (10.9× dse)

**Hypothesis:** People do not infer one concept that covers all the examples together; they store each example as a separate exemplar and judge the target by its average similarity to the stored exemplars. The similarity of the target to one exemplar is Shepard-style generalization through consequential regions: how likely a number concept (an arithmetic or digit rule, or a range of a typical width) that contains that single exemplar also contains the target, with smaller concepts counting more. Because each exemplar is generalized from on its own and the results are summed, a target resembling only some of the examples gets partial "yes", and evidence never tightens jointly across examples as it would under concept inference.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/962b9e6d929c436d/repo/_runs/cell_1/experiment1/model_loop/models/pruned/exemplar_consequential_region_similarity.py`
