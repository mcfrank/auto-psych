# Refinement menu

The models you may refine, other than the incumbent `decade_aligned_range_concepts`. Each entry is one mechanism: its hypothesis in full as its author stated it, how it stands, and where its source is. This is a menu, not a list of what is ruled out: a pruned model lost to the best model on the data it was scored on, by the stated margin, but its mechanism may be partly right — and a model that lost narrowly is the most promising target here. Read the source of the model you pick before you write anything.

## Live models other than the incumbent (in the set; best first)

### decade_concept_bayesian_learning — rank 1, 4.1 ± 6.3 nats behind the best (0.6× dse: statistically tied with the best), ELPD-LOO -1595.9

**Hypothesis:** People do the same conservative Bayesian concept learning as the best model (42 arithmetic rules and midpoint-weighted intervals of 1-100, evidence weight growing with set size, attention lapses), but they also hold decades — "numbers in the 20s", "numbers in the 60s" (same tens digit) — as named concepts with their own inferred prior share, split equally across the ten decades. Because a decade is a privileged, a-priori-likely range rather than one of thousands of arbitrary intervals, a target just outside the examples' range is endorsed more when it shares their tens decade than when an equally close target crosses a decade boundary.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/80aefb383357367b/repo/_runs/cell_1/experiment1/model_loop/models/decade_concept_bayesian_learning.py`

### decade_concepts_lapsing_learner — rank 2, 4.4 ± 5.8 nats behind the best (0.7× dse: statistically tied with the best), ELPD-LOO -1596.2

**Hypothesis:** Refinement of `lapsing_growing_evidence_concepts`: people do the same conservative Bayesian concept learning (42 arithmetic rules plus midpoint-weighted intervals, evidence weight growing with set size, blind "yes"/"no" lapses), but they also treat the decades of the written number system (1-9, 10-19, ..., 90-99) as privileged, ready-made concepts with their own inferred prior share, over and above the same ranges' standing as ordinary intervals. The one change is this decade hypothesis family, which predicts that a nearby target outside the examples' range is endorsed more when it shares the examples' tens digit than when it crosses into the next decade — the same-decade bonus the incumbent's critique flags.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/80aefb383357367b/repo/_runs/cell_1/experiment1/model_loop/models/decade_concepts_lapsing_learner.py`

### decade_aligned_interval_concepts — rank 3, 6.1 ± 3.8 nats behind the best (1.6× dse: statistically tied with the best), ELPD-LOO -1597.9

**Hypothesis:** Refinement of `lapsing_growing_evidence_concepts`: people do the same conservative Bayesian concept learning over the 42 arithmetic rules and midpoint-weighted intervals of 1-100, with evidence weight growing with set size and attention lapses, but they mentally carve the number line into decades, so number ranges whose edges fall on decade boundaries (e.g. 20-29, 30-59, 1-10) are a priori more plausible than arbitrary ranges by an inferred factor. The one change is this decade-aligned prior bonus on intervals, which predicts that a nearby target outside the examples' span is endorsed more often when it shares a decade with an example (25 after 21, 23) than when it lies just across a decade boundary (31 after 27, 29) — the same-decade bonus the incumbent's critique flags.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/80aefb383357367b/repo/_runs/cell_1/experiment1/model_loop/models/decade_aligned_interval_concepts.py`

### lapsing_growing_evidence_concepts — rank 4, 6.9 ± 5.8 nats behind the best (1.2× dse: statistically tied with the best), ELPD-LOO -1598.7

**Hypothesis:** Refinement of `sample_size_growing_evidence_concepts`: people do the same conservative Bayesian concept learning over the 42 arithmetic rules and midpoint-weighted intervals of 1-100, with evidence weight growing with set size, but on some trials they answer without consulting their concept at all, guessing "yes" or "no" with their own inferred rates. The one change is this attention-lapse response floor and ceiling (an inferred chance of a blind "yes" and a separate chance of a blind "no"), which predicts a steady baseline of "yes" to targets the concept clearly excludes and occasional "no" to targets it clearly contains, where the incumbent puts both at near-certainty.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/80aefb383357367b/repo/_runs/cell_1/experiment1/model_loop/models/lapsing_growing_evidence_concepts.py`

### sample_size_growing_evidence_concepts — rank 5, 11.6 ± 4.3 nats behind the best (2.7× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1603.4

**Hypothesis:** Refinement of `conservative_evidence_interval_concepts`: people do the same conservative Bayesian concept learning over the 42 arithmetic rules and midpoint-weighted intervals of 1-100, but how much weight each example's evidence receives grows with the number of examples shown, because a single example gives little reason to believe it was sampled from a narrow concept, while several examples that all fit a tight concept convince people the examples are "strong" samples. The one change is that the evidence-weighting strength is a power function of the set size (an inferred growth exponent) rather than one constant, which predicts broader generalization from a single example than the incumbent, the discrepancy its critique flags.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/80aefb383357367b/repo/_runs/cell_1/experiment1/model_loop/models/sample_size_growing_evidence_concepts.py`

### soft_criterion_concept_decision — rank 6, 18.4 ± 11.8 nats behind the best (1.6× dse: statistically tied with the best), ELPD-LOO -1610.2

**Hypothesis:** People infer the hidden concept by plain Bayesian concept learning over arithmetic rules and numeric intervals (the simplest current inference, with the size principle), but they do not report that belief by probability matching: they answer "yes" when their confidence that the target belongs clears a personal criterion, applied noisily. How high the criterion sits (a bias toward "yes" or "no") and how sharply confidence is turned into a choice are the only changes, so moderately believed targets are pushed toward one answer and near-certain beliefs are softened toward the other.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/80aefb383357367b/repo/_runs/cell_1/experiment1/model_loop/models/soft_criterion_concept_decision.py`

### iter1_candidate4 — rank 7, 18.9 ± 7.9 nats behind the best (2.4× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1610.7

**Hypothesis:** Refinement of `conservative_evidence_interval_concepts`: people do the same conservative Bayesian concept learning over the 42 arithmetic rules and every interval of 1-100 (inferred rule weights, a Beta midpoint prior on intervals, size-principle sampling with 10% noise, examples' evidence weighted by an inferred strength), but they also hold a prior preference for intervals of intermediate size (Tenenbaum's Erlang size prior, scale inferred, grafted from `bp_rules_erlang_interval_size`) that multiplies the midpoint prior. The one change is this size prior, which discounts tiny near-singleton intervals a priori and so predicts broader generalization from a single example — where the incumbent says "yes" too rarely — while leaving multi-example sets, whose evidence already rules out tiny intervals, largely unchanged.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/80aefb383357367b/repo/_runs/cell_1/experiment1/model_loop/models/iter1_candidate4.py`

### first_example_weak_erlang_concepts — rank 8, 21.8 ± 10.8 nats behind the best (2.0× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1613.6

**Hypothesis:** Refinement of `first_example_weak_sampling_concepts`: people update their concept one example at a time, and the first example only rules out concepts that do not contain it, while every later example is treated as drawn from the concept (size principle, weighted by an inferred strength) — but people also hold a prior preference for number ranges of intermediate size (Tenenbaum's Erlang size prior on intervals, scale inferred, grafted from `iter1_candidate4`). The one change is this size prior: because a single example carries no size-principle evidence under this account, what people generalize to from one example is set entirely by their prior, and the size prior predicts a moderately wide range around a lone example rather than the broad ranges near 1-100 that a size-blind prior favours.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/80aefb383357367b/repo/_runs/cell_1/experiment1/model_loop/models/first_example_weak_erlang_concepts.py`

### first_example_weak_sampling_concepts — rank 9, 24.7 ± 11.1 nats behind the best (2.2× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1616.5

**Hypothesis:** People update their concept one example at a time, and the size principle only kicks in once there is a coincidence to explain: the first example merely rules out concepts that do not contain it (any consistent concept stays as plausible as its prior says), while every later example is treated as drawn from the concept, so small concepts that keep fitting become favoured ("suspicious coincidence"). This predicts broad generalization from a single example and a sharp tightening only when a second consistent example arrives.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/80aefb383357367b/repo/_runs/cell_1/experiment1/model_loop/models/first_example_weak_sampling_concepts.py`

### conservative_evidence_interval_concepts — rank 10, 26.7 ± 8.7 nats behind the best (3.1× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1618.5

**Hypothesis:** Refinement of `bigelow_piantadosi_interval`: people do the same Bayesian concept learning over 42 arithmetic rules and every interval of 1-100 (a prior weight inferred for each rule, an interval prior given by a Beta density over its midpoint, examples sampled from the concept by the size principle with 10% noise), but they are conservative updaters: they weight the evidence of the examples by a single inferred strength rather than at full Bayesian value, so the posterior over concepts is flatter (strength below 1) or sharper (above 1) than the ideal learner's. The one change is this evidence-weighting exponent on the examples' likelihood, which governs how quickly generalization tightens around the narrowest consistent concept as examples accumulate.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/80aefb383357367b/repo/_runs/cell_1/experiment1/model_loop/models/conservative_evidence_interval_concepts.py`

### recency_decayed_evidence_concepts — rank 11, 31.7 ± 8.8 nats behind the best (3.6× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1623.5

**Hypothesis:** People learn the program's concept by conservative Bayesian concept learning over arithmetic rules and numeric intervals (size principle), but their memory of the examples fades with how long ago each was shown: the last example counts as full evidence, and each example further back in the sequence counts as progressively weaker evidence, by a constant inferred retention factor per position. Generalization is therefore driven mostly by the most recent examples, so reordering the same examples changes which targets get a "yes" (targets that fit only the first example are endorsed less often than targets that fit only the last).

**Source:** `/scr/wurgaft/auto-psych/agent_trees/80aefb383357367b/repo/_runs/cell_1/experiment1/model_loop/models/recency_decayed_evidence_concepts.py`

### size_principle_strength_intervals — rank 12, 53.6 ± 11.4 nats behind the best (4.7× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1645.4

**Hypothesis:** Refinement of bigelow_piantadosi_interval: people use the same Bayesian concept learning over the 42 rules and the midpoint-weighted intervals of 1-100, but they apply the size principle only partially — how strongly a small concept is favoured for fitting the examples tightly is a graded, inferred strength, between weak sampling (examples merely consistent with a concept, no preference for small concepts) and the full strong sampling the incumbent assumes. The one change is this inferred size-principle strength, which predicts broader generalization than the incumbent (e.g. more "yes" to other even numbers after 16, 8, 64) if people underweight concept size.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/80aefb383357367b/repo/_runs/cell_1/experiment1/model_loop/models/size_principle_strength_intervals.py`

### digit_place_similarity_gradient — rank 13, 55.6 ± 20.9 nats behind the best (2.7× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1647.4

**Hypothesis:** People judge a target by its similarity to the examples in their written decimal form rather than on the number line: they compare the target with each example place by place, and every mismatch in the units digit or in the tens digit multiplies similarity down by its own inferred factor. The more similar the target is on average to the examples under this digit-place metric, the more likely people are to say "yes" — so 21 after 91, or 58 after 28 and 98, is endorsed despite being numerically far away, while 88 after 89 is rejected despite being adjacent.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/80aefb383357367b/repo/_runs/cell_1/experiment1/model_loop/models/digit_place_similarity_gradient.py`

### shared_property_overlap — rank 14, 64.6 ± 18.6 nats behind the best (3.5× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1656.4

**Hypothesis:** People do not weigh concepts by how tightly they fit the examples (no size principle); instead they note every salient number property that all the examples share (parity, final digit, decade, contained digit, prime, square, cube, power-of-k, multiple-of-k) and say "yes" more often the larger the fraction of those shared properties the target also has, with every shared property counting equally. This disagrees most sharply with the best model on targets that share the examples' broad properties but not their narrowest one (e.g. 14 after 28, 98: even but not ending in 8), which this account rates as half-matching while the size-principle model gives them almost no generalization.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/80aefb383357367b/repo/_runs/cell_1/experiment1/model_loop/models/shared_property_overlap.py`

### exemplar_feature_similarity_gcm — rank 15, 73.2 ± 21.0 nats behind the best (3.5× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1665.0

**Hypothesis:** People do not induce a rule; they judge the target by its similarity to the stored examples (exemplar generalization, as in the Generalized Context Model). Each number is represented by its magnitude and its salient arithmetic/digit properties (parity, primality, square, cube, multiples, last and tens digit), and the more similar the target is on average to the examples — similarity decaying exponentially with a weighted distance in this space — the more likely people are to say yes, with no hypothesis space, posterior, or size principle involved.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/80aefb383357367b/repo/_runs/cell_1/experiment1/model_loop/models/exemplar_feature_similarity_gcm.py`

### coherence_attention_exemplar_similarity — rank 16, 97.3 ± 22.4 nats behind the best (4.3× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1689.1

**Hypothesis:** People do not infer a concept; they judge the target by its similarity to the stored examples (exemplar generalization), but their attention is drawn to the dimensions on which the stored examples cohere: every property all the examples share (parity, final digit, decade, contained digit, prime, square, cube, power-of-k, multiple-of-k) gets attention in proportion to how distinctive it is in 1-100, so a target that lacks a rare shared property (e.g. "ends in 8", "in the 20s") is judged dissimilar to every example, while closeness on the number line counts on a scale that stretches with the examples' spread. Saying "yes" is a Luce choice between the target's mean similarity to the examples and a fixed criterion.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/80aefb383357367b/repo/_runs/cell_1/experiment1/model_loop/models/coherence_attention_exemplar_similarity.py`

### transformational_distance_similarity — rank 17, 130.6 ± 23.6 nats behind the best (5.5× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1722.4

**Hypothesis:** People do not induce a concept or measure closeness on the number line; they judge whether the target can be made from one of the examples by a short chain of simple mental operations on numbers (add or subtract one, add or subtract ten, double, halve, reverse the digits), so the fewer operations it takes to turn the examples into the target, the more likely they say "yes" (transformational similarity). This predicts "yes" to 32 after 16, to 61 after 16, and to 40 after 20 even though they are far away on the number line, while a numerically moderate jump that needs many steps (e.g. 47 after 20) is rejected.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/80aefb383357367b/repo/_runs/cell_1/experiment1/model_loop/models/transformational_distance_similarity.py`

### range_scaled_magnitude_gradient — rank 18, 131.0 ± 23.9 nats behind the best (5.5× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1722.8

**Hypothesis:** People do not consider arithmetic rules or properties at all; they judge a target purely by its closeness on the number line to the nearest example, with a generalization gradient whose width stretches with the spread of the examples. A single example (no spread) still supports a fixed baseline width of generalization around it, so "yes" falls off smoothly with numerical distance, more slowly when the examples cover a wide range.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/80aefb383357367b/repo/_runs/cell_1/experiment1/model_loop/models/range_scaled_magnitude_gradient.py`

### bp_rules_erlang_interval_size — rank 19, 182.8 ± 28.6 nats behind the best (6.4× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1774.6

**Hypothesis:** Refinement of `bigelow_piantadosi_independent`: people do Bayesian concept learning over the same 42 rules (each with its own inferred prior weight) and every interval of 1-100, with examples sampled from the concept (the size principle, 10% noise), and answer "yes" with the posterior probability that the concept contains the target. The one change is that intervals are not equally likely a priori: people prefer intervals of intermediate size, with an Erlang prior on an interval's size whose scale is inferred (Tenenbaum's size prior grafted onto the independent-rule-weight model), so a few nearby examples generalize to a moderately wide range rather than to the very tightest or the very widest interval.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/80aefb383357367b/repo/_runs/cell_1/experiment1/model_loop/models/bp_rules_erlang_interval_size.py`

### bigelow_piantadosi_interval — rank 20, 202.3 ± 28.7 nats behind the best (7.1× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1794.1

**Hypothesis:** Bayesian concept learning over 42 rules (powers of 2-10, multiples of 3-12, final digit 1-9, contained digit 1-9, primes, even, odd, squares, cubes) and every interval of 1-100, with a prior weight inferred for each rule separately and a share p_mix for the rules as a group. An interval's prior follows a Beta density over its midpoint, with both shapes inferred, so concepts in some parts of 1-100 are favoured. Examples are sampled from the concept (the size principle), 10% of them uniformly from 1-100, and "yes" is the posterior probability that the concept contains the target.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/80aefb383357367b/repo/_runs/cell_1/experiment1/model_loop/models/bigelow_piantadosi_interval.py`

### bigelow_piantadosi_independent — rank 21, 208.4 ± 31.6 nats behind the best (6.6× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1800.2

**Hypothesis:** Bayesian concept learning over 42 rules (powers of 2-10, multiples of 3-12, final digit 1-9, contained digit 1-9, primes, even, odd, squares, cubes) and every interval of 1-100, with a prior weight inferred for each rule separately and a share p_mix for the rules as a group; intervals are equally likely. Examples are sampled from the concept (the size principle), 10% of them uniformly from 1-100, and "yes" is the posterior probability that the concept contains the target.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/80aefb383357367b/repo/_runs/cell_1/experiment1/model_loop/models/bigelow_piantadosi_independent.py`

### tenenbaum_rules_similarity — rank 22, 295.2 ± 45.7 nats behind the best (6.5× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1887.0

**Hypothesis:** Bayesian concept learning over 33 mathematical rules (odd, even, squares, cubes, primes, multiples of 3-12, powers of 2-10, ending in 1-9) and every interval of 1-100. The rules share a prior fraction lambda equally; the intervals share the rest by an Erlang density on their size (scale sigma). Examples are sampled from the concept (the size principle), 10% of them uniformly from 1-100, and "yes" is the posterior probability that the concept contains the target.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/80aefb383357367b/repo/_runs/cell_1/experiment1/model_loop/models/tenenbaum_rules_similarity.py`

## Pruned models (out of the set; narrowest margin first)

No model has been pruned yet in this project.
