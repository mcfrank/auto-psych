# Refinement menu

The models you may refine, other than the incumbent `monotone_order_range_prior_shift`. Each entry is one mechanism: its hypothesis in full as its author stated it, how it stands, and where its source is. This is a menu, not a list of what is ruled out: a pruned model lost to the best model on the data it was scored on, by the stated margin, but its mechanism may be partly right — and a model that lost narrowly is the most promising target here. Read the source of the model you pick before you write anything.

## Live models other than the incumbent (in the set; best first)

### parity_restricted_concepts — rank 1, 4.1 ± 10.9 nats behind the best (0.4× dse: statistically tied with the best), ELPD-LOO -1553.4

**Hypothesis:** Refinement of `tempered_size_principle_independent`: people do the same tempered-evidence Bayesian concept learning over the 45 individually weighted rules and all intervals of 1-100, but parity is a basic feature they can conjoin with any concept, so they also entertain the odd-only and even-only version of each rule and interval (e.g. "odd multiples of 3", "even numbers between 4 and 96"), which together receive an inferred share of prior belief. The one change is adding these parity-restricted concepts (share 0 recovers the incumbent); because such a concept is half the size, the size principle favours it whenever all examples share a parity, so people reject opposite-parity targets — including far, weakly supported ones outside the examples' range — more firmly than the incumbent predicts.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/8a52ff3d6e8e6ef7/repo/_runs/cell_1/experiment1/model_loop/models/parity_restricted_concepts.py`

### decade_place_value_concepts — rank 2, 4.6 ± 10.5 nats behind the best (0.4× dse: statistically tied with the best), ELPD-LOO -1553.9

**Hypothesis:** Refinement of `parity_restricted_concepts`: people do the same tempered-evidence Bayesian concept learning over the 45 weighted rules, all intervals of 1-100, and their odd-only and even-only versions. The one change is that people also read numbers by place value, so they entertain decade concepts: "numbers whose tens digit is one of these", e.g. "numbers in the 20s, 30s or 60s". These concepts get an inferred share of prior belief (share 0 recovers the incumbent), and the size principle favours the smallest one, the decades the examples occupy. This predicts that a target with a different final digit is endorsed more when an example sits in its decade (62,32,22 → 25; 73,33,3 → 36) than when none does (63,23,43 → 35), whatever order the examples are shown in.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/8a52ff3d6e8e6ef7/repo/_runs/cell_1/experiment1/model_loop/models/decade_place_value_concepts.py`

### serial_position_weighted_evidence — rank 3, 5.1 ± 5.0 nats behind the best (1.0× dse: statistically tied with the best), ELPD-LOO -1554.5

**Hypothesis:** Refinement of `tempered_size_principle_independent`: people do the same tempered Bayesian concept learning over the 45 individually weighted rules and all intervals of 1-100, but they do not weigh every example equally as evidence — how much an example counts depends on its serial position in the list, with evidence weight shifting smoothly toward the first examples (primacy) or the last ones (recency) by an inferred serial-position gradient, while the total evidence strength stays the same. The one change is this order-dependent weighting (a zero gradient recovers the incumbent); it predicts that the same examples shown in a different order yield different generalization, and in particular that an example that breaks the pattern of the others (e.g. an odd number among evens) undermines the tight concept more when it sits at the favoured end of the list.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/8a52ff3d6e8e6ef7/repo/_runs/cell_1/experiment1/model_loop/models/serial_position_weighted_evidence.py`

### place_value_context_exemplar — rank 4, 32.7 ± 22.9 nats behind the best (1.4× dse: statistically tied with the best), ELPD-LOO -1582.1

**Hypothesis:** People do not infer a rule; they judge a target by its summed similarity to the stored examples, as in the context (exemplar) model of classification, where each number is represented by its place-value digits — its tens digit and its units digit — together with its parity and its magnitude, and similarity to an example shrinks multiplicatively with each mismatching dimension, with an inferred attention weight per dimension. "Yes" grows with the target's total similarity to the examples relative to an internal criterion, so a target that shares its tens digit with one example (e.g. 25 after 62, 32, 22) gets partial endorsement even when it breaks the examples' shared final digit, whereas the same target is rejected when no example lies in its decade.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/8a52ff3d6e8e6ef7/repo/_runs/cell_1/experiment1/model_loop/models/place_value_context_exemplar.py`

### iter2_candidate1 — rank 5, 45.0 ± 26.6 nats behind the best (1.7× dse: statistically tied with the best), ELPD-LOO -1594.3

**Hypothesis:** People judge a target by Shepard-style exponential generalization in a psychological space whose dimensions are weighted by selective attention, and the examples set that attention: they attend to the properties the examples agree on (more strongly the more consistently the examples share a property and the more distinctive it is), and to magnitude in proportion to how tightly the examples cluster. The probability of "yes" falls off exponentially with the target's attention-weighted distance from the examples, so missing a consistently shared, distinctive property (e.g. the common last digit) or lying well outside a tight example range sharply lowers endorsement, while properties the examples do not agree on barely matter.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/8a52ff3d6e8e6ef7/repo/_runs/cell_1/experiment1/model_loop/models/iter2_candidate1.py`

### tempered_size_principle_independent — rank 6, 50.3 ± 13.4 nats behind the best (3.7× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1599.7

**Hypothesis:** Refinement of `bigelow_piantadosi_independent`: people do the same Bayesian concept learning over 45 rules (each with its own prior weight) and all intervals, but they under-trust the size principle — they treat the examples as weaker evidence than a fully rational learner would, as if the likelihood of the examples were tempered by an inferred exponent (sampling-assumption strength), so their posterior is less concentrated on the smallest consistent rule. The one change is this free evidence-strength exponent (1 recovers the original model); it predicts lower endorsement of minimal-rule members and graded endorsement among them, because members also covered by competing concepts (nearby intervals, other rules) keep more support.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/8a52ff3d6e8e6ef7/repo/_runs/cell_1/experiment1/model_loop/models/tempered_size_principle_independent.py`

### sampled_hypothesis_majority_vote — rank 7, 53.3 ± 15.2 nats behind the best (3.5× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1602.7

**Hypothesis:** Refinement of tempered_size_principle_independent: people do the same tempered Bayesian concept learning over the 45 rules and all intervals, but they cannot hold the whole posterior in mind — they bring a small number of candidate concepts to mind by sampling them from their posterior and answer "yes" if the majority of those sampled concepts contain the target. The one change is this limited-sample majority vote (with an inferred number of samples; one sample recovers the original model), which makes answers more all-or-none than the posterior itself: likely members are endorsed more consistently and weakly supported targets, such as numbers beyond the range of the examples, are endorsed less.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/8a52ff3d6e8e6ef7/repo/_runs/cell_1/experiment1/model_loop/models/sampled_hypothesis_majority_vote.py`

### sampled_concepts_criterion_vote — rank 8, 55.9 ± 15.9 nats behind the best (3.5× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1605.2

**Hypothesis:** Refinement of `sampled_hypothesis_majority_vote`: people do the same tempered Bayesian concept learning over the 45 individually weighted rules and all intervals, and answer by bringing a few candidate concepts to mind by sampling them from their posterior — but they say "yes" only if a criterion number of the sampled concepts contains the target, and that criterion is inferred rather than fixed at a simple majority (equivalently, they set how many sampled concepts must favour the target and how many may go against it). The one change is this free decision criterion (equal counts recover majority vote, a single sample recovers the incumbent); a demanding criterion predicts that targets supported only by a minority of concepts — such as numbers far outside the examples' range that fit no small rule — are rejected more firmly than majority vote allows, while targets in the dominant concept are still endorsed.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/8a52ff3d6e8e6ef7/repo/_runs/cell_1/experiment1/model_loop/models/sampled_concepts_criterion_vote.py`

### compact_interval_tempered_bayes — rank 9, 60.0 ± 16.8 nats behind the best (3.6× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1609.4

**Hypothesis:** Refinement of `tempered_size_principle_independent`: people do the same tempered-evidence Bayesian concept learning over the 45 rules (each with its own prior weight) and all intervals of 1-100, but they do not consider all intervals equally plausible a priori — they prefer compact magnitude ranges, with an Erlang prior over interval size whose scale is inferred (the one change, taken from `tenenbaum_rules_similarity`). Because the tempered evidence no longer spreads interval belief onto wide ranges, people confine magnitude-based generalization close to the [min, max] range of the examples and endorse targets outside it less than the incumbent predicts.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/8a52ff3d6e8e6ef7/repo/_runs/cell_1/experiment1/model_loop/models/compact_interval_tempered_bayes.py`

### hierarchical_tempered_evidence — rank 10, 64.1 ± 15.4 nats behind the best (4.2× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1613.5

**Hypothesis:** Refinement of `tempered_size_principle_independent`: people under-trust the examples only when deciding *which kind* of concept and which rule produced them, but once they entertain a magnitude range they apply the size principle at full strength, so the range they imagine stays tight around the examples. The one change is where the inferred evidence-strength exponent applies: it tempers the evidence for each rule and for the interval family as a whole, while belief among intervals follows the untempered size principle; this predicts firmer rejection of targets far outside the examples' range that no small rule covers, while keeping graded endorsement of rule members.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/8a52ff3d6e8e6ef7/repo/_runs/cell_1/experiment1/model_loop/models/hierarchical_tempered_evidence.py`

### size_penalized_interval_prior — rank 11, 66.2 ± 17.0 nats behind the best (3.9× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1615.6

**Hypothesis:** Refinement of `tempered_size_principle_independent`: people do the same tempered Bayesian concept learning over 45 individually weighted rules and all intervals of 1-100, but a priori they expect a magnitude-based concept to be a narrow range rather than treating all intervals as equally likely — their prior on an interval falls off as a power of its width, with the steepness inferred. The one change is this size-penalizing interval prior (steepness 0 recovers a uniform interval prior); it predicts that people confine generalization more tightly to the range spanned by the examples, and endorse targets outside that range less, than the incumbent allows.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/8a52ff3d6e8e6ef7/repo/_runs/cell_1/experiment1/model_loop/models/size_penalized_interval_prior.py`

### conservative_evidence_concept_learning — rank 12, 87.3 ± 22.8 nats behind the best (3.8× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1636.6

**Hypothesis:** Refinement of `bigelow_piantadosi_mixture`: people do the same Bayesian concept learning over the 45 rules and all intervals of 1-100 with the size principle, but they under-weight the evidence of the examples, as if each example counted for only a fraction of a full observation (a conservative, tempered likelihood). The one change is a free evidence-weight exponent on the likelihood of the examples, so confidence in the smallest consistent rule grows more slowly with examples and more posterior stays on broader rules and intervals — making learners less sure of the winning rule than the ideal learner.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/8a52ff3d6e8e6ef7/repo/_runs/cell_1/experiment1/model_loop/models/conservative_evidence_concept_learning.py`

### domain_specific_evidence_weight — rank 13, 93.6 ± 23.2 nats behind the best (4.0× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1642.9

**Hypothesis:** Refinement of `conservative_evidence_concept_learning`: people do the same Bayesian concept learning over the 45 equally weighted rules and all intervals of 1-100 with the size principle, but how strongly they weigh the examples as evidence depends on the kind of concept — they are conservative about mathematical rules yet apply the size principle sharply to magnitude ranges, so once they think "a range of numbers" they confine it tightly to the span of the examples. The one change is splitting the single evidence-weight exponent into one for rule hypotheses and one for interval hypotheses; it predicts less endorsement of targets just outside the examples' [min, max] range (relative to inside it) than a single shared tempering allows, while keeping graded endorsement of rule members.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/8a52ff3d6e8e6ef7/repo/_runs/cell_1/experiment1/model_loop/models/domain_specific_evidence_weight.py`

### log_odds_belief_readout — rank 14, 99.9 ± 23.0 nats behind the best (4.3× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1649.3

**Hypothesis:** Refinement of bigelow_piantadosi_mixture: people do the same Bayesian concept learning over the 45 rules and all intervals of 1-100 with the size principle, but they do not answer "yes" with probability equal to their belief that the concept contains the target; instead they read out that belief through a distorted, linear-in-log-odds probability weighting (as in judgments of probability and decisions under risk), which compresses confident beliefs toward the middle and shifts them by a response bias. The one change is this readout: confident beliefs that a minimal-rule member is in the concept are endorsed less than certainly, and confident beliefs that a number is out are rejected less than certainly.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/8a52ff3d6e8e6ef7/repo/_runs/cell_1/experiment1/model_loop/models/log_odds_belief_readout.py`

### belief_criterion_threshold — rank 15, 103.9 ± 23.4 nats behind the best (4.4× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1653.2

**Hypothesis:** People do the simplest Bayesian concept learning (45 equally likely rules and all intervals of 1-100, with the size principle), but they do not answer "yes" with probability equal to their belief; instead they compare their belief that the target is in the concept against an internal criterion of how sure they must be, and say "yes" only when the belief exceeds that criterion, with the comparison blurred by Gaussian decision noise (criterion and noise inferred). Because the criterion acts on the probability itself, targets with middling belief — such as far-away numbers supported only by broad rules or wide intervals — are rejected more firmly than belief alone predicts, while targets the learner is fairly sure of are endorsed nearly always.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/8a52ff3d6e8e6ef7/repo/_runs/cell_1/experiment1/model_loop/models/belief_criterion_threshold.py`

### feature_overlap_similarity — rank 16, 114.3 ± 27.3 nats behind the best (4.2× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1663.7

**Hypothesis:** People do not infer a single hidden rule; instead they judge the target by its featural similarity to the examples they were shown. Each number is represented by its properties (parity, primality, being a square/cube/power, divisibility, final digit, digits it contains, and which decade it lies in), and the more properties the target shares with each example on average (feature overlap relative to the properties either has), the more likely people say yes — so endorsement is graded even among numbers that fit a rule, and partial matches such as a shared final digit or nearby magnitude still earn some yeses.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/8a52ff3d6e8e6ef7/repo/_runs/cell_1/experiment1/model_loop/models/feature_overlap_similarity.py`

### featural_exemplar_similarity — rank 17, 117.5 ± 27.2 nats behind the best (4.3× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1666.8

**Hypothesis:** People do not pick a single best-fitting rule; they judge a target by its graded exemplar similarity to the examples they saw, where two numbers are similar to the extent that they are close on the number line and share mathematical/digit features (parity, last digit, multiples, powers, primality, digits contained), and "yes" rises with the target's average similarity to the examples. This disagrees most sharply with the rule-based Bayesian model on members of the winning rule that are far from or featurally unlike the examples (which it endorses fully and this model only partly, producing graded typicality), and on non-members that are numerically close to the examples or share their last digit (which it rejects and this model partially endorses).

**Source:** `/scr/wurgaft/auto-psych/agent_trees/8a52ff3d6e8e6ef7/repo/_runs/cell_1/experiment1/model_loop/models/featural_exemplar_similarity.py`

### bounded_range_rule_learning — rank 18, 119.3 ± 35.5 nats behind the best (3.4× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1668.7

**Hypothesis:** People believe the program produces numbers that follow a mathematical rule but only within a bounded range spanned by the examples: they do Bayesian concept learning over "rule restricted to a range" concepts (any of the 45 rules, or no rule at all, intersected with an interval that covers every example), weighting each by the size principle with an inferred, tempered evidence strength. Because a rule restricted to the examples' range is a smaller, better-fitting concept than the unbounded rule, people generalize mainly to rule members inside the [min, max] span of the examples and endorse rule members far outside that span much less.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/8a52ff3d6e8e6ef7/repo/_runs/cell_1/experiment1/model_loop/models/bounded_range_rule_learning.py`

### featural_contrast_similarity — rank 19, 131.0 ± 28.0 nats behind the best (4.7× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1680.4

**Hypothesis:** People do not pick a single rule; they judge a target by featural similarity to the examples, in the manner of Tversky's contrast model. Each number is represented by its properties (parity, primality, being a square/cube/power, divisibility, final digit, digits it contains, decade), and the more of the examples' shared properties the target also has (weighted by how distinctive each property is), and the fewer extra properties the examples lack, the more likely people are to say yes — so members of a rule are graded by how many other properties they share with the examples, and non-members that share e.g. a last digit still get partial endorsement.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/8a52ff3d6e8e6ef7/repo/_runs/cell_1/experiment1/model_loop/models/featural_contrast_similarity.py`

### outlier_tolerant_concept_learning — rank 20, 175.0 ± 24.4 nats behind the best (7.2× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1724.3

**Hypothesis:** People do normative Bayesian concept learning over the 45 mathematical rules (each with its own prior weight) and all intervals of 1-100, using the size principle, but they mis-estimate how the examples were generated: they believe a much larger share of the examples are random strays from 1-100 than the ideal learner assumes (an inferred noise belief instead of a fixed 10%). Because a stray-tolerant learner keeps believing in a tight concept (a small rule or a narrow interval around the examples) even when one example does not fit it, belief stays on tight hypotheses rather than spreading to broad rules and wide intervals. So members of the tight hypotheses keep graded endorsement, while far-away non-members, which only broad hypotheses cover, are rejected more firmly.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/8a52ff3d6e8e6ef7/repo/_runs/cell_1/experiment1/model_loop/models/outlier_tolerant_concept_learning.py`

### bigelow_piantadosi_mixture — rank 21, 253.9 ± 39.8 nats behind the best (6.4× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1803.3

**Hypothesis:** Bayesian concept learning over 45 rules (odd, even, primes, squares, cubes, powers of 2-10, powers of 2 without 32 or with 37, multiples of 3-9, 11 and 12, ending in 0-9, containing the digit 0-9) and every interval of 1-100. A rule is chosen with probability lambda_0, and the rules are equally likely; each interval has prior (1 - lambda_0) / 100^2. Examples are sampled from the concept (the size principle), 10% of them uniformly from 1-100, and "yes" is the posterior probability that the concept contains the target.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/8a52ff3d6e8e6ef7/repo/_runs/cell_1/experiment1/model_loop/models/bigelow_piantadosi_mixture.py`

### bigelow_piantadosi_independent — rank 22, 273.7 ± 43.9 nats behind the best (6.2× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1823.1

**Hypothesis:** Bayesian concept learning over 45 rules (odd, even, primes, squares, cubes, powers of 2-10, powers of 2 without 32 or with 37, multiples of 3-9, 11 and 12, ending in 0-9, containing the digit 0-9) and every interval of 1-100, with a prior weight inferred for each rule separately and a share lambda_0 for the rules as a group; each interval has prior (1 - lambda_0) / 100^2. Examples are sampled from the concept (the size principle), 10% of them uniformly from 1-100, and "yes" is the posterior probability that the concept contains the target.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/8a52ff3d6e8e6ef7/repo/_runs/cell_1/experiment1/model_loop/models/bigelow_piantadosi_independent.py`

### tenenbaum_rules_similarity — rank 23, 565.2 ± 83.1 nats behind the best (6.8× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -2114.5

**Hypothesis:** Bayesian concept learning over 33 mathematical rules (odd, even, squares, cubes, primes, multiples of 3-12, powers of 2-10, ending in 1-9) and every interval of 1-100. The rules share a prior fraction lambda equally; the intervals share the rest by an Erlang density on their size (scale sigma). Examples are sampled from the concept (the size principle), 10% of them uniformly from 1-100, and "yes" is the posterior probability that the concept contains the target.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/8a52ff3d6e8e6ef7/repo/_runs/cell_1/experiment1/model_loop/models/tenenbaum_rules_similarity.py`

### predictive_output_probability_readout — no comparison row

**Hypothesis:** People do the same tempered Bayesian concept learning over the 45 individually weighted rules and all intervals of 1-100, but when asked whether the program would produce a target they do not ask "is it in the concept?" — they judge how probable it is that the program, sampling from its concept, would output that number next (so a number in a small concept seems far likelier to come out than one in a big concept), and say "yes" when that predicted chance is high enough relative to an internal criterion, with graded noise. This disagrees most sharply with the best model's membership readout on members of broad rules: rule members outside the examples' range (supported only by large rules such as "multiples of 3" or "odd") are endorsed much less than in-range members, which also get the high output probability of tight intervals around the examples, and members of very large rules (odd, even) are endorsed less than members of small ones.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/8a52ff3d6e8e6ef7/repo/_runs/cell_1/experiment1/model_loop/models/predictive_output_probability_readout.py`

### log_number_line_ranges — no comparison row

**Hypothesis:** Refinement of `monotone_order_range_prior_shift`: people do the same tempered, serial-position-weighted Bayesian concept learning over the 45 weighted rules and all intervals of 1-100 (with the sorted-order shift toward ranges). The one change is that people think of a magnitude range on a compressed, roughly logarithmic mental number line. So they assume a range's examples are spread evenly along that compressed line, and they judge a range's size by its length on that line. The degree of compression is inferred; no compression recovers the original model. This predicts that generalization from a range is asymmetric: extending a range downward toward small numbers costs more than extending it upward by the same amount. So targets above the examples' range are endorsed more than targets equally far below it, and small examples count as tighter evidence than large ones.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/8a52ff3d6e8e6ef7/repo/_runs/cell_1/experiment1/model_loop/models/log_number_line_ranges.py`

### extrapolation_verification_failure — no comparison row

**Hypothesis:** People infer the hidden concept by the same tempered, serial-position-weighted Bayesian concept learning as the best model (45 weighted rules and all intervals, with sorted lists shifting belief toward ranges), but to answer they must mentally verify that the target follows from the examples, and this extrapolation becomes unreliable beyond the span of numbers they have actually seen: the farther the target lies outside the examples' min–max range, the more likely the check fails and they answer "no", at an inferred rate per unit of distance. This predicts that members of a broad arithmetic rule (e.g. multiples of 3 after 63, 27, 99, 33) are endorsed less outside the examples' range than inside it, more steeply than belief in the concept alone implies, while inside-range judgments are unchanged.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/8a52ff3d6e8e6ef7/repo/_runs/cell_1/experiment1/model_loop/models/extrapolation_verification_failure.py`

### parity_conjoined_order_shift — no comparison row

**Hypothesis:** Refinement of the incumbent `monotone_order_range_prior_shift`: people do the same tempered, serial-position-weighted Bayesian concept learning over the 45 weighted rules and all intervals of 1-100, with sorted lists of three or more examples shifting prior belief from rules toward magnitude ranges. The one change, taken from `parity_restricted_concepts`, is that parity is a basic feature people conjoin with any concept, so they also entertain the odd-only and even-only version of every rule and range (e.g. "odd multiples of 3", "even numbers between 4 and 96"), which share an inferred part of prior belief (share 0 recovers the incumbent). Because these concepts are half the size, the size principle favours them whenever all examples share a parity, so people reject opposite-parity targets — even inside the examples' range and even for members of the winning arithmetic rule — more firmly than the incumbent predicts, while order effects are kept.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/8a52ff3d6e8e6ef7/repo/_runs/cell_1/experiment1/model_loop/models/parity_conjoined_order_shift.py`

## Pruned models (out of the set; narrowest margin first)

No model has been pruned yet in this project.
