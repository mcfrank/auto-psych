# Tried before

72 hypotheses proposed earlier in this project are no longer in the model set. Read them before you propose:

- A *pruned* entry entered the set and later lost to the best model by the stated margin, on the data available then. Its mechanism may be partly right: a model that changes it substantively is welcome, but do not re-propose it unchanged or merely re-parameterised, under any name. Pruned models stay readable under `models/pruned/`.
- A *rejected* entry never entered the set. If it was a near-duplicate of a model still in the set, that region is already covered: do not re-propose it. If it failed on its code or its fit (see its outcome), the idea itself was never tested and a correct implementation may be worth trying.

### featural_contrast_similarity — pruned (experiment1 end of experiment)

**Outcome:** 154.9 nats behind extrapolation_verification_failure (5.6× dse)

**Hypothesis:** People do not pick a single rule; they judge a target by featural similarity to the examples, in the manner of Tversky's contrast model. Each number is represented by its properties (parity, primality, being a square/cube/power, divisibility, final digit, digits it contains, decade), and the more of the examples' shared properties the target also has (weighted by how distinctive each property is), and the fewer extra properties the examples lack, the more likely people are to say yes — so members of a rule are graded by how many other properties they share with the examples, and non-members that share e.g. a last digit still get partial endorsement.

### featural_exemplar_similarity — pruned (experiment1 end of experiment)

**Outcome:** 141.4 nats behind extrapolation_verification_failure (5.3× dse)

**Hypothesis:** People do not pick a single best-fitting rule; they judge a target by its graded exemplar similarity to the examples they saw, where two numbers are similar to the extent that they are close on the number line and share mathematical/digit features (parity, last digit, multiples, powers, primality, digits contained), and "yes" rises with the target's average similarity to the examples. This disagrees most sharply with the rule-based Bayesian model on members of the winning rule that are far from or featurally unlike the examples (which it endorses fully and this model only partly, producing graded typicality), and on non-members that are numerically close to the examples or share their last digit (which it rejects and this model partially endorses).

### feature_overlap_similarity — pruned (experiment1 end of experiment)

**Outcome:** 138.2 nats behind extrapolation_verification_failure (5.2× dse)

**Hypothesis:** People do not infer a single hidden rule; instead they judge the target by its featural similarity to the examples they were shown. Each number is represented by its properties (parity, primality, being a square/cube/power, divisibility, final digit, digits it contains, and which decade it lies in), and the more properties the target shares with each example on average (feature overlap relative to the properties either has), the more likely people say yes — so endorsement is graded even among numbers that fit a rule, and partial matches such as a shared final digit or nearby magnitude still earn some yeses.

### conservative_evidence_concept_learning — pruned (experiment1 end of experiment)

**Outcome:** 111.2 nats behind extrapolation_verification_failure (4.9× dse)

**Hypothesis:** Refinement of `bigelow_piantadosi_mixture`: people do the same Bayesian concept learning over the 45 rules and all intervals of 1-100 with the size principle, but they under-weight the evidence of the examples, as if each example counted for only a fraction of a full observation (a conservative, tempered likelihood). The one change is a free evidence-weight exponent on the likelihood of the examples, so confidence in the smallest consistent rule grows more slowly with examples and more posterior stays on broader rules and intervals — making learners less sure of the winning rule than the ideal learner.

### tempered_evidence_concept_learning — rejected (experiment1 round 0 candidate 4 refine incumbent bigelow_piantadosi_mixture)

**Outcome:** predicts like existing model 'conservative_evidence_concept_learning' (p_yes RMSE 0.00008 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of conservative_evidence_concept_learning, not a new hypothesis.

**Hypothesis:** Refinement of bigelow_piantadosi_mixture: the same Bayesian concept learner over 45 rules and all intervals of 1-100 (rules chosen with probability lambda_0 and equally likely, each interval (1 - lambda_0)/100^2, examples sampled from the concept with 10% noise), except that people under-weight the evidence of the examples — the likelihood of the examples is raised to a single inferred exponent (tempered sampling evidence) before being combined with the prior. With an exponent below 1 the size principle is weakened, so people are less confident in the smallest consistent rule and spread belief onto broader rules and intervals, endorsing minimal-rule members less and nearby non-members more.

### tempered_size_principle_independent — pruned (experiment1 end of experiment)

**Outcome:** 74.2 nats behind extrapolation_verification_failure (5.4× dse)

**Hypothesis:** Refinement of `bigelow_piantadosi_independent`: people do the same Bayesian concept learning over 45 rules (each with its own prior weight) and all intervals, but they under-trust the size principle — they treat the examples as weaker evidence than a fully rational learner would, as if the likelihood of the examples were tempered by an inferred exponent (sampling-assumption strength), so their posterior is less concentrated on the smallest consistent rule. The one change is this free evidence-strength exponent (1 recovers the original model); it predicts lower endorsement of minimal-rule members and graded endorsement among them, because members also covered by competing concepts (nearby intervals, other rules) keep more support.

### log_odds_belief_readout — pruned (experiment1 end of experiment)

**Outcome:** 123.8 nats behind extrapolation_verification_failure (5.4× dse)

**Hypothesis:** Refinement of bigelow_piantadosi_mixture: people do the same Bayesian concept learning over the 45 rules and all intervals of 1-100 with the size principle, but they do not answer "yes" with probability equal to their belief that the concept contains the target; instead they read out that belief through a distorted, linear-in-log-odds probability weighting (as in judgments of probability and decisions under risk), which compresses confident beliefs toward the middle and shifts them by a response bias. The one change is this readout: confident beliefs that a minimal-rule member is in the concept are endorsed less than certainly, and confident beliefs that a number is out are rejected less than certainly.

### sampled_hypothesis_majority_vote — pruned (experiment1 end of experiment)

**Outcome:** 77.2 nats behind extrapolation_verification_failure (5.0× dse)

**Hypothesis:** Refinement of tempered_size_principle_independent: people do the same tempered Bayesian concept learning over the 45 rules and all intervals, but they cannot hold the whole posterior in mind — they bring a small number of candidate concepts to mind by sampling them from their posterior and answer "yes" if the majority of those sampled concepts contain the target. The one change is this limited-sample majority vote (with an inferred number of samples; one sample recovers the original model), which makes answers more all-or-none than the posterior itself: likely members are endorsed more consistently and weakly supported targets, such as numbers beyond the range of the examples, are endorsed less.

### outlier_tolerant_concept_learning — pruned (experiment1 end of experiment)

**Outcome:** 198.9 nats behind extrapolation_verification_failure (7.8× dse)

**Hypothesis:** People do normative Bayesian concept learning over the 45 mathematical rules (each with its own prior weight) and all intervals of 1-100, using the size principle, but they mis-estimate how the examples were generated: they believe a much larger share of the examples are random strays from 1-100 than the ideal learner assumes (an inferred noise belief instead of a fixed 10%). Because a stray-tolerant learner keeps believing in a tight concept (a small rule or a narrow interval around the examples) even when one example does not fit it, belief stays on tight hypotheses rather than spreading to broad rules and wide intervals. So members of the tight hypotheses keep graded endorsement, while far-away non-members, which only broad hypotheses cover, are rejected more firmly.

### bounded_range_rule_learning — pruned (experiment1 end of experiment)

**Outcome:** 143.2 nats behind extrapolation_verification_failure (4.5× dse)

**Hypothesis:** People believe the program produces numbers that follow a mathematical rule but only within a bounded range spanned by the examples: they do Bayesian concept learning over "rule restricted to a range" concepts (any of the 45 rules, or no rule at all, intersected with an interval that covers every example), weighting each by the size principle with an inferred, tempered evidence strength. Because a rule restricted to the examples' range is a smaller, better-fitting concept than the unbounded rule, people generalize mainly to rule members inside the [min, max] span of the examples and endorse rule members far outside that span much less.

### size_penalized_interval_prior — pruned (experiment1 end of experiment)

**Outcome:** 90.1 nats behind extrapolation_verification_failure (5.4× dse)

**Hypothesis:** Refinement of `tempered_size_principle_independent`: people do the same tempered Bayesian concept learning over 45 individually weighted rules and all intervals of 1-100, but a priori they expect a magnitude-based concept to be a narrow range rather than treating all intervals as equally likely — their prior on an interval falls off as a power of its width, with the steepness inferred. The one change is this size-penalizing interval prior (steepness 0 recovers a uniform interval prior); it predicts that people confine generalization more tightly to the range spanned by the examples, and endorse targets outside that range less, than the incumbent allows.

### compact_interval_tempered_bayes — pruned (experiment1 end of experiment)

**Outcome:** 83.9 nats behind extrapolation_verification_failure (5.1× dse)

**Hypothesis:** Refinement of `tempered_size_principle_independent`: people do the same tempered-evidence Bayesian concept learning over the 45 rules (each with its own prior weight) and all intervals of 1-100, but they do not consider all intervals equally plausible a priori — they prefer compact magnitude ranges, with an Erlang prior over interval size whose scale is inferred (the one change, taken from `tenenbaum_rules_similarity`). Because the tempered evidence no longer spreads interval belief onto wide ranges, people confine magnitude-based generalization close to the [min, max] range of the examples and endorse targets outside it less than the incumbent predicts.

### domain_specific_evidence_weight — pruned (experiment1 end of experiment)

**Outcome:** 117.5 nats behind extrapolation_verification_failure (5.1× dse)

**Hypothesis:** Refinement of `conservative_evidence_concept_learning`: people do the same Bayesian concept learning over the 45 equally weighted rules and all intervals of 1-100 with the size principle, but how strongly they weigh the examples as evidence depends on the kind of concept — they are conservative about mathematical rules yet apply the size principle sharply to magnitude ranges, so once they think "a range of numbers" they confine it tightly to the span of the examples. The one change is splitting the single evidence-weight exponent into one for rule hypotheses and one for interval hypotheses; it predicts less endorsement of targets just outside the examples' [min, max] range (relative to inside it) than a single shared tempering allows, while keeping graded endorsement of rule members.

### serial_position_weighted_evidence — pruned (experiment1 end of experiment)

**Outcome:** 29.0 nats behind extrapolation_verification_failure (3.7× dse)

**Hypothesis:** Refinement of `tempered_size_principle_independent`: people do the same tempered Bayesian concept learning over the 45 individually weighted rules and all intervals of 1-100, but they do not weigh every example equally as evidence — how much an example counts depends on its serial position in the list, with evidence weight shifting smoothly toward the first examples (primacy) or the last ones (recency) by an inferred serial-position gradient, while the total evidence strength stays the same. The one change is this order-dependent weighting (a zero gradient recovers the incumbent); it predicts that the same examples shown in a different order yield different generalization, and in particular that an example that breaks the pattern of the others (e.g. an odd number among evens) undermines the tight concept more when it sits at the favoured end of the list.

### iter2_candidate1 — pruned (experiment1 end of experiment)

**Outcome:** 68.9 nats behind extrapolation_verification_failure (2.7× dse)

**Hypothesis:** People judge a target by Shepard-style exponential generalization in a psychological space whose dimensions are weighted by selective attention, and the examples set that attention: they attend to the properties the examples agree on (more strongly the more consistently the examples share a property and the more distinctive it is), and to magnitude in proportion to how tightly the examples cluster. The probability of "yes" falls off exponentially with the target's attention-weighted distance from the examples, so missing a consistently shared, distinctive property (e.g. the common last digit) or lying well outside a tight example range sharply lowers endorsement, while properties the examples do not agree on barely matter.

### belief_criterion_threshold — pruned (experiment1 end of experiment)

**Outcome:** 127.8 nats behind extrapolation_verification_failure (5.5× dse)

**Hypothesis:** People do the simplest Bayesian concept learning (45 equally likely rules and all intervals of 1-100, with the size principle), but they do not answer "yes" with probability equal to their belief; instead they compare their belief that the target is in the concept against an internal criterion of how sure they must be, and say "yes" only when the belief exceeds that criterion, with the comparison blurred by Gaussian decision noise (criterion and noise inferred). Because the criterion acts on the probability itself, targets with middling belief — such as far-away numbers supported only by broad rules or wide intervals — are rejected more firmly than belief alone predicts, while targets the learner is fairly sure of are endorsed nearly always.

### hierarchical_tempered_evidence — pruned (experiment1 end of experiment)

**Outcome:** 88.0 nats behind extrapolation_verification_failure (5.7× dse)

**Hypothesis:** Refinement of `tempered_size_principle_independent`: people under-trust the examples only when deciding *which kind* of concept and which rule produced them, but once they entertain a magnitude range they apply the size principle at full strength, so the range they imagine stays tight around the examples. The one change is where the inferred evidence-strength exponent applies: it tempers the evidence for each rule and for the interval family as a whole, while belief among intervals follows the untempered size principle; this predicts firmer rejection of targets far outside the examples' range that no small rule covers, while keeping graded endorsement of rule members.

### parity_restricted_concepts — pruned (experiment1 end of experiment)

**Outcome:** 28.0 nats behind extrapolation_verification_failure (2.4× dse)

**Hypothesis:** Refinement of `tempered_size_principle_independent`: people do the same tempered-evidence Bayesian concept learning over the 45 individually weighted rules and all intervals of 1-100, but parity is a basic feature they can conjoin with any concept, so they also entertain the odd-only and even-only version of each rule and interval (e.g. "odd multiples of 3", "even numbers between 4 and 96"), which together receive an inferred share of prior belief. The one change is adding these parity-restricted concepts (share 0 recovers the incumbent); because such a concept is half the size, the size principle favours it whenever all examples share a parity, so people reject opposite-parity targets — including far, weakly supported ones outside the examples' range — more firmly than the incumbent predicts.

### sampled_concepts_criterion_vote — pruned (experiment1 end of experiment)

**Outcome:** 79.8 nats behind extrapolation_verification_failure (4.9× dse)

**Hypothesis:** Refinement of `sampled_hypothesis_majority_vote`: people do the same tempered Bayesian concept learning over the 45 individually weighted rules and all intervals, and answer by bringing a few candidate concepts to mind by sampling them from their posterior — but they say "yes" only if a criterion number of the sampled concepts contains the target, and that criterion is inferred rather than fixed at a simple majority (equivalently, they set how many sampled concepts must favour the target and how many may go against it). The one change is this free decision criterion (equal counts recover majority vote, a single sample recovers the incumbent); a demanding criterion predicts that targets supported only by a minority of concepts — such as numbers far outside the examples' range that fit no small rule — are rejected more firmly than majority vote allows, while targets in the dominant concept are still endorsed.

### ordered_sweep_concepts — dropped (experiment1 round 3)

**Outcome:** fold 0 too slow to fit: the fit of 'ordered_sweep_concepts' (target_accept 0.99) was still sampling after the 30-minute limit and was stopped. Every sampling run of a candidate's admission fit has a 30-minute limit. Make the model cheaper to evaluate and easier to sample: vectorise the likelihood over trials (no Python loops, pytensor scan or per-trial subgraphs), compute features once per unique sequence (in compute_features or prepare_observed, not in the graph), and drop or merge parameters the data barely constrain — a weakly identified posterior makes NUTS take maximal-length trajectories.

**Hypothesis:** People entertain one more kind of concept than the current models allow: an ordered sweep, in which the program runs through a range of numbers in increasing or decreasing order, so examples shown in sorted order are much more probable under a sweep than under a randomly sampled concept (by the number of possible orderings), and a sweep receives an inferred share of prior belief. All else is the parity-restricted tempered Bayesian concept learner; because sorted presentation shifts belief from the tight "ends in d" rule toward a sweep over the examples' magnitude range, people endorse in-range targets that break the shared digit more when the examples come in sorted order than when the same kind of set is shuffled.

### place_value_context_exemplar — pruned (experiment1 end of experiment)

**Outcome:** 56.6 nats behind extrapolation_verification_failure (2.5× dse)

**Hypothesis:** People do not infer a rule; they judge a target by its summed similarity to the stored examples, as in the context (exemplar) model of classification, where each number is represented by its place-value digits — its tens digit and its units digit — together with its parity and its magnitude, and similarity to an example shrinks multiplicatively with each mismatching dimension, with an inferred attention weight per dimension. "Yes" grows with the target's total similarity to the examples relative to an internal criterion, so a target that shares its tens digit with one example (e.g. 25 after 62, 32, 22) gets partial endorsement even when it breaks the examples' shared final digit, whereas the same target is rejected when no example lies in its decade.

### seriation_primed_range_concepts — dropped (experiment1 round 3)

**Outcome:** fold 0 too slow to fit: the fit of 'seriation_primed_range_concepts' (target_accept 0.99) was still sampling after the 30-minute limit and was stopped. Every sampling run of a candidate's admission fit has a 30-minute limit. Make the model cheaper to evaluate and easier to sample: vectorise the likelihood over trials (no Python loops, pytensor scan or per-trial subgraphs), compute features once per unique sequence (in compute_features or prepare_observed, not in the graph), and drop or merge parameters the data barely constrain — a weakly identified posterior makes NUTS take maximal-length trajectories.

**Hypothesis:** When three or more examples are shown as an orderly, sorted sequence (ascending or descending), people perceive them as a progression along the number line, and this perceptual grouping brings magnitude-range interpretations ("numbers between 22 and 62") to mind, whereas a scrambled list leaves property rules such as "ends in 2" dominant. People otherwise infer the hidden concept by tempered Bayesian concept learning over rules, intervals and their odd/even versions, but an orderly presentation raises their prior belief in a magnitude-range concept by an inferred amount. So for the same digit-sharing examples, in-range targets lacking the shared digit are endorsed more when the list is sorted than when it is shuffled.

### orderly_list_discounted_evidence — dropped (experiment1 round 3)

**Outcome:** fold 0 too slow to fit: the fit of 'orderly_list_discounted_evidence' (target_accept 0.99) was still sampling after the 30-minute limit and was stopped. Every sampling run of a candidate's admission fit has a 30-minute limit. Make the model cheaper to evaluate and easier to sample: vectorise the likelihood over trials (no Python loops, pytensor scan or per-trial subgraphs), compute features once per unique sequence (in compute_features or prepare_observed, not in the graph), and drop or merge parameters the data barely constrain — a weakly identified posterior makes NUTS take maximal-length trajectories.

**Hypothesis:** Refinement of `parity_restricted_concepts`: people do the same tempered-evidence Bayesian concept learning over the 45 weighted rules, all intervals, and their odd-only and even-only versions. But the size principle rests on believing the examples were drawn at random from the concept, and a list shown in sorted order (ascending or descending, three or more examples) looks deliberately arranged rather than randomly sampled. So people trust such orderly examples less as evidence. The one change is an inferred reduction of evidence strength for orderly lists (zero recovers the incumbent). It predicts that the same examples shown in sorted order (e.g. 62,32,22) support the tight "ends in 2" rule less, and non-members are endorsed more, than when shown in a scrambled order (e.g. 63,23,43).

### orderly_list_weak_sampling — rejected (experiment1 round 3 candidate 4 refine incumbent parity_restricted_concepts)

**Outcome:** predicts like existing model 'orderly_list_discounted_evidence' (p_yes RMSE 0.00105 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of orderly_list_discounted_evidence, not a new hypothesis.

**Hypothesis:** Refinement of `parity_restricted_concepts`: people do the same tempered-evidence Bayesian concept learning over the 45 individually weighted rules, all intervals of 1-100, and their odd-only and even-only versions. The one change is that how strongly they apply the size principle depends on how the examples are shown. When three or more examples come in sorted order (ascending or descending), people take the list to have been picked systematically, not sampled at random from the concept. They then treat the examples as weaker evidence for the smallest consistent concept, by an inferred factor (a factor of zero recovers the incumbent). This predicts that the same digit-sharing examples (e.g. 62,32,22 vs 63,23,43) loosen the "ends in d" rule and earn more yeses for targets with other final digits when they are shown in sorted order than when they are shuffled.

### monotone_order_range_prior_shift — pruned (experiment1 end of experiment)

**Outcome:** 23.9 nats behind extrapolation_verification_failure (3.0× dse)

**Hypothesis:** Refinement of `serial_position_weighted_evidence`: people do the same tempered, serial-position-weighted Bayesian concept learning over the 45 individually weighted rules and all intervals of 1-100, but when three or more examples are shown in sorted (steadily rising or steadily falling) order, they read the list as a progression along the number line and so shift prior belief from mathematical/digit rules toward magnitude-range concepts, by an inferred amount. The one change is this order-induced shift in the rule-versus-range prior (zero shift recovers the original model); it predicts that a sorted list such as 62, 32, 22 loosens the "ends in 2" rule, so in-range or nearby numbers lacking the shared digit are endorsed more than when the same kind of set is shown in scrambled order (e.g. 63, 23, 43).

### decade_place_value_concepts — pruned (experiment1 end of experiment)

**Outcome:** 28.5 nats behind extrapolation_verification_failure (2.6× dse)

**Hypothesis:** Refinement of `parity_restricted_concepts`: people do the same tempered-evidence Bayesian concept learning over the 45 weighted rules, all intervals of 1-100, and their odd-only and even-only versions. The one change is that people also read numbers by place value, so they entertain decade concepts: "numbers whose tens digit is one of these", e.g. "numbers in the 20s, 30s or 60s". These concepts get an inferred share of prior belief (share 0 recovers the incumbent), and the size principle favours the smallest one, the decades the examples occupy. This predicts that a target with a different final digit is endorsed more when an example sits in its decade (62,32,22 → 25; 73,33,3 → 36) than when none does (63,23,43 → 35), whatever order the examples are shown in.

### predictive_output_probability_readout — pruned (experiment1 end of experiment)

**Outcome:** 202.1 nats behind extrapolation_verification_failure (6.8× dse)

**Hypothesis:** People do the same tempered Bayesian concept learning over the 45 individually weighted rules and all intervals of 1-100, but when asked whether the program would produce a target they do not ask "is it in the concept?" — they judge how probable it is that the program, sampling from its concept, would output that number next (so a number in a small concept seems far likelier to come out than one in a big concept), and say "yes" when that predicted chance is high enough relative to an internal criterion, with graded noise. This disagrees most sharply with the best model's membership readout on members of broad rules: rule members outside the examples' range (supported only by large rules such as "multiples of 3" or "odd") are endorsed much less than in-range members, which also get the high output probability of tight intervals around the examples, and members of very large rules (odd, even) are endorsed less than members of small ones.

### log_number_line_ranges — dropped (experiment1 round 4)

**Outcome:** fold 0 too slow to fit: the fit of 'log_number_line_ranges' (target_accept 0.99) was still sampling after the 30-minute limit and was stopped. Every sampling run of a candidate's admission fit has a 30-minute limit. Make the model cheaper to evaluate and easier to sample: vectorise the likelihood over trials (no Python loops, pytensor scan or per-trial subgraphs), compute features once per unique sequence (in compute_features or prepare_observed, not in the graph), and drop or merge parameters the data barely constrain — a weakly identified posterior makes NUTS take maximal-length trajectories.

**Hypothesis:** Refinement of `monotone_order_range_prior_shift`: people do the same tempered, serial-position-weighted Bayesian concept learning over the 45 weighted rules and all intervals of 1-100 (with the sorted-order shift toward ranges). The one change is that people think of a magnitude range on a compressed, roughly logarithmic mental number line. So they assume a range's examples are spread evenly along that compressed line, and they judge a range's size by its length on that line. The degree of compression is inferred; no compression recovers the original model. This predicts that generalization from a range is asymmetric: extending a range downward toward small numbers costs more than extending it upward by the same amount. So targets above the examples' range are endorsed more than targets equally far below it, and small examples count as tighter evidence than large ones.

### extrapolation_verification_failure — pruned (experiment2 end of experiment)

**Outcome:** 173.4 nats behind consistent_examples_bounded_rules (6.5× dse)

**Hypothesis:** People infer the hidden concept by the same tempered, serial-position-weighted Bayesian concept learning as the best model (45 weighted rules and all intervals, with sorted lists shifting belief toward ranges), but to answer they must mentally verify that the target follows from the examples, and this extrapolation becomes unreliable beyond the span of numbers they have actually seen: the farther the target lies outside the examples' min–max range, the more likely the check fails and they answer "no", at an inferred rate per unit of distance. This predicts that members of a broad arithmetic rule (e.g. multiples of 3 after 63, 27, 99, 33) are endorsed less outside the examples' range than inside it, more steeply than belief in the concept alone implies, while inside-range judgments are unchanged.

### parity_conjoined_order_shift — dropped (experiment1 round 4)

**Outcome:** fold 0 too slow to fit: the fit of 'parity_conjoined_order_shift' (target_accept 0.99) was still sampling after the 30-minute limit and was stopped. Every sampling run of a candidate's admission fit has a 30-minute limit. Make the model cheaper to evaluate and easier to sample: vectorise the likelihood over trials (no Python loops, pytensor scan or per-trial subgraphs), compute features once per unique sequence (in compute_features or prepare_observed, not in the graph), and drop or merge parameters the data barely constrain — a weakly identified posterior makes NUTS take maximal-length trajectories.

**Hypothesis:** Refinement of the incumbent `monotone_order_range_prior_shift`: people do the same tempered, serial-position-weighted Bayesian concept learning over the 45 weighted rules and all intervals of 1-100, with sorted lists of three or more examples shifting prior belief from rules toward magnitude ranges. The one change, taken from `parity_restricted_concepts`, is that parity is a basic feature people conjoin with any concept, so they also entertain the odd-only and even-only version of every rule and range (e.g. "odd multiples of 3", "even numbers between 4 and 96"), which share an inferred part of prior belief (share 0 recovers the incumbent). Because these concepts are half the size, the size principle favours them whenever all examples share a parity, so people reject opposite-parity targets — even inside the examples' range and even for members of the winning arithmetic rule — more firmly than the incumbent predicts, while order effects are kept.

### parity_conjoined_order_shift_2 — rejected (experiment1 round 4 candidate 4 refine incumbent monotone_order_range_prior_shift)

**Outcome:** predicts like existing model 'parity_conjoined_order_shift' (p_yes RMSE 0.00000 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of parity_conjoined_order_shift, not a new hypothesis.

**Hypothesis:** Refinement of `monotone_order_range_prior_shift`: people do the same tempered, serial-position-weighted Bayesian concept learning over the 45 weighted rules and all intervals of 1-100, with the same sorted-order shift of prior belief toward ranges, but parity is a basic feature they conjoin with any concept, so they also entertain the odd-only and even-only version of each rule and range ("odd multiples of 3", "even numbers between 4 and 96"), which share an inferred part of the prior (share 0 recovers the incumbent). The one change is adding these parity-restricted concepts, taken from `parity_restricted_concepts`. Because such a concept is half the size, the size principle favours it whenever all examples share a parity, so rule members of the opposite parity are rejected — e.g. after 69, 9, 39, 21 the even multiples of 3 above the examples (72, 84, 96) are endorsed much less than the odd ones inside the range, which is the out-of-range drop the incumbent underpredicts.

### range_bounded_parity_concepts — pruned (experiment2 end of experiment)

**Outcome:** 95.7 nats behind consistent_examples_bounded_rules (6.0× dse)

**Hypothesis:** Refinement of `parity_restricted_concepts`: people do the same tempered-evidence Bayesian concept learning over the 45 weighted rules, all intervals of 1-100, and their odd-only and even-only versions. The one change is a single added component, taken from `bounded_range_rule_learning`: people can also restrict a rule to a range of numbers that covers the examples (e.g. "multiples of 3 between 20 and 100"), and these range-bounded rules get an inferred share of the prior for rules (share 0 recovers `parity_restricted_concepts`). A bounded rule is smaller than the full rule, so the size principle favours it. This predicts that people endorse members of the examples' rule that lie outside the examples' min-max range less than members inside it, and that this gap is larger than the rule/interval mixture allows (63,27,99,33 → 3 below 45).

### range_bounded_rules_order_shift — rejected (experiment1 round 4 candidate 4 refine incumbent monotone_order_range_prior_shift repair 1)

**Outcome:** too slow to fit: the fit of 'range_bounded_rules_order_shift' (target_accept 0.99) was still sampling after the 30-minute limit and was stopped. Every sampling run of a candidate's admission fit has a 30-minute limit. Make the model cheaper to evaluate and easier to sample: vectorise the likelihood over trials (no Python loops, pytensor scan or per-trial subgraphs), compute features once per unique sequence (in compute_features or prepare_observed, not in the graph), and drop or merge parameters the data barely constrain — a weakly identified posterior makes NUTS take maximal-length trajectories.

**Hypothesis:** Refinement of the incumbent `monotone_order_range_prior_shift`: people do the same tempered, serial-position-weighted Bayesian concept learning over the 45 weighted rules and all intervals of 1-100, with sorted lists of three or more examples shifting prior belief from rules toward ranges. The one change, a component taken from `bounded_range_rule_learning`, is that people can also think of a rule as holding only over a stretch of the number line that covers the examples ("multiples of 3 between 9 and 70"), and these range-bounded rules take an inferred share of the rule prior (share 0 recovers the incumbent). Because a bounded rule is smaller than the full rule, the size principle favours it, so members of the winning arithmetic rule that lie outside the examples' min-max range are endorsed less than members inside it — the out-of-range drop the incumbent underpredicts — while order effects are kept.

### tenenbaum_rules_similarity — pruned (experiment1 end of experiment)

**Outcome:** 589.0 nats behind extrapolation_verification_failure (7.0× dse)

**Hypothesis:** Bayesian concept learning over 33 mathematical rules (odd, even, squares, cubes, primes, multiples of 3-12, powers of 2-10, ending in 1-9) and every interval of 1-100. The rules share a prior fraction lambda equally; the intervals share the rest by an Erlang density on their size (scale sigma). Examples are sampled from the concept (the size principle), 10% of them uniformly from 1-100, and "yes" is the posterior probability that the concept contains the target.

### bigelow_piantadosi_mixture — pruned (experiment1 end of experiment)

**Outcome:** 277.8 nats behind extrapolation_verification_failure (7.0× dse)

**Hypothesis:** Bayesian concept learning over 45 rules (odd, even, primes, squares, cubes, powers of 2-10, powers of 2 without 32 or with 37, multiples of 3-9, 11 and 12, ending in 0-9, containing the digit 0-9) and every interval of 1-100. A rule is chosen with probability lambda_0, and the rules are equally likely; each interval has prior (1 - lambda_0) / 100^2. Examples are sampled from the concept (the size principle), 10% of them uniformly from 1-100, and "yes" is the posterior probability that the concept contains the target.

### bigelow_piantadosi_independent — pruned (experiment1 end of experiment)

**Outcome:** 297.6 nats behind extrapolation_verification_failure (6.7× dse)

**Hypothesis:** Bayesian concept learning over 45 rules (odd, even, primes, squares, cubes, powers of 2-10, powers of 2 without 32 or with 37, multiples of 3-9, 11 and 12, ending in 0-9, containing the digit 0-9) and every interval of 1-100, with a prior weight inferred for each rule separately and a share lambda_0 for the rules as a group; each interval has prior (1 - lambda_0) / 100^2. Examples are sampled from the concept (the size principle), 10% of them uniformly from 1-100, and "yes" is the posterior probability that the concept contains the target.

### sample_size_neglect_evidence — pruned (experiment2 end of experiment)

**Outcome:** 83.5 nats behind consistent_examples_bounded_rules (5.6× dse)

**Hypothesis:** People do the same tempered Bayesian concept learning as the best model (45 weighted rules, all intervals, their odd/even versions and range-bounded rules), with one distortion: they neglect sample size, so the total weight they give the examples grows less than in proportion to how many examples they saw — the first example counts heavily, and each further example adds less. This predicts that a single example already pins down its most salient rule (e.g. 10 → multiples of 10), producing many yeses for that rule's members, while adding more examples sharpens generalization far less than an ideal learner would.

### weak_sampling_hypothesis_elimination — pruned (experiment2 end of experiment)

**Outcome:** 94.2 nats behind consistent_examples_bounded_rules (4.2× dse)

**Hypothesis:** People learn the hidden concept by eliminating hypotheses rather than by weighing how typical the examples are: they keep every rule (one of 45 mathematical or digit rules, each with its own prior plausibility) or range of numbers that is consistent with the examples, treating an example that does not fit only as a possible stray, and they answer "yes" with their total belief in the surviving concepts that contain the target. Because there is no size principle (weak sampling), a small concept gains nothing over a broad one that also fits, so people generalize broadly — especially from a single example — and adding consistent examples does not sharpen belief toward the smallest rule.

### position_growing_evidence_weight — dropped (experiment2 round 0)

**Outcome:** fold 0 too slow to fit: the fit of 'position_growing_evidence_weight' (target_accept 0.99) was still sampling after the 30-minute limit and was stopped. Every sampling run of a candidate's admission fit has a 30-minute limit. Make the model cheaper to evaluate and easier to sample: vectorise the likelihood over trials (no Python loops, pytensor scan or per-trial subgraphs), compute features once per unique sequence (in compute_features or prepare_observed, not in the graph), and drop or merge parameters the data barely constrain — a weakly identified posterior makes NUTS take maximal-length trajectories.

**Hypothesis:** People revise a running guess about the concept as each example arrives, and how much an example sharpens that guess grows with its serial position: the first example only seeds a broad set of candidate concepts (mathematical rules and number ranges) and counts as weak evidence, while each later example, arriving once a guess is already forming, is weighed more heavily as confirmation, with evidence weight rising as a power of the example's position in the list. This predicts broad generalization from a single example, generalization that sharpens faster than in proportion to the number of examples, and that the same examples shown in a different order lead to different generalization because the later ones dominate.

### sublinear_evidence_accumulation — rejected (experiment2 round 0 candidate 3 refine incumbent range_bounded_parity_concepts)

**Outcome:** too slow to fit: the fit of 'sublinear_evidence_accumulation' (target_accept 0.99) was still sampling after the 30-minute limit and was stopped. Every sampling run of a candidate's admission fit has a 30-minute limit. Make the model cheaper to evaluate and easier to sample: vectorise the likelihood over trials (no Python loops, pytensor scan or per-trial subgraphs), compute features once per unique sequence (in compute_features or prepare_observed, not in the graph), and drop or merge parameters the data barely constrain — a weakly identified posterior makes NUTS take maximal-length trajectories.

**Hypothesis:** Refinement of `range_bounded_parity_concepts`: people do the same tempered-evidence Bayesian concept learning over the 45 weighted rules, all intervals, their odd-only and even-only versions and the range-bounded rules, but they do not count the examples as independent evidence. Each additional example is treated as partly redundant with the ones already seen, so the strength of the evidence grows sublinearly with the number of examples, at an inferred rate (no discount recovers the incumbent). This predicts that a single example already drives sharp rule-based generalization (10 → 60, 70, 80, 90; 11 → 22, 55, 99), which addresses the critique that people generalize more broadly from one example than the model allows, while three or four examples tighten generalization less than independent sampling implies.

### example_count_scaled_evidence — rejected (experiment2 round 0 candidate 4 refine incumbent range_bounded_parity_concepts)

**Outcome:** too slow to fit: the fit of 'example_count_scaled_evidence' (target_accept 0.99) was still sampling after the 30-minute limit and was stopped. Every sampling run of a candidate's admission fit has a 30-minute limit. Make the model cheaper to evaluate and easier to sample: vectorise the likelihood over trials (no Python loops, pytensor scan or per-trial subgraphs), compute features once per unique sequence (in compute_features or prepare_observed, not in the graph), and drop or merge parameters the data barely constrain — a weakly identified posterior makes NUTS take maximal-length trajectories.

**Hypothesis:** Refinement of `range_bounded_parity_concepts`: people do the same tempered-evidence Bayesian concept learning over the 45 weighted rules, all intervals of 1-100, their odd-only and even-only versions, and range-bounded rules, but how strongly they apply the size principle to each example is not fixed — it grows with the number of examples shown, by an inferred power of the set size (power 0 recovers the incumbent). A single example raises little sense of a "suspicious coincidence", so people treat it as weak evidence and generalize broadly from it (e.g. many multiples of 10 after just "10", well above what the incumbent allows), and only commit to the tightest concept covering the examples once several examples agree.

### single_example_rule_reading — pruned (experiment2 end of experiment)

**Outcome:** 82.8 nats behind consistent_examples_bounded_rules (4.6× dse)

**Hypothesis:** Refinement of `parity_restricted_concepts` (people do tempered-evidence Bayesian concept learning over the 45 weighted rules, all intervals of 1-100, and their odd-only and even-only versions). The one change: a magnitude range is a concept defined by its endpoints, so a single example cannot suggest one — when only one example is shown, people read it as an instance of a mathematical or digit property, and prior belief shifts from range concepts to rules by an inferred amount (zero shift recovers the original model). This predicts that from a lone example people endorse the members of the rules that contain it broadly (11 → 22, 55, 99; 10 → 60, 90) rather than confining belief to a tight range around the example, while judgments about sets of two or more examples are unchanged.

### example_spread_range_bounds — rejected (experiment2 round 0 candidate 3 refine incumbent range_bounded_parity_concepts repair 1)

**Outcome:** too slow to fit: the fit of 'example_spread_range_bounds' (target_accept 0.99) was still sampling after the 30-minute limit and was stopped. Every sampling run of a candidate's admission fit has a 30-minute limit. Make the model cheaper to evaluate and easier to sample: vectorise the likelihood over trials (no Python loops, pytensor scan or per-trial subgraphs), compute features once per unique sequence (in compute_features or prepare_observed, not in the graph), and drop or merge parameters the data barely constrain — a weakly identified posterior makes NUTS take maximal-length trajectories.

**Hypothesis:** Refinement of `range_bounded_parity_concepts`: people do the same tempered-evidence Bayesian concept learning over the 45 weighted rules, all intervals of 1-100, their odd-only and even-only versions, and rules restricted to a range covering the examples. The one change is that a range bound is something people read off the spread of the examples, so they restrict a rule to a range only when at least two different examples mark out that range: a single example suggests no range bound and is generalized through the unbounded rules it belongs to (no new parameters). This predicts broader rule-based generalization from one example than the incumbent allows (10 → 60, 70, 80, 90; 11 → 55, 66, 99), while generalization from two or more examples still falls off for rule members outside the examples' span.

### range_size_prior_concepts — pruned (experiment2 end of experiment)

**Outcome:** 69.3 nats behind consistent_examples_bounded_rules (4.7× dse)

**Hypothesis:** Refinement of `range_bounded_parity_concepts`: people do the same tempered-evidence Bayesian concept learning over the 45 weighted rules, all intervals of 1-100, their odd-only and even-only versions, and range-bounded rules, but they do not think every magnitude range is equally plausible a priori — a tiny range such as "numbers from 10 to 12" seems an unlikely thing for a program to produce, so a range's prior plausibility grows (or shrinks) as an inferred power of its size (power 0 recovers the incumbent). The one change is this size-dependent prior on ranges; because the size principle favours tiny ranges most when there is only one example, a broad-range prior moves belief from tiny ranges around a lone example to the rules it belongs to, so people generalize from a single example (10 → 80, 11 → 55) much more than the incumbent allows, while several examples still pin down a tight concept.

### normalized_attention_similarity — pruned (experiment2 end of experiment)

**Outcome:** 199.3 nats behind consistent_examples_bounded_rules (6.1× dse)

**Hypothesis:** People judge a target by its similarity to the examples over number properties (parity, primality, squares, cubes, powers, multiples, final digit, digits contained), where the examples set a fixed budget of attention that is shared among the properties according to how consistently the examples have them. The target's dissimilarity is the fraction of that attention falling on properties it lacks, and the chance of "yes" falls off exponentially with it (Shepard-style generalization). Because the budget is fixed, a lone example spreads attention over its many properties, so a target sharing most of them (10 → 80, 6 → 18) is still judged very similar and broadly endorsed, while several examples that agree on one property (all ending in 2) concentrate attention there and sharply reject targets lacking it.

### inattentive_guess_lapse — dropped (experiment2 round 1)

**Outcome:** fold 0 too slow to fit: the fit of 'inattentive_guess_lapse' (target_accept 0.99) was still sampling after the 30-minute limit and was stopped. Every sampling run of a candidate's admission fit has a 30-minute limit. Make the model cheaper to evaluate and easier to sample: vectorise the likelihood over trials (no Python loops, pytensor scan or per-trial subgraphs), compute features once per unique sequence (in compute_features or prepare_observed, not in the graph), and drop or merge parameters the data barely constrain — a weakly identified posterior makes NUTS take maximal-length trajectories.

**Hypothesis:** People infer the hidden concept by simple hypothesis elimination — they keep every mathematical/digit rule (each with its own prior plausibility) or range of numbers consistent with the examples, with no size principle — but on a fixed, inferred share of trials they do not consult that belief at all and instead answer by an inattentive guess with their own inferred bias toward "yes". This decision-level lapse predicts that even the examples themselves are endorsed well short of certainty (about 70%), and that targets the belief firmly rejects still collect a floor of yeses set by the guessing bias, compressing all judgments toward that bias by the same proportion.

### open_ended_multiples_concepts — pruned (experiment2 end of experiment)

**Outcome:** 71.5 nats behind consistent_examples_bounded_rules (4.8× dse)

**Hypothesis:** People do the same tempered-evidence Bayesian concept learning as `range_size_prior_concepts` (45 weighted rules, size-weighted intervals of 1-100, their odd-only and even-only versions, and range-bounded rules), but their space of concepts also includes an open-ended family of "multiples of k" concepts for larger steps k from 13 to 50 (e.g. multiples of 25, of 33, of 14), beyond the few small multiples in the fixed rule list. This family gets an inferred share of prior belief, and within it a step's prior plausibility falls off in inverse proportion to k (larger steps come to mind less easily). Because the size principle strongly favours such a small concept when it contains the examples, people generalize a lone example to its own multiples (25 → 50, 75, 100; 33 → 66, 99; 14 → 28, 42) much more than the fixed rule list allows.

### range_prior_sample_size_neglect — pruned (experiment2 end of experiment)

**Outcome:** 62.8 nats behind consistent_examples_bounded_rules (4.4× dse)

**Hypothesis:** Refinement of `range_size_prior_concepts`: people do the same tempered-evidence Bayesian concept learning over the 45 weighted rules, all intervals of 1-100 (with a prior that grows as a power of a range's size), their odd-only and even-only versions, and range-bounded rules. The one change is taken from `sample_size_neglect_evidence`: people neglect sample size, so each example's evidential weight shrinks as more examples are shown. The total weight of the examples grows less than in proportion to their number (no neglect recovers the incumbent). This predicts that a lone example is weighed heavily enough to pin down the small rule it belongs to (6 → 12, 18; 11 → 55, 66; 25 → 50, 100), so its members are endorsed far more than the incumbent allows, while larger sets sharpen generalization less than an ideal learner's would.

### evidence_neglect_prior_lapse — dropped (experiment2 round 1)

**Outcome:** fold 0 too slow to fit: the fit of 'evidence_neglect_prior_lapse' (target_accept 0.99) was still sampling after the 30-minute limit and was stopped. Every sampling run of a candidate's admission fit has a 30-minute limit. Make the model cheaper to evaluate and easier to sample: vectorise the likelihood over trials (no Python loops, pytensor scan or per-trial subgraphs), compute features once per unique sequence (in compute_features or prepare_observed, not in the graph), and drop or merge parameters the data barely constrain — a weakly identified posterior makes NUTS take maximal-length trajectories.

**Hypothesis:** Refinement of `range_size_prior_concepts`. People do the same tempered-evidence Bayesian concept learning as that model: the 45 weighted rules, all intervals with a size-dependent prior, the odd-only and even-only versions, and range-bounded rules. The one change: on an inferred fraction of trials people do not use the examples and answer from prior belief alone, saying "yes" with the prior probability that the program's concept contains the target (a fraction of 0 recovers the incumbent). Prior belief favours broad concepts, so these evidence-neglect lapses raise endorsement of non-example targets the size principle would reject (most of all after a single example, where people generalize more than the incumbent allows) and lower endorsement of the examples themselves, which people accept only about 70% of the time.

### neglect_evidence_guessing_lapse — pruned (experiment2 end of experiment)

**Outcome:** 79.5 nats behind consistent_examples_bounded_rules (5.1× dse)

**Hypothesis:** Refinement of `sample_size_neglect_evidence`: people do the same tempered Bayesian concept learning with sample-size neglect (45 weighted rules, all intervals, their odd/even versions and range-bounded rules, with each extra example adding less evidence), but they do not consult that concept on every trial. On some trials, at an inferred lapse rate, they skip the inference and answer by a biased guess, saying "yes" at an inferred guessing rate whatever the target is. The one change is this guessing lapse (a lapse rate of 0 recovers the original model). It predicts that even targets that are themselves examples, or that clearly belong to the inferred concept, are sometimes rejected (examples get about 0.7–0.9 yes, not 1), and that clear non-members still get a floor of yeses.

### arithmetic_relation_exemplar — pruned (experiment2 end of experiment)

**Outcome:** 121.7 nats behind consistent_examples_bounded_rules (4.0× dse)

**Hypothesis:** People do not infer a hidden rule; they judge a target by its similarity to the stored examples, as in an exemplar (context) model, where two numbers are similar to the extent that they stand in familiar arithmetic relations — one is a multiple of the other, they share a common factor of 3 or more, they end in the same digit, they have the same parity — and each relation the pair lacks shrinks their similarity by an inferred factor. The target's average similarity to the examples, against an internal criterion, sets the chance of "yes", so a lone example is generalized to its own multiples (6 → 18, 25 → 100, 11 → 55) because each such target is arithmetically close to it, while numbers sharing none of these relations with any example are rejected.

### pedagogical_sampling_concepts — pruned (experiment2 end of experiment)

**Outcome:** 131.5 nats behind consistent_examples_bounded_rules (6.5× dse)

**Hypothesis:** People assume the examples were chosen by a helpful teacher, not sampled at random: the program picked numbers that would best lead a learner to its concept (pedagogical sampling). So an example counts as evidence for a concept, whether a mathematical rule or a range, in proportion to how strongly that example on its own would make a naive learner think of that concept, compared with the other numbers the teacher could have shown; people then answer "yes" with their resulting belief that the concept contains the target. This predicts that concepts for which the shown examples would be poor, unrepresentative teaching choices lose belief even when they contain the examples, unlike under the size principle, where any concept that contains the examples gains in proportion to how small it is.

### prior_fallback_evidence_lapse — dropped (experiment2 round 2)

**Outcome:** fold 0 too slow to fit: the fit of 'prior_fallback_evidence_lapse' (target_accept 0.99) was still sampling after the 30-minute limit and was stopped. Every sampling run of a candidate's admission fit has a 30-minute limit. Make the model cheaper to evaluate and easier to sample: vectorise the likelihood over trials (no Python loops, pytensor scan or per-trial subgraphs), compute features once per unique sequence (in compute_features or prepare_observed, not in the graph), and drop or merge parameters the data barely constrain — a weakly identified posterior makes NUTS take maximal-length trajectories.

**Hypothesis:** People infer the hidden concept from the examples by the same Bayesian concept learning as the current best model: weighted mathematical/digit rules, magnitude ranges, their odd/even versions and range-bounded rules, with sample-size neglect. On some trials, at an inferred rate, they do not use the examples at all. On those trials they answer from prior belief alone, saying "yes" as often as they think a program would produce the target before seeing any examples. This disagrees most sharply with the best model in two places. First, the examples themselves are accepted well short of certainty. Second, numbers outside a tight concept (e.g. non-matching digits after 63, 23, 43) keep a floor of yeses that is higher for a-priori plausible numbers, such as members of broad rules like even numbers or multiples of 10, and lower for idiosyncratic ones.

### range_sized_bounded_rules — pruned (experiment2 end of experiment)

**Outcome:** 61.5 nats behind consistent_examples_bounded_rules (4.7× dse)

**Hypothesis:** Refinement of `range_prior_sample_size_neglect` (tempered Bayesian concept learning with sample-size neglect over 45 weighted rules, size-weighted intervals of 1-100, their odd/even versions and range-bounded rules). The one change is that the range-size prior people hold applies to every concept defined by a range, so it applies to a rule confined to a range ("multiples of 11 between 5 and 15") as it does to a plain interval. A bounded rule's prior plausibility grows by the same inferred power of how many members it keeps, rather than being equal for every range, and no new parameter is added. Tiny bounded versions of a rule, such as {11} or {25}, then seem implausible. From a lone example people should generalize to the whole rule it belongs to (11 → 55, 66; 25 → 50, 100), while sets of several examples keep a range-bounded reading.

### digit_rules_need_comparison — pruned (experiment2 end of experiment)

**Outcome:** 43.9 nats behind consistent_examples_bounded_rules (4.8× dse)

**Hypothesis:** Refinement of `range_prior_sample_size_neglect` (the same tempered, sample-size-neglecting Bayesian concept learning over the 45 weighted rules, size-weighted intervals of 1-100, their odd/even versions and range-bounded rules). The one change: a digit pattern ("ends in 6", "contains a 1") is something people notice only by comparing examples, so when a single example is shown, digit-based rules hardly come to mind, and their prior is discounted by an inferred amount (zero recovers the incumbent). Belief then moves to the lone example's arithmetic rules. People should generalize a lone example to its multiples (6 → 12, 18; 11 → 55, 66; 10 → 80) more than the incumbent allows, and endorse numbers that share only its digits (14 → 74, 94) less. Judgments about sets of two or more examples are unchanged.

### open_multiples_guessing_lapse — pruned (experiment2 end of experiment)

**Outcome:** 70.7 nats behind consistent_examples_bounded_rules (4.6× dse)

**Hypothesis:** Refinement of `open_ended_multiples_concepts`: people do the same tempered-evidence Bayesian concept learning over the 45 weighted rules, size-weighted intervals of 1-100, their odd-only and even-only versions, range-bounded rules and the open-ended "multiples of k" family (k from 13 to 50). The one change, taken from `neglect_evidence_guessing_lapse`, is that people do not consult this concept on every trial: at an inferred lapse rate they skip the inference and give a biased guess, answering "yes" at an inferred guessing rate whatever the target is (a lapse rate of 0 recovers the original model). Because the guessing lapse, not weak evidence, accounts for the rejected examples and the floor of yeses, the concept inference itself can be sharper, so members of the examples' rule (including the lone example's own multiples, 25 → 50, 100) are endorsed more and clear non-members less, apart from the guessing floor.

### downward_extrapolation_reluctance — dropped (experiment2 round 3)

**Outcome:** fold 0 too slow to fit: the fit of 'downward_extrapolation_reluctance' (target_accept 0.99) was still sampling after the 30-minute limit and was stopped. Every sampling run of a candidate's admission fit has a 30-minute limit. Make the model cheaper to evaluate and easier to sample: vectorise the likelihood over trials (no Python loops, pytensor scan or per-trial subgraphs), compute features once per unique sequence (in compute_features or prepare_observed, not in the graph), and drop or merge parameters the data barely constrain — a weakly identified posterior makes NUTS take maximal-length trajectories.

**Hypothesis:** People infer the program's concept from the examples as the current best model does (tempered, sample-size-neglecting Bayesian concept learning over weighted rules, size-weighted ranges, their odd/even versions and range-bounded rules, with digit rules discounted for a lone example), but they treat the examples as where the program's output starts, and they think of it as producing numbers counting upward. Accepting a number smaller than every example therefore means extrapolating backward, which people trust less the farther the target lies below the smallest example: the chance of saying yes is the belief that the concept contains the target, discounted by an inferred rate for each step below the smallest example, while targets above or among the examples are not discounted. This predicts that a digit rule's members below the examples (48,78,38,98 → 8, 18; 97,77,87,47 → 7) are endorsed less than its members above them, and in general that people generalize less downward than upward.

### attention_lapse_prior_judgment — dropped (experiment2 round 3)

**Outcome:** fold 0 too slow to fit: the fit of 'attention_lapse_prior_judgment' (target_accept 0.99) was still sampling after the 30-minute limit and was stopped. Every sampling run of a candidate's admission fit has a 30-minute limit. Make the model cheaper to evaluate and easier to sample: vectorise the likelihood over trials (no Python loops, pytensor scan or per-trial subgraphs), compute features once per unique sequence (in compute_features or prepare_observed, not in the graph), and drop or merge parameters the data barely constrain — a weakly identified posterior makes NUTS take maximal-length trajectories.

**Hypothesis:** People do not attend to the examples on every trial. They infer the hidden concept from the examples, using the same Bayesian concept learning as `digit_rules_need_comparison`. But on an inferred share of trials their attention lapses: they do not look at the examples and judge the target alone, answering "yes" with their prior belief that a program would produce that number. This predicts that examples and clear members of a confidently inferred rule are endorsed well short of certainty (a digit rule extended outside the examples' range gets about 0.85 yes, not 0.95). Non-members keep a target-dependent floor of yeses that is higher for numbers belonging to broad, a-priori plausible concepts.

### examples_forgotten_prior_answer — rejected (experiment2 round 3 candidate 2 lens 4)

**Outcome:** predicts like existing model 'attention_lapse_prior_judgment' (p_yes RMSE 0.00195 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of attention_lapse_prior_judgment, not a new hypothesis.

**Hypothesis:** People learn the program's concept by the same tempered Bayesian concept learning as the current best model: weighted mathematical/digit rules, magnitude ranges, odd/even versions and range-bounded rules, with sample-size neglect, and digit rules discounted after a lone example. They have one resource limit, in memory: on some trials, at an inferred rate, the examples are no longer available when the target is judged, so people answer from prior belief alone, saying "yes" as often as they would expect a program to produce that number before seeing any examples. This predicts that confident judgments fall short of certainty. Members of a clear digit rule outside the examples' range, and the examples themselves, get about 0.85 yes, not 0.95. Clear non-members keep a floor of yeses that is higher for a-priori plausible numbers (members of broad rules and wide ranges).

### digit_comparison_guessing_lapse — dropped (experiment2 round 3)

**Outcome:** fold 0 too slow to fit: the fit of 'digit_comparison_guessing_lapse' (target_accept 0.99) was still sampling after the 30-minute limit and was stopped. Every sampling run of a candidate's admission fit has a 30-minute limit. Make the model cheaper to evaluate and easier to sample: vectorise the likelihood over trials (no Python loops, pytensor scan or per-trial subgraphs), compute features once per unique sequence (in compute_features or prepare_observed, not in the graph), and drop or merge parameters the data barely constrain — a weakly identified posterior makes NUTS take maximal-length trajectories.

**Hypothesis:** Refinement of `digit_rules_need_comparison`: people learn concepts the same way. They do tempered, sample-size-neglecting Bayesian concept learning over the 45 weighted rules, size-weighted intervals, their odd/even versions and range-bounded rules, and they discount digit rules when only one example is shown. The one change, taken from `open_multiples_guessing_lapse`, is that people do not consult this concept on every trial. On an inferred share of trials they skip the inference and guess, answering "yes" at an inferred guessing rate whatever the target is (a lapse rate of 0 recovers the incumbent). This predicts that even clear members of a firmly inferred rule are endorsed well short of certainty: the shared final digit beyond the examples' range (59,49,99,69 → 9, 19; 97,77,87,47 → 7) gets about 0.8–0.9 yes, not 0.95. The examples themselves get only about 0.7–0.85, and clear non-members keep a small floor of yeses.

### comparison_dependent_concepts — pruned (experiment2 end of experiment)

**Outcome:** 22.5 nats behind consistent_examples_bounded_rules (4.5× dse)

**Hypothesis:** Refinement of `digit_rules_need_comparison` (tempered, sample-size-neglecting Bayesian concept learning over the 45 weighted rules, size-weighted intervals, their odd/even versions and range-bounded rules, with digit rules discounted when only one example is shown). The one change is to say why the discount happens. Any concept that is defined by a relation among the examples can be noticed only by comparing examples. That covers a shared digit pattern and also a rule confined to the span the examples cover ("multiples of 6 between 4 and 9"). So when a single example is shown, range-bounded rules lose prior plausibility by the same inferred amount as digit rules, and no new parameter is added. People should then generalize a lone example to the whole of its arithmetic rule, including far members (6 → 18, 24; 11 → 55, 99; 25 → 100; 10 → 80), more than the incumbent allows, while judgments about sets of two or more examples are unchanged.

### digit_neglect_sized_bounded_rules — pruned (experiment2 end of experiment)

**Outcome:** 57.4 nats behind consistent_examples_bounded_rules (6.2× dse)

**Hypothesis:** Refinement of `range_sized_bounded_rules` (tempered, sample-size-neglecting Bayesian concept learning over the 45 weighted rules, size-weighted intervals, their odd/even versions, and range-bounded rules whose prior grows with how many members they keep). The one change, grafted from `digit_rules_need_comparison`: a digit pattern ("ends in 6", "contains a 1") is noticed only by comparing examples, so when a single example is shown the digit-based rules hardly come to mind and their prior is discounted by an inferred amount (zero recovers `range_sized_bounded_rules`). The sized range prior already makes tiny bounded rules around a lone example implausible, so the belief taken from digit rules moves to the example's whole arithmetic rules. People should then generalize a lone example to its multiples (6 → 12, 18; 11 → 55, 66; 25 → 50, 100) more than either parent model allows, and endorse its digit neighbours (14 → 74) less, while judgments about sets of two or more examples are unchanged.

### last_example_working_memory — rejected (experiment2 round 3 candidate 2 lens 4 repair 1)

**Outcome:** too slow to fit: the fit of 'last_example_working_memory' (target_accept 0.99) was still sampling after the 30-minute limit and was stopped. Every sampling run of a candidate's admission fit has a 30-minute limit. Make the model cheaper to evaluate and easier to sample: vectorise the likelihood over trials (no Python loops, pytensor scan or per-trial subgraphs), compute features once per unique sequence (in compute_features or prepare_observed, not in the graph), and drop or merge parameters the data barely constrain — a weakly identified posterior makes NUTS take maximal-length trajectories.

**Hypothesis:** People infer the program's concept by the same tempered Bayesian concept learning as the current best model (weighted mathematical/digit rules, size-weighted magnitude ranges, odd/even versions and range-bounded rules, with sample-size neglect and digit rules discounted after a lone example), but working memory is limited: on an inferred share of trials only the most recently shown example is still held in mind when the target is judged, so they generalize from that last example alone. This predicts that earlier examples are endorsed less than the last one, that after several examples people still partly generalize broadly from the final example (to its multiples, digits and neighbourhood), and that the same examples in a different order lead to different judgments.

### rule_only_size_principle — dropped (experiment2 round 4)

**Outcome:** fold 0 too slow to fit: the fit of 'rule_only_size_principle' (target_accept 0.99) was still sampling after the 30-minute limit and was stopped. Every sampling run of a candidate's admission fit has a 30-minute limit. Make the model cheaper to evaluate and easier to sample: vectorise the likelihood over trials (no Python loops, pytensor scan or per-trial subgraphs), compute features once per unique sequence (in compute_features or prepare_observed, not in the graph), and drop or merge parameters the data barely constrain — a weakly identified posterior makes NUTS take maximal-length trajectories.

**Hypothesis:** Refinement of `comparison_dependent_concepts`. The learner is the same: tempered, sample-size-neglecting Bayesian concept learning over weighted rules, size-weighted ranges, their odd/even versions and range-bounded rules, with lone-example discounts. The one change is that people apply the size principle only to rules, whose members they can list. For a magnitude range they only check whether it admits each example, and every admitted example counts as equally likely however wide the range is (as if drawn from a window of one inferred reference width), so how broad a range they favour comes from their prior alone. A range that covers all the examples then beats a small rule that must treat some examples as strays. For sets that no rule covers (71,61,26,81; 70,88,7,22), people should endorse targets inside the examples' span more, and targets outside it less, than the incumbent predicts.

### serial_position_primacy_concepts — dropped (experiment2 round 4)

**Outcome:** fold 0 too slow to fit: the fit of 'serial_position_primacy_concepts' (target_accept 0.99) was still sampling after the 30-minute limit and was stopped. Every sampling run of a candidate's admission fit has a 30-minute limit. Make the model cheaper to evaluate and easier to sample: vectorise the likelihood over trials (no Python loops, pytensor scan or per-trial subgraphs), compute features once per unique sequence (in compute_features or prepare_observed, not in the graph), and drop or merge parameters the data barely constrain — a weakly identified posterior makes NUTS take maximal-length trajectories.

**Hypothesis:** Refinement of `comparison_dependent_concepts`. People learn the program's concept in the same way: tempered, sample-size-neglecting Bayesian concept learning over weighted rules, size-weighted ranges, odd/even versions and range-bounded rules, with comparison-dependent concepts discounted for a lone example. The one change is that examples do not all count equally as evidence. How much an example counts depends on where it appears in the list, so evidence is shifted smoothly toward the first examples (primacy: the opening examples anchor the guess and later ones revise it less) or toward the last ones (recency), by an inferred gradient, while the total evidence stays the same. This predicts that the same examples in a different order produce different generalization; in particular, an example that breaks the others' pattern weakens the tight concept more when it sits at the favoured end of the list.

### coherence_relative_similarity — pruned (experiment2 end of experiment)

**Outcome:** 226.6 nats behind consistent_examples_bounded_rules (6.2× dse)

**Hypothesis:** People judge a target by its similarity to the examples, where two numbers are similar to the extent that they share number properties (parity, primality, being a square/cube/power, divisibility, final digit, digits contained, and a magnitude band of width 20), each shared property counting by how rare, and so informative, it is. The criterion is relative: people say "yes" when the target is about as similar to the examples, on average, as the examples are to one another, so a set whose members are loosely related (no common rule) accepts loosely related targets such as nearby numbers, while a tightly related set (all ending in 2) demands the same tight relation. A lone example, which has no partner to compare with, is judged against a fixed default criterion.

### span_concepts_need_comparison — dropped (experiment2 round 4)

**Outcome:** fold 0 too slow to fit: the fit of 'span_concepts_need_comparison' (target_accept 0.99) was still sampling after the 30-minute limit and was stopped. Every sampling run of a candidate's admission fit has a 30-minute limit. Make the model cheaper to evaluate and easier to sample: vectorise the likelihood over trials (no Python loops, pytensor scan or per-trial subgraphs), compute features once per unique sequence (in compute_features or prepare_observed, not in the graph), and drop or merge parameters the data barely constrain — a weakly identified posterior makes NUTS take maximal-length trajectories.

**Hypothesis:** Refinement of `comparison_dependent_concepts` (tempered, sample-size-neglecting Bayesian concept learning over the 45 weighted rules, size-weighted intervals, their odd/even versions and range-bounded rules, where concepts that can only be noticed by comparing examples, such as digit patterns and rules bounded to the examples' span, lose prior plausibility when only one example is shown). The one change is to apply that principle to every concept defined by a span of examples. A plain magnitude range ("numbers from 20 to 40") is fixed by where the examples lie relative to one another, so a lone example cannot suggest it either. When a single example is shown, ranges are discounted by the same inferred amount as digit and bounded rules, and no parameter is added. A lone example should then be read almost purely as an instance of an arithmetic rule: its far rule members (25 → 100, 66 → 99, 10 → 90) are endorsed more and non-members that lie close to it (6 → 9, 6 → 20, 1 → 27) less than the incumbent predicts, while sets of two or more examples are unchanged.

### comparison_gated_sized_bounded_rules — pruned (experiment2 end of experiment)

**Outcome:** 39.6 nats behind consistent_examples_bounded_rules (5.3× dse)

**Hypothesis:** Refinement of `digit_neglect_sized_bounded_rules`. People do tempered, sample-size-neglecting Bayesian concept learning over the 45 weighted rules, size-weighted intervals, their odd/even versions, and range-bounded rules whose prior grows with how many members they keep. In that model, digit rules are already discounted when only one example is shown. The one change, grafted from `comparison_dependent_concepts`: a rule confined to the span the examples cover also needs a comparison between examples, so with a lone example range-bounded rules lose prior plausibility by the same inferred discount as digit rules. No parameter is added. People should therefore generalize a lone example to the whole of its arithmetic rule, including far members (6 → 18, 24; 25 → 100; 11 → 99), more than either parent allows. For sets of two or more examples, a sized range-bounded reading is kept.

### log_odds_response_bias — dropped (experiment3 round 0)

**Outcome:** fold 0 too slow to fit: the fit of 'log_odds_response_bias' (target_accept 0.99) was still sampling after the 30-minute limit and was stopped. Every sampling run of a candidate's admission fit has a 30-minute limit. Make the model cheaper to evaluate and easier to sample: vectorise the likelihood over trials (no Python loops, pytensor scan or per-trial subgraphs), compute features once per unique sequence (in compute_features or prepare_observed, not in the graph), and drop or merge parameters the data barely constrain — a weakly identified posterior makes NUTS take maximal-length trajectories.

**Hypothesis:** People infer the program's concept exactly as the current best model says (tempered, sample-size-neglecting Bayesian concept learning over weighted rules, size-weighted ranges, their odd/even versions and fully consistent range-bounded rules), but their yes/no answer does not report that belief directly: they apply a fixed response criterion that shifts the odds of their belief by a constant amount toward "yes" (or "no"). Because a constant shift in odds moves uncertain beliefs much more than confident ones, targets people are unsure about, such as in-range numbers after examples that no rule fully explains, should be endorsed noticeably more (for a yes-bias) than the belief alone predicts, while examples and clear non-members barely change.

### round_number_range_concepts — dropped (experiment3 round 0)

**Outcome:** fold 0 too slow to fit: the fit of 'round_number_range_concepts' (target_accept 0.99) was still sampling after the 30-minute limit and was stopped. Every sampling run of a candidate's admission fit has a 30-minute limit. Make the model cheaper to evaluate and easier to sample: vectorise the likelihood over trials (no Python loops, pytensor scan or per-trial subgraphs), compute features once per unique sequence (in compute_features or prepare_observed, not in the graph), and drop or merge parameters the data barely constrain — a weakly identified posterior makes NUTS take maximal-length trajectories.

**Hypothesis:** People think of magnitude ranges in round numbers: besides mathematical and digit rules and arbitrary ranges, they explicitly entertain the few ranges whose ends fall on decade boundaries ("21 to 50", "the 60s and 70s"), and because these round ranges are few, each is a priori far more plausible than an arbitrary range. So when no rule fits all the examples, or only two examples are shown, people generalize to the round range that covers the examples, endorsing numbers inside that decade-bounded span more, and numbers beyond its decade boundaries less, than learners with only rules and arbitrary ranges would.

### stray_aware_consistent_concepts — dropped (experiment3 round 0)

**Outcome:** fold 0 too slow to fit: the fit of 'stray_aware_consistent_concepts' (target_accept 0.99) was still sampling after the 30-minute limit and was stopped. Every sampling run of a candidate's admission fit has a 30-minute limit. Make the model cheaper to evaluate and easier to sample: vectorise the likelihood over trials (no Python loops, pytensor scan or per-trial subgraphs), compute features once per unique sequence (in compute_features or prepare_observed, not in the graph), and drop or merge parameters the data barely constrain — a weakly identified posterior makes NUTS take maximal-length trajectories.

**Hypothesis:** Refinement of `consistent_examples_bounded_rules`. People use the same tempered Bayesian concept learning over weighted rules, size-weighted ranges, their odd/even versions and bounded rules that every example must fit, and they still neglect sample size and discount comparison-based rules when shown one example. The one change: the share of examples people believe are random strays is inferred instead of fixed at 10%. If people expect few strays, a rule that one example breaks loses most of its belief. So for sets that no rule fits fully, people fall back on the range covering all the examples: they should endorse in-range numbers more and out-of-range numbers less than the incumbent predicts.

### stray_averse_concept_learning — rejected (experiment3 round 0 candidate 4 refine incumbent consistent_examples_bounded_rules)

**Outcome:** predicts like existing model 'stray_aware_consistent_concepts' (p_yes RMSE 0.00054 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of stray_aware_consistent_concepts, not a new hypothesis.

**Hypothesis:** Refinement of `consistent_examples_bounded_rules` (the same tempered, sample-size-neglecting Bayesian concept learning over the 45 weighted rules, size-weighted intervals, their odd/even versions and bounded rules that every example fits, with digit and bounded rules discounted when only one example is shown). The one change is in how often people think a shown example is a stray that the program did not produce from its concept. That model fixes this rate at one in ten; here it is inferred, and the claim is that people think strays are rarer. An example that breaks a rule then counts strongly against that rule, so for sets that no rule fits fully, belief moves from rules that most examples fit to the magnitude range the examples span. In-range numbers should then be endorsed more for such rule-less sets, and for pairs of examples, than the incumbent predicts.
