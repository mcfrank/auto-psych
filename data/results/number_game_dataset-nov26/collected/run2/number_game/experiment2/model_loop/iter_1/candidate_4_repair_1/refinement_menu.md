# Refinement menu

The models you may refine, other than the incumbent `seed_overshadowing_last_digit_recency`. Each entry is one mechanism: its hypothesis in full as its author stated it, how it stands, and where its source is. This is a menu, not a list of what is ruled out: a pruned model lost to the best model on the data it was scored on, by the stated margin, but its mechanism may be partly right — and a model that lost narrowly is the most promising target here. Read the source of the model you pick before you write anything.

## Live models other than the incumbent (in the set; best first)

### seed_multiple_digit_overshadowing — rank 1, 4.3 ± 9.8 nats behind the best (0.4× dse: statistically tied with the best), ELPD-LOO -3199.3

**Hypothesis:** Refinement of `final_digit_overshadowing_overlap` by one grafted component, taken from `seed_multiple_salience_overlap`. People still say "yes" in proportion to the salience-weighted share of the examples' shared property profile that the target also has, and a final digit visibly shared by several examples still overshadows every other shared property. The one change is that people also notice when the smallest example divides every other example (7 and 63, 8 and 64). They read that number as the program's seed and add "multiples of the seed" to the shared profile with its own salience, so multiples of the seed that break other shared properties such as parity (14 or 28 for 7, 63) are endorsed far more than the incumbent predicts.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/2485af553ddb9dd3/repo/_runs/cell_1/experiment2/model_loop/models/seed_multiple_digit_overshadowing.py`

### seed_multiple_digit_overshadowing_overlap — rank 2, 13.7 ± 10.4 nats behind the best (1.3× dse: statistically tied with the best), ELPD-LOO -3208.6

**Hypothesis:** Refinement of `final_digit_overshadowing_overlap` by one change, a component grafted from `seed_multiple_salience_overlap`: people also notice a relation between the examples themselves, namely whether the smallest example divides every other example (7, 63; 9, 81; 8, 64). When it does, "multiples of that seed number" joins the examples' shared profile as a property with its own salience. Otherwise the judgment is unchanged: people say "yes" in proportion to the salience-weighted share of the shared profile the target has, and a final digit visibly shared by several examples still overshadows every other shared property, the seed property included. Compared with the incumbent, multiples of the seed that break other shared properties, such as 14 and 28 for the set 7, 63, are endorsed more.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/2485af553ddb9dd3/repo/_runs/cell_1/experiment2/model_loop/models/seed_multiple_digit_overshadowing_overlap.py`

### last_example_digit_recency_overlap — rank 3, 22.4 ± 7.9 nats behind the best (2.8× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -3217.3

**Hypothesis:** Refinement of `final_digit_overshadowing_overlap` by one order-dependent change: people revise a running guess as they read the examples, and the final digit of the last example shown lingers in that guess as a candidate pattern ("ends in 2") even when the earlier examples do not all share it. They still say "yes" in proportion to the salience-weighted share of the examples' shared profile that the target has, but that profile also contains "ends in the same digit as the last example", with a fraction of the final-digit salience. So the same examples in a different order are generalized differently: targets sharing the last-shown example's final digit are endorsed more than targets sharing the first-shown example's.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/2485af553ddb9dd3/repo/_runs/cell_1/experiment2/model_loop/models/last_example_digit_recency_overlap.py`

### graded_consensus_feature_overlap — rank 4, 52.2 ± 15.8 nats behind the best (3.3× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -3247.1

**Hypothesis:** Refinement of `final_digit_overshadowing_overlap` by one change: people judge a target by how much of the examples' property profile it shares, but that profile is not limited to properties that *every* example has. Each property held by some of the examples enters with a weight that grows with the fraction of examples holding it, and how steeply that weight falls off is inferred. A property held by most examples (three of 69, 9, 39, 21 end in 9) or by half of them (13, 91 end in 1 and 3) still pulls generalization toward targets that have it, rather than being ignored as it is when only unanimous properties count.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/2485af553ddb9dd3/repo/_runs/cell_1/experiment2/model_loop/models/graded_consensus_feature_overlap.py`

### final_digit_overshadowing_overlap — rank 5, 95.7 ± 17.8 nats behind the best (5.4× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -3290.6

**Hypothesis:** Refinement of `one_sided_magnitude_bound_overlap` by one change: a final digit that is visibly shared by all of several examples (e.g. 28, 98) overshadows every other shared property. People still say "yes" in proportion to the salience-weighted share of the examples' shared profile that the target has, but when the examples visibly share a final digit, all the other shared properties (parity, multiples, contained digits, magnitude bounds, and so on) lose an inferred fraction of their salience. This disagrees most sharply with the incumbent on shared-final-digit sets: targets that keep the digit but break the other shared properties (28, 98 -> 18, 38) are endorsed far more, and targets that fit the other properties but not the digit (28, 98 -> 14, 56) far less.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/2485af553ddb9dd3/repo/_runs/cell_1/experiment2/model_loop/models/final_digit_overshadowing_overlap.py`

### seed_multiple_salience_overlap — rank 6, 104.1 ± 20.0 nats behind the best (5.2× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -3299.0

**Hypothesis:** People notice a relation between the examples themselves that the property-overlap models ignore: whether the smallest example divides every other example (7 and 63, 9 and 81, 8 and 64). When it does, they read that smallest number as the program's seed and take "multiples of the seed" as a salient property of the set, with its own salience. As in the overlap account, they still say "yes" in proportion to the salience-weighted share of the examples' common properties that the target also has, so a multiple of the seed is endorsed even when it breaks other shared properties such as parity (e.g. 14 and 28 for 7, 63).

**Source:** `/scr/wurgaft/auto-psych/agent_trees/2485af553ddb9dd3/repo/_runs/cell_1/experiment2/model_loop/models/seed_multiple_salience_overlap.py`

### leaky_memory_example_elimination — rank 7, 139.4 ± 18.3 nats behind the best (7.6× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -3334.3

**Hypothesis:** People generalize like a weak-sampling Bayesian learner over salient number properties (the incumbent's salience-weighted overlap, with a visible shared final digit overshadowing other properties), but with one resource limit: working memory for the examples fades, so each earlier example is only remembered — and so only rules out the properties it lacks — with a probability that falls with how many examples were shown after it. Properties of the most recent example that an earlier, partly forgotten example contradicts therefore keep some weight, so targets resembling the last-shown example (e.g. sharing its final digit) are endorsed more than targets resembling the first.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/2485af553ddb9dd3/repo/_runs/cell_1/experiment2/model_loop/models/leaky_memory_example_elimination.py`

### graded_magnitude_band_overlap — no comparison row

**Hypothesis:** Refinement of `seed_overshadowing_last_digit_recency` by one change to its magnitude property. People do not treat "at least as large as the smallest example" and "no larger than the largest example" as all-or-none tests. They judge how close in magnitude the target is to the range the examples span, and that magnitude similarity falls off smoothly (a broad, bell-shaped gradient) with the target's distance below the smallest or above the largest example. So a target just outside the examples keeps nearly full magnitude credit, and only far-away targets lose it. Everything else is unchanged: "yes" follows the salience-weighted share of the examples' shared profile the target has, including the seed-multiple property, final-digit overshadowing and the lingering last-example digit. The main consequence is broader generalization from a single example, where every non-example target lies outside the examples' "range".

**Source:** `/scr/wurgaft/auto-psych/agent_trees/2485af553ddb9dd3/repo/_runs/cell_1/experiment2/model_loop/models/graded_magnitude_band_overlap.py`

### gap_step_progression_overlap — no comparison row

**Hypothesis:** People's space of candidate concepts includes an arithmetic-progression concept that the current models lack. When the examples are evenly spaced (e.g. 8 and 16, or 2 and 8), people entertain "start at the smallest example and keep adding the gap between the examples" (8, 16, 24, 32, ...; 2, 8, 14, 20, ...) as one more property of the set, with its own inferred prior salience. Otherwise they judge the target as before, by the salience-weighted share of the examples' shared property profile it has, so targets on the upward continuation of the progression are endorsed more even when they break other shared properties such as being a power of 2.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/2485af553ddb9dd3/repo/_runs/cell_1/experiment2/model_loop/models/gap_step_progression_overlap.py`

### single_example_leniency_overlap — no comparison row

**Hypothesis:** Refinement of `seed_overshadowing_last_digit_recency` by one change: when only a single example is shown, people know they cannot yet tell which of its properties the program cares about, so they adopt a more lenient criterion and lean toward "yes" for any target, beyond what the overlap with that example's property profile alone implies. Everything else is unchanged (salience-weighted share of the examples' shared profile, multiples of the seed, a visibly shared final digit overshadowing other properties, and the last example's final digit lingering), so the change predicts higher endorsement of non-example targets for one-example sets than for sets of two to four examples, and no difference among the multi-example set sizes.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/2485af553ddb9dd3/repo/_runs/cell_1/experiment2/model_loop/models/single_example_leniency_overlap.py`

## Pruned models (out of the set; narrowest margin first)

### parity_split_one_sided_bound_overlap — pruned (experiment1 end of experiment)

**Margin:** 19.6 nats behind seed_multiple_salience_overlap (2.5× dse)

**Hypothesis:** Refinement of `parity_direction_salience_overlap`: people note every salient mathematical property all the examples share (each property family with its own salience, "all odd" and "all even" with separate saliences, multiples of 5/10 their own family, no size principle) and say "yes" in proportion to the salience-weighted share of that profile the target also has. The one change, grafted from `one_sided_magnitude_bound_overlap`, replaces the two-sided "within the examples' span" property with two one-sided magnitude bounds, "at least as large as the smallest example" and "no larger than the largest example". Each bound has its own salience, so going past the examples upward can cost a target a different amount than going below them, on top of the directional parity asymmetry.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/2485af553ddb9dd3/repo/_runs/cell_1/experiment1/model_loop/models/pruned/parity_split_one_sided_bound_overlap.py`

### one_sided_magnitude_bound_overlap — pruned (experiment1 end of experiment)

**Margin:** 21.6 nats behind seed_multiple_salience_overlap (2.9× dse)

**Hypothesis:** Refinement of `decimal_multiple_salience_overlap`: people still note every salient property all the examples share (each property family with its own salience, base-ten multiples their own family, no size principle) and say "yes" in proportion to the salience-weighted share of that profile the target also has. The one change is to the magnitude concepts people consider: instead of one two-sided "within the examples' span" property, they entertain two separate one-sided bounds — "at least as large as the smallest example" and "no larger than the largest example" — each with its own salience, so if the open-ended "this big or bigger" bound is the more salient, targets above all the examples are endorsed more readily than equally distant targets below them.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/2485af553ddb9dd3/repo/_runs/cell_1/experiment1/model_loop/models/pruned/one_sided_magnitude_bound_overlap.py`

### surface_digit_search_termination — pruned (experiment1 end of experiment)

**Margin:** 24.4 nats behind seed_multiple_salience_overlap (3.1× dse)

**Hypothesis:** People search the examples for shared properties in order of how visible they are, beginning with the written surface of the numbers, and when that first glance turns up a shared final digit they largely stop searching: the deeper arithmetic properties the examples also happen to share (primes, squares/cubes, powers, multiples) are only partly noticed. They then say "yes" in proportion to the salience-weighted share of the noticed shared profile the target also has (with one-sided magnitude bounds and no size principle), so when a common last digit is present, a target that keeps that digit is endorsed strongly even if it breaks an incidental arithmetic pattern such as "multiple of 7".

**Source:** `/scr/wurgaft/auto-psych/agent_trees/2485af553ddb9dd3/repo/_runs/cell_1/experiment1/model_loop/models/pruned/surface_digit_search_termination.py`

### salience_scaled_decisiveness_overlap — pruned (experiment1 end of experiment)

**Margin:** 25.0 nats behind seed_multiple_salience_overlap (2.7× dse)

**Hypothesis:** Refinement of `one_sided_magnitude_bound_overlap`: people still note every salient property all the examples share (each property family with its own salience, base-ten multiples their own family, two one-sided magnitude bounds, no size principle) and judge a target by the salience-weighted share of that profile it also has. The one change is that how decisively they act on that share depends on how salient the examples' common pattern is: when the examples share a highly noticeable property (such as a common final digit or being powers of the same number), a target that fits the profile is endorsed much more confidently than one that fully fits a profile made only of weak, unremarkable properties (such as merely being even), so the pull toward "yes" grows with the total salience of the shared profile rather than with the matched share alone.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/2485af553ddb9dd3/repo/_runs/cell_1/experiment1/model_loop/models/pruned/salience_scaled_decisiveness_overlap.py`

### shared_final_digit_rule_overlap — pruned (experiment1 end of experiment)

**Margin:** 26.0 nats behind seed_multiple_salience_overlap (2.8× dse)

**Hypothesis:** Refinement of `one_sided_magnitude_bound_overlap`: people still note every salient property all the examples share (each family with its own salience, base-ten multiples their own family, two one-sided magnitude bounds, no size principle) and say "yes" in proportion to the salience-weighted share of that profile the target also has. The one change is that a final digit shared by two or more examples is not just one diluted property in that profile. People read it as a self-evident rule ("they all end in 8"), so a target that ends in that digit gets an extra, independent boost toward "yes". That boost does not shrink when the examples also happen to share other, coincidental properties (e.g. 28 and 98 both being multiples of 7).

**Source:** `/scr/wurgaft/auto-psych/agent_trees/2485af553ddb9dd3/repo/_runs/cell_1/experiment1/model_loop/models/pruned/shared_final_digit_rule_overlap.py`

### parity_direction_salience_overlap — pruned (experiment1 end of experiment)

**Margin:** 27.2 nats behind seed_multiple_salience_overlap (2.6× dse)

**Hypothesis:** Refinement of `family_salience_feature_overlap` (carried over in the form its successor `decimal_multiple_salience_overlap` established, with multiples of 5 and 10 as their own salience family): people note every salient mathematical property the examples all share and say "yes" in proportion to the salience-weighted share of that profile the target also has, with no size principle. The one new change, grafted from the directional parity contrast of `asymmetric_decimal_contrast_similarity`, is that "all odd" and "all even" are not one parity family with a single salience: each has its own inferred salience, so a shared odd (or even) pattern can weigh far more in the judgment than the opposite parity pattern does, and breaking it costs a target correspondingly more.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/2485af553ddb9dd3/repo/_runs/cell_1/experiment1/model_loop/models/pruned/parity_direction_salience_overlap.py`

### directional_span_salience_overlap — pruned (experiment1 end of experiment)

**Margin:** 31.6 nats behind seed_multiple_salience_overlap (3.3× dse)

**Hypothesis:** Refinement of `decimal_multiple_salience_overlap`: people still note every salient mathematical property the examples all share (with multiples of 5/10 as their own salient family) and say "yes" in proportion to the salience-weighted share of that profile the target also has, with no size principle. The one change is the "lies within the examples' range" property: instead of an all-or-none, symmetric in-span check, a target outside the span keeps partial credit that fades with its distance from the span, and it fades at a different, inferred rate above the largest example than below the smallest — people readily extend a pattern upward to larger numbers (as if the program counts up) but much less downward.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/2485af553ddb9dd3/repo/_runs/cell_1/experiment1/model_loop/models/pruned/directional_span_salience_overlap.py`

### directional_span_salience_overlap_2 — pruned (experiment1 end of experiment)

**Margin:** 31.6 nats behind seed_multiple_salience_overlap (3.3× dse)

**Hypothesis:** Refinement of `decimal_multiple_salience_overlap`: people still note every salient mathematical property the examples all share (with multiples of 5/10 as their own salience family) and say "yes" in proportion to the salience-weighted share of that profile the target also has, with no size principle. The one change is to the "lies within the examples' span" property: instead of an all-or-none, symmetric in-span test, the span match fades gradually with how far the target lies beyond the examples, at separately inferred rates above the largest example and below the smallest one — people extend a set upward toward larger numbers more readily than downward toward smaller ones.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/2485af553ddb9dd3/repo/_runs/cell_1/experiment1/model_loop/models/pruned/directional_span_salience_overlap_2.py`

### decimal_multiple_salience_overlap — pruned (experiment1 end of experiment)

**Margin:** 32.7 nats behind seed_multiple_salience_overlap (3.3× dse)

**Hypothesis:** Refinement of `family_salience_feature_overlap`: people still note every salient mathematical property the examples all share and say "yes" in proportion to the salience-weighted share of that profile the target also has, with no size principle. The one change is that multiples of 5 and of 10 — the round, base-ten multiples — form their own salience family, separate from other multiples (of 3, 4, 6, 7, 8, 9, 11, 12), because decimal notation makes them far more noticeable; so when examples like 90 and 45 share both "multiple of 5" and "multiple of 9", people generalize along the multiple-of-5 property much more than along the other one.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/2485af553ddb9dd3/repo/_runs/cell_1/experiment1/model_loop/models/pruned/decimal_multiple_salience_overlap.py`

### family_salience_feature_overlap — pruned (experiment1 end of experiment)

**Margin:** 56.1 nats behind seed_multiple_salience_overlap (4.2× dse)

**Hypothesis:** Refinement of `shared_feature_overlap`: people still note every salient mathematical property the examples all share and say "yes" in proportion to how much of that shared profile the target also has, with no size principle. The one change is that the properties are not equally salient: each family of property (parity, primes, squares/cubes, powers, multiples, final digit, contained digit, and lying within the examples' span) gets its own inferred salience weight, so a shared feature of a highly salient kind (e.g. a common final digit) dominates the judgment while shared features of less noticed kinds barely count.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/2485af553ddb9dd3/repo/_runs/cell_1/experiment1/model_loop/models/pruned/family_salience_feature_overlap.py`

### position_weighted_feature_tally — pruned (experiment1 end of experiment)

**Margin:** 60.9 nats behind seed_multiple_salience_overlap (4.4× dse)

**Hypothesis:** People build up their sense of which mathematical properties matter by reading the examples in order and keeping a running tally of each property (parity, primes, squares/cubes, powers, multiples, final digit, contained digit), in which examples at certain positions count more than others — the most recent ones if memory fades, the first ones if the first example anchors the guess. They then say "yes" in proportion to how much of that order-weighted, salience-weighted property profile (plus lying within the examples' span) the target also has, so a property carried by the favoured examples but not by the rest still drives generalization, and the same examples in a different order would be generalized differently.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/2485af553ddb9dd3/repo/_runs/cell_1/experiment1/model_loop/models/pruned/position_weighted_feature_tally.py`

### violation_penalty_feature_test — pruned (experiment1 end of experiment)

**Margin:** 77.0 nats behind seed_multiple_salience_overlap (3.9× dse)

**Hypothesis:** Refinement of `family_salience_feature_overlap`: people still note the salient mathematical properties all the examples share (parity, primes, squares/cubes, powers, multiples, final digit, contained digit, lying within the examples' span), each family with its own salience, and use no size principle. The one change is the combination rule: instead of a compensatory share of the profile the target matches (where matching parity or span can make up for breaking the shared final digit), each shared property the target violates independently cuts the chance of a "yes" by an amount set by its family's salience — a non-compensatory, noisy-AND test in which violating a highly salient shared property (e.g. a common final digit) is close to disqualifying no matter what else the target shares.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/2485af553ddb9dd3/repo/_runs/cell_1/experiment1/model_loop/models/pruned/violation_penalty_feature_test.py`

### evidence_weighted_concept_learning — pruned (experiment1 end of experiment)

**Margin:** 91.8 nats behind seed_multiple_salience_overlap (4.8× dse)

**Hypothesis:** People do Bayesian concept learning over mathematical rules and numeric intervals with the size principle, as in the Independent model of Bigelow & Piantadosi, but they systematically mis-weight the evidence of the examples relative to their prior (Edwards-style conservatism or over-confidence): the examples' likelihood enters their belief update raised to a single inferred power. When that power is below one, people update too little from the examples, so tight rules and narrow ranges dominate less and generalization stays broader than the normative learner's; above one, they over-commit to the tightest fitting concept.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/2485af553ddb9dd3/repo/_runs/cell_1/experiment1/model_loop/models/pruned/evidence_weighted_concept_learning.py`

### evidence_weighted_concepts_with_lapse — pruned (experiment1 end of experiment)

**Margin:** 93.5 nats behind seed_multiple_salience_overlap (5.1× dse)

**Hypothesis:** Refinement of `evidence_weighted_concept_learning`: people still do Bayesian concept learning over the same 45 mathematical rules and all intervals of 1-100 with the size principle, mis-weighting the examples' evidence by one inferred power, and usually answer "yes" with the posterior probability that the concept contains the target. The one change is that on an inferred fraction of trials people do not consult the inferred concept at all (inattention) and instead answer from a fixed, inferred bias toward "yes", so targets that clearly violate the concept still get a floor of endorsement and targets that clearly fit it are sometimes rejected, rather than near-certain all-or-none answers.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/2485af553ddb9dd3/repo/_runs/cell_1/experiment1/model_loop/models/pruned/evidence_weighted_concepts_with_lapse.py`

### conservative_size_principle_conjunction — pruned (experiment1 end of experiment)

**Margin:** 97.9 nats behind seed_multiple_salience_overlap (5.1× dse)

**Hypothesis:** Refinement of rule_within_range_conjunction: people still infer a conjunctive concept — a mathematical rule (or "any number") restricted to a range of magnitudes — choosing rule and range together by Bayesian size-principle reasoning, and say "yes" with the posterior probability that the concept contains the target. The one change is that people are conservative about the evidence: they weigh how well each candidate concept explains the examples with an inferred strength (the likelihood raised to a power below one if they are conservative), so the size principle pulls them less sharply toward the tightest rule and range, and they keep partial belief in broader concepts, such as any number within the examples' span, even when a narrow rule fits better.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/2485af553ddb9dd3/repo/_runs/cell_1/experiment1/model_loop/models/pruned/conservative_size_principle_conjunction.py`

### log_scale_feature_prototype — pruned (experiment1 end of experiment)

**Margin:** 143.3 nats behind seed_multiple_salience_overlap (5.8× dse)

**Hypothesis:** People do not infer a rule; they abstract a prototype from the examples — the average profile of mathematical properties the examples have (how many are even, odd, prime, squares, powers, multiples of each number, multiples of 5/10, end in each digit, contain each digit) together with a central magnitude on a compressed, logarithmic number line — and say "yes" in proportion to how similar the target is to that prototype, with similarity falling off with the attention-weighted mismatch in properties and the log-scale distance from the prototype's centre. Because the number line is logarithmic, a target the same numerical distance above the examples is closer to the prototype than one below them, so people extend upward more readily than downward.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/2485af553ddb9dd3/repo/_runs/cell_1/experiment1/model_loop/models/pruned/log_scale_feature_prototype.py`

### recency_weighted_example_evidence — pruned (experiment1 end of experiment)

**Margin:** 149.6 nats behind seed_multiple_salience_overlap (5.6× dse)

**Hypothesis:** People do Bayesian concept learning over mathematical rules and magnitude intervals (size principle), but they read the examples one at a time and memory for earlier examples fades: each example's evidential force is discounted by how many examples were read after it. Earlier examples therefore constrain the inferred concept less than the most recent one, so the same examples shown in a different order would be generalized differently, with targets resembling the last-shown example favoured.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/2485af553ddb9dd3/repo/_runs/cell_1/experiment1/model_loop/models/pruned/recency_weighted_example_evidence.py`

### rule_within_range_conjunction — pruned (experiment1 end of experiment)

**Margin:** 150.7 nats behind seed_multiple_salience_overlap (6.2× dse)

**Hypothesis:** People infer a conjunctive concept: a mathematical rule (e.g. "even", "multiples of 7", or "any number") restricted to a range of magnitudes, and they infer both the rule and the range from the examples by Bayesian size-principle reasoning. Because the range can be tight around the examples, a target that fits the rule but lies far outside the span of the examples is judged less likely to be produced than one inside it.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/2485af553ddb9dd3/repo/_runs/cell_1/experiment1/model_loop/models/pruned/rule_within_range_conjunction.py`

### asymmetric_decimal_contrast_similarity — pruned (experiment1 end of experiment)

**Margin:** 152.5 nats behind seed_multiple_salience_overlap (5.8× dse)

**Hypothesis:** People judge a target by its asymmetric (Tversky-style) similarity to every example at once: similarity falls off with numerical distance and with mismatches in the decimal form — a different final digit, and the divisibility-by-2 and divisibility-by-5 properties visible in that digit — and the target must resemble all the examples (the distances to each example add up, so one badly mismatched example drags the judgment down). The comparison is directional: a property the examples have that the target lacks (e.g. examples even or multiples of 5, target not) costs a different, inferred amount than a property the target has that the examples lack, so violating an even or base-10 pattern is punished more (or less) than violating an odd pattern.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/2485af553ddb9dd3/repo/_runs/cell_1/experiment1/model_loop/models/pruned/asymmetric_decimal_contrast_similarity.py`

### rule_range_width_power_prior — pruned (experiment1 end of experiment)

**Margin:** 153.9 nats behind seed_multiple_salience_overlap (6.2× dse)

**Hypothesis:** Refinement of `rule_within_range_conjunction`: people still infer a conjunctive concept, a mathematical rule (or "any number") restricted to a range of magnitudes, inferring rule and range from the examples by Bayesian size-principle reasoning and answering "yes" with the posterior probability that the concept contains the target. The one change is the prior over the restricting range: instead of every sub-range being equally likely, a range's prior weight scales as a power of its width (exponent inferred), so people may come in expecting narrow ranges hugging the examples (generalizing less beyond them) or broad ones (generalizing further) — the width preference, not just the examples, governs how far a rule is extended outside the examples' span.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/2485af553ddb9dd3/repo/_runs/cell_1/experiment1/model_loop/models/pruned/rule_range_width_power_prior.py`

### shared_feature_overlap — pruned (experiment1 end of experiment)

**Margin:** 157.2 nats behind seed_multiple_salience_overlap (7.5× dse)

**Hypothesis:** People do not weigh hypotheses by how tightly they fit the examples (no size principle); instead they note every salient mathematical property the examples all share (parity, prime, square, cube, powers, multiples, final digit, contained digit, and lying within the examples' span) and say "yes" in proportion to the fraction of those shared properties the target also has. This disagrees most sharply with the Bayesian rule-learning model where the examples share both a broad and a narrow property (e.g. 16,32,2,8 are all even and all powers of 2): the Bayesian model concentrates on the narrow rule and rejects merely-even targets, whereas feature-overlap gives such targets graded, intermediate endorsement, and it gives far out-of-range rule-matching targets less endorsement because they fail the shared "within the span" property.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/2485af553ddb9dd3/repo/_runs/cell_1/experiment1/model_loop/models/pruned/shared_feature_overlap.py`

### interval_width_power_prior — pruned (experiment1 end of experiment)

**Margin:** 171.0 nats behind seed_multiple_salience_overlap (6.2× dse)

**Hypothesis:** A refinement of bigelow_piantadosi_independent: people still do Bayesian concept learning over the same 45 rules (each with its own inferred prior weight, lambda_0 for the rules as a group) and all intervals of 1-100, with the size principle and 10% uniform example noise, answering "yes" with the posterior probability that the concept contains the target. The one change is that intervals are no longer equally likely a priori: an interval's prior weight scales as a power of its width (exponent inferred), so people can prefer narrow, tight ranges around the examples (or broad ones), which governs how far they generalize beyond the range of the examples.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/2485af553ddb9dd3/repo/_runs/cell_1/experiment1/model_loop/models/pruned/interval_width_power_prior.py`

### erlang_interval_independent_rules — pruned (experiment1 end of experiment)

**Margin:** 182.1 nats behind seed_multiple_salience_overlap (6.2× dse)

**Hypothesis:** Refinement of bigelow_piantadosi_independent: people do Bayesian concept learning over the same 45 rules, each with its own inferred prior weight, plus intervals of 1-100, sampling examples from the concept (size principle, 10% noise) and saying "yes" with the posterior probability that the concept contains the target. The one change is the interval prior: instead of every interval being equally likely a priori, the prior over intervals follows an Erlang density on interval size (scale inferred, as in tenenbaum_rules_similarity), so people favour intervals of a characteristic, typically modest, width and generalize less far beyond the range of the examples.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/2485af553ddb9dd3/repo/_runs/cell_1/experiment1/model_loop/models/pruned/erlang_interval_independent_rules.py`

### decimal_exemplar_similarity — pruned (experiment1 end of experiment)

**Margin:** 194.6 nats behind seed_multiple_salience_overlap (7.6× dse)

**Hypothesis:** People do not induce rules or concepts at all; they judge a target by its summed similarity to the remembered examples (exemplar generalization, as in the Generalized Context Model). Numbers are represented in their decimal written form, so similarity decays exponentially with a weighted city-block distance that combines how far apart the numbers are in magnitude and whether they share their ones digit, with a learned attention weight between the two; the higher the total similarity relative to a response criterion, the more likely a "yes".

**Source:** `/scr/wurgaft/auto-psych/agent_trees/2485af553ddb9dd3/repo/_runs/cell_1/experiment1/model_loop/models/pruned/decimal_exemplar_similarity.py`

### family_weighted_rule_prior — pruned (experiment1 end of experiment)

**Margin:** 196.4 nats behind seed_multiple_salience_overlap (5.6× dse)

**Hypothesis:** Refinement of `bigelow_piantadosi_mixture`: people do Bayesian concept learning over the same 45 rules and all intervals of 1-100 (size principle, 10% noisy examples, "yes" = posterior probability the concept contains the target), but their prior over rules is organised by kind of rule rather than flat. The one change: the rules are grouped into seven families (parity, primes, squares/cubes, powers, multiples, final digit, contained digit), each family gets its own inferred share of the rule prior, and the rules within a family are equally likely — so a few interpretable family preferences (e.g. arithmetic multiples and powers over digit-based rules) rather than one weight per rule or none explain which rules people readily entertain.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/2485af553ddb9dd3/repo/_runs/cell_1/experiment1/model_loop/models/pruned/family_weighted_rule_prior.py`

### bigelow_piantadosi_independent — pruned (experiment1 end of experiment)

**Margin:** 201.3 nats behind seed_multiple_salience_overlap (6.8× dse)

**Hypothesis:** Bayesian concept learning over 45 rules (odd, even, primes, squares, cubes, powers of 2-10, powers of 2 without 32 or with 37, multiples of 3-9, 11 and 12, ending in 0-9, containing the digit 0-9) and every interval of 1-100, with a prior weight inferred for each rule separately and a share lambda_0 for the rules as a group; each interval has prior (1 - lambda_0) / 100^2. Examples are sampled from the concept (the size principle), 10% of them uniformly from 1-100, and "yes" is the posterior probability that the concept contains the target.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/2485af553ddb9dd3/repo/_runs/cell_1/experiment1/model_loop/models/pruned/bigelow_piantadosi_independent.py`

### upward_counting_progression — pruned (experiment1 end of experiment)

**Margin:** 208.3 nats behind seed_multiple_salience_overlap (8.3× dse)

**Hypothesis:** People think of the hidden program as a counter: it starts at some number and keeps adding a fixed step, producing numbers upward until it runs out of room at 100. They infer the step and the starting point from the examples (preferring starting points at or just below the smallest example and favouring counters that would make the examples likely to be hit, so more examples narrow the guess), and say "yes" with their belief that such a counter would reach the target — so a target beyond the largest example on the counter's path is readily accepted, while one below the smallest example is accepted only if the counter is thought to start that low.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/2485af553ddb9dd3/repo/_runs/cell_1/experiment1/model_loop/models/pruned/upward_counting_progression.py`

### posterior_confidence_criterion — pruned (experiment1 end of experiment)

**Margin:** 241.7 nats behind seed_multiple_salience_overlap (9.5× dse)

**Hypothesis:** People infer the program's concept exactly as the simplest Bayesian rule-and-interval learner does (equally likely rules, size principle), but they do not answer "yes" with the probability they believe the target belongs (probability matching); instead they compare their confidence to an internal criterion and answer "yes" only when it is high enough, with a soft, inferred-steepness threshold. This sharpens fairly-likely targets toward "yes" and doubtful ones toward "no", so sets differ more starkly in overall endorsement and targets violating the favoured rule are rejected more firmly than graded posterior beliefs would imply.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/2485af553ddb9dd3/repo/_runs/cell_1/experiment1/model_loop/models/pruned/posterior_confidence_criterion.py`

### most_specific_rule_winner_take_all — pruned (experiment1 end of experiment)

**Margin:** 263.5 nats behind seed_multiple_salience_overlap (10.4× dse)

**Hypothesis:** People do not weigh many hypotheses at once; they settle on a single concept, the most specific (smallest) mathematical rule that every example satisfies (e.g. "powers of 2" rather than "even" for 16, 32, 2, 8), and answer "yes" essentially only to targets that obey that one rule, with occasional noisy answers. Only when no mathematical rule fits all the examples do they fall back on the range spanned by the examples. This predicts all-or-none generalization: merely-even targets for a powers-of-2 set, and out-of-range rule-matching targets, get no graded endorsement beyond noise.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/2485af553ddb9dd3/repo/_runs/cell_1/experiment1/model_loop/models/pruned/most_specific_rule_winner_take_all.py`

### bigelow_piantadosi_mixture — pruned (experiment1 end of experiment)

**Margin:** 339.7 nats behind seed_multiple_salience_overlap (9.0× dse)

**Hypothesis:** Bayesian concept learning over 45 rules (odd, even, primes, squares, cubes, powers of 2-10, powers of 2 without 32 or with 37, multiples of 3-9, 11 and 12, ending in 0-9, containing the digit 0-9) and every interval of 1-100. A rule is chosen with probability lambda_0, and the rules are equally likely; each interval has prior (1 - lambda_0) / 100^2. Examples are sampled from the concept (the size principle), 10% of them uniformly from 1-100, and "yes" is the posterior probability that the concept contains the target.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/2485af553ddb9dd3/repo/_runs/cell_1/experiment1/model_loop/models/pruned/bigelow_piantadosi_mixture.py`

### tenenbaum_rules_similarity — pruned (experiment1 end of experiment)

**Margin:** 361.6 nats behind seed_multiple_salience_overlap (7.9× dse)

**Hypothesis:** Bayesian concept learning over 33 mathematical rules (odd, even, squares, cubes, primes, multiples of 3-12, powers of 2-10, ending in 1-9) and every interval of 1-100. The rules share a prior fraction lambda equally; the intervals share the rest by an Erlang density on their size (scale sigma). Examples are sampled from the concept (the size principle), 10% of them uniformly from 1-100, and "yes" is the posterior probability that the concept contains the target.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/2485af553ddb9dd3/repo/_runs/cell_1/experiment1/model_loop/models/pruned/tenenbaum_rules_similarity.py`
