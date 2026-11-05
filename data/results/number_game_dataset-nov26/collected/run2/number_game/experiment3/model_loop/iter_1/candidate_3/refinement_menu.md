# Refinement menu

The models you may refine, other than the incumbent `single_example_random_program_doubt`. Each entry is one mechanism: its hypothesis in full as its author stated it, how it stands, and where its source is. This is a menu, not a list of what is ruled out: a pruned model lost to the best model on the data it was scored on, by the stated margin, but its mechanism may be partly right — and a model that lost narrowly is the most promising target here. Read the source of the model you pick before you write anything.

## Live models other than the incumbent (in the set; best first)

### size_discounted_anything_concept — rank 1, 14.9 ± 6.5 nats behind the best (2.3× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -4763.3

**Hypothesis:** People's space of candidate concepts includes "the program can produce any number at all", a concept every target fits, which they weigh by the size principle: because it is the broadest concept, each additional example that happens to fit a narrower pattern makes it less credible, by an inferred factor per example. So after one example the anything-goes concept carries much of the judgment and people say "yes" broadly with little regard to the target's distance or properties, while with three or four examples it has faded and "yes" follows the salience-weighted share of the examples' shared profile (with the in-span interval concept) as in the current best account.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/2485af553ddb9dd3/repo/_runs/cell_1/experiment3/model_loop/models/size_discounted_anything_concept.py`

### range_band_single_leniency — rank 2, 17.1 ± 8.2 nats behind the best (2.1× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -4765.5

**Hypothesis:** Refinement of `range_tightened_magnitude_band` (chosen: statistically tied with the best and already separating broad single-example from tight multi-example magnitude generalization) by one grafted component, taken from `progression_band_single_leniency`: when only a single example is shown, people know they cannot yet tell which of its properties the program cares about, so they adopt a laxer decision criterion and lean toward "yes" for every target by an inferred amount. Everything else is the refined model unchanged (salience-weighted share of the examples' shared profile with a magnitude band that is broad for one example and tighter outside the span of several, progression concept, seed multiples, final-digit overshadowing, lingering first- and last-example digits), so one-example sets are endorsed more, and more uniformly across distance from the example, than the profile match alone implies — the single_vs_multi_yes_gap and single_example_distance_gradient discrepancies the best model fails — with no change for two- to four-example sets.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/2485af553ddb9dd3/repo/_runs/cell_1/experiment3/model_loop/models/range_band_single_leniency.py`

### evidence_scaled_decisiveness — rank 3, 28.4 ± 11.3 nats behind the best (2.5× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -4776.8

**Hypothesis:** People infer the program's concept as the current best account says (the salience-weighted share of the examples' shared property profile, including the stretch between the examples), but the decision rule that turns that belief into a yes/no answer depends on how much evidence they have: the fewer examples shown, the noisier and less decisive their answers, drifting toward a coin flip. So after a single example poorly matching targets are endorsed more and targets are less sharply differentiated, while with three or four examples the same degree of match is acted on more decisively.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/2485af553ddb9dd3/repo/_runs/cell_1/experiment3/model_loop/models/evidence_scaled_decisiveness.py`

### span_interval_similarity — rank 4, 29.5 ± 13.1 nats behind the best (2.3× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -4777.9

**Hypothesis:** When two or more examples are shown, people also treat the stretch of the number line between the smallest and largest example as a candidate concept in its own right ("numbers in this range"), and a target earns credit for it by its similarity to that interval: full credit anywhere inside it, decaying exponentially (Shepard's generalization gradient, at an inferred and fairly steep rate) with the target's distance outside it. This interval similarity enters, with its own salience, the salience-weighted share of the examples' shared profile that drives "yes" in the current best account, so targets lying between the examples are endorsed more than equally nondescript targets outside them, even when they share no digit, parity or seed-multiple with the examples.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/2485af553ddb9dd3/repo/_runs/cell_1/experiment3/model_loop/models/span_interval_similarity.py`

### range_tightened_magnitude_band — rank 5, 32.7 ± 13.4 nats behind the best (2.5× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -4781.1

**Hypothesis:** Refinement of `progression_band_first_digit_primacy` (the incumbent) by one change to its graded magnitude band: the band's width depends on whether the examples mark out a range. A single example gives no range, so magnitude credit falls off broadly around it, but once two or more examples are shown people take the span they cover as the program's range, and magnitude credit falls off more steeply (by an inferred fraction of the single-example width) below the smallest and above the largest example. Everything else is the incumbent unchanged, so in multi-example sets targets inside the examples' range that share no other property are endorsed more than targets outside it (the inrange_nonshared_rate critique), while single-example sets still generalize broadly.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/2485af553ddb9dd3/repo/_runs/cell_1/experiment3/model_loop/models/range_tightened_magnitude_band.py`

### span_sharpened_magnitude_band — rank 6, 33.9 ± 13.3 nats behind the best (2.5× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -4782.3

**Hypothesis:** Refinement of the incumbent `progression_band_first_digit_primacy` by one change to its graded magnitude band: how sharply magnitude credit falls off outside the examples depends on whether the examples mark out a span. A lone example gives no edges, so people generalize by magnitude broadly; once two or more examples are shown, their smallest and largest values become real boundaries, and credit falls off much more steeply (its own inferred width) beyond them. Everything else is the incumbent unchanged, so in multi-example sets, targets inside the examples' range that share nothing else with them are endorsed more than equally nondescript targets outside it, closing the inrange_nonshared_rate critique, while single-example generalization stays broad.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/2485af553ddb9dd3/repo/_runs/cell_1/experiment3/model_loop/models/span_sharpened_magnitude_band.py`

### incremental_band_narrowing — rank 7, 35.7 ± 14.2 nats behind the best (2.5× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -4784.1

**Hypothesis:** People read the examples one at a time and keep a running guess about how widely the program's numbers spread in magnitude. After a single example the guess is broad, but each additional example revises it and tightens how far beyond the examples' range they are willing to generalize. So targets outside the range lose endorsement faster with distance the more examples have been shown, while targets inside the range keep full magnitude credit, and the rest of the judgment (the salience-weighted share of the examples' shared properties) is unchanged.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/2485af553ddb9dd3/repo/_runs/cell_1/experiment3/model_loop/models/incremental_band_narrowing.py`

### salience_scaled_span_decisiveness — rank 8, 41.1 ± 13.4 nats behind the best (3.1× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -4789.5

**Hypothesis:** Refinement of the incumbent `span_interval_similarity` by one change, grafted from `salience_scaled_decisiveness_overlap`: how decisively people act on a target's match to the examples' shared profile grows with how much the examples visibly have in common. When the examples share many or highly salient properties (a common final digit, a seed and its multiples, an arithmetic progression), a well-matching target is endorsed and a mismatching one rejected far more sharply than when the examples share only a few weak properties, where answers stay close to the baseline whatever the target; the judged share itself (including the in-span interval similarity) is the incumbent's unchanged, so items are differentiated more sharply on strongly patterned sets (the item_mean_variance critique).

**Source:** `/scr/wurgaft/auto-psych/agent_trees/2485af553ddb9dd3/repo/_runs/cell_1/experiment3/model_loop/models/salience_scaled_span_decisiveness.py`

### first_impression_span_interior — rank 9, 64.5 ± 16.7 nats behind the best (3.9× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -4812.9

**Hypothesis:** Refinement of `first_impression_anchoring` by one change: when two or more examples are shown, people also take "a number lying between the smallest and the largest example" as a candidate property of the program (the range the examples fill in), with its own inferred salience, overshadowed like the other non-digit properties by a visible shared final digit. Everything else is unchanged (the first impression's contradicted properties keep a fraction of their weight, graded magnitude band, progression concept, seed multiples, the last example's lingering digit), so on multi-example sets targets strictly inside the examples' range are endorsed more than targets just outside it even when they share no digit, parity or seed-multiple with the examples, closing the inrange_nonshared_rate critique.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/2485af553ddb9dd3/repo/_runs/cell_1/experiment3/model_loop/models/first_impression_span_interior.py`

### progression_band_first_digit_primacy — rank 10, 67.7 ± 16.5 nats behind the best (4.1× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -4816.1

**Hypothesis:** Refinement of `progression_graded_magnitude_band` (the incumbent) by one grafted component, taken from `graded_band_first_digit_primacy`: when several examples are shown without a shared final digit, the final digit of the FIRST example also lingers as a candidate pattern ("ends in 3"), with its own inferred salience, because the first example anchors people's initial guess before the later examples arrive. Everything else is the incumbent's judgment unchanged (salience-weighted share of the examples' shared profile, with the graded magnitude band, the "start at the smallest and add the gap" progression concept, multiples of the seed, final-digit overshadowing and the last example's lingering digit), so targets sharing the first-shown example's final digit are endorsed more than targets sharing no example's final digit, closing the first_example_digit_lift critique the incumbent fails.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/2485af553ddb9dd3/repo/_runs/cell_1/experiment3/model_loop/models/progression_band_first_digit_primacy.py`

### coincidence_gated_first_digit_primacy — rank 11, 95.3 ± 16.5 nats behind the best (5.8× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -4843.7

**Hypothesis:** Refinement of `coincidence_gated_pattern_belief` by one grafted component, taken from `graded_band_first_digit_primacy`: when people believe the program follows a rule, the final digit of the FIRST example shown also lingers as a candidate pattern ("ends in 3"), with its own inferred salience, because the first example anchors their initial guess before the later examples arrive. Everything else is unchanged: people first judge from how striking the examples' agreement is whether the program is patterned at all (a single example or broadly shared properties leave them suspecting it is random, so they lean toward "yes"), and otherwise say "yes" by the salience-weighted share of the examples' shared profile the target has, the last example's lingering digit included. So on multi-example sets without a shared final digit, targets sharing the first-shown example's final digit are endorsed more than targets sharing no example's final digit, which is the first_example_digit_lift discrepancy that the best model misses.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/2485af553ddb9dd3/repo/_runs/cell_1/experiment3/model_loop/models/coincidence_gated_first_digit_primacy.py`

### first_impression_anchoring — rank 12, 112.5 ± 19.7 nats behind the best (5.7× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -4860.9

**Hypothesis:** People read the examples in order and form their first guess about the program from the first example alone, entertaining every salient property it has (its parity, the numbers it is a multiple of, its digits, its final digit). Later examples should rule out the first example's properties they lack, but people adjust insufficiently from this first impression: contradicted first-example properties keep a fraction of their weight, so targets that share properties with the first-shown example — not only its final digit — are endorsed more than the examples' common profile alone implies, and the same examples in a different order are generalized differently.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/2485af553ddb9dd3/repo/_runs/cell_1/experiment3/model_loop/models/first_impression_anchoring.py`

### evidence_tuned_attention_exemplar — rank 13, 320.5 ± 34.1 nats behind the best (9.4× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -5068.9

**Hypothesis:** People do not infer a concept; they store each example shown and judge a target by its average similarity to those stored examples, where similarity falls off exponentially with the attention-weighted number of the example's properties (parity, primality, squares/cubes, powers, multiples, base-ten multiples, final digit, contained digits) the target lacks, plus their distance in magnitude. Attention to a property is learned from the examples themselves: it grows with how many of the stored examples carry that property, shrunk toward zero by a prior pseudo-count of uninformative observations, so a single example draws only weak attention to any of its features (broad, nearly distance-flat generalization), while each further example sharpens attention to the features they have in common.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/2485af553ddb9dd3/repo/_runs/cell_1/experiment3/model_loop/models/evidence_tuned_attention_exemplar.py`

## Pruned models (out of the set; narrowest margin first)

### progression_graded_magnitude_band — pruned (experiment2 end of experiment)

**Margin:** 15.6 nats behind incremental_band_narrowing; retired to keep the live set at 8 models: ELPD-LOO rank 10 of 10

**Hypothesis:** Refinement of `gap_step_progression_overlap` by one change, grafted from `graded_magnitude_band_overlap`: people do not treat "at least as large as the smallest example" and "no larger than the largest example" as all-or-none tests, but give a target magnitude credit that falls off smoothly (a bell-shaped gradient of inferred width) with its distance below the smallest or above the largest example. Everything else is unchanged — the salience-weighted share of the examples' shared profile, including the "start at the smallest and add the gap" progression concept, multiples of the seed, final-digit overshadowing and the last example's lingering digit — so the change predicts broader, distance-graded generalization from a single example (near targets endorsed more than far ones), which the incumbent misses.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/2485af553ddb9dd3/repo/_runs/cell_1/experiment2/model_loop/models/pruned/progression_graded_magnitude_band.py`

### coincidence_tilted_overlap — pruned (experiment2 end of experiment)

**Margin:** 15.6 nats behind incremental_band_narrowing; retired to keep the live set at 8 models: ELPD-LOO rank 9 of 10

**Hypothesis:** People weigh each pattern the examples share by how much of a coincidence that sharing would be. A pattern that covers few numbers (powers of 3, multiples of 11, squares, a narrow magnitude bound) carries more weight than a broad one (even, contains a 4), and this tilt toward the rarer patterns grows with each example beyond the first that confirms them. A single example therefore carries no coincidence at all. Otherwise "yes" follows the share of that weighted profile the target has, as in the current best account. So the more examples share a narrow pattern, the more firmly people reject targets that lack it, even when those targets keep the broad shared properties.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/2485af553ddb9dd3/repo/_runs/cell_1/experiment2/model_loop/models/pruned/coincidence_tilted_overlap.py`

### specificity_weighted_overlap — pruned (experiment2 end of experiment)

**Margin:** 17.3 nats behind incremental_band_narrowing (2.1× dse)

**Hypothesis:** People weigh each property the examples share the way a normative Bayesian learner would, by the size principle — a property that picks out few numbers (powers of 2, cubes, multiples of 11) is a more suspicious coincidence and so counts for more than a broad one (odd, contains a 3) — but they apply it only partially, discounting the narrowness of a property by an inferred power below the normative one (a conservative mis-weighting of specificity). "Yes" then follows the specificity-weighted share of the examples' shared property profile the target has (with the current best account's progression, seed-multiple, graded magnitude, overshadowing and lingering-digit properties), so a lone example with narrow properties (64) is generalized more narrowly than a nondescript one (37), making breadth vary from seed to seed.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/2485af553ddb9dd3/repo/_runs/cell_1/experiment2/model_loop/models/pruned/specificity_weighted_overlap.py`

### graded_band_first_digit_primacy — pruned (experiment2 end of experiment)

**Margin:** 19.1 nats behind incremental_band_narrowing (2.7× dse)

**Hypothesis:** Refinement of `graded_magnitude_band_overlap` by one grafted component, taken from `first_last_digit_anchoring`: the final digit of the FIRST example shown also lingers as a candidate pattern ("ends in 3"), with its own inferred salience, because the first example anchors people's initial guess before the later examples arrive. Everything else is unchanged: "yes" follows the salience-weighted share of the examples' shared profile the target has, with magnitude credit that falls off smoothly outside the examples' range, multiples of the seed, a visibly shared final digit overshadowing other properties, and the last example's lingering digit. So on multi-example sets without a shared final digit, targets sharing the first-shown example's final digit are endorsed more than targets sharing no example's final digit, while single-example sets still generalize broadly by magnitude.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/2485af553ddb9dd3/repo/_runs/cell_1/experiment2/model_loop/models/pruned/graded_band_first_digit_primacy.py`

### parity_split_one_sided_bound_overlap — pruned (experiment1 end of experiment)

**Margin:** 19.6 nats behind seed_multiple_salience_overlap (2.5× dse)

**Hypothesis:** Refinement of `parity_direction_salience_overlap`: people note every salient mathematical property all the examples share (each property family with its own salience, "all odd" and "all even" with separate saliences, multiples of 5/10 their own family, no size principle) and say "yes" in proportion to the salience-weighted share of that profile the target also has. The one change, grafted from `one_sided_magnitude_bound_overlap`, replaces the two-sided "within the examples' span" property with two one-sided magnitude bounds, "at least as large as the smallest example" and "no larger than the largest example". Each bound has its own salience, so going past the examples upward can cost a target a different amount than going below them, on top of the directional parity asymmetry.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/2485af553ddb9dd3/repo/_runs/cell_1/experiment1/model_loop/models/pruned/parity_split_one_sided_bound_overlap.py`

### coincidence_gated_pattern_belief — pruned (experiment2 end of experiment)

**Margin:** 20.8 nats behind incremental_band_narrowing (2.1× dse)

**Hypothesis:** People first ask whether the examples look patterned at all, or could be numbers the program produced at random. They decide this by coincidence detection: the more striking the agreement among the examples (several examples all falling in a small category such as "powers of 2" or "ends in 8"), the more sure they are that the program follows a rule. A single example, or several that share only broad properties, gives no such evidence, so people suspect the program produces almost any number and lean toward "yes" for every target. When they do believe in a rule, they judge the target as the current best account does, by the salience-weighted share of the examples' shared property profile it has.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/2485af553ddb9dd3/repo/_runs/cell_1/experiment2/model_loop/models/pruned/coincidence_gated_pattern_belief.py`

### one_sided_magnitude_bound_overlap — pruned (experiment1 end of experiment)

**Margin:** 21.6 nats behind seed_multiple_salience_overlap (2.9× dse)

**Hypothesis:** Refinement of `decimal_multiple_salience_overlap`: people still note every salient property all the examples share (each property family with its own salience, base-ten multiples their own family, no size principle) and say "yes" in proportion to the salience-weighted share of that profile the target also has. The one change is to the magnitude concepts people consider: instead of one two-sided "within the examples' span" property, they entertain two separate one-sided bounds — "at least as large as the smallest example" and "no larger than the largest example" — each with its own salience, so if the open-ended "this big or bigger" bound is the more salient, targets above all the examples are endorsed more readily than equally distant targets below them.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/2485af553ddb9dd3/repo/_runs/cell_1/experiment1/model_loop/models/pruned/one_sided_magnitude_bound_overlap.py`

### single_example_magnitude_neighborhood — pruned (experiment2 end of experiment)

**Margin:** 22.9 nats behind incremental_band_narrowing (2.4× dse)

**Hypothesis:** When the program has shown only a single example, that number's magnitude itself becomes one of the candidate properties people consider ("numbers about this size"), and a target earns credit for it that falls off smoothly with its distance from the example, in both directions. Otherwise "yes" follows the salience-weighted share of the examples' shared property profile the target has, as in the current best model (arithmetic-progression concept, seed multiples, final-digit overshadowing, the last example's lingering digit). The disagreement with the best model lives in one-example sets: targets near the lone example, above or below it and even when they break its parity or digit properties, are endorsed more, and far targets above it are no longer favored just for passing the "at least as large" bound.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/2485af553ddb9dd3/repo/_runs/cell_1/experiment2/model_loop/models/pruned/single_example_magnitude_neighborhood.py`

### progression_single_example_leniency — pruned (experiment2 end of experiment)

**Margin:** 23.0 nats behind incremental_band_narrowing (2.4× dse)

**Hypothesis:** Refinement of `gap_step_progression_overlap` by one change, grafted from `single_example_leniency_overlap`: when only a single example is shown, people know they cannot yet tell which of its properties the program cares about, so they adopt a more lenient criterion and lean toward "yes" for every target by an inferred amount, beyond what the overlap with that example's property profile implies. Everything else is the incumbent's judgment unchanged (salience-weighted share of the examples' shared profile, including the arithmetic-progression concept for evenly spaced examples, multiples of the seed, final-digit overshadowing and the last example's lingering digit), so non-example targets of one-example sets are endorsed more than after two or more examples, addressing the critique that the incumbent under-generalizes from a single example.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/2485af553ddb9dd3/repo/_runs/cell_1/experiment2/model_loop/models/pruned/progression_single_example_leniency.py`

### progression_band_single_leniency — pruned (experiment2 end of experiment)

**Margin:** 23.0 nats behind incremental_band_narrowing (2.5× dse)

**Hypothesis:** Refinement of the incumbent `progression_graded_magnitude_band` by one grafted component, taken from `progression_single_example_leniency`: when only a single example is shown, people know they cannot yet tell which of its properties the program cares about, so they adopt a more lenient criterion and lean toward "yes" for every target by an inferred amount, beyond what the overlap with that example's profile implies. Everything else is the incumbent unchanged (salience-weighted share of the examples' shared profile, with magnitude credit falling off smoothly outside the examples' range, the "start at the smallest and add the gap" progression concept, seed multiples, final-digit overshadowing and the last example's lingering digit), so the graded band shapes which targets a lone example generalizes to while the leniency raises how often people say yes to them at all — non-example targets of one-example sets are endorsed more than the band alone predicts, with no change for sets of two to four examples.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/2485af553ddb9dd3/repo/_runs/cell_1/experiment2/model_loop/models/pruned/progression_band_single_leniency.py`

### surface_digit_search_termination — pruned (experiment1 end of experiment)

**Margin:** 24.4 nats behind seed_multiple_salience_overlap (3.1× dse)

**Hypothesis:** People search the examples for shared properties in order of how visible they are, beginning with the written surface of the numbers, and when that first glance turns up a shared final digit they largely stop searching: the deeper arithmetic properties the examples also happen to share (primes, squares/cubes, powers, multiples) are only partly noticed. They then say "yes" in proportion to the salience-weighted share of the noticed shared profile the target also has (with one-sided magnitude bounds and no size principle), so when a common last digit is present, a target that keeps that digit is endorsed strongly even if it breaks an incidental arithmetic pattern such as "multiple of 7".

**Source:** `/scr/wurgaft/auto-psych/agent_trees/2485af553ddb9dd3/repo/_runs/cell_1/experiment1/model_loop/models/pruned/surface_digit_search_termination.py`

### salience_scaled_decisiveness_overlap — pruned (experiment1 end of experiment)

**Margin:** 25.0 nats behind seed_multiple_salience_overlap (2.7× dse)

**Hypothesis:** Refinement of `one_sided_magnitude_bound_overlap`: people still note every salient property all the examples share (each property family with its own salience, base-ten multiples their own family, two one-sided magnitude bounds, no size principle) and judge a target by the salience-weighted share of that profile it also has. The one change is that how decisively they act on that share depends on how salient the examples' common pattern is: when the examples share a highly noticeable property (such as a common final digit or being powers of the same number), a target that fits the profile is endorsed much more confidently than one that fully fits a profile made only of weak, unremarkable properties (such as merely being even), so the pull toward "yes" grows with the total salience of the shared profile rather than with the matched share alone.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/2485af553ddb9dd3/repo/_runs/cell_1/experiment1/model_loop/models/pruned/salience_scaled_decisiveness_overlap.py`

### gap_step_progression_overlap — pruned (experiment2 end of experiment)

**Margin:** 25.4 nats behind incremental_band_narrowing (3.2× dse)

**Hypothesis:** People's space of candidate concepts includes an arithmetic-progression concept that the current models lack. When the examples are evenly spaced (e.g. 8 and 16, or 2 and 8), people entertain "start at the smallest example and keep adding the gap between the examples" (8, 16, 24, 32, ...; 2, 8, 14, 20, ...) as one more property of the set, with its own inferred prior salience. Otherwise they judge the target as before, by the salience-weighted share of the examples' shared property profile it has, so targets on the upward continuation of the progression are endorsed more even when they break other shared properties such as being a power of 2.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/2485af553ddb9dd3/repo/_runs/cell_1/experiment2/model_loop/models/pruned/gap_step_progression_overlap.py`

### shared_final_digit_rule_overlap — pruned (experiment1 end of experiment)

**Margin:** 26.0 nats behind seed_multiple_salience_overlap (2.8× dse)

**Hypothesis:** Refinement of `one_sided_magnitude_bound_overlap`: people still note every salient property all the examples share (each family with its own salience, base-ten multiples their own family, two one-sided magnitude bounds, no size principle) and say "yes" in proportion to the salience-weighted share of that profile the target also has. The one change is that a final digit shared by two or more examples is not just one diluted property in that profile. People read it as a self-evident rule ("they all end in 8"), so a target that ends in that digit gets an extra, independent boost toward "yes". That boost does not shrink when the examples also happen to share other, coincidental properties (e.g. 28 and 98 both being multiples of 7).

**Source:** `/scr/wurgaft/auto-psych/agent_trees/2485af553ddb9dd3/repo/_runs/cell_1/experiment1/model_loop/models/pruned/shared_final_digit_rule_overlap.py`

### parity_direction_salience_overlap — pruned (experiment1 end of experiment)

**Margin:** 27.2 nats behind seed_multiple_salience_overlap (2.6× dse)

**Hypothesis:** Refinement of `family_salience_feature_overlap` (carried over in the form its successor `decimal_multiple_salience_overlap` established, with multiples of 5 and 10 as their own salience family): people note every salient mathematical property the examples all share and say "yes" in proportion to the salience-weighted share of that profile the target also has, with no size principle. The one new change, grafted from the directional parity contrast of `asymmetric_decimal_contrast_similarity`, is that "all odd" and "all even" are not one parity family with a single salience: each has its own inferred salience, so a shared odd (or even) pattern can weigh far more in the judgment than the opposite parity pattern does, and breaking it costs a target correspondingly more.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/2485af553ddb9dd3/repo/_runs/cell_1/experiment1/model_loop/models/pruned/parity_direction_salience_overlap.py`

### graded_magnitude_band_overlap — pruned (experiment2 end of experiment)

**Margin:** 27.7 nats behind incremental_band_narrowing (2.9× dse)

**Hypothesis:** Refinement of `seed_overshadowing_last_digit_recency` by one change to its magnitude property. People do not treat "at least as large as the smallest example" and "no larger than the largest example" as all-or-none tests. They judge how close in magnitude the target is to the range the examples span, and that magnitude similarity falls off smoothly (a broad, bell-shaped gradient) with the target's distance below the smallest or above the largest example. So a target just outside the examples keeps nearly full magnitude credit, and only far-away targets lose it. Everything else is unchanged: "yes" follows the salience-weighted share of the examples' shared profile the target has, including the seed-multiple property, final-digit overshadowing and the lingering last-example digit. The main consequence is broader generalization from a single example, where every non-example target lies outside the examples' "range".

**Source:** `/scr/wurgaft/auto-psych/agent_trees/2485af553ddb9dd3/repo/_runs/cell_1/experiment2/model_loop/models/pruned/graded_magnitude_band_overlap.py`

### directional_span_salience_overlap — pruned (experiment1 end of experiment)

**Margin:** 31.6 nats behind seed_multiple_salience_overlap (3.3× dse)

**Hypothesis:** Refinement of `decimal_multiple_salience_overlap`: people still note every salient mathematical property the examples all share (with multiples of 5/10 as their own salient family) and say "yes" in proportion to the salience-weighted share of that profile the target also has, with no size principle. The one change is the "lies within the examples' range" property: instead of an all-or-none, symmetric in-span check, a target outside the span keeps partial credit that fades with its distance from the span, and it fades at a different, inferred rate above the largest example than below the smallest — people readily extend a pattern upward to larger numbers (as if the program counts up) but much less downward.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/2485af553ddb9dd3/repo/_runs/cell_1/experiment1/model_loop/models/pruned/directional_span_salience_overlap.py`

### directional_span_salience_overlap_2 — pruned (experiment1 end of experiment)

**Margin:** 31.6 nats behind seed_multiple_salience_overlap (3.3× dse)

**Hypothesis:** Refinement of `decimal_multiple_salience_overlap`: people still note every salient mathematical property the examples all share (with multiples of 5/10 as their own salience family) and say "yes" in proportion to the salience-weighted share of that profile the target also has, with no size principle. The one change is to the "lies within the examples' span" property: instead of an all-or-none, symmetric in-span test, the span match fades gradually with how far the target lies beyond the examples, at separately inferred rates above the largest example and below the smallest one — people extend a set upward toward larger numbers more readily than downward toward smaller ones.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/2485af553ddb9dd3/repo/_runs/cell_1/experiment1/model_loop/models/pruned/directional_span_salience_overlap_2.py`

### single_example_leniency_overlap — pruned (experiment2 end of experiment)

**Margin:** 31.6 nats behind incremental_band_narrowing (3.1× dse)

**Hypothesis:** Refinement of `seed_overshadowing_last_digit_recency` by one change: when only a single example is shown, people know they cannot yet tell which of its properties the program cares about, so they adopt a more lenient criterion and lean toward "yes" for any target, beyond what the overlap with that example's property profile alone implies. Everything else is unchanged (salience-weighted share of the examples' shared profile, multiples of the seed, a visibly shared final digit overshadowing other properties, and the last example's final digit lingering), so the change predicts higher endorsement of non-example targets for one-example sets than for sets of two to four examples, and no difference among the multi-example set sizes.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/2485af553ddb9dd3/repo/_runs/cell_1/experiment2/model_loop/models/pruned/single_example_leniency_overlap.py`

### decimal_multiple_salience_overlap — pruned (experiment1 end of experiment)

**Margin:** 32.7 nats behind seed_multiple_salience_overlap (3.3× dse)

**Hypothesis:** Refinement of `family_salience_feature_overlap`: people still note every salient mathematical property the examples all share and say "yes" in proportion to the salience-weighted share of that profile the target also has, with no size principle. The one change is that multiples of 5 and of 10 — the round, base-ten multiples — form their own salience family, separate from other multiples (of 3, 4, 6, 7, 8, 9, 11, 12), because decimal notation makes them far more noticeable; so when examples like 90 and 45 share both "multiple of 5" and "multiple of 9", people generalize along the multiple-of-5 property much more than along the other one.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/2485af553ddb9dd3/repo/_runs/cell_1/experiment1/model_loop/models/pruned/decimal_multiple_salience_overlap.py`

### first_last_digit_anchoring — pruned (experiment2 end of experiment)

**Margin:** 35.3 nats behind incremental_band_narrowing (4.6× dse)

**Hypothesis:** Refinement of `seed_overshadowing_last_digit_recency` by one change: the final digit of the FIRST example shown also lingers as a candidate pattern ("ends in 3"), not only that of the last, because the first example anchors people's initial guess before the later ones arrive (primacy alongside recency), with its own inferred salience. Otherwise the judgment is unchanged: "yes" follows the salience-weighted share of the examples' shared profile the target has, with multiples of the seed, a visibly shared final digit overshadowing other properties, and the last example's lingering digit. So on multi-example sets without a shared final digit, targets sharing the first-shown example's final digit are endorsed more than targets sharing no example's final digit, which the incumbent does not predict.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/2485af553ddb9dd3/repo/_runs/cell_1/experiment2/model_loop/models/pruned/first_last_digit_anchoring.py`

### single_example_open_generalization — pruned (experiment2 end of experiment)

**Margin:** 36.6 nats behind incremental_band_narrowing (3.2× dse)

**Hypothesis:** Refinement of `seed_multiple_digit_overshadowing` by one change: when the program has shown only a single example, people see no evidence that it is restricted to any particular pattern, so they lean toward "yes" for every target, by an inferred amount, on top of the usual judgment. Otherwise the judgment is unchanged: "yes" follows the salience-weighted share of the examples' shared property profile that the target has, a final digit visibly shared by several examples overshadows the other properties, and "multiples of the smallest example" joins the profile when it divides all the others. So non-example targets for one-example sets (e.g. 48 -> 18, 60, 72) are endorsed more than the same degree of profile match would earn after two or more examples. This addresses the critique that the best model under-generalizes from single examples.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/2485af553ddb9dd3/repo/_runs/cell_1/experiment2/model_loop/models/pruned/single_example_open_generalization.py`

### seed_overshadowing_last_digit_recency — pruned (experiment2 end of experiment)

**Margin:** 42.3 nats behind incremental_band_narrowing (4.4× dse)

**Hypothesis:** Refinement of `seed_multiple_digit_overshadowing_overlap` by one grafted component, taken from `last_example_digit_recency_overlap`: people read the examples in order, and the final digit of the last example shown lingers in their guess as a candidate pattern ("ends in 1") even when the earlier examples do not share it. Otherwise the judgment is unchanged: "yes" follows the salience-weighted share of the examples' shared profile the target has, with "multiples of the smallest example" in that profile when it divides all the others, and a final digit visibly shared by all examples overshadowing every other property. Compared with the seed-overshadowing model, targets that share the last-shown example's final digit are endorsed more than targets that share the first-shown example's, so the same examples in a different order are generalized differently.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/2485af553ddb9dd3/repo/_runs/cell_1/experiment2/model_loop/models/pruned/seed_overshadowing_last_digit_recency.py`

### seed_multiple_digit_overshadowing — pruned (experiment2 end of experiment)

**Margin:** 46.7 nats behind incremental_band_narrowing (4.2× dse)

**Hypothesis:** Refinement of `final_digit_overshadowing_overlap` by one grafted component, taken from `seed_multiple_salience_overlap`. People still say "yes" in proportion to the salience-weighted share of the examples' shared property profile that the target also has, and a final digit visibly shared by several examples still overshadows every other shared property. The one change is that people also notice when the smallest example divides every other example (7 and 63, 8 and 64). They read that number as the program's seed and add "multiples of the seed" to the shared profile with its own salience, so multiples of the seed that break other shared properties such as parity (14 or 28 for 7, 63) are endorsed far more than the incumbent predicts.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/2485af553ddb9dd3/repo/_runs/cell_1/experiment2/model_loop/models/pruned/seed_multiple_digit_overshadowing.py`

### seed_multiple_digit_overshadowing_overlap — pruned (experiment2 end of experiment)

**Margin:** 56.0 nats behind incremental_band_narrowing (4.4× dse)

**Hypothesis:** Refinement of `final_digit_overshadowing_overlap` by one change, a component grafted from `seed_multiple_salience_overlap`: people also notice a relation between the examples themselves, namely whether the smallest example divides every other example (7, 63; 9, 81; 8, 64). When it does, "multiples of that seed number" joins the examples' shared profile as a property with its own salience. Otherwise the judgment is unchanged: people say "yes" in proportion to the salience-weighted share of the shared profile the target has, and a final digit visibly shared by several examples still overshadows every other shared property, the seed property included. Compared with the incumbent, multiples of the seed that break other shared properties, such as 14 and 28 for the set 7, 63, are endorsed more.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/2485af553ddb9dd3/repo/_runs/cell_1/experiment2/model_loop/models/pruned/seed_multiple_digit_overshadowing_overlap.py`

### family_salience_feature_overlap — pruned (experiment1 end of experiment)

**Margin:** 56.1 nats behind seed_multiple_salience_overlap (4.2× dse)

**Hypothesis:** Refinement of `shared_feature_overlap`: people still note every salient mathematical property the examples all share and say "yes" in proportion to how much of that shared profile the target also has, with no size principle. The one change is that the properties are not equally salient: each family of property (parity, primes, squares/cubes, powers, multiples, final digit, contained digit, and lying within the examples' span) gets its own inferred salience weight, so a shared feature of a highly salient kind (e.g. a common final digit) dominates the judgment while shared features of less noticed kinds barely count.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/2485af553ddb9dd3/repo/_runs/cell_1/experiment1/model_loop/models/pruned/family_salience_feature_overlap.py`

### position_weighted_feature_tally — pruned (experiment1 end of experiment)

**Margin:** 60.9 nats behind seed_multiple_salience_overlap (4.4× dse)

**Hypothesis:** People build up their sense of which mathematical properties matter by reading the examples in order and keeping a running tally of each property (parity, primes, squares/cubes, powers, multiples, final digit, contained digit), in which examples at certain positions count more than others — the most recent ones if memory fades, the first ones if the first example anchors the guess. They then say "yes" in proportion to how much of that order-weighted, salience-weighted property profile (plus lying within the examples' span) the target also has, so a property carried by the favoured examples but not by the rest still drives generalization, and the same examples in a different order would be generalized differently.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/2485af553ddb9dd3/repo/_runs/cell_1/experiment1/model_loop/models/pruned/position_weighted_feature_tally.py`

### last_example_digit_recency_overlap — pruned (experiment2 end of experiment)

**Margin:** 64.8 nats behind incremental_band_narrowing (5.3× dse)

**Hypothesis:** Refinement of `final_digit_overshadowing_overlap` by one order-dependent change: people revise a running guess as they read the examples, and the final digit of the last example shown lingers in that guess as a candidate pattern ("ends in 2") even when the earlier examples do not all share it. They still say "yes" in proportion to the salience-weighted share of the examples' shared profile that the target has, but that profile also contains "ends in the same digit as the last example", with a fraction of the final-digit salience. So the same examples in a different order are generalized differently: targets sharing the last-shown example's final digit are endorsed more than targets sharing the first-shown example's.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/2485af553ddb9dd3/repo/_runs/cell_1/experiment2/model_loop/models/pruned/last_example_digit_recency_overlap.py`

### undiluted_magnitude_range_dimension — pruned (experiment2 end of experiment)

**Margin:** 69.7 nats behind incremental_band_narrowing (5.7× dse)

**Hypothesis:** People judge magnitude separately from the examples' mathematical properties: whether the target falls within the range the examples span (with credit fading smoothly with its distance below the smallest or above the largest example) is a dimension of its own that pushes the decision directly, not one property diluted among however many properties the examples happen to share. So a target inside the examples' range that shares none of their parity, digit or multiple properties is still endorsed clearly more than an equally unrelated target outside the range, even for sets with rich shared profiles, while the rest of the judgment (the salience-weighted share of the shared profile, with progression, seed multiples, final-digit overshadowing and the first and last examples' lingering digits) is that of the best model.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/2485af553ddb9dd3/repo/_runs/cell_1/experiment2/model_loop/models/pruned/undiluted_magnitude_range_dimension.py`

### violation_penalty_feature_test — pruned (experiment1 end of experiment)

**Margin:** 77.0 nats behind seed_multiple_salience_overlap (3.9× dse)

**Hypothesis:** Refinement of `family_salience_feature_overlap`: people still note the salient mathematical properties all the examples share (parity, primes, squares/cubes, powers, multiples, final digit, contained digit, lying within the examples' span), each family with its own salience, and use no size principle. The one change is the combination rule: instead of a compensatory share of the profile the target matches (where matching parity or span can make up for breaking the shared final digit), each shared property the target violates independently cuts the chance of a "yes" by an amount set by its family's salience — a non-compensatory, noisy-AND test in which violating a highly salient shared property (e.g. a common final digit) is close to disqualifying no matter what else the target shares.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/2485af553ddb9dd3/repo/_runs/cell_1/experiment1/model_loop/models/pruned/violation_penalty_feature_test.py`

### inattentive_guessing_lapse — pruned (experiment2 end of experiment)

**Margin:** 88.8 nats behind incremental_band_narrowing (6.3× dse)

**Hypothesis:** People judge each target the way the current best account says: by the salience-weighted share of the examples' shared property profile the target has. That profile includes multiples of the smallest example when it divides the others, a shared final digit that overshadows other properties, and the last example's lingering final digit. The new claim is about the decision rule. On some fraction of trials, which the model infers, people do not consult that judgment at all and answer by guessing, saying "yes" at their own inferred guessing rate. This puts a floor under endorsement of targets that clearly do not fit and a ceiling under targets that clearly fit, and the same floor and ceiling apply to every set.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/2485af553ddb9dd3/repo/_runs/cell_1/experiment2/model_loop/models/pruned/inattentive_guessing_lapse.py`

### evidence_weighted_concept_learning — pruned (experiment1 end of experiment)

**Margin:** 91.8 nats behind seed_multiple_salience_overlap (4.8× dse)

**Hypothesis:** People do Bayesian concept learning over mathematical rules and numeric intervals with the size principle, as in the Independent model of Bigelow & Piantadosi, but they systematically mis-weight the evidence of the examples relative to their prior (Edwards-style conservatism or over-confidence): the examples' likelihood enters their belief update raised to a single inferred power. When that power is below one, people update too little from the examples, so tight rules and narrow ranges dominate less and generalization stays broader than the normative learner's; above one, they over-commit to the tightest fitting concept.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/2485af553ddb9dd3/repo/_runs/cell_1/experiment1/model_loop/models/pruned/evidence_weighted_concept_learning.py`

### evidence_weighted_concepts_with_lapse — pruned (experiment1 end of experiment)

**Margin:** 93.5 nats behind seed_multiple_salience_overlap (5.1× dse)

**Hypothesis:** Refinement of `evidence_weighted_concept_learning`: people still do Bayesian concept learning over the same 45 mathematical rules and all intervals of 1-100 with the size principle, mis-weighting the examples' evidence by one inferred power, and usually answer "yes" with the posterior probability that the concept contains the target. The one change is that on an inferred fraction of trials people do not consult the inferred concept at all (inattention) and instead answer from a fixed, inferred bias toward "yes", so targets that clearly violate the concept still get a floor of endorsement and targets that clearly fit it are sometimes rejected, rather than near-certain all-or-none answers.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/2485af553ddb9dd3/repo/_runs/cell_1/experiment1/model_loop/models/pruned/evidence_weighted_concepts_with_lapse.py`

### graded_consensus_feature_overlap — pruned (experiment2 end of experiment)

**Margin:** 94.6 nats behind incremental_band_narrowing (6.1× dse)

**Hypothesis:** Refinement of `final_digit_overshadowing_overlap` by one change: people judge a target by how much of the examples' property profile it shares, but that profile is not limited to properties that *every* example has. Each property held by some of the examples enters with a weight that grows with the fraction of examples holding it, and how steeply that weight falls off is inferred. A property held by most examples (three of 69, 9, 39, 21 end in 9) or by half of them (13, 91 end in 1 and 3) still pulls generalization toward targets that have it, rather than being ignored as it is when only unanimous properties count.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/2485af553ddb9dd3/repo/_runs/cell_1/experiment2/model_loop/models/pruned/graded_consensus_feature_overlap.py`

### conservative_size_principle_conjunction — pruned (experiment1 end of experiment)

**Margin:** 97.9 nats behind seed_multiple_salience_overlap (5.1× dse)

**Hypothesis:** Refinement of rule_within_range_conjunction: people still infer a conjunctive concept — a mathematical rule (or "any number") restricted to a range of magnitudes — choosing rule and range together by Bayesian size-principle reasoning, and say "yes" with the posterior probability that the concept contains the target. The one change is that people are conservative about the evidence: they weigh how well each candidate concept explains the examples with an inferred strength (the likelihood raised to a power below one if they are conservative), so the size principle pulls them less sharply toward the tightest rule and range, and they keep partial belief in broader concepts, such as any number within the examples' span, even when a narrow rule fits better.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/2485af553ddb9dd3/repo/_runs/cell_1/experiment1/model_loop/models/pruned/conservative_size_principle_conjunction.py`

### final_digit_overshadowing_overlap — pruned (experiment2 end of experiment)

**Margin:** 138.0 nats behind incremental_band_narrowing (7.3× dse)

**Hypothesis:** Refinement of `one_sided_magnitude_bound_overlap` by one change: a final digit that is visibly shared by all of several examples (e.g. 28, 98) overshadows every other shared property. People still say "yes" in proportion to the salience-weighted share of the examples' shared profile that the target has, but when the examples visibly share a final digit, all the other shared properties (parity, multiples, contained digits, magnitude bounds, and so on) lose an inferred fraction of their salience. This disagrees most sharply with the incumbent on shared-final-digit sets: targets that keep the digit but break the other shared properties (28, 98 -> 18, 38) are endorsed far more, and targets that fit the other properties but not the digit (28, 98 -> 14, 56) far less.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/2485af553ddb9dd3/repo/_runs/cell_1/experiment2/model_loop/models/pruned/final_digit_overshadowing_overlap.py`

### log_scale_feature_prototype — pruned (experiment1 end of experiment)

**Margin:** 143.3 nats behind seed_multiple_salience_overlap (5.8× dse)

**Hypothesis:** People do not infer a rule; they abstract a prototype from the examples — the average profile of mathematical properties the examples have (how many are even, odd, prime, squares, powers, multiples of each number, multiples of 5/10, end in each digit, contain each digit) together with a central magnitude on a compressed, logarithmic number line — and say "yes" in proportion to how similar the target is to that prototype, with similarity falling off with the attention-weighted mismatch in properties and the log-scale distance from the prototype's centre. Because the number line is logarithmic, a target the same numerical distance above the examples is closer to the prototype than one below them, so people extend upward more readily than downward.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/2485af553ddb9dd3/repo/_runs/cell_1/experiment1/model_loop/models/pruned/log_scale_feature_prototype.py`

### seed_multiple_salience_overlap — pruned (experiment2 end of experiment)

**Margin:** 146.4 nats behind incremental_band_narrowing (6.6× dse)

**Hypothesis:** People notice a relation between the examples themselves that the property-overlap models ignore: whether the smallest example divides every other example (7 and 63, 9 and 81, 8 and 64). When it does, they read that smallest number as the program's seed and take "multiples of the seed" as a salient property of the set, with its own salience. As in the overlap account, they still say "yes" in proportion to the salience-weighted share of the examples' common properties that the target also has, so a multiple of the seed is endorsed even when it breaks other shared properties such as parity (e.g. 14 and 28 for 7, 63).

**Source:** `/scr/wurgaft/auto-psych/agent_trees/2485af553ddb9dd3/repo/_runs/cell_1/experiment2/model_loop/models/pruned/seed_multiple_salience_overlap.py`

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

### property_exemplar_similarity — pruned (experiment2 end of experiment)

**Margin:** 181.2 nats behind incremental_band_narrowing (7.0× dse)

**Hypothesis:** People do not infer a concept from the examples; they store each example shown as a separate exemplar, described by its mathematical properties (parity, primality, squares/cubes, powers, multiples, base-ten multiples, final digit, contained digits) and its magnitude, and judge a target by its average similarity to the stored examples. Similarity to each example falls off exponentially with the attention-weighted number of properties on which the target and that example differ plus their numerical distance, and the more similar the target is to the examples on average, the more likely a "yes". So every example counts equally whatever its position, a target sharing a property with only some examples (e.g. the first example's final digit) still gains, and a single example is generalized to numbers close in magnitude.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/2485af553ddb9dd3/repo/_runs/cell_1/experiment2/model_loop/models/pruned/property_exemplar_similarity.py`

### leaky_memory_example_elimination — pruned (experiment2 end of experiment)

**Margin:** 181.7 nats behind incremental_band_narrowing (8.8× dse)

**Hypothesis:** People generalize like a weak-sampling Bayesian learner over salient number properties (the incumbent's salience-weighted overlap, with a visible shared final digit overshadowing other properties), but with one resource limit: working memory for the examples fades, so each earlier example is only remembered — and so only rules out the properties it lacks — with a probability that falls with how many examples were shown after it. Properties of the most recent example that an earlier, partly forgotten example contradicts therefore keep some weight, so targets resembling the last-shown example (e.g. sharing its final digit) are endorsed more than targets resembling the first.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/2485af553ddb9dd3/repo/_runs/cell_1/experiment2/model_loop/models/pruned/leaky_memory_example_elimination.py`

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
