# Existing hypotheses

Each model below is ONE cognitive hypothesis, with how it stands on the current data by ELPD-LOO (best first). `elpd_diff ± dse` is a model's deficit against the best and the standard error of that difference (with the responses to each stimulus pair counted together, as they are correlated): within about 2·dse the two are statistically tied on this data; beyond it the model has lost. "PSIS-LOO unreliable" means the estimate itself is untrustworthy (too many high-Pareto-k trials), not that the model is bad. Propose a hypothesis that is genuinely different from these, or a refinement of a single one of them — never a combination of several.

## final_digit_overshadowing_overlap  — rank 0, the best model on this data, ELPD-LOO -3290.6

Refinement of `one_sided_magnitude_bound_overlap` by one change: a final digit that is visibly shared by all of several examples (e.g. 28, 98) overshadows every other shared property. People still say "yes" in proportion to the salience-weighted share of the examples' shared profile that the target has, but when the examples visibly share a final digit, all the other shared properties (parity, multiples, contained digits, magnitude bounds, and so on) lose an inferred fraction of their salience. This disagrees most sharply with the incumbent on shared-final-digit sets: targets that keep the digit but break the other shared properties (28, 98 -> 18, 38) are endorsed far more, and targets that fit the other properties but not the digit (28, 98 -> 14, 56) far less.

## seed_multiple_salience_overlap  — rank 1, 8.4 ± 21.9 nats behind the best (0.4× dse: statistically tied with the best), ELPD-LOO -3299.0

People notice a relation between the examples themselves that the property-overlap models ignore: whether the smallest example divides every other example (7 and 63, 9 and 81, 8 and 64). When it does, they read that smallest number as the program's seed and take "multiples of the seed" as a salient property of the set, with its own salience. As in the overlap account, they still say "yes" in proportion to the salience-weighted share of the examples' common properties that the target also has, so a multiple of the seed is endorsed even when it breaks other shared properties such as parity (e.g. 14 and 28 for 7, 63).

## leaky_memory_example_elimination  — no comparison row

People generalize like a weak-sampling Bayesian learner over salient number properties (the incumbent's salience-weighted overlap, with a visible shared final digit overshadowing other properties), but with one resource limit: working memory for the examples fades, so each earlier example is only remembered — and so only rules out the properties it lacks — with a probability that falls with how many examples were shown after it. Properties of the most recent example that an earlier, partly forgotten example contradicts therefore keep some weight, so targets resembling the last-shown example (e.g. sharing its final digit) are endorsed more than targets resembling the first.

## graded_consensus_feature_overlap  — no comparison row

Refinement of `final_digit_overshadowing_overlap` by one change: people judge a target by how much of the examples' property profile it shares, but that profile is not limited to properties that *every* example has. Each property held by some of the examples enters with a weight that grows with the fraction of examples holding it, and how steeply that weight falls off is inferred. A property held by most examples (three of 69, 9, 39, 21 end in 9) or by half of them (13, 91 end in 1 and 3) still pulls generalization toward targets that have it, rather than being ignored as it is when only unanimous properties count.

## last_example_digit_recency_overlap  — no comparison row

Refinement of `final_digit_overshadowing_overlap` by one order-dependent change: people revise a running guess as they read the examples, and the final digit of the last example shown lingers in that guess as a candidate pattern ("ends in 2") even when the earlier examples do not all share it. They still say "yes" in proportion to the salience-weighted share of the examples' shared profile that the target has, but that profile also contains "ends in the same digit as the last example", with a fraction of the final-digit salience. So the same examples in a different order are generalized differently: targets sharing the last-shown example's final digit are endorsed more than targets sharing the first-shown example's.

## seed_multiple_digit_overshadowing  — no comparison row

Refinement of `final_digit_overshadowing_overlap` by one grafted component, taken from `seed_multiple_salience_overlap`. People still say "yes" in proportion to the salience-weighted share of the examples' shared property profile that the target also has, and a final digit visibly shared by several examples still overshadows every other shared property. The one change is that people also notice when the smallest example divides every other example (7 and 63, 8 and 64). They read that number as the program's seed and add "multiples of the seed" to the shared profile with its own salience, so multiples of the seed that break other shared properties such as parity (14 or 28 for 7, 63) are endorsed far more than the incumbent predicts.

## seed_multiple_digit_overshadowing_overlap  — no comparison row

Refinement of `final_digit_overshadowing_overlap` by one change, a component grafted from `seed_multiple_salience_overlap`: people also notice a relation between the examples themselves, namely whether the smallest example divides every other example (7, 63; 9, 81; 8, 64). When it does, "multiples of that seed number" joins the examples' shared profile as a property with its own salience. Otherwise the judgment is unchanged: people say "yes" in proportion to the salience-weighted share of the shared profile the target has, and a final digit visibly shared by several examples still overshadows every other shared property, the seed property included. Compared with the incumbent, multiples of the seed that break other shared properties, such as 14 and 28 for the set 7, 63, are endorsed more.
