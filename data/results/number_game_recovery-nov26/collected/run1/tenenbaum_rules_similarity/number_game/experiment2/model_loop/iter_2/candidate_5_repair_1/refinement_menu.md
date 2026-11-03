# Refinement menu

The models you may refine, other than the incumbent `threshold_concepts_erlang_interval`. Each entry is one mechanism: its hypothesis in full as its author stated it, how it stands, and where its source is. This is a menu, not a list of what is ruled out: a pruned model lost to the best model on the data it was scored on, by the stated margin, but its mechanism may be partly right — and a model that lost narrowly is the most promising target here. Read the source of the model you pick before you write anything.

## Live models other than the incumbent (in the set; best first)

### centre_weighted_range_sampling — rank 1, 0.3 ± 1.4 nats behind the best (0.2× dse: statistically tied with the best), ELPD-LOO -2259.2

**Hypothesis:** Refinement of `erlang_interval_size_family_prior`: people do the same Bayesian concept learning over number rules and numeric ranges, with the size principle, a prior mass per concept family (exception rules in their own family) and an Erlang prior over range width, but they do not expect a range-based program to produce every number in its range equally often — they expect its outputs to cluster toward the middle of the range, with how strongly they do so inferred from the data (none recovers the parent model). The one change is this centre-weighted sampling from ranges: examples sitting at a range's edge are a "suspicious coincidence" for it, so ranges that are centred on the examples and extend beyond the lowest and highest example gain support, and people say "yes" to targets somewhat past the examples' extremes more often than a uniform-sampling learner predicts.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/043c9ecd8d452a7b/repo/_runs/cell_1/experiment2/model_loop/models/centre_weighted_range_sampling.py`

### set_size_amplified_size_principle — rank 2, 0.9 ± 1.4 nats behind the best (0.7× dse: statistically tied with the best), ELPD-LOO -2259.8

**Hypothesis:** Refinement of `threshold_concepts_erlang_interval`: people do the same Bayesian concept learning over rules, bounded numeric ranges (with an Erlang prior on width) and one-sided thresholds, with the size principle and a prior mass per concept family, but the "suspicious coincidence" does not accumulate additively over examples. The one change is that each example's evidential weight scales with a learned power of how many examples were shown, so the more examples a set has, the more decisively each one favours tight concepts over broad ones (zero recovers the incumbent): generalisation narrows faster from one to four examples than an ideal learner predicts.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/043c9ecd8d452a7b/repo/_runs/cell_1/experiment2/model_loop/models/set_size_amplified_size_principle.py`

### threshold_concepts_primes_family — rank 3, 1.1 ± 1.6 nats behind the best (0.7× dse: statistically tied with the best), ELPD-LOO -2259.9

**Hypothesis:** Refinement of `threshold_concepts_erlang_interval`: people do the same Bayesian concept learning over number rules, bounded numeric ranges (with an Erlang prior on width) and one-sided magnitude thresholds (their own family), with the size principle and a prior mass per concept family, but they do not treat "prime numbers" as just one more member of a shared "special sets" family with squares, cubes and powers. The one change is that primes form their own concept family with a separately inferred prior mass, so examples such as 3, 7 or 53, 2 can be read as "primes" strongly enough that people endorse other primes (13, 29, 5, 11) and reject nearby composites, independently of how readily they read 81, 25 as "squares" or 8, 64 as "powers".

**Source:** `/scr/wurgaft/auto-psych/agent_trees/043c9ecd8d452a7b/repo/_runs/cell_1/experiment2/model_loop/models/threshold_concepts_primes_family.py`

### erlang_interval_size_family_prior — rank 4, 1.8 ± 1.3 nats behind the best (1.4× dse: statistically tied with the best), ELPD-LOO -2260.7

**Hypothesis:** Refinement of `exception_rule_family_prior_2`: people do the same Bayesian concept learning over rules and intervals, with the size principle and a separate prior mass per concept family (exception rules in their own family), but they do not regard all numeric intervals as equally plausible a priori. The one change is an Erlang-shaped prior over interval width with a learned characteristic size (as in Tenenbaum's number game): very narrow and very wide ranges are a priori unlikely, so after a single example or tightly clustered examples people generalise to a band of moderate width around them rather than collapsing onto the tightest interval, and after widely spread examples broad ranges are penalised relative to natural rules.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/043c9ecd8d452a7b/repo/_runs/cell_1/experiment2/model_loop/models/erlang_interval_size_family_prior.py`

### powers_own_family_erlang_interval — rank 5, 2.3 ± 1.5 nats behind the best (1.6× dse: statistically tied with the best), ELPD-LOO -2261.2

**Hypothesis:** Refinement of `erlang_interval_size_family_prior`: people do the same Bayesian concept learning over rules and numeric intervals, with the size principle, a prior mass per concept family (exception rules in their own family) and an Erlang prior over interval width, but they do not lump "powers of m" (2, 4, 8, 16, …; 3, 9, 27, 81; 8, 64; …) together with primes, squares and cubes. The one change is that repeated-multiplication rules form their own concept family with a separately inferred prior mass, so if people find powers an especially natural kind of program, examples like 8 and 64 are read as "powers of 8" (rejecting 2, 4, 16 and 32), and sets of powers of 2 crowd out "even" more strongly than the current model predicts.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/043c9ecd8d452a7b/repo/_runs/cell_1/experiment2/model_loop/models/powers_own_family_erlang_interval.py`

### weber_scaled_interval_size_prior — rank 6, 2.5 ± 1.4 nats behind the best (1.9× dse: statistically tied with the best), ELPD-LOO -2261.4

**Hypothesis:** Refinement of `erlang_interval_size_family_prior`: people do the same Bayesian concept learning over rules and numeric intervals, with the size principle, a prior mass per concept family (exception rules in their own family) and an Erlang prior over interval size, but they judge how big a range is relative to its magnitude (Weber's law) rather than in absolute units. The one change is that the Erlang prior is placed on the interval's width divided by a learned power of its location on the number line, so a range of ten numbers around 85 counts as "narrower" than one around 8; people therefore generalise to wider bands of nearby numbers after large examples than after small ones, with the strength of this magnitude scaling inferred from the data (zero recovers the incumbent).

**Source:** `/scr/wurgaft/auto-psych/agent_trees/043c9ecd8d452a7b/repo/_runs/cell_1/experiment2/model_loop/models/weber_scaled_interval_size_prior.py`

### erlang_interval_size_exception_prior — rank 7, 2.9 ± 1.4 nats behind the best (2.1× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -2261.8

**Hypothesis:** Refinement of `exception_rule_family_prior`: people do the same Bayesian concept learning over rules and intervals, with a prior mass per concept family (exception rules as their own family) and a learned strength of the size principle, but they do not consider every numeric interval equally likely a priori. The one change is an Erlang prior on interval size with a learned typical size, as in Tenenbaum's number game: people expect a magnitude-based program to cover a range of moderate width, so very narrow intervals hugging the examples and very broad ones are both discounted, which reshapes how far "yes" answers extend beyond the examples in magnitude.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/043c9ecd8d452a7b/repo/_runs/cell_1/experiment2/model_loop/models/erlang_interval_size_exception_prior.py`

### digit_sum_concepts_threshold_erlang — rank 8, 3.0 ± 2.1 nats behind the best (1.4× dse: statistically tied with the best), ELPD-LOO -2261.9

**Hypothesis:** Refinement of `threshold_concepts_erlang_interval`: people do the same Bayesian concept learning over number rules, bounded numeric ranges (with an Erlang prior on width) and one-sided thresholds, with the size principle and a prior mass per concept family. The one change is that their space of candidate programs also includes digit-sum concepts — "numbers whose digits add up to s" (e.g. 5, 14, 23, 32, 41, 50 for s = 5) — as a new family with its own inferred prior mass, shared equally among the 18 possible sums. When the examples happen to share a digit sum (especially a single example with a small, rare sum), people endorse other numbers with that digit sum more, and neighbours with a different digit sum less, than the incumbent predicts.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/043c9ecd8d452a7b/repo/_runs/cell_1/experiment2/model_loop/models/digit_sum_concepts_threshold_erlang.py`

### primes_own_family_erlang_interval — rank 9, 4.0 ± 2.3 nats behind the best (1.8× dse: statistically tied with the best), ELPD-LOO -2262.9

**Hypothesis:** Refinement of `powers_own_family_erlang_interval`: people do the same Bayesian concept learning over rules and numeric intervals, with the size principle, a prior mass per concept family (exception rules and powers of m each in their own family) and an Erlang prior over interval width, but they do not treat "prime numbers" as the same kind of program as squares and cubes. The one change is that primes form their own concept family with a separately inferred prior mass, so how readily people read examples such as 3, 7 or 53, 2 as "primes" (endorsing 29 or 59 and rejecting nearby composites) is no longer tied to how readily they read 81, 25 as "squares".

**Source:** `/scr/wurgaft/auto-psych/agent_trees/043c9ecd8d452a7b/repo/_runs/cell_1/experiment2/model_loop/models/primes_own_family_erlang_interval.py`

### weber_noisy_interval_membership — rank 10, 5.2 ± 2.2 nats behind the best (2.3× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -2264.0

**Hypothesis:** People do Bayesian concept learning over natural rules and numeric ranges as in the best model (size principle, a prior mass per concept family, an Erlang prior on range width), but they check whether a target falls inside a magnitude range using their approximate, noisy sense of number rather than exact arithmetic. The one distortion is this Weber-like perceptual noise, which grows in proportion to a number's size: symbolic rules (even, ends in 3, powers of 2) are checked exactly, but the edges of a range are blurred, so large targets just outside a range are often accepted and large targets just inside are sometimes rejected, while range boundaries among small numbers stay sharp.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/043c9ecd8d452a7b/repo/_runs/cell_1/experiment2/model_loop/models/weber_noisy_interval_membership.py`

### exception_rule_family_prior — rank 11, 15.8 ± 6.0 nats behind the best (2.6× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -2274.7

**Hypothesis:** Refinement of `size_principle_strength_family_prior`: people do the same Bayesian concept learning over rules and intervals, with a prior weight per concept family and a learned strength of the size principle, but they do not treat ad-hoc "exception" rules (powers of 2 except 32, powers of 2 plus 37) as being as natural as clean mathematical sets. The one change is to give these exception rules their own concept family with its own inferred prior mass, so people can find "powers of 2, or 37" implausible and reject the 37 lure for sets of powers of 2 while still generalising to the other powers of 2.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/043c9ecd8d452a7b/repo/_runs/cell_1/experiment2/model_loop/models/exception_rule_family_prior.py`

### power_base_family_prior — rank 12, 16.2 ± 6.2 nats behind the best (2.6× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -2275.0

**Hypothesis:** Refinement of `exception_rule_family_prior_2`: people do the same Bayesian concept learning over rules and intervals with the size principle, a separate prior mass per concept family and the exception rules in their own family, but they do not lump "powers of a base" (powers of 2, 3, ... 10) together with primes, squares and cubes as one "special sets" family. The one change is to give powers-of-a-base rules their own family with its own inferred prior mass: people readily think of a set as successive powers (or repeated multiplication) of a number, while primes, squares and cubes each have a separate, independently inferred plausibility, so generalisation after power-like sets (e.g. 8, 64) and after square- or prime-like sets (e.g. 81, 9 or 3, 31) can differ in strength instead of being tied to a shared per-rule prior.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/043c9ecd8d452a7b/repo/_runs/cell_1/experiment2/model_loop/models/power_base_family_prior.py`

### exception_rule_family_prior_2 — rank 13, 16.5 ± 6.2 nats behind the best (2.7× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -2275.3

**Hypothesis:** Refinement of `family_weighted_concept_prior`: people do the same Bayesian concept learning over rules and intervals with the size principle and a separate prior mass per concept family, but the two gerrymandered exception rules ("powers of 2 except 32" and "powers of 2 plus 37") are not natural mathematical sets — people give them their own, separately inferred (and expectedly tiny) prior mass rather than an equal share of the "special sets" family alongside primes, squares and powers. The one change is this separate family for exception rules, so for sets of powers of 2 people should reject the lure 37 nearly always while still endorsing the other powers of 2.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/043c9ecd8d452a7b/repo/_runs/cell_1/experiment2/model_loop/models/exception_rule_family_prior_2.py`

### parity_bounded_range_concepts — rank 14, 17.0 ± 6.3 nats behind the best (2.7× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -2275.9

**Hypothesis:** Refinement of `exception_rule_family_prior_2`: people do the same Bayesian concept learning over rules and numeric intervals, with the size principle and a prior mass per concept family (the exception rules in their own family), but their hypothesis space also contains conjunctive "parity within a range" concepts — "even numbers between 20 and 60", "odd numbers from 9 to 69" — as a new family with its own inferred prior mass. The one change is this composed rule-plus-magnitude family: when examples are all even (or all odd) and clustered in magnitude, people generalise to same-parity numbers near the examples while rejecting both nearby numbers of the other parity and same-parity numbers far outside the examples' range, which neither a pure parity rule nor a pure interval predicts.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/043c9ecd8d452a7b/repo/_runs/cell_1/experiment2/model_loop/models/parity_bounded_range_concepts.py`

### recognition_shortcut_for_examples — rank 15, 31.0 ± 11.4 nats behind the best (2.7× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -2289.9

**Hypothesis:** People infer the hidden program as in the best current model (Bayesian concept learning over number rules, bounded ranges and one-sided thresholds) and answer most targets by probability matching on that belief, but when the target is one of the numbers they were just shown they do not consult their inferred concept at all: they answer from recognition, saying "yes" at a single fixed rate that does not depend on the set. So the yes rate for a repeated example is the same whether one example or four were shown and whatever concept they suggest, unlike the best model, where it falls as tight concepts that exclude random-looking examples gain belief.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/043c9ecd8d452a7b/repo/_runs/cell_1/experiment2/model_loop/models/recognition_shortcut_for_examples.py`

### log_number_line_ranges — rank 16, 303.8 ± 34.6 nats behind the best (8.8× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -2562.7

**Hypothesis:** People do Bayesian concept learning over number rules (with a prior mass per concept family, the exception rules in their own family) and numeric ranges with the size principle, but they represent magnitude on a compressed, logarithmic mental number line: a range is a band on that line, so ranges are a priori equally likely wherever they sit on the log line and their plausibility depends on their ratio width (e.g. 5-10 is as natural as 50-100), not their width in units. As a result, magnitude-based generalisation scales with the examples' size — after small examples people generalise only to nearby small numbers, after large examples to a proportionally much wider band.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/043c9ecd8d452a7b/repo/_runs/cell_1/experiment2/model_loop/models/log_number_line_ranges.py`

### newest_example_size_principle_wsls — rank 17, 308.3 ± 26.0 nats behind the best (11.8× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -2567.1

**Hypothesis:** People hold a single running guess about the program (a rule or a numeric range, weighed by family plausibility and an Erlang prior on range width) and revise it example by example in the order shown: they keep their current guess with a probability that grows with how well it explains the newest example, and otherwise draw a fresh guess — but when they redraw they judge the "suspicious coincidence" (the size principle) only for the newest example, merely requiring the guess to cover the earlier ones. So evidence for tight concepts does not compound across examples as in an ideal Bayesian learner, and which example drives specificity depends on the order shown (the last examples, plus whatever guess has survived from the first).

**Source:** `/scr/wurgaft/auto-psych/agent_trees/043c9ecd8d452a7b/repo/_runs/cell_1/experiment2/model_loop/models/newest_example_size_principle_wsls.py`

### average_digit_magnitude_similarity — rank 18, 788.3 ± 40.6 nats behind the best (19.4× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -3047.2

**Hypothesis:** People do not infer a rule; they judge a target by its average similarity to the examples, where two numbers are similar if they are close in magnitude and share their final digit, with similarity decaying exponentially along the number line and dropping by a fixed factor when the final digits differ. Because the similarity is averaged rather than summed over examples, "yes" depends on the proportion of the examples the target resembles, not on how many examples were shown, so adding dissimilar examples to a set dilutes generalisation to a target near one of them.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/043c9ecd8d452a7b/repo/_runs/cell_1/experiment2/model_loop/models/average_digit_magnitude_similarity.py`

### shared_category_exemplar_averaging — no comparison row

**Hypothesis:** People do not infer one program that explains all the examples together; they store each example as a separate exemplar and judge the target by its average similarity to the stored examples, where two numbers are similar to the degree that they fall together in the familiar number categories people know (parity, multiples, final or contained digits, primes, squares, powers, magnitude ranges and thresholds), with small, distinctive shared categories counting more than broad ones (Shepard's additive-clustering view of number similarity). Because each example generalises on its own and the similarities are averaged rather than multiplied, the "suspicious coincidence" of several examples sharing a tight category never accumulates: a target similar to only one example of a set gets partial "yes" support, and adding examples does not sharpen generalisation the way it does for a Bayesian concept learner.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/043c9ecd8d452a7b/repo/_runs/cell_1/experiment2/model_loop/models/shared_category_exemplar_averaging.py`

### range_size_principle_strength — no comparison row

**Hypothesis:** Refinement of `threshold_concepts_erlang_interval`: people do the same Bayesian concept learning over number rules, bounded numeric ranges (with an Erlang prior on width) and one-sided magnitude thresholds (their own family), with a prior mass per concept family, but they do not apply the "suspicious coincidence" of the size principle to magnitude concepts as strongly as to arithmetic and digit rules. The one change is that, for ranges and thresholds only, how much an example's probability shrinks with the concept's width is governed by a learned exponent (one recovers the incumbent, zero means ranges are treated as mere filters the examples pass), so how tightly people's magnitude-based generalisation hugs the examples, relative to rule-based generalisation, is inferred from the data rather than fixed by the ideal learner.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/043c9ecd8d452a7b/repo/_runs/cell_1/experiment2/model_loop/models/range_size_principle_strength.py`

### centre_weighted_ranges_with_thresholds — no comparison row

**Hypothesis:** Refinement of `threshold_concepts_erlang_interval`, adding one component from `centre_weighted_range_sampling`. People do the same Bayesian concept learning over number rules, bounded ranges (with an Erlang prior on width) and one-sided thresholds (their own family), with the size principle and a prior mass per concept family. The one change is that people expect a bounded-range program ("numbers between 30 and 50") to produce numbers clustered toward the middle of its range, while a threshold program ("numbers above 60") still produces its members evenly. How strongly ranges cluster is inferred from the data, and zero recovers the incumbent. An example at a range's edge is then a suspicious coincidence, so ranges centred on the examples that reach past them gain belief relative to thresholds and rules.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/043c9ecd8d452a7b/repo/_runs/cell_1/experiment2/model_loop/models/centre_weighted_ranges_with_thresholds.py`

## Pruned models (out of the set; narrowest margin first)

### offset_progression_concepts — pruned (experiment1 end of experiment)

**Margin:** 2.6 nats behind powers_own_family_erlang_interval; retired to keep the live set at 8 models: ELPD-LOO rank 10 of 12

**Hypothesis:** People learn the hidden program's concept by Bayesian inference over natural rules and numeric ranges (with the size principle and a separate prior plausibility for each family of concepts), and their space of candidate rules also includes arithmetic progressions that do not start at zero — "every k-th number starting from r", such as 4, 11, 18, 25, … or the odd multiples of 3 — as their own family with its own prior plausibility. When the examples are evenly spaced by a common step, people therefore endorse targets that continue the same progression and reject nearby targets that fall off it.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/043c9ecd8d452a7b/repo/_runs/cell_1/experiment1/model_loop/models/pruned/offset_progression_concepts.py`

### mixed_weak_strong_sampling — pruned (experiment1 end of experiment)

**Margin:** 2.6 nats behind powers_own_family_erlang_interval; retired to keep the live set at 8 models: ELPD-LOO rank 9 of 12

**Hypothesis:** People do Bayesian concept learning over the same rules and numeric intervals (with a prior mass per concept family, the exception rules in their own family), but they do not assume every example was deliberately drawn from the concept: they believe each example is, with a learned probability, merely a number picked at random that happened to pass the program's test (weak sampling) rather than one the program generated (strong sampling, the size principle). The one change is this inferred mix of weak and strong sampling, which blunts the size principle in proportion to how large a concept is, so it disagrees most with the best model on multi-example sets consistent with both a tight and a broad natural rule — e.g. other even numbers after 16, 8, 64, or in-range targets that break a shared final digit — which it predicts are endorsed more often.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/043c9ecd8d452a7b/repo/_runs/cell_1/experiment1/model_loop/models/pruned/mixed_weak_strong_sampling.py`

### digit_vocabulary_concepts — pruned (experiment1 end of experiment)

**Margin:** 3.1 nats behind powers_own_family_erlang_interval; retired to keep the live set at 8 models: ELPD-LOO rank 11 of 12

**Hypothesis:** People learn the hidden program's concept by Bayesian inference over natural number rules and numeric ranges (with the size principle and a prior plausibility per concept family), but they also entertain "digit vocabulary" concepts — the program writes numbers using only a particular set of digits — as their own family. When all the examples are spelled with a small set of digits, that coincidence makes such a concept convincing, so people endorse targets that recombine the examples' digits (e.g. 76 after 27, 37, 67) more than neighbouring targets that introduce a new digit.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/043c9ecd8d452a7b/repo/_runs/cell_1/experiment1/model_loop/models/pruned/digit_vocabulary_concepts.py`

### leaky_working_memory_concept_learning — pruned (experiment1 end of experiment)

**Margin:** 3.4 nats behind powers_own_family_erlang_interval; retired to keep the live set at 8 models: ELPD-LOO rank 12 of 12

**Hypothesis:** People do the same Bayesian concept learning as the best model (rules and numeric intervals, the size principle, a prior mass per concept family with the exception rules in their own family), but they reason only from the examples they still hold in working memory: each shown example is independently retained with an inferred probability and forgotten otherwise, and a forgotten example simply does not enter the inference. Their "yes" rate is therefore the average, over which examples happened to be remembered, of the posterior probability that the concept contains the target — so multi-example sets generalise more broadly and less rule-tightly than a learner using every example would predict.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/043c9ecd8d452a7b/repo/_runs/cell_1/experiment1/model_loop/models/pruned/leaky_working_memory_concept_learning.py`

### size_principle_strength_family_prior — pruned (experiment1 end of experiment)

**Margin:** 11.8 nats behind powers_own_family_erlang_interval (2.4× dse)

**Hypothesis:** Refinement of `family_weighted_concept_prior`: people do the same Bayesian concept learning over rules and intervals with a separate prior weight per concept family, but they treat the examples as more (or less) diagnostic than single independent random draws — the evidence the examples provide is raised to a learned "sampling strength" that sharpens or flattens the size principle. The one change is this learned strength of the size principle: if people over-weight how suspicious it is that every example ends in the same digit, a tight rule such as "ends in 3" decisively beats a broad interval spanning the examples, so in-range targets that break the shared final digit are rejected.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/043c9ecd8d452a7b/repo/_runs/cell_1/experiment1/model_loop/models/pruned/size_principle_strength_family_prior.py`

### family_weighted_concept_prior — pruned (experiment1 end of experiment)

**Margin:** 11.9 nats behind powers_own_family_erlang_interval (2.4× dse)

**Hypothesis:** Refinement of `bigelow_piantadosi_mixture`: people do Bayesian concept learning over the same rules and intervals with the size principle, but they do not treat all rules as equally likely a priori. Instead, each family of concepts — parity (odd/even), special mathematical sets (primes, squares, cubes, powers), multiples of k, "ends in digit d", "contains digit d", and numeric intervals — has its own prior probability, shared equally by that family's members. So a broad but natural rule like "even" or "multiples of 3" can outweigh a tighter but implausible one like "contains the digit 2", which the size principle alone would favor.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/043c9ecd8d452a7b/repo/_runs/cell_1/experiment1/model_loop/models/pruned/family_weighted_concept_prior.py`

### posterior_sharpened_concept_selection — pruned (experiment1 end of experiment)

**Margin:** 12.9 nats behind powers_own_family_erlang_interval (2.7× dse)

**Hypothesis:** Refinement of `family_weighted_concept_prior`: people weigh the same rules and intervals by prior plausibility and the size principle, but they do not probability-match over the whole posterior — when settling on what the program does, they favour the most convincing concepts more strongly than Bayes would (or less, if their selection is noisy), as if choosing among concepts with a sharpening exponent on each concept's posterior weight. The one change is this inferred decision temperature: a sharpening exponent above one lets a clearly winning rule like "ends in 3" crowd out the many loosely fitting intervals, so in-range targets that break the shared final digit are rejected more often.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/043c9ecd8d452a7b/repo/_runs/cell_1/experiment1/model_loop/models/pruned/posterior_sharpened_concept_selection.py`

### family_weighted_learned_reliability — pruned (experiment1 end of experiment)

**Margin:** 14.7 nats behind powers_own_family_erlang_interval (2.9× dse)

**Hypothesis:** Refinement of `family_weighted_concept_prior`: people do the same Bayesian concept learning over rules and intervals with the size principle and a separate prior mass for each concept family, but they do not assume a fixed 10% of the examples are random noise — how much they trust the examples to come from the hidden concept is a free, inferred parameter (the component taken from `learned_example_reliability_ceiling`). The one change is this learned example reliability: if people trust the examples more, tight intervals that cover only one example and write the others off as noise lose their support, so for sets whose examples share a final digit people reject in-range targets that break that digit more firmly than the incumbent predicts.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/043c9ecd8d452a7b/repo/_runs/cell_1/experiment1/model_loop/models/pruned/family_weighted_learned_reliability.py`

### family_prior_reliability_ceiling — pruned (experiment1 end of experiment)

**Margin:** 17.8 nats behind powers_own_family_erlang_interval (3.3× dse)

**Hypothesis:** Refinement of `learned_example_reliability_ceiling`: people do Bayesian concept learning over the same 45 rules and all intervals with the size principle, trust the examples to a learned degree, and endorse a believed member only up to a ceiling — but they do not treat all rules as equally likely a priori. The one change is grafting the family-weighted concept prior: each family of concepts (parity, special mathematical sets, multiples of k, "ends in digit d", "contains digit d", intervals) has its own prior mass shared by its members, so a natural rule like "ends in 3" can outweigh the interval spanning the examples, which sharpens rejection of in-range targets that break a shared final digit.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/043c9ecd8d452a7b/repo/_runs/cell_1/experiment1/model_loop/models/pruned/family_prior_reliability_ceiling.py`

### family_prior_confidence_threshold — pruned (experiment1 end of experiment)

**Margin:** 31.6 nats behind powers_own_family_erlang_interval (4.5× dse)

**Hypothesis:** Refinement of `size_principle_strength_family_prior`: people do the same Bayesian concept learning over rules and intervals, with a prior weight per concept family and a learned strength of the size principle, but they do not answer "yes" with the probability that the concept contains the target; instead they apply a noisy confidence criterion, saying "yes" only when their belief that the target belongs clears a learned threshold (the decision stage taken from `membership_confidence_threshold`). The one change is this thresholded decision rule: targets backed by only a minority of the posterior — such as 37 after powers of 2, which only the ad-hoc "powers of 2, or 37" rule contains — fall below the criterion and are rejected far more often than probability matching predicts, while confidently believed members are endorsed nearly always.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/043c9ecd8d452a7b/repo/_runs/cell_1/experiment1/model_loop/models/pruned/family_prior_confidence_threshold.py`

### generative_production_likelihood — pruned (experiment1 end of experiment)

**Margin:** 127.5 nats behind powers_own_family_erlang_interval (5.2× dse)

**Hypothesis:** People do not answer by asking whether the target belongs to the concept they have inferred; they answer by how readily they can imagine the program actually producing that target next — the probability of the target as the program's next output, which is high for targets in small, tight concepts (a lone powers-of-2 value, a narrow interval around clustered examples) and low for targets that belong only to broad concepts (all even numbers, a wide interval) even when membership is certain. Their yes rate rises smoothly with this predicted production probability, relative to what a program emitting numbers at random would give.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/043c9ecd8d452a7b/repo/_runs/cell_1/experiment1/model_loop/models/pruned/generative_production_likelihood.py`

### learned_example_reliability_ceiling — pruned (experiment1 end of experiment)

**Margin:** 261.7 nats behind powers_own_family_erlang_interval (9.6× dse)

**Hypothesis:** Refinement of `concept_learning_endorsement_ceiling`: people do Bayesian concept learning over the same 45 equally likely rules and all intervals (size principle, rule-vs-interval prior), and endorse a believed member only up to a fixed ceiling, but they do not assume a fixed 10% of the examples are random noise — how much they trust the examples to come from the hidden concept is itself a free parameter. The one change is this learned example-reliability: if people trust the examples more than the original model assumes, a single example that breaks a rule (an odd number among evens, a different final digit) rules that concept out more decisively, sharpening rejection of parity- and digit-violating targets.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/043c9ecd8d452a7b/repo/_runs/cell_1/experiment1/model_loop/models/pruned/learned_example_reliability_ceiling.py`

### concept_learning_endorsement_ceiling — pruned (experiment1 end of experiment)

**Margin:** 272.5 nats behind powers_own_family_erlang_interval (10.0× dse)

**Hypothesis:** Refinement of `bigelow_piantadosi_mixture`: people do the same Bayesian concept learning over rules and intervals (size principle, rule-vs-interval prior lambda_0), but they do not say "yes" whenever they believe the concept contains the target — they endorse a target only with a fixed probability (an endorsement ceiling below 1), so "yes" is that ceiling times the posterior probability of membership. The one change is this conservative endorsement ceiling, which lowers yes rates uniformly for every believed member (near-example, same-parity, same-digit targets alike) instead of letting the posterior map straight onto yes.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/043c9ecd8d452a7b/repo/_runs/cell_1/experiment1/model_loop/models/pruned/concept_learning_endorsement_ceiling.py`

### bigelow_piantadosi_mixture — pruned (experiment1 end of experiment)

**Margin:** 330.3 nats behind powers_own_family_erlang_interval (8.5× dse)

**Hypothesis:** Bayesian concept learning over 45 rules (odd, even, primes, squares, cubes, powers of 2-10, powers of 2 without 32 or with 37, multiples of 3-9, 11 and 12, ending in 0-9, containing the digit 0-9) and every interval of 1-100. A rule is chosen with probability lambda_0, and the rules are equally likely; each interval has prior (1 - lambda_0) / 100^2. Examples are sampled from the concept (the size principle), 10% of them uniformly from 1-100, and "yes" is the posterior probability that the concept contains the target.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/043c9ecd8d452a7b/repo/_runs/cell_1/experiment1/model_loop/models/pruned/bigelow_piantadosi_mixture.py`

### membership_confidence_threshold — pruned (experiment1 end of experiment)

**Margin:** 333.1 nats behind powers_own_family_erlang_interval (10.9× dse)

**Hypothesis:** People infer the hidden concept as in plain Bayesian concept learning (equally likely rules versus intervals, with the size principle), but they do not answer "yes" with the probability that the concept contains the target; instead they apply a noisy confidence criterion, saying "yes" only when their belief that the target belongs clears a learned threshold. Targets supported by only part of the posterior — such as 37 for powers-of-2 examples, which only the ad-hoc "powers of 2, or 37" rule contains — therefore fall below the criterion and are rejected far more often than probability matching predicts, while targets believed members are endorsed nearly always.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/043c9ecd8d452a7b/repo/_runs/cell_1/experiment1/model_loop/models/pruned/membership_confidence_threshold.py`

### evidence_overweighted_size_principle — pruned (experiment1 end of experiment)

**Margin:** 368.3 nats behind powers_own_family_erlang_interval (9.7× dse)

**Hypothesis:** People do Bayesian concept learning over the same rules (equally likely) and numeric intervals with the size principle, but they over-weight the evidence: each example counts as if it were seen more than once, so the likelihood of the examples is raised to a power above one (an "evidence-overweighting" distortion of the normative posterior). This makes tight rules that fit every example (e.g. "ends in 3" for 13, 23, 43) crowd out broader intervals more decisively than the normative model allows, so in-range targets that break such a rule are rejected more sharply.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/043c9ecd8d452a7b/repo/_runs/cell_1/experiment1/model_loop/models/pruned/evidence_overweighted_size_principle.py`

### shared_property_matching — pruned (experiment1 end of experiment)

**Margin:** 383.7 nats behind powers_own_family_erlang_interval (12.6× dse)

**Hypothesis:** People do not weigh competing concepts by how well they explain the examples; instead they note every salient number property that all the examples share (parity, final digit, tens digit, being a multiple or power of something, prime, square, containing a digit) and say "yes" to a target in proportion to how many of those shared properties it also has. A target is rejected to the degree it breaks the examples' common properties, regardless of how close it lies in magnitude to any example.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/043c9ecd8d452a7b/repo/_runs/cell_1/experiment1/model_loop/models/pruned/shared_property_matching.py`

### rule_only_concept_learning — pruned (experiment1 end of experiment)

**Margin:** 398.2 nats behind powers_own_family_erlang_interval (12.3× dse)

**Hypothesis:** People generalize only through arithmetic and digit rules, never through numerical magnitude: they weigh the 45 candidate rules (parity, primes, squares, cubes, powers, multiples of k, "ends in d", "contains d") by how well each explains the examples under the size principle, and say "yes" with the posterior probability that the target obeys the rule, with no interval or closeness-based hypotheses at all. A target lying right between the examples but breaking every rule they share (e.g. 23 for 21, 41, 81) should therefore be rejected, not endorsed as nearby.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/043c9ecd8d452a7b/repo/_runs/cell_1/experiment1/model_loop/models/pruned/rule_only_concept_learning.py`

### shared_property_profile_match — pruned (experiment1 end of experiment)

**Margin:** 424.5 nats behind powers_own_family_erlang_interval (11.6× dse)

**Hypothesis:** People do not average over candidate concepts or generalise by numerical closeness; instead they note which arithmetic and digit properties (parity, primality, squares, powers, multiples, final digit, contained digit) every example shares, and say "yes" in proportion to how much of that shared-property profile the target also has, with rarer (more distinctive) properties counting more. This disagrees most sharply with the best model on targets just next to an example that break the shared properties (e.g. 17 for the set 16, 8, 64), which it predicts are usually rejected because there is no interval or magnitude-based generalisation at all.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/043c9ecd8d452a7b/repo/_runs/cell_1/experiment1/model_loop/models/pruned/shared_property_profile_match.py`

### distinctive_feature_matching — pruned (experiment1 end of experiment)

**Margin:** 441.3 nats behind powers_own_family_erlang_interval (11.7× dse)

**Hypothesis:** People do not entertain explicit hypotheses about the hidden program; instead they judge a target by feature matching: they notice which arithmetic and digit properties (parity, primality, being a square or a power, being a multiple of k, final digit, contained digits, decade) all the examples have in common, and say "yes" in proportion to how much of that common-feature profile the target also shows, with rarer (more distinctive) shared properties counting more than common ones. Numerical closeness matters only insofar as it produces shared features, so a target missing a distinctive common property is rejected even if it is adjacent to an example.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/043c9ecd8d452a7b/repo/_runs/cell_1/experiment1/model_loop/models/pruned/distinctive_feature_matching.py`

### log_ratio_adaptive_similarity_gradient — pruned (experiment1 end of experiment)

**Margin:** 473.7 nats behind powers_own_family_erlang_interval (15.2× dse)

**Hypothesis:** People judge a target by its similarity in magnitude to the examples, measured as the ratio distance (difference in log magnitude) to the nearest example, so a fixed absolute gap matters more among small numbers than large ones. Their willingness to say "yes" falls off smoothly with that ratio distance, and the tolerance of the gradient widens in proportion to how spread out (in log magnitude) the examples are, so tightly clustered examples yield a sharp generalization gradient and widely scattered examples a broad one.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/043c9ecd8d452a7b/repo/_runs/cell_1/experiment1/model_loop/models/pruned/log_ratio_adaptive_similarity_gradient.py`

### digit_parity_exemplar_similarity — pruned (experiment1 end of experiment)

**Margin:** 514.2 nats behind powers_own_family_erlang_interval (13.5× dse)

**Hypothesis:** People do not infer a hidden rule; they store the shown examples as exemplars and judge a target by its summed similarity to them, as in an exemplar (generalized context) model. Each number is represented on three psychological dimensions — its magnitude, its parity, and its final digit — and similarity to an example decays exponentially with magnitude distance and with each mismatch in parity or final digit; the probability of "yes" is the summed similarity relative to a fixed background response criterion.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/043c9ecd8d452a7b/repo/_runs/cell_1/experiment1/model_loop/models/pruned/digit_parity_exemplar_similarity.py`

### first_example_anchored_then_filtered — pruned (experiment1 end of experiment)

**Margin:** 613.9 nats behind powers_own_family_erlang_interval (11.3× dse)

**Hypothesis:** People form their guess about the program from the first example alone — weighing rules and numeric intervals by family-level prior plausibility and by how suspicious it would be to draw that first number from each concept (the size principle) — and then use each later example only as a consistency check: a concept that fails to contain a later example is (nearly) discarded, but a later example that fits does not make tight concepts more convincing than broad ones. So the first example sets how specific the guess is, and the rest merely prune it; e.g. after 16, 8, 64 a broad rule like "even" keeps the share it had after seeing 16 alone instead of being crowded out by "powers of 2".

**Source:** `/scr/wurgaft/auto-psych/agent_trees/043c9ecd8d452a7b/repo/_runs/cell_1/experiment1/model_loop/models/pruned/first_example_anchored_then_filtered.py`
