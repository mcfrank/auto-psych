# Refinement menu

The models you may refine, other than the incumbent `graded_prior_digit_sum_concepts`. Each entry is one mechanism: its hypothesis in full as its author stated it, how it stands, and where its source is. This is a menu, not a list of what is ruled out: a pruned model lost to the best model on the data it was scored on, by the stated margin, but its mechanism may be partly right — and a model that lost narrowly is the most promising target here. Read the source of the model you pick before you write anything.

## Live models other than the incumbent (in the set; best first)

### centre_weighted_ranges_digit_sums — rank 1, 0.9 ± 1.6 nats behind the best (0.6× dse: statistically tied with the best), ELPD-LOO -3523.7

**Hypothesis:** Refinement of `centre_weighted_range_sampling`, adding one component from `digit_sum_concepts_threshold_erlang`: people do the same Bayesian concept learning over number rules and numeric ranges, with the size principle, a prior mass per concept family (exception rules in their own family), an Erlang prior over range width, and range outputs expected to cluster toward the middle of the range. The one change is that the programs people consider also include digit-sum concepts, "numbers whose digits add up to s" (e.g. 5, 14, 23, 32, 41, 50), as a new family with its own inferred prior mass shared equally among the 18 possible sums. Because these concepts are tiny, examples that happen to share a digit sum are a suspicious coincidence, so people endorse other numbers with that sum, and reject nearby numbers with a different sum, more than the incumbent predicts.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/043c9ecd8d452a7b/repo/_runs/cell_1/experiment3/model_loop/models/centre_weighted_ranges_digit_sums.py`

### centre_weighted_range_sampling — rank 2, 1.3 ± 1.9 nats behind the best (0.7× dse: statistically tied with the best), ELPD-LOO -3524.0

**Hypothesis:** Refinement of `erlang_interval_size_family_prior`: people do the same Bayesian concept learning over number rules and numeric ranges, with the size principle, a prior mass per concept family (exception rules in their own family) and an Erlang prior over range width, but they do not expect a range-based program to produce every number in its range equally often — they expect its outputs to cluster toward the middle of the range, with how strongly they do so inferred from the data (none recovers the parent model). The one change is this centre-weighted sampling from ranges: examples sitting at a range's edge are a "suspicious coincidence" for it, so ranges that are centred on the examples and extend beyond the lowest and highest example gain support, and people say "yes" to targets somewhat past the examples' extremes more often than a uniform-sampling learner predicts.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/043c9ecd8d452a7b/repo/_runs/cell_1/experiment3/model_loop/models/centre_weighted_range_sampling.py`

### location_biased_range_prior — rank 3, 1.4 ± 1.9 nats behind the best (0.8× dse: statistically tied with the best), ELPD-LOO -3524.2

**Hypothesis:** People infer the hidden program as a Bayesian concept learner over number rules (including digit-sum rules such as "digits add up to 5") and numeric ranges of a typical width, but they do not regard a range as equally plausible wherever it sits on the number line: they expect a magnitude-based program to live in a particular region (lower or higher numbers), so a range's prior plausibility rises or falls steadily with where its middle lies, with the direction and strength inferred from the data. The sharpest disagreement with the best model is in the direction of magnitude generalisation: after a single example or a tight cluster in mid-range, this model predicts asymmetric "yes" answers, more for targets on the favoured side of the examples than for equally distant targets on the other side, whereas the best model generalises symmetrically.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/043c9ecd8d452a7b/repo/_runs/cell_1/experiment3/model_loop/models/location_biased_range_prior.py`

### round_edge_ranges_digit_sums — rank 4, 1.7 ± 2.4 nats behind the best (0.7× dse: statistically tied with the best), ELPD-LOO -3524.4

**Hypothesis:** Refinement of `centre_weighted_ranges_digit_sums`, adding one component from `round_number_range_boundaries`: people do the same Bayesian concept learning over number rules, digit-sum rules ("numbers whose digits add up to s", one family with an equal share per sum) and numeric ranges (Erlang width prior, outputs clustered toward the range's middle), with the size principle and a prior mass per concept family. The one change is that multiples of ten act as reference points when people judge how plausible a range is a priori. A range whose lower or upper edge falls exactly on a multiple of ten is judged more or less likely than an otherwise identical range, with the direction and strength inferred from the data. So the digit-sum coincidence decides which rule-based numbers are endorsed, while the round-number bias decides whether magnitude-based "yes" answers stop just before or exactly at a nearby multiple of ten (e.g. whether 40 is endorsed after examples in the low thirties).

**Source:** `/scr/wurgaft/auto-psych/agent_trees/043c9ecd8d452a7b/repo/_runs/cell_1/experiment3/model_loop/models/round_edge_ranges_digit_sums.py`

### centre_weighted_round_edge_ranges — rank 5, 2.2 ± 3.0 nats behind the best (0.7× dse: statistically tied with the best), ELPD-LOO -3524.9

**Hypothesis:** Refinement of `centre_weighted_ranges_with_thresholds`, adding one component from `round_number_range_boundaries`. People do the same Bayesian concept learning over number rules, bounded ranges (Erlang width prior, outputs expected to cluster toward the range's middle) and one-sided thresholds, with the size principle and a prior mass per concept family; the one change is that multiples of ten act as reference points in judging how plausible a bounded range is a priori, so a range whose lower or upper edge falls exactly on a multiple of ten is judged more or less likely than an otherwise identical range, with the direction and strength of this bias inferred from the data. Centre-weighting decides how far past the examples a range is expected to reach, and the round-number bias decides whether "yes" answers stop just before or exactly at a nearby multiple of ten (e.g. whether 40 is endorsed after examples in the low thirties).

**Source:** `/scr/wurgaft/auto-psych/agent_trees/043c9ecd8d452a7b/repo/_runs/cell_1/experiment3/model_loop/models/centre_weighted_round_edge_ranges.py`

### centre_weighted_ranges_with_thresholds — rank 6, 2.6 ± 2.9 nats behind the best (0.9× dse: statistically tied with the best), ELPD-LOO -3525.4

**Hypothesis:** Refinement of `threshold_concepts_erlang_interval`, adding one component from `centre_weighted_range_sampling`. People do the same Bayesian concept learning over number rules, bounded ranges (with an Erlang prior on width) and one-sided thresholds (their own family), with the size principle and a prior mass per concept family. The one change is that people expect a bounded-range program ("numbers between 30 and 50") to produce numbers clustered toward the middle of its range, while a threshold program ("numbers above 60") still produces its members evenly. How strongly ranges cluster is inferred from the data, and zero recovers the incumbent. An example at a range's edge is then a suspicious coincidence, so ranges centred on the examples that reach past them gain belief relative to thresholds and rules.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/043c9ecd8d452a7b/repo/_runs/cell_1/experiment3/model_loop/models/centre_weighted_ranges_with_thresholds.py`

### graded_digit_sums_primes_family — rank 7, 5.1 ± 2.6 nats behind the best (2.0× dse: statistically tied with the best), ELPD-LOO -3527.8

**Hypothesis:** Refinement of `graded_prior_digit_sum_concepts`, adding one component from `centre_weighted_ranges_primes_family`: people do the same Bayesian concept learning over number rules, digit-sum rules (with a smoothly graded prior over the sum) and centre-weighted numeric ranges with an Erlang width prior, the size principle and a prior mass per concept family. The one change is that "prime numbers" is no longer lumped with squares, cubes and powers in a shared special-sets family but forms its own family with a separately inferred prior mass, so how readily people read prime-like sets as "primes" — endorsing other primes and rejecting nearby composites — no longer depends on how plausible squares or powers are.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/043c9ecd8d452a7b/repo/_runs/cell_1/experiment3/model_loop/models/graded_digit_sums_primes_family.py`

### centre_weighted_ranges_primes_family — rank 8, 5.9 ± 3.0 nats behind the best (2.0× dse: statistically tied with the best), ELPD-LOO -3528.7

**Hypothesis:** Refinement of `centre_weighted_range_sampling`, adding one component from `without_replacement_primes_family`: people do the same Bayesian concept learning over number rules and numeric ranges, with the size principle, a prior mass per concept family (exception rules in their own family), an Erlang prior over range width and range outputs expected to cluster toward the middle of the range. The one change is that "prime numbers" is no longer lumped with squares, cubes and powers in a shared special-sets family but forms its own family with a separately inferred prior mass, so how readily people read prime-like sets (e.g. 3, 7 or 53, 2) as "primes" — endorsing other primes and rejecting nearby composites — no longer depends on how plausible squares or powers are.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/043c9ecd8d452a7b/repo/_runs/cell_1/experiment3/model_loop/models/centre_weighted_ranges_primes_family.py`

### first_example_evidence_weight — rank 9, 8.6 ± 5.5 nats behind the best (1.6× dse: statistically tied with the best), ELPD-LOO -3531.3

**Hypothesis:** People infer the hidden program as an ideal Bayesian concept learner would (number rules, bounded ranges with a round-number bias on their edges, and one-sided thresholds, weighed by family plausibility, range width and the size principle, with examples drawn without repetition), but they revise their belief example by example in the order shown and the first example carries its own evidential weight: met before they have any guess about the program, the opening example counts as a looser (or, if the data say so, stronger) piece of evidence than each later example, which counts normally. So the "suspicious coincidence" that the first example points to a tight concept is discounted relative to later examples, which matters most for single-example sets and for which example of a set is shown first.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/043c9ecd8d452a7b/repo/_runs/cell_1/experiment3/model_loop/models/first_example_evidence_weight.py`

### without_replacement_size_principle — rank 10, 8.8 ± 5.2 nats behind the best (1.7× dse: statistically tied with the best), ELPD-LOO -3531.6

**Hypothesis:** Refinement of `threshold_concepts_erlang_interval`: people do the same Bayesian concept learning over number rules, bounded ranges (with an Erlang prior on width) and one-sided thresholds, with a prior mass per concept family, but they believe the program never repeats itself — each example is a member of the concept not already shown, drawn without replacement. The one change is this without-replacement size principle, which makes the "suspicious coincidence" grow faster than an ideal with-replacement learner's whenever the examples use up a large share of a small concept: sets that nearly exhaust a tiny rule or a narrow range (e.g. 4 of the 7 powers of 2, or 11, 15 filling a five-number band) favour that tight concept even more strongly, so people reject targets just outside it more often than the best model predicts, while sets that are few relative to every candidate concept behave as before.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/043c9ecd8d452a7b/repo/_runs/cell_1/experiment3/model_loop/models/without_replacement_size_principle.py`

### round_number_range_boundaries — rank 11, 9.0 ± 5.3 nats behind the best (1.7× dse: statistically tied with the best), ELPD-LOO -3531.7

**Hypothesis:** People infer the hidden program as an ideal Bayesian concept learner would (number rules, bounded ranges and one-sided thresholds, weighed by family plausibility, range width and the size principle, with examples drawn without repetition), but their sense of how plausible a bounded range is is biased by round numbers: multiples of ten act as cognitive reference points, so a range whose lower or upper boundary falls exactly on one (e.g. 20-37 or 40-60) is judged more, or less, likely a priori than an otherwise identical range, with the direction and strength of this round-number bias inferred from the data. This shifts where people stop saying "yes" near multiples of ten — for example whether 40 is included after examples in the low thirties — while leaving rule-based generalisation unchanged.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/043c9ecd8d452a7b/repo/_runs/cell_1/experiment3/model_loop/models/round_number_range_boundaries.py`

### without_replacement_primes_family — rank 12, 11.0 ± 5.4 nats behind the best (2.0× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -3533.7

**Hypothesis:** Refinement of `without_replacement_size_principle`, adding one component from `threshold_concepts_primes_family`: people do the same Bayesian concept learning over number rules, bounded ranges (with an Erlang prior on width) and one-sided thresholds, with a prior mass per concept family, and they believe the program never repeats itself, so examples are drawn from the concept without replacement. The one change is that "prime numbers" is no longer lumped with squares, cubes and powers in a shared special-sets family but forms its own family with a separately inferred prior mass, so prime-like sets (e.g. 3, 7 or 53, 2) are read as "primes" — people endorse other primes and reject nearby composites — independently of how plausible squares or powers are, while the without-replacement coincidence still sharpens generalisation for sets that use up much of a small concept.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/043c9ecd8d452a7b/repo/_runs/cell_1/experiment3/model_loop/models/without_replacement_primes_family.py`

### graded_belief_criterion_decision — rank 13, 11.0 ± 4.5 nats behind the best (2.4× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -3533.8

**Hypothesis:** People infer the hidden program as an ideal Bayesian concept learner would (number rules, bounded ranges with a typical width, and one-sided thresholds, with the size principle and examples drawn without repetition), but they do not simply say "yes" with the probability they believe the target belongs. Instead they weigh how strongly they believe it belongs against a personal response criterion, with some response noise: the criterion sets a bias toward "yes" or "no", and the noise sets whether answers are more decisive than the belief (a soft threshold) or more hedged than it, so partly-believed targets are pushed toward or away from the extremes in the same way for every set.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/043c9ecd8d452a7b/repo/_runs/cell_1/experiment3/model_loop/models/graded_belief_criterion_decision.py`

### round_number_ranges_primes_family — rank 14, 11.9 ± 5.6 nats behind the best (2.1× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -3534.6

**Hypothesis:** Refinement of `round_number_range_boundaries`, adding one component from `without_replacement_primes_family`: people do the same Bayesian concept learning over number rules, bounded ranges (Erlang width prior, with a learned round-number bias on ranges whose boundaries fall on multiples of ten) and one-sided thresholds, with a prior mass per concept family and examples drawn without replacement. The one change is that "prime numbers" is no longer lumped with squares, cubes and powers in a shared special-sets family but forms its own family with a separately inferred prior mass, so prime-like sets (e.g. 3, 7 or 53, 2) are read as "primes" — people endorse other primes and reject nearby composites — independently of how plausible squares or powers are, while the round-number bias still shapes where magnitude-based "yes" answers stop.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/043c9ecd8d452a7b/repo/_runs/cell_1/experiment3/model_loop/models/round_number_ranges_primes_family.py`

### weak_sampling_hypothesis_elimination — rank 15, 596.5 ± 39.3 nats behind the best (15.2× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -4119.2

**Hypothesis:** People consider the same kinds of candidate programs as an ideal concept learner — number rules (parity, primes, squares, powers, multiples, final or contained digits, digit sums), each kind with its own prior plausibility, and numeric ranges of a typical width — but they treat each example only as a test the program passed, not as a sample it generated: programs that do not contain an example are eliminated, and every program consistent with the examples keeps its prior share however large it is, with no "suspicious coincidence" favouring tight concepts. They say "yes" in proportion to how much of the surviving, prior-weighted belief contains the target, so adding consistent examples does not narrow generalisation (e.g. after 16, 8, 64 "even" stays as credible as "powers of 2").

**Source:** `/scr/wurgaft/auto-psych/agent_trees/043c9ecd8d452a7b/repo/_runs/cell_1/experiment3/model_loop/models/weak_sampling_hypothesis_elimination.py`

### attention_weighted_set_prototype — rank 16, 995.9 ± 46.5 nats behind the best (21.4× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -4518.7

**Hypothesis:** People do not infer a hidden rule; they abstract a single prototype of the example set — its typical magnitude and spread on the number line, and the profile of familiar number properties (parity, multiples, final and contained digits, primes, squares, powers) the examples tend to have — and say "yes" to a target in proportion to its similarity to that prototype. Attention to a property grows with how consistently the examples share it, and rarer, more distinctive properties count more, so a target is rejected mainly for lacking properties that all the examples have in common or for lying far from the examples' typical magnitude relative to their spread.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/043c9ecd8d452a7b/repo/_runs/cell_1/experiment3/model_loop/models/attention_weighted_set_prototype.py`

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

### threshold_concepts_erlang_interval — pruned (experiment2 end of experiment)

**Margin:** 4.8 nats behind first_example_evidence_weight (2.1× dse)

**Hypothesis:** Refinement of `erlang_interval_size_family_prior`: people do the same Bayesian concept learning over rules and numeric intervals, with the size principle, a prior mass per concept family (exception rules in their own family) and an Erlang prior over the width of bounded ranges, but they treat one-sided magnitude thresholds ("numbers above 60", "numbers below 20") as a distinct kind of program rather than as just very wide intervals that happen to touch 1 or 100. The one change is that these threshold concepts form their own family with a separately inferred prior mass, shared equally among them and exempt from the width prior, so when all the examples are large (or all small) people extend "yes" all the way to the end of the number line, not just to a band of moderate width around the examples.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/043c9ecd8d452a7b/repo/_runs/cell_1/experiment2/model_loop/models/pruned/threshold_concepts_erlang_interval.py`

### range_size_principle_strength — pruned (experiment2 end of experiment)

**Margin:** 5.2 nats behind first_example_evidence_weight (2.2× dse)

**Hypothesis:** Refinement of `threshold_concepts_erlang_interval`: people do the same Bayesian concept learning over number rules, bounded numeric ranges (with an Erlang prior on width) and one-sided magnitude thresholds (their own family), with a prior mass per concept family, but they do not apply the "suspicious coincidence" of the size principle to magnitude concepts as strongly as to arithmetic and digit rules. The one change is that, for ranges and thresholds only, how much an example's probability shrinks with the concept's width is governed by a learned exponent (one recovers the incumbent, zero means ranges are treated as mere filters the examples pass), so how tightly people's magnitude-based generalisation hugs the examples, relative to rule-based generalisation, is inferred from the data rather than fixed by the ideal learner.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/043c9ecd8d452a7b/repo/_runs/cell_1/experiment2/model_loop/models/pruned/range_size_principle_strength.py`

### set_size_amplified_size_principle — pruned (experiment2 end of experiment)

**Margin:** 5.7 nats behind first_example_evidence_weight (2.3× dse)

**Hypothesis:** Refinement of `threshold_concepts_erlang_interval`: people do the same Bayesian concept learning over rules, bounded numeric ranges (with an Erlang prior on width) and one-sided thresholds, with the size principle and a prior mass per concept family, but the "suspicious coincidence" does not accumulate additively over examples. The one change is that each example's evidential weight scales with a learned power of how many examples were shown, so the more examples a set has, the more decisively each one favours tight concepts over broad ones (zero recovers the incumbent): generalisation narrows faster from one to four examples than an ideal learner predicts.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/043c9ecd8d452a7b/repo/_runs/cell_1/experiment2/model_loop/models/pruned/set_size_amplified_size_principle.py`

### round_number_ranges_digit_sums — pruned (experiment2 end of experiment)

**Margin:** 6.5 nats behind first_example_evidence_weight (2.4× dse)

**Hypothesis:** Refinement of `round_number_range_boundaries`, adding one component from `digit_sum_concepts_threshold_erlang`: people do the same Bayesian concept learning over number rules, bounded ranges (Erlang width prior, with a learned round-number bias on ranges whose boundaries fall on multiples of ten) and one-sided thresholds, with a prior mass per concept family and examples drawn without replacement. The one change is that their space of candidate programs also includes digit-sum concepts — "numbers whose digits add up to s" (e.g. 5, 14, 23, 32, 41, 50) — as a new family with its own inferred prior mass, shared equally among the 18 possible sums. Because these concepts are tiny, examples that share a digit sum are a suspicious coincidence, so people endorse other numbers with that sum and reject neighbours with a different sum more than the incumbent predicts, while the round-number bias still shapes where magnitude-based "yes" answers stop.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/043c9ecd8d452a7b/repo/_runs/cell_1/experiment2/model_loop/models/pruned/round_number_ranges_digit_sums.py`

### erlang_interval_size_family_prior — pruned (experiment2 end of experiment)

**Margin:** 6.6 nats behind first_example_evidence_weight (2.3× dse)

**Hypothesis:** Refinement of `exception_rule_family_prior_2`: people do the same Bayesian concept learning over rules and intervals, with the size principle and a separate prior mass per concept family (exception rules in their own family), but they do not regard all numeric intervals as equally plausible a priori. The one change is an Erlang-shaped prior over interval width with a learned characteristic size (as in Tenenbaum's number game): very narrow and very wide ranges are a priori unlikely, so after a single example or tightly clustered examples people generalise to a band of moderate width around them rather than collapsing onto the tightest interval, and after widely spread examples broad ranges are penalised relative to natural rules.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/043c9ecd8d452a7b/repo/_runs/cell_1/experiment2/model_loop/models/pruned/erlang_interval_size_family_prior.py`

### without_replacement_digit_sum_concepts — pruned (experiment2 end of experiment)

**Margin:** 6.7 nats behind first_example_evidence_weight (2.5× dse)

**Hypothesis:** Refinement of `without_replacement_size_principle`, adding one component from `digit_sum_concepts_threshold_erlang`: people do the same Bayesian concept learning over number rules, bounded ranges (with an Erlang prior on width) and one-sided thresholds, with a prior mass per concept family and examples believed to be drawn from the concept without replacement, but their space of candidate programs also includes digit-sum concepts — "numbers whose digits add up to s" (e.g. 5, 14, 23, 32, 41, 50) — as a new family with its own inferred prior mass, shared equally among the 18 possible sums. The one change is this digit-sum family: because these concepts are tiny, examples that happen to share a digit sum are an especially suspicious coincidence when drawn without replacement, so people endorse other numbers with that digit sum, and reject neighbours with a different sum, more than the incumbent predicts.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/043c9ecd8d452a7b/repo/_runs/cell_1/experiment2/model_loop/models/pruned/without_replacement_digit_sum_concepts.py`

### powers_own_family_erlang_interval — pruned (experiment2 end of experiment)

**Margin:** 7.2 nats behind first_example_evidence_weight (2.4× dse)

**Hypothesis:** Refinement of `erlang_interval_size_family_prior`: people do the same Bayesian concept learning over rules and numeric intervals, with the size principle, a prior mass per concept family (exception rules in their own family) and an Erlang prior over interval width, but they do not lump "powers of m" (2, 4, 8, 16, …; 3, 9, 27, 81; 8, 64; …) together with primes, squares and cubes. The one change is that repeated-multiplication rules form their own concept family with a separately inferred prior mass, so if people find powers an especially natural kind of program, examples like 8 and 64 are read as "powers of 8" (rejecting 2, 4, 16 and 32), and sets of powers of 2 crowd out "even" more strongly than the current model predicts.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/043c9ecd8d452a7b/repo/_runs/cell_1/experiment2/model_loop/models/pruned/powers_own_family_erlang_interval.py`

### weber_scaled_interval_size_prior — pruned (experiment2 end of experiment)

**Margin:** 7.4 nats behind first_example_evidence_weight (2.6× dse)

**Hypothesis:** Refinement of `erlang_interval_size_family_prior`: people do the same Bayesian concept learning over rules and numeric intervals, with the size principle, a prior mass per concept family (exception rules in their own family) and an Erlang prior over interval size, but they judge how big a range is relative to its magnitude (Weber's law) rather than in absolute units. The one change is that the Erlang prior is placed on the interval's width divided by a learned power of its location on the number line, so a range of ten numbers around 85 counts as "narrower" than one around 8; people therefore generalise to wider bands of nearby numbers after large examples than after small ones, with the strength of this magnitude scaling inferred from the data (zero recovers the incumbent).

**Source:** `/scr/wurgaft/auto-psych/agent_trees/043c9ecd8d452a7b/repo/_runs/cell_1/experiment2/model_loop/models/pruned/weber_scaled_interval_size_prior.py`

### primes_family_set_size_amplified — pruned (experiment2 end of experiment)

**Margin:** 7.6 nats behind first_example_evidence_weight (2.3× dse)

**Hypothesis:** Refinement of `set_size_amplified_size_principle`, adding one component from `threshold_concepts_primes_family`. People do the same Bayesian concept learning over number rules, bounded ranges (with an Erlang prior on width) and one-sided thresholds, with a prior mass per concept family and a size principle whose per-example weight grows with how many examples were shown. The one change is that "prime numbers" leaves the shared special-sets family (squares, cubes, powers) and becomes its own family with a separately inferred prior mass, so prime-like sets such as 3, 7 or 53, 2 are read as "primes" (endorsing other primes, rejecting nearby composites) independently of how plausible squares or powers are, and this reading tightens with more prime examples faster than an ideal learner's would.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/043c9ecd8d452a7b/repo/_runs/cell_1/experiment2/model_loop/models/pruned/primes_family_set_size_amplified.py`

### erlang_interval_size_exception_prior — pruned (experiment2 end of experiment)

**Margin:** 7.7 nats behind first_example_evidence_weight (2.6× dse)

**Hypothesis:** Refinement of `exception_rule_family_prior`: people do the same Bayesian concept learning over rules and intervals, with a prior mass per concept family (exception rules as their own family) and a learned strength of the size principle, but they do not consider every numeric interval equally likely a priori. The one change is an Erlang prior on interval size with a learned typical size, as in Tenenbaum's number game: people expect a magnitude-based program to cover a range of moderate width, so very narrow intervals hugging the examples and very broad ones are both discounted, which reshapes how far "yes" answers extend beyond the examples in magnitude.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/043c9ecd8d452a7b/repo/_runs/cell_1/experiment2/model_loop/models/pruned/erlang_interval_size_exception_prior.py`

### digit_sum_concepts_threshold_erlang — pruned (experiment2 end of experiment)

**Margin:** 7.8 nats behind first_example_evidence_weight (2.5× dse)

**Hypothesis:** Refinement of `threshold_concepts_erlang_interval`: people do the same Bayesian concept learning over number rules, bounded numeric ranges (with an Erlang prior on width) and one-sided thresholds, with the size principle and a prior mass per concept family. The one change is that their space of candidate programs also includes digit-sum concepts — "numbers whose digits add up to s" (e.g. 5, 14, 23, 32, 41, 50 for s = 5) — as a new family with its own inferred prior mass, shared equally among the 18 possible sums. When the examples happen to share a digit sum (especially a single example with a small, rare sum), people endorse other numbers with that digit sum more, and neighbours with a different digit sum less, than the incumbent predicts.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/043c9ecd8d452a7b/repo/_runs/cell_1/experiment2/model_loop/models/pruned/digit_sum_concepts_threshold_erlang.py`

### without_replacement_primes_digit_sums — pruned (experiment2 end of experiment)

**Margin:** 8.2 nats behind first_example_evidence_weight (2.3× dse)

**Hypothesis:** Refinement of `without_replacement_primes_family`, adding one component from `digit_sum_concepts_threshold_erlang`: people do the same Bayesian concept learning over number rules (with "prime numbers" as its own family), bounded ranges (with an Erlang prior on width) and one-sided thresholds, with a prior mass per concept family, and they believe the program never repeats itself, so examples are drawn from the concept without replacement. The one change is that their space of candidate programs also includes digit-sum concepts — "numbers whose digits add up to s" (e.g. 5, 14, 23, 32, 41, 50) — as a new family with its own inferred prior mass, shared equally among the 18 possible sums. When the examples happen to share a digit sum, people endorse other numbers with that sum and reject neighbours with a different sum more than the parent predicts, and this works independently of how readily they read prime-like sets as "primes".

**Source:** `/scr/wurgaft/auto-psych/agent_trees/043c9ecd8d452a7b/repo/_runs/cell_1/experiment2/model_loop/models/pruned/without_replacement_primes_digit_sums.py`

### set_size_amplified_digit_sum_concepts — pruned (experiment2 end of experiment)

**Margin:** 8.7 nats behind first_example_evidence_weight (2.6× dse)

**Hypothesis:** Refinement of `set_size_amplified_size_principle`, adding one component from `digit_sum_concepts_threshold_erlang`. People do the same Bayesian concept learning over number rules, bounded ranges (with an Erlang prior on width) and one-sided thresholds, with a prior mass per concept family and a size principle whose per-example weight grows with how many examples were shown. The one change is that their space of candidate programs also includes digit-sum concepts — "numbers whose digits add up to s" (e.g. 5, 14, 23, 32, 41, 50) — as a new family with its own inferred prior mass shared equally among the 18 sums, so when the examples share a digit sum people endorse other numbers with that sum (and reject neighbours with a different sum) more than the parent predicts, and this reading sharpens with more examples faster than an ideal learner's would.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/043c9ecd8d452a7b/repo/_runs/cell_1/experiment2/model_loop/models/pruned/set_size_amplified_digit_sum_concepts.py`

### primes_own_family_erlang_interval — pruned (experiment2 end of experiment)

**Margin:** 8.9 nats behind first_example_evidence_weight (2.5× dse)

**Hypothesis:** Refinement of `powers_own_family_erlang_interval`: people do the same Bayesian concept learning over rules and numeric intervals, with the size principle, a prior mass per concept family (exception rules and powers of m each in their own family) and an Erlang prior over interval width, but they do not treat "prime numbers" as the same kind of program as squares and cubes. The one change is that primes form their own concept family with a separately inferred prior mass, so how readily people read examples such as 3, 7 or 53, 2 as "primes" (endorsing 29 or 59 and rejecting nearby composites) is no longer tied to how readily they read 81, 25 as "squares".

**Source:** `/scr/wurgaft/auto-psych/agent_trees/043c9ecd8d452a7b/repo/_runs/cell_1/experiment2/model_loop/models/pruned/primes_own_family_erlang_interval.py`

### weber_noisy_interval_membership — pruned (experiment2 end of experiment)

**Margin:** 10.0 nats behind first_example_evidence_weight (3.0× dse)

**Hypothesis:** People do Bayesian concept learning over natural rules and numeric ranges as in the best model (size principle, a prior mass per concept family, an Erlang prior on range width), but they check whether a target falls inside a magnitude range using their approximate, noisy sense of number rather than exact arithmetic. The one distortion is this Weber-like perceptual noise, which grows in proportion to a number's size: symbolic rules (even, ends in 3, powers of 2) are checked exactly, but the edges of a range are blurred, so large targets just outside a range are often accepted and large targets just inside are sometimes rejected, while range boundaries among small numbers stay sharp.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/043c9ecd8d452a7b/repo/_runs/cell_1/experiment2/model_loop/models/pruned/weber_noisy_interval_membership.py`

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

### exception_rule_family_prior — pruned (experiment2 end of experiment)

**Margin:** 20.7 nats behind first_example_evidence_weight (3.2× dse)

**Hypothesis:** Refinement of `size_principle_strength_family_prior`: people do the same Bayesian concept learning over rules and intervals, with a prior weight per concept family and a learned strength of the size principle, but they do not treat ad-hoc "exception" rules (powers of 2 except 32, powers of 2 plus 37) as being as natural as clean mathematical sets. The one change is to give these exception rules their own concept family with its own inferred prior mass, so people can find "powers of 2, or 37" implausible and reject the 37 lure for sets of powers of 2 while still generalising to the other powers of 2.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/043c9ecd8d452a7b/repo/_runs/cell_1/experiment2/model_loop/models/pruned/exception_rule_family_prior.py`

### power_base_family_prior — pruned (experiment2 end of experiment)

**Margin:** 21.0 nats behind first_example_evidence_weight (3.2× dse)

**Hypothesis:** Refinement of `exception_rule_family_prior_2`: people do the same Bayesian concept learning over rules and intervals with the size principle, a separate prior mass per concept family and the exception rules in their own family, but they do not lump "powers of a base" (powers of 2, 3, ... 10) together with primes, squares and cubes as one "special sets" family. The one change is to give powers-of-a-base rules their own family with its own inferred prior mass: people readily think of a set as successive powers (or repeated multiplication) of a number, while primes, squares and cubes each have a separate, independently inferred plausibility, so generalisation after power-like sets (e.g. 8, 64) and after square- or prime-like sets (e.g. 81, 9 or 3, 31) can differ in strength instead of being tied to a shared per-rule prior.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/043c9ecd8d452a7b/repo/_runs/cell_1/experiment2/model_loop/models/pruned/power_base_family_prior.py`

### exception_rule_family_prior_2 — pruned (experiment2 end of experiment)

**Margin:** 21.3 nats behind first_example_evidence_weight (3.2× dse)

**Hypothesis:** Refinement of `family_weighted_concept_prior`: people do the same Bayesian concept learning over rules and intervals with the size principle and a separate prior mass per concept family, but the two gerrymandered exception rules ("powers of 2 except 32" and "powers of 2 plus 37") are not natural mathematical sets — people give them their own, separately inferred (and expectedly tiny) prior mass rather than an equal share of the "special sets" family alongside primes, squares and powers. The one change is this separate family for exception rules, so for sets of powers of 2 people should reject the lure 37 nearly always while still endorsing the other powers of 2.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/043c9ecd8d452a7b/repo/_runs/cell_1/experiment2/model_loop/models/pruned/exception_rule_family_prior_2.py`

### parity_bounded_range_concepts — pruned (experiment2 end of experiment)

**Margin:** 21.8 nats behind first_example_evidence_weight (3.2× dse)

**Hypothesis:** Refinement of `exception_rule_family_prior_2`: people do the same Bayesian concept learning over rules and numeric intervals, with the size principle and a prior mass per concept family (the exception rules in their own family), but their hypothesis space also contains conjunctive "parity within a range" concepts — "even numbers between 20 and 60", "odd numbers from 9 to 69" — as a new family with its own inferred prior mass. The one change is this composed rule-plus-magnitude family: when examples are all even (or all odd) and clustered in magnitude, people generalise to same-parity numbers near the examples while rejecting both nearby numbers of the other parity and same-parity numbers far outside the examples' range, which neither a pure parity rule nor a pure interval predicts.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/043c9ecd8d452a7b/repo/_runs/cell_1/experiment2/model_loop/models/pruned/parity_bounded_range_concepts.py`

### family_prior_confidence_threshold — pruned (experiment1 end of experiment)

**Margin:** 31.6 nats behind powers_own_family_erlang_interval (4.5× dse)

**Hypothesis:** Refinement of `size_principle_strength_family_prior`: people do the same Bayesian concept learning over rules and intervals, with a prior weight per concept family and a learned strength of the size principle, but they do not answer "yes" with the probability that the concept contains the target; instead they apply a noisy confidence criterion, saying "yes" only when their belief that the target belongs clears a learned threshold (the decision stage taken from `membership_confidence_threshold`). The one change is this thresholded decision rule: targets backed by only a minority of the posterior — such as 37 after powers of 2, which only the ad-hoc "powers of 2, or 37" rule contains — fall below the criterion and are rejected far more often than probability matching predicts, while confidently believed members are endorsed nearly always.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/043c9ecd8d452a7b/repo/_runs/cell_1/experiment1/model_loop/models/pruned/family_prior_confidence_threshold.py`

### recognition_shortcut_for_examples — pruned (experiment2 end of experiment)

**Margin:** 35.8 nats behind first_example_evidence_weight (3.1× dse)

**Hypothesis:** People infer the hidden program as in the best current model (Bayesian concept learning over number rules, bounded ranges and one-sided thresholds) and answer most targets by probability matching on that belief, but when the target is one of the numbers they were just shown they do not consult their inferred concept at all: they answer from recognition, saying "yes" at a single fixed rate that does not depend on the set. So the yes rate for a repeated example is the same whether one example or four were shown and whatever concept they suggest, unlike the best model, where it falls as tight concepts that exclude random-looking examples gain belief.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/043c9ecd8d452a7b/repo/_runs/cell_1/experiment2/model_loop/models/pruned/recognition_shortcut_for_examples.py`

### limited_family_consideration_set — pruned (experiment2 end of experiment)

**Margin:** 57.8 nats behind first_example_evidence_weight (5.4× dse)

**Hypothesis:** People do not search the whole space of possible programs on every trial: each kind of program (parity, special sets such as primes, squares and powers, multiples, final digit, contained digit, exception rules, bounded ranges, one-sided thresholds) comes to mind only with its own probability, and people do Bayesian concept learning with the size principle only over the kinds that came to mind, giving each considered kind an equal share of their prior. When the kind of program that best fits the examples is not thought of, they fall back on whatever kind they did consider, so "yes" is the average, over which kinds came to mind, of the probability that the inferred program produces the target. How available each kind is to recall replaces a fixed prior plausibility per kind.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/043c9ecd8d452a7b/repo/_runs/cell_1/experiment2/model_loop/models/pruned/limited_family_consideration_set.py`

### gaussian_prototype_magnitude_concepts — pruned (experiment2 end of experiment)

**Margin:** 97.3 nats behind first_example_evidence_weight (6.6× dse)

**Hypothesis:** People do Bayesian concept learning over number rules and one-sided thresholds as in the best model, but they do not represent a magnitude-based program as a sharp-edged range; they represent it as a graded prototype — a typical value on the number line with a spread — that produces numbers mostly near its centre and fewer the further away they lie. A target then belongs to such a concept to the degree it is close to the prototype's centre, so magnitude generalisation falls off smoothly with distance from the examples instead of dropping at a boundary.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/043c9ecd8d452a7b/repo/_runs/cell_1/experiment2/model_loop/models/pruned/gaussian_prototype_magnitude_concepts.py`

### pedagogical_teacher_sampling — pruned (experiment2 end of experiment)

**Margin:** 107.0 nats behind first_example_evidence_weight (6.1× dse)

**Hypothesis:** People assume the program's examples were chosen to be helpful, as a teacher would choose them, rather than drawn at random from its concept (rational pedagogy): a teacher holding a concept picks examples in proportion to how strongly each one, on its own, would point a naive learner (one with no preference among kinds of concept) to that concept. People judge each candidate rule, range or threshold by how likely such a teacher would be to have shown exactly these examples, so examples that would point equally well to many other concepts count as weak evidence and a concept is favoured when the examples are the ones most diagnostic of it; they then say "yes" with the probability that the inferred concept contains the target.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/043c9ecd8d452a7b/repo/_runs/cell_1/experiment2/model_loop/models/pruned/pedagogical_teacher_sampling.py`

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

### log_number_line_ranges — pruned (experiment2 end of experiment)

**Margin:** 308.7 nats behind first_example_evidence_weight (8.9× dse)

**Hypothesis:** People do Bayesian concept learning over number rules (with a prior mass per concept family, the exception rules in their own family) and numeric ranges with the size principle, but they represent magnitude on a compressed, logarithmic mental number line: a range is a band on that line, so ranges are a priori equally likely wherever they sit on the log line and their plausibility depends on their ratio width (e.g. 5-10 is as natural as 50-100), not their width in units. As a result, magnitude-based generalisation scales with the examples' size — after small examples people generalise only to nearby small numbers, after large examples to a proportionally much wider band.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/043c9ecd8d452a7b/repo/_runs/cell_1/experiment2/model_loop/models/pruned/log_number_line_ranges.py`

### newest_example_size_principle_wsls — pruned (experiment2 end of experiment)

**Margin:** 313.1 nats behind first_example_evidence_weight (11.7× dse)

**Hypothesis:** People hold a single running guess about the program (a rule or a numeric range, weighed by family plausibility and an Erlang prior on range width) and revise it example by example in the order shown: they keep their current guess with a probability that grows with how well it explains the newest example, and otherwise draw a fresh guess — but when they redraw they judge the "suspicious coincidence" (the size principle) only for the newest example, merely requiring the guess to cover the earlier ones. So evidence for tight concepts does not compound across examples as in an ideal Bayesian learner, and which example drives specificity depends on the order shown (the last examples, plus whatever guess has survived from the first).

**Source:** `/scr/wurgaft/auto-psych/agent_trees/043c9ecd8d452a7b/repo/_runs/cell_1/experiment2/model_loop/models/pruned/newest_example_size_principle_wsls.py`

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

### conjunctive_category_similarity — pruned (experiment2 end of experiment)

**Margin:** 734.8 nats behind first_example_evidence_weight (17.9× dse)

**Hypothesis:** People do not infer a program; they judge a target by its similarity to the examples, where two numbers are similar to the degree that they fall together in familiar number categories (parity, multiples, final or contained digit, primes, squares, powers, magnitude ranges and thresholds), small distinctive shared categories counting more than broad ones. The similarities are combined conjunctively, as a geometric mean over the examples, so a target must resemble every example: being very dissimilar to any single example drags "yes" down sharply, unlike an average in which closeness to one example is enough. The willingness to say "yes" rises smoothly (logistically) with this geometric-mean similarity on a log scale.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/043c9ecd8d452a7b/repo/_runs/cell_1/experiment2/model_loop/models/pruned/conjunctive_category_similarity.py`

### shared_category_exemplar_averaging — pruned (experiment2 end of experiment)

**Margin:** 774.5 nats behind first_example_evidence_weight (18.4× dse)

**Hypothesis:** People do not infer one program that explains all the examples together; they store each example as a separate exemplar and judge the target by its average similarity to the stored examples, where two numbers are similar to the degree that they fall together in the familiar number categories people know (parity, multiples, final or contained digits, primes, squares, powers, magnitude ranges and thresholds), with small, distinctive shared categories counting more than broad ones (Shepard's additive-clustering view of number similarity). Because each example generalises on its own and the similarities are averaged rather than multiplied, the "suspicious coincidence" of several examples sharing a tight category never accumulates: a target similar to only one example of a set gets partial "yes" support, and adding examples does not sharpen generalisation the way it does for a Bayesian concept learner.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/043c9ecd8d452a7b/repo/_runs/cell_1/experiment2/model_loop/models/pruned/shared_category_exemplar_averaging.py`

### average_digit_magnitude_similarity — pruned (experiment2 end of experiment)

**Margin:** 793.1 nats behind first_example_evidence_weight (19.5× dse)

**Hypothesis:** People do not infer a rule; they judge a target by its average similarity to the examples, where two numbers are similar if they are close in magnitude and share their final digit, with similarity decaying exponentially along the number line and dropping by a fixed factor when the final digits differ. Because the similarity is averaged rather than summed over examples, "yes" depends on the proportion of the examples the target resembles, not on how many examples were shown, so adding dissimilar examples to a set dilutes generalisation to a target near one of them.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/043c9ecd8d452a7b/repo/_runs/cell_1/experiment2/model_loop/models/pruned/average_digit_magnitude_similarity.py`

### simplicity_description_length_prior — pruned (experiment2 end of experiment)

**Margin:** 910.8 nats behind first_example_evidence_weight (13.5× dse)

**Hypothesis:** People infer the hidden program as a Bayesian concept learner would — weighing number rules, bounded ranges and one-sided thresholds by how well they explain the examples, with the size principle and examples drawn without repetition — but their sense of how plausible a program is a priori comes from a single simplicity bias rather than learned preferences for particular kinds of concept: every program is penalised by the same amount for each symbol it takes to state it ("odd" is one, "multiples of 7" two, "between 23 and 41" three, "powers of 2 except 32" four). So how readily people read examples as a rule versus a range versus a threshold is fixed by description length alone, with one inferred cost per symbol.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/043c9ecd8d452a7b/repo/_runs/cell_1/experiment2/model_loop/models/pruned/simplicity_description_length_prior.py`
