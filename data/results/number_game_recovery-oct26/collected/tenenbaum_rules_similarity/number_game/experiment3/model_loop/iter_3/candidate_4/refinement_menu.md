# Refinement menu

The models you may refine, other than the incumbent `sequence_opens_bottom_ends_top`. Each entry is one mechanism: its hypothesis in full as its author stated it, how it stands, and where its source is. This is a menu, not a list of what is ruled out: a pruned model lost to the best model on the data it was scored on, by the stated margin, but its mechanism may be partly right — and a model that lost narrowly is the most promising target here. Read the source of the model you pick before you write anything.

## Live models other than the incumbent (in the set; best first)

### compressed_rule_extension — rank 1, 0.1 ± 2.5 nats behind the best (0.1× dse: statistically tied with the best), ELPD-LOO -3098.2

**Hypothesis:** People think of an arithmetic property such as "even numbers" or "multiples of 5" by its description rather than by listing its members, so they underestimate how many numbers it contains. When they weigh how well a concept explains the examples (the size principle), they treat a rule as smaller than it really is by an inferred degree, but they judge a numeric range at its true size, because its extent is visible on the number line. Otherwise they reason as in Bayesian concept learning over rules, ranges and range-restricted rules, with the last-shown largest example marking a range's top. So broad properties like parity keep more support against tight ranges, and people generalize by the shared property beyond the examples' span (yes to same-parity numbers far away, no to opposite-parity neighbours just outside) more than an ideal learner would.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment3/model_loop/models/compressed_rule_extension.py`

### parity_representativeness_bias — rank 2, 0.3 ± 2.7 nats behind the best (0.1× dse: statistically tied with the best), ELPD-LOO -3098.4

**Hypothesis:** People do the same Bayesian concept learning as the best model (arithmetic rules, ranges of 1-100 and range-restricted rules, with the size principle and the last-shown largest example marking a range's top), but they judge how likely the examples are under a concept partly by how representative they look: a concept that contains both odd and even numbers (a range, multiples of 3, squares) is expected to produce a parity-mixed sample, so a set whose examples all share a parity seems less likely under it than chance implies, with each further same-parity example costing an inferred extra penalty. Parity-pure concepts (even numbers, odd numbers, multiples of 4) pay nothing, so after examples like 22,46,68 people say "yes" to even numbers outside the examples' span (e.g. 8 or 90) more, and to odd numbers inside it less, than the ideal learner does.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment3/model_loop/models/parity_representativeness_bias.py`

### last_example_marks_range_top — rank 3, 1.1 ± 1.6 nats behind the best (0.7× dse: statistically tied with the best), ELPD-LOO -3099.1

**Hypothesis:** Refinement of `top_marking_range_sampling`: people do the same Bayesian concept learning over arithmetic rules (primes excluding 1), every range of 1-100 and range-restricted rules, and believe that, with some inferred odds, the program deliberately shows its range's top number and draws the other examples at random — but they believe the program ends its demonstration on that top number. The one change: the marked top is read from the order of presentation, so a range ending at the largest example gets the generative top-marking credit (larger for wider ranges) only when the largest example is the last one shown, and then at full strength whatever the set size, while sets whose largest example comes earlier get no edge credit — after 20,35,60 people say "yes" to 45 but not 62, whereas after 60,35,20 they extend above 60 as readily as below 20.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment3/model_loop/models/last_example_marks_range_top.py`

### parity_kind_last_shown_top_marking — rank 4, 1.3 ± 1.9 nats behind the best (0.7× dse: statistically tied with the best), ELPD-LOO -3099.3

**Hypothesis:** Refinement of `last_example_marks_range_top`: people do the same Bayesian concept learning over arithmetic rules (primes excluding 1), every range of 1-100 and range-restricted rules, with the program believed to end its demonstration on its range's top number. The one change: "even numbers" and "odd numbers" are a kind of rule in their own right, with their own inferred prior share, rather than sharing one "special sets" share equally with primes, squares and cubes. So when all the examples share a parity (e.g. 22,36,48), parity stays a live explanation next to the ranges, and people say "yes" to same-parity numbers outside the examples' span (e.g. 80) more than to opposite-parity ones (81), which addresses the critique that the incumbent under-uses parity beyond the range.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment3/model_loop/models/parity_kind_last_shown_top_marking.py`

### examples_bracket_range_ends — rank 5, 1.4 ± 3.1 nats behind the best (0.5× dse: statistically tied with the best), ELPD-LOO -3099.5

**Hypothesis:** People believe the program often demonstrates a range by showing both of its ends: with some inferred odds, the smallest and the largest example are the range's bottom and top numbers (whatever order they are shown in) and any other examples are random draws. Otherwise they reason as in the best model's Bayesian concept learning (arithmetic rules perceived as smaller than they are, ranges of 1-100 and range-restricted rules, a last-shown largest example marking a range's top). So the range spanning exactly the examples is favoured far more than chance sampling implies, most of all for widely separated pairs. After 22,46 people say "yes" to odd numbers between the examples (e.g. 33) more, and "no" to neighbours just outside (21 or 47) more, than the best model predicts.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment3/model_loop/models/examples_bracket_range_ends.py`

### multi_example_sequence_top_marking — rank 6, 1.5 ± 1.7 nats behind the best (0.8× dse: statistically tied with the best), ELPD-LOO -3099.5

**Hypothesis:** Refinement of `last_example_marks_range_top`: people do the same Bayesian concept learning over arithmetic rules (primes excluding 1), every range of 1-100 and range-restricted rules, and believe that, with inferred odds, the program deliberately ends its demonstration on its range's top number. The one change: people read this marking from the sequence only when there is a sequence. When several examples are shown and the largest comes last, they credit ranges ending there, as the incumbent does. A lone example is not taken to mark anything, so it is treated as an ordinary random draw. So after a single example (e.g. 60) "yes" spreads about as far above it as below it (62 as readily as 58). After 20,35,60, generalization still stops at 60 while spreading below 20.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment3/model_loop/models/multi_example_sequence_top_marking.py`

### parity_restricted_range_concepts — rank 7, 1.6 ± 2.0 nats behind the best (0.8× dse: statistically tied with the best), ELPD-LOO -3099.6

**Hypothesis:** Refinement of `last_example_marks_range_top`. People do the same Bayesian concept learning over arithmetic rules (primes excluding 1), ranges of 1-100 and range-restricted rules, and they read a last-shown largest example as marking the top of the range. The one change: "even (or odd) numbers within a range" is a family of concepts in its own right, with its own inferred prior plausibility. It is not a minor range-restricted rule whose plausibility is diluted by the weight of the parity rule among all rules. So when every example shares a parity (e.g. 66,78 or 83,77), people readily read the set as "even/odd numbers around here" and say "no" to opposite-parity numbers near and beyond the examples. Other range-restricted rules ("multiples of 7 in the 20s") keep a separate share.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment3/model_loop/models/parity_restricted_range_concepts.py`

### opening_example_marks_range_edge — rank 8, 1.9 ± 2.9 nats behind the best (0.7× dse: statistically tied with the best), ELPD-LOO -3099.9

**Hypothesis:** Refinement of `compressed_rule_extension`: people do the same Bayesian concept learning over arithmetic rules (perceived as smaller than they are), ranges of 1-100 and range-restricted rules, with a last-shown largest example marking a range's top. The one change is that people also read the opening of a sequence. When several examples are shown and the first one is the smallest, or the largest, they believe (with inferred odds) that the program deliberately started on that edge of its range and drew the rest at random. Ranges with that end at the first example then gain credit, more for wider ranges. So after 20,35,60 people say "no" to 17 more readily, and after 60,35,20 to 63, than after the same numbers in an order that does not open on an extreme.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment3/model_loop/models/opening_example_marks_range_edge.py`

### last_shown_top_discounted_ranges — rank 9, 2.1 ± 2.4 nats behind the best (0.9× dse: statistically tied with the best), ELPD-LOO -3100.1

**Hypothesis:** Refinement of `upper_edge_discounted_range_size`: people do the same Bayesian concept learning over arithmetic rules (primes excluding 1), every range of 1-100 and range-restricted rules, applying the size principle fully to rules but only partly to ranges (judging a range as if it were wider than it is). The one change, taken from `last_example_marks_range_top`, is how they read the top edge. They do not give a fixed bonus to any range ending at the largest example. Instead, they believe the program may deliberately end its demonstration on its range's top number, so a range ending at the largest example gets generative credit (larger for wider ranges) only when that example is shown last. Because ranges are only weakly penalised for size, "yes" still extends a little above a last-shown maximum (addressing the critique that the order-based model over-sharpens that edge). Sets whose largest example comes earlier get no edge credit and generalize upward freely.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment3/model_loop/models/last_shown_top_discounted_ranges.py`

### uncertain_sampling_sequence_top — rank 10, 2.3 ± 2.4 nats behind the best (1.0× dse: statistically tied with the best), ELPD-LOO -3100.3

**Hypothesis:** Refinement of `uncertain_sampling_top_marking`: people do Bayesian concept learning over arithmetic rules (primes excluding 1) and ranges of 1-100, unsure whether the examples were drawn from the program's concept (strong sampling) or are merely numbers it accepts (weak sampling), so broad concepts such as "even numbers" keep a floor of support beyond the examples' span (the critique's parity gap), and they believe the program, with inferred odds, deliberately shows its range's top number. The one change, taken from `multi_example_sequence_top_marking`: the marked top is read from the order of presentation, so only a sequence of several examples that ends on its largest example blocks generalization above it (after 20,35,60 "yes" to 45 but not 62), while sets whose largest example comes earlier, and single examples, get no edge credit and generalize upward as freely as downward.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment3/model_loop/models/uncertain_sampling_sequence_top.py`

### limited_capacity_example_memory — rank 11, 2.6 ± 2.7 nats behind the best (1.0× dse: statistically tied with the best), ELPD-LOO -3100.6

**Hypothesis:** People can hold only a limited number of examples in working memory while judging: when a set has more examples than their capacity (which varies from trial to trial, between two and four, with inferred probabilities), they keep the last example shown plus a random selection of the earlier ones and forget the rest. They then reason exactly as in the best model's Bayesian concept learning (arithmetic rules perceived as smaller than they are, ranges of 1-100 and range-restricted rules, a last-shown largest example marking a range's top) from only the examples they still hold, so sets of three or four examples tighten generalization less than an ideal learner predicts and a forgotten example is itself sometimes rejected, while one- and two-example sets are judged in full.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment3/model_loop/models/limited_capacity_example_memory.py`

### digit_sum_concepts — rank 12, 2.6 ± 2.1 nats behind the best (1.3× dse: statistically tied with the best), ELPD-LOO -3100.7

**Hypothesis:** People do the same Bayesian concept learning as the best model (arithmetic rules, ranges of 1-100 and range-restricted rules, with the size principle and the last-shown largest example marking a range's top), but their space of candidate concepts also includes digit-sum concepts — "numbers whose digits add up to the same total" (e.g. 18, 27, 36, 45 … 90 for a total of nine; 15, 24, 33, 42, 51, 60 for six) — as a kind of rule of its own with an inferred prior share. So examples that share a digit sum (36,45,54,81 or 15,51) lead people to say "yes" to other numbers with that digit sum (90 or 27; 24 or 60) but not to rule-mates that break it (99 after 36,45,54,81).

**Source:** `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment3/model_loop/models/digit_sum_concepts.py`

### span_anchored_width_expectation — rank 13, 3.1 ± 3.2 nats behind the best (1.0× dse: statistically tied with the best), ELPD-LOO -3101.2

**Hypothesis:** People do Bayesian concept learning as in the best model (arithmetic rules with primes excluding 1 perceived as smaller than they are, every range of 1-100 and range-restricted rules, the size principle, the last-shown largest example marking a range's top), but their expectation of how wide the program's range is gets anchored on how widely the examples are spread, so the spread is counted twice: once as evidence and again in the prior. The one distortion: the typical range width people expect grows (or shrinks) with the examples' span by an inferred power, so widely spread sets lead people to expect, and generalize to, ranges reaching further past the outermost examples than the ideal learner would, and tightly clustered or single examples to ranges hugging them.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment3/model_loop/models/span_anchored_width_expectation.py`

### top_marking_range_sampling — rank 14, 3.3 ± 2.7 nats behind the best (1.2× dse: statistically tied with the best), ELPD-LOO -3101.3

**Hypothesis:** Refinement of `upper_edge_marking_ranges`: people do the same Bayesian concept learning over arithmetic rules (primes excluding 1), every range of 1-100 and range-restricted rules, and believe the program shows where its range stops at the top — but they reason about this generatively rather than granting a fixed bonus. The one change: people assume that, with some inferred odds, the program deliberately shows its range's top number and draws the other examples at random, so the largest example is explained by marking instead of by chance; a range ending at the largest example is therefore credited in proportion to how unlikely a random draw would have been to hit that number — far more for wide ranges than narrow ones, and less as more examples are shown — so single examples and sparse sets license especially strong downward generalization (after 60, "yes" to 40 but not 62), while dense sets fall back toward ordinary size-principle ranges.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment3/model_loop/models/top_marking_range_sampling.py`

### last_example_marks_range_edge — rank 15, 3.3 ± 2.3 nats behind the best (1.5× dse: statistically tied with the best), ELPD-LOO -3101.4

**Hypothesis:** Refinement of `last_example_marks_range_top`: people do the same Bayesian concept learning over arithmetic rules (primes excluding 1), every range of 1-100 and range-restricted rules, and believe that, with some inferred odds, the program ends its demonstration on a boundary of its range and draws the other examples at random. The one change: the last example marks whichever boundary it is, not only the top — a sequence ending on its largest example marks the range's top (as in the incumbent), while a sequence ending on its smallest example marks the range's bottom with the same generative credit (larger for wider ranges), so after 60,35,20 people say "yes" to 45 but less to 15, the mirror image of 20,35,60, and only sets ending on a middle example get no edge credit.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment3/model_loop/models/last_example_marks_range_edge.py`

### breadth_adaptive_yes_criterion — rank 16, 3.4 ± 3.4 nats behind the best (1.0× dse: statistically tied with the best), ELPD-LOO -3101.4

**Hypothesis:** People infer the program's concept by plain Bayesian concept learning over arithmetic rules (primes excluding 1) and numeric ranges, but the criterion they use to say "yes" is not fixed: it adapts to how many of the numbers 1-100 they believe the program produces. When the examples suggest a broad concept (e.g. "even numbers", a wide range) they shift how sure they must be that a target belongs before answering "yes" by an inferred amount, relative to sets that suggest a narrow concept (e.g. powers of 4), so the same membership belief leads to different answers depending on the expected breadth of the concept.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment3/model_loop/models/breadth_adaptive_yes_criterion.py`

### upper_edge_marking_ranges — rank 17, 3.4 ± 2.9 nats behind the best (1.2× dse: statistically tied with the best), ELPD-LOO -3101.4

**Hypothesis:** Refinement of `range_restricted_rule_concepts` (with its prime concept corrected to exclude 1, as in the incumbent): people do the same Bayesian concept learning over arithmetic rules, every range of 1-100 and range-restricted rules, but they assume the program's examples mark where its range stops at the top — the largest example shown is taken as evidence that the range ends there. The one change, grafted in one-sided form from `endpoint_marking_interval_sampling`: a range whose upper end coincides with the largest example gets an inferred evidence bonus, while its lower end is unconstrained, so people generalize downward from the examples more readily than upward (after 16 they accept 15 or 13 more than 17 or 19; after 83,77 they reject 84).

**Source:** `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment3/model_loop/models/upper_edge_marking_ranges.py`

### prime_concept_excludes_one — rank 18, 3.6 ± 3.0 nats behind the best (1.2× dse: statistically tied with the best), ELPD-LOO -3101.6

**Hypothesis:** Refinement of `range_restricted_rule_concepts`: people do the same Bayesian concept learning over arithmetic rules, every range of 1-100 and range-restricted rules (kind-level rule prior, Erlang width prior with a decade-boundary preference, size principle with 10% random examples, "yes" = posterior probability the concept contains the target), but their concept of "prime numbers" is the mathematical one, which excludes 1. The one change is removing 1 from the prime concept: when 1 is among the examples (e.g. 31,3,1) people cannot explain the set as primes and fall back on "odd numbers" (yes to 9 or 25, no special yes to 2), and after a set of primes (e.g. 67,31,17,61) they answer "no" to 1 rather than accepting it as a prime.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment3/model_loop/models/prime_concept_excludes_one.py`

### spread_sampled_ranges — rank 19, 4.1 ± 2.5 nats behind the best (1.6× dse: statistically tied with the best), ELPD-LOO -3102.2

**Hypothesis:** People learn the program's concept by Bayesian inference over arithmetic rules and numeric ranges, as in the best model, but with an inferred probability they believe a program that produces a range spreads its examples evenly across it (one example from each equal stretch of the range) instead of scattering them at random. A set of several evenly spaced examples is then far likelier under the range they span than under a rule or a random draw, while a range reaching well past the outermost examples, or a set with clumped examples, gets no such boost. The sharpest disagreement with the best model is on widely and evenly spaced sets of three or four examples (e.g. 20,35,68,92): this model says "yes" more readily to any number inside their span, whatever its parity or divisibility, and its range-based "yes" stops close to the outermost examples.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment3/model_loop/models/spread_sampled_ranges.py`

### upper_edge_discounted_range_size — rank 20, 4.2 ± 3.4 nats behind the best (1.2× dse: statistically tied with the best), ELPD-LOO -3102.2

**Hypothesis:** Refinement of `upper_edge_marking_ranges`: people do the same Bayesian concept learning over arithmetic rules (primes excluding 1), every range of 1-100 and range-restricted rules, and they take the largest example as marking where the program's range stops at the top. The one change, grafted from `range_discounted_size_principle`: people apply the size principle at full strength only to arithmetic rules and only partly to numeric ranges, judging a range's fit to the examples as if it were wider than it is (by an inferred degree), so it is the marked top edge — not the size principle — that stops range generalization going upward, while downward from the examples "yes" spreads further across their neighbourhood than strong sampling predicts.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment3/model_loop/models/upper_edge_discounted_range_size.py`

### uncertain_sampling_assumption — rank 21, 4.5 ± 3.3 nats behind the best (1.4× dse: statistically tied with the best), ELPD-LOO -3102.6

**Hypothesis:** People do Bayesian concept learning over arithmetic rules (primes excluding 1) and numeric ranges of 1-100 (moderate, decade-bounded ranges preferred, the largest example marking a range's top), but they are unsure how the examples were chosen: with an inferred probability they assume the examples were drawn from the program's concept (strong sampling, so a small concept that fits is strongly favoured), and otherwise that the examples are merely numbers the program happens to accept (weak sampling, so every concept containing them stays as plausible as it was a priori). Their "yes" averages over both readings, so broad concepts such as "even numbers" or wide ranges keep a floor of support however many examples fit a narrower one, and generalization tightens with more examples less than an ideal strong-sampling learner predicts — for rules and ranges alike.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment3/model_loop/models/uncertain_sampling_assumption.py`

### uncertain_sampling_top_marking — rank 22, 4.9 ± 3.3 nats behind the best (1.5× dse: statistically tied with the best), ELPD-LOO -3102.9

**Hypothesis:** Refinement of `uncertain_sampling_assumption`. People still do Bayesian concept learning over arithmetic rules (primes excluding 1) and ranges of 1-100, and they are still unsure whether the examples were drawn from the program's concept (strong sampling) or are just numbers it happens to accept (weak sampling). The one change, taken from `top_marking_range_sampling`, is how the largest example marks the top of a range: instead of a fixed bonus, people think that with some inferred odds the program deliberately showed the top of its range and drew the other examples at random. A range ending at the largest example is then credited by how unlikely a chance draw would have been to hit that number under each sampling reading. Under strong sampling this credit is large for wide ranges and shrinks as more examples are shown; under weak sampling it is flat. So downward generalization from sparse sets is strong, and upward generalization is blocked, under both readings.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment3/model_loop/models/uncertain_sampling_top_marking.py`

### top_marking_uncertain_sampling — rank 23, 5.1 ± 2.9 nats behind the best (1.7× dse: statistically tied with the best), ELPD-LOO -3103.1

**Hypothesis:** Refinement of `top_marking_range_sampling`: people do the same Bayesian concept learning over arithmetic rules (primes excluding 1), every range of 1-100 and range-restricted rules, and believe that with inferred odds the program deliberately shows its range's top number and draws the other examples at random — but, the one change grafted from `uncertain_sampling_assumption`, they are unsure whether the remaining examples were drawn from the program's concept (strong sampling, size principle) or are merely numbers it happens to accept (weak sampling), and average over both readings with an inferred probability. So the marked top still blocks upward generalization, while broad concepts ("even numbers", wide ranges ending at the largest example) keep a floor of support however many examples fit a narrower one, and "yes" tightens with more examples less than the incumbent predicts.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment3/model_loop/models/top_marking_uncertain_sampling.py`

### information_weighted_prototype_matching — rank 24, 1027.1 ± 49.4 nats behind the best (20.8× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -4125.1

**Hypothesis:** People do not infer a rule; they abstract a single prototype from the examples — the examples' average magnitude and how far they spread, plus the proportion of examples that have each familiar number property (even, multiple of 3, 4, 5 or 10, prime, square, cube, power of two, final digit) — and say "yes" to a number in proportion to how closely it matches that prototype, where distance in magnitude is judged relative to the examples' spread and missing a property most examples share costs more the rarer that property is among numbers. Because the comparison is to the central tendency rather than to the stored examples, an example that is atypical of the rest (such as 100 after 31,19,49) is itself often rejected, and generalization spreads symmetrically around the examples' centre.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment3/model_loop/models/information_weighted_prototype_matching.py`

## Pruned models (out of the set; narrowest margin first)

### top_example_teacher_ranges — pruned (experiment2 end of experiment)

**Margin:** 2.6 nats behind last_example_marks_range_top; retired to keep the live set at 8 models: ELPD-LOO rank 10 of 19

**Hypothesis:** Refinement of `upper_edge_marking_ranges`: people do the same Bayesian concept learning over arithmetic rules (primes excluding 1), every range of 1-100 and range-restricted rules, and they take the largest example as marking the top of the program's range — but the one change is the form of that marking: rather than a constant evidence bonus for any range ending at the largest example, people believe that, with an inferred probability, the program deliberately showed the top of its range and drew the other examples at random. Under this teacher story the largest example pays no size-principle penalty for a range that ends at it, so the advantage of a top-marked range grows with its width: "yes" spreads broadly downward from the largest example (most strongly after a single example, e.g. after 60 people accept 35 or 45 but reject 63), while generalization above it stays blocked.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment2/model_loop/models/pruned/top_example_teacher_ranges.py`

### teacher_sampling_upper_edge_ranges — pruned (experiment2 end of experiment)

**Margin:** 2.6 nats behind last_example_marks_range_top; retired to keep the live set at 8 models: ELPD-LOO rank 9 of 19

**Hypothesis:** Refinement of `pedagogical_teacher_sampling`: people assume a helpful teacher chose the examples, so distinctive examples (numbers few plausible concepts would produce) count as stronger evidence for the concepts that contain them, over arithmetic rules (primes excluding 1) and ranges of 1-100. The one change, grafted from `upper_edge_marking_ranges`: people also take the largest example as marking where the program's range stops at the top, so a range ending exactly at the largest example gets an inferred evidence bonus while its lower end is unconstrained — people generalize downward from the examples more readily than upward (after 83,77 they reject 84 but accept 75).

**Source:** `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment2/model_loop/models/pruned/teacher_sampling_upper_edge_ranges.py`

### two_cluster_range_concepts — pruned (experiment2 end of experiment)

**Margin:** 3.0 nats behind last_example_marks_range_top; retired to keep the live set at 8 models: ELPD-LOO rank 11 of 19

**Hypothesis:** People do Bayesian concept learning over arithmetic rules (primes excluding 1, kind-level rule prior) and numeric ranges of 1-100 (width prior favouring moderate, decade-bounded ranges, the top of a range marked by the largest example, size principle with 10% random examples), but they can also think of the program's numbers as lying in two separate clusters — two equally wide stretches of the number line with a gap between them ("numbers near 10 and numbers near 80") — a kind of concept with its own inferred prior share and a width prior shared with ranges. The sharpest disagreement with the best model lives in the gap between widely separated examples (e.g. 45 after 10,80, or 35 after 5,65): the best model must cover such examples with one wide range or a rule and so says "yes" across the gap, whereas this model says "yes" near each example and "no" in the empty middle.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment2/model_loop/models/pruned/two_cluster_range_concepts.py`

### range_discounted_size_principle — pruned (experiment2 end of experiment)

**Margin:** 3.1 nats behind last_example_marks_range_top; retired to keep the live set at 8 models: ELPD-LOO rank 12 of 19

**Hypothesis:** Refinement of `prime_concept_excludes_one`: people do the same Bayesian concept learning over arithmetic rules (primes excluding 1), ranges of 1-100 and range-restricted rules, but they apply the size principle at full strength only to arithmetic rules and only partly to numeric ranges. The one change: when judging how well a range explains the examples, people treat the range as if it were wider than it is — its effective size lies between its true width and the whole 1-100 scale, by an inferred degree — so more examples tighten range-based generalization less than rule-based generalization, and clustered examples are explained by ranges (with "yes" spreading further across the examples' neighbourhood) more readily than strong sampling predicts.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment2/model_loop/models/pruned/range_discounted_size_principle.py`

### pedagogical_teacher_sampling — pruned (experiment2 end of experiment)

**Margin:** 3.3 nats behind last_example_marks_range_top; retired to keep the live set at 8 models: ELPD-LOO rank 13 of 19

**Hypothesis:** People assume the examples were chosen by a helpful teacher rather than drawn at random from the program's concept: a teacher who wants them to guess the concept picks the numbers that point most clearly to it, i.e. numbers that few other plausible concepts would also produce. Reasoning about this choice (pedagogical sampling), people take an example that is distinctive — unlikely under the concepts they find plausible in general — as stronger evidence for the concepts containing it than a commonplace example, with an inferred strength of this teacher assumption (zero gives the ordinary random-sampling learner), and say "yes" with the resulting posterior probability that the concept contains the target.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment2/model_loop/models/pruned/pedagogical_teacher_sampling.py`

### odd_composite_upper_edge_ranges — pruned (experiment2 end of experiment)

**Margin:** 3.5 nats behind last_example_marks_range_top; retired to keep the live set at 8 models: ELPD-LOO rank 14 of 19

**Hypothesis:** Refinement of `odd_composite_concept`: people do Bayesian concept learning over arithmetic rules (primes excluding 1), every range of 1-100 and range-restricted rules, with "odd numbers that are not prime" as a kind of rule of its own, so sets of odd composites (9,15,25) lead to "yes" for other odd non-primes but "no" for odd primes. The one change, grafted from `upper_edge_marking_ranges`: people take the largest example shown as marking where the program's range stops at the top, so a range ending exactly at the largest example gets an inferred evidence bonus while its lower end is unconstrained, and people generalize downward from the examples more readily than upward.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment2/model_loop/models/pruned/odd_composite_upper_edge_ranges.py`

### first_shown_top_marking — pruned (experiment2 end of experiment)

**Margin:** 3.7 nats behind last_example_marks_range_top; retired to keep the live set at 8 models: ELPD-LOO rank 15 of 19

**Hypothesis:** People do Bayesian concept learning over arithmetic rules (primes excluding 1), ranges of 1-100 and range-restricted rules, and they believe the program, with some inferred odds, opens its sequence by showing the top number of its range and then draws the remaining examples at random — so the order of the examples matters: only the first example shown can mark the range's top. When the first example is also the largest (e.g. 83,77 or 60), ranges ending there gain credit that grows with their width and "yes" spreads downward but stops above it; when the largest example comes later (e.g. 77,83) it is read as an ordinary random draw, so the same examples in that order license more generalization above the largest example.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment2/model_loop/models/pruned/first_shown_top_marking.py`

### odd_composite_concept — pruned (experiment2 end of experiment)

**Margin:** 3.9 nats behind last_example_marks_range_top; retired to keep the live set at 8 models: ELPD-LOO rank 16 of 19

**Hypothesis:** People do the same Bayesian concept learning as the best model (arithmetic rules with a kind-level prior, every range of 1-100, range-restricted rules, size principle with 10% random examples, "yes" = posterior probability the concept contains the target), but their space of concepts also includes a rule with exceptions, "odd numbers that are not prime" (9, 15, 21, 25, 27, 33, ...), as a kind of rule of its own with an inferred prior share. When the examples are all odd composites (e.g. 9,15,25 or 63,27,99,33), this smaller concept wins under the size principle, so people say "yes" to other odd non-primes but "no" to odd primes such as 7, 11 or 53, which "odd numbers" would accept.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment2/model_loop/models/pruned/odd_composite_concept.py`

### log_odds_yes_bias_concept_learning — pruned (experiment2 end of experiment)

**Margin:** 4.0 nats behind last_example_marks_range_top; retired to keep the live set at 8 models: ELPD-LOO rank 17 of 19

**Hypothesis:** People infer the program's concept by plain Bayesian concept learning over arithmetic rules and numeric ranges (rules of one kind equally likely, a width prior favouring moderately wide and decade-bounded ranges, the size principle, primes excluding 1), but their answer is not a direct readout of their belief: they carry a constant response bias toward "yes" or toward "no" that shifts their confidence by the same amount on the log-odds scale on every trial. The bias therefore moves answers most for targets the learner is unsure about (posterior near one half) and hardly at all for targets that are clearly in or clearly out of the concept.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment2/model_loop/models/pruned/log_odds_yes_bias_concept_learning.py`

### compressed_number_line_size_principle — pruned (experiment2 end of experiment)

**Margin:** 4.4 nats behind last_example_marks_range_top; retired to keep the live set at 8 models: ELPD-LOO rank 18 of 19

**Hypothesis:** People do Bayesian concept learning over arithmetic rules (primes excluding 1, rules of one kind equally likely) and numeric ranges of 1-100 (a width prior favouring moderate, decade-bounded ranges), with the size principle, but they judge how big a range is on a compressed, logarithmic mental number line: a stretch among large numbers (e.g. 81-90) feels smaller than an equally long stretch among small numbers (e.g. 1-10), by an inferred degree of compression. Because the size principle rewards small concepts, examples among large numbers are explained by tighter ranges, so people generalize less far from high examples like 83,77 than from low examples like 3,7.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment2/model_loop/models/pruned/compressed_number_line_size_principle.py`

### extreme_examples_attended_sampling — pruned (experiment2 end of experiment)

**Margin:** 4.6 nats behind last_example_marks_range_top; retired to keep the live set at 8 models: ELPD-LOO rank 19 of 19

**Hypothesis:** People attend to the extreme examples: they treat the smallest and the largest example shown as samples drawn from the program's concept, but the examples lying between them are encoded only as confirming that the concept includes them, so they count as weaker evidence under the size principle by an inferred degree (zero: mere consistency; one: the ideal learner). Otherwise people do Bayesian concept learning over arithmetic rules (primes excluding 1, kind-level prior) and ranges of 1-100 (Erlang width prior, decade-boundary preference, the largest example marking the top of the range), so adding examples inside the span of the first two tightens generalization less than an ideal learner predicts, while examples that extend the span tighten it fully.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment2/model_loop/models/pruned/extreme_examples_attended_sampling.py`

### named_sets_distinct_familiarity — pruned (experiment2 end of experiment)

**Margin:** 8.6 nats behind last_example_marks_range_top (2.1× dse)

**Hypothesis:** Refinement of `prime_concept_excludes_one`: people do the same Bayesian concept learning (arithmetic rules with a kind-level prior, every range of 1-100 with an Erlang width prior and decade-boundary preference, range-restricted rules, size principle with 10% random examples, primes excluding 1, "yes" = posterior probability the concept contains the target), but the familiar named sets — primes, even numbers, odd numbers, squares and cubes — are not equally likely a priori. The one change: each named set gets its own inferred prior weight instead of sharing one "special sets" weight equally, so how readily people reach for "even/odd" versus "squares" or "primes" (e.g. after 3,7 or 1,4,9 or 31,17) reflects how familiar each concept is.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment2/model_loop/models/pruned/named_sets_distinct_familiarity.py`

### erlang_interval_size_prior — pruned (experiment1 end of experiment)

**Margin:** 30.4 nats behind scale_end_avoiding_ranges (4.1× dse)

**Hypothesis:** Refinement of bigelow_piantadosi_interval: people do Bayesian concept learning over the same 42 rules and every interval of 1-100 (size principle, 10% of examples random, "yes" = posterior probability the concept contains the target), but their prior over intervals depends on how wide an interval is rather than where it lies. Following Tenenbaum's original number game, interval width has an Erlang prior with an inferred scale, so people a priori favour moderately sized ranges and disfavour both tiny and near-total ranges, which controls how far they generalize beyond the examples' span.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment1/model_loop/models/pruned/erlang_interval_size_prior.py`

### decade_bounded_interval_prior — pruned (experiment1 end of experiment)

**Margin:** 30.4 nats behind scale_end_avoiding_ranges (4.1× dse)

**Hypothesis:** Refinement of `located_sized_interval_prior`: people do Bayesian concept learning over the same 42 arithmetic rules and every interval of 1-100 (inferred rule weights and rule share, size principle with 10% random examples, "yes" = posterior probability the concept contains the target, Erlang prior over interval width with an inferred scale), but their prior over where a range lies is not a smooth density over its midpoint: it favours ranges bounded by round decade numbers ("the 20s", "1 to 50", "51 to 100") by an inferred factor. The one change is replacing the Beta-over-midpoint location prior with this round-boundary preference, so generalization from examples tends to stop at decade boundaries — "yes" stays high up to the edge of the decade the examples fall in and drops sharply past it, a sharper edge near the examples' span than a width-only prior predicts.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment1/model_loop/models/pruned/decade_bounded_interval_prior.py`

### anchored_endpoint_or_decade_ranges — pruned (experiment1 end of experiment)

**Margin:** 30.6 nats behind scale_end_avoiding_ranges (4.1× dse)

**Hypothesis:** Refinement of `endpoint_marking_interval_sampling`: people do Bayesian concept learning over 42 arithmetic rules and every interval of 1-100 (inferred rule weights and rule share, Erlang prior over interval width, size principle with 10% random examples, "yes" = posterior probability the concept contains the target), and they assume the program's examples tend to mark the ends of its range — but a range's ends can also be anchored at salient round numbers. The one change is grafting the incumbent's round-decade boundary preference onto the endpoint-marking model: ranges bounded by decade numbers ("the 20s", "1 to 50") get an inferred prior bonus alongside the bonus for ends coinciding with the smallest or largest example, so generalization stops either exactly at the examples' span or at the edge of the enclosing decade, whichever anchors are available.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment1/model_loop/models/pruned/anchored_endpoint_or_decade_ranges.py`

### endpoint_marking_interval_sampling — pruned (experiment1 end of experiment)

**Margin:** 30.8 nats behind scale_end_avoiding_ranges (4.1× dse)

**Hypothesis:** Refinement of erlang_interval_size_prior: people do the same Bayesian concept learning over 42 arithmetic rules and every interval of 1-100 (Erlang prior over interval width, size principle with 10% random examples, "yes" = posterior probability the concept contains the target), but when two or more distinct examples are shown they assume the program's examples tend to mark the ends of its range (edge-marking, pedagogical sampling). The one change is an inferred evidence bonus for each end of a candidate range that coincides with the smallest or largest example, so the range that exactly spans the examples is favoured over wider ones, and people say "yes" inside the examples' span but sharply less just beyond it.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment1/model_loop/models/pruned/endpoint_marking_interval_sampling.py`

### erlang_interval_response_lapse — pruned (experiment1 end of experiment)

**Margin:** 31.3 nats behind scale_end_avoiding_ranges (4.3× dse)

**Hypothesis:** Refinement of erlang_interval_size_prior: people do the same Bayesian concept learning (42 rules plus every interval of 1-100 with an Erlang prior over interval width, size principle, 10% random examples), but they do not always answer from that posterior — on an inferred share of trials they lapse and give a guessed answer, saying "yes" at their own inferred base rate. The one change is this response-lapse stage, which keeps a floor of "yes" for targets the posterior rules out and of "no" for targets it is sure of, instead of answering exactly at the posterior probability.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment1/model_loop/models/pruned/erlang_interval_response_lapse.py`

### weber_noisy_range_membership — pruned (experiment1 end of experiment)

**Margin:** 31.8 nats behind scale_end_avoiding_ranges (4.2× dse)

**Hypothesis:** People do Bayesian concept learning over arithmetic rules and numeric ranges (size principle, a prior favouring moderately wide ranges), but they check whether a target falls inside a range on a noisy analog mental number line whose noise grows in proportion to magnitude (Weber's law), while rule membership (even, multiple of 5, ...) is checked exactly. So the edge of a range is blurred rather than sharp: targets just outside the range sometimes count as "in" and targets just inside sometimes as "out", and this blur is wider among large numbers than among small ones.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment1/model_loop/models/pruned/weber_noisy_range_membership.py`

### gamma_shape_interval_width_prior — pruned (experiment1 end of experiment)

**Margin:** 32.5 nats behind scale_end_avoiding_ranges (4.3× dse)

**Hypothesis:** Refinement of erlang_interval_size_prior: people do the same Bayesian concept learning over 42 arithmetic rules and every interval of 1-100 (inferred rule weights and rule share, size principle with 10% random examples, "yes" = posterior probability the concept contains the target), but their prior over interval width is a gamma density whose shape is inferred along with its typical width, rather than an Erlang with its shape fixed at two. The one change is this free shape: a large shape means people expect ranges of a characteristic width and sharply reject ranges much wider than the examples' span (a sharp edge at the span), while a small shape means a long-tailed preference that generalizes smoothly past the edge — addressing the critique that people's yes rate drops more sharply at the span edge than the incumbent predicts.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment1/model_loop/models/pruned/gamma_shape_interval_width_prior.py`

### primacy_weighted_evidence — pruned (experiment1 end of experiment)

**Margin:** 32.7 nats behind scale_end_avoiding_ranges (4.5× dse)

**Hypothesis:** People do Bayesian concept learning over arithmetic rules and ranges of 1-100 (size principle, a prior favouring moderately wide ranges), but they anchor on their first impression: the first example shown counts fully as evidence about the concept, and each later example counts for progressively less, as attention and updating decline with each new example (a primacy effect). So the concept people settle on is pulled toward what the first example suggests, and the same examples shown in a different order lead to different generalization.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment1/model_loop/models/pruned/primacy_weighted_evidence.py`

### few_sample_majority_vote — pruned (experiment1 end of experiment)

**Margin:** 32.9 nats behind scale_end_avoiding_ranges (4.3× dse)

**Hypothesis:** People do Bayesian concept learning over arithmetic rules and numeric ranges (as in the best model, with an Erlang width prior on ranges), but they cannot compute the full posterior: they bring to mind a small number of concepts, each sampled in proportion to its posterior probability, and answer "yes" if most of the sampled concepts contain the target (ties broken at random). How many concepts a person samples varies from trial to trial with an inferred mean; sampling only one reproduces the ideal posterior predictive, while sampling several makes answers more all-or-none than the posterior probability.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment1/model_loop/models/pruned/few_sample_majority_vote.py`

### recency_weighted_evidence — pruned (experiment1 end of experiment)

**Margin:** 33.4 nats behind scale_end_avoiding_ranges (4.3× dse)

**Hypothesis:** People do Bayesian concept learning over arithmetic rules and ranges of 1-100 (size principle, with a prior favouring moderately wide ranges), but their memory of the examples is leaky: the most recently shown example counts fully as evidence, while each earlier example counts for progressively less, as if partly forgotten. Because earlier examples carry less weight, people stay less committed to the narrow concept the whole set picks out, and the more examples came before the last one, the further their generalization departs from the ideal learner's.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment1/model_loop/models/pruned/recency_weighted_evidence.py`

### per_endpoint_round_boundary_prior — pruned (experiment1 end of experiment)

**Margin:** 33.5 nats behind scale_end_avoiding_ranges (4.4× dse)

**Hypothesis:** Refinement of `decade_bounded_interval_prior`: people do the same Bayesian concept learning over 42 arithmetic rules and every interval of 1-100 (inferred rule weights and rule share, size principle with 10% random examples, Erlang prior over interval width, "yes" = posterior probability the concept contains the target) and they favour ranges with round boundaries, but they judge roundness one endpoint at a time and under either everyday convention. The one change: each end of a range that falls on a round decade number — a lower end like 1, 10, 11, 20, 21, ... or an upper end like 9, 10, 19, 20, ... 100 (so "the 20s" as 20-29, 21-30 or 20-30) — multiplies the range's prior by an inferred factor, so a range with one round end and one end hugging the examples is also favoured, instead of only ranges whose both ends sit at the incumbent's strict 21-30 convention.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment1/model_loop/models/pruned/per_endpoint_round_boundary_prior.py`

### offset_arithmetic_progression_concepts — pruned (experiment1 end of experiment)

**Margin:** 34.5 nats behind scale_end_avoiding_ranges (4.5× dse)

**Hypothesis:** People do Bayesian concept learning as in the best model (42 arithmetic rules and every interval of 1-100, with an Erlang width prior and a preference for decade-bounded ranges, size principle, "yes" = posterior probability the concept contains the target), but their space of candidate rules also includes offset arithmetic progressions — "every 4th number starting from 3" (3, 7, 11, ...), "every 6th number from 5" (5, 11, ..., 77, 83) — for every step from 3 to 12 and every non-zero starting offset. These progressions together receive an inferred share of prior belief, split equally among them, so examples whose gaps share a common step lead people to say "yes" to targets that continue the same counting pattern, even when they fall outside the examples' range and no multiple, digit or interval rule covers them.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment1/model_loop/models/pruned/offset_arithmetic_progression_concepts.py`

### located_sized_interval_prior — pruned (experiment1 end of experiment)

**Margin:** 35.8 nats behind scale_end_avoiding_ranges (4.5× dse)

**Hypothesis:** Refinement of `bigelow_piantadosi_interval`: people do Bayesian concept learning over the same 42 rules and every interval of 1-100 (inferred rule weights and rule share, size principle with 10% random examples, "yes" = posterior probability the concept contains the target), and they keep its prior over where an interval lies (a Beta density over the interval's midpoint). The one change is that the interval prior also depends on how wide the interval is, an Erlang over width with an inferred scale taken from `erlang_interval_size_prior`. The claim is that people's prior over ranges depends on both their location and their size, so generalization beyond the examples depends both on how wide a range people expect and on which part of 1-100 the examples fall in (for example, extending a range further toward small numbers than toward large ones).

**Source:** `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment1/model_loop/models/pruned/located_sized_interval_prior.py`

### noisy_criterion_threshold — pruned (experiment1 end of experiment)

**Margin:** 38.6 nats behind scale_end_avoiding_ranges (4.9× dse)

**Hypothesis:** People do the same Bayesian concept learning as the simplest well-fitting model (42 arithmetic rules plus every interval of 1-100 with an Erlang prior over interval width, size principle, 10% random examples), but they do not report their posterior belief directly: they say "yes" only when their belief that the target belongs to the concept clears an internal criterion of how sure they must be, and that comparison is noisy. The criterion's level and the noise are inferred, so answers follow an S-shaped function of the posterior centred on the criterion (not necessarily at 50%), producing floors of "no" for examples and of "yes" for excluded targets in proportion to how far each belief sits from the criterion.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment1/model_loop/models/pruned/noisy_criterion_threshold.py`

### bigelow_piantadosi_interval — pruned (experiment1 end of experiment)

**Margin:** 39.3 nats behind scale_end_avoiding_ranges (4.7× dse)

**Hypothesis:** Bayesian concept learning over 42 rules (powers of 2-10, multiples of 3-12, final digit 1-9, contained digit 1-9, primes, even, odd, squares, cubes) and every interval of 1-100, with a prior weight inferred for each rule separately and a share p_mix for the rules as a group. An interval's prior follows a Beta density over its midpoint, with both shapes inferred, so concepts in some parts of 1-100 are favoured. Examples are sampled from the concept (the size principle), 10% of them uniformly from 1-100, and "yes" is the posterior probability that the concept contains the target.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment1/model_loop/models/pruned/bigelow_piantadosi_interval.py`

### tempered_evidence_interval — pruned (experiment1 end of experiment)

**Margin:** 41.6 nats behind scale_end_avoiding_ranges (4.9× dse)

**Hypothesis:** A refinement of `bigelow_piantadosi_interval`: people do Bayesian concept learning over the same 42 rules and all intervals of 1-100 (inferred rule weights, a rule share, and a Beta prior over interval midpoints), but they under-weight the evidence of the examples — they treat the examples as only partly informative, raising the size-principle likelihood to an inferred power below or above one rather than using it at full strength. The one change is this evidence-tempering exponent: with weaker evidence, the posterior stays spread over broader concepts and "yes" generalizes further from the examples than strong sampling predicts.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment1/model_loop/models/pruned/tempered_evidence_interval.py`

### bigelow_piantadosi_independent — pruned (experiment1 end of experiment)

**Margin:** 42.5 nats behind scale_end_avoiding_ranges (4.9× dse)

**Hypothesis:** Bayesian concept learning over 42 rules (powers of 2-10, multiples of 3-12, final digit 1-9, contained digit 1-9, primes, even, odd, squares, cubes) and every interval of 1-100, with a prior weight inferred for each rule separately and a share p_mix for the rules as a group; intervals are equally likely. Examples are sampled from the concept (the size principle), 10% of them uniformly from 1-100, and "yes" is the posterior probability that the concept contains the target.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment1/model_loop/models/pruned/bigelow_piantadosi_independent.py`

### noise_tolerant_concept_learning — pruned (experiment1 end of experiment)

**Margin:** 43.9 nats behind scale_end_avoiding_ranges (5.0× dse)

**Hypothesis:** Refinement of `bigelow_piantadosi_independent`: people do Bayesian concept learning over the same 42 arithmetic rules and all intervals of 1-100, weighing each by how well it explains the examples under the size principle, but the share of examples they believe are random noise (not drawn from the hidden concept) is a free quantity inferred from the data rather than fixed at 10%. A learner who expects more noise keeps concepts that miss some of the examples, and so is less certain even about targets among the examples and generalizes more loosely — which is the one change from the original model.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment1/model_loop/models/pruned/noise_tolerant_concept_learning.py`

### range_restricted_rule_concepts — pruned (experiment2 end of experiment)

**Margin:** 51.1 nats behind last_example_marks_range_top (3.4× dse)

**Hypothesis:** People do the same Bayesian concept learning as the best model (42 arithmetic rules and every interval of 1-100, a kind-level rule prior, an Erlang width prior favouring decade-bounded ranges, size principle with 10% random examples, "yes" = posterior probability the concept contains the target), but their hypothesis space also contains range-restricted rules — conjunctions such as "even numbers in the 60s-70s" or "multiples of 4 below 20" — which receive an inferred share of prior belief and, being smaller, are strongly favoured by the size principle when the examples cluster. The sharpest disagreement with the best model lives on rule-consistent targets far from the examples' span (e.g. 12 or 30 after 66,78; 64 or 100 after 4,16,12): the best model says "yes" to every member of the rule wherever it lies, while this model predicts that "yes" to rule members falls off outside the region the examples occupy.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment2/model_loop/models/pruned/range_restricted_rule_concepts.py`

### family_pooled_rule_prior — pruned (experiment2 end of experiment)

**Margin:** 52.4 nats behind last_example_marks_range_top (3.3× dse)

**Hypothesis:** Refinement of `decade_bounded_interval_prior`: people do the same Bayesian concept learning over 42 arithmetic rules and every interval of 1-100 (rule share, size principle with 10% random examples, Erlang prior over interval width, a preference for decade-bounded ranges, "yes" = posterior probability the concept contains the target), but their prior over rules is set by the kind of rule, not rule by rule. The one change: people hold a belief about how likely each kind of rule is (powers, multiples, final digit, contained digit, and special sets such as primes, evens, odds, squares, cubes) and treat all rules of one kind as equally likely, so "multiples of 7" is as plausible a priori as "multiples of 4", instead of the incumbent's separate free weight for each of the 42 rules.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment2/model_loop/models/pruned/family_pooled_rule_prior.py`

### target_as_weak_evidence — pruned (experiment2 end of experiment)

**Margin:** 52.5 nats behind last_example_marks_range_top (3.3× dse)

**Hypothesis:** People do Bayesian concept learning over arithmetic rules and ranges of 1-100 (as in `family_pooled_rule_prior`: kind-level rule prior, size principle with 10% random examples, Erlang width prior, a preference for decade-bounded ranges), but being asked about a number is itself taken as weak evidence: people partly treat the queried target as if the program had produced it, adding it to the examples with an inferred fraction of an example's weight before judging whether the concept contains it. This suggestion (positive-test) bias inflates "yes" most for targets that fit a small, plausible concept alongside the examples, and least for targets that only a broad or implausible concept would contain.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment2/model_loop/models/pruned/target_as_weak_evidence.py`

### random_forgetting_range_restricted_rules — pruned (experiment2 end of experiment)

**Margin:** 52.7 nats behind last_example_marks_range_top (3.5× dse)

**Hypothesis:** Refinement of `random_example_forgetting`: people do Bayesian concept learning over 42 arithmetic rules and every range of 1-100 (kind-level rule prior, size principle with 10% random examples, Erlang width prior with a preference for decade-bounded ranges, "yes" = posterior probability the concept contains the target) from only the examples they still hold in memory, each shown example being independently forgotten with an inferred probability. The one change, grafted from `range_restricted_rule_concepts`: the concepts people entertain also include range-restricted rules ("even numbers in the 60s", "multiples of 4 below 20") with their own inferred share of prior belief, so the remembered examples can pick out a rule confined to the region they occupy — forgetting loosens generalization on large sets while the conjunctions confine "yes" to rule members near the remembered examples, and the two together predict how far rule-based "yes" extends beyond the examples' span.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment2/model_loop/models/pruned/random_forgetting_range_restricted_rules.py`

### log_scale_erlang_interval_prior — pruned (experiment1 end of experiment)

**Margin:** 53.4 nats behind scale_end_avoiding_ranges (5.4× dse)

**Hypothesis:** Refinement of erlang_interval_size_prior: people do the same Bayesian concept learning over 42 arithmetic rules and every interval of 1-100 (inferred rule weights and rule share, size principle with 10% random examples, "yes" = posterior probability the concept contains the target, Erlang prior over interval width with an inferred scale), but they judge how wide a range is on a compressed, logarithmic number line (by the ratio of its endpoints, Weber-like) rather than by how many numbers it holds. The one change is this log-scale width in the Erlang prior: a range of fixed span counts as narrower among large numbers than among small ones, so people generalize further from examples like 83,77 than from examples like 3,7.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment1/model_loop/models/pruned/log_scale_erlang_interval_prior.py`

### random_example_forgetting — pruned (experiment2 end of experiment)

**Margin:** 53.6 nats behind last_example_marks_range_top (3.5× dse)

**Hypothesis:** People do Bayesian concept learning over arithmetic rules and ranges of 1-100 (as in the best model: kind-level rule prior, size principle, a width prior and a preference for decade-bounded ranges), but their working memory for the examples is unreliable: each example shown is independently lost from memory with an inferred probability (at least one is always retained), and people generalize only from the examples they still hold. So their judgment is an average of what the ideal learner would conclude from each remembered subset of the examples — larger sets are more often reduced to fewer examples, which weakens the size principle's tightening and lets broader concepts and single-example generalizations survive, irrespective of the order the examples were shown in.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment2/model_loop/models/pruned/random_example_forgetting.py`

### scale_end_avoiding_ranges — pruned (experiment2 end of experiment)

**Margin:** 54.6 nats behind last_example_marks_range_top (3.5× dse)

**Hypothesis:** People do Bayesian concept learning over 42 arithmetic rules and every interval of 1-100 exactly as in `family_pooled_rule_prior` (rule-kind prior shares, size principle with 10% random examples, Erlang prior over interval width, a preference for decade-bounded ranges, "yes" = posterior probability the concept contains the target), but they believe the program's range is unlikely to run all the way to the ends of the 1-100 scale. Each end of a candidate range that sits at 1 or at 100 changes its prior by an inferred factor, so people generalize less toward the extremes of the scale — fewer "yes" answers to numbers in the 90s above high examples (and near 1 below low ones) than a learner who treats the scale's ends like any other number.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment2/model_loop/models/pruned/scale_end_avoiding_ranges.py`

### weber_blurred_ranges_kind_pooled_rules — pruned (experiment2 end of experiment)

**Margin:** 55.0 nats behind last_example_marks_range_top (3.5× dse)

**Hypothesis:** Refinement of `weber_noisy_range_membership`: people do Bayesian concept learning over 42 arithmetic rules and every range of 1-100 (size principle, a prior favouring moderately wide ranges), and they check whether a target falls inside a range on a noisy mental number line whose noise grows with magnitude (Weber's law), so range edges are blurred, more among large numbers, while rule membership is checked exactly. The one change, grafted from `family_pooled_rule_prior`: people's prior over rules is set by the kind of rule (powers, multiples, final digit, contained digit, special sets), with all rules of one kind equally likely, instead of a separate free weight for each of the 42 rules — so the blurred-edge account stops being handicapped by an overfitted rule prior, and range generalization ends in a magnitude-scaled soft edge rather than at decade boundaries.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment2/model_loop/models/pruned/weber_blurred_ranges_kind_pooled_rules.py`

### noise_rate_inferring_restricted_rules — pruned (experiment2 end of experiment)

**Margin:** 58.2 nats behind last_example_marks_range_top (3.8× dse)

**Hypothesis:** Refinement of `range_restricted_rule_concepts`: people do the same Bayesian concept learning over arithmetic rules, plain ranges of 1-100 and range-restricted rules ("even numbers in the 60s-70s"), with the size principle and "yes" = posterior probability the concept contains the target. The one change is how they treat noisy examples: instead of assuming that exactly 10% of the program's examples are random, people are unsure how many of the examples are random noise and infer the noise rate jointly with the concept for each set, from an inferred prior belief over noise levels. So a set with one stray example (such as 8,24,48,63) can still be read as a tight rule plus one random number, while a set that fits a tight concept exactly drives the belief in noise down and sharpens generalization.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment2/model_loop/models/pruned/noise_rate_inferring_restricted_rules.py`

### parity_kind_rule_prior — pruned (experiment2 end of experiment)

**Margin:** 61.2 nats behind last_example_marks_range_top (4.0× dse)

**Hypothesis:** Refinement of `family_pooled_rule_prior`: people do the same Bayesian concept learning over 42 arithmetic rules and every interval of 1-100 (size principle with 10% random examples, Erlang prior over interval width, a preference for decade-bounded ranges, "yes" = posterior probability the concept contains the target, rules of one kind equally likely a priori), but parity is a kind of rule in its own right. The one change: "even numbers" and "odd numbers" form their own kind with their own inferred prior share, instead of being lumped with primes, squares and cubes as one "special sets" kind, so how readily people fall back on odd/even after examples like 3,7 or 66,78 is set apart from how readily they think of primes, squares or cubes.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment2/model_loop/models/pruned/parity_kind_rule_prior.py`

### win_stay_lose_shift_concept — pruned (experiment2 end of experiment)

**Margin:** 104.0 nats behind last_example_marks_range_top (6.4× dse)

**Hypothesis:** People do not weigh all the examples jointly; they hold a single running guess about the concept and revise it one example at a time, in the order shown (win-stay, lose-shift). After the first example they adopt one concept drawn in proportion to its posterior plausibility (arithmetic rules with a kind-level prior, ranges with a width prior favouring decade-bounded ranges, size principle); each later example that their current guess contains leaves the guess unchanged, and only an example the guess fails to contain makes them draw a new guess from the posterior given all the examples seen so far. So a broad concept adopted early (e.g. "even numbers" or a wide range) survives later examples that would have narrowed an ideal learner's belief, and the same examples in a different order lead to different generalization.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment2/model_loop/models/pruned/win_stay_lose_shift_concept.py`

### minimal_consistent_concept — pruned (experiment1 end of experiment)

**Margin:** 240.9 nats behind scale_end_avoiding_ranges (10.8× dse)

**Hypothesis:** People do not average over many candidate concepts; they commit to the single smallest concept that contains every example — the tightest arithmetic rule (powers, multiples, final digit, contained digit, primes, even, odd, squares, cubes) or, if no rule is tighter, the exact range spanned by the examples — and say "yes" to a target essentially only if it belongs to that one concept. Apart from a constant rate of lapses in either direction, how far an excluded target lies from the examples does not matter.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment1/model_loop/models/pruned/minimal_consistent_concept.py`

### arithmetic_step_lattice — pruned (experiment1 end of experiment)

**Margin:** 251.6 nats behind scale_end_avoiding_ranges (10.6× dse)

**Hypothesis:** People read the spacing between the examples as a fixed counting step (the largest step that divides every gap between examples, e.g. 3,7 → step 4) and accept a target to the extent that it can be reached from an example by repeatedly adding or subtracting that step. Targets off this arithmetic lattice get only a baseline "yes" rate, and acceptance of on-lattice targets falls off with the number of steps needed to reach them from the nearest example.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment1/model_loop/models/pruned/arithmetic_step_lattice.py`

### decade_block_range_concepts — pruned (experiment2 end of experiment)

**Margin:** 260.1 nats behind last_example_marks_range_top (7.7× dse)

**Hypothesis:** People do Bayesian concept learning over arithmetic rules (powers, multiples, final digit, contained digit, primes, even, odd, squares, cubes; rules of one kind equally likely a priori) and numeric ranges, with the size principle, but they only ever think of ranges as whole blocks of decades ("the 80s and 90s", "1 to 30", "41 to 60") — never as a range with arbitrary ends hugging the examples. So range-based generalization is all-or-none within a decade: a target in the same decade block as the examples is accepted as readily as one right next to them, and acceptance drops abruptly at decade boundaries, with no gradient tied to the examples' exact span.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment2/model_loop/models/pruned/decade_block_range_concepts.py`

### featural_exemplar_similarity — pruned (experiment1 end of experiment)

**Margin:** 268.5 nats behind scale_end_avoiding_ranges (10.6× dse)

**Hypothesis:** People do not induce a rule; they judge the target by its summed similarity to the remembered examples (exemplar generalization). Similarity decays exponentially with psychological distance, which combines how far apart the two numbers are in magnitude and how many of their salient surface features (parity, final digit, decade, divisibility by 3 and 5, primality, squareness, power of two) they fail to share, and the more similar the target is to the examples on average, the more likely a "yes".

**Source:** `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment1/model_loop/models/pruned/featural_exemplar_similarity.py`

### span_scaled_hull_similarity — pruned (experiment1 end of experiment)

**Margin:** 283.2 nats behind scale_end_avoiding_ranges (11.1× dse)

**Hypothesis:** People judge a target by its numerical closeness to the range the examples span: every target inside the examples' range counts as fully similar, and similarity outside it decays exponentially with the distance from the nearest edge of that range. The width of this generalization gradient scales with how widely the examples are spread, so tightly clustered examples give a steep fall-off at the range edge and widely spread examples a gentle one; the more similar the target, the more likely a "yes".

**Source:** `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment1/model_loop/models/pruned/span_scaled_hull_similarity.py`

### shepard_exemplar_similarity_average — pruned (experiment1 end of experiment)

**Margin:** 289.5 nats behind scale_end_avoiding_ranges (10.3× dse)

**Hypothesis:** People do not infer one concept from the examples jointly; they store each example as a separate exemplar and judge the target by its average similarity to the stored exemplars (an exemplar-memory account). The similarity of a target to one exemplar is Shepard's consequential-region generalization: how likely a familiar number concept (an arithmetic rule, or a range with a preference for moderate widths) that contains that single exemplar also contains the target. Because each exemplar generalizes on its own and the similarities are averaged rather than combined as joint evidence, adding examples that share a narrow rule (e.g. 4, 16, 12) does not sharpen generalization the way it would for an ideal learner.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment1/model_loop/models/pruned/shepard_exemplar_similarity_average.py`

### weak_sampling_concept_learning — pruned (experiment1 end of experiment)

**Margin:** 289.6 nats behind scale_end_avoiding_ranges (10.7× dse)

**Hypothesis:** People generalize by Bayesian concept learning over the same rules and intervals, but they treat the examples only as consistent with the program's concept (weak sampling), not as drawn from it, so a large concept is not penalized for making the examples a coincidence. A concept's support is its prior times how many examples it fails to contain, with no size principle. This disagrees most sharply with the best model on sets of several examples that fit a narrow rule (e.g. 4,16,12 or 10,50,100) and on far targets: it predicts broad "yes" to targets in wide intervals and larger rules (even numbers, multiples of 2) instead of tightening around the examples as more of them arrive.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment1/model_loop/models/pruned/weak_sampling_concept_learning.py`

### one_sided_magnitude_threshold_concepts — pruned (experiment2 end of experiment)

**Margin:** 506.1 nats behind last_example_marks_range_top (9.9× dse)

**Hypothesis:** People do Bayesian concept learning over arithmetic rules (primes excluding 1, rules of one kind equally likely) with the size principle, but they do not represent numeric ranges with two ends: the only magnitude concepts they entertain are one-sided thresholds — "numbers up to N" or "numbers N and above" (with round decade thresholds preferred and an inferred preference for upper bounds over lower bounds). So magnitude-based "yes" never stops on both sides of the examples: once people judge by size, they accept everything below the largest example (or everything above the smallest), however far away, rather than only the examples' neighbourhood.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment2/model_loop/models/pruned/one_sided_magnitude_threshold_concepts.py`

### attention_focused_exemplar_similarity — pruned (experiment2 end of experiment)

**Margin:** 750.0 nats behind last_example_marks_range_top (17.0× dse)

**Hypothesis:** People judge a target by its summed similarity to the remembered examples on a multidimensional representation of numbers (magnitude, parity, final digit, decade, divisibility by 3, 4 and 5, primality, squareness, power of two), where the metric is an attention-weighted city-block distance and similarity decays exponentially with that distance. The examples steer attention: dimensions on which the examples agree with each other (e.g. all even, all in the 60s, all ending in 3) draw attention in proportion to that agreement, by an inferred degree of focusing, so a target that differs from the examples on a dimension they share is judged far, while differences on dimensions the examples vary on barely matter.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment2/model_loop/models/pruned/attention_focused_exemplar_similarity.py`

### common_feature_contrast_similarity — pruned (experiment2 end of experiment)

**Margin:** 785.1 nats behind last_example_marks_range_top (19.3× dse)

**Hypothesis:** People judge a target by its similarity to the example set as a whole, not by inducing a concept: the psychological similarity is the share of the properties that all the examples have in common (parity, multiples, final digit, contained digit, powers, primes, squares, cubes, decade) that the target also has, each property counted by how distinctive it is (rarer properties count more), together with how close the target is in magnitude to the nearest example on a compressed (logarithmic) distance scale. The probability of "yes" rises smoothly with this similarity, so a target matching the set's distinctive shared properties and lying near an example is accepted, and acceptance falls off gradually — never all-or-none — as it loses shared properties or moves away.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment2/model_loop/models/pruned/common_feature_contrast_similarity.py`

### soft_min_exemplar_generalization — pruned (experiment2 end of experiment)

**Margin:** 899.0 nats behind last_example_marks_range_top (22.3× dse)

**Hypothesis:** People do not infer one concept from the examples jointly; they store each example as a separate exemplar and judge the target by its similarity to each stored exemplar, where the similarity of a target to one exemplar is Shepard's consequential-region generalization (how likely a familiar number concept — an arithmetic rule, or a range with a preference for moderate, decade-bounded widths — that contains that exemplar also contains the target). The similarities are then combined conjunctively, as a soft minimum dominated by the least similar exemplar (the target must resemble every example, with an inferred degree of strictness), so a target is accepted only if it fits with all of the examples, but adding examples never tightens generalization by the size principle beyond what each exemplar alone implies.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment2/model_loop/models/pruned/soft_min_exemplar_generalization.py`
