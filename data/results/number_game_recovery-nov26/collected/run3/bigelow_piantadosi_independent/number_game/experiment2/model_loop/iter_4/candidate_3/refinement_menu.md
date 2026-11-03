# Refinement menu

The models you may refine, other than the incumbent `prime_over_odd_reading`. Each entry is one mechanism: its hypothesis in full as its author stated it, how it stands, and where its source is. This is a menu, not a list of what is ruled out: a pruned model lost to the best model on the data it was scored on, by the stated margin, but its mechanism may be partly right — and a model that lost narrowly is the most promising target here. Read the source of the model you pick before you write anything.

## Live models other than the incumbent (in the set; best first)

### prime_salient_marked_odd — rank 1, 3.4 ± 2.0 nats behind the best (1.7× dse: statistically tied with the best), ELPD-LOO -2203.9

**Hypothesis:** Refinement of `prime_salient_hundred_chart` (everything kept: primitive-built rules with a per-primitive cost, final-digit discount, costs on powers/cubes and squares, times-table-graded multiples, Erlang ranges with a bonus for whole hundred-chart rows, a fitted prior bonus for the familiar category "prime numbers", the size principle and near-all-or-none commitment). The one change, taken from `marked_odd_parity_concepts`, is that "odd" is a marked, negatively defined category ("not even"), so the rule "odd numbers" carries a fitted extra prior cost that "even numbers" does not. As a result, after all-prime sets of odd primes (53, 47, 41; 67, 31, 17, 61) "odd" competes less with "primes", and people say yes to other primes but no to odd composites (9, 21, 63, 81) more sharply than prime salience alone predicts, while after all-even sets they still extend "even" readily.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/d6b6238b22d937a5/repo/_runs/cell_1/experiment2/model_loop/models/prime_salient_marked_odd.py`

### hidden_divisibility_by_three — rank 2, 5.3 ± 6.2 nats behind the best (0.9× dse: statistically tied with the best), ELPD-LOO -2205.9

**Hypothesis:** Refinement of `prime_salient_marked_odd` (everything kept: primitive-built rules with a per-primitive cost, final-digit discount, costs on powers/cubes and squares, times-table-graded multiples, Erlang ranges with a bonus for whole hundred-chart rows, a prime bonus, a marked-odd cost, the size principle and near-all-or-none commitment). The one change is that divisibility by 3 cannot be read off a numeral's last digit the way divisibility by 2, 5 or 10 can; it needs a digit-sum check. So every rule whose membership test contains that check ("multiples of 3", "of 6", "of 9", "of 12") carries one shared, fitted extra prior cost. As a result, after odd multiples of 3 such as 69, 9, 39, 21 or 63, 27, 99, 33, people seldom commit to "multiples of 3". They extend "odd", a final digit or a range instead, and say yes to odd non-multiples of 3 (19, 29, 49, 23, 37) about as readily as to other multiples of 3.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/d6b6238b22d937a5/repo/_runs/cell_1/experiment2/model_loop/models/hidden_divisibility_by_three.py`

### prime_salient_hundred_chart — rank 3, 10.4 ± 8.0 nats behind the best (1.3× dse: statistically tied with the best), ELPD-LOO -2210.9

**Hypothesis:** Refinement of `hundred_chart_row_ranges` (everything kept: primitive-built rules with a per-primitive cost, final-digit discount, costs on powers/cubes and squares, times-table-graded multiples, Erlang ranges with a bonus for whole hundred-chart rows, the size principle and near-all-or-none commitment). The one change, taken from `prime_salient_concepts`, is that "prime numbers" is a familiar school-math category kept ready to hand, so it gets a fitted prior bonus over other compound rules. So after all-prime sets such as 2, 47 or 67, 31, 17, 61, people commit to "primes": they say yes to other primes more than to odd composite numbers, which the current best model gets backwards.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/d6b6238b22d937a5/repo/_runs/cell_1/experiment2/model_loop/models/prime_salient_hundred_chart.py`

### marked_odd_parity_concepts — rank 4, 14.0 ± 3.7 nats behind the best (3.7× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -2214.5

**Hypothesis:** "Odd" is a marked, negatively defined category: people think of an odd number as one that is "not even" rather than by a positive property, so "odd numbers" is harder to entertain as the program's rule than "even numbers", even though both are equally simple and equally large. As a result, after all-odd examples (e.g. 19, 73 or 7, 31) people often reject other odd numbers (63, 81, 87, 21) and fall back on a range or a narrower rule. After all-even examples they readily extend "even". Everything else in how they infer the rule is unchanged: primitive-built rules, the size principle, hundred-chart ranges and near-all-or-none commitment.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/d6b6238b22d937a5/repo/_runs/cell_1/experiment2/model_loop/models/marked_odd_parity_concepts.py`

### hundred_chart_row_ranges — rank 5, 20.3 ± 8.9 nats behind the best (2.3× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -2220.8

**Hypothesis:** Refinement of `factor_graded_multiples_concepts`: everything is kept (primitive-built rules with a fitted cost per extra primitive, a final-digit discount, extra costs for powers/cubes and for squares, times-table-graded multiples costs, Erlang-weighted ranges, the size principle and near-all-or-none commitment). The one change is how people picture numerical ranges: they lay 1-100 out as a hundred chart of ten rows (1-10, 11-20, ..., 91-100), so ranges made of whole rows ("1 to 50", "21 to 40", "81 to 90", "1 to 100") get a fitted prior bonus over ranges with arbitrary endpoints. When no rule explains the examples well, people therefore fall back on a broad round block of the chart and say yes fairly uniformly to targets within it, rather than to a tight range hugging the examples.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/d6b6238b22d937a5/repo/_runs/cell_1/experiment2/model_loop/models/hundred_chart_row_ranges.py`

### sieve_prime_concepts — rank 6, 41.3 ± 16.0 nats behind the best (2.6× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -2241.8

**Hypothesis:** People recognise "prime numbers" not by knowing the true primes but by a quick divisibility sieve: a number counts as prime-like if it is not even, does not end in 0 or 5, and is not in the 3 times table (2, 3 and 5 themselves included). This sieve concept is a familiar, readily entertained rule with its own fitted prior bonus, layered on the best model's concept search (primitive costs, final-digit discount, power/square costs, factor-graded multiples, hundred-chart row ranges, size principle, near-all-or-none commitment), so after all-prime sets such as 53, 47, 41 or 19, 73 people say yes to other primes and no to odd multiples of 3 or 5. The sharpest disagreement with the best model (and with true-prime salience) lives at odd composites that survive the sieve, 49, 77 and 91: after a prime set people should say yes to them about as often as to real primes.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/d6b6238b22d937a5/repo/_runs/cell_1/experiment2/model_loop/models/sieve_prime_concepts.py`

### prime_salient_concepts — rank 7, 92.5 ± 14.3 nats behind the best (6.5× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -2293.1

**Hypothesis:** Refinement of `factor_graded_multiples_concepts` that changes how likely one concept is a priori. "Prime numbers" is a familiar school-math category. People keep it ready to hand, so it carries a fitted prior bonus over other compound rules (multiples, powers, final digits, digits contained). Everything else is kept: primitive costs, the final-digit discount, the costs on powers and squares, factor-graded multiples, Erlang ranges, the size principle and near-all-or-none commitment. So after sets such as 2, 47 or 53, 2 or 67, 31, 17, 61, people commit to "primes": they say yes to other primes (5, 13, 73, 89) and no to non-prime numbers inside the examples' range, more often than the current best model predicts.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/d6b6238b22d937a5/repo/_runs/cell_1/experiment2/model_loop/models/prime_salient_concepts.py`

### factor_graded_multiples_concepts — rank 8, 103.7 ± 14.9 nats behind the best (6.9× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -2304.2

**Hypothesis:** Refinement of `costly_squares_final_digit_concepts`. Everything is kept: concepts built from mental primitives with a fitted cost per extra primitive, a discount for "ends in d" rules, separate extra costs for powers/cubes and for perfect squares, Erlang-weighted ranges, the size principle and near-all-or-none commitment. The one change is that "multiples of m" rules are not equally available for every m. People check divisibility by recalling a times table, and larger tables are less fluent, so the prior cost of a multiples rule grows with the logarithm of its factor, at a fitted rate. People therefore readily read sets as multiples of 3, 4 or 5 (e.g. 20, 32, 92 as multiples of 4; 90, 100, 5 as multiples of 5). After 14, 98, 91 or 8, 64 they seldom commit to "multiples of 7" or "multiples of 8" and say no to other multiples (7, 35, 63; 24, 40, 56).

**Source:** `/scr/wurgaft/auto-psych/agent_trees/d6b6238b22d937a5/repo/_runs/cell_1/experiment2/model_loop/models/factor_graded_multiples_concepts.py`

### forward_counting_graded_multiples — rank 9, 108.9 ± 15.0 nats behind the best (7.3× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -2309.5

**Hypothesis:** Refinement of `forward_counting_final_digit_concepts`: people build number concepts from mental primitives (a cost per extra primitive, an extra cost on powers/squares/cubes, a discount on "ends in d" rules, Erlang-weighted ranges, the size principle, near-all-or-none commitment) and picture a range-type program as counting upward from its start, so they generalise a little more above the examples than below. The one change, taken from `factor_graded_multiples_concepts`, is that "multiples of m" rules are not equally available for every m: divisibility is checked by recalling a times table, and larger tables are less fluent, so a multiples rule's prior cost grows with the logarithm of its factor at a fitted rate. People therefore readily read sets as multiples of 3, 4 or 5 but seldom commit to multiples of 7, 8, 11 or 12, and when they do not, they fall back on an upward-tilted range around the examples.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/d6b6238b22d937a5/repo/_runs/cell_1/experiment2/model_loop/models/forward_counting_graded_multiples.py`

### forward_counting_final_digit_concepts — rank 10, 116.0 ± 16.5 nats behind the best (7.1× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -2316.5

**Hypothesis:** Refinement of the pruned `final_digit_multiplicative_concepts`: people still build number concepts from mental primitives (a fitted cost per extra primitive, a fitted extra cost on powers, squares and cubes, a fitted prior discount on salient "ends in d" rules, Erlang-weighted ranges, the size principle and near-all-or-none commitment). The one change, taken from `forward_counting_range_sampling`, is how people think a range-type program produces its numbers: they picture it counting upward from where the range starts, so the examples are read as early, low members of the range, with the strength of this forward-counting assumption fitted. So when a set is explained by a range, people say yes more to numbers a little above the examples than to numbers equally far below them, while their rule-based answers (parity, multiples, final digits) are unchanged.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/d6b6238b22d937a5/repo/_runs/cell_1/experiment2/model_loop/models/forward_counting_final_digit_concepts.py`

### chain_length_graded_powers — rank 11, 123.3 ± 19.7 nats behind the best (6.3× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -2323.8

**Hypothesis:** Refinement of `factor_graded_multiples_concepts`. Everything in it is kept: concepts built from mental primitives with a fitted cost per extra primitive, the "ends in d" discount, extra costs for powers/cubes and for squares, factor-graded multiples costs, Erlang-weighted ranges, the size principle and near-all-or-none commitment. The one change is that a "powers of b" rule is held in mind as a chain of repeated multiplications, so it is harder to conceive the longer the chain it spans within 1-100. Its prior cost grows with the logarithm of the chain's length at a fitted rate. Short chains made of familiar times-table facts (1, 8, 64; 1, 9, 81; 1, 10, 100) are therefore readily entertained, while long chains such as powers of 2 stay costly. After 8, 64, people commit to "powers of 8" and say no to the other multiples of 8 (24, 40, 56) more than the incumbent predicts.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/d6b6238b22d937a5/repo/_runs/cell_1/experiment2/model_loop/models/chain_length_graded_powers.py`

### first_example_overridden_concepts — rank 12, 143.8 ± 17.6 nats behind the best (8.2× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -2344.3

**Hypothesis:** People revise their guess about the concept as each example arrives, so the later examples carry the final say: the first example shown is the one most readily dismissed as a stray that the program did not really mean, while later examples are trusted to belong to the concept. The concept they settle on (built from primitives with simplicity costs, a final-digit discount, extra costs for powers and squares, the size principle and near-all-or-none commitment) may therefore ignore a first example that breaks the pattern the later ones share, and reordering the same examples (e.g. 8,16 versus 16,8) changes which concept wins.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/d6b6238b22d937a5/repo/_runs/cell_1/experiment2/model_loop/models/first_example_overridden_concepts.py`

### costly_squares_final_digit_concepts — rank 13, 146.6 ± 18.6 nats behind the best (7.9× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -2347.1

**Hypothesis:** Refinement of `final_digit_multiplicative_concepts`. People still build number concepts from mental primitives: a fitted cost per extra primitive, a fitted prior discount for salient "ends in d" rules, Erlang-weighted intervals, the size principle and near-all-or-none commitment to the best-explaining concept. The one change splits the single cost on repeated-multiplication concepts in two. "Perfect squares" (a number multiplied by itself) pays its own fitted extra cost, separate from the cost shared by powers of a base (and their exception variants) and cubes. The claim is that self-multiplication is the hardest of these to conceive, harder than repeatedly multiplying by a fixed base. So after square sets such as 100, 1, 16 or 4, 64, 25, 81, people rarely commit to "squares". Instead they fall back on other rules, ranges or scattered guesses. Meanwhile, powers of 2 or 3 stay more available than a single cost on all of these concepts allows.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/d6b6238b22d937a5/repo/_runs/cell_1/experiment2/model_loop/models/costly_squares_final_digit_concepts.py`

### repeat_averse_concept_learning — rank 14, 149.1 ± 18.8 nats behind the best (7.9× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -2349.6

**Hypothesis:** People infer the hidden concept as a Bayesian learner would: they build number rules from simple primitives and weigh them, and numeric ranges, by the size principle. The one distortion is that they treat the program as producing distinct numbers, as if it sampled without replacement, so a number already shown as an example seems less likely to come up again. As a result they sometimes say "no" to a target that is itself one of the examples, even though it certainly fits the concept, while their answers to every other target are unchanged.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/d6b6238b22d937a5/repo/_runs/cell_1/experiment2/model_loop/models/repeat_averse_concept_learning.py`

### small_number_expectation_concepts — rank 15, 163.2 ± 20.5 nats behind the best (8.0× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -2363.8

**Hypothesis:** Refinement of `costly_squares_final_digit_concepts`: people infer the hidden concept exactly as in that model (primitive-built rules with a per-primitive cost, a final-digit discount, extra costs on powers/cubes and on squares, Erlang-weighted intervals, the size principle, near-all-or-none commitment). The one change is a small-number expectation: people believe a computer program is more likely to produce small numbers than large ones, so their readiness to say "yes" falls steadily with the target's magnitude at a fitted rate. For the same concept belief, a target in the 80s or 90s gets fewer yeses than one in the teens or 20s.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/d6b6238b22d937a5/repo/_runs/cell_1/experiment2/model_loop/models/small_number_expectation_concepts.py`

### place_value_disjunctive_contains — rank 16, 286.3 ± 27.4 nats behind the best (10.5× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -2486.8

**Hypothesis:** Refinement of `multiplicative_not_exponential_concepts`: people still build number concepts from mental primitives (a fitted cost per extra primitive, a fitted extra cost on powers, squares and cubes, Erlang-weighted intervals, the size principle and near-all-or-none commitment to the best-explaining concept). The one change is how "contains the digit d" is represented: people read numerals by place value (tens digit, ones digit), so "contains d" is not an elementary property but a disjunction, "the ones digit is d OR the tens digit is d", and it costs one more primitive than "ends in d" or "multiples of m", priced by the same per-primitive cost with no new free parameter. So when every example contains a digit (e.g. 92, 82, 27, 42), people seldom commit to "contains 2" and say no to targets that contain the digit without sharing an example's ending (23, 29, 12), as the critique reports.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/d6b6238b22d937a5/repo/_runs/cell_1/experiment2/model_loop/models/place_value_disjunctive_contains.py`

### diagnostic_attention_exemplar_similarity — rank 17, 520.4 ± 36.0 nats behind the best (14.4× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -2720.9

**Hypothesis:** People do not infer a hidden concept: they hold the examples in memory as exemplars and say yes to a target the more similar it is, on average, to those stored examples. Similarity is judged over familiar number properties (parity, primality, squares, cubes, powers of 2, multiples, final digit, decade, digits contained) and magnitude. Attention shifts to whatever the examples conspicuously share, with more attention the rarer the shared property is and, for magnitude, the more tightly the examples cluster. So after an all-prime set, a non-prime target mismatches every example on a heavily attended dimension and is rejected, while other primes are accepted.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/d6b6238b22d937a5/repo/_runs/cell_1/experiment2/model_loop/models/diagnostic_attention_exemplar_similarity.py`

### spacing_scaled_distance_gradient — rank 18, 766.4 ± 45.0 nats behind the best (17.0× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -2967.0

**Hypothesis:** People judge a target by its numerical distance to the nearest example: willingness to say yes falls off exponentially (a Shepard generalization gradient) as the target moves away from the closest example on the number line. The width of that gradient is not fixed but scales with how widely spaced the examples are, so tightly clustered examples (e.g. 60, 62, 64) produce a narrow gradient that rejects numbers a few steps away, while widely spaced examples (e.g. 16, 8, 64) produce a broad gradient that accepts numbers far from any example.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/d6b6238b22d937a5/repo/_runs/cell_1/experiment2/model_loop/models/spacing_scaled_distance_gradient.py`

## Pruned models (out of the set; narrowest margin first)

### final_digit_multiplicative_concepts — pruned (experiment1 end of experiment)

**Margin:** 8.0 nats behind costly_squares_final_digit_concepts (2.3× dse)

**Hypothesis:** Refinement of `multiplicative_not_exponential_concepts`: people still build number concepts from mental primitives. They pay a fitted cost per extra primitive and a fitted extra cost for concepts built on repeated multiplication (powers, squares, cubes), they weigh examples by the size principle, and they commit nearly all-or-none to the concept that best explains the examples. The one change, taken from `final_digit_salient_concept_search`, is that a written number's final digit is perceptually privileged: "ends in d" rules get a fitted prior discount over other compound rules, including "contains the digit d". So when every example contains some digit, people read the set as "ends in d" where they can, and say no to targets that only contain that digit somewhere else. This addresses the critique that people use "contains digit d" less than the incumbent predicts.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/d6b6238b22d937a5/repo/_runs/cell_1/experiment1/model_loop/models/pruned/final_digit_multiplicative_concepts.py`

### contains_digit_costly_concepts — pruned (experiment1 end of experiment)

**Margin:** 15.7 nats behind costly_squares_final_digit_concepts (2.2× dse)

**Hypothesis:** Refinement of `multiplicative_not_exponential_concepts`: people still build number concepts from simple primitives (a fitted cost per extra primitive, the size principle, Erlang-weighted intervals, near-all-or-none commitment to the best-explaining concept), with an extra cost on powers, squares and cubes. The one change is that "contains the digit d anywhere" is a hard, rarely entertained concept, because it ignores where the digit is written. It pays its own fitted prior cost, so people seldom commit to it. The sharpest disagreement with the best model is on sets whose examples all contain a digit d (e.g. 92, 82, 27, 42). There, people say no to targets that contain d without sharing an example's last digit (e.g. 23, 29, 12), while the best model predicts yes.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/d6b6238b22d937a5/repo/_runs/cell_1/experiment1/model_loop/models/pruned/contains_digit_costly_concepts.py`

### ascending_range_sampling_concepts — pruned (experiment1 end of experiment)

**Margin:** 28.3 nats behind costly_squares_final_digit_concepts (3.0× dse)

**Hypothesis:** People think of a range-type program as counting upward from where its range starts, so they expect the examples they have seen to be bunched toward the bottom of the range, with the rest of the range lying above them. A range that starts just below the examples and runs well above them therefore explains the examples better than one centred on them, and people generalise more to numbers a little above the examples than to numbers equally far below. Rule-type concepts (parity, multiples, final digits, with powers, squares and cubes costing extra) are judged as before.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/d6b6238b22d937a5/repo/_runs/cell_1/experiment1/model_loop/models/pruned/ascending_range_sampling_concepts.py`

### forward_counting_range_sampling — pruned (experiment1 end of experiment)

**Margin:** 28.8 nats behind costly_squares_final_digit_concepts (3.0× dse)

**Hypothesis:** People assume the hidden program produces a numerical range by counting upward from its starting point, so the examples they see are taken to be early, low members of the range rather than evenly spread across it; a range that starts near the examples and extends above them therefore explains the examples better than one extending below them, with the strength (and direction) of this counting-forward assumption fitted from data. Everything else follows the best model (primitive-built rules with a per-primitive cost and an extra cost on powers/squares/cubes, the size principle, Erlang-weighted ranges, near-all-or-none commitment), so people generalise from a set to numbers just above the examples more than to numbers just below them.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/d6b6238b22d937a5/repo/_runs/cell_1/experiment1/model_loop/models/pruned/forward_counting_range_sampling.py`

### multiplicative_not_exponential_concepts — pruned (experiment1 end of experiment)

**Margin:** 29.4 nats behind costly_squares_final_digit_concepts (3.2× dse)

**Hypothesis:** People's concept space for numbers is built on multiplication, not exponentiation: they readily entertain parity, "multiples of m", final digits, digits contained and numerical ranges, but concepts that require repeated multiplication — powers of a base, perfect squares, perfect cubes — are a priori much less accessible, carrying a fitted extra cost on top of the ordinary per-primitive simplicity cost, size principle and near-all-or-none commitment. So after 32, 4, 16 or 4, 64 people generalise to other multiples of 4 (12, 72) rather than restricting themselves to powers of 2 or 4, extending shared-divisor rules more than a learner who finds powers as natural as multiples.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/d6b6238b22d937a5/repo/_runs/cell_1/experiment1/model_loop/models/pruned/multiplicative_not_exponential_concepts.py`

### costly_power_rule_simplicity_search — pruned (experiment1 end of experiment)

**Margin:** 32.1 nats behind costly_squares_final_digit_concepts (3.2× dse)

**Hypothesis:** Refinement of `simplicity_cost_concept_search`: people still build concepts from simple mental primitives, paying a fitted cost per extra primitive, weighing examples by the size principle and committing nearly all-or-none to the best-explaining concept. The one change is that rules built on repeated multiplication — squares, cubes, powers of a number and their exception variants — are harder to conceive than other rules of the same length and pay a fitted extra prior cost, so when examples such as 32, 4, 16 fit both "powers of 2" and "multiples of 4", people favour the multiplicative rule and extend it to new multiples (12, 72) far more than the size principle predicts — addressing the critique that people extend shared-divisor rules more than the best model does.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/d6b6238b22d937a5/repo/_runs/cell_1/experiment1/model_loop/models/pruned/costly_power_rule_simplicity_search.py`

### final_digit_salient_concept_search — pruned (experiment1 end of experiment)

**Margin:** 40.7 nats behind costly_squares_final_digit_concepts (3.2× dse)

**Hypothesis:** Refinement of `erlang_interval_simplicity_search`: people still build concepts from mental primitives (a fitted cost per extra primitive, Erlang-weighted intervals, the size principle and near-all-or-none commitment to the best-explaining concept). The one change is that the final digit of a written number is a privileged, perceptually salient feature: rules of the form "ends in d" get a fitted prior discount compared with other compound rules (multiples, powers, primes, contains-a-digit). So after 62, 32, 22 or 92, 82, 27, 42 people commit to "ends in 2" instead of "even numbers" or "contains a 2", and say no to other even numbers and to the rest of the 20s.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/d6b6238b22d937a5/repo/_runs/cell_1/experiment1/model_loop/models/pruned/final_digit_salient_concept_search.py`

### notation_cheap_simplicity_search — pruned (experiment1 end of experiment)

**Margin:** 59.4 nats behind costly_squares_final_digit_concepts (4.2× dse)

**Hypothesis:** Refinement of `erlang_interval_simplicity_search`: people still build concepts from mental primitives (a fitted cost per extra primitive, the size principle, an Erlang size prior on intervals, near-all-or-none commitment to the best-explaining concept). The one change is that concepts read straight off the written numeral — "ends in 2", "contains a 7" — cost a different, fitted amount from arithmetic concepts that must be computed (primes, squares, cubes, powers, multiples of m), instead of every two-primitive rule costing the same. If digit-notation concepts are cheaper, people more readily read a set like 92, 82, 27, 42 as "ends in 2" with one stray, and reject numbers that only share a range with the examples or are merely odd/even neighbours of them.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/d6b6238b22d937a5/repo/_runs/cell_1/experiment1/model_loop/models/pruned/notation_cheap_simplicity_search.py`

### erlang_interval_simplicity_search — pruned (experiment1 end of experiment)

**Margin:** 79.7 nats behind costly_squares_final_digit_concepts (4.9× dse)

**Hypothesis:** Refinement of `simplicity_cost_concept_search`: people still build concepts from mental primitives, paying a fitted cost per extra primitive for rules, weighing examples by the size principle and committing nearly all-or-none to the best-explaining concept. The one change (taken from `tenenbaum_rules_similarity`) is the prior over numerical intervals: instead of treating every stretch of the number line as equally natural, people favour intervals of a typical width (an Erlang prior on interval size with a fitted scale), so tight ranges hugging the examples are a priori unlikely and rules win more often — predicting less "same decade / nearby number" generalisation and stricter respect for shared properties such as parity than the incumbent.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/d6b6238b22d937a5/repo/_runs/cell_1/experiment1/model_loop/models/pruned/erlang_interval_simplicity_search.py`

### simplicity_cost_concept_search — pruned (experiment1 end of experiment)

**Margin:** 88.6 nats behind costly_squares_final_digit_concepts (5.0× dse)

**Hypothesis:** People search for a concept by building it from simple mental primitives, and every extra primitive a concept needs costs them: the simplest concepts (odd, even) come to mind readily, while compound concepts (multiples of 7, ending in 2, powers of 3, a rule with an exception) are penalised in proportion to how many primitives describe them, on top of the size principle and the near-all-or-none commitment to the best-explaining concept. This complexity cost makes people fall back on broad parity rules much more than the size principle alone predicts — e.g. saying yes to any even number after 22, 62, 26, 82 rather than only to numbers ending in 2.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/d6b6238b22d937a5/repo/_runs/cell_1/experiment1/model_loop/models/pruned/simplicity_cost_concept_search.py`

### no_default_lapse_simplicity_search — pruned (experiment1 end of experiment)

**Margin:** 90.2 nats behind costly_squares_final_digit_concepts (5.1× dse)

**Hypothesis:** Refinement of `simplicity_cost_concept_search`: people still search for a concept built from simple primitives (a fitted cost per extra primitive, the size principle, near-all-or-none commitment to the best concept), and on a fraction of trials they lapse and answer without consulting the concept. The one change is what a lapse produces: instead of a coin-flip yes/no, a lapsing person falls back on a learned default answer — a fitted, typically low, base rate of saying yes (a "when in doubt, say no" bias) — so parity-violating and merely nearby targets get fewer stray yeses and repeated answers to the same item are more consistent than the coin-flip lapse predicts.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/d6b6238b22d937a5/repo/_runs/cell_1/experiment1/model_loop/models/pruned/no_default_lapse_simplicity_search.py`

### parity_salient_concept_learning — pruned (experiment1 end of experiment)

**Margin:** 104.1 nats behind costly_squares_final_digit_concepts (5.1× dse)

**Hypothesis:** Refinement of `concept_posterior_with_lapse_guessing`: people do Bayesian concept learning over the same 45 rules and all intervals of 1-100 (size principle, lapse-and-guess response stage), but parity — odd versus even — is a privileged, highly accessible concept whose prior is boosted above the other rules by a fitted salience factor. The one change is this non-uniform rule prior favouring odd/even, which lets parity survive the size principle against tighter rules and intervals, so people say yes to parity-matching targets of an all-odd or all-even set far more than the uniform-prior learner predicts.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/d6b6238b22d937a5/repo/_runs/cell_1/experiment1/model_loop/models/pruned/parity_salient_concept_learning.py`

### interval_breadth_biased_commitment — pruned (experiment1 end of experiment)

**Margin:** 105.5 nats behind costly_squares_final_digit_concepts (5.6× dse)

**Hypothesis:** Refinement of `breadth_biased_rule_commitment`: people commit nearly all-or-none to the concept (a mathematical rule or a numerical interval) that best explains the examples under the size principle, with broad rules a priori more natural in proportion to their size. The one change is that the same size-graded naturalness applies to numerical ranges: instead of all intervals being equally likely a priori, an interval's prior grows (or shrinks) with its width by a fitted breadth exponent, so if people favour broad ranges, tight "nearby number" intervals rarely win and people say yes to non-example numbers in the examples' own decade less than a flat interval prior predicts.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/d6b6238b22d937a5/repo/_runs/cell_1/experiment1/model_loop/models/pruned/interval_breadth_biased_commitment.py`

### breadth_biased_rule_commitment — pruned (experiment1 end of experiment)

**Margin:** 108.9 nats behind costly_squares_final_digit_concepts (5.6× dse)

**Hypothesis:** Refinement of `sharpened_map_concept_selection`: people still commit, nearly all-or-none, to the one or few concepts (a mathematical rule or a numerical interval) that best explain the examples under the size principle, and answer yes only if the target is in that concept. The one change is the prior over rules: rather than treating all rules as equally likely, people find broad, familiar rules (odd, even, and other large categories) a priori more natural in proportion to how many numbers they cover, with the strength of this breadth bias fitted from data — so a broad rule like parity can win over tighter rules and intervals when all examples share it, predicting stronger parity-based generalization than the incumbent, as the critique reports.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/d6b6238b22d937a5/repo/_runs/cell_1/experiment1/model_loop/models/pruned/breadth_biased_rule_commitment.py`

### serial_position_weighted_concept_search — pruned (experiment1 end of experiment)

**Margin:** 113.3 nats behind costly_squares_final_digit_concepts (5.3× dse)

**Hypothesis:** People search for the simplest concept that explains the examples (complexity-costed rules and intervals, size principle, near-all-or-none commitment), but they do not weigh the examples equally: the evidence each example contributes depends on its position in the order shown, growing or shrinking steadily from the first example to the last at a fitted rate. If the first example anchors the guess (primacy), a concept fitting the early examples survives a later example that breaks it; if the latest example dominates (recency), the concept is pulled toward what the last examples share.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/d6b6238b22d937a5/repo/_runs/cell_1/experiment1/model_loop/models/pruned/serial_position_weighted_concept_search.py`

### weakened_size_principle_sampling — pruned (experiment1 end of experiment)

**Margin:** 129.3 nats behind costly_squares_final_digit_concepts (6.2× dse)

**Hypothesis:** People do Bayesian concept learning over mathematical rules and numerical intervals, but they apply the size principle only partially: they discount the evidence that a small concept explains the examples better than a large one, as if the examples were only partly sampled from the concept rather than just happening to fall in it. This weakened size penalty keeps broad concepts such as "odd numbers", "even numbers" or a wide stretch of the number line alive, so people generalise along parity and across a region of numbers more than an ideal strong-sampling learner does.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/d6b6238b22d937a5/repo/_runs/cell_1/experiment1/model_loop/models/pruned/weakened_size_principle_sampling.py`

### confidence_criterion_decision — pruned (experiment1 end of experiment)

**Margin:** 143.9 nats behind costly_squares_final_digit_concepts (6.5× dse)

**Hypothesis:** People infer how likely the target is to belong to the hidden concept exactly as a simple Bayesian concept learner would (rules and intervals under the size principle), but they do not report that belief by probability matching: they compare it against a personal criterion of how sure they must be to say "yes", answering yes when their confidence exceeds the criterion and no otherwise, with only a little noise around the cut-off. This soft-threshold decision rule makes answers far more all-or-none than the belief itself, and shifts borderline targets toward "no" (or "yes") depending on whether the criterion sits above (or below) even odds.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/d6b6238b22d937a5/repo/_runs/cell_1/experiment1/model_loop/models/pruned/confidence_criterion_decision.py`

### sharpened_map_concept_selection — pruned (experiment1 end of experiment)

**Margin:** 151.3 nats behind costly_squares_final_digit_concepts (6.4× dse)

**Hypothesis:** People do not average over all concepts consistent with the examples; they commit, nearly all-or-none, to the one or few concepts (a mathematical rule or a tight numerical interval) that best explain the examples under the size principle, and answer yes only if the target falls in the concept they committed to. This sharpened, winner-take-most choice of concept makes answers more deterministic and leaves far less graded "nearby number" generalization than full Bayesian averaging predicts.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/d6b6238b22d937a5/repo/_runs/cell_1/experiment1/model_loop/models/pruned/sharpened_map_concept_selection.py`

### parity_restricted_range_commitment — pruned (experiment1 end of experiment)

**Margin:** 155.8 nats behind costly_squares_final_digit_concepts (6.7× dse)

**Hypothesis:** Refinement of `sharpened_map_concept_selection`: people still commit, nearly all-or-none, to the concept that best explains the examples under the size principle, but the concepts they can commit to include parity-restricted ranges ("even numbers between 16 and 64", "odd numbers from 31 to 47") alongside plain mathematical rules and plain intervals. The one change is adding these parity-by-interval conjunctions to the hypothesis space (with a fitted share of the interval prior), so that when all examples share parity people say yes to parity-matching targets inside the examples' range and no to opposite-parity near neighbours and to numbers outside the range.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/d6b6238b22d937a5/repo/_runs/cell_1/experiment1/model_loop/models/pruned/parity_restricted_range_commitment.py`

### concept_posterior_with_lapse_guessing — pruned (experiment1 end of experiment)

**Margin:** 164.7 nats behind costly_squares_final_digit_concepts (6.5× dse)

**Hypothesis:** Refinement of `bigelow_piantadosi_mixture`: people do Bayesian concept learning over the same 45 rules and all intervals of 1-100 (size principle, rule prior lambda_0), but they do not answer every trial from that posterior — on a fixed fraction of trials (a lapse) they ignore the examples and guess "yes" with a fixed base rate. The one change is this lapse-and-guess response stage, which makes answers to the same (set, target) less deterministic than the concept posterior alone implies and pulls confident predictions toward the guess rate.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/d6b6238b22d937a5/repo/_runs/cell_1/experiment1/model_loop/models/pruned/concept_posterior_with_lapse_guessing.py`

### tempered_posterior_commitment — pruned (experiment1 end of experiment)

**Margin:** 173.7 nats behind costly_squares_final_digit_concepts (6.6× dse)

**Hypothesis:** People hold a belief over candidate concepts (mathematical rules and numeric intervals, with the size principle) whose commitment differs from Bayesian averaging by a single tempering exponent: the posterior over concepts is raised to a free power and renormalised, so people either commit almost entirely to the best concept (exponent above 1) or hedge conservatively across many weaker ones (exponent below 1). It disagrees most sharply with the averaging model on sets where a rule and an interval compete, such as targets just outside the examples' range or numeric neighbours that fit no rule, because there the weight on the runner-up concepts (set by the exponent) decides the answer.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/d6b6238b22d937a5/repo/_runs/cell_1/experiment1/model_loop/models/pruned/tempered_posterior_commitment.py`

### overweighted_evidence_rules_intervals — pruned (experiment1 end of experiment)

**Margin:** 204.4 nats behind costly_squares_final_digit_concepts (7.8× dse)

**Hypothesis:** Refinement of `tenenbaum_rules_similarity`: people do Bayesian concept learning over the same mathematical rules and Erlang-weighted intervals, but they treat the examples as stronger evidence than ideal random sampling warrants — the size-principle likelihood is raised to a fitted power (an "evidence-weight" exponent), so the posterior concentrates on the tightest rules and intervals consistent with the examples. The one change is this fitted exponent (fixed at 1 in the original), which predicts less generalization to numerical neighbours and to numbers just outside the examples' range than the ideal learner, as the critique of the incumbent reports.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/d6b6238b22d937a5/repo/_runs/cell_1/experiment1/model_loop/models/pruned/overweighted_evidence_rules_intervals.py`

### range_bounded_rule_learning — pruned (experiment1 end of experiment)

**Margin:** 233.3 nats behind costly_squares_final_digit_concepts (8.1× dse)

**Hypothesis:** People never entertain an unbounded mathematical rule on its own: every concept they consider is a rule confined to a numerical range (e.g. "even numbers between 20 and 60", with "any number between 20 and 60" as the plain-interval case), and they infer which range-bounded rule produced the examples under the size principle, answering yes by how much of that inference contains the target. Because a rule bounded tightly around the examples is always a smaller, better explanation than the same rule across all of 1-100, this predicts strong use of shared properties like parity inside the examples' span but sharply less generalization of those properties to targets outside it.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/d6b6238b22d937a5/repo/_runs/cell_1/experiment1/model_loop/models/pruned/range_bounded_rule_learning.py`

### tversky_prototype_feature_contrast — pruned (experiment1 end of experiment)

**Margin:** 258.2 nats behind costly_squares_final_digit_concepts (9.2× dse)

**Hypothesis:** People judge membership by feature matching rather than by Bayesian inference over concepts: they abstract a prototype made of every property all the examples share (parity, divisibility, being a square/cube/prime/power, last digit, digits contained, decade, lying within the examples' range), and say "yes" the more of those shared properties the target has, penalised for each shared property it lacks and, more weakly, for each property it has that the prototype lacks (Tversky's contrast model of similarity). There is no size principle: a broad shared property counts as much as a narrow one.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/d6b6238b22d937a5/repo/_runs/cell_1/experiment1/model_loop/models/pruned/tversky_prototype_feature_contrast.py`

### additive_cluster_exemplar_similarity — pruned (experiment1 end of experiment)

**Margin:** 263.6 nats behind costly_squares_final_digit_concepts (9.5× dse)

**Hypothesis:** People do not infer a hidden concept; they compare the target to each stored example, one at a time, and judge how similar the two numbers are by the stored number categories they have in common (odd/even, multiples of m, squares, powers, last digit, digits contained, and stretches of the number line), with a shared small category counting far more than a shared broad one. They say yes the more the target resembles every example (a geometric-mean, all-examples-must-match comparison), so a target sharing a divisor with all the examples, or lying close to all of them, is accepted in graded fashion.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/d6b6238b22d937a5/repo/_runs/cell_1/experiment1/model_loop/models/pruned/additive_cluster_exemplar_similarity.py`

### stray_tolerant_concept_learning — pruned (experiment1 end of experiment)

**Margin:** 268.1 nats behind costly_squares_final_digit_concepts (7.1× dse)

**Hypothesis:** Refinement of bigelow_piantadosi_mixture: people do Bayesian concept learning over the same 45 rules and all intervals of 1-100 with the same grammar prior, and say yes with the posterior probability that the concept contains the target. The one change is that how often people think an example is a stray number unrelated to the concept is learned from the data rather than fixed at 10%; tolerating more strays lets a scattered set be read as a tight rule plus an outlier, so people generalise less to the examples' numerical neighbours than the incumbent predicts.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/d6b6238b22d937a5/repo/_runs/cell_1/experiment1/model_loop/models/pruned/stray_tolerant_concept_learning.py`

### bigelow_piantadosi_mixture — pruned (experiment1 end of experiment)

**Margin:** 302.0 nats behind costly_squares_final_digit_concepts (7.0× dse)

**Hypothesis:** Bayesian concept learning over 45 rules (odd, even, primes, squares, cubes, powers of 2-10, powers of 2 without 32 or with 37, multiples of 3-9, 11 and 12, ending in 0-9, containing the digit 0-9) and every interval of 1-100. A rule is chosen with probability lambda_0, and the rules are equally likely; each interval has prior (1 - lambda_0) / 100^2. Examples are sampled from the concept (the size principle), 10% of them uniformly from 1-100, and "yes" is the posterior probability that the concept contains the target.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/d6b6238b22d937a5/repo/_runs/cell_1/experiment1/model_loop/models/pruned/bigelow_piantadosi_mixture.py`

### stepwise_progression_induction — pruned (experiment1 end of experiment)

**Margin:** 331.2 nats behind costly_squares_final_digit_concepts (9.8× dse)

**Hypothesis:** People assume the hidden program is a counting procedure: it starts at some number and steps upward by a fixed stride until it stops (e.g. "from 22 to 82 in steps of 4", "from 8 to 64 in steps of 8", or "every number from 40 to 60" for a stride of 1). They infer which start, stride and stop produced the examples — preferring small strides and, by the size principle, short progressions — and say yes to a target if the inferred progressions land on it. This predicts generalisation along shared step sizes (multiples of m, numbers spaced by the examples' common difference) more than fixed rule lists do, and less generalisation to immediate neighbours (1-3 away) that fall between the steps than to numbers a full stride away.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/d6b6238b22d937a5/repo/_runs/cell_1/experiment1/model_loop/models/pruned/stepwise_progression_induction.py`

### tenenbaum_rules_similarity — pruned (experiment1 end of experiment)

**Margin:** 357.8 nats behind costly_squares_final_digit_concepts (7.4× dse)

**Hypothesis:** Bayesian concept learning over 33 mathematical rules (odd, even, squares, cubes, primes, multiples of 3-12, powers of 2-10, ending in 1-9) and every interval of 1-100. The rules share a prior fraction lambda equally; the intervals share the rest by an Erlang density on their size (scale sigma). Examples are sampled from the concept (the size principle), 10% of them uniformly from 1-100, and "yes" is the posterior probability that the concept contains the target.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/d6b6238b22d937a5/repo/_runs/cell_1/experiment1/model_loop/models/pruned/tenenbaum_rules_similarity.py`

### digit_space_exemplar_similarity — pruned (experiment1 end of experiment)

**Margin:** 359.4 nats behind costly_squares_final_digit_concepts (11.1× dse)

**Hypothesis:** People judge a target by its exemplar similarity to the examples in a digit-based representation of numbers: each number is coded by its tens digit and its ones digit, the psychological distance between the target and an example is a weighted city-block distance (a graded penalty for each step apart in tens digit plus a penalty for a differing ones digit, with fitted attention weights on the two dimensions), and similarity falls off exponentially with that distance. The probability of saying yes rises with the target's summed similarity to all the examples, so targets sharing the examples' final digit or lying in their decades are accepted in graded fashion, with no rule inference or size principle.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/d6b6238b22d937a5/repo/_runs/cell_1/experiment1/model_loop/models/pruned/digit_space_exemplar_similarity.py`
