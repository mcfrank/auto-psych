# Refinement menu

The models you may refine, other than the incumbent `iter2_candidate1`. Each entry is one mechanism: its hypothesis in full as its author stated it, how it stands, and where its source is. This is a menu, not a list of what is ruled out: a pruned model lost to the best model on the data it was scored on, by the stated margin, but its mechanism may be partly right — and a model that lost narrowly is the most promising target here. Read the source of the model you pick before you write anything.

## Live models other than the incumbent (in the set; best first)

### graded_explaining_away_by_specificity — rank 1, 4.5 ± 6.8 nats behind the best (0.7× dse: statistically tied with the best), ELPD-LOO -4140.9

**Hypothesis:** When people compare the examples, they notice the most specific property all of them share (for example, all ending in 3), and they treat broader properties that it implies (being odd, containing a 3) as by-products of it rather than as separate candidate rules. A broader rule is set aside more completely the more specific the shared property that implies it, while people otherwise infer the concept by tempered Bayesian concept learning over rules and magnitude ranges. So after sets like 63,23,43 or 93,43,83,53, same-parity numbers with a different final digit should be endorsed less than the current best model predicts, while a lone example, which allows no comparison, is unaffected.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/8a52ff3d6e8e6ef7/repo/_runs/cell_1/experiment3/model_loop/models/graded_explaining_away_by_specificity.py`

### range_evidence_with_guessing_lapse — rank 2, 8.6 ± 15.2 nats behind the best (0.6× dse: statistically tied with the best), ELPD-LOO -4145.0

**Hypothesis:** I refine `range_specific_evidence_strength`, keeping its tempered, sample-size-neglecting Bayesian concept learning over weighted rules, size-weighted ranges that have their own evidence strength, odd/even versions and fully consistent bounded rules, with comparison-based rules discounted for a lone example. The one change is a component taken from `guessing_lapse_consistent_concepts`: on some trials people do not consult their inferred concept at all and guess instead, saying "yes" at an inferred rate whatever the target is. Then the occasional "yes" to numbers that no plausible concept contains comes from guessing, not from a prior that favours very wide ranges. So the ranges people imagine, which already have their own evidence strength, should stay closer to the examples, and every target, even the examples themselves, keeps a floor and a ceiling set by guessing.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/8a52ff3d6e8e6ef7/repo/_runs/cell_1/experiment3/model_loop/models/range_evidence_with_guessing_lapse.py`

### shared_rule_explains_away_supersets — rank 3, 11.0 ± 5.2 nats behind the best (2.1× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -4147.4

**Hypothesis:** Refinement of `range_specific_evidence_strength` (the same tempered, sample-size-neglecting Bayesian concept learning over weighted rules, size-weighted ranges with their own evidence strength, odd/even versions and fully consistent bounded rules, with comparison-based rules discounted for a lone example). The one change is explaining away: when people compare two or more examples and see that all of them share a specific property (all end in 3), they treat any broader property it implies (being odd, containing a 3) as a by-product of that specific property, not as a separate candidate rule, so every rule that strictly contains another rule all the examples fit loses prior plausibility by an inferred amount (no loss recovers the incumbent). After sets such as 63,23,43 or 93,43,83,53, odd targets with a different final digit (31, 35, 39) should then be endorsed much less than the incumbent predicts, while targets with the specific property and sets without nested rules barely change.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/8a52ff3d6e8e6ef7/repo/_runs/cell_1/experiment3/model_loop/models/shared_rule_explains_away_supersets.py`

### range_specific_evidence_strength — rank 4, 16.0 ± 14.4 nats behind the best (1.1× dse: statistically tied with the best), ELPD-LOO -4152.4

**Hypothesis:** Refinement of `consistent_examples_bounded_rules`. People use the same tempered, sample-size-neglecting Bayesian concept learning over weighted rules, size-weighted ranges, their odd/even versions and bounded rules that every example must fit, and they still discount comparison-based rules when shown one example. The one change: people do not weigh the examples as equally strong evidence for every kind of concept. Once they think of a magnitude range, the size principle acts on it with its own strength, which differs from the strength used for mathematical and digit rules by an inferred amount (no difference recovers the incumbent). This separates how tightly the imagined range fits the examples from how sharply people choose among rules, so it should change how much people endorse in-range relative to out-of-range targets for pairs of examples and for sets that no rule fits fully, while judgments driven by a rule that fits fully stay nearly the same.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/8a52ff3d6e8e6ef7/repo/_runs/cell_1/experiment3/model_loop/models/range_specific_evidence_strength.py`

### range_evidence_accumulates_with_n — rank 5, 16.8 ± 14.5 nats behind the best (1.2× dse: statistically tied with the best), ELPD-LOO -4153.2

**Hypothesis:** Refinement of `range_specific_evidence_strength` (the same tempered, sample-size-neglecting Bayesian concept learning over weighted rules, size-weighted ranges with their own evidence strength, odd/even versions and fully consistent bounded rules, with comparison-based rules discounted for a lone example). The one change: sample-size neglect differs by kind of concept. People judge how tightly a magnitude range fits by how densely the examples fill it, so each further example adds evidence for a range at its own rate, which differs from the rate for rules by an inferred amount (no difference recovers the incumbent). If ranges neglect sample size less than rules, then for sets of three or four examples that no rule fits (71,61,26,81; 70,88,7,22) people should endorse targets outside the examples' span less than the incumbent predicts, while pairs and rule-driven judgments barely change.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/8a52ff3d6e8e6ef7/repo/_runs/cell_1/experiment3/model_loop/models/range_evidence_accumulates_with_n.py`

### specific_rule_explains_away_with_lapse — rank 6, 17.1 ± 11.3 nats behind the best (1.5× dse: statistically tied with the best), ELPD-LOO -4153.4

**Hypothesis:** I refine `guessing_lapse_consistent_concepts` (tempered, sample-size-neglecting Bayesian concept learning over weighted rules, size-weighted ranges, their odd/even versions and bounded rules that every example must fit, with a guessing lapse on some trials). The one change, taken from `specific_rule_preempts_supersets`: when people compare two or more examples and see that they all share a specific property (all end in 3), they explain away any broader property it implies (being odd, containing a 3) as a by-product rather than a candidate rule, so each rule that strictly contains another rule every example fits keeps only an inferred fraction of its plausibility. After sets such as 63,23,43 or 93,43,83,53, people should then rarely endorse same-parity numbers with a different final digit (31, 35, 39), less often than either parent predicts, while targets with the specific property, lone examples, and sets without nested rules stay as the guessing-lapse model predicts.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/8a52ff3d6e8e6ef7/repo/_runs/cell_1/experiment3/model_loop/models/specific_rule_explains_away_with_lapse.py`

### guessing_lapse_consistent_concepts — rank 7, 17.9 ± 15.4 nats behind the best (1.2× dse: statistically tied with the best), ELPD-LOO -4154.3

**Hypothesis:** I refine `consistent_examples_bounded_rules` (tempered, sample-size-neglecting Bayesian concept learning over weighted rules, size-weighted ranges, their odd/even versions, and bounded rules that every example must fit). The one change is a component taken from `neglect_evidence_guessing_lapse`: on some trials people do not consult their inferred concept at all. On those trials they guess, saying "yes" at an inferred rate whatever the target is. In the parent model the occasional "yes" to numbers that no plausible concept contains could only come from a prior that favours very wide ranges. With guessing to account for it, people's ranges should stay close to the examples: when no rule fits, numbers inside the examples' span get relatively more endorsement and numbers outside it less, and every target, even the examples themselves, keeps a floor and ceiling set by guessing.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/8a52ff3d6e8e6ef7/repo/_runs/cell_1/experiment3/model_loop/models/guessing_lapse_consistent_concepts.py`

### consistent_examples_bounded_rules — rank 8, 27.5 ± 14.5 nats behind the best (1.9× dse: statistically tied with the best), ELPD-LOO -4163.9

**Hypothesis:** Refinement of `comparison_dependent_concepts` (the same tempered, sample-size-neglecting Bayesian concept learning over the 45 weighted rules, size-weighted intervals, their odd/even versions and range-bounded rules, with digit rules and bounded rules discounted when only one example is shown). The one change takes that model's own premise further. A rule confined to the examples' span ("ends in 3 between 48 and 63") is built by comparing the examples and seeing that they share the rule, so people entertain it only when every example fits the rule. A bounded rule that one of the examples breaks is never considered, rather than kept with that example written off as a stray, and no parameter is added. People should then drop a rule that most examples share but one breaks, and fall back on the examples' range or on whole rules. In-range numbers lacking the majority pattern (48,63,53 → 49, 51) should be endorsed more than the incumbent predicts, while sets that a rule fits fully are unchanged.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/8a52ff3d6e8e6ef7/repo/_runs/cell_1/experiment3/model_loop/models/consistent_examples_bounded_rules.py`

### stray_rate_inferred_concepts — rank 9, 39.4 ± 16.0 nats behind the best (2.5× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -4175.8

**Hypothesis:** Refinement of `comparison_dependent_concepts`. It keeps the same tempered, sample-size-neglecting Bayesian concept learning over the 45 weighted rules, size-weighted intervals, their odd/even versions and range-bounded rules, with digit and bounded rules discounted for a lone example. The one change: people do not assume a fixed 10% of the examples are random strays. How likely they think a stray example is gets inferred. If people think the program's examples are rarely strays, any rule that one example breaks (whole, bounded or parity-restricted) loses nearly all belief, and that belief moves to the ranges that cover every example. In-range numbers should then be endorsed more than either earlier model predicts for sets no rule fully fits and for pairs of examples, and less outside the range.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/8a52ff3d6e8e6ef7/repo/_runs/cell_1/experiment3/model_loop/models/stray_rate_inferred_concepts.py`

### hundreds_chart_block_concepts — rank 10, 65.7 ± 26.8 nats behind the best (2.4× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -4202.1

**Hypothesis:** People picture the numbers 1-100 laid out as a ten-by-ten hundreds chart (rows are decades, columns are final digits), and the "magnitude" concepts they entertain are rectangular blocks of that chart rather than stretches of the number line. They weigh these blocks, together with familiar mathematical and digit rules, by how small a concept can still cover the examples (the size principle), and say "yes" to a target with their belief that the concept contains it. So examples sharing a final digit are read as a narrow column block that excludes other final digits even of the same parity, and when no rule fits, people endorse in-range targets only if the target's final digit falls within the span of the examples' final digits.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/8a52ff3d6e8e6ef7/repo/_runs/cell_1/experiment3/model_loop/models/hundreds_chart_block_concepts.py`

### most_specific_noticed_rule — rank 11, 234.8 ± 41.2 nats behind the best (5.7× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -4371.2

**Hypothesis:** People do not weigh many candidate concepts at once; they search for a single rule and stop at the first one they notice. Each familiar property that every example shares (a parity, multiples of some number, a special number kind such as squares, powers or primes, a final digit, a contained digit) is noticed with a probability that depends on its kind, and among the shared properties they notice people adopt the most specific one and say "yes" to targets that have it. Only when they notice no shared property do they fall back on the stretch of the number line the examples span, endorsing targets inside it and fewer the farther a target lies outside it, so sets with no common rule give little generalization beyond their range.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/8a52ff3d6e8e6ef7/repo/_runs/cell_1/experiment3/model_loop/models/most_specific_noticed_rule.py`

### central_tendency_feature_prototype — rank 12, 285.9 ± 39.2 nats behind the best (7.3× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -4422.3

**Hypothesis:** People do not infer a rule. They boil the examples down to one prototype, which records how often each familiar number property (odd/even, prime, square, cube, power, multiple of k, final digit, digit contained) occurs among the examples and where the examples sit on average on the number line. They say "yes" to a target according to its similarity to that prototype. A property counts toward the comparison more steeply the more consistently the examples share it, and the target loses similarity for the share of that weight that falls on properties it lacks and for its distance from the examples' central magnitude. This predicts that when no property is shared by every example, generalization rests on closeness to the examples' centre, so targets between the examples are endorsed more than equally related targets outside their span, while a property shared by all the examples dominates the judgment.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/8a52ff3d6e8e6ef7/repo/_runs/cell_1/experiment3/model_loop/models/central_tendency_feature_prototype.py`

## Pruned models (out of the set; narrowest margin first)

### comparison_dependent_concepts — pruned (experiment2 end of experiment)

**Margin:** 22.5 nats behind consistent_examples_bounded_rules (4.5× dse)

**Hypothesis:** Refinement of `digit_rules_need_comparison` (tempered, sample-size-neglecting Bayesian concept learning over the 45 weighted rules, size-weighted intervals, their odd/even versions and range-bounded rules, with digit rules discounted when only one example is shown). The one change is to say why the discount happens. Any concept that is defined by a relation among the examples can be noticed only by comparing examples. That covers a shared digit pattern and also a rule confined to the span the examples cover ("multiples of 6 between 4 and 9"). So when a single example is shown, range-bounded rules lose prior plausibility by the same inferred amount as digit rules, and no new parameter is added. People should then generalize a lone example to the whole of its arithmetic rule, including far members (6 → 18, 24; 11 → 55, 99; 25 → 100; 10 → 80), more than the incumbent allows, while judgments about sets of two or more examples are unchanged.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/8a52ff3d6e8e6ef7/repo/_runs/cell_1/experiment2/model_loop/models/pruned/comparison_dependent_concepts.py`

### monotone_order_range_prior_shift — pruned (experiment1 end of experiment)

**Margin:** 23.9 nats behind extrapolation_verification_failure (3.0× dse)

**Hypothesis:** Refinement of `serial_position_weighted_evidence`: people do the same tempered, serial-position-weighted Bayesian concept learning over the 45 individually weighted rules and all intervals of 1-100, but when three or more examples are shown in sorted (steadily rising or steadily falling) order, they read the list as a progression along the number line and so shift prior belief from mathematical/digit rules toward magnitude-range concepts, by an inferred amount. The one change is this order-induced shift in the rule-versus-range prior (zero shift recovers the original model); it predicts that a sorted list such as 62, 32, 22 loosens the "ends in 2" rule, so in-range or nearby numbers lacking the shared digit are endorsed more than when the same kind of set is shown in scrambled order (e.g. 63, 23, 43).

**Source:** `/scr/wurgaft/auto-psych/agent_trees/8a52ff3d6e8e6ef7/repo/_runs/cell_1/experiment1/model_loop/models/pruned/monotone_order_range_prior_shift.py`

### parity_restricted_concepts — pruned (experiment1 end of experiment)

**Margin:** 28.0 nats behind extrapolation_verification_failure (2.4× dse)

**Hypothesis:** Refinement of `tempered_size_principle_independent`: people do the same tempered-evidence Bayesian concept learning over the 45 individually weighted rules and all intervals of 1-100, but parity is a basic feature they can conjoin with any concept, so they also entertain the odd-only and even-only version of each rule and interval (e.g. "odd multiples of 3", "even numbers between 4 and 96"), which together receive an inferred share of prior belief. The one change is adding these parity-restricted concepts (share 0 recovers the incumbent); because such a concept is half the size, the size principle favours it whenever all examples share a parity, so people reject opposite-parity targets — including far, weakly supported ones outside the examples' range — more firmly than the incumbent predicts.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/8a52ff3d6e8e6ef7/repo/_runs/cell_1/experiment1/model_loop/models/pruned/parity_restricted_concepts.py`

### decade_place_value_concepts — pruned (experiment1 end of experiment)

**Margin:** 28.5 nats behind extrapolation_verification_failure (2.6× dse)

**Hypothesis:** Refinement of `parity_restricted_concepts`: people do the same tempered-evidence Bayesian concept learning over the 45 weighted rules, all intervals of 1-100, and their odd-only and even-only versions. The one change is that people also read numbers by place value, so they entertain decade concepts: "numbers whose tens digit is one of these", e.g. "numbers in the 20s, 30s or 60s". These concepts get an inferred share of prior belief (share 0 recovers the incumbent), and the size principle favours the smallest one, the decades the examples occupy. This predicts that a target with a different final digit is endorsed more when an example sits in its decade (62,32,22 → 25; 73,33,3 → 36) than when none does (63,23,43 → 35), whatever order the examples are shown in.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/8a52ff3d6e8e6ef7/repo/_runs/cell_1/experiment1/model_loop/models/pruned/decade_place_value_concepts.py`

### serial_position_weighted_evidence — pruned (experiment1 end of experiment)

**Margin:** 29.0 nats behind extrapolation_verification_failure (3.7× dse)

**Hypothesis:** Refinement of `tempered_size_principle_independent`: people do the same tempered Bayesian concept learning over the 45 individually weighted rules and all intervals of 1-100, but they do not weigh every example equally as evidence — how much an example counts depends on its serial position in the list, with evidence weight shifting smoothly toward the first examples (primacy) or the last ones (recency) by an inferred serial-position gradient, while the total evidence strength stays the same. The one change is this order-dependent weighting (a zero gradient recovers the incumbent); it predicts that the same examples shown in a different order yield different generalization, and in particular that an example that breaks the pattern of the others (e.g. an odd number among evens) undermines the tight concept more when it sits at the favoured end of the list.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/8a52ff3d6e8e6ef7/repo/_runs/cell_1/experiment1/model_loop/models/pruned/serial_position_weighted_evidence.py`

### comparison_gated_sized_bounded_rules — pruned (experiment2 end of experiment)

**Margin:** 39.6 nats behind consistent_examples_bounded_rules (5.3× dse)

**Hypothesis:** Refinement of `digit_neglect_sized_bounded_rules`. People do tempered, sample-size-neglecting Bayesian concept learning over the 45 weighted rules, size-weighted intervals, their odd/even versions, and range-bounded rules whose prior grows with how many members they keep. In that model, digit rules are already discounted when only one example is shown. The one change, grafted from `comparison_dependent_concepts`: a rule confined to the span the examples cover also needs a comparison between examples, so with a lone example range-bounded rules lose prior plausibility by the same inferred discount as digit rules. No parameter is added. People should therefore generalize a lone example to the whole of its arithmetic rule, including far members (6 → 18, 24; 25 → 100; 11 → 99), more than either parent allows. For sets of two or more examples, a sized range-bounded reading is kept.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/8a52ff3d6e8e6ef7/repo/_runs/cell_1/experiment2/model_loop/models/pruned/comparison_gated_sized_bounded_rules.py`

### digit_rules_need_comparison — pruned (experiment2 end of experiment)

**Margin:** 43.9 nats behind consistent_examples_bounded_rules (4.8× dse)

**Hypothesis:** Refinement of `range_prior_sample_size_neglect` (the same tempered, sample-size-neglecting Bayesian concept learning over the 45 weighted rules, size-weighted intervals of 1-100, their odd/even versions and range-bounded rules). The one change: a digit pattern ("ends in 6", "contains a 1") is something people notice only by comparing examples, so when a single example is shown, digit-based rules hardly come to mind, and their prior is discounted by an inferred amount (zero recovers the incumbent). Belief then moves to the lone example's arithmetic rules. People should generalize a lone example to its multiples (6 → 12, 18; 11 → 55, 66; 10 → 80) more than the incumbent allows, and endorse numbers that share only its digits (14 → 74, 94) less. Judgments about sets of two or more examples are unchanged.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/8a52ff3d6e8e6ef7/repo/_runs/cell_1/experiment2/model_loop/models/pruned/digit_rules_need_comparison.py`

### place_value_context_exemplar — pruned (experiment1 end of experiment)

**Margin:** 56.6 nats behind extrapolation_verification_failure (2.5× dse)

**Hypothesis:** People do not infer a rule; they judge a target by its summed similarity to the stored examples, as in the context (exemplar) model of classification, where each number is represented by its place-value digits — its tens digit and its units digit — together with its parity and its magnitude, and similarity to an example shrinks multiplicatively with each mismatching dimension, with an inferred attention weight per dimension. "Yes" grows with the target's total similarity to the examples relative to an internal criterion, so a target that shares its tens digit with one example (e.g. 25 after 62, 32, 22) gets partial endorsement even when it breaks the examples' shared final digit, whereas the same target is rejected when no example lies in its decade.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/8a52ff3d6e8e6ef7/repo/_runs/cell_1/experiment1/model_loop/models/pruned/place_value_context_exemplar.py`

### digit_neglect_sized_bounded_rules — pruned (experiment2 end of experiment)

**Margin:** 57.4 nats behind consistent_examples_bounded_rules (6.2× dse)

**Hypothesis:** Refinement of `range_sized_bounded_rules` (tempered, sample-size-neglecting Bayesian concept learning over the 45 weighted rules, size-weighted intervals, their odd/even versions, and range-bounded rules whose prior grows with how many members they keep). The one change, grafted from `digit_rules_need_comparison`: a digit pattern ("ends in 6", "contains a 1") is noticed only by comparing examples, so when a single example is shown the digit-based rules hardly come to mind and their prior is discounted by an inferred amount (zero recovers `range_sized_bounded_rules`). The sized range prior already makes tiny bounded rules around a lone example implausible, so the belief taken from digit rules moves to the example's whole arithmetic rules. People should then generalize a lone example to its multiples (6 → 12, 18; 11 → 55, 66; 25 → 50, 100) more than either parent model allows, and endorse its digit neighbours (14 → 74) less, while judgments about sets of two or more examples are unchanged.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/8a52ff3d6e8e6ef7/repo/_runs/cell_1/experiment2/model_loop/models/pruned/digit_neglect_sized_bounded_rules.py`

### range_sized_bounded_rules — pruned (experiment2 end of experiment)

**Margin:** 61.5 nats behind consistent_examples_bounded_rules (4.7× dse)

**Hypothesis:** Refinement of `range_prior_sample_size_neglect` (tempered Bayesian concept learning with sample-size neglect over 45 weighted rules, size-weighted intervals of 1-100, their odd/even versions and range-bounded rules). The one change is that the range-size prior people hold applies to every concept defined by a range, so it applies to a rule confined to a range ("multiples of 11 between 5 and 15") as it does to a plain interval. A bounded rule's prior plausibility grows by the same inferred power of how many members it keeps, rather than being equal for every range, and no new parameter is added. Tiny bounded versions of a rule, such as {11} or {25}, then seem implausible. From a lone example people should generalize to the whole rule it belongs to (11 → 55, 66; 25 → 50, 100), while sets of several examples keep a range-bounded reading.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/8a52ff3d6e8e6ef7/repo/_runs/cell_1/experiment2/model_loop/models/pruned/range_sized_bounded_rules.py`

### range_prior_sample_size_neglect — pruned (experiment2 end of experiment)

**Margin:** 62.8 nats behind consistent_examples_bounded_rules (4.4× dse)

**Hypothesis:** Refinement of `range_size_prior_concepts`: people do the same tempered-evidence Bayesian concept learning over the 45 weighted rules, all intervals of 1-100 (with a prior that grows as a power of a range's size), their odd-only and even-only versions, and range-bounded rules. The one change is taken from `sample_size_neglect_evidence`: people neglect sample size, so each example's evidential weight shrinks as more examples are shown. The total weight of the examples grows less than in proportion to their number (no neglect recovers the incumbent). This predicts that a lone example is weighed heavily enough to pin down the small rule it belongs to (6 → 12, 18; 11 → 55, 66; 25 → 50, 100), so its members are endorsed far more than the incumbent allows, while larger sets sharpen generalization less than an ideal learner's would.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/8a52ff3d6e8e6ef7/repo/_runs/cell_1/experiment2/model_loop/models/pruned/range_prior_sample_size_neglect.py`

### range_size_prior_concepts — pruned (experiment2 end of experiment)

**Margin:** 69.3 nats behind consistent_examples_bounded_rules (4.7× dse)

**Hypothesis:** Refinement of `range_bounded_parity_concepts`: people do the same tempered-evidence Bayesian concept learning over the 45 weighted rules, all intervals of 1-100, their odd-only and even-only versions, and range-bounded rules, but they do not think every magnitude range is equally plausible a priori — a tiny range such as "numbers from 10 to 12" seems an unlikely thing for a program to produce, so a range's prior plausibility grows (or shrinks) as an inferred power of its size (power 0 recovers the incumbent). The one change is this size-dependent prior on ranges; because the size principle favours tiny ranges most when there is only one example, a broad-range prior moves belief from tiny ranges around a lone example to the rules it belongs to, so people generalize from a single example (10 → 80, 11 → 55) much more than the incumbent allows, while several examples still pin down a tight concept.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/8a52ff3d6e8e6ef7/repo/_runs/cell_1/experiment2/model_loop/models/pruned/range_size_prior_concepts.py`

### open_multiples_guessing_lapse — pruned (experiment2 end of experiment)

**Margin:** 70.7 nats behind consistent_examples_bounded_rules (4.6× dse)

**Hypothesis:** Refinement of `open_ended_multiples_concepts`: people do the same tempered-evidence Bayesian concept learning over the 45 weighted rules, size-weighted intervals of 1-100, their odd-only and even-only versions, range-bounded rules and the open-ended "multiples of k" family (k from 13 to 50). The one change, taken from `neglect_evidence_guessing_lapse`, is that people do not consult this concept on every trial: at an inferred lapse rate they skip the inference and give a biased guess, answering "yes" at an inferred guessing rate whatever the target is (a lapse rate of 0 recovers the original model). Because the guessing lapse, not weak evidence, accounts for the rejected examples and the floor of yeses, the concept inference itself can be sharper, so members of the examples' rule (including the lone example's own multiples, 25 → 50, 100) are endorsed more and clear non-members less, apart from the guessing floor.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/8a52ff3d6e8e6ef7/repo/_runs/cell_1/experiment2/model_loop/models/pruned/open_multiples_guessing_lapse.py`

### open_ended_multiples_concepts — pruned (experiment2 end of experiment)

**Margin:** 71.5 nats behind consistent_examples_bounded_rules (4.8× dse)

**Hypothesis:** People do the same tempered-evidence Bayesian concept learning as `range_size_prior_concepts` (45 weighted rules, size-weighted intervals of 1-100, their odd-only and even-only versions, and range-bounded rules), but their space of concepts also includes an open-ended family of "multiples of k" concepts for larger steps k from 13 to 50 (e.g. multiples of 25, of 33, of 14), beyond the few small multiples in the fixed rule list. This family gets an inferred share of prior belief, and within it a step's prior plausibility falls off in inverse proportion to k (larger steps come to mind less easily). Because the size principle strongly favours such a small concept when it contains the examples, people generalize a lone example to its own multiples (25 → 50, 75, 100; 33 → 66, 99; 14 → 28, 42) much more than the fixed rule list allows.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/8a52ff3d6e8e6ef7/repo/_runs/cell_1/experiment2/model_loop/models/pruned/open_ended_multiples_concepts.py`

### tempered_size_principle_independent — pruned (experiment1 end of experiment)

**Margin:** 74.2 nats behind extrapolation_verification_failure (5.4× dse)

**Hypothesis:** Refinement of `bigelow_piantadosi_independent`: people do the same Bayesian concept learning over 45 rules (each with its own prior weight) and all intervals, but they under-trust the size principle — they treat the examples as weaker evidence than a fully rational learner would, as if the likelihood of the examples were tempered by an inferred exponent (sampling-assumption strength), so their posterior is less concentrated on the smallest consistent rule. The one change is this free evidence-strength exponent (1 recovers the original model); it predicts lower endorsement of minimal-rule members and graded endorsement among them, because members also covered by competing concepts (nearby intervals, other rules) keep more support.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/8a52ff3d6e8e6ef7/repo/_runs/cell_1/experiment1/model_loop/models/pruned/tempered_size_principle_independent.py`

### sampled_hypothesis_majority_vote — pruned (experiment1 end of experiment)

**Margin:** 77.2 nats behind extrapolation_verification_failure (5.0× dse)

**Hypothesis:** Refinement of tempered_size_principle_independent: people do the same tempered Bayesian concept learning over the 45 rules and all intervals, but they cannot hold the whole posterior in mind — they bring a small number of candidate concepts to mind by sampling them from their posterior and answer "yes" if the majority of those sampled concepts contain the target. The one change is this limited-sample majority vote (with an inferred number of samples; one sample recovers the original model), which makes answers more all-or-none than the posterior itself: likely members are endorsed more consistently and weakly supported targets, such as numbers beyond the range of the examples, are endorsed less.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/8a52ff3d6e8e6ef7/repo/_runs/cell_1/experiment1/model_loop/models/pruned/sampled_hypothesis_majority_vote.py`

### neglect_evidence_guessing_lapse — pruned (experiment2 end of experiment)

**Margin:** 79.5 nats behind consistent_examples_bounded_rules (5.1× dse)

**Hypothesis:** Refinement of `sample_size_neglect_evidence`: people do the same tempered Bayesian concept learning with sample-size neglect (45 weighted rules, all intervals, their odd/even versions and range-bounded rules, with each extra example adding less evidence), but they do not consult that concept on every trial. On some trials, at an inferred lapse rate, they skip the inference and answer by a biased guess, saying "yes" at an inferred guessing rate whatever the target is. The one change is this guessing lapse (a lapse rate of 0 recovers the original model). It predicts that even targets that are themselves examples, or that clearly belong to the inferred concept, are sometimes rejected (examples get about 0.7–0.9 yes, not 1), and that clear non-members still get a floor of yeses.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/8a52ff3d6e8e6ef7/repo/_runs/cell_1/experiment2/model_loop/models/pruned/neglect_evidence_guessing_lapse.py`

### sampled_concepts_criterion_vote — pruned (experiment1 end of experiment)

**Margin:** 79.8 nats behind extrapolation_verification_failure (4.9× dse)

**Hypothesis:** Refinement of `sampled_hypothesis_majority_vote`: people do the same tempered Bayesian concept learning over the 45 individually weighted rules and all intervals, and answer by bringing a few candidate concepts to mind by sampling them from their posterior — but they say "yes" only if a criterion number of the sampled concepts contains the target, and that criterion is inferred rather than fixed at a simple majority (equivalently, they set how many sampled concepts must favour the target and how many may go against it). The one change is this free decision criterion (equal counts recover majority vote, a single sample recovers the incumbent); a demanding criterion predicts that targets supported only by a minority of concepts — such as numbers far outside the examples' range that fit no small rule — are rejected more firmly than majority vote allows, while targets in the dominant concept are still endorsed.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/8a52ff3d6e8e6ef7/repo/_runs/cell_1/experiment1/model_loop/models/pruned/sampled_concepts_criterion_vote.py`

### single_example_rule_reading — pruned (experiment2 end of experiment)

**Margin:** 82.8 nats behind consistent_examples_bounded_rules (4.6× dse)

**Hypothesis:** Refinement of `parity_restricted_concepts` (people do tempered-evidence Bayesian concept learning over the 45 weighted rules, all intervals of 1-100, and their odd-only and even-only versions). The one change: a magnitude range is a concept defined by its endpoints, so a single example cannot suggest one — when only one example is shown, people read it as an instance of a mathematical or digit property, and prior belief shifts from range concepts to rules by an inferred amount (zero shift recovers the original model). This predicts that from a lone example people endorse the members of the rules that contain it broadly (11 → 22, 55, 99; 10 → 60, 90) rather than confining belief to a tight range around the example, while judgments about sets of two or more examples are unchanged.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/8a52ff3d6e8e6ef7/repo/_runs/cell_1/experiment2/model_loop/models/pruned/single_example_rule_reading.py`

### sample_size_neglect_evidence — pruned (experiment2 end of experiment)

**Margin:** 83.5 nats behind consistent_examples_bounded_rules (5.6× dse)

**Hypothesis:** People do the same tempered Bayesian concept learning as the best model (45 weighted rules, all intervals, their odd/even versions and range-bounded rules), with one distortion: they neglect sample size, so the total weight they give the examples grows less than in proportion to how many examples they saw — the first example counts heavily, and each further example adds less. This predicts that a single example already pins down its most salient rule (e.g. 10 → multiples of 10), producing many yeses for that rule's members, while adding more examples sharpens generalization far less than an ideal learner would.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/8a52ff3d6e8e6ef7/repo/_runs/cell_1/experiment2/model_loop/models/pruned/sample_size_neglect_evidence.py`

### compact_interval_tempered_bayes — pruned (experiment1 end of experiment)

**Margin:** 83.9 nats behind extrapolation_verification_failure (5.1× dse)

**Hypothesis:** Refinement of `tempered_size_principle_independent`: people do the same tempered-evidence Bayesian concept learning over the 45 rules (each with its own prior weight) and all intervals of 1-100, but they do not consider all intervals equally plausible a priori — they prefer compact magnitude ranges, with an Erlang prior over interval size whose scale is inferred (the one change, taken from `tenenbaum_rules_similarity`). Because the tempered evidence no longer spreads interval belief onto wide ranges, people confine magnitude-based generalization close to the [min, max] range of the examples and endorse targets outside it less than the incumbent predicts.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/8a52ff3d6e8e6ef7/repo/_runs/cell_1/experiment1/model_loop/models/pruned/compact_interval_tempered_bayes.py`

### hierarchical_tempered_evidence — pruned (experiment1 end of experiment)

**Margin:** 88.0 nats behind extrapolation_verification_failure (5.7× dse)

**Hypothesis:** Refinement of `tempered_size_principle_independent`: people under-trust the examples only when deciding *which kind* of concept and which rule produced them, but once they entertain a magnitude range they apply the size principle at full strength, so the range they imagine stays tight around the examples. The one change is where the inferred evidence-strength exponent applies: it tempers the evidence for each rule and for the interval family as a whole, while belief among intervals follows the untempered size principle; this predicts firmer rejection of targets far outside the examples' range that no small rule covers, while keeping graded endorsement of rule members.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/8a52ff3d6e8e6ef7/repo/_runs/cell_1/experiment1/model_loop/models/pruned/hierarchical_tempered_evidence.py`

### size_penalized_interval_prior — pruned (experiment1 end of experiment)

**Margin:** 90.1 nats behind extrapolation_verification_failure (5.4× dse)

**Hypothesis:** Refinement of `tempered_size_principle_independent`: people do the same tempered Bayesian concept learning over 45 individually weighted rules and all intervals of 1-100, but a priori they expect a magnitude-based concept to be a narrow range rather than treating all intervals as equally likely — their prior on an interval falls off as a power of its width, with the steepness inferred. The one change is this size-penalizing interval prior (steepness 0 recovers a uniform interval prior); it predicts that people confine generalization more tightly to the range spanned by the examples, and endorse targets outside that range less, than the incumbent allows.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/8a52ff3d6e8e6ef7/repo/_runs/cell_1/experiment1/model_loop/models/pruned/size_penalized_interval_prior.py`

### weak_sampling_hypothesis_elimination — pruned (experiment2 end of experiment)

**Margin:** 94.2 nats behind consistent_examples_bounded_rules (4.2× dse)

**Hypothesis:** People learn the hidden concept by eliminating hypotheses rather than by weighing how typical the examples are: they keep every rule (one of 45 mathematical or digit rules, each with its own prior plausibility) or range of numbers that is consistent with the examples, treating an example that does not fit only as a possible stray, and they answer "yes" with their total belief in the surviving concepts that contain the target. Because there is no size principle (weak sampling), a small concept gains nothing over a broad one that also fits, so people generalize broadly — especially from a single example — and adding consistent examples does not sharpen belief toward the smallest rule.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/8a52ff3d6e8e6ef7/repo/_runs/cell_1/experiment2/model_loop/models/pruned/weak_sampling_hypothesis_elimination.py`

### range_bounded_parity_concepts — pruned (experiment2 end of experiment)

**Margin:** 95.7 nats behind consistent_examples_bounded_rules (6.0× dse)

**Hypothesis:** Refinement of `parity_restricted_concepts`: people do the same tempered-evidence Bayesian concept learning over the 45 weighted rules, all intervals of 1-100, and their odd-only and even-only versions. The one change is a single added component, taken from `bounded_range_rule_learning`: people can also restrict a rule to a range of numbers that covers the examples (e.g. "multiples of 3 between 20 and 100"), and these range-bounded rules get an inferred share of the prior for rules (share 0 recovers `parity_restricted_concepts`). A bounded rule is smaller than the full rule, so the size principle favours it. This predicts that people endorse members of the examples' rule that lie outside the examples' min-max range less than members inside it, and that this gap is larger than the rule/interval mixture allows (63,27,99,33 → 3 below 45).

**Source:** `/scr/wurgaft/auto-psych/agent_trees/8a52ff3d6e8e6ef7/repo/_runs/cell_1/experiment2/model_loop/models/pruned/range_bounded_parity_concepts.py`

### conservative_evidence_concept_learning — pruned (experiment1 end of experiment)

**Margin:** 111.2 nats behind extrapolation_verification_failure (4.9× dse)

**Hypothesis:** Refinement of `bigelow_piantadosi_mixture`: people do the same Bayesian concept learning over the 45 rules and all intervals of 1-100 with the size principle, but they under-weight the evidence of the examples, as if each example counted for only a fraction of a full observation (a conservative, tempered likelihood). The one change is a free evidence-weight exponent on the likelihood of the examples, so confidence in the smallest consistent rule grows more slowly with examples and more posterior stays on broader rules and intervals — making learners less sure of the winning rule than the ideal learner.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/8a52ff3d6e8e6ef7/repo/_runs/cell_1/experiment1/model_loop/models/pruned/conservative_evidence_concept_learning.py`

### domain_specific_evidence_weight — pruned (experiment1 end of experiment)

**Margin:** 117.5 nats behind extrapolation_verification_failure (5.1× dse)

**Hypothesis:** Refinement of `conservative_evidence_concept_learning`: people do the same Bayesian concept learning over the 45 equally weighted rules and all intervals of 1-100 with the size principle, but how strongly they weigh the examples as evidence depends on the kind of concept — they are conservative about mathematical rules yet apply the size principle sharply to magnitude ranges, so once they think "a range of numbers" they confine it tightly to the span of the examples. The one change is splitting the single evidence-weight exponent into one for rule hypotheses and one for interval hypotheses; it predicts less endorsement of targets just outside the examples' [min, max] range (relative to inside it) than a single shared tempering allows, while keeping graded endorsement of rule members.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/8a52ff3d6e8e6ef7/repo/_runs/cell_1/experiment1/model_loop/models/pruned/domain_specific_evidence_weight.py`

### arithmetic_relation_exemplar — pruned (experiment2 end of experiment)

**Margin:** 121.7 nats behind consistent_examples_bounded_rules (4.0× dse)

**Hypothesis:** People do not infer a hidden rule; they judge a target by its similarity to the stored examples, as in an exemplar (context) model, where two numbers are similar to the extent that they stand in familiar arithmetic relations — one is a multiple of the other, they share a common factor of 3 or more, they end in the same digit, they have the same parity — and each relation the pair lacks shrinks their similarity by an inferred factor. The target's average similarity to the examples, against an internal criterion, sets the chance of "yes", so a lone example is generalized to its own multiples (6 → 18, 25 → 100, 11 → 55) because each such target is arithmetically close to it, while numbers sharing none of these relations with any example are rejected.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/8a52ff3d6e8e6ef7/repo/_runs/cell_1/experiment2/model_loop/models/pruned/arithmetic_relation_exemplar.py`

### log_odds_belief_readout — pruned (experiment1 end of experiment)

**Margin:** 123.8 nats behind extrapolation_verification_failure (5.4× dse)

**Hypothesis:** Refinement of bigelow_piantadosi_mixture: people do the same Bayesian concept learning over the 45 rules and all intervals of 1-100 with the size principle, but they do not answer "yes" with probability equal to their belief that the concept contains the target; instead they read out that belief through a distorted, linear-in-log-odds probability weighting (as in judgments of probability and decisions under risk), which compresses confident beliefs toward the middle and shifts them by a response bias. The one change is this readout: confident beliefs that a minimal-rule member is in the concept are endorsed less than certainly, and confident beliefs that a number is out are rejected less than certainly.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/8a52ff3d6e8e6ef7/repo/_runs/cell_1/experiment1/model_loop/models/pruned/log_odds_belief_readout.py`

### belief_criterion_threshold — pruned (experiment1 end of experiment)

**Margin:** 127.8 nats behind extrapolation_verification_failure (5.5× dse)

**Hypothesis:** People do the simplest Bayesian concept learning (45 equally likely rules and all intervals of 1-100, with the size principle), but they do not answer "yes" with probability equal to their belief; instead they compare their belief that the target is in the concept against an internal criterion of how sure they must be, and say "yes" only when the belief exceeds that criterion, with the comparison blurred by Gaussian decision noise (criterion and noise inferred). Because the criterion acts on the probability itself, targets with middling belief — such as far-away numbers supported only by broad rules or wide intervals — are rejected more firmly than belief alone predicts, while targets the learner is fairly sure of are endorsed nearly always.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/8a52ff3d6e8e6ef7/repo/_runs/cell_1/experiment1/model_loop/models/pruned/belief_criterion_threshold.py`

### pedagogical_sampling_concepts — pruned (experiment2 end of experiment)

**Margin:** 131.5 nats behind consistent_examples_bounded_rules (6.5× dse)

**Hypothesis:** People assume the examples were chosen by a helpful teacher, not sampled at random: the program picked numbers that would best lead a learner to its concept (pedagogical sampling). So an example counts as evidence for a concept, whether a mathematical rule or a range, in proportion to how strongly that example on its own would make a naive learner think of that concept, compared with the other numbers the teacher could have shown; people then answer "yes" with their resulting belief that the concept contains the target. This predicts that concepts for which the shown examples would be poor, unrepresentative teaching choices lose belief even when they contain the examples, unlike under the size principle, where any concept that contains the examples gains in proportion to how small it is.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/8a52ff3d6e8e6ef7/repo/_runs/cell_1/experiment2/model_loop/models/pruned/pedagogical_sampling_concepts.py`

### feature_overlap_similarity — pruned (experiment1 end of experiment)

**Margin:** 138.2 nats behind extrapolation_verification_failure (5.2× dse)

**Hypothesis:** People do not infer a single hidden rule; instead they judge the target by its featural similarity to the examples they were shown. Each number is represented by its properties (parity, primality, being a square/cube/power, divisibility, final digit, digits it contains, and which decade it lies in), and the more properties the target shares with each example on average (feature overlap relative to the properties either has), the more likely people say yes — so endorsement is graded even among numbers that fit a rule, and partial matches such as a shared final digit or nearby magnitude still earn some yeses.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/8a52ff3d6e8e6ef7/repo/_runs/cell_1/experiment1/model_loop/models/pruned/feature_overlap_similarity.py`

### featural_exemplar_similarity — pruned (experiment1 end of experiment)

**Margin:** 141.4 nats behind extrapolation_verification_failure (5.3× dse)

**Hypothesis:** People do not pick a single best-fitting rule; they judge a target by its graded exemplar similarity to the examples they saw, where two numbers are similar to the extent that they are close on the number line and share mathematical/digit features (parity, last digit, multiples, powers, primality, digits contained), and "yes" rises with the target's average similarity to the examples. This disagrees most sharply with the rule-based Bayesian model on members of the winning rule that are far from or featurally unlike the examples (which it endorses fully and this model only partly, producing graded typicality), and on non-members that are numerically close to the examples or share their last digit (which it rejects and this model partially endorses).

**Source:** `/scr/wurgaft/auto-psych/agent_trees/8a52ff3d6e8e6ef7/repo/_runs/cell_1/experiment1/model_loop/models/pruned/featural_exemplar_similarity.py`

### bounded_range_rule_learning — pruned (experiment1 end of experiment)

**Margin:** 143.2 nats behind extrapolation_verification_failure (4.5× dse)

**Hypothesis:** People believe the program produces numbers that follow a mathematical rule but only within a bounded range spanned by the examples: they do Bayesian concept learning over "rule restricted to a range" concepts (any of the 45 rules, or no rule at all, intersected with an interval that covers every example), weighting each by the size principle with an inferred, tempered evidence strength. Because a rule restricted to the examples' range is a smaller, better-fitting concept than the unbounded rule, people generalize mainly to rule members inside the [min, max] span of the examples and endorse rule members far outside that span much less.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/8a52ff3d6e8e6ef7/repo/_runs/cell_1/experiment1/model_loop/models/pruned/bounded_range_rule_learning.py`

### featural_contrast_similarity — pruned (experiment1 end of experiment)

**Margin:** 154.9 nats behind extrapolation_verification_failure (5.6× dse)

**Hypothesis:** People do not pick a single rule; they judge a target by featural similarity to the examples, in the manner of Tversky's contrast model. Each number is represented by its properties (parity, primality, being a square/cube/power, divisibility, final digit, digits it contains, decade), and the more of the examples' shared properties the target also has (weighted by how distinctive each property is), and the fewer extra properties the examples lack, the more likely people are to say yes — so members of a rule are graded by how many other properties they share with the examples, and non-members that share e.g. a last digit still get partial endorsement.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/8a52ff3d6e8e6ef7/repo/_runs/cell_1/experiment1/model_loop/models/pruned/featural_contrast_similarity.py`

### extrapolation_verification_failure — pruned (experiment2 end of experiment)

**Margin:** 173.4 nats behind consistent_examples_bounded_rules (6.5× dse)

**Hypothesis:** People infer the hidden concept by the same tempered, serial-position-weighted Bayesian concept learning as the best model (45 weighted rules and all intervals, with sorted lists shifting belief toward ranges), but to answer they must mentally verify that the target follows from the examples, and this extrapolation becomes unreliable beyond the span of numbers they have actually seen: the farther the target lies outside the examples' min–max range, the more likely the check fails and they answer "no", at an inferred rate per unit of distance. This predicts that members of a broad arithmetic rule (e.g. multiples of 3 after 63, 27, 99, 33) are endorsed less outside the examples' range than inside it, more steeply than belief in the concept alone implies, while inside-range judgments are unchanged.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/8a52ff3d6e8e6ef7/repo/_runs/cell_1/experiment2/model_loop/models/pruned/extrapolation_verification_failure.py`

### outlier_tolerant_concept_learning — pruned (experiment1 end of experiment)

**Margin:** 198.9 nats behind extrapolation_verification_failure (7.8× dse)

**Hypothesis:** People do normative Bayesian concept learning over the 45 mathematical rules (each with its own prior weight) and all intervals of 1-100, using the size principle, but they mis-estimate how the examples were generated: they believe a much larger share of the examples are random strays from 1-100 than the ideal learner assumes (an inferred noise belief instead of a fixed 10%). Because a stray-tolerant learner keeps believing in a tight concept (a small rule or a narrow interval around the examples) even when one example does not fit it, belief stays on tight hypotheses rather than spreading to broad rules and wide intervals. So members of the tight hypotheses keep graded endorsement, while far-away non-members, which only broad hypotheses cover, are rejected more firmly.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/8a52ff3d6e8e6ef7/repo/_runs/cell_1/experiment1/model_loop/models/pruned/outlier_tolerant_concept_learning.py`

### normalized_attention_similarity — pruned (experiment2 end of experiment)

**Margin:** 199.3 nats behind consistent_examples_bounded_rules (6.1× dse)

**Hypothesis:** People judge a target by its similarity to the examples over number properties (parity, primality, squares, cubes, powers, multiples, final digit, digits contained), where the examples set a fixed budget of attention that is shared among the properties according to how consistently the examples have them. The target's dissimilarity is the fraction of that attention falling on properties it lacks, and the chance of "yes" falls off exponentially with it (Shepard-style generalization). Because the budget is fixed, a lone example spreads attention over its many properties, so a target sharing most of them (10 → 80, 6 → 18) is still judged very similar and broadly endorsed, while several examples that agree on one property (all ending in 2) concentrate attention there and sharply reject targets lacking it.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/8a52ff3d6e8e6ef7/repo/_runs/cell_1/experiment2/model_loop/models/pruned/normalized_attention_similarity.py`

### predictive_output_probability_readout — pruned (experiment1 end of experiment)

**Margin:** 202.1 nats behind extrapolation_verification_failure (6.8× dse)

**Hypothesis:** People do the same tempered Bayesian concept learning over the 45 individually weighted rules and all intervals of 1-100, but when asked whether the program would produce a target they do not ask "is it in the concept?" — they judge how probable it is that the program, sampling from its concept, would output that number next (so a number in a small concept seems far likelier to come out than one in a big concept), and say "yes" when that predicted chance is high enough relative to an internal criterion, with graded noise. This disagrees most sharply with the best model's membership readout on members of broad rules: rule members outside the examples' range (supported only by large rules such as "multiples of 3" or "odd") are endorsed much less than in-range members, which also get the high output probability of tight intervals around the examples, and members of very large rules (odd, even) are endorsed less than members of small ones.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/8a52ff3d6e8e6ef7/repo/_runs/cell_1/experiment1/model_loop/models/pruned/predictive_output_probability_readout.py`

### coherence_relative_similarity — pruned (experiment2 end of experiment)

**Margin:** 226.6 nats behind consistent_examples_bounded_rules (6.2× dse)

**Hypothesis:** People judge a target by its similarity to the examples, where two numbers are similar to the extent that they share number properties (parity, primality, being a square/cube/power, divisibility, final digit, digits contained, and a magnitude band of width 20), each shared property counting by how rare, and so informative, it is. The criterion is relative: people say "yes" when the target is about as similar to the examples, on average, as the examples are to one another, so a set whose members are loosely related (no common rule) accepts loosely related targets such as nearby numbers, while a tightly related set (all ending in 2) demands the same tight relation. A lone example, which has no partner to compare with, is judged against a fixed default criterion.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/8a52ff3d6e8e6ef7/repo/_runs/cell_1/experiment2/model_loop/models/pruned/coherence_relative_similarity.py`

### bigelow_piantadosi_mixture — pruned (experiment1 end of experiment)

**Margin:** 277.8 nats behind extrapolation_verification_failure (7.0× dse)

**Hypothesis:** Bayesian concept learning over 45 rules (odd, even, primes, squares, cubes, powers of 2-10, powers of 2 without 32 or with 37, multiples of 3-9, 11 and 12, ending in 0-9, containing the digit 0-9) and every interval of 1-100. A rule is chosen with probability lambda_0, and the rules are equally likely; each interval has prior (1 - lambda_0) / 100^2. Examples are sampled from the concept (the size principle), 10% of them uniformly from 1-100, and "yes" is the posterior probability that the concept contains the target.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/8a52ff3d6e8e6ef7/repo/_runs/cell_1/experiment1/model_loop/models/pruned/bigelow_piantadosi_mixture.py`

### bigelow_piantadosi_independent — pruned (experiment1 end of experiment)

**Margin:** 297.6 nats behind extrapolation_verification_failure (6.7× dse)

**Hypothesis:** Bayesian concept learning over 45 rules (odd, even, primes, squares, cubes, powers of 2-10, powers of 2 without 32 or with 37, multiples of 3-9, 11 and 12, ending in 0-9, containing the digit 0-9) and every interval of 1-100, with a prior weight inferred for each rule separately and a share lambda_0 for the rules as a group; each interval has prior (1 - lambda_0) / 100^2. Examples are sampled from the concept (the size principle), 10% of them uniformly from 1-100, and "yes" is the posterior probability that the concept contains the target.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/8a52ff3d6e8e6ef7/repo/_runs/cell_1/experiment1/model_loop/models/pruned/bigelow_piantadosi_independent.py`

### tenenbaum_rules_similarity — pruned (experiment1 end of experiment)

**Margin:** 589.0 nats behind extrapolation_verification_failure (7.0× dse)

**Hypothesis:** Bayesian concept learning over 33 mathematical rules (odd, even, squares, cubes, primes, multiples of 3-12, powers of 2-10, ending in 1-9) and every interval of 1-100. The rules share a prior fraction lambda equally; the intervals share the rest by an Erlang density on their size (scale sigma). Examples are sampled from the concept (the size principle), 10% of them uniformly from 1-100, and "yes" is the posterior probability that the concept contains the target.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/8a52ff3d6e8e6ef7/repo/_runs/cell_1/experiment1/model_loop/models/pruned/tenenbaum_rules_similarity.py`
