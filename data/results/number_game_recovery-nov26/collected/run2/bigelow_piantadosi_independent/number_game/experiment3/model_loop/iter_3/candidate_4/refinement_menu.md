# Refinement menu

The models you may refine, other than the incumbent `effortful_computed_rule_search`. Each entry is one mechanism: its hypothesis in full as its author stated it, how it stands, and where its source is. This is a menu, not a list of what is ruled out: a pruned model lost to the best model on the data it was scored on, by the stated margin, but its mechanism may be partly right — and a model that lost narrowly is the most promising target here. Read the source of the model you pick before you write anything.

## Live models other than the incumbent (in the set; best first)

### individual_range_sampling_looseness — rank 1, 12.6 ± 13.0 nats behind the best (1.0× dse: statistically tied with the best), ELPD-LOO -3512.2

**Hypothesis:** Refinement of `loose_range_sampling_sparse_rules`: people do the same capacity-limited Bayesian learning over arithmetic rules (kind plausibilities, sparse familiarities within kinds) and numerical ranges, reading the examples with one sampling strength for rules and a separate, looser one for stretches of the number line. The single change is that how loosely the examples are read as evidence for a tight range is a stable trait of each participant rather than a population constant (each person's range sampling strength is scaled by a personal factor with a fixed spread around the population, inferred from their answers; the rule strength stays shared). Because the examples' range evidence grows with every example, this predicts individual differences that grow with set size: some participants keep to the examples' span and reject far rule members, others favour the fitting rule, consistently across every set each person sees, while for single examples people barely differ.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/fc3d1e4e3afb1c37/repo/_runs/cell_1/experiment3/model_loop/models/individual_range_sampling_looseness.py`

### magnitude_structure_attention_tradeoff — rank 2, 14.9 ± 13.1 nats behind the best (1.1× dse: statistically tied with the best), ELPD-LOO -3514.4

**Hypothesis:** Refinement of `individual_range_sampling_looseness`: people do the same capacity-limited Bayesian learning over arithmetic rules (kind plausibilities, sparse familiarities within kinds) and numerical ranges, reading the examples with one sampling strength for rules and a looser one for ranges, and each participant has a stable personal trait. The single change is what that trait is: instead of setting only how literally a person reads the examples as evidence for a tight range, it sets how the person divides attention between the examples' magnitudes and their arithmetic structure, so the same personal factor that strengthens their range evidence weakens their rule evidence by the same proportion (and the reverse). This predicts a within-person trade-off, consistent across every set a person sees: people who confine their yeses to the examples' span also extend fitting rules beyond it less, and people who extend rules fill in the span less, with the split growing as examples accumulate.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/fc3d1e4e3afb1c37/repo/_runs/cell_1/experiment3/model_loop/models/magnitude_structure_attention_tradeoff.py`

### loose_range_individual_rule_leaning — rank 3, 15.9 ± 13.7 nats behind the best (1.2× dse: statistically tied with the best), ELPD-LOO -3515.4

**Hypothesis:** Refinement of `individual_rule_range_leaning`. People learn the hidden concept by the same capacity-limited Bayesian weighing of arithmetic rules (sparse rule familiarities within kinds) against numerical ranges, with the same parity-conflict veto, and each participant has a stable personal leaning that raises or lowers the prior odds of every rule against the ranges. The single change, grafted from `loose_range_sampling_sparse_rules`, is that people do not read the examples with one sampling strength for both kinds of explanation: they weigh the evidence that a range is tight on its own, looser scale. So the personal leaning decides between a rule and a less literally sampled range. Participants who lean toward ranges then generalize somewhat beyond the examples' span, instead of stopping sharply at its edges, and as examples accumulate a fitting rule overtakes the tightest covering range faster for everyone.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/fc3d1e4e3afb1c37/repo/_runs/cell_1/experiment3/model_loop/models/loose_range_individual_rule_leaning.py`

### individual_rule_leaning_range_looseness — rank 4, 16.4 ± 13.4 nats behind the best (1.2× dse: statistically tied with the best), ELPD-LOO -3515.9

**Hypothesis:** Refinement of `individual_range_sampling_looseness`: people do the same capacity-limited Bayesian learning over arithmetic rules (kind plausibilities, sparse familiarities within kinds) and numerical ranges, reading the examples more loosely for ranges than for rules, with each participant's range looseness a stable personal trait. The single change, a component taken from `loose_range_individual_rule_leaning`, is that participants also differ in how readily they reach for a rule rather than a range before seeing any example (a personal shift of the prior odds of every rule against the ranges, with a fixed spread around the population). Because this leaning does not grow with the evidence, it predicts individual differences in rule-versus-range generalization that are already present for small sets, and that trade off yeses to in-span non-members against yeses to distant rule members, whereas range looseness alone makes people differ only as examples accumulate.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/fc3d1e4e3afb1c37/repo/_runs/cell_1/experiment3/model_loop/models/individual_rule_leaning_range_looseness.py`

### loose_range_sampling_sparse_rules — rank 5, 16.4 ± 14.1 nats behind the best (1.2× dse: statistically tied with the best), ELPD-LOO -3516.0

**Hypothesis:** Refinement of `sparse_rule_familiarity_within_kinds`: people do the same capacity-limited Bayesian learning over rules (with sparse familiarity within each kind) and numerical ranges. The single change is that people do not apply the size principle equally to both kinds of explanation: they read the examples with one sampling strength when weighing arithmetic rules and with a separate, looser one when weighing stretches of the number line, so the examples count for less as evidence that a range is tight. This predicts that sets no rule fits are generalized somewhat beyond the examples' span, and that as examples accumulate, a fitting rule overtakes the tightest covering range faster than a single shared sampling strength allows.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/fc3d1e4e3afb1c37/repo/_runs/cell_1/experiment3/model_loop/models/loose_range_sampling_sparse_rules.py`

### individual_working_memory_capacity — rank 6, 16.9 ± 13.6 nats behind the best (1.2× dse: statistically tied with the best), ELPD-LOO -3516.5

**Hypothesis:** People learn the hidden concept by the same Bayesian weighing of arithmetic rules (kind plausibilities, sparse familiarities within kinds) and stretches of the number line as the best model, with a looser reading of the examples as evidence for a tight range, but they are limited in how much of the examples' evidence they can hold and integrate at once, and this working-memory capacity is a stable trait that differs between people. A participant with a tighter capacity limit draws less extra evidence from each additional example, so for that person larger sets are generalized more broadly (broad rules and wide ranges keep more belief), while people barely differ on single examples, and each person's breadth holds consistently across every set they see.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/fc3d1e4e3afb1c37/repo/_runs/cell_1/experiment3/model_loop/models/individual_working_memory_capacity.py`

### pedagogical_example_selection — rank 7, 21.6 ± 15.8 nats behind the best (1.4× dse: statistically tied with the best), ELPD-LOO -3521.2

**Hypothesis:** People infer the hidden concept by the same capacity-limited Bayesian weighing of arithmetic rules (sparse rule familiarities within kinds) and numerical ranges as the best model, but they do not assume the examples are a random draw from the concept: they assume a helpful teacher picked them, favouring numbers that by themselves would single that concept out (e.g. numbers few other concepts contain, such as 64 for powers of 2 or the numbers near the ends of a range). The examples are therefore read as pointing to the concept that makes them the most telling choices, not just to the smallest concept that contains them. The sharpest disagreement with the best model is in sets made of "telling" numbers, which this account generalizes along the concept those numbers single out, and in sets of numbers that many concepts contain, which it treats as weak evidence and generalizes more broadly than the size principle does.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/fc3d1e4e3afb1c37/repo/_runs/cell_1/experiment3/model_loop/models/pedagogical_example_selection.py`

### without_replacement_example_sampling — rank 8, 22.1 ± 16.4 nats behind the best (1.3× dse: statistically tied with the best), ELPD-LOO -3521.7

**Hypothesis:** People learn the hidden concept by the same capacity-limited Bayesian weighing of arithmetic rules (each kind with its own plausibility, a few rules standing out as especially familiar or unfamiliar) and stretches of the number line as the best model, but they understand that the examples shown are different numbers drawn from the program one after another without repeats, so each example is judged against the members of a concept not yet shown. This makes a small concept that the examples nearly use up (e.g. 1, 8, 27 as the cubes, or a range only a little wider than the examples) a far more striking explanation than independent draws would make it, so such sets are generalized more narrowly to just the concept's few remaining members, while sets fitting large concepts are judged as before.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/fc3d1e4e3afb1c37/repo/_runs/cell_1/experiment3/model_loop/models/without_replacement_example_sampling.py`

### individual_leaning_loose_ranges — rank 9, 24.9 ± 15.4 nats behind the best (1.6× dse: statistically tied with the best), ELPD-LOO -3524.4

**Hypothesis:** Refinement of `loose_range_sampling_sparse_rules`: people do the same capacity-limited Bayesian learning over arithmetic rules (sparse familiarities within kinds) and numerical ranges, and they read the examples as more loosely sampled when weighing ranges than when weighing rules. The single change is a component taken from `individual_rule_range_leaning`: participants differ in how readily they reach for a rule rather than a range. Each person has a stable leaning that shifts the prior odds of every rule against the ranges, drawn from a fixed population spread, and an unidentified person is treated as the population average over that spread. The model predicts that the same set is generalized by its rule (yeses to distant rule members) by some participants and by loose proximity (yeses to nearby non-members, even somewhat beyond the examples' span) by others, consistently across every set each person sees.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/fc3d1e4e3afb1c37/repo/_runs/cell_1/experiment3/model_loop/models/individual_leaning_loose_ranges.py`

### individual_rule_range_leaning — rank 10, 25.0 ± 14.3 nats behind the best (1.7× dse: statistically tied with the best), ELPD-LOO -3524.6

**Hypothesis:** Refinement of `parity_veto_sparse_familiar_rules`: people learn the hidden concept by the same capacity-limited Bayesian weighing of arithmetic rules (sparse rule familiarities within kinds) against numerical ranges, with the same parity-conflict veto, but they differ from one another in how readily they reach for a rule rather than a range — each participant has a stable personal leaning that raises or lowers the prior odds of every rule against the ranges. The single change is this participant-level rule-versus-range leaning (people differ by a fixed spread around the population, each person's leaning inferred from the data), predicting that the same set is generalized by its rule (yeses to distant rule members) by some participants and by proximity (yeses to nearby non-members) by others, consistently across all the sets each person sees.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/fc3d1e4e3afb1c37/repo/_runs/cell_1/experiment3/model_loop/models/individual_rule_range_leaning.py`

### parity_veto_sparse_familiar_rules — rank 11, 26.5 ± 14.7 nats behind the best (1.8× dse: statistically tied with the best), ELPD-LOO -3526.1

**Hypothesis:** Refinement of `parity_conflict_decision_veto`: people infer the hidden concept by capacity-limited Bayesian learning over rules (multiples of 5 as its own base-ten kind) and numerical ranges, and a target whose parity matches none of the examples triggers an automatic parity conflict that, on a fixed fraction of such trials, overrides the inferred concept with a "no". The single change is grafting sparse rule familiarity (from `sparse_rule_familiarity_within_kinds`) onto the concept-learning stage: most rules share their kind's plausibility, and only a few stand out as exceptionally familiar (multiples of 11, powers of 10) or unfamiliar (multiples of 7, "contains the digit 1"). The claim is that the decision-stage parity veto survives once rule priors are calibrated this way: opposite-parity targets in all-odd or all-even sets (e.g. odd numbers inside the span of 20, 32, 92) get fewer yeses than the inferred concept implies, while same-parity targets are unaffected.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/fc3d1e4e3afb1c37/repo/_runs/cell_1/experiment3/model_loop/models/parity_veto_sparse_familiar_rules.py`

### sparse_rule_familiarity_within_kinds — rank 12, 28.0 ± 14.8 nats behind the best (1.9× dse: statistically tied with the best), ELPD-LOO -3527.6

**Hypothesis:** Refinement of `rule_familiarity_within_kinds`: people do the same capacity-limited Bayesian learning over rules and numerical ranges, with a prior plausibility for each kind of rule and rule-specific familiarities within each kind, but familiarity is sparse: most rules of a kind are about as familiar as one another and simply share the kind's plausibility, while only a few stand out as exceptionally familiar (e.g. multiples of 11, powers of 10) or exceptionally unfamiliar (e.g. multiples of 7, "contains the digit 1"). The single change is replacing the incumbent's evenly spread (normal) familiarities with heavy-tailed ones, which predicts that rules with no standout familiarity are generalized exactly as their kind's other rules are, instead of each being nudged up or down by a moderate amount, while the few exceptional rules keep their large deviations.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/fc3d1e4e3afb1c37/repo/_runs/cell_1/experiment3/model_loop/models/sparse_rule_familiarity_within_kinds.py`

### triangular_number_concept — rank 13, 30.0 ± 14.7 nats behind the best (2.0× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -3529.6

**Hypothesis:** People infer the hidden concept by the same capacity-limited Bayesian learning over arithmetic rules (each kind of rule with its own plausibility, a few rules standing out as especially familiar or unfamiliar) and numerical ranges as the best model, with the same parity-conflict veto at the answer, but their space of candidate concepts also contains the triangular numbers (1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 66, 78, 91): a well-known figurate sequence, built by adding successive whole numbers, that no multiples, parity, power or digit rule covers. It is its own kind of rule, held a priori with a plausibility inferred from the data. This predicts that sets made of triangular numbers (e.g. 15, 21, 45 or 10, 55, 28) get yeses to distant triangular targets such as 66, 78 or 91, and fewer yeses to non-triangular numbers between the examples, than a learner without this concept would give.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/fc3d1e4e3afb1c37/repo/_runs/cell_1/experiment3/model_loop/models/triangular_number_concept.py`

### rule_or_range_kind_commitment — rank 14, 30.1 ± 14.8 nats behind the best (2.0× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -3529.6

**Hypothesis:** People learn the hidden concept from the examples by the same capacity-limited Bayesian weighing of arithmetic rules and numerical ranges as the best model, but they do not blend the two kinds of explanation when they answer: they first commit to whether the program follows an arithmetic rule or produces a stretch of the number line, by comparing how well each kind as a whole explains the examples more decisively than an ideal learner would, and then judge the target only by the concepts of the kind they chose. This predicts that when rules and ranges both explain the examples moderately well, answers follow whichever kind is ahead more sharply than the blend implies, e.g. fewer yeses to far rule-members when ranges are slightly favored, and fewer yeses to in-span rule-violators when a rule is slightly favored.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/fc3d1e4e3afb1c37/repo/_runs/cell_1/experiment3/model_loop/models/rule_or_range_kind_commitment.py`

### individual_yes_criterion_rule_leaning — rank 15, 34.6 ± 14.7 nats behind the best (2.4× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -3534.1

**Hypothesis:** Refinement of `loose_range_individual_rule_leaning`. People learn the hidden concept as that model says: capacity-limited Bayesian weighing of arithmetic rules (sparse familiarities within kinds) against numerical ranges, with looser sampling for ranges, a parity-conflict veto, and a personal leaning toward rules or ranges. The single change is at the answer stage: each participant also has a stable response criterion, a personal readiness to say "yes" at a given strength of belief that the target is in the concept. It shifts the log-odds of every answer by the same personal amount (people differ by a fixed spread; an unidentified person is averaged over it). So participants differ consistently in their overall yes-rate across every set and target, most visibly for targets the concept leaves uncertain and least for near-certain members and non-members, independently of whether they lean toward rules or ranges.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/fc3d1e4e3afb1c37/repo/_runs/cell_1/experiment3/model_loop/models/individual_yes_criterion_rule_leaning.py`

### individual_yes_response_bias — rank 16, 35.9 ± 15.0 nats behind the best (2.4× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -3535.5

**Hypothesis:** People all learn the hidden concept the same way (the capacity-limited Bayesian weighing of arithmetic rules and loosely sampled numerical ranges of the current models), but they differ stably in how readily they say "yes" given the same belief: each participant has a personal yes-proneness at the decision stage that shifts the odds of answering yes on every trial. This predicts individual differences concentrated on uncertain targets (near the span edges, far rule members, small sets) and pointing the same way for a person across all sets, rather than differences that depend on whether a rule or a range explains the set.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/fc3d1e4e3afb1c37/repo/_runs/cell_1/experiment3/model_loop/models/individual_yes_response_bias.py`

### criterion_gated_concept_belief — rank 17, 96.3 ± 17.0 nats behind the best (5.7× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -3595.8

**Hypothesis:** People infer the hidden concept by the same capacity-limited Bayesian learning over rules (with kind plausibilities and a few exceptionally familiar or unfamiliar rules) and numerical ranges as the simplest current model, but they do not answer by probability matching to their belief. They say "yes" only when their belief that the target belongs to the concept clears an internal criterion of how sure they must be, and this comparison is noisy from trial to trial. This predicts that yes-rates are a steep S-shaped function of belief around the criterion, not a one-for-one copy of it: moderately supported targets on either side of the criterion are answered more unanimously than their belief implies.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/fc3d1e4e3afb1c37/repo/_runs/cell_1/experiment3/model_loop/models/criterion_gated_concept_belief.py`

### conjunctive_category_exemplar_similarity — rank 18, 587.4 ± 44.3 nats behind the best (13.2× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -4086.9

**Hypothesis:** People do not infer a hidden concept; they compare the target with each remembered example, judging two numbers similar to the extent that they belong together to the number categories stored in memory (odd, even, primes, squares, powers, multiples, final or contained digit, and stretches of the number line), with small, distinctive shared categories counting more than broad ones. Because the target has to resemble every example, these comparisons combine conjunctively: the more uniformly similar the target is to all the examples (one weak match pulls it down sharply), the more likely people are to say yes.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/fc3d1e4e3afb1c37/repo/_runs/cell_1/experiment3/model_loop/models/conjunctive_category_exemplar_similarity.py`

## Pruned models (out of the set; narrowest margin first)

### recency_weighted_even_odd_family_prior — pruned (experiment1 end of experiment)

**Margin:** 2.8 nats behind familiarity_relaxed_size_principle (2.1× dse)

**Hypothesis:** Refinement of `even_odd_split_family_prior`: people do the same Bayesian concept learning over the same 45 rules (with separate prior plausibilities for even, odd and the other kinds of rule) and all intervals of 1-100 with a tempered size principle, but they do not weigh every example equally — they read the examples in order and each example's evidence counts more the later it was shown (a fitted order gradient, with the total amount of evidence held fixed). The single change is grafting this recency weighting of the examples (from `recency_weighted_concept_learning`) onto the incumbent, motivated by the critique that targets near the last-shown example get more yeses than targets near the first-shown one, which an order-blind learner cannot produce.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/fc3d1e4e3afb1c37/repo/_runs/cell_1/experiment1/model_loop/models/pruned/recency_weighted_even_odd_family_prior.py`

### even_odd_split_family_prior — pruned (experiment1 end of experiment)

**Margin:** 4.1 nats behind familiarity_relaxed_size_principle (2.2× dse)

**Hypothesis:** Refinement of `rule_family_prior_tempered_mixture`: people do the same Bayesian concept learning over the same 45 rules and all intervals of 1-100, with a tempered size principle and a prior over kinds of rule, but they do not treat "even" and "odd" as one kind of rule with a single shared plausibility — "even numbers" and "odd numbers" are separate kinds, each with its own prior plausibility inferred from the data. The single change is splitting the parity family in two, predicting that an all-even set of examples triggers parity generalization (yeses to distant same-parity targets) more readily than an all-odd set does.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/fc3d1e4e3afb1c37/repo/_runs/cell_1/experiment1/model_loop/models/pruned/even_odd_split_family_prior.py`

### recency_decay_even_odd_family_prior — pruned (experiment1 end of experiment)

**Margin:** 4.8 nats behind familiarity_relaxed_size_principle (2.1× dse)

**Hypothesis:** Refinement of `even_odd_split_family_prior`: people do the same Bayesian concept learning over the same 45 rules (with separate prior plausibilities for even, odd and the other kinds of rule) and all intervals of 1-100 with a tempered size principle, but they read the examples one at a time and their memory of earlier examples fades, so each example's evidence counts less the more examples were shown after it. The single change is grafting this recency-weighted evidence (from `recency_weighted_concept_learning`) onto the incumbent, motivated by the critique that people say yes more to targets near the last-shown example than to targets near the first-shown one, which the order-blind incumbent cannot produce.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/fc3d1e4e3afb1c37/repo/_runs/cell_1/experiment1/model_loop/models/pruned/recency_decay_even_odd_family_prior.py`

### stochastic_example_recall_sampling — pruned (experiment1 end of experiment)

**Margin:** 5.6 nats behind familiarity_relaxed_size_principle (2.7× dse)

**Hypothesis:** People do not reason from all the examples they were shown, but from the examples they can recall when judging the target: the most recent example is always remembered, while each earlier example is recalled only by chance, less often the further back it was shown, and the learner infers the hidden concept (by Bayesian rule-and-range inference) from whichever examples were recalled. Because the answer averages over these all-or-none recall outcomes, targets near the last example are endorsed more than targets near the first, and sets shown earlier in a sequence partly behave like smaller, broader-generalizing sets.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/fc3d1e4e3afb1c37/repo/_runs/cell_1/experiment1/model_loop/models/pruned/stochastic_example_recall_sampling.py`

### arithmetic_progression_concepts — pruned (experiment1 end of experiment)

**Margin:** 5.9 nats behind familiarity_relaxed_size_principle (2.8× dse)

**Hypothesis:** Refinement of `even_odd_split_family_prior`: people do the same Bayesian concept learning (tempered size principle, separate even and odd kinds, a prior over kinds of rule), but their space of concepts also includes regularly spaced sequences that do not start at zero — "every m-th number starting from r" (numbers leaving the same remainder r when divided by m, for spacings 3–12, e.g. 1, 5, 9, 13, …), a new kind of rule with its own prior plausibility inferred from the data. The single change is adding this arithmetic-progression kind, predicting that sets whose examples share a common spacing (such as 5, 9, 17, 33) earn yeses for distant targets continuing that spacing, which no multiple, parity or final-digit rule covers.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/fc3d1e4e3afb1c37/repo/_runs/cell_1/experiment1/model_loop/models/pruned/arithmetic_progression_concepts.py`

### previous_set_intrusion_concept_learning — pruned (experiment1 end of experiment)

**Margin:** 7.0 nats behind familiarity_relaxed_size_principle (3.2× dse)

**Hypothesis:** People infer the hidden concept by Bayesian rule-and-range inference (tempered size principle, a prior over kinds of rule with separate even and odd kinds), but the examples of the previously shown, different set linger in memory and intrude as extra, down-weighted evidence about the current set's concept (proactive interference between trials). This disagrees most sharply with the order-blind, trial-independent best model whenever a set follows an unrelated set: the pooled, partly conflicting examples push belief toward broad ranges and broad rules (more yeses to distant targets), or, when the previous set shares the current set's pattern (e.g. both even), toward that shared rule — so the same set and target get different answers depending on what was shown just before.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/fc3d1e4e3afb1c37/repo/_runs/cell_1/experiment1/model_loop/models/pruned/previous_set_intrusion_concept_learning.py`

### rule_family_prior_tempered_mixture — pruned (experiment1 end of experiment)

**Margin:** 19.6 nats behind familiarity_relaxed_size_principle (4.0× dse)

**Hypothesis:** Refinement of `tempered_size_principle_mixture`: people do the same Bayesian concept learning over the same 45 rules and all intervals of 1-100 with a partially applied (tempered) size principle, but they do not treat every rule as equally plausible a priori — they hold a prior over *kinds* of rule (parity, primes, squares/cubes, powers, multiples, final digit, contained digit), each kind having its own plausibility, inferred from the data, and the rules within a kind sharing it equally. The single change is this family-structured prior in place of the uniform rule prior: it predicts that whole kinds of rule (e.g. "contains the digit 2" or "multiples of 4") are rarely entertained even when examples fit them, while parity and final-digit rules compete strongly with numerical intervals.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/fc3d1e4e3afb1c37/repo/_runs/cell_1/experiment1/model_loop/models/pruned/rule_family_prior_tempered_mixture.py`

### erlang_interval_family_prior — pruned (experiment1 end of experiment)

**Margin:** 20.6 nats behind familiarity_relaxed_size_principle (3.8× dse)

**Hypothesis:** Refinement of `rule_family_prior_tempered_mixture`: people do the same Bayesian concept learning over the same 45 rules (with a prior over kinds of rule) and all intervals of 1-100 with a tempered size principle, but they do not consider every numerical range equally plausible a priori — they expect ranges of a typical, intermediate width (an Erlang prior on interval size with a fitted scale, as in Tenenbaum's number game), so very narrow and very broad ranges are discounted before any examples are seen. The single change is grafting this size-graded interval prior (from `learned_rule_prior_erlang_intervals`) onto the incumbent in place of its uniform interval prior, predicting that magnitude-based generalization from one or two examples extends to a characteristic distance around them instead of collapsing onto the tightest covering range or spreading over the whole 1-100 line.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/fc3d1e4e3afb1c37/repo/_runs/cell_1/experiment1/model_loop/models/pruned/erlang_interval_family_prior.py`

### recency_weighted_rule_family_prior — pruned (experiment1 end of experiment)

**Margin:** 21.1 nats behind familiarity_relaxed_size_principle (4.0× dse)

**Hypothesis:** Refinement of `rule_family_prior_tempered_mixture`: people do the same Bayesian concept learning over the same 45 rules (with a prior over kinds of rule) and all intervals of 1-100 with a tempered size principle, but they read the examples in the order shown and their memory of earlier examples fades, so each example's evidence counts less the further back it was shown and the most recent examples dominate which concept is inferred. The single change is grafting this recency-weighted evidence (from `recency_weighted_concept_learning`) onto the family-prior model, motivated by the critique that targets near the last-shown example are endorsed more than targets near the first-shown one, which no order-blind learner can produce.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/fc3d1e4e3afb1c37/repo/_runs/cell_1/experiment1/model_loop/models/pruned/recency_weighted_rule_family_prior.py`

### sticky_running_guess_revision — pruned (experiment2 end of experiment)

**Margin:** 22.9 nats behind parity_veto_sparse_familiar_rules (2.9× dse)

**Hypothesis:** People read the examples in the order shown and hold a single running guess about the program's concept (a mathematical rule or a numerical range), formed from the first example by the same capacity-limited Bayesian learning as the best model; each new example only revises that guess when it has to — a guess the new example fits is kept with a fitted probability (stickiness), and only otherwise is a new guess drawn from everything seen so far. Because guesses formed from early examples can outlive the evidence of later ones, the same examples shown in a different order are generalized differently, and broad concepts suggested by the first example survive longer than an order-blind learner allows.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/fc3d1e4e3afb1c37/repo/_runs/cell_1/experiment2/model_loop/models/pruned/sticky_running_guess_revision.py`

### familiar_rules_parity_conflict_veto — pruned (experiment2 end of experiment)

**Margin:** 23.6 nats behind parity_veto_sparse_familiar_rules (3.2× dse)

**Hypothesis:** Refinement of `rule_familiarity_within_kinds`: people do the same capacity-limited Bayesian learning over rules and numerical ranges, with each kind of rule having its own prior plausibility and each rule its own familiarity within that kind. On top of this, a number's parity is registered automatically, so when the target's parity matches none of the examples, that conflict intrudes at the moment of answering and, on a fixed fraction of those trials, overrides the inferred concept and produces a "no". The single change is grafting this decision-stage parity veto (from `parity_conflict_decision_veto`) onto the familiarity model. It predicts that in all-odd or all-even sets, opposite-parity targets that the inferred concept covers (for example odd numbers inside the span of 20, 32, 92) get fewer yeses than the concept implies, while same-parity targets are unaffected.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/fc3d1e4e3afb1c37/repo/_runs/cell_1/experiment2/model_loop/models/pruned/familiar_rules_parity_conflict_veto.py`

### rule_familiarity_within_kinds — pruned (experiment2 end of experiment)

**Margin:** 30.2 nats behind parity_veto_sparse_familiar_rules (3.6× dse)

**Hypothesis:** Refinement of `base_ten_multiples_capacity_limited`: people do the same capacity-limited Bayesian learning over rules and numerical ranges, with a prior plausibility for each kind of rule (odd, even, primes, squares/cubes, powers, multiples, final digit, contained digit, and multiples of 5 as its own base-ten kind), but the rules within a kind are not equally familiar — each rule carries its own familiarity that raises or lowers its prior plausibility relative to the other rules of its kind (a substantial, fixed amount of variation, with each rule's familiarity inferred from the data). The single change is replacing the equal sharing of a kind's plausibility among its rules with these rule-specific familiarities, predicting that some rules (e.g. multiples of 7, or "contains the digit 1") are rarely entertained even when the examples fit them, so those sets are generalized by range rather than by the rule, while other rules of the same kind (e.g. powers of 10, multiples of 11) are adopted readily.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/fc3d1e4e3afb1c37/repo/_runs/cell_1/experiment2/model_loop/models/pruned/rule_familiarity_within_kinds.py`

### recency_weighted_concept_learning — pruned (experiment1 end of experiment)

**Margin:** 47.2 nats behind familiarity_relaxed_size_principle (2.2× dse)

**Hypothesis:** People infer the hidden concept by Bayesian inference over mathematical rules (each with its own prior plausibility) and numerical intervals, but they read the examples one at a time and their memory of earlier examples fades: each example's evidence counts less the further back in the sequence it was shown, so the most recent examples dominate which concept is inferred. This predicts that the same set shown in a different order yields different generalization, with targets that fit the last examples (but not the first ones) endorsed more than an ideal, order-blind learner would.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/fc3d1e4e3afb1c37/repo/_runs/cell_1/experiment1/model_loop/models/pruned/recency_weighted_concept_learning.py`

### repeated_digit_base_ten_rules — pruned (experiment2 end of experiment)

**Margin:** 47.2 nats behind parity_veto_sparse_familiar_rules (2.8× dse)

**Hypothesis:** Refinement of `base_ten_multiples_capacity_limited`: people do the same capacity-limited Bayesian learning over rules and numerical ranges, with "multiples of 5" as its own base-ten-salient kind of rule, but their space of candidate rules also includes a digit pattern that base-ten notation makes visually obvious and that the incumbent lacks entirely: numbers written with a repeated digit (11, 22, 33, ..., 99), a new kind of rule with its own prior plausibility inferred from the data. The single change is adding this repeated-digit rule, predicting that sets such as 3, 33, 99 are generalized to other repdigits (88, 44) that no multiples, parity or final-digit rule covers, and that a single repdigit example (e.g. 33 or 44) earns fewer yeses for numerically close non-repdigits.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/fc3d1e4e3afb1c37/repo/_runs/cell_1/experiment2/model_loop/models/pruned/repeated_digit_base_ten_rules.py`

### first_example_win_stay_lose_sample — pruned (experiment1 end of experiment)

**Margin:** 50.6 nats behind familiarity_relaxed_size_principle (6.1× dse)

**Hypothesis:** People do not weigh all examples together; they form a guess about the concept from the first example they see and then test it against the later examples ("win-stay, lose-sample"). If every later example fits the guess formed from the first example, they keep it; only when an example contradicts it do they abandon it and re-infer the concept from all the examples. Because the first example's guess survives whenever it is consistent, broad concepts suggested by the first example (e.g. "even", a wide range around it) persist far more than an ideal learner allows, and the same set in a different order can yield different generalization.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/fc3d1e4e3afb1c37/repo/_runs/cell_1/experiment1/model_loop/models/pruned/first_example_win_stay_lose_sample.py`

### parity_conflict_decision_veto — pruned (experiment2 end of experiment)

**Margin:** 52.5 nats behind parity_veto_sparse_familiar_rules (2.9× dse)

**Hypothesis:** People infer the hidden concept by the same capacity-limited Bayesian learning over rules (with multiples of 5 as its own, base-ten-salient kind of rule) and numerical ranges as the best model, but a number's parity is registered automatically and effortlessly, so at the moment of judging a target whose parity matches none of the examples, that conflict intrudes on the decision and, on a fixed fraction of such trials, overrides the inferred concept and produces a "no". This decision-stage parity interference predicts that in all-odd or all-even sets, opposite-parity targets that the inferred concept covers (for example odd numbers inside the span of 20, 32, 92) get fewer yeses than the concept alone implies, while same-parity targets are unaffected.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/fc3d1e4e3afb1c37/repo/_runs/cell_1/experiment2/model_loop/models/pruned/parity_conflict_decision_veto.py`

### base_ten_multiples_capacity_limited — pruned (experiment2 end of experiment)

**Margin:** 58.2 nats behind parity_veto_sparse_familiar_rules (3.2× dse)

**Hypothesis:** Refinement of `capacity_limited_evidence_integration`: people do the same capacity-limited Bayesian learning over rules and numerical ranges, with each added example contributing less evidence, but "multiples of 5" is not just one more multiples rule sharing a single plausibility with multiples of 3, 4, 7, 11 and the rest — because it is aligned with our base-ten number notation, it is its own, far more readily entertained kind of rule with a separate prior plausibility inferred from the data. The single change is splitting multiples of 5 out of the multiples family (as even and odd are already split out of parity), predicting that sets such as 90, 100, 5 are generalized to every multiple of 5 across the number line, while sets fitting a less familiar multiple (e.g. 42, 35, 21, 84 as multiples of 7) are generalized by that rule less than the incumbent predicts.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/fc3d1e4e3afb1c37/repo/_runs/cell_1/experiment2/model_loop/models/pruned/base_ten_multiples_capacity_limited.py`

### learned_rule_prior_erlang_intervals — pruned (experiment1 end of experiment)

**Margin:** 58.4 nats behind familiarity_relaxed_size_principle (2.2× dse)

**Hypothesis:** Refinement of `learned_rule_prior_mixture`: people learn concepts by the same Bayesian inference over 45 mathematical rules (each with its own learned prior plausibility) and all intervals of 1-100 with the full size principle, but they do not consider every numerical range equally plausible a priori — they expect ranges of a typical, intermediate width (an Erlang prior on interval size with a fitted scale, as in Tenenbaum's number game), so very narrow and very broad ranges are discounted before any examples are seen. The single change is replacing the incumbent's uniform prior over intervals with this size-graded interval prior (taken from `tenenbaum_rules_similarity`), predicting that magnitude-based generalization from one or two examples extends to a characteristic distance around them rather than collapsing onto the tightest range that covers them.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/fc3d1e4e3afb1c37/repo/_runs/cell_1/experiment1/model_loop/models/pruned/learned_rule_prior_erlang_intervals.py`

### narrow_range_base_ten_multiples — pruned (experiment2 end of experiment)

**Margin:** 58.6 nats behind parity_veto_sparse_familiar_rules (3.2× dse)

**Hypothesis:** Refinement of `base_ten_multiples_capacity_limited`: people do the same capacity-limited Bayesian learning over rules and numerical ranges, with "multiples of 5" being its own readily entertained kind of rule because it fits base-ten notation, and with each added example contributing less evidence. But before seeing any example they also expect the program's numerical range to be narrow: a range's prior plausibility falls as a fitted power of its width, instead of every range being equally plausible. The single change is grafting this narrow-range expectation (from `narrow_range_prior_capacity_limited`) onto the base-ten model. Because the expectation comes before the evidence, the capacity limit does not dilute it, so for sets that no rule explains (e.g. mixed odd and even numbers) people keep their yeses closer to the examples' span than the base-ten model predicts, while its multiples-of-5 generalization stays the same.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/fc3d1e4e3afb1c37/repo/_runs/cell_1/experiment2/model_loop/models/pruned/narrow_range_base_ten_multiples.py`

### extreme_examples_anchor_range_edges — pruned (experiment2 end of experiment)

**Margin:** 60.8 nats behind parity_veto_sparse_familiar_rules (3.4× dse)

**Hypothesis:** People do the same capacity-limited Bayesian learning over mathematical rules and numerical ranges as the best model (with "multiples of 5" as its own, readily entertained kind of rule), but they read the smallest and the largest example as marking where the program's range stops: a range whose lower edge sits exactly at the smallest example, or whose upper edge sits exactly at the largest example, is entertained more readily before any evidence is weighed (a fitted boost per anchored edge). This predicts that, when ranges carry the judgment, yeses fall off right at the extreme examples — fewer yeses to targets just beyond them, more to targets inside their span — rather than tapering gradually past them as the size principle alone implies.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/fc3d1e4e3afb1c37/repo/_runs/cell_1/experiment2/model_loop/models/pruned/extreme_examples_anchor_range_edges.py`

### query_as_evidence_concept_learning — pruned (experiment2 end of experiment)

**Margin:** 62.0 nats behind parity_veto_sparse_familiar_rules (3.3× dse)

**Hypothesis:** People infer the hidden concept by the same capacity-limited Bayesian learning over rules (with multiples of 5 as its own, base-ten-salient kind of rule) and numerical ranges as the best model, but they do not treat the question as neutral: being asked about a particular number is itself read as information about the program, so before answering, the learner updates on the queried number as if it were a weak, extra observation (a pragmatic "why are they asking about this?" inference), with its strength and direction inferred from the data. If the question is read as a hint the target is produced, targets covered only by small concepts gain yeses; if it is read as a lure, fewer yeses go to every target and narrow concepts lose more than broad ones, which also lowers yeses on small sets, where the queried number weighs most against the little evidence the examples give.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/fc3d1e4e3afb1c37/repo/_runs/cell_1/experiment2/model_loop/models/pruned/query_as_evidence_concept_learning.py`

### base_ten_fluency_graded_multiples — pruned (experiment2 end of experiment)

**Margin:** 81.5 nats behind parity_veto_sparse_familiar_rules (3.9× dse)

**Hypothesis:** Refinement of `multiplication_fluency_rule_prior`: people do capacity-limited Bayesian learning over rules and numerical ranges, and a "multiples of m" rule comes to mind less readily the larger m is, because checking divisibility by a larger number is less fluent mental arithmetic. The single change is grafting the base-ten account of multiples of 5 (from `base_ten_multiples_capacity_limited`) onto it: multiples of 5 are not a point on that fluency gradient but their own, separately plausible kind of rule because they fit our decimal notation, so the gradient describes only the non-decimal bases (3, 4, 6, 7, 8, 9, 11, 12). This predicts that sets such as 90, 100, 5 are generalized to every multiple of 5, while sets fitting multiples of 7, 9, 11 or 12 (e.g. 42, 35, 21, 84) are generalized less to the rule's distant members and more to non-member numbers within the examples' span than the best model predicts.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/fc3d1e4e3afb1c37/repo/_runs/cell_1/experiment2/model_loop/models/pruned/base_ten_fluency_graded_multiples.py`

### multiplication_fluency_rule_prior — pruned (experiment2 end of experiment)

**Margin:** 104.6 nats behind parity_veto_sparse_familiar_rules (4.6× dse)

**Hypothesis:** Refinement of `capacity_limited_evidence_integration`: people do the same capacity-limited Bayesian learning over rules (separate prior plausibilities for even, odd and the other kinds of rule) and numerical ranges, but a "multiples of m" rule is entertained less readily the larger m is, because checking divisibility by a larger number is less fluent mental arithmetic (multiples of 3 or 4 come to mind easily, multiples of 7, 9, 11 or 12 hardly do). The single change is a fitted fluency gradient that lowers each multiples rule's prior plausibility as a power of its base. The disagreement with the best model lives in sets that fit multiples of a large base (e.g. 42, 35, 21, 84 or 36, 45, 54, 81): the best model generalizes them to the rule's distant members (91, 98, 90, 99), whereas this model predicts fewer yeses there and more yeses to non-member numbers inside the examples' span, which are carried by ranges.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/fc3d1e4e3afb1c37/repo/_runs/cell_1/experiment2/model_loop/models/pruned/multiplication_fluency_rule_prior.py`

### capacity_limited_evidence_integration — pruned (experiment2 end of experiment)

**Margin:** 114.8 nats behind parity_veto_sparse_familiar_rules (4.7× dse)

**Hypothesis:** People infer the hidden concept by Bayesian learning over mathematical rules (with separate prior plausibilities for even, odd and the other kinds of rule) and numerical intervals, but they have a limited capacity for integrating examples: the examples of a set compete for the same limited working memory, so each additional example adds less evidence than the one before, and the total evidence grows more slowly than the number of examples. This predicts that larger sets are generalized more broadly (broad rules and wide ranges keep more weight) than an ideal learner, which counts every example fully, would allow, while single-example sets are judged as the ideal learner does.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/fc3d1e4e3afb1c37/repo/_runs/cell_1/experiment2/model_loop/models/pruned/capacity_limited_evidence_integration.py`

### approximate_magnitude_target_encoding — pruned (experiment2 end of experiment)

**Margin:** 115.6 nats behind parity_veto_sparse_familiar_rules (4.8× dse)

**Hypothesis:** People infer the hidden concept by capacity-limited Bayesian learning over mathematical rules and numerical ranges, but they check the target against that concept through an approximate, logarithmically compressed sense of its magnitude rather than its exact identity: the target is confused with numbers of similar size, more so for larger numbers (Weber's law), and the answer is the concept's average endorsement over those confusable numbers. This predicts that targets just outside a rule but numerically close to members — especially large targets — get more yeses than an exact-identity learner allows (the rule-violating near-target leak), while small targets are judged nearly exactly.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/fc3d1e4e3afb1c37/repo/_runs/cell_1/experiment2/model_loop/models/pruned/approximate_magnitude_target_encoding.py`

### proximity_leak_capacity_limited — pruned (experiment2 end of experiment)

**Margin:** 115.6 nats behind parity_veto_sparse_familiar_rules (4.8× dse)

**Hypothesis:** Refinement of `capacity_limited_evidence_integration`: people do the same capacity-limited Bayesian learning over rules (separate prior plausibilities for even, odd and the other kinds of rule) and numerical ranges, with each added example contributing less evidence, but when the target falls outside the concept they have inferred, its numerical closeness to one of the examples can still pull them into saying yes — a proximity leak whose strength falls off exponentially with the distance to the nearest example (fitted leak strength and fall-off). The single change is this similarity override of the inferred concept, motivated by the critique that, for sets fitting a small rule, targets that break the rule but lie within a few units of an example get more yeses than the incumbent predicts.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/fc3d1e4e3afb1c37/repo/_runs/cell_1/experiment2/model_loop/models/pruned/proximity_leak_capacity_limited.py`

### narrow_range_prior_capacity_limited — pruned (experiment2 end of experiment)

**Margin:** 117.7 nats behind parity_veto_sparse_familiar_rules (4.8× dse)

**Hypothesis:** Refinement of `capacity_limited_evidence_integration`: people do the same capacity-limited Bayesian learning over rules (separate prior plausibilities for even, odd and the other kinds of rule) and numerical ranges, with each added example contributing less evidence, but before seeing any example they expect the program's numerical range to be narrow — a range's prior plausibility falls as a fitted power of its width, instead of all ranges being equally plausible. The single change is this narrow-range prior expectation, which, unlike the evidence from the examples, is not diluted by the capacity limit as sets grow, so for sets that no rule explains (e.g. mixing odd and even numbers) people keep their yeses inside the examples' span more sharply than the incumbent predicts, as the mixed-parity range-gap critique shows.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/fc3d1e4e3afb1c37/repo/_runs/cell_1/experiment2/model_loop/models/pruned/narrow_range_prior_capacity_limited.py`

### pair_contrast_capacity_limited — pruned (experiment2 end of experiment)

**Margin:** 117.8 nats behind parity_veto_sparse_familiar_rules (4.9× dse)

**Hypothesis:** Refinement of `capacity_limited_evidence_integration`: people do the same capacity-limited Bayesian learning over rules (separate prior plausibilities for even, odd and the other kinds of rule) and numerical ranges, with each added example contributing less evidence, but a pair of examples is read as a deliberate contrast — two numbers shown together invite a direct comparison that makes the pair look especially diagnostic, so a two-example set is taken as more tightly sampled than the smooth capacity limit implies. The single change is a fitted pair-comparison boost to the sampling strength of two-example sets, predicting that people generalize more narrowly (fewer yeses) from two examples than from three or four, the non-monotone two-example dip the critique reports.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/fc3d1e4e3afb1c37/repo/_runs/cell_1/experiment2/model_loop/models/pruned/pair_contrast_capacity_limited.py`

### magnitude_privileged_size_principle — pruned (experiment2 end of experiment)

**Margin:** 119.2 nats behind parity_veto_sparse_familiar_rules (4.9× dse)

**Hypothesis:** People infer the hidden concept by Bayesian learning over mathematical rules (with separate prior plausibilities for even, odd and the other kinds of rule) and numerical ranges, with capacity-limited integration of the examples, but they mis-weight the two kinds of evidence: the "suspicious coincidence" that the examples all fall in a narrow stretch of the number line is applied more strongly than the coincidence that they all obey an arithmetic rule, because magnitude is perceived effortlessly while rule membership must be computed. This predicts that when no rule fits well (for example sets mixing odd and even numbers) people confine their yeses to the span of the examples more sharply than a learner who applies one sampling strength to every concept.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/fc3d1e4e3afb1c37/repo/_runs/cell_1/experiment2/model_loop/models/pruned/magnitude_privileged_size_principle.py`

### learned_outlier_tolerance_capacity — pruned (experiment2 end of experiment)

**Margin:** 123.3 nats behind parity_veto_sparse_familiar_rules (5.0× dse)

**Hypothesis:** Refinement of `capacity_limited_evidence_integration`: people do the same capacity-limited Bayesian concept learning over rules (separate prior plausibilities for even, odd and the other kinds of rule) and numerical ranges, but how much they suspect the program of producing stray, random numbers is not fixed at 10% — their belief in such "noise" examples is a free parameter inferred from the data. The single change is learning this outlier tolerance: if people trust every example as a genuine member of the concept, ranges that fail to cover all examples are ruled out, so in sets with no fitting rule (e.g. mixed odd and even numbers) they say yes inside the span of the examples and no outside it more sharply than a learner who tolerates stray examples.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/fc3d1e4e3afb1c37/repo/_runs/cell_1/experiment2/model_loop/models/pruned/learned_outlier_tolerance_capacity.py`

### two_item_span_evidence_integration — pruned (experiment2 end of experiment)

**Margin:** 123.5 nats behind parity_veto_sparse_familiar_rules (5.1× dse)

**Hypothesis:** Refinement of `capacity_limited_evidence_integration`: people do the same Bayesian learning over rules (separate prior plausibilities for even, odd and the other kinds of rule) and numerical ranges, but their limited capacity is a fixed span of two examples rather than a smooth power-law dilution — the first two examples of a set are each integrated in full, and only examples beyond that span are integrated partially (a fitted fraction of an example's evidence each). The single change is replacing the incumbent's gradual n^-gamma discount with this two-item span, predicting that two-example sets are judged as conservatively as an ideal learner would (fewer yeses), while three- and four-example sets gain little extra evidence and so generalize relatively broadly — the two-example dip the critique reports.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/fc3d1e4e3afb1c37/repo/_runs/cell_1/experiment2/model_loop/models/pruned/two_item_span_evidence_integration.py`

### exposure_weakened_size_principle — pruned (experiment2 end of experiment)

**Margin:** 125.0 nats behind parity_veto_sparse_familiar_rules (4.8× dse)

**Hypothesis:** Refinement of `even_odd_split_family_prior`: people do the same Bayesian concept learning over the same 45 rules (separate prior plausibilities for even, odd and the other kinds of rule) and all intervals of 1-100 with a tempered size principle, but the strength with which they apply the size principle is not fixed — each further question about the same set of examples within a session weakens it (a fitted power-law decay with the number of earlier trials on that set), so the learner who is repeatedly probed about one set comes to treat its examples as less diagnostic of a small concept and generalizes more broadly. The single change is making the sampling strength depend on the set's exposure history, motivated by the critique that first encounters of a set are judged less generously than later ones, which an exposure-blind learner cannot produce.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/fc3d1e4e3afb1c37/repo/_runs/cell_1/experiment2/model_loop/models/pruned/exposure_weakened_size_principle.py`

### overload_interference_narrow_range — pruned (experiment2 end of experiment)

**Margin:** 126.0 nats behind parity_veto_sparse_familiar_rules (5.0× dse)

**Hypothesis:** Refinement of `narrow_range_prior_capacity_limited`: people do the same Bayesian learning over rules (separate prior plausibilities for even, odd and the other kinds of rule) and numerical ranges, with a prior expectation that the program's range is narrow, but the examples on display interfere with one another in working memory, so every additional example degrades the encoding of *all* the examples by a fixed proportion. The single change is replacing the power-law dilution of evidence with this multiplicative interference: total evidence first rises with set size and can then fall once the display overloads memory, so two-example sets are judged most conservatively and three- and four-example sets more generously, as the critique's two-example dip shows (a power-law dilution can only make evidence grow monotonically).

**Source:** `/scr/wurgaft/auto-psych/agent_trees/fc3d1e4e3afb1c37/repo/_runs/cell_1/experiment2/model_loop/models/pruned/overload_interference_narrow_range.py`

### range_sharpened_exposure_size_principle — pruned (experiment2 end of experiment)

**Margin:** 127.7 nats behind parity_veto_sparse_familiar_rules (5.0× dse)

**Hypothesis:** Refinement of `exposure_weakened_size_principle`: people do the same Bayesian concept learning over the same 45 rules (separate prior plausibilities for even, odd and the other kinds of rule) and all intervals of 1-100, with a size principle that weakens as the same set is probed again, but they do not apply the size principle equally to the two kinds of concept — magnitude ranges are read as tightly sampled (a separate, fitted sampling strength for intervals), so the range spanned by the examples is taken far more literally than an arithmetic rule's extension. The single change is splitting the one sampling strength into a rule strength and an interval strength, motivated by the critique that in mixed-parity sets (where ranges dominate) people separate targets inside the examples' range from those outside it more sharply than the model predicts.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/fc3d1e4e3afb1c37/repo/_runs/cell_1/experiment2/model_loop/models/pruned/range_sharpened_exposure_size_principle.py`

### kind_specific_sampling_assumption — pruned (experiment2 end of experiment)

**Margin:** 128.1 nats behind parity_veto_sparse_familiar_rules (4.9× dse)

**Hypothesis:** People infer the hidden concept by Bayesian learning over mathematical rules (with separate prior plausibilities for even, odd and the other kinds of rule) and numerical ranges, but they make different sampling assumptions for the two kinds of concept: they treat the examples as deliberately drawn from a magnitude range (a strong size principle, so the tightest range covering the examples wins sharply), while they treat a mathematical rule as merely compatible with the examples, discounting how suspiciously the examples fit a small rule (a separately fitted, weaker size principle). This predicts sharp yes/no edges at the examples' smallest and largest values when no rule explains them (e.g. mixed-parity sets), while broad rules such as parity still survive when the examples do fit them.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/fc3d1e4e3afb1c37/repo/_runs/cell_1/experiment2/model_loop/models/pruned/kind_specific_sampling_assumption.py`

### familiarity_relaxed_size_principle — pruned (experiment2 end of experiment)

**Margin:** 129.3 nats behind parity_veto_sparse_familiar_rules (4.7× dse)

**Hypothesis:** People do the same recency-weighted Bayesian concept learning over rules and numerical ranges as the best model, but how strongly they apply the size principle depends on how familiar the example set is: the first time a participant meets a set they read its examples as tightly sampled from a small concept, and each further time they are asked about that same set (having already been probed with other targets for it) they discount that sampling evidence more, so their generalization broadens. This predicts fewer yeses on a set's first encounter than on its repeats, concentrated on targets outside the tightest range or rule covering the examples.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/fc3d1e4e3afb1c37/repo/_runs/cell_1/experiment2/model_loop/models/pruned/familiarity_relaxed_size_principle.py`

### tempered_learned_rule_prior — pruned (experiment2 end of experiment)

**Margin:** 134.5 nats behind parity_veto_sparse_familiar_rules (6.9× dse)

**Hypothesis:** Refinement of `learned_rule_prior_mixture`: people learn concepts by the same Bayesian inference over 45 mathematical rules (each with its own learned prior plausibility) and all intervals of 1-100, but they apply the size principle only partially — they discount the evidence that the examples were sampled from a small concept, so the likelihood of the examples is tempered by a free sampling-strength exponent (1 = full strong sampling, 0 = weak sampling). The single change is grafting this tempered size principle (from `tempered_size_principle_mixture`) onto the learned-rule-prior model, predicting that broad favoured rules and wide intervals keep more posterior weight after a few examples than an ideal strong-sampling learner allows.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/fc3d1e4e3afb1c37/repo/_runs/cell_1/experiment2/model_loop/models/pruned/tempered_learned_rule_prior.py`

### recency_family_prior_sharpened_response — pruned (experiment2 end of experiment)

**Margin:** 137.5 nats behind parity_veto_sparse_familiar_rules (4.9× dse)

**Hypothesis:** Refinement of `recency_weighted_even_odd_family_prior`: people do the same recency-weighted Bayesian concept learning over the same 45 rules (separate prior plausibilities for even, odd and the other kinds of rule) and all intervals of 1-100 with a tempered size principle, but they do not answer by probability matching to their belief that the target is in the concept — they turn that belief into a yes/no answer more (or less) decisively, following the belief's log-odds scaled by a free decisiveness exponent. The single change is grafting this sharpened belief-to-response mapping (from `response_sharpened_tempered_rule_prior`) onto the incumbent, predicting that targets the learner judges probably in (or probably out of) the concept are answered yes (or no) more consistently across people than the graded posterior itself implies.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/fc3d1e4e3afb1c37/repo/_runs/cell_1/experiment2/model_loop/models/pruned/recency_family_prior_sharpened_response.py`

### response_sharpened_tempered_rule_prior — pruned (experiment2 end of experiment)

**Margin:** 138.2 nats behind parity_veto_sparse_familiar_rules (7.1× dse)

**Hypothesis:** Refinement of `tempered_learned_rule_prior`: people do the same Bayesian concept learning over 45 rules (each with its own learned prior plausibility) and all intervals of 1-100 with a tempered size principle, but they turn their belief that the target is in the concept into a yes/no answer more decisively than probability matching — their answer follows the belief's log-odds scaled by a free decisiveness exponent, so targets the learner thinks are probably in (or probably out of) the concept are answered yes (or no) more consistently than the belief itself. The single change is this sharpened belief-to-response mapping (a one-parameter log-odds scaling, from `llo_distorted_concept_posterior`) grafted onto `tempered_learned_rule_prior`, motivated by the critique that people's repeated answers to the same set and target are more unanimous than the graded posterior predicts.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/fc3d1e4e3afb1c37/repo/_runs/cell_1/experiment2/model_loop/models/pruned/response_sharpened_tempered_rule_prior.py`

### surprise_weighted_sequential_updating — pruned (experiment2 end of experiment)

**Margin:** 151.0 nats behind parity_veto_sparse_familiar_rules (5.2× dse)

**Hypothesis:** People read the examples one at a time, revising a running Bayesian guess about the concept (over mathematical rules, with separate prior plausibilities for even, odd and the other kinds of rule, and numerical ranges), and how much each new example counts depends on how surprising it is given the guess formed from the examples before it: an example the current guess predicted well is skimmed over, while an example that violates it (breaks the parity pattern, falls outside the range seen so far) captures attention and is weighted more heavily. Because the weight depends on the preceding examples, the same set shown in a different order is generalized differently, and pattern-breaking sets such as mixed-parity ones end up with their range taken more sharply than an order-blind learner would allow.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/fc3d1e4e3afb1c37/repo/_runs/cell_1/experiment2/model_loop/models/pruned/surprise_weighted_sequential_updating.py`

### tempered_size_principle_mixture — pruned (experiment1 end of experiment)

**Margin:** 160.0 nats behind familiarity_relaxed_size_principle (7.2× dse)

**Hypothesis:** Refinement of `bigelow_piantadosi_mixture`: people do the same Bayesian concept learning over the same 45 rules and all intervals of 1-100, but they apply the size principle only partially — they discount the evidence that the examples were sampled from a small concept, so the likelihood of the examples is tempered by a free "sampling strength" exponent (1 = the incumbent's full strong sampling, 0 = weak sampling). With weaker sampling, broad rules such as odd/even and digit rules keep more posterior weight against tight intervals, predicting stronger parity generalization, more yeses to far targets and less pure numerical-proximity generalization.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/fc3d1e4e3afb1c37/repo/_runs/cell_1/experiment1/model_loop/models/pruned/tempered_size_principle_mixture.py`

### llo_distorted_concept_posterior — pruned (experiment1 end of experiment)

**Margin:** 171.0 nats behind familiarity_relaxed_size_principle (7.9× dse)

**Hypothesis:** People infer the hidden concept as an ideal Bayesian learner would — over the same 45 mathematical rules (equally likely a priori) and all intervals of 1-100, with the size principle — but they misread their own belief when turning it into a yes/no answer: the posterior probability that the target is in the concept is distorted linearly in log-odds (Zhang & Maloney-style probability weighting), compressed toward a reference point. As a result, near-certain members get fewer yeses and near-certain non-members more yeses than the ideal learner predicts, while the ordering of targets is left intact.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/fc3d1e4e3afb1c37/repo/_runs/cell_1/experiment1/model_loop/models/pruned/llo_distorted_concept_posterior.py`

### learned_rule_prior_mixture — pruned (experiment2 end of experiment)

**Margin:** 179.0 nats behind parity_veto_sparse_familiar_rules (6.8× dse)

**Hypothesis:** Refinement of bigelow_piantadosi_mixture: people learn concepts by the same Bayesian inference over 45 mathematical rules and all intervals of 1-100 with the size principle, but they do not hold the rules as equally likely a priori — each rule (e.g. odd, even, ends in a digit) carries its own prior plausibility, inferred from the data. The single change is replacing the uniform prior over rules with a learned, non-uniform one (the core idea of Bigelow & Piantadosi's prior inference), so favoured rules such as parity and shared final digits can dominate interval-based similarity when the examples support them.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/fc3d1e4e3afb1c37/repo/_runs/cell_1/experiment2/model_loop/models/pruned/learned_rule_prior_mixture.py`

### criterion_thresholded_concept_belief — pruned (experiment2 end of experiment)

**Margin:** 180.6 nats behind parity_veto_sparse_familiar_rules (6.0× dse)

**Hypothesis:** People infer the hidden concept by ordinary Bayesian learning over mathematical rules (with separate prior plausibilities for even, odd and the other kinds of rule) and numerical ranges with a tempered size principle, but they do not answer by probability matching: they say yes only when their belief that the target is in the concept clears an internal criterion of how sure they must be, applied with some trial-to-trial noise. This predicts that targets with intermediate support — common after two examples, when several rules and ranges are still in play — are rejected more often than their belief implies, so moderately uncertain sets get fewer yeses than probability matching allows, while clearly supported and clearly unsupported targets are answered nearly unanimously.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/fc3d1e4e3afb1c37/repo/_runs/cell_1/experiment2/model_loop/models/pruned/criterion_thresholded_concept_belief.py`

### lapse_guess_concept_learning — pruned (experiment1 end of experiment)

**Margin:** 200.8 nats behind familiarity_relaxed_size_principle (9.7× dse)

**Hypothesis:** People infer the hidden concept as an ideal Bayesian learner would — over the same 45 mathematical rules (equally likely a priori) and all intervals of 1-100, with the size principle — and normally answer by probability matching to their belief that the target is in the concept, but on a fixed fraction of trials they do not consult that belief at all and instead guess, with their own overall bias toward "yes" or "no". This lapse-and-guess decision rule predicts a floor of yeses for clear non-members and a ceiling below one for clear members (including the examples themselves), both set by the same lapse rate and guessing bias and independent of how certain the inference is.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/fc3d1e4e3afb1c37/repo/_runs/cell_1/experiment1/model_loop/models/pruned/lapse_guess_concept_learning.py`

### tempered_rules_similarity — pruned (experiment1 end of experiment)

**Margin:** 207.7 nats behind familiarity_relaxed_size_principle (7.8× dse)

**Hypothesis:** Refinement of `tenenbaum_rules_similarity`: people do Bayesian concept learning over the same mathematical rules and numerical intervals, but update conservatively — they under-weight the evidence of the examples relative to their prior (the example likelihood is raised to a fitted power below one, a weakened size principle). As a result broad concepts like "even" or "odd" and wide intervals survive a few examples better than for an ideal learner, producing more yeses to distant parity-matching targets and less sharply peaked generalization to numbers right next to the examples.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/fc3d1e4e3afb1c37/repo/_runs/cell_1/experiment1/model_loop/models/pruned/tempered_rules_similarity.py`

### featural_exemplar_similarity — pruned (experiment1 end of experiment)

**Margin:** 211.1 nats behind familiarity_relaxed_size_principle (8.5× dse)

**Hypothesis:** People do not induce a hidden rule; they judge a target by its featural similarity to the remembered examples (exemplar generalization over symbolic number features). A target is accepted in proportion to its average similarity to the examples, where similarity decays with the number of salient surface features it fails to share with each example — its parity, its final digit, and its decade — rather than with its numerical distance.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/fc3d1e4e3afb1c37/repo/_runs/cell_1/experiment1/model_loop/models/pruned/featural_exemplar_similarity.py`

### digit_parity_exemplar_similarity — pruned (experiment1 end of experiment)

**Margin:** 219.9 nats behind familiarity_relaxed_size_principle (9.0× dse)

**Hypothesis:** People do not induce a hidden rule at all; they judge a target by its exemplar similarity to the examples shown, where two numbers are similar to the extent they share surface features (same parity, same final digit) and are close in magnitude, and "yes" grows with the target's average similarity to the examples. This disagrees most sharply with the rule-based Bayesian best model on targets that match only some examples' final digit or share the examples' parity without fitting any whole rule (graded yeses where the rule model says no), and on near-in-magnitude targets that break the examples' parity/digit pattern (fewer yeses than interval hypotheses predict).

**Source:** `/scr/wurgaft/auto-psych/agent_trees/fc3d1e4e3afb1c37/repo/_runs/cell_1/experiment1/model_loop/models/pruned/digit_parity_exemplar_similarity.py`

### ones_digit_exemplar_similarity — pruned (experiment1 end of experiment)

**Margin:** 244.0 nats behind familiarity_relaxed_size_principle (9.4× dse)

**Hypothesis:** People judge a target by comparing its final (ones) digit with the final digit of each example, ignoring magnitude and mathematical rules: a target is endorsed in proportion to how many examples it resembles in surface form, where an example with the identical final digit counts fully and an example whose final digit merely has the same parity (odd/even) counts partially. This graded, example-by-example digit matching predicts yeses for targets that share a final digit with only some of the examples and for parity-matching targets far from every example.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/fc3d1e4e3afb1c37/repo/_runs/cell_1/experiment1/model_loop/models/pruned/ones_digit_exemplar_similarity.py`

### recency_weighted_category_exemplar — pruned (experiment1 end of experiment)

**Margin:** 246.8 nats behind familiarity_relaxed_size_principle (9.4× dse)

**Hypothesis:** People do not infer a hidden concept; they judge a target by its summed similarity to the remembered examples (an exemplar account), where two numbers are similar to the extent they are close in magnitude and share membership in stored number categories (even, odd, primes, squares, powers, multiples, final digit, contained digit), with membership in small, distinctive categories counting more than in broad ones. The examples are held in memory with strengths that fade with how far back in the sequence they were shown, so the most recently seen examples dominate the comparison — predicting more yeses to targets resembling the last example than to targets resembling only the first.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/fc3d1e4e3afb1c37/repo/_runs/cell_1/experiment1/model_loop/models/pruned/recency_weighted_category_exemplar.py`

### divisor_overlap_similarity — pruned (experiment1 end of experiment)

**Margin:** 350.0 nats behind familiarity_relaxed_size_principle (10.7× dse)

**Hypothesis:** People judge a target by its arithmetic similarity to the examples rather than by magnitude: two numbers are similar to the extent that they share divisors (the overlap of their sets of divisors greater than one), and the willingness to say "yes" rises smoothly with the target's average divisor-overlap similarity to the examples. Because even numbers all share the divisor 2 while odd numbers share no common divisor, this predicts strong parity-like generalization from even sets, much weaker generalization from odd sets, and graded yeses for targets sharing factors (e.g. multiples of 3 or 5) with some of the examples.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/fc3d1e4e3afb1c37/repo/_runs/cell_1/experiment1/model_loop/models/pruned/divisor_overlap_similarity.py`

### bigelow_piantadosi_mixture — pruned (experiment1 end of experiment)

**Margin:** 393.7 nats behind familiarity_relaxed_size_principle (8.5× dse)

**Hypothesis:** Bayesian concept learning over 45 rules (odd, even, primes, squares, cubes, powers of 2-10, powers of 2 without 32 or with 37, multiples of 3-9, 11 and 12, ending in 0-9, containing the digit 0-9) and every interval of 1-100. A rule is chosen with probability lambda_0, and the rules are equally likely; each interval has prior (1 - lambda_0) / 100^2. Examples are sampled from the concept (the size principle), 10% of them uniformly from 1-100, and "yes" is the posterior probability that the concept contains the target.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/fc3d1e4e3afb1c37/repo/_runs/cell_1/experiment1/model_loop/models/pruned/bigelow_piantadosi_mixture.py`

### selective_attention_exemplar_similarity — pruned (experiment2 end of experiment)

**Margin:** 416.0 nats behind parity_veto_sparse_familiar_rules (13.5× dse)

**Hypothesis:** People do not infer a hidden concept; they judge a target by its average similarity to the stored examples (an exemplar account), where similarity is computed over the number's magnitude and its membership in familiar number categories (even, odd, primes, squares, powers, multiples, final digit, contained digit), and attention is allocated selectively by the examples themselves: the categories that all examples share draw attention (with a fitted dependence on how broad each category is), and whatever attention they draw is taken away from magnitude. So when the examples share a category, targets breaking it are judged dissimilar even if numerically close, but when the examples share nothing (e.g. mixed parity) attention falls entirely on magnitude and yeses follow numerical proximity to the examples.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/fc3d1e4e3afb1c37/repo/_runs/cell_1/experiment2/model_loop/models/pruned/selective_attention_exemplar_similarity.py`

### tenenbaum_rules_similarity — pruned (experiment1 end of experiment)

**Margin:** 536.0 nats behind familiarity_relaxed_size_principle (9.0× dse)

**Hypothesis:** Bayesian concept learning over 33 mathematical rules (odd, even, squares, cubes, primes, multiples of 3-12, powers of 2-10, ending in 1-9) and every interval of 1-100. The rules share a prior fraction lambda equally; the intervals share the rest by an Erlang density on their size (scale sigma). Examples are sampled from the concept (the size principle), 10% of them uniformly from 1-100, and "yes" is the posterior probability that the concept contains the target.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/fc3d1e4e3afb1c37/repo/_runs/cell_1/experiment1/model_loop/models/pruned/tenenbaum_rules_similarity.py`

### feature_contrast_set_similarity — pruned (experiment2 end of experiment)

**Margin:** 555.8 nats behind parity_veto_sparse_familiar_rules (14.5× dse)

**Hypothesis:** People do not infer a hidden concept; they judge the target by its similarity to the example set on a feature-contrast metric: what counts is how many of the properties all the examples share (parity, primes, squares, powers, multiples, final or contained digit, weighted by how rare and thus distinctive each property is) the target also has, relative to the shared properties it lacks, together with how close the target lies to the nearest example on the number line (an exponentially falling gradient). The more similar the target is on this metric, the more likely people say yes, so a target that matches the examples' common distinctive properties is accepted even far from them, while a target that breaks them is accepted only if numerically very close to an example.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/fc3d1e4e3afb1c37/repo/_runs/cell_1/experiment2/model_loop/models/pruned/feature_contrast_set_similarity.py`

### interval_only_size_principle — pruned (experiment1 end of experiment)

**Margin:** 659.8 nats behind familiarity_relaxed_size_principle (17.5× dse)

**Hypothesis:** People generalize from the examples purely by numerical magnitude, ignoring every arithmetic or digit rule: they infer which contiguous range of 1-100 the program draws from, favouring tighter ranges that still cover all the examples (the size principle) under a prior preference for ranges of a typical width, and say "yes" with the probability that the target lies inside the inferred range. This predicts near-certain yeses between the smallest and largest example, a smooth fall-off just outside them that sharpens as more examples are shown, and no special boost for targets that share the examples' parity, final digit or multiplicative structure.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/fc3d1e4e3afb1c37/repo/_runs/cell_1/experiment1/model_loop/models/pruned/interval_only_size_principle.py`

### spread_scaled_distance_gradient — pruned (experiment2 end of experiment)

**Margin:** 738.2 nats behind parity_veto_sparse_familiar_rules (16.5× dse)

**Hypothesis:** People judge a target by its numerical closeness to the nearest example shown, with willingness to say yes falling off exponentially (a Shepard generalization gradient) with the distance between the target and that example on the 1-100 number line. The breadth of this gradient is not fixed but is set by how dispersed the examples are: tightly clustered examples produce a narrow gradient, widely scattered examples a broad one, so the spread of the examples tells people how far from them the program is likely to reach.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/fc3d1e4e3afb1c37/repo/_runs/cell_1/experiment2/model_loop/models/pruned/spread_scaled_distance_gradient.py`
