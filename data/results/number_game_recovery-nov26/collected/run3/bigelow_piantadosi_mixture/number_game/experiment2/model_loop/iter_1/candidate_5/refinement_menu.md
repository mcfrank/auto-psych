# Refinement menu

The models you may refine, other than the incumbent `first_example_anchored_reliability`. Each entry is one mechanism: its hypothesis in full as its author stated it, how it stands, and where its source is. This is a menu, not a list of what is ruled out: a pruned model lost to the best model on the data it was scored on, by the stated margin, but its mechanism may be partly right — and a model that lost narrowly is the most promising target here. Read the source of the model you pick before you write anything.

## Live models other than the incumbent (in the set; best first)

### family_pooled_uncertain_reliability — rank 1, 2.5 ± 3.8 nats behind the best (0.7× dse: statistically tied with the best), ELPD-LOO -2056.6

**Hypothesis:** Refinement of `uncertain_reliability_noise_concepts`: people do the same Bayesian concept learning over the 45 rules and all intervals of 1-100 with the size principle, holding an uncertain belief about how reliable the program is (how often an example is a stray draw from 1-100) and averaging over it for each example set — but their prior over rules is organised by kind of rule rather than set rule by rule. The one change, taken from `family_pooled_rule_prior`, is this family-structured, partially pooled rule prior: each rule's plausibility is drawn around a shared level for its family (parity, primes, squares and cubes, powers, multiples, final digit, contained digit) with an inferred spread, so rules of the same kind (e.g. "contains a 3" and "contains a 7") are about equally plausible unless the evidence says otherwise, which predicts more uniform generalisation across rules within a family than the incumbent's independent rule weights.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/f8fb9bba6e6fa0a9/repo/_runs/cell_1/experiment2/model_loop/models/family_pooled_uncertain_reliability.py`

### family_pooled_win_stay_lose_sample — rank 2, 11.7 ± 4.2 nats behind the best (2.8× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -2065.8

**Hypothesis:** Refinement of `win_stay_lose_sample_concepts`: people learn the concept one example at a time, holding a single working guess (a rule or magnitude range picked by size-principle Bayesian concept learning) that they keep with some probability while it still contains each new example and otherwise re-pick from the posterior given all examples so far — so early examples shape generalisation more than later ones. The one change, taken from `family_pooled_rule_prior`, is the prior from which guesses are picked: rules are a priori plausible by kind (parity, primes, squares and cubes, powers, multiples, final digit, contained digit), each rule's weight drawn around a shared family level with an inferred spread, rather than an independently learned weight for every rule, so rules of the same kind (e.g. "contains a 3" and "contains a 7") are about equally likely to be adopted as the working guess.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/f8fb9bba6e6fa0a9/repo/_runs/cell_1/experiment2/model_loop/models/family_pooled_win_stay_lose_sample.py`

### family_pooled_rule_prior — rank 3, 15.9 ± 4.5 nats behind the best (3.5× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -2069.9

**Hypothesis:** Refinement of `inferred_example_noise_concepts`: people do the same Bayesian concept learning over the 45 rules and all intervals of 1-100 (size principle, an inferred rate at which examples are stray draws), but their prior over rules is organised by kind of rule rather than set rule by rule: each rule's plausibility is drawn around a shared level for its family (parity, primes, squares and cubes, powers, multiples, final digit, contained digit), with an inferred spread, so rules of the same kind (e.g. "contains a 3" and "contains a 7") are about equally plausible unless the evidence says otherwise. The one change is this family-structured, partially pooled rule prior, which replaces the incumbent's independent weight for every rule and predicts more uniform generalisation across rules within a family.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/f8fb9bba6e6fa0a9/repo/_runs/cell_1/experiment2/model_loop/models/family_pooled_rule_prior.py`

### uncertain_reliability_noise_concepts — rank 4, 45.0 ± 10.6 nats behind the best (4.3× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -2099.1

**Hypothesis:** Refinement of `inferred_example_noise_concepts`: people do the same Bayesian concept learning over the 45 rules (each with its own inferred prior weight) and all intervals of 1-100 with the size principle, allowing that some examples are stray draws from 1-100 — but they do not know the program's reliability, so instead of assuming one fixed stray rate they hold a belief over how reliable the program is and average over it for each example set. The one change is this uncertainty about the noise rate, which makes strays exchangeable but correlated within a set: one example that breaks a tight rule makes people suspect the program is unreliable on that set, so they discount the remaining examples too and generalise more broadly, while a set fully consistent with a tight rule is taken as strong evidence for it.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/f8fb9bba6e6fa0a9/repo/_runs/cell_1/experiment2/model_loop/models/uncertain_reliability_noise_concepts.py`

### digit_sum_concepts — rank 5, 47.6 ± 10.8 nats behind the best (4.4× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -2101.7

**Hypothesis:** Refinement of `uncertain_reliability_noise_concepts`: people do the same Bayesian concept learning over the 45 rules and all intervals of 1-100 (size principle, uncertain belief in how often examples are stray draws), but they also entertain digit-arithmetic concepts — "numbers whose digits add up to s", one concept for each digit sum from 1 to 18 — which together hold a small a-priori share of the rule prior (about 4%, inferred around that value) split equally among them. The one change is this enlarged concept space, which predicts that sets whose examples share a digit sum (e.g. 23, 41, 50, 14) draw "yes" for other numbers with that digit sum (32, 5, 68) even though no arithmetic rule, final digit or magnitude range explains them.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/f8fb9bba6e6fa0a9/repo/_runs/cell_1/experiment2/model_loop/models/digit_sum_concepts.py`

### compressed_number_line_ranges — rank 6, 48.1 ± 10.7 nats behind the best (4.5× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -2102.1

**Hypothesis:** People do Bayesian concept learning over mathematical rules and magnitude ranges of 1-100 (size principle, a prior weight per rule, an uncertain belief in how often an example is a stray draw), but they represent magnitude ranges on a compressed mental number line, on which small numbers take up more room than large ones, and they treat a range's examples as spread evenly along that line, not evenly over the integers. So ranges among small numbers feel mentally wide and give weak evidence, which makes generalisation broad, while equally wide ranges among large numbers feel narrow and tighten quickly. Within a range, its lower members are more typical than its upper ones, so people extend a range further upward than downward. How compressed the line is gets inferred, from logarithmic to linear.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/f8fb9bba6e6fa0a9/repo/_runs/cell_1/experiment2/model_loop/models/compressed_number_line_ranges.py`

### inner_loop_model — rank 7, 49.0 ± 11.1 nats behind the best (4.4× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -2103.0

**Hypothesis:** Refinement of `win_stay_lose_sample_concepts`: people learn the concept one example at a time, holding a single working guess (a rule or magnitude range) that they keep with some probability while it still contains each new example and otherwise re-pick from the Bayesian posterior given the examples so far, so early examples shape generalisation more than later ones. The one change, taken from `uncertain_reliability_noise_concepts`, is the posterior people re-pick from: they are unsure how reliable the program is and average over that uncertainty, so strays are correlated within a set. One example that breaks a tight rule makes the remaining ones suspect too and pushes re-picked guesses toward broader concepts, while a fully consistent prefix strongly favours the tight rule. This predicts order effects (from the sticky guess) that interact with how consistent the early examples are.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/f8fb9bba6e6fa0a9/repo/_runs/cell_1/experiment2/model_loop/models/inner_loop_model.py`

### kind_specific_reliability_concepts — rank 8, 69.9 ± 12.0 nats behind the best (5.8× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -2124.0

**Hypothesis:** People do Bayesian concept learning over mathematical rules and magnitude ranges of 1-100 (size principle, a family-organised prior over rules, an uncertain belief about how often an example is a stray draw from 1-100). But their belief about how reliable the program is depends on what kind of program they suppose it is: they expect a program running a mathematical rule to produce strays at a different rate from one producing numbers in a magnitude range, rather than holding one reliability belief whichever kind of concept is true. So a single off-pattern example counts differently against a rule than against a range, which shifts how readily people give up a tight rule for a broad range (or the reverse) when a set contains an outlier.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/f8fb9bba6e6fa0a9/repo/_runs/cell_1/experiment2/model_loop/models/kind_specific_reliability_concepts.py`

### complement_stray_reliability_concepts — rank 9, 80.5 ± 13.1 nats behind the best (6.1× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -2134.6

**Hypothesis:** Refinement of `uncertain_reliability_noise_concepts`: people do the same Bayesian concept learning over the 45 rules (each with its own inferred prior weight) and all intervals of 1-100 with the size principle, and they are just as unsure how reliable the program is on each set — but they think of a stray example as a mistake, a number that does *not* belong to the concept, rather than a draw that could land anywhere in 1-100. The one change is this stray distribution, drawn from the concept's complement: an example that a broad concept (e.g. "even", "contains a 3") misses is better explained as a stray than under a uniform stray generator, whereas a tight concept gains nothing. So people keep broad rules that miss one example more readily, and examples that a concept already contains are always taken as genuine evidence for it.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/f8fb9bba6e6fa0a9/repo/_runs/cell_1/experiment2/model_loop/models/complement_stray_reliability_concepts.py`

### familiar_number_strays — rank 10, 125.2 ± 18.3 nats behind the best (6.9× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -2179.3

**Hypothesis:** Refinement of `family_pooled_uncertain_reliability`: people do the same Bayesian concept learning over the 45 rules (with the family-structured, partially pooled rule prior) and all intervals of 1-100, with the size principle and an uncertain belief about how reliable the program is on each set. The one change is what people think a stray example is: some strays are random numbers from 1-100, but a share of them (how large is inferred) are "familiar" numbers, i.e. a member of some well-known rule picked at random, so how likely a number is to turn up as a stray depends on how many small, familiar rules produce it, whichever rules the learner currently favours. Thus examples that many small rules produce (1, 2, 4, 8, 16, 64, 81, 100) are easily written off as strays and constrain the concept less, while unremarkable examples that few rules produce (e.g. 58, 87, 94) are hard to dismiss and the concept has to cover them.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/f8fb9bba6e6fa0a9/repo/_runs/cell_1/experiment2/model_loop/models/familiar_number_strays.py`

### strong_weak_sampling_mixture — rank 11, 135.1 ± 18.0 nats behind the best (7.5× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -2189.1

**Hypothesis:** Refinement of `family_pooled_uncertain_reliability`: people do the same Bayesian concept learning over the 45 rules (plausibility organised by rule family) and all intervals of 1-100, with an uncertain belief in how often examples are stray draws. The one change is the sampling assumption: people are unsure whether the examples were sampled at random from the concept (so a small concept that fits gains strongly, the size principle) or were just numbers the program happened to produce that are consistent with it (so a concept's size carries no evidence), and they weigh both possibilities, with how much they trust random sampling inferred. This predicts broader, less size-sensitive generalisation than the incumbent, except where the examples fit a tight concept so well that the random-sampling explanation dominates.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/f8fb9bba6e6fa0a9/repo/_runs/cell_1/experiment2/model_loop/models/strong_weak_sampling_mixture.py`

### weak_sampling_concepts — rank 12, 205.2 ± 22.7 nats behind the best (9.0× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -2259.2

**Hypothesis:** People treat the examples only as numbers the program is allowed to produce, not as random samples drawn from its concept (weak sampling): a candidate rule or magnitude range gains plausibility merely by containing the examples (and loses it for each example it misses, which they may write off as a stray), but not for being small, so a concept's plausibility rests on its prior — one plausibility per kind of rule (parity, primes, squares and cubes, powers, multiples, final digit, contained digit) shared by the rules of that kind, and a preference for narrower ranges — and does not tighten as more consistent examples arrive. This predicts that adding examples consistent with both a broad and a tight concept (e.g. 16, 8, 64 under "even" and "powers of 2") leaves generalisation about as broad as after the first example, unlike the size principle.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/f8fb9bba6e6fa0a9/repo/_runs/cell_1/experiment2/model_loop/models/weak_sampling_concepts.py`

## Pruned models (out of the set; narrowest margin first)

### digit_slip_stray_concepts — pruned (experiment1 end of experiment)

**Margin:** 3.8 nats behind iter4_candidate5; retired to keep the live set at 8 models: ELPD-LOO rank 9 of 19

**Hypothesis:** Refinement of `uncertain_reliability_noise_concepts`: people do the same Bayesian concept learning over the 45 rules (each with its own inferred prior weight) and all intervals of 1-100 with the size principle, and they are just as unsure how reliable the program is on each set. The one change is what they think a stray example is: besides a number drawn at random from 1-100, a stray can be a slip, i.e. a member of the concept miswritten (one digit changed, added or dropped, or two digits swapped) so that it lands outside the concept, and how often strays are slips is inferred. So an example that is a slip of a tight concept's member (83 next to 38, 58, 68 under "ends in 8") is easily explained away and the tight concept survives, while an example that no slip of any member could produce counts against the concept as fully as before.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/f8fb9bba6e6fa0a9/repo/_runs/cell_1/experiment1/model_loop/models/pruned/digit_slip_stray_concepts.py`

### inferred_example_noise_concepts — pruned (experiment1 end of experiment)

**Margin:** 3.9 nats behind iter4_candidate5; retired to keep the live set at 8 models: ELPD-LOO rank 10 of 19

**Hypothesis:** Refinement of `bigelow_piantadosi_independent`: people do the same Bayesian concept learning over its 45 rules (each with its own inferred prior weight) and all intervals of 1-100 with the size principle, but how much they trust the examples to come from the concept is a free parameter rather than fixed at 90%. The one change is this inferred example-noise rate: the more people believe some examples are stray draws from 1-100, the more readily they keep a tight rule or range that misses one example, rather than retreating to a broad concept that covers all of them.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/f8fb9bba6e6fa0a9/repo/_runs/cell_1/experiment1/model_loop/models/pruned/inferred_example_noise_concepts.py`

### without_replacement_example_sampling — pruned (experiment1 end of experiment)

**Margin:** 4.5 nats behind iter4_candidate5; retired to keep the live set at 8 models: ELPD-LOO rank 11 of 19

**Hypothesis:** Refinement of the size-principle Bayesian concept learner (45 rules with their own prior weights plus all magnitude intervals of 1-100, with an uncertain belief about how often an example is a stray draw): people know the program never repeats a number within a set, so they reason as if the genuine examples were drawn from the concept without replacement, not independently. This makes each additional example much stronger evidence for small concepts that the examples nearly exhaust (cubes, high powers, very narrow ranges) while leaving large concepts unchanged, so the disagreement with the best model lies in sets that sit inside a tiny concept: there people should commit to that concept and reject neighbouring or merely broader-rule targets more sharply than independent sampling predicts.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/f8fb9bba6e6fa0a9/repo/_runs/cell_1/experiment1/model_loop/models/pruned/without_replacement_example_sampling.py`

### bigelow_piantadosi_independent — pruned (experiment1 end of experiment)

**Margin:** 4.6 nats behind iter4_candidate5; retired to keep the live set at 8 models: ELPD-LOO rank 12 of 19

**Hypothesis:** Bayesian concept learning over 45 rules (odd, even, primes, squares, cubes, powers of 2-10, powers of 2 without 32 or with 37, multiples of 3-9, 11 and 12, ending in 0-9, containing the digit 0-9) and every interval of 1-100, with a prior weight inferred for each rule separately and a share lambda_0 for the rules as a group; each interval has prior (1 - lambda_0) / 100^2. Examples are sampled from the concept (the size principle), 10% of them uniformly from 1-100, and "yes" is the posterior probability that the concept contains the target.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/f8fb9bba6e6fa0a9/repo/_runs/cell_1/experiment1/model_loop/models/pruned/bigelow_piantadosi_independent.py`

### win_stay_lose_sample_concepts — pruned (experiment1 end of experiment)

**Margin:** 4.7 nats behind iter4_candidate5; retired to keep the live set at 8 models: ELPD-LOO rank 13 of 19

**Hypothesis:** People learn the concept one example at a time, holding a single working guess rather than the full posterior: after the first example they pick a rule or magnitude range by Bayesian concept learning (size principle, a prior weight per rule), and as each later example arrives they keep their current guess with some probability if it still contains that example (win-stay), and otherwise re-pick a guess from the Bayesian posterior given all examples seen so far (lose-sample). The one free mechanism is this stickiness of the running guess, which makes judgments depend on the order of the examples: a broad concept that fits the first example (e.g. "even" after 16) survives later examples it happens to contain (8, 64) instead of being displaced by a tighter rule (powers of 2), so early examples shape generalization more than later ones.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/f8fb9bba6e6fa0a9/repo/_runs/cell_1/experiment1/model_loop/models/pruned/win_stay_lose_sample_concepts.py`

### interval_width_prior_concepts — pruned (experiment1 end of experiment)

**Margin:** 4.9 nats behind iter4_candidate5; retired to keep the live set at 8 models: ELPD-LOO rank 14 of 19

**Hypothesis:** Refinement of `bigelow_piantadosi_independent`: the same Bayesian concept learner over its 45 rules (each with its own inferred prior weight) and all intervals of 1-100, with the size principle, but people's prior over magnitude intervals depends on the interval's width — each interval's prior is proportional to its width raised to an inferred exponent (0 recovers the incumbent's uniform interval prior; negative values mean narrow ranges like "numbers in the 60s" are a priori more plausible than wide ones). The single change is this width-dependent interval prior, which governs how readily people generalise to nearby versus distant magnitudes when no rule explains the examples.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/f8fb9bba6e6fa0a9/repo/_runs/cell_1/experiment1/model_loop/models/pruned/interval_width_prior_concepts.py`

### any_number_generator_concepts — pruned (experiment1 end of experiment)

**Margin:** 5.1 nats behind iter4_candidate5; retired to keep the live set at 8 models: ELPD-LOO rank 15 of 19

**Hypothesis:** Refinement of `bigelow_piantadosi_independent`: people do the same Bayesian concept learning over its 45 rules (each with its own inferred prior weight) and all intervals of 1-100 with the size principle, but they also seriously entertain the possibility that the program has no structure at all and simply produces any number from 1 to 100, giving this "anything goes" hypothesis its own inferred prior weight. The one change is this unstructured-generator hypothesis: because it explains each example only as well as chance, the size principle makes it lose to any fitting rule or range as examples accumulate, so it predicts stray "yes" answers to unrelated targets that are common after one or two examples (or when nothing fits) and vanish after four consistent ones — unlike a constant response lapse.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/f8fb9bba6e6fa0a9/repo/_runs/cell_1/experiment1/model_loop/models/pruned/any_number_generator_concepts.py`

### bayesian_rules_with_response_lapse — pruned (experiment1 end of experiment)

**Margin:** 5.5 nats behind iter4_candidate5 (2.0× dse)

**Hypothesis:** Refinement of bigelow_piantadosi_independent: people do the same Bayesian concept learning over the 45 rules and all intervals (size principle, a prior weight inferred per rule), but on a fixed fraction of trials they do not consult that posterior at all and answer yes or no at random. The one change is this response-lapse rate, which lets the model explain "no" answers to targets that are themselves examples and stray "yes" answers to targets that no plausible concept contains.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/f8fb9bba6e6fa0a9/repo/_runs/cell_1/experiment1/model_loop/models/pruned/bayesian_rules_with_response_lapse.py`

### log_odds_compressed_concept_posterior — pruned (experiment1 end of experiment)

**Margin:** 5.5 nats behind iter4_candidate5 (2.0× dse)

**Hypothesis:** Refinement of `bigelow_piantadosi_independent`: people do the same Bayesian concept learning over the 45 rules and all intervals of 1-100 (size principle, a prior weight inferred per rule), but they do not report the posterior probability that the target belongs faithfully — their subjective sense of that probability is distorted linearly in log-odds, so confident beliefs are pulled toward (or pushed away from) even odds, as in classic probability-weighting. The one change is this single probability-distortion slope, which predicts that near-certain "yes" and near-certain "no" cases are both answered less extremely, with the compression strongest for the most confident judgments, unlike a fixed random-response lapse that adds the same floor everywhere.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/f8fb9bba6e6fa0a9/repo/_runs/cell_1/experiment1/model_loop/models/pruned/log_odds_compressed_concept_posterior.py`

### leaky_example_memory_concepts — pruned (experiment1 end of experiment)

**Margin:** 5.7 nats behind iter4_candidate5 (2.1× dse)

**Hypothesis:** People generalize by Bayesian concept learning over mathematical rules and magnitude intervals (the size principle, a prior weight per rule), but their working memory is leaky: each example shown is independently retained with some probability and the forgotten ones play no part in the inference. So a judgment is the rule-and-interval posterior computed from whichever examples survive, which predicts weaker rule-based tightening and broader generalization for larger example sets than full Bayesian learning, and occasional "no" answers to targets that were themselves examples.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/f8fb9bba6e6fa0a9/repo/_runs/cell_1/experiment1/model_loop/models/pruned/leaky_example_memory_concepts.py`

### tenenbaum_digit_rule_inventory — pruned (experiment1 end of experiment)

**Margin:** 6.6 nats behind iter4_candidate5; retired to keep the live set at 8 models: ELPD-LOO rank 16 of 19

**Hypothesis:** Refinement of `tenenbaum_rules_similarity`: people do Tenenbaum's Bayesian concept learning (rules share one prior fraction equally, intervals are weighted by an Erlang density on their size, examples are sampled from the concept), but the rules they entertain are the richer inventory of the Bigelow-Piantadosi grammar — notably "contains the digit d" and "ends in 0", plus the two near-miss powers-of-2 sets — instead of Tenenbaum's 33. The one change is the hypothesis space; the claim is that what the people consider, not a separately learned weight for every rule, is what separates the two models, so equal rule weights with two free parameters should suffice.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/f8fb9bba6e6fa0a9/repo/_runs/cell_1/experiment1/model_loop/models/pruned/tenenbaum_digit_rule_inventory.py`

### exemplar_retrieved_number_concepts — pruned (experiment1 end of experiment)

**Margin:** 7.1 nats behind iter4_candidate5 (2.2× dse)

**Hypothesis:** People do not reason about which concept is literally consistent with the examples; they hold number concepts in memory as stored sets of member numbers (the familiar rules like even, squares, powers of 2, multiples, final and contained digits, and magnitude ranges), and judge by exemplar similarity on the number line. Each stored concept is retrieved in proportion to how similar the examples are, on average, to its stored members (a concept with few members that lie near every example is retrieved most strongly, so an example close to but not in a concept still partly supports it), and a target is endorsed by its similarity to the nearest stored member of the retrieved concepts, so endorsement falls off gradedly for targets just outside a concept rather than dropping to zero.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/f8fb9bba6e6fa0a9/repo/_runs/cell_1/experiment1/model_loop/models/pruned/exemplar_retrieved_number_concepts.py`

### pedagogical_teacher_sampling — pruned (experiment1 end of experiment)

**Margin:** 7.2 nats behind iter4_candidate5 (2.1× dse)

**Hypothesis:** People reason about the examples as a communicative act: they assume the program's examples were picked by a helpful teacher who chooses numbers that point to the intended concept and away from other plausible concepts, rather than drawn at random from it. So an example that belongs to few other plausible rules or ranges (e.g. 64, which says "powers of 2") counts as strong evidence for the concepts containing it, while an example that belongs to many (e.g. 16 or 50) counts as weak evidence; the strength of this pedagogical assumption is inferred, and with none people reduce to ordinary size-principle Bayesian concept learning over the 45 rules and all intervals of 1-100.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/f8fb9bba6e6fa0a9/repo/_runs/cell_1/experiment1/model_loop/models/pruned/pedagogical_teacher_sampling.py`

### inferred_outlier_rate_rule_inventory — pruned (experiment1 end of experiment)

**Margin:** 7.6 nats behind iter4_candidate5; retired to keep the live set at 8 models: ELPD-LOO rank 17 of 19

**Hypothesis:** Refinement of `tenenbaum_digit_rule_inventory`: people do the same Bayesian concept learning over the 45 Bigelow-Piantadosi rules (one equal, shared prior weight) and all magnitude intervals (Erlang prior on size), with examples sampled from the concept, but how strongly they believe an example may be a random stray rather than a draw from the concept is a property of the learner, inferred from the data instead of fixed at 10%. The one change is this inferred outlier rate: a high rate means people tolerate examples that break a rule and weaken the size principle (generalising more broadly), while a low rate means a single inconsistent example rules a concept out and tight concepts dominate.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/f8fb9bba6e6fa0a9/repo/_runs/cell_1/experiment1/model_loop/models/pruned/inferred_outlier_rate_rule_inventory.py`

### rule_generated_stray_concepts — pruned (experiment1 end of experiment)

**Margin:** 8.5 nats behind iter4_candidate5; retired to keep the live set at 8 models: ELPD-LOO rank 18 of 19

**Hypothesis:** Refinement of `uncertain_reliability_noise_concepts`: people do the same Bayesian concept learning over the 45 rules (each with its own inferred prior weight) and all intervals of 1-100 with the size principle, and are just as unsure how reliable the program is on each set — but they think of a stray example as a number that some *other* mathematical rule produced (a rule picked by how plausible they find rules, then any of its members), rather than a draw that is equally likely to be any number from 1 to 100. The one change is this rule-generated stray distribution: examples that many small rules produce (1, 2, 4, 8, 16, 64, 81, 100) are easily written off as strays, so they constrain the concept less, while "unremarkable" examples that few rules produce (e.g. 58, 87, 94) are hard to dismiss and the concept must cover them.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/f8fb9bba6e6fa0a9/repo/_runs/cell_1/experiment1/model_loop/models/pruned/rule_generated_stray_concepts.py`

### response_bias_concept_learner — pruned (experiment1 end of experiment)

**Margin:** 9.3 nats behind iter4_candidate5; retired to keep the live set at 8 models: ELPD-LOO rank 19 of 19

**Hypothesis:** People infer the hidden concept by simple Bayesian concept learning (rules from the Bigelow-Piantadosi inventory sharing one equal prior weight, magnitude intervals weighted by their size, examples sampled from the concept), but they answer with a response bias: a constant shift in the log-odds of their belief toward "yes" (liberal criterion) or toward "no" (conservative criterion), the same on every trial. The single change from the simplest Bayesian model is this criterion shift at the decision stage, which predicts that uncertain targets (near 50% belief) move most while near-certain ones barely move, and that both the endorsement of example-consistent targets and the rejection of implausible ones shift in the same direction.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/f8fb9bba6e6fa0a9/repo/_runs/cell_1/experiment1/model_loop/models/pruned/response_bias_concept_learner.py`

### predictive_surprise_endorsement — pruned (experiment1 end of experiment)

**Margin:** 51.2 nats behind iter4_candidate5 (5.3× dse)

**Hypothesis:** People answer "would the program produce this number?" by imagining the target as the program's next output and asking how expected it would be, not by checking whether it belongs to the concept they have inferred. They learn the concept by size-principle Bayesian inference over the 45 rules and all intervals of 1-100, unsure how reliable the program is. Then they say yes in proportion to how much more likely the target is as the next example than a random number, so members of tight concepts (64 after "16, 8") are endorsed more strongly than members of broad concepts that are just as certainly consistent (any even number after "16, 8"), and endorsement is graded rather than all-or-none.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/f8fb9bba6e6fa0a9/repo/_runs/cell_1/experiment1/model_loop/models/pruned/predictive_surprise_endorsement.py`

### shared_property_contrast_similarity — pruned (experiment1 end of experiment)

**Margin:** 120.7 nats behind iter4_candidate5 (6.9× dse)

**Hypothesis:** People do not infer a single hidden concept; they judge a target by its feature-based similarity to the examples as a group. Each number is represented by its mathematical properties (parity, primality, being a square, cube or power, multiples, final digit, digits it contains) and by the magnitude bands it falls in, and a target is endorsed to the extent that it possesses the properties the examples have in common, with each property's weight growing with how many of the examples share it and with how distinctive (rare among 1-100) it is. Endorsement therefore falls off gradedly as the target lacks more of the salient shared properties, with no posterior averaging over concepts and no all-or-none membership.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/f8fb9bba6e6fa0a9/repo/_runs/cell_1/experiment1/model_loop/models/pruned/shared_property_contrast_similarity.py`

### map_single_concept_commitment — pruned (experiment1 end of experiment)

**Margin:** 244.6 nats behind iter4_candidate5 (9.6× dse)

**Hypothesis:** People do not average over all the concepts consistent with the examples; they commit to the single concept that best explains them — the most probable rule or magnitude interval under a size-principle Bayesian evaluation with a flat prior (rules as a group as likely as intervals as a group) — and answer "yes" exactly when the target belongs to that one concept, apart from occasional response errors. This predicts all-or-none generalization: no graded falloff near the examples' range and no partial endorsement of targets that only some plausible concepts contain (e.g. 32 after "16,8,64" is rejected if the smaller "powers of 2 without 32" wins).

**Source:** `/scr/wurgaft/auto-psych/agent_trees/f8fb9bba6e6fa0a9/repo/_runs/cell_1/experiment1/model_loop/models/pruned/map_single_concept_commitment.py`

### tenenbaum_rules_similarity — pruned (experiment1 end of experiment)

**Margin:** 316.2 nats behind iter4_candidate5 (6.6× dse)

**Hypothesis:** Bayesian concept learning over 33 mathematical rules (odd, even, squares, cubes, primes, multiples of 3-12, powers of 2-10, ending in 1-9) and every interval of 1-100. The rules share a prior fraction lambda equally; the intervals share the rest by an Erlang density on their size (scale sigma). Examples are sampled from the concept (the size principle), 10% of them uniformly from 1-100, and "yes" is the posterior probability that the concept contains the target.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/f8fb9bba6e6fa0a9/repo/_runs/cell_1/experiment1/model_loop/models/pruned/tenenbaum_rules_similarity.py`

### digit_exemplar_similarity — pruned (experiment1 end of experiment)

**Margin:** 353.5 nats behind iter4_candidate5 (10.6× dse)

**Hypothesis:** People do not induce a rule at all; they judge a target by its summed similarity to the remembered examples, where numbers are represented by their two decimal digits: a target feels like it "belongs" to the extent that its tens digit is close to an example's tens digit and its ones digit matches an example's ones digit. Attention is split between the tens (magnitude) and ones (final-digit) dimensions, similarity decays exponentially with this digit-wise distance, and the probability of saying yes rises with the average similarity of the target to the examples.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/f8fb9bba6e6fa0a9/repo/_runs/cell_1/experiment1/model_loop/models/pruned/digit_exemplar_similarity.py`

### digit_overlap_exemplar_similarity — pruned (experiment1 end of experiment)

**Margin:** 389.6 nats behind iter4_candidate5 (11.9× dse)

**Hypothesis:** People do not induce mathematical rules or magnitude intervals at all; they judge a target by its surface resemblance to the examples, treating each number as the set of decimal digits it is written with, and say "yes" in proportion to the average digit-overlap similarity between the target and the examples (an exemplar-similarity process over written digits). This disagrees most sharply with the rule-based Bayesian model on targets that share a digit with some examples but fit no rule or interval consistent with all of them (e.g. 12 after "31,19,49,100"), and on rule-consistent targets written with entirely different digits (e.g. 2 after "16,8,64"), where it predicts low endorsement.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/f8fb9bba6e6fa0a9/repo/_runs/cell_1/experiment1/model_loop/models/pruned/digit_overlap_exemplar_similarity.py`

### spread_scaled_range_generalization — pruned (experiment1 end of experiment)

**Margin:** 421.3 nats behind iter4_candidate5 (12.7× dse)

**Hypothesis:** People do not search over mathematical rules; they treat the examples as a sample from a band of magnitudes and generalize by how far the target falls outside the span the examples cover, with a tolerance that grows with how spread out the examples are. A target inside the span of the examples is endorsed at a constant high rate, and endorsement decays smoothly with the target's distance beyond the nearest edge, more slowly for widely spaced examples than for tightly clustered ones (or a single example).

**Source:** `/scr/wurgaft/auto-psych/agent_trees/f8fb9bba6e6fa0a9/repo/_runs/cell_1/experiment1/model_loop/models/pruned/spread_scaled_range_generalization.py`
