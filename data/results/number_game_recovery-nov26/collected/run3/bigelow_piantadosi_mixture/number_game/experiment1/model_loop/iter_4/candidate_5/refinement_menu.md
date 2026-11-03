# Refinement menu

The models you may refine, other than the incumbent `uncertain_reliability_noise_concepts`. Each entry is one mechanism: its hypothesis in full as its author stated it, how it stands, and where its source is. This is a menu, not a list of what is ruled out: a pruned model lost to the best model on the data it was scored on, by the stated margin, but its mechanism may be partly right — and a model that lost narrowly is the most promising target here. Read the source of the model you pick before you write anything.

## Live models other than the incumbent (in the set; best first)

### digit_sum_concepts — rank 1, 0.1 ± 0.4 nats behind the best (0.1× dse: statistically tied with the best), ELPD-LOO -835.7

**Hypothesis:** Refinement of `uncertain_reliability_noise_concepts`: people do the same Bayesian concept learning over the 45 rules and all intervals of 1-100 (size principle, uncertain belief in how often examples are stray draws), but they also entertain digit-arithmetic concepts — "numbers whose digits add up to s", one concept for each digit sum from 1 to 18 — which together hold a small a-priori share of the rule prior (about 4%, inferred around that value) split equally among them. The one change is this enlarged concept space, which predicts that sets whose examples share a digit sum (e.g. 23, 41, 50, 14) draw "yes" for other numbers with that digit sum (32, 5, 68) even though no arithmetic rule, final digit or magnitude range explains them.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/f8fb9bba6e6fa0a9/repo/_runs/cell_1/experiment1/model_loop/models/digit_sum_concepts.py`

### family_pooled_uncertain_reliability — rank 2, 0.3 ± 3.5 nats behind the best (0.1× dse: statistically tied with the best), ELPD-LOO -836.0

**Hypothesis:** Refinement of `uncertain_reliability_noise_concepts`: people do the same Bayesian concept learning over the 45 rules and all intervals of 1-100 with the size principle, holding an uncertain belief about how reliable the program is (how often an example is a stray draw from 1-100) and averaging over it for each example set — but their prior over rules is organised by kind of rule rather than set rule by rule. The one change, taken from `family_pooled_rule_prior`, is this family-structured, partially pooled rule prior: each rule's plausibility is drawn around a shared level for its family (parity, primes, squares and cubes, powers, multiples, final digit, contained digit) with an inferred spread, so rules of the same kind (e.g. "contains a 3" and "contains a 7") are about equally plausible unless the evidence says otherwise, which predicts more uniform generalisation across rules within a family than the incumbent's independent rule weights.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/f8fb9bba6e6fa0a9/repo/_runs/cell_1/experiment1/model_loop/models/family_pooled_uncertain_reliability.py`

### complement_stray_reliability_concepts — rank 3, 0.3 ± 3.2 nats behind the best (0.1× dse: statistically tied with the best), ELPD-LOO -836.0

**Hypothesis:** Refinement of `uncertain_reliability_noise_concepts`: people do the same Bayesian concept learning over the 45 rules (each with its own inferred prior weight) and all intervals of 1-100 with the size principle, and they are just as unsure how reliable the program is on each set — but they think of a stray example as a mistake, a number that does *not* belong to the concept, rather than a draw that could land anywhere in 1-100. The one change is this stray distribution, drawn from the concept's complement: an example that a broad concept (e.g. "even", "contains a 3") misses is better explained as a stray than under a uniform stray generator, whereas a tight concept gains nothing. So people keep broad rules that miss one example more readily, and examples that a concept already contains are always taken as genuine evidence for it.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/f8fb9bba6e6fa0a9/repo/_runs/cell_1/experiment1/model_loop/models/complement_stray_reliability_concepts.py`

### family_pooled_rule_prior — rank 4, 1.1 ± 3.9 nats behind the best (0.3× dse: statistically tied with the best), ELPD-LOO -836.8

**Hypothesis:** Refinement of `inferred_example_noise_concepts`: people do the same Bayesian concept learning over the 45 rules and all intervals of 1-100 (size principle, an inferred rate at which examples are stray draws), but their prior over rules is organised by kind of rule rather than set rule by rule: each rule's plausibility is drawn around a shared level for its family (parity, primes, squares and cubes, powers, multiples, final digit, contained digit), with an inferred spread, so rules of the same kind (e.g. "contains a 3" and "contains a 7") are about equally plausible unless the evidence says otherwise. The one change is this family-structured, partially pooled rule prior, which replaces the incumbent's independent weight for every rule and predicts more uniform generalisation across rules within a family.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/f8fb9bba6e6fa0a9/repo/_runs/cell_1/experiment1/model_loop/models/family_pooled_rule_prior.py`

### family_pooled_win_stay_lose_sample — rank 5, 1.7 ± 4.1 nats behind the best (0.4× dse: statistically tied with the best), ELPD-LOO -837.3

**Hypothesis:** Refinement of `win_stay_lose_sample_concepts`: people learn the concept one example at a time, holding a single working guess (a rule or magnitude range picked by size-principle Bayesian concept learning) that they keep with some probability while it still contains each new example and otherwise re-pick from the posterior given all examples so far — so early examples shape generalisation more than later ones. The one change, taken from `family_pooled_rule_prior`, is the prior from which guesses are picked: rules are a priori plausible by kind (parity, primes, squares and cubes, powers, multiples, final digit, contained digit), each rule's weight drawn around a shared family level with an inferred spread, rather than an independently learned weight for every rule, so rules of the same kind (e.g. "contains a 3" and "contains a 7") are about equally likely to be adopted as the working guess.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/f8fb9bba6e6fa0a9/repo/_runs/cell_1/experiment1/model_loop/models/family_pooled_win_stay_lose_sample.py`

### inferred_example_noise_concepts — rank 6, 3.9 ± 2.2 nats behind the best (1.8× dse: statistically tied with the best), ELPD-LOO -839.6

**Hypothesis:** Refinement of `bigelow_piantadosi_independent`: people do the same Bayesian concept learning over its 45 rules (each with its own inferred prior weight) and all intervals of 1-100 with the size principle, but how much they trust the examples to come from the concept is a free parameter rather than fixed at 90%. The one change is this inferred example-noise rate: the more people believe some examples are stray draws from 1-100, the more readily they keep a tight rule or range that misses one example, rather than retreating to a broad concept that covers all of them.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/f8fb9bba6e6fa0a9/repo/_runs/cell_1/experiment1/model_loop/models/inferred_example_noise_concepts.py`

### bigelow_piantadosi_independent — rank 7, 4.6 ± 2.5 nats behind the best (1.8× dse: statistically tied with the best), ELPD-LOO -840.3

**Hypothesis:** Bayesian concept learning over 45 rules (odd, even, primes, squares, cubes, powers of 2-10, powers of 2 without 32 or with 37, multiples of 3-9, 11 and 12, ending in 0-9, containing the digit 0-9) and every interval of 1-100, with a prior weight inferred for each rule separately and a share lambda_0 for the rules as a group; each interval has prior (1 - lambda_0) / 100^2. Examples are sampled from the concept (the size principle), 10% of them uniformly from 1-100, and "yes" is the posterior probability that the concept contains the target.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/f8fb9bba6e6fa0a9/repo/_runs/cell_1/experiment1/model_loop/models/bigelow_piantadosi_independent.py`

### win_stay_lose_sample_concepts — rank 8, 4.7 ± 2.9 nats behind the best (1.7× dse: statistically tied with the best), ELPD-LOO -840.4

**Hypothesis:** People learn the concept one example at a time, holding a single working guess rather than the full posterior: after the first example they pick a rule or magnitude range by Bayesian concept learning (size principle, a prior weight per rule), and as each later example arrives they keep their current guess with some probability if it still contains that example (win-stay), and otherwise re-pick a guess from the Bayesian posterior given all examples seen so far (lose-sample). The one free mechanism is this stickiness of the running guess, which makes judgments depend on the order of the examples: a broad concept that fits the first example (e.g. "even" after 16) survives later examples it happens to contain (8, 64) instead of being displaced by a tighter rule (powers of 2), so early examples shape generalization more than later ones.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/f8fb9bba6e6fa0a9/repo/_runs/cell_1/experiment1/model_loop/models/win_stay_lose_sample_concepts.py`

### interval_width_prior_concepts — rank 9, 4.9 ± 2.7 nats behind the best (1.8× dse: statistically tied with the best), ELPD-LOO -840.6

**Hypothesis:** Refinement of `bigelow_piantadosi_independent`: the same Bayesian concept learner over its 45 rules (each with its own inferred prior weight) and all intervals of 1-100, with the size principle, but people's prior over magnitude intervals depends on the interval's width — each interval's prior is proportional to its width raised to an inferred exponent (0 recovers the incumbent's uniform interval prior; negative values mean narrow ranges like "numbers in the 60s" are a priori more plausible than wide ones). The single change is this width-dependent interval prior, which governs how readily people generalise to nearby versus distant magnitudes when no rule explains the examples.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/f8fb9bba6e6fa0a9/repo/_runs/cell_1/experiment1/model_loop/models/interval_width_prior_concepts.py`

### any_number_generator_concepts — rank 10, 5.0 ± 2.6 nats behind the best (1.9× dse: statistically tied with the best), ELPD-LOO -840.7

**Hypothesis:** Refinement of `bigelow_piantadosi_independent`: people do the same Bayesian concept learning over its 45 rules (each with its own inferred prior weight) and all intervals of 1-100 with the size principle, but they also seriously entertain the possibility that the program has no structure at all and simply produces any number from 1 to 100, giving this "anything goes" hypothesis its own inferred prior weight. The one change is this unstructured-generator hypothesis: because it explains each example only as well as chance, the size principle makes it lose to any fitting rule or range as examples accumulate, so it predicts stray "yes" answers to unrelated targets that are common after one or two examples (or when nothing fits) and vanish after four consistent ones — unlike a constant response lapse.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/f8fb9bba6e6fa0a9/repo/_runs/cell_1/experiment1/model_loop/models/any_number_generator_concepts.py`

### log_odds_compressed_concept_posterior — rank 11, 5.5 ± 2.6 nats behind the best (2.1× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -841.2

**Hypothesis:** Refinement of `bigelow_piantadosi_independent`: people do the same Bayesian concept learning over the 45 rules and all intervals of 1-100 (size principle, a prior weight inferred per rule), but they do not report the posterior probability that the target belongs faithfully — their subjective sense of that probability is distorted linearly in log-odds, so confident beliefs are pulled toward (or pushed away from) even odds, as in classic probability-weighting. The one change is this single probability-distortion slope, which predicts that near-certain "yes" and near-certain "no" cases are both answered less extremely, with the compression strongest for the most confident judgments, unlike a fixed random-response lapse that adds the same floor everywhere.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/f8fb9bba6e6fa0a9/repo/_runs/cell_1/experiment1/model_loop/models/log_odds_compressed_concept_posterior.py`

### bayesian_rules_with_response_lapse — rank 12, 5.5 ± 2.6 nats behind the best (2.1× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -841.2

**Hypothesis:** Refinement of bigelow_piantadosi_independent: people do the same Bayesian concept learning over the 45 rules and all intervals (size principle, a prior weight inferred per rule), but on a fixed fraction of trials they do not consult that posterior at all and answer yes or no at random. The one change is this response-lapse rate, which lets the model explain "no" answers to targets that are themselves examples and stray "yes" answers to targets that no plausible concept contains.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/f8fb9bba6e6fa0a9/repo/_runs/cell_1/experiment1/model_loop/models/bayesian_rules_with_response_lapse.py`

### leaky_example_memory_concepts — rank 13, 5.7 ± 2.8 nats behind the best (2.1× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -841.3

**Hypothesis:** People generalize by Bayesian concept learning over mathematical rules and magnitude intervals (the size principle, a prior weight per rule), but their working memory is leaky: each example shown is independently retained with some probability and the forgotten ones play no part in the inference. So a judgment is the rule-and-interval posterior computed from whichever examples survive, which predicts weaker rule-based tightening and broader generalization for larger example sets than full Bayesian learning, and occasional "no" answers to targets that were themselves examples.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/f8fb9bba6e6fa0a9/repo/_runs/cell_1/experiment1/model_loop/models/leaky_example_memory_concepts.py`

### tenenbaum_digit_rule_inventory — rank 14, 6.5 ± 5.1 nats behind the best (1.3× dse: statistically tied with the best), ELPD-LOO -842.2

**Hypothesis:** Refinement of `tenenbaum_rules_similarity`: people do Tenenbaum's Bayesian concept learning (rules share one prior fraction equally, intervals are weighted by an Erlang density on their size, examples are sampled from the concept), but the rules they entertain are the richer inventory of the Bigelow-Piantadosi grammar — notably "contains the digit d" and "ends in 0", plus the two near-miss powers-of-2 sets — instead of Tenenbaum's 33. The one change is the hypothesis space; the claim is that what the people consider, not a separately learned weight for every rule, is what separates the two models, so equal rule weights with two free parameters should suffice.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/f8fb9bba6e6fa0a9/repo/_runs/cell_1/experiment1/model_loop/models/tenenbaum_digit_rule_inventory.py`

### exemplar_retrieved_number_concepts — rank 15, 7.1 ± 3.3 nats behind the best (2.1× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -842.8

**Hypothesis:** People do not reason about which concept is literally consistent with the examples; they hold number concepts in memory as stored sets of member numbers (the familiar rules like even, squares, powers of 2, multiples, final and contained digits, and magnitude ranges), and judge by exemplar similarity on the number line. Each stored concept is retrieved in proportion to how similar the examples are, on average, to its stored members (a concept with few members that lie near every example is retrieved most strongly, so an example close to but not in a concept still partly supports it), and a target is endorsed by its similarity to the nearest stored member of the retrieved concepts, so endorsement falls off gradedly for targets just outside a concept rather than dropping to zero.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/f8fb9bba6e6fa0a9/repo/_runs/cell_1/experiment1/model_loop/models/exemplar_retrieved_number_concepts.py`

### pedagogical_teacher_sampling — rank 16, 7.1 ± 3.5 nats behind the best (2.0× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -842.8

**Hypothesis:** People reason about the examples as a communicative act: they assume the program's examples were picked by a helpful teacher who chooses numbers that point to the intended concept and away from other plausible concepts, rather than drawn at random from it. So an example that belongs to few other plausible rules or ranges (e.g. 64, which says "powers of 2") counts as strong evidence for the concepts containing it, while an example that belongs to many (e.g. 16 or 50) counts as weak evidence; the strength of this pedagogical assumption is inferred, and with none people reduce to ordinary size-principle Bayesian concept learning over the 45 rules and all intervals of 1-100.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/f8fb9bba6e6fa0a9/repo/_runs/cell_1/experiment1/model_loop/models/pedagogical_teacher_sampling.py`

### inferred_outlier_rate_rule_inventory — rank 17, 7.5 ± 5.2 nats behind the best (1.5× dse: statistically tied with the best), ELPD-LOO -843.2

**Hypothesis:** Refinement of `tenenbaum_digit_rule_inventory`: people do the same Bayesian concept learning over the 45 Bigelow-Piantadosi rules (one equal, shared prior weight) and all magnitude intervals (Erlang prior on size), with examples sampled from the concept, but how strongly they believe an example may be a random stray rather than a draw from the concept is a property of the learner, inferred from the data instead of fixed at 10%. The one change is this inferred outlier rate: a high rate means people tolerate examples that break a rule and weaken the size principle (generalising more broadly), while a low rate means a single inconsistent example rules a concept out and tight concepts dominate.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/f8fb9bba6e6fa0a9/repo/_runs/cell_1/experiment1/model_loop/models/inferred_outlier_rate_rule_inventory.py`

### response_bias_concept_learner — rank 18, 9.3 ± 5.8 nats behind the best (1.6× dse: statistically tied with the best), ELPD-LOO -845.0

**Hypothesis:** People infer the hidden concept by simple Bayesian concept learning (rules from the Bigelow-Piantadosi inventory sharing one equal prior weight, magnitude intervals weighted by their size, examples sampled from the concept), but they answer with a response bias: a constant shift in the log-odds of their belief toward "yes" (liberal criterion) or toward "no" (conservative criterion), the same on every trial. The single change from the simplest Bayesian model is this criterion shift at the decision stage, which predicts that uncertain targets (near 50% belief) move most while near-certain ones barely move, and that both the endorsement of example-consistent targets and the rejection of implausible ones shift in the same direction.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/f8fb9bba6e6fa0a9/repo/_runs/cell_1/experiment1/model_loop/models/response_bias_concept_learner.py`

### shared_property_contrast_similarity — rank 19, 120.6 ± 17.4 nats behind the best (6.9× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -956.3

**Hypothesis:** People do not infer a single hidden concept; they judge a target by its feature-based similarity to the examples as a group. Each number is represented by its mathematical properties (parity, primality, being a square, cube or power, multiples, final digit, digits it contains) and by the magnitude bands it falls in, and a target is endorsed to the extent that it possesses the properties the examples have in common, with each property's weight growing with how many of the examples share it and with how distinctive (rare among 1-100) it is. Endorsement therefore falls off gradedly as the target lacks more of the salient shared properties, with no posterior averaging over concepts and no all-or-none membership.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/f8fb9bba6e6fa0a9/repo/_runs/cell_1/experiment1/model_loop/models/shared_property_contrast_similarity.py`

### map_single_concept_commitment — rank 20, 244.6 ± 26.0 nats behind the best (9.4× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1080.3

**Hypothesis:** People do not average over all the concepts consistent with the examples; they commit to the single concept that best explains them — the most probable rule or magnitude interval under a size-principle Bayesian evaluation with a flat prior (rules as a group as likely as intervals as a group) — and answer "yes" exactly when the target belongs to that one concept, apart from occasional response errors. This predicts all-or-none generalization: no graded falloff near the examples' range and no partial endorsement of targets that only some plausible concepts contain (e.g. 32 after "16,8,64" is rejected if the smaller "powers of 2 without 32" wins).

**Source:** `/scr/wurgaft/auto-psych/agent_trees/f8fb9bba6e6fa0a9/repo/_runs/cell_1/experiment1/model_loop/models/map_single_concept_commitment.py`

### tenenbaum_rules_similarity — rank 21, 316.2 ± 48.0 nats behind the best (6.6× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1151.9

**Hypothesis:** Bayesian concept learning over 33 mathematical rules (odd, even, squares, cubes, primes, multiples of 3-12, powers of 2-10, ending in 1-9) and every interval of 1-100. The rules share a prior fraction lambda equally; the intervals share the rest by an Erlang density on their size (scale sigma). Examples are sampled from the concept (the size principle), 10% of them uniformly from 1-100, and "yes" is the posterior probability that the concept contains the target.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/f8fb9bba6e6fa0a9/repo/_runs/cell_1/experiment1/model_loop/models/tenenbaum_rules_similarity.py`

### digit_exemplar_similarity — rank 22, 353.5 ± 33.5 nats behind the best (10.5× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1189.2

**Hypothesis:** People do not induce a rule at all; they judge a target by its summed similarity to the remembered examples, where numbers are represented by their two decimal digits: a target feels like it "belongs" to the extent that its tens digit is close to an example's tens digit and its ones digit matches an example's ones digit. Attention is split between the tens (magnitude) and ones (final-digit) dimensions, similarity decays exponentially with this digit-wise distance, and the probability of saying yes rises with the average similarity of the target to the examples.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/f8fb9bba6e6fa0a9/repo/_runs/cell_1/experiment1/model_loop/models/digit_exemplar_similarity.py`

### digit_overlap_exemplar_similarity — rank 23, 389.6 ± 33.1 nats behind the best (11.8× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1225.2

**Hypothesis:** People do not induce mathematical rules or magnitude intervals at all; they judge a target by its surface resemblance to the examples, treating each number as the set of decimal digits it is written with, and say "yes" in proportion to the average digit-overlap similarity between the target and the examples (an exemplar-similarity process over written digits). This disagrees most sharply with the rule-based Bayesian model on targets that share a digit with some examples but fit no rule or interval consistent with all of them (e.g. 12 after "31,19,49,100"), and on rule-consistent targets written with entirely different digits (e.g. 2 after "16,8,64"), where it predicts low endorsement.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/f8fb9bba6e6fa0a9/repo/_runs/cell_1/experiment1/model_loop/models/digit_overlap_exemplar_similarity.py`

### spread_scaled_range_generalization — rank 24, 421.2 ± 33.5 nats behind the best (12.6× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1256.9

**Hypothesis:** People do not search over mathematical rules; they treat the examples as a sample from a band of magnitudes and generalize by how far the target falls outside the span the examples cover, with a tolerance that grows with how spread out the examples are. A target inside the span of the examples is endorsed at a constant high rate, and endorsement decays smoothly with the target's distance beyond the nearest edge, more slowly for widely spaced examples than for tightly clustered ones (or a single example).

**Source:** `/scr/wurgaft/auto-psych/agent_trees/f8fb9bba6e6fa0a9/repo/_runs/cell_1/experiment1/model_loop/models/spread_scaled_range_generalization.py`

## Pruned models (out of the set; narrowest margin first)

No model has been pruned yet in this project.
