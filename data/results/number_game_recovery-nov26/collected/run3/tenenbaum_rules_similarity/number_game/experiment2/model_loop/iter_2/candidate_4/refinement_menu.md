# Refinement menu

The models you may refine, other than the incumbent `beta_share_width_exception_concepts`. Each entry is one mechanism: its hypothesis in full as its author stated it, how it stands, and where its source is. This is a menu, not a list of what is ruled out: a pruned model lost to the best model on the data it was scored on, by the stated margin, but its mechanism may be partly right — and a model that lost narrowly is the most promising target here. Read the source of the model you pick before you write anything.

## Live models other than the incumbent (in the set; best first)

### uncertainty_shifted_criterion — rank 1, 0.6 ± 0.8 nats behind the best (0.8× dse: statistically tied with the best), ELPD-LOO -2285.8

**Hypothesis:** People infer the hidden concept exactly as the best model's Bayesian concept learner does, but their yes/no decision criterion depends on how sure they are which concept the program uses: the more their belief is spread over many competing concepts (rules and range widths), the more they shift their answer away from the inferred membership probability toward "no" (uncertainty aversion) or, if the inferred shift is negative, toward "yes". So two targets with the same membership probability get different answers depending on whether the examples point clearly to one concept or leave many open.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/570a40651ab1ca31/repo/_runs/cell_1/experiment2/model_loop/models/uncertainty_shifted_criterion.py`

### beta_share_width_prior_leaking_judgment — rank 2, 1.5 ± 1.9 nats behind the best (0.8× dse: statistically tied with the best), ELPD-LOO -2286.6

**Hypothesis:** Refinement of `beta_share_width_exception_concepts` with one component taken from `prior_leaking_membership_judgment`: people infer the hidden concept exactly as the incumbent's Bayesian learner does (family-organised rules with exception rules as their own kind, magnitude ranges judged as a typical share of the 1–100 line, the size principle), but their yes/no judgment is pulled by how plausible the target was before any examples were shown. The log-odds of saying yes is the posterior log-odds that the concept contains the target, shifted by an inferred share of the target's prior log-odds of membership (under the same share-of-the-line range prior), so a priori typical numbers — mid-range numbers that most moderately wide ranges contain, and numbers belonging to many rules — get more "yes" than the examples alone warrant (or less, if the share is negative).

**Source:** `/scr/wurgaft/auto-psych/agent_trees/570a40651ab1ca31/repo/_runs/cell_1/experiment2/model_loop/models/beta_share_width_prior_leaking_judgment.py`

### beta_share_parity_range_concepts — rank 3, 1.8 ± 1.6 nats behind the best (1.1× dse: statistically tied with the best), ELPD-LOO -2286.9

**Hypothesis:** Refinement of `beta_share_width_exception_concepts` with one component taken from `parity_within_range_concepts`: people do the same Bayesian concept learning over rules (family-organised prior, exception rules as their own kind) and magnitude ranges judged as a typical share of the 1–100 line, with the size principle, but they can also entertain the conjunctive concept "odd (or even) numbers within a range" (e.g. "even numbers above 60") as its own kind of concept, with its own inferred plausibility and the same share-of-the-line width prior as plain ranges. So after 4, 66, 78, 96 people say "yes" to large even numbers (70, 74) but "no" to small ones and to odd numbers between the examples, and after 23, 35, 95, 83 they reject even numbers between the examples, rather than spreading "yes" over every number in the range or over all even (odd) numbers.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/570a40651ab1ca31/repo/_runs/cell_1/experiment2/model_loop/models/beta_share_parity_range_concepts.py`

### prior_leaking_membership_judgment — rank 4, 3.8 ± 3.1 nats behind the best (1.2× dse: statistically tied with the best), ELPD-LOO -2289.0

**Hypothesis:** People infer the hidden concept as a normative Bayesian concept learner does (rules with a family-organised prior and arbitrary-exception rules as their own kind, magnitude ranges with a preferred width, the size principle), but their yes/no judgment is distorted by how plausible the target was before they saw any examples: the log-odds of saying yes is the posterior log-odds that the concept contains the target, shifted by an inferred share of the target's prior log-odds of membership. So a priori "typical" numbers (mid-range, even, multiples) get more "yes" than the examples warrant (prior leakage), or, if the share is negative, less (contrast against what was expected anyway).

**Source:** `/scr/wurgaft/auto-psych/agent_trees/570a40651ab1ca31/repo/_runs/cell_1/experiment2/model_loop/models/prior_leaking_membership_judgment.py`

### inferred_stray_rate_concepts — rank 5, 5.9 ± 3.0 nats behind the best (2.0× dse: statistically tied with the best), ELPD-LOO -2291.1

**Hypothesis:** Refinement of `gamma_width_exception_family_concepts`: people do the same Bayesian concept learning over rules (family-organised prior, exception rules as their own kind) and magnitude ranges with a gamma width preference and the size principle, and they also allow that any example may be a stray number the program produced at random rather than a member of the concept. The one change is that how often they expect such strays is no longer fixed at one example in ten but is a belief inferred from their judgments: the more strays they expect, the more readily they keep concepts that miss an example (and so sometimes reject an example they were shown), and the weaker the size principle becomes.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/570a40651ab1ca31/repo/_runs/cell_1/experiment2/model_loop/models/inferred_stray_rate_concepts.py`

### gamma_width_exception_family_concepts — rank 6, 6.0 ± 3.1 nats behind the best (1.9× dse: statistically tied with the best), ELPD-LOO -2291.2

**Hypothesis:** Refinement of `gamma_width_family_prior_concepts`: people do the same Bayesian concept learning over rules and magnitude ranges (size principle, family-organised rule prior, and a gamma prior over range width whose sharpness is inferred along with its typical width). The one change, taken from `arbitrary_exception_rule_prior`, is that rules with an arbitrary exception ("powers of 2 except 32", "powers of 2 plus 37") form their own kind of rule with their own, presumably very low, plausibility, instead of being ordinary power rules. So people expect ranges of a sharply characteristic width and also firmly reject 37 (and accept 32) after seeing powers of 2.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/570a40651ab1ca31/repo/_runs/cell_1/experiment2/model_loop/models/gamma_width_exception_family_concepts.py`

### first_example_trust_concepts — rank 7, 6.8 ± 3.1 nats behind the best (2.2× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -2292.0

**Hypothesis:** People treat the first example they see differently from the later ones when judging how far each example is a genuine output of the hidden concept: the first example sets the concepts they entertain and is trusted to belong to the concept at its own inferred rate. Each later example is trusted at the usual fixed rate, and one that does not fit can be dismissed as a stray output. Apart from this, they generalise by the same Bayesian concept learning as the best model (size principle, rule prior organised by kind with exception rules as their own kind, gamma prior over range width). So if the first example is trusted more, an odd-one-out example is discounted when it comes late in the set (8, 24, 48, 63 still suggests multiples of 8) but forces a broader concept when it comes first.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/570a40651ab1ca31/repo/_runs/cell_1/experiment2/model_loop/models/first_example_trust_concepts.py`

### gamma_width_parity_range_concepts — rank 8, 7.6 ± 3.5 nats behind the best (2.2× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -2292.8

**Hypothesis:** Refinement of the incumbent `parity_within_range_concepts`: people do the same Bayesian concept learning over rules (family-organised rule prior, exception rules as their own kind), magnitude ranges and "odd (or even) numbers within a range", with the size principle. The one change, taken from `gamma_width_exception_family_concepts`, is the shape of their expectation about how wide a range (plain or parity-restricted) is: a gamma prior whose sharpness is inferred along with its typical width, instead of the fixed, loose Erlang form — so people may expect ranges of one sharply characteristic width, strongly disfavouring both very narrow and very wide ranges around the examples.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/570a40651ab1ca31/repo/_runs/cell_1/experiment2/model_loop/models/gamma_width_parity_range_concepts.py`

### without_replacement_sampling_concepts — rank 9, 13.8 ± 4.3 nats behind the best (3.2× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -2299.0

**Hypothesis:** People do the same Bayesian concept learning as the best model (rules with a family-organised prior, arbitrary-exception rules as their own kind, magnitude ranges of a characteristic width), but they assume the program never repeats a number, so they treat the examples as drawn from the concept without replacement: each further example is one of the concept's members not yet shown. As a result, examples that nearly exhaust a small concept (e.g. 8, 64 or 81, 9) point to that concept, such as cubes or powers of 3, much more strongly than under sampling with replacement, and generalisation narrows faster onto small, nearly exhausted rules.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/570a40651ab1ca31/repo/_runs/cell_1/experiment2/model_loop/models/without_replacement_sampling_concepts.py`

### gamma_width_doubling_chain_concepts — rank 10, 18.0 ± 6.8 nats behind the best (2.7× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -2303.2

**Hypothesis:** Refinement of `doubling_chain_concepts`: people do Bayesian concept learning over arithmetic rules (organised by kind, with exception rules as their own kind), magnitude ranges and odd/even-within-a-range concepts, using the size principle, and they also entertain multiplicative chains — each number double (or triple) the one before, starting from a number other than 1 (e.g. 5, 10, 20, 40, 80) — as a kind of rule with its own plausibility. The one change, taken from the best model `gamma_width_exception_family_concepts`, is their expectation about how wide a range is: a gamma prior whose sharpness is inferred along with its typical width, in place of the fixed, loose Erlang form. With ranges judged as the best model judges them, chains are put to a fair test: after a chain-like set such as 5, 10, 20 this model still predicts "yes" to 40 and 80 and "no" to other multiples of 5 such as 15, 25 and 30, where the best model spreads "yes" over all multiples of 5.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/570a40651ab1ca31/repo/_runs/cell_1/experiment2/model_loop/models/gamma_width_doubling_chain_concepts.py`

### example_recall_limited_concepts — rank 11, 19.2 ± 7.2 nats behind the best (2.7× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -2304.4

**Hypothesis:** People do the same Bayesian concept learning as the best model (rules with a family-organised prior and exception rules as their own kind, magnitude ranges and odd/even-within-a-range concepts with a preferred width, size principle), but their memory for the examples is limited: each example shown is retained, independently, only with an inferred probability, and they generalise from whichever examples they still hold. So a judgment is sometimes based on only part of the set, which broadens generalisation for longer sets (a forgotten example no longer rules out a broader or different concept) in proportion to how many examples could have been lost.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/570a40651ab1ca31/repo/_runs/cell_1/experiment2/model_loop/models/example_recall_limited_concepts.py`

### parity_within_range_concepts — rank 12, 19.5 ± 7.3 nats behind the best (2.7× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -2304.7

**Hypothesis:** Refinement of the incumbent `arbitrary_exception_rule_prior`: people do the same Bayesian concept learning over rules (family-organised rule prior, exception rules as their own kind) and magnitude ranges with a preferred width and the size principle, but they can also entertain a conjunctive kind of concept, "odd (or even) numbers within a range" (e.g. "even numbers above 60"), with its own inferred plausibility and the same width preference. The one change is adding this parity-within-a-range kind of concept, so that after 4, 66, 78, 96 people say yes to large even numbers (70, 74) but no to small ones (8, 22), and after 23, 35, 95, 83 they reject even numbers between the examples, rather than spreading "yes" evenly over all even (or all in-range) numbers.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/570a40651ab1ca31/repo/_runs/cell_1/experiment2/model_loop/models/parity_within_range_concepts.py`

### decade_and_parity_range_concepts — rank 13, 21.5 ± 7.9 nats behind the best (2.7× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -2306.7

**Hypothesis:** Refinement of `decade_concept_family_prior` (tied with the best): people do the same Bayesian concept learning over rules (family-organised prior, exception-laden rules as their own kind, and decades such as "numbers in the 60s" as their own kind of rule) and magnitude ranges with a preferred width and the size principle. The one change, taken from `parity_within_range_concepts`, is that people can also entertain "odd (or even) numbers within a range" as a kind of concept with its own inferred plausibility, so examples sharing a tens digit (61, 64, 67) confine "yes" to that decade while examples such as 4, 66, 78, 96 confine it to large even numbers — two ready-made ways of bounding generalisation that the incumbent and the decade model each have only one of.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/570a40651ab1ca31/repo/_runs/cell_1/experiment2/model_loop/models/decade_and_parity_range_concepts.py`

### rule_noticing_grows_with_examples — rank 14, 22.0 ± 7.7 nats behind the best (2.9× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -2307.2

**Hypothesis:** People do the same Bayesian concept learning over arithmetic rules, magnitude ranges and odd/even-within-a-range concepts as the best model, but whether a rule even comes to mind depends on how many examples they have seen: the prior odds of thinking of a rule rather than a magnitude range grow steadily, at an inferred rate, with each additional example. So after one or two examples people generalise mostly by magnitude, and after four examples they look for a rule more readily than the examples' likelihood alone would warrant.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/570a40651ab1ca31/repo/_runs/cell_1/experiment2/model_loop/models/rule_noticing_grows_with_examples.py`

### multiples_within_range_concepts — rank 15, 22.5 ± 7.6 nats behind the best (3.0× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -2307.7

**Hypothesis:** Refinement of the incumbent `parity_within_range_concepts`: people do the same Bayesian concept learning over rules (family-organised prior, exception rules as their own kind), magnitude ranges and "odd (or even) numbers within a range", with the Erlang width preference and the size principle. The one change is that the conjunctive "rule within a range" idea extends beyond parity: people also entertain "multiples of m within a range" (e.g. "multiples of 3 between 10 and 60", "multiples of 9 from 27 to 81") as a further kind of concept with its own inferred plausibility and the same width preference. So after 39, 60, 12, 45 or 63, 27, 81 they accept multiples near the examples (e.g. 36, 54) but reject multiples far outside the examples' span (e.g. 90, 99, or 3), instead of spreading "yes" over all multiples of 3 or over every number in the range.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/570a40651ab1ca31/repo/_runs/cell_1/experiment2/model_loop/models/multiples_within_range_concepts.py`

### doubling_chain_concepts — rank 16, 27.6 ± 9.0 nats behind the best (3.1× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -2312.8

**Hypothesis:** People do the same Bayesian concept learning as the best model (family-organised rule prior with exception rules as their own kind, magnitude ranges and odd/even-within-a-range concepts with a preferred width, and the size principle), but they also hold a kind of concept the best model lacks: multiplicative chains, in which each number is double (or triple) the one before, starting from some number other than 1 (e.g. 3, 6, 12, 24, 48, 96 or 5, 10, 20, 40, 80 or 2, 6, 18, 54). This kind has its own inferred plausibility. The disagreement with the best model shows up on example sets that form such a chain (e.g. 5, 10, 20 or 6, 12, 24): there this model predicts "yes" to the next links (40, 80 or 48, 96) and "no" to other multiples (15, 25, 30 or 18, 30, 36), whereas the best model spreads "yes" over all multiples of 5 (or 6) and nearby numbers.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/570a40651ab1ca31/repo/_runs/cell_1/experiment2/model_loop/models/doubling_chain_concepts.py`

### off_by_one_power_concepts — rank 17, 109.4 ± 21.5 nats behind the best (5.1× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -2394.6

**Hypothesis:** People do the same Bayesian concept learning as the best model (rules with a family-organised prior and arbitrary-exception rules as their own kind, magnitude ranges with a beta prior over the share of the 1–100 line they cover, and the size principle), but their space of rules also contains near-misses of the sparse multiplicative rules: "one more than" or "one less than" a power or a square (e.g. 3, 5, 9, 17, 33, 65 or 1, 3, 7, 15, 31, 63, or 2, 5, 10, 17, 26, 37), forming their own kind of rule with its own inferred, presumably low, plausibility. So after examples such as 31, 3, 1, 15 or 33, 17, 5, 9 people say "yes" to the next shifted power (63, 65) and "no" to other odd numbers or nearby numbers, rather than spreading "yes" over all odd numbers or a range.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/570a40651ab1ca31/repo/_runs/cell_1/experiment2/model_loop/models/off_by_one_power_concepts.py`

### gaussian_neighbourhood_magnitude_concepts — rank 18, 210.3 ± 29.3 nats behind the best (7.2× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -2495.5

**Hypothesis:** People generalise by rules exactly as the best Bayesian concept learner does, but their magnitude-based concepts are not hard-edged ranges: such a concept is a neighbourhood around a central number, membership falls off smoothly with numerical distance from that centre along a Gaussian similarity gradient whose breadth they infer from the examples (narrow neighbourhoods favoured by the size principle), and examples are produced in proportion to that similarity. A target's graded similarity to the inferred neighbourhood is the probability they say "yes", so "yes" fades gradually with distance from the examples' centre instead of being flat inside the examples' range and stopping at an edge.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/570a40651ab1ca31/repo/_runs/cell_1/experiment2/model_loop/models/gaussian_neighbourhood_magnitude_concepts.py`

## Pruned models (out of the set; narrowest margin first)

### arbitrary_exception_rule_prior — pruned (experiment1 end of experiment)

**Margin:** 3.8 nats behind gamma_width_parity_range_concepts; retired to keep the live set at 8 models: ELPD-LOO rank 9 of 10

**Hypothesis:** Refinement of `erlang_interval_family_prior_concepts`: people do the same Bayesian concept learning over rules and magnitude ranges (size principle, Erlang prior over range width, prior shared by kind of rule), but a rule with an arbitrary exception ("powers of 2 except 32", "powers of 2 plus 37") is not just another power rule to them — it belongs to its own kind, with its own inferred, presumably very low, plausibility. The one change is giving these exception-laden rules their own family in the rule prior, so after seeing powers of 2 people firmly reject 37 (and accept 32), instead of giving 37 the roughly one-in-three "yes" that the incumbent predicts.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/570a40651ab1ca31/repo/_runs/cell_1/experiment1/model_loop/models/pruned/arbitrary_exception_rule_prior.py`

### decade_concept_family_prior — pruned (experiment1 end of experiment)

**Margin:** 4.3 nats behind gamma_width_parity_range_concepts; retired to keep the live set at 8 models: ELPD-LOO rank 10 of 10

**Hypothesis:** Refinement of `arbitrary_exception_rule_prior` (the incumbent): people do the same Bayesian concept learning over rules and magnitude ranges (size principle, Erlang prior over range width, rule prior shared by kind of rule, exception-laden rules as their own kind), but they also hold a ready-made kind of concept that the incumbent lacks: the decade, "numbers in the 60s" (same tens digit), as a distinct kind of rule with its own inferred plausibility. The one change is adding the ten decade concepts as a new family in the rule prior, so when the examples share a tens digit (e.g. 61, 64, 67) people confine "yes" sharply to that decade, more firmly than an arbitrary range of the same width.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/570a40651ab1ca31/repo/_runs/cell_1/experiment1/model_loop/models/pruned/decade_concept_family_prior.py`

### gamma_width_family_prior_concepts — pruned (experiment1 end of experiment)

**Margin:** 10.1 nats behind gamma_width_parity_range_concepts (2.2× dse)

**Hypothesis:** Refinement of `erlang_interval_family_prior_concepts` (the incumbent): people do the same Bayesian concept learning over the 45 rules (with the family-organised rule prior) and magnitude ranges of 1-100, with the size principle. The one change is the shape of their prior expectation about how wide a range is: instead of a fixed Erlang form (whose preference for a typical width is always of the same, fairly loose shape), the width prior is a gamma distribution whose shape is inferred along with its typical width, so people may hold a sharp expectation that ranges have one characteristic width (strongly disfavouring both narrow and very wide ranges) or, at the other extreme, a monotone preference for the narrowest range that fits the examples.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/570a40651ab1ca31/repo/_runs/cell_1/experiment1/model_loop/models/pruned/gamma_width_family_prior_concepts.py`

### erlang_interval_family_prior_concepts — pruned (experiment1 end of experiment)

**Margin:** 11.0 nats behind gamma_width_parity_range_concepts (2.4× dse)

**Hypothesis:** Refinement of `erlang_interval_size_concept_learning` (the narrowest-losing live model): people do Bayesian concept learning over the same 45 rules and magnitude ranges, with the size principle and an Erlang prior that favours ranges of a characteristic, inferred width over very wide or single-number ones. The one change is how prior belief is shared among the rules: instead of an independent weight for each of the 45 rules, people first favour a kind of rule (parity, primes, squares/cubes, powers, multiples, last digit, contained digit), each kind with its own inferred plausibility, and treat rules of the same kind as equally likely. So whole kinds of rule (e.g. "contains the digit d") can be implausible a priori, while magnitude generalisation follows ranges of a preferred size.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/570a40651ab1ca31/repo/_runs/cell_1/experiment1/model_loop/models/pruned/erlang_interval_family_prior_concepts.py`

### rule_family_prior_concepts — pruned (experiment1 end of experiment)

**Margin:** 12.6 nats behind gamma_width_parity_range_concepts (2.6× dse)

**Hypothesis:** Refinement of `bigelow_piantadosi_mixture`: people do Bayesian concept learning over the same 45 rules and all intervals (size principle, 10% of examples random), but their prior over rules is organised by kind of rule rather than flat: they first favour a family of rule (parity, primes, squares/cubes, powers, multiples, last digit, contained digit), each with its own inferred plausibility, and rules within a family are equally likely. The one change is replacing the mixture's uniform rule prior with this inferred family-level prior, so whole kinds of rule (e.g. "contains the digit d") can be a priori implausible while others (multiples, last digit) dominate generalization.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/570a40651ab1ca31/repo/_runs/cell_1/experiment1/model_loop/models/pruned/rule_family_prior_concepts.py`

### iter2_candidate3 — pruned (experiment1 end of experiment)

**Margin:** 12.7 nats behind gamma_width_parity_range_concepts (2.8× dse)

**Hypothesis:** Refinement of the incumbent `erlang_interval_family_prior_concepts`: people do the same Bayesian concept learning (45 rules with a family-organised prior, magnitude ranges with an Erlang width prior, size principle), but on some judgments they do not consult the inferred concept at all and simply guess, answering "yes" with their own guessing bias. The one change is this attentional lapse: on an inferred share of trials the answer is a biased guess, so targets the learner is certain about are still sometimes answered the other way — "yes" to numbers outside every plausible concept and "no" to numbers inside it — rather than with near-zero or near-one probability.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/570a40651ab1ca31/repo/_runs/cell_1/experiment1/model_loop/models/pruned/iter2_candidate3.py`

### serial_position_weighted_evidence — pruned (experiment1 end of experiment)

**Margin:** 14.2 nats behind gamma_width_parity_range_concepts (2.9× dse)

**Hypothesis:** People do Bayesian concept learning over the same rules (with a family-organised rule prior) and magnitude ranges as the best model, but they update sequentially as the examples are read and do not weigh every example equally: each example's evidence is weighted by its position in the sequence, decaying (or growing) geometrically with how far it lies before the last example shown, at an inferred rate. A recency rate means earlier examples are partly forgotten, so longer sets narrow generalisation less than the size principle predicts; a primacy rate means early examples are overcounted, so longer sets narrow it more.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/570a40651ab1ca31/repo/_runs/cell_1/experiment1/model_loop/models/pruned/serial_position_weighted_evidence.py`

### lapse_guess_concept_learning — pruned (experiment1 end of experiment)

**Margin:** 14.4 nats behind gamma_width_parity_range_concepts (3.0× dse)

**Hypothesis:** People infer how likely the target is to belong to the hidden concept by Bayesian concept learning over rules (with a family-organised rule prior) and magnitude ranges, using the size principle, and normally answer with that probability; but on some fraction of judgments they lapse (inattention, a slip, not engaging with the examples) and answer by a fixed guessing tendency instead, saying "yes" with an inferred guessing rate regardless of the examples. So even targets the concept clearly excludes get a floor of "yes" answers, and targets it clearly includes get a ceiling below certainty, both set by one lapse rate and one guessing bias.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/570a40651ab1ca31/repo/_runs/cell_1/experiment1/model_loop/models/pruned/lapse_guess_concept_learning.py`

### evidence_overweighting_concept_learning — pruned (experiment1 end of experiment)

**Margin:** 14.9 nats behind gamma_width_parity_range_concepts (3.1× dse)

**Hypothesis:** People do Bayesian concept learning over the same rules (with a family-organised rule prior) and magnitude intervals as the best model, using the size principle, but they mis-weight the evidence: each example's likelihood is raised to an inferred power, so the examples count for more (or less) than a normative learner would let them. If people over-weight evidence, their generalisation collapses onto the smallest fitting concept faster as examples accumulate than the size principle predicts, which would explain why their yes-rate drops more from two- to four-example sets than the best model expects.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/570a40651ab1ca31/repo/_runs/cell_1/experiment1/model_loop/models/pruned/evidence_overweighting_concept_learning.py`

### offset_progression_concepts — pruned (experiment1 end of experiment)

**Margin:** 14.9 nats behind gamma_width_parity_range_concepts (2.3× dse)

**Hypothesis:** Refinement of `arbitrary_exception_rule_prior`: people do the same Bayesian concept learning over rules and magnitude ranges (size principle, Erlang prior over range width, prior shared by kind of rule), but their space of rules also contains arithmetic progressions that do not start at zero — "every third number starting from 1", "odd multiples of 3" (3, 9, 15, ...), "every fourth number starting from 2" — i.e. all numbers leaving the same nonzero remainder under a small step of 3 to 6. These offset progressions form their own kind of rule with its own inferred, presumably low, plausibility, so after examples such as 69, 9, 39, 21 people can generalise only to the odd multiples of 3 and reject other odd numbers such as 19.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/570a40651ab1ca31/repo/_runs/cell_1/experiment1/model_loop/models/pruned/offset_progression_concepts.py`

### log_number_line_range_concepts — pruned (experiment1 end of experiment)

**Margin:** 23.7 nats behind gamma_width_parity_range_concepts (4.1× dse)

**Hypothesis:** Refinement of `rule_family_prior_concepts`: people do the same Bayesian concept learning over the 45 rules (with the family-organised rule prior) and magnitude ranges of 1-100, with size-principle sampling, but they picture the ranges on a compressed, roughly logarithmic mental number line rather than a linear one. The one change is the prior over ranges: a range's endpoints are equally likely to fall anywhere on that mental line, with the degree of compression inferred, so ranges among small numbers (where numbers are spread out on the mental line) are a priori more plausible than equally long ranges among large numbers (where they are crowded together). As a result, examples among large numbers are explained by a range less readily and by a rule more often than the incumbent predicts.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/570a40651ab1ca31/repo/_runs/cell_1/experiment1/model_loop/models/pruned/log_number_line_range_concepts.py`

### membership_criterion_threshold — pruned (experiment1 end of experiment)

**Margin:** 34.2 nats behind gamma_width_parity_range_concepts (4.6× dse)

**Hypothesis:** People infer how likely the target is to belong to the hidden concept exactly as the current best model's Bayesian concept learner does (same rules and magnitude ranges, family-organised rule prior, size principle), but they do not report that probability directly: they answer "yes" only when the inferred membership probability exceeds an internal decision criterion, and this criterion varies noisily from judgment to judgment. So targets with middling membership probability are mostly rejected (or mostly accepted, depending on where the criterion sits), and answers are more all-or-none than probability matching predicts.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/570a40651ab1ca31/repo/_runs/cell_1/experiment1/model_loop/models/pruned/membership_criterion_threshold.py`

### win_stay_lose_shift_concept_testing — pruned (experiment1 end of experiment)

**Margin:** 49.0 nats behind gamma_width_parity_range_concepts (5.0× dse)

**Hypothesis:** People learn the concept by sequential hypothesis testing ("win-stay, lose-shift") rather than by weighing all examples at once: on reading the first example they pick one concept (a rule or a magnitude range, with the same family-organised rule prior and preferred-width range prior as the best model) in proportion to how well it explains that example, keep it as long as each new example fits it, and only when an example contradicts it do they pick a fresh concept in proportion to how well it explains all examples seen so far. They answer "yes" if the concept they end up holding contains the target, so the first example shapes generalisation most and later examples that fit a broad initial guess do not narrow it.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/570a40651ab1ca31/repo/_runs/cell_1/experiment1/model_loop/models/pruned/win_stay_lose_shift_concept_testing.py`

### erlang_interval_size_concept_learning — pruned (experiment1 end of experiment)

**Margin:** 92.2 nats behind gamma_width_parity_range_concepts (7.4× dse)

**Hypothesis:** A refinement of bigelow_piantadosi_independent: people do Bayesian concept learning over the same 45 rules and all intervals of 1-100 (prior weight per rule, share lambda_0 for rules, size-principle sampling with 10% random examples, "yes" = posterior probability the concept contains the target). The one change is the prior over intervals: instead of every interval being equally likely (which makes wide ranges a priori far more numerous), people expect a range of a characteristic width, with an Erlang prior over interval size whose scale is inferred from the data (as in Tenenbaum's number game), so moderately narrow ranges are preferred a priori and very wide or single-number ranges are penalised.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/570a40651ab1ca31/repo/_runs/cell_1/experiment1/model_loop/models/pruned/erlang_interval_size_concept_learning.py`

### bigelow_piantadosi_independent — pruned (experiment1 end of experiment)

**Margin:** 95.2 nats behind gamma_width_parity_range_concepts (7.8× dse)

**Hypothesis:** Bayesian concept learning over 45 rules (odd, even, primes, squares, cubes, powers of 2-10, powers of 2 without 32 or with 37, multiples of 3-9, 11 and 12, ending in 0-9, containing the digit 0-9) and every interval of 1-100, with a prior weight inferred for each rule separately and a share lambda_0 for the rules as a group; each interval has prior (1 - lambda_0) / 100^2. Examples are sampled from the concept (the size principle), 10% of them uniformly from 1-100, and "yes" is the posterior probability that the concept contains the target.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/570a40651ab1ca31/repo/_runs/cell_1/experiment1/model_loop/models/pruned/bigelow_piantadosi_independent.py`

### weak_sampling_concept_learning — pruned (experiment1 end of experiment)

**Margin:** 213.0 nats behind gamma_width_parity_range_concepts (10.2× dse)

**Hypothesis:** People entertain the same kinds of number concepts (arithmetic rules and magnitude ranges) but treat the examples only as consistent with the hidden concept rather than as random samples drawn from it (weak sampling, no size principle), so every concept that contains the examples stays plausible in proportion to its prior regardless of how many other numbers it includes. The sharpest disagreement with the size-principle best model lives in sets that fit a small rule nested in a larger one (e.g. 16,32,2,8 or 20,32,92): this model predicts broad "yes" to targets that only the larger concept contains (other even numbers, numbers anywhere in a wide range), whereas the best model confines "yes" to the smallest fitting concept.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/570a40651ab1ca31/repo/_runs/cell_1/experiment1/model_loop/models/pruned/weak_sampling_concept_learning.py`

### bigelow_piantadosi_mixture — pruned (experiment1 end of experiment)

**Margin:** 264.6 nats behind gamma_width_parity_range_concepts (9.4× dse)

**Hypothesis:** Bayesian concept learning over 45 rules (odd, even, primes, squares, cubes, powers of 2-10, powers of 2 without 32 or with 37, multiples of 3-9, 11 and 12, ending in 0-9, containing the digit 0-9) and every interval of 1-100. A rule is chosen with probability lambda_0, and the rules are equally likely; each interval has prior (1 - lambda_0) / 100^2. Examples are sampled from the concept (the size principle), 10% of them uniformly from 1-100, and "yes" is the posterior probability that the concept contains the target.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/570a40651ab1ca31/repo/_runs/cell_1/experiment1/model_loop/models/pruned/bigelow_piantadosi_mixture.py`

### exemplar_featural_similarity_gcm — pruned (experiment1 end of experiment)

**Margin:** 400.3 nats behind gamma_width_parity_range_concepts (14.7× dse)

**Hypothesis:** People do not infer a rule at all: they judge the target by its exemplar similarity to the examples shown, as in a generalized context model. Each example is compared with the target on a few perceptual dimensions of the number — its magnitude, its final digit, its tens digit and its parity — with attention shared among these dimensions, similarity decays exponentially with attention-weighted mismatch, and the more similar the target is on average to the examples, the more likely they say yes.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/570a40651ab1ca31/repo/_runs/cell_1/experiment1/model_loop/models/pruned/exemplar_featural_similarity_gcm.py`

### property_matching_exemplar_similarity — pruned (experiment1 end of experiment)

**Margin:** 404.0 nats behind gamma_width_parity_range_concepts (14.8× dse)

**Hypothesis:** People do not infer a hidden concept: they store the shown examples as exemplars and judge the target by how similar it is to them, where two numbers are similar to the extent that they share familiar mathematical properties (parity, primality, being a square or cube, being a power of some base, being a multiple of some number, final digit, digits contained) and are close in magnitude. Each kind of property receives its own attention weight, similarity decays exponentially with the attention-weighted count of properties on which target and example differ, and people say "yes" with a Luce-choice probability that grows with the target's average similarity to the stored examples relative to a fixed response criterion.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/570a40651ab1ca31/repo/_runs/cell_1/experiment1/model_loop/models/pruned/property_matching_exemplar_similarity.py`

### residue_class_progression — pruned (experiment1 end of experiment)

**Margin:** 457.7 nats behind gamma_width_parity_range_concepts (12.1× dse)

**Hypothesis:** People assume the hidden program produces an arithmetic progression: all numbers in 1-100 that leave the same remainder when divided by some step size (step 1 = every number, step 2 = all odds or all evens, step 10 = a shared final digit, step 12 = every twelfth number, and so on). They consider every step size whose residue class contains all the examples, favour smaller steps a priori but larger (tighter) steps when several examples all fit them (the size principle), and say "yes" with the posterior probability that the target lies in the same progression as the examples.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/570a40651ab1ca31/repo/_runs/cell_1/experiment1/model_loop/models/pruned/residue_class_progression.py`

### spread_scaled_distance_gradient — pruned (experiment1 end of experiment)

**Margin:** 459.8 nats behind gamma_width_parity_range_concepts (16.0× dse)

**Hypothesis:** People judge the target purely by its numerical closeness to the nearest example on the 1-100 number line, with a Shepard-style exponential generalization gradient whose width is set by how spread out the examples are: they say "yes" with a probability that falls off exponentially with the target's distance from the nearest example, and widely spaced examples (large average gap between neighbouring examples) license a proportionally broader gradient than tightly clustered ones.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/570a40651ab1ca31/repo/_runs/cell_1/experiment1/model_loop/models/pruned/spread_scaled_distance_gradient.py`
