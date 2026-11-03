# Existing hypotheses

Each model below is ONE cognitive hypothesis, with how it stands on the current data by ELPD-LOO (best first). `elpd_diff ± dse` is a model's deficit against the best and the standard error of that difference (with the responses to each stimulus pair counted together, as they are correlated): within about 2·dse the two are statistically tied on this data; beyond it the model has lost. "PSIS-LOO unreliable" means the estimate itself is untrustworthy (too many high-Pareto-k trials), not that the model is bad. Propose a hypothesis that is genuinely different from these, or a refinement of a single one of them — never a combination of several.

## parity_within_range_concepts  — rank 0, the best model on this data, ELPD-LOO -1072.9

Refinement of the incumbent `arbitrary_exception_rule_prior`: people do the same Bayesian concept learning over rules (family-organised rule prior, exception rules as their own kind) and magnitude ranges with a preferred width and the size principle, but they can also entertain a conjunctive kind of concept, "odd (or even) numbers within a range" (e.g. "even numbers above 60"), with its own inferred plausibility and the same width preference. The one change is adding this parity-within-a-range kind of concept, so that after 4, 66, 78, 96 people say yes to large even numbers (70, 74) but no to small ones (8, 22), and after 23, 35, 95, 83 they reject even numbers between the examples, rather than spreading "yes" evenly over all even (or all in-range) numbers.

## gamma_width_exception_family_concepts  — rank 1, 2.7 ± 3.6 nats behind the best (0.8× dse: statistically tied with the best), ELPD-LOO -1075.6

Refinement of `gamma_width_family_prior_concepts`: people do the same Bayesian concept learning over rules and magnitude ranges (size principle, family-organised rule prior, and a gamma prior over range width whose sharpness is inferred along with its typical width). The one change, taken from `arbitrary_exception_rule_prior`, is that rules with an arbitrary exception ("powers of 2 except 32", "powers of 2 plus 37") form their own kind of rule with their own, presumably very low, plausibility, instead of being ordinary power rules. So people expect ranges of a sharply characteristic width and also firmly reject 37 (and accept 32) after seeing powers of 2.

## arbitrary_exception_rule_prior  — rank 2, 3.5 ± 3.5 nats behind the best (1.0× dse: statistically tied with the best), ELPD-LOO -1076.4

Refinement of `erlang_interval_family_prior_concepts`: people do the same Bayesian concept learning over rules and magnitude ranges (size principle, Erlang prior over range width, prior shared by kind of rule), but a rule with an arbitrary exception ("powers of 2 except 32", "powers of 2 plus 37") is not just another power rule to them — it belongs to its own kind, with its own inferred, presumably very low, plausibility. The one change is giving these exception-laden rules their own family in the rule prior, so after seeing powers of 2 people firmly reject 37 (and accept 32), instead of giving 37 the roughly one-in-three "yes" that the incumbent predicts.

## decade_concept_family_prior  — rank 3, 3.9 ± 3.4 nats behind the best (1.1× dse: statistically tied with the best), ELPD-LOO -1076.8

Refinement of `arbitrary_exception_rule_prior` (the incumbent): people do the same Bayesian concept learning over rules and magnitude ranges (size principle, Erlang prior over range width, rule prior shared by kind of rule, exception-laden rules as their own kind), but they also hold a ready-made kind of concept that the incumbent lacks: the decade, "numbers in the 60s" (same tens digit), as a distinct kind of rule with its own inferred plausibility. The one change is adding the ten decade concepts as a new family in the rule prior, so when the examples share a tens digit (e.g. 61, 64, 67) people confine "yes" sharply to that decade, more firmly than an arbitrary range of the same width.

## gamma_width_family_prior_concepts  — rank 4, 9.8 ± 4.7 nats behind the best (2.1× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1082.7

Refinement of `erlang_interval_family_prior_concepts` (the incumbent): people do the same Bayesian concept learning over the 45 rules (with the family-organised rule prior) and magnitude ranges of 1-100, with the size principle. The one change is the shape of their prior expectation about how wide a range is: instead of a fixed Erlang form (whose preference for a typical width is always of the same, fairly loose shape), the width prior is a gamma distribution whose shape is inferred along with its typical width, so people may hold a sharp expectation that ranges have one characteristic width (strongly disfavouring both narrow and very wide ranges) or, at the other extreme, a monotone preference for the narrowest range that fits the examples.

## erlang_interval_family_prior_concepts  — rank 5, 10.7 ± 4.5 nats behind the best (2.4× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1083.6

Refinement of `erlang_interval_size_concept_learning` (the narrowest-losing live model): people do Bayesian concept learning over the same 45 rules and magnitude ranges, with the size principle and an Erlang prior that favours ranges of a characteristic, inferred width over very wide or single-number ones. The one change is how prior belief is shared among the rules: instead of an independent weight for each of the 45 rules, people first favour a kind of rule (parity, primes, squares/cubes, powers, multiples, last digit, contained digit), each kind with its own inferred plausibility, and treat rules of the same kind as equally likely. So whole kinds of rule (e.g. "contains the digit d") can be implausible a priori, while magnitude generalisation follows ranges of a preferred size.

## rule_family_prior_concepts  — rank 6, 12.2 ± 4.8 nats behind the best (2.5× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1085.1

Refinement of `bigelow_piantadosi_mixture`: people do Bayesian concept learning over the same 45 rules and all intervals (size principle, 10% of examples random), but their prior over rules is organised by kind of rule rather than flat: they first favour a family of rule (parity, primes, squares/cubes, powers, multiples, last digit, contained digit), each with its own inferred plausibility, and rules within a family are equally likely. The one change is replacing the mixture's uniform rule prior with this inferred family-level prior, so whole kinds of rule (e.g. "contains the digit d") can be a priori implausible while others (multiples, last digit) dominate generalization.

## iter2_candidate3  — rank 7, 12.4 ± 4.6 nats behind the best (2.7× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1085.3

Refinement of the incumbent `erlang_interval_family_prior_concepts`: people do the same Bayesian concept learning (45 rules with a family-organised prior, magnitude ranges with an Erlang width prior, size principle), but on some judgments they do not consult the inferred concept at all and simply guess, answering "yes" with their own guessing bias. The one change is this attentional lapse: on an inferred share of trials the answer is a biased guess, so targets the learner is certain about are still sometimes answered the other way — "yes" to numbers outside every plausible concept and "no" to numbers inside it — rather than with near-zero or near-one probability.

## serial_position_weighted_evidence  — rank 8, 13.9 ± 4.8 nats behind the best (2.9× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1086.8

People do Bayesian concept learning over the same rules (with a family-organised rule prior) and magnitude ranges as the best model, but they update sequentially as the examples are read and do not weigh every example equally: each example's evidence is weighted by its position in the sequence, decaying (or growing) geometrically with how far it lies before the last example shown, at an inferred rate. A recency rate means earlier examples are partly forgotten, so longer sets narrow generalisation less than the size principle predicts; a primacy rate means early examples are overcounted, so longer sets narrow it more.

## lapse_guess_concept_learning  — rank 9, 14.1 ± 4.8 nats behind the best (2.9× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1087.0

People infer how likely the target is to belong to the hidden concept by Bayesian concept learning over rules (with a family-organised rule prior) and magnitude ranges, using the size principle, and normally answer with that probability; but on some fraction of judgments they lapse (inattention, a slip, not engaging with the examples) and answer by a fixed guessing tendency instead, saying "yes" with an inferred guessing rate regardless of the examples. So even targets the concept clearly excludes get a floor of "yes" answers, and targets it clearly includes get a ceiling below certainty, both set by one lapse rate and one guessing bias.

## offset_progression_concepts  — rank 10, 14.5 ± 6.3 nats behind the best (2.3× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1087.4

Refinement of `arbitrary_exception_rule_prior`: people do the same Bayesian concept learning over rules and magnitude ranges (size principle, Erlang prior over range width, prior shared by kind of rule), but their space of rules also contains arithmetic progressions that do not start at zero — "every third number starting from 1", "odd multiples of 3" (3, 9, 15, ...), "every fourth number starting from 2" — i.e. all numbers leaving the same nonzero remainder under a small step of 3 to 6. These offset progressions form their own kind of rule with its own inferred, presumably low, plausibility, so after examples such as 69, 9, 39, 21 people can generalise only to the odd multiples of 3 and reject other odd numbers such as 19.

## evidence_overweighting_concept_learning  — rank 11, 14.5 ± 4.9 nats behind the best (3.0× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1087.4

People do Bayesian concept learning over the same rules (with a family-organised rule prior) and magnitude intervals as the best model, using the size principle, but they mis-weight the evidence: each example's likelihood is raised to an inferred power, so the examples count for more (or less) than a normative learner would let them. If people over-weight evidence, their generalisation collapses onto the smallest fitting concept faster as examples accumulate than the size principle predicts, which would explain why their yes-rate drops more from two- to four-example sets than the best model expects.

## log_number_line_range_concepts  — rank 12, 23.4 ± 5.8 nats behind the best (4.1× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1096.3

Refinement of `rule_family_prior_concepts`: people do the same Bayesian concept learning over the 45 rules (with the family-organised rule prior) and magnitude ranges of 1-100, with size-principle sampling, but they picture the ranges on a compressed, roughly logarithmic mental number line rather than a linear one. The one change is the prior over ranges: a range's endpoints are equally likely to fall anywhere on that mental line, with the degree of compression inferred, so ranges among small numbers (where numbers are spread out on the mental line) are a priori more plausible than equally long ranges among large numbers (where they are crowded together). As a result, examples among large numbers are explained by a range less readily and by a rule more often than the incumbent predicts.

## membership_criterion_threshold  — rank 13, 33.9 ± 7.5 nats behind the best (4.5× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1106.8

People infer how likely the target is to belong to the hidden concept exactly as the current best model's Bayesian concept learner does (same rules and magnitude ranges, family-organised rule prior, size principle), but they do not report that probability directly: they answer "yes" only when the inferred membership probability exceeds an internal decision criterion, and this criterion varies noisily from judgment to judgment. So targets with middling membership probability are mostly rejected (or mostly accepted, depending on where the criterion sits), and answers are more all-or-none than probability matching predicts.

## win_stay_lose_shift_concept_testing  — rank 14, 48.6 ± 9.8 nats behind the best (5.0× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1121.6

People learn the concept by sequential hypothesis testing ("win-stay, lose-shift") rather than by weighing all examples at once: on reading the first example they pick one concept (a rule or a magnitude range, with the same family-organised rule prior and preferred-width range prior as the best model) in proportion to how well it explains that example, keep it as long as each new example fits it, and only when an example contradicts it do they pick a fresh concept in proportion to how well it explains all examples seen so far. They answer "yes" if the concept they end up holding contains the target, so the first example shapes generalisation most and later examples that fit a broad initial guess do not narrow it.

## erlang_interval_size_concept_learning  — rank 15, 91.8 ± 12.4 nats behind the best (7.4× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1164.7

A refinement of bigelow_piantadosi_independent: people do Bayesian concept learning over the same 45 rules and all intervals of 1-100 (prior weight per rule, share lambda_0 for rules, size-principle sampling with 10% random examples, "yes" = posterior probability the concept contains the target). The one change is the prior over intervals: instead of every interval being equally likely (which makes wide ranges a priori far more numerous), people expect a range of a characteristic width, with an Erlang prior over interval size whose scale is inferred from the data (as in Tenenbaum's number game), so moderately narrow ranges are preferred a priori and very wide or single-number ranges are penalised.

## bigelow_piantadosi_independent  — rank 16, 94.9 ± 12.2 nats behind the best (7.7× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1167.8

Bayesian concept learning over 45 rules (odd, even, primes, squares,
cubes, powers of 2-10, powers of 2 without 32 or with 37, multiples of
3-9, 11 and 12, ending in 0-9, containing the digit 0-9) and every
interval of 1-100, with a prior weight inferred for each rule separately
and a share lambda_0 for the rules as a group; each interval has prior
(1 - lambda_0) / 100^2. Examples are sampled from the concept (the size
principle), 10% of them uniformly from 1-100, and "yes" is the posterior
probability that the concept contains the target.

## weak_sampling_concept_learning  — rank 17, 212.6 ± 20.9 nats behind the best (10.2× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1285.5

People entertain the same kinds of number concepts (arithmetic rules and magnitude ranges) but treat the examples only as consistent with the hidden concept rather than as random samples drawn from it (weak sampling, no size principle), so every concept that contains the examples stays plausible in proportion to its prior regardless of how many other numbers it includes. The sharpest disagreement with the size-principle best model lives in sets that fit a small rule nested in a larger one (e.g. 16,32,2,8 or 20,32,92): this model predicts broad "yes" to targets that only the larger concept contains (other even numbers, numbers anywhere in a wide range), whereas the best model confines "yes" to the smallest fitting concept.

## bigelow_piantadosi_mixture  — rank 18, 264.2 ± 28.1 nats behind the best (9.4× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1337.1

Bayesian concept learning over 45 rules (odd, even, primes, squares,
cubes, powers of 2-10, powers of 2 without 32 or with 37, multiples of
3-9, 11 and 12, ending in 0-9, containing the digit 0-9) and every
interval of 1-100. A rule is chosen with probability lambda_0, and the
rules are equally likely; each interval has prior (1 - lambda_0) / 100^2.
Examples are sampled from the concept (the size principle), 10% of them
uniformly from 1-100, and "yes" is the posterior probability that the
concept contains the target.

## exemplar_featural_similarity_gcm  — rank 19, 399.9 ± 27.2 nats behind the best (14.7× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1472.8

People do not infer a rule at all: they judge the target by its exemplar similarity to the examples shown, as in a generalized context model. Each example is compared with the target on a few perceptual dimensions of the number — its magnitude, its final digit, its tens digit and its parity — with attention shared among these dimensions, similarity decays exponentially with attention-weighted mismatch, and the more similar the target is on average to the examples, the more likely they say yes.

## property_matching_exemplar_similarity  — rank 20, 403.6 ± 27.3 nats behind the best (14.8× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1476.5

People do not infer a hidden concept: they store the shown examples as exemplars and judge the target by how similar it is to them, where two numbers are similar to the extent that they share familiar mathematical properties (parity, primality, being a square or cube, being a power of some base, being a multiple of some number, final digit, digits contained) and are close in magnitude. Each kind of property receives its own attention weight, similarity decays exponentially with the attention-weighted count of properties on which target and example differ, and people say "yes" with a Luce-choice probability that grows with the target's average similarity to the stored examples relative to a fixed response criterion.

## residue_class_progression  — rank 21, 457.3 ± 37.8 nats behind the best (12.1× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1530.2

People assume the hidden program produces an arithmetic progression: all numbers in 1-100 that leave the same remainder when divided by some step size (step 1 = every number, step 2 = all odds or all evens, step 10 = a shared final digit, step 12 = every twelfth number, and so on). They consider every step size whose residue class contains all the examples, favour smaller steps a priori but larger (tighter) steps when several examples all fit them (the size principle), and say "yes" with the posterior probability that the target lies in the same progression as the examples.

## spread_scaled_distance_gradient  — rank 22, 459.4 ± 28.7 nats behind the best (16.0× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1532.3

People judge the target purely by its numerical closeness to the nearest example on the 1-100 number line, with a Shepard-style exponential generalization gradient whose width is set by how spread out the examples are: they say "yes" with a probability that falls off exponentially with the target's distance from the nearest example, and widely spaced examples (large average gap between neighbouring examples) license a proportionally broader gradient than tightly clustered ones.

## doubling_chain_concepts  — no comparison row

People do the same Bayesian concept learning as the best model (family-organised rule prior with exception rules as their own kind, magnitude ranges and odd/even-within-a-range concepts with a preferred width, and the size principle), but they also hold a kind of concept the best model lacks: multiplicative chains, in which each number is double (or triple) the one before, starting from some number other than 1 (e.g. 3, 6, 12, 24, 48, 96 or 5, 10, 20, 40, 80 or 2, 6, 18, 54). This kind has its own inferred plausibility. The disagreement with the best model shows up on example sets that form such a chain (e.g. 5, 10, 20 or 6, 12, 24): there this model predicts "yes" to the next links (40, 80 or 48, 96) and "no" to other multiples (15, 25, 30 or 18, 30, 36), whereas the best model spreads "yes" over all multiples of 5 (or 6) and nearby numbers.

## rule_noticing_grows_with_examples  — no comparison row

People do the same Bayesian concept learning over arithmetic rules, magnitude ranges and odd/even-within-a-range concepts as the best model, but whether a rule even comes to mind depends on how many examples they have seen: the prior odds of thinking of a rule rather than a magnitude range grow steadily, at an inferred rate, with each additional example. So after one or two examples people generalise mostly by magnitude, and after four examples they look for a rule more readily than the examples' likelihood alone would warrant.

## gamma_width_parity_range_concepts  — no comparison row

Refinement of the incumbent `parity_within_range_concepts`: people do the same Bayesian concept learning over rules (family-organised rule prior, exception rules as their own kind), magnitude ranges and "odd (or even) numbers within a range", with the size principle. The one change, taken from `gamma_width_exception_family_concepts`, is the shape of their expectation about how wide a range (plain or parity-restricted) is: a gamma prior whose sharpness is inferred along with its typical width, instead of the fixed, loose Erlang form — so people may expect ranges of one sharply characteristic width, strongly disfavouring both very narrow and very wide ranges around the examples.
