# Tried before

81 hypotheses proposed earlier in this project are no longer in the model set. Read them before you propose:

- A *pruned* entry entered the set and later lost to the best model by the stated margin, on the data available then. Its mechanism may be partly right: a model that changes it substantively is welcome, but do not re-propose it unchanged or merely re-parameterised, under any name. Pruned models stay readable under `models/pruned/`.
- A *rejected* entry never entered the set. If it was a near-duplicate of a model still in the set, that region is already covered: do not re-propose it. If it failed on its code or its fit (see its outcome), the idea itself was never tested and a correct implementation may be worth trying.

### exemplar_featural_similarity_gcm — pruned (experiment1 end of experiment)

**Outcome:** 400.3 nats behind gamma_width_parity_range_concepts (14.7× dse)

**Hypothesis:** People do not infer a rule at all: they judge the target by its exemplar similarity to the examples shown, as in a generalized context model. Each example is compared with the target on a few perceptual dimensions of the number — its magnitude, its final digit, its tens digit and its parity — with attention shared among these dimensions, similarity decays exponentially with attention-weighted mismatch, and the more similar the target is on average to the examples, the more likely they say yes.

### weak_sampling_concept_learning — pruned (experiment1 end of experiment)

**Outcome:** 213.0 nats behind gamma_width_parity_range_concepts (10.2× dse)

**Hypothesis:** People entertain the same kinds of number concepts (arithmetic rules and magnitude ranges) but treat the examples only as consistent with the hidden concept rather than as random samples drawn from it (weak sampling, no size principle), so every concept that contains the examples stays plausible in proportion to its prior regardless of how many other numbers it includes. The sharpest disagreement with the size-principle best model lives in sets that fit a small rule nested in a larger one (e.g. 16,32,2,8 or 20,32,92): this model predicts broad "yes" to targets that only the larger concept contains (other even numbers, numbers anywhere in a wide range), whereas the best model confines "yes" to the smallest fitting concept.

### residue_class_progression — pruned (experiment1 end of experiment)

**Outcome:** 457.7 nats behind gamma_width_parity_range_concepts (12.1× dse)

**Hypothesis:** People assume the hidden program produces an arithmetic progression: all numbers in 1-100 that leave the same remainder when divided by some step size (step 1 = every number, step 2 = all odds or all evens, step 10 = a shared final digit, step 12 = every twelfth number, and so on). They consider every step size whose residue class contains all the examples, favour smaller steps a priori but larger (tighter) steps when several examples all fit them (the size principle), and say "yes" with the posterior probability that the target lies in the same progression as the examples.

### inferred_sampling_noise_concept_learning — rejected (experiment1 round 0 candidate 3 refine incumbent bigelow_piantadosi_independent)

**Outcome:** predicts like existing model 'bigelow_piantadosi_independent' (p_yes RMSE 0.00165 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of bigelow_piantadosi_independent, not a new hypothesis.

**Hypothesis:** A refinement of bigelow_piantadosi_independent: people do Bayesian concept learning over the same 45 rules and all intervals of 1-100, with a prior weight for each rule and a share lambda_0 for rules, and answer "yes" with the posterior probability that the concept contains the target. The one change is that how far people trust the examples to have come from the concept (the strength of the size principle, versus examples being random numbers from 1-100) is inferred from their judgments rather than fixed at 90%, so the data decide how strongly small concepts are favoured and how readily a concept is kept despite an example it does not contain.

### inferred_outlier_rate_concept_learning — rejected (experiment1 round 0 candidate 4 refine incumbent bigelow_piantadosi_independent)

**Outcome:** predicts like existing model 'bigelow_piantadosi_independent' (p_yes RMSE 0.00130 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of bigelow_piantadosi_independent, not a new hypothesis.

**Hypothesis:** A refinement of `bigelow_piantadosi_independent`: people do Bayesian concept learning over the same 45 rules and all intervals of 1-100 with independently weighted rules and the size principle, but how far they trust each example to come from the concept is not fixed at 90%. Instead the rate at which examples are believed to be random noise drawn uniformly from 1-100 is a free parameter inferred from the data, so that sets containing an "odd one out" (e.g. 8, 24, 48, 63) are explained by a rule plus an outlier to the degree people actually tolerate outliers.

### rule_family_prior_concepts — pruned (experiment1 end of experiment)

**Outcome:** 12.6 nats behind gamma_width_parity_range_concepts (2.6× dse)

**Hypothesis:** Refinement of `bigelow_piantadosi_mixture`: people do Bayesian concept learning over the same 45 rules and all intervals (size principle, 10% of examples random), but their prior over rules is organised by kind of rule rather than flat: they first favour a family of rule (parity, primes, squares/cubes, powers, multiples, last digit, contained digit), each with its own inferred plausibility, and rules within a family are equally likely. The one change is replacing the mixture's uniform rule prior with this inferred family-level prior, so whole kinds of rule (e.g. "contains the digit d") can be a priori implausible while others (multiples, last digit) dominate generalization.

### erlang_interval_size_concept_learning — pruned (experiment1 end of experiment)

**Outcome:** 92.2 nats behind gamma_width_parity_range_concepts (7.4× dse)

**Hypothesis:** A refinement of bigelow_piantadosi_independent: people do Bayesian concept learning over the same 45 rules and all intervals of 1-100 (prior weight per rule, share lambda_0 for rules, size-principle sampling with 10% random examples, "yes" = posterior probability the concept contains the target). The one change is the prior over intervals: instead of every interval being equally likely (which makes wide ranges a priori far more numerous), people expect a range of a characteristic width, with an Erlang prior over interval size whose scale is inferred from the data (as in Tenenbaum's number game), so moderately narrow ranges are preferred a priori and very wide or single-number ranges are penalised.

### interval_size_erlang_prior — rejected (experiment1 round 0 candidate 4 refine incumbent bigelow_piantadosi_independent repair 1)

**Outcome:** predicts like existing model 'erlang_interval_size_concept_learning' (p_yes RMSE 0.00000 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of erlang_interval_size_concept_learning, not a new hypothesis.

**Hypothesis:** A refinement of `bigelow_piantadosi_independent`: people do Bayesian concept learning over the same 45 independently weighted rules and all intervals of 1-100 with the size principle (10% of examples treated as noise), but their prior over magnitude ranges is not flat over endpoint pairs. Instead people expect a range of a characteristic size: the prior on an interval's width follows an Erlang distribution (as in Tenenbaum's number game) whose scale is inferred from the data, so ranges much narrower or much wider than that typical size are judged implausible a priori, changing how far "yes" extends around examples that suggest a magnitude range.

### serial_position_weighted_evidence — pruned (experiment1 end of experiment)

**Outcome:** 14.2 nats behind gamma_width_parity_range_concepts (2.9× dse)

**Hypothesis:** People do Bayesian concept learning over the same rules (with a family-organised rule prior) and magnitude ranges as the best model, but they update sequentially as the examples are read and do not weigh every example equally: each example's evidence is weighted by its position in the sequence, decaying (or growing) geometrically with how far it lies before the last example shown, at an inferred rate. A recency rate means earlier examples are partly forgotten, so longer sets narrow generalisation less than the size principle predicts; a primacy rate means early examples are overcounted, so longer sets narrow it more.

### evidence_overweighting_concept_learning — pruned (experiment1 end of experiment)

**Outcome:** 14.9 nats behind gamma_width_parity_range_concepts (3.1× dse)

**Hypothesis:** People do Bayesian concept learning over the same rules (with a family-organised rule prior) and magnitude intervals as the best model, using the size principle, but they mis-weight the evidence: each example's likelihood is raised to an inferred power, so the examples count for more (or less) than a normative learner would let them. If people over-weight evidence, their generalisation collapses onto the smallest fitting concept faster as examples accumulate than the size principle predicts, which would explain why their yes-rate drops more from two- to four-example sets than the best model expects.

### posterior_sharpened_concept_selection — rejected (experiment1 round 1 candidate 2 lens 5)

**Outcome:** predicts like existing model 'evidence_overweighting_concept_learning' (p_yes RMSE 0.00121 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of evidence_overweighting_concept_learning, not a new hypothesis.

**Hypothesis:** People do not average their predictions over every number concept consistent with the examples; they commit to (approximately) the single most plausible concept and generalize only from it. Concretely, given the same rules and magnitude ranges, family-organised rule prior and size-principle likelihood as the current best model, the concept a person generalizes from is chosen with probability proportional to its posterior raised to an inferred decisiveness exponent (above 1 = near-MAP hypothesis selection, so generalization collapses onto the smallest well-fitting concept more sharply as examples accumulate), and "yes" means the chosen concept contains the target.

### iter1_candidate3 — rejected (experiment1 round 1 candidate 3 refine incumbent rule_family_prior_concepts)

**Outcome:** predicts like existing model 'rule_family_prior_concepts' (p_yes RMSE 0.00180 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of rule_family_prior_concepts, not a new hypothesis.

**Hypothesis:** Refinement of `rule_family_prior_concepts`: people do the same Bayesian concept learning over the 45 rules (with the inferred family-level rule prior) and all intervals, with "yes" the posterior probability that the concept contains the target. The one change is how strongly people believe the examples were sampled from the concept: instead of a fixed 90% (10% of examples random noise), the share of examples they treat as genuine samples from the concept is a free parameter inferred from the data, so the strength of the size principle — how sharply generalization narrows as consistent examples accumulate, and how readily an inconsistent example is dismissed as noise — is estimated rather than assumed.

### inferred_sampling_reliability_concepts — rejected (experiment1 round 1 candidate 4 refine incumbent rule_family_prior_concepts)

**Outcome:** predicts like existing model 'evidence_overweighting_concept_learning' (p_yes RMSE 0.00000 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of evidence_overweighting_concept_learning, not a new hypothesis.

**Hypothesis:** Refinement of `rule_family_prior_concepts`: people do the same Bayesian concept learning over the 45 rules (with the family-organised rule prior) and all intervals of 1-100, with size-principle sampling, but how strongly they treat the examples as drawn from the concept is not fixed at 90% — the share of examples they believe are genuine samples of the concept (versus random numbers from 1-100) is a free quantity inferred from their judgments. The one change is replacing the fixed 10% noise rate with this inferred sampling reliability, which sets how sharply each extra example narrows generalization to the smallest concept that fits (the critique shows people narrow with more examples differently from what the fixed rate predicts) and how readily a concept that misses one example is still entertained.

### erlang_interval_family_prior_concepts — pruned (experiment1 end of experiment)

**Outcome:** 11.0 nats behind gamma_width_parity_range_concepts (2.4× dse)

**Hypothesis:** Refinement of `erlang_interval_size_concept_learning` (the narrowest-losing live model): people do Bayesian concept learning over the same 45 rules and magnitude ranges, with the size principle and an Erlang prior that favours ranges of a characteristic, inferred width over very wide or single-number ones. The one change is how prior belief is shared among the rules: instead of an independent weight for each of the 45 rules, people first favour a kind of rule (parity, primes, squares/cubes, powers, multiples, last digit, contained digit), each kind with its own inferred plausibility, and treat rules of the same kind as equally likely. So whole kinds of rule (e.g. "contains the digit d") can be implausible a priori, while magnitude generalisation follows ranges of a preferred size.

### membership_criterion_threshold — pruned (experiment1 end of experiment)

**Outcome:** 34.2 nats behind gamma_width_parity_range_concepts (4.6× dse)

**Hypothesis:** People infer how likely the target is to belong to the hidden concept exactly as the current best model's Bayesian concept learner does (same rules and magnitude ranges, family-organised rule prior, size principle), but they do not report that probability directly: they answer "yes" only when the inferred membership probability exceeds an internal decision criterion, and this criterion varies noisily from judgment to judgment. So targets with middling membership probability are mostly rejected (or mostly accepted, depending on where the criterion sits), and answers are more all-or-none than probability matching predicts.

### family_prior_erlang_interval_concepts — rejected (experiment1 round 1 candidate 3 refine incumbent rule_family_prior_concepts repair 1)

**Outcome:** predicts like existing model 'erlang_interval_family_prior_concepts' (p_yes RMSE 0.00176 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of erlang_interval_family_prior_concepts, not a new hypothesis.

**Hypothesis:** Refinement of `rule_family_prior_concepts`: people do the same Bayesian concept learning over the 45 rules, with the inferred family-level rule prior, and over intervals of 1-100, using the size principle (10% of examples random), with "yes" the posterior probability that the concept contains the target. The one change, taken from `erlang_interval_size_concept_learning`, is the prior over intervals: instead of every interval being equally likely, which makes wide ranges far more numerous a priori, people expect a range of a characteristic width (an Erlang prior over interval size with an inferred scale). Narrow, magnitude-based generalisation is then favoured from the first few examples, so adding examples narrows the range less than the flat-interval model predicts.

### log_number_line_range_concepts — pruned (experiment1 end of experiment)

**Outcome:** 23.7 nats behind gamma_width_parity_range_concepts (4.1× dse)

**Hypothesis:** Refinement of `rule_family_prior_concepts`: people do the same Bayesian concept learning over the 45 rules (with the family-organised rule prior) and magnitude ranges of 1-100, with size-principle sampling, but they picture the ranges on a compressed, roughly logarithmic mental number line rather than a linear one. The one change is the prior over ranges: a range's endpoints are equally likely to fall anywhere on that mental line, with the degree of compression inferred, so ranges among small numbers (where numbers are spread out on the mental line) are a priori more plausible than equally long ranges among large numbers (where they are crowded together). As a result, examples among large numbers are explained by a range less readily and by a rule more often than the incumbent predicts.

### win_stay_lose_shift_concept_testing — pruned (experiment1 end of experiment)

**Outcome:** 49.0 nats behind gamma_width_parity_range_concepts (5.0× dse)

**Hypothesis:** People learn the concept by sequential hypothesis testing ("win-stay, lose-shift") rather than by weighing all examples at once: on reading the first example they pick one concept (a rule or a magnitude range, with the same family-organised rule prior and preferred-width range prior as the best model) in proportion to how well it explains that example, keep it as long as each new example fits it, and only when an example contradicts it do they pick a fresh concept in proportion to how well it explains all examples seen so far. They answer "yes" if the concept they end up holding contains the target, so the first example shapes generalisation most and later examples that fit a broad initial guess do not narrow it.

### spread_scaled_distance_gradient — pruned (experiment1 end of experiment)

**Outcome:** 459.8 nats behind gamma_width_parity_range_concepts (16.0× dse)

**Hypothesis:** People judge the target purely by its numerical closeness to the nearest example on the 1-100 number line, with a Shepard-style exponential generalization gradient whose width is set by how spread out the examples are: they say "yes" with a probability that falls off exponentially with the target's distance from the nearest example, and widely spaced examples (large average gap between neighbouring examples) license a proportionally broader gradient than tightly clustered ones.

### lapse_guess_concept_learning — pruned (experiment1 end of experiment)

**Outcome:** 14.4 nats behind gamma_width_parity_range_concepts (3.0× dse)

**Hypothesis:** People infer how likely the target is to belong to the hidden concept by Bayesian concept learning over rules (with a family-organised rule prior) and magnitude ranges, using the size principle, and normally answer with that probability; but on some fraction of judgments they lapse (inattention, a slip, not engaging with the examples) and answer by a fixed guessing tendency instead, saying "yes" with an inferred guessing rate regardless of the examples. So even targets the concept clearly excludes get a floor of "yes" answers, and targets it clearly includes get a ceiling below certainty, both set by one lapse rate and one guessing bias.

### iter2_candidate3 — pruned (experiment1 end of experiment)

**Outcome:** 12.7 nats behind gamma_width_parity_range_concepts (2.8× dse)

**Hypothesis:** Refinement of the incumbent `erlang_interval_family_prior_concepts`: people do the same Bayesian concept learning (45 rules with a family-organised prior, magnitude ranges with an Erlang width prior, size principle), but on some judgments they do not consult the inferred concept at all and simply guess, answering "yes" with their own guessing bias. The one change is this attentional lapse: on an inferred share of trials the answer is a biased guess, so targets the learner is certain about are still sometimes answered the other way — "yes" to numbers outside every plausible concept and "no" to numbers inside it — rather than with near-zero or near-one probability.

### gamma_width_family_prior_concepts — pruned (experiment1 end of experiment)

**Outcome:** 10.1 nats behind gamma_width_parity_range_concepts (2.2× dse)

**Hypothesis:** Refinement of `erlang_interval_family_prior_concepts` (the incumbent): people do the same Bayesian concept learning over the 45 rules (with the family-organised rule prior) and magnitude ranges of 1-100, with the size principle. The one change is the shape of their prior expectation about how wide a range is: instead of a fixed Erlang form (whose preference for a typical width is always of the same, fairly loose shape), the width prior is a gamma distribution whose shape is inferred along with its typical width, so people may hold a sharp expectation that ranges have one characteristic width (strongly disfavouring both narrow and very wide ranges) or, at the other extreme, a monotone preference for the narrowest range that fits the examples.

### arbitrary_exception_rule_prior — pruned (experiment1 end of experiment)

**Outcome:** 3.8 nats behind gamma_width_parity_range_concepts; retired to keep the live set at 8 models: ELPD-LOO rank 9 of 10

**Hypothesis:** Refinement of `erlang_interval_family_prior_concepts`: people do the same Bayesian concept learning over rules and magnitude ranges (size principle, Erlang prior over range width, prior shared by kind of rule), but a rule with an arbitrary exception ("powers of 2 except 32", "powers of 2 plus 37") is not just another power rule to them — it belongs to its own kind, with its own inferred, presumably very low, plausibility. The one change is giving these exception-laden rules their own family in the rule prior, so after seeing powers of 2 people firmly reject 37 (and accept 32), instead of giving 37 the roughly one-in-three "yes" that the incumbent predicts.

### offset_progression_concepts — pruned (experiment1 end of experiment)

**Outcome:** 14.9 nats behind gamma_width_parity_range_concepts (2.3× dse)

**Hypothesis:** Refinement of `arbitrary_exception_rule_prior`: people do the same Bayesian concept learning over rules and magnitude ranges (size principle, Erlang prior over range width, prior shared by kind of rule), but their space of rules also contains arithmetic progressions that do not start at zero — "every third number starting from 1", "odd multiples of 3" (3, 9, 15, ...), "every fourth number starting from 2" — i.e. all numbers leaving the same nonzero remainder under a small step of 3 to 6. These offset progressions form their own kind of rule with its own inferred, presumably low, plausibility, so after examples such as 69, 9, 39, 21 people can generalise only to the odd multiples of 3 and reject other odd numbers such as 19.

### property_matching_exemplar_similarity — pruned (experiment1 end of experiment)

**Outcome:** 404.0 nats behind gamma_width_parity_range_concepts (14.8× dse)

**Hypothesis:** People do not infer a hidden concept: they store the shown examples as exemplars and judge the target by how similar it is to them, where two numbers are similar to the extent that they share familiar mathematical properties (parity, primality, being a square or cube, being a power of some base, being a multiple of some number, final digit, digits contained) and are close in magnitude. Each kind of property receives its own attention weight, similarity decays exponentially with the attention-weighted count of properties on which target and example differ, and people say "yes" with a Luce-choice probability that grows with the target's average similarity to the stored examples relative to a fixed response criterion.

### generative_next_output_judgment — dropped (experiment1 round 3)

**Outcome:** MCMC fit failed on CV fold 1 (RuntimeError: warm-start NUTS: the Hessian at the posterior mode is not positive definite (some parameter direction is pinned down by neither the data nor the prior))

**Hypothesis:** People read the question "would the program produce this number?" generatively: instead of asking whether the target belongs to the hidden concept, they judge how likely the program is to produce that particular number as its next output (averaging over the concepts that explain the examples, each of which spreads its output thinly when it contains many numbers), and say "yes" in proportion to how that likelihood compares with the most likely next output, with an inferred sensitivity. So targets that only a large concept contains (any even number, anywhere in a wide range) get far fewer "yes" answers than their membership probability implies, while targets of a small, well-fitting concept are accepted.

### parity_within_range_concepts — pruned (experiment2 end of experiment)

**Outcome:** 20.7 nats behind mixed_strong_weak_sampling_concepts (2.8× dse)

**Hypothesis:** Refinement of the incumbent `arbitrary_exception_rule_prior`: people do the same Bayesian concept learning over rules (family-organised rule prior, exception rules as their own kind) and magnitude ranges with a preferred width and the size principle, but they can also entertain a conjunctive kind of concept, "odd (or even) numbers within a range" (e.g. "even numbers above 60"), with its own inferred plausibility and the same width preference. The one change is adding this parity-within-a-range kind of concept, so that after 4, 66, 78, 96 people say yes to large even numbers (70, 74) but no to small ones (8, 22), and after 23, 35, 95, 83 they reject even numbers between the examples, rather than spreading "yes" evenly over all even (or all in-range) numbers.

### decade_concept_family_prior — pruned (experiment1 end of experiment)

**Outcome:** 4.3 nats behind gamma_width_parity_range_concepts; retired to keep the live set at 8 models: ELPD-LOO rank 10 of 10

**Hypothesis:** Refinement of `arbitrary_exception_rule_prior` (the incumbent): people do the same Bayesian concept learning over rules and magnitude ranges (size principle, Erlang prior over range width, rule prior shared by kind of rule, exception-laden rules as their own kind), but they also hold a ready-made kind of concept that the incumbent lacks: the decade, "numbers in the 60s" (same tens digit), as a distinct kind of rule with its own inferred plausibility. The one change is adding the ten decade concepts as a new family in the rule prior, so when the examples share a tens digit (e.g. 61, 64, 67) people confine "yes" sharply to that decade, more firmly than an arbitrary range of the same width.

### gamma_width_exception_family_concepts — pruned (experiment2 end of experiment)

**Outcome:** 7.2 nats behind mixed_strong_weak_sampling_concepts; retired to keep the live set at 8 models: ELPD-LOO rank 14 of 15

**Hypothesis:** Refinement of `gamma_width_family_prior_concepts`: people do the same Bayesian concept learning over rules and magnitude ranges (size principle, family-organised rule prior, and a gamma prior over range width whose sharpness is inferred along with its typical width). The one change, taken from `arbitrary_exception_rule_prior`, is that rules with an arbitrary exception ("powers of 2 except 32", "powers of 2 plus 37") form their own kind of rule with their own, presumably very low, plausibility, instead of being ordinary power rules. So people expect ranges of a sharply characteristic width and also firmly reject 37 (and accept 32) after seeing powers of 2.

### doubling_chain_concepts — pruned (experiment2 end of experiment)

**Outcome:** 28.8 nats behind mixed_strong_weak_sampling_concepts (3.2× dse)

**Hypothesis:** People do the same Bayesian concept learning as the best model (family-organised rule prior with exception rules as their own kind, magnitude ranges and odd/even-within-a-range concepts with a preferred width, and the size principle), but they also hold a kind of concept the best model lacks: multiplicative chains, in which each number is double (or triple) the one before, starting from some number other than 1 (e.g. 3, 6, 12, 24, 48, 96 or 5, 10, 20, 40, 80 or 2, 6, 18, 54). This kind has its own inferred plausibility. The disagreement with the best model shows up on example sets that form such a chain (e.g. 5, 10, 20 or 6, 12, 24): there this model predicts "yes" to the next links (40, 80 or 48, 96) and "no" to other multiples (15, 25, 30 or 18, 30, 36), whereas the best model spreads "yes" over all multiples of 5 (or 6) and nearby numbers.

### rule_noticing_grows_with_examples — pruned (experiment2 end of experiment)

**Outcome:** 23.3 nats behind mixed_strong_weak_sampling_concepts (3.0× dse)

**Hypothesis:** People do the same Bayesian concept learning over arithmetic rules, magnitude ranges and odd/even-within-a-range concepts as the best model, but whether a rule even comes to mind depends on how many examples they have seen: the prior odds of thinking of a rule rather than a magnitude range grow steadily, at an inferred rate, with each additional example. So after one or two examples people generalise mostly by magnitude, and after four examples they look for a rule more readily than the examples' likelihood alone would warrant.

### example_recall_limited_concepts — pruned (experiment2 end of experiment)

**Outcome:** 20.5 nats behind mixed_strong_weak_sampling_concepts (2.8× dse)

**Hypothesis:** People do the same Bayesian concept learning as the best model (rules with a family-organised prior and exception rules as their own kind, magnitude ranges and odd/even-within-a-range concepts with a preferred width, size principle), but their memory for the examples is limited: each example shown is retained, independently, only with an inferred probability, and they generalise from whichever examples they still hold. So a judgment is sometimes based on only part of the set, which broadens generalisation for longer sets (a forgotten example no longer rules out a broader or different concept) in proportion to how many examples could have been lost.

### gamma_width_parity_range_concepts — pruned (experiment2 end of experiment)

**Outcome:** 8.9 nats behind mixed_strong_weak_sampling_concepts (2.2× dse)

**Hypothesis:** Refinement of the incumbent `parity_within_range_concepts`: people do the same Bayesian concept learning over rules (family-organised rule prior, exception rules as their own kind), magnitude ranges and "odd (or even) numbers within a range", with the size principle. The one change, taken from `gamma_width_exception_family_concepts`, is the shape of their expectation about how wide a range (plain or parity-restricted) is: a gamma prior whose sharpness is inferred along with its typical width, instead of the fixed, loose Erlang form — so people may expect ranges of one sharply characteristic width, strongly disfavouring both very narrow and very wide ranges around the examples.

### gamma_width_parity_range_concepts_2 — rejected (experiment1 round 4 candidate 5 refine chosen)

**Outcome:** predicts like existing model 'gamma_width_parity_range_concepts' (p_yes RMSE 0.00000 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of gamma_width_parity_range_concepts, not a new hypothesis.

**Hypothesis:** Refinement of `gamma_width_exception_family_concepts`: people do the same Bayesian concept learning over rules and magnitude ranges (size principle, family-organised rule prior with exception-laden rules as their own kind, and a gamma prior over range width whose sharpness is inferred along with its typical width). The one change, taken from `parity_within_range_concepts`, is that people can also entertain a conjunctive kind of concept, "odd (or even) numbers within a range" (e.g. "even numbers above 60"), with its own inferred plausibility and the same sharply characteristic width preference as plain ranges. So after 4, 66, 78, 96 people say yes to large even numbers but no to small ones, and the ranges they generalise over — plain or parity-restricted — are of one characteristic width rather than extending loosely beyond the examples.

### multiples_within_range_concepts — pruned (experiment2 end of experiment)

**Outcome:** 23.7 nats behind mixed_strong_weak_sampling_concepts (3.1× dse)

**Hypothesis:** Refinement of the incumbent `parity_within_range_concepts`: people do the same Bayesian concept learning over rules (family-organised prior, exception rules as their own kind), magnitude ranges and "odd (or even) numbers within a range", with the Erlang width preference and the size principle. The one change is that the conjunctive "rule within a range" idea extends beyond parity: people also entertain "multiples of m within a range" (e.g. "multiples of 3 between 10 and 60", "multiples of 9 from 27 to 81") as a further kind of concept with its own inferred plausibility and the same width preference. So after 39, 60, 12, 45 or 63, 27, 81 they accept multiples near the examples (e.g. 36, 54) but reject multiples far outside the examples' span (e.g. 90, 99, or 3), instead of spreading "yes" over all multiples of 3 or over every number in the range.

### decade_and_parity_range_concepts — pruned (experiment2 end of experiment)

**Outcome:** 22.8 nats behind mixed_strong_weak_sampling_concepts (2.8× dse)

**Hypothesis:** Refinement of `decade_concept_family_prior` (tied with the best): people do the same Bayesian concept learning over rules (family-organised prior, exception-laden rules as their own kind, and decades such as "numbers in the 60s" as their own kind of rule) and magnitude ranges with a preferred width and the size principle. The one change, taken from `parity_within_range_concepts`, is that people can also entertain "odd (or even) numbers within a range" as a kind of concept with its own inferred plausibility, so examples sharing a tens digit (61, 64, 67) confine "yes" to that decade while examples such as 4, 66, 78, 96 confine it to large even numbers — two ready-made ways of bounding generalisation that the incumbent and the decade model each have only one of.

### bigelow_piantadosi_mixture — pruned (experiment1 end of experiment)

**Outcome:** 264.6 nats behind gamma_width_parity_range_concepts (9.4× dse)

**Hypothesis:** Bayesian concept learning over 45 rules (odd, even, primes, squares, cubes, powers of 2-10, powers of 2 without 32 or with 37, multiples of 3-9, 11 and 12, ending in 0-9, containing the digit 0-9) and every interval of 1-100. A rule is chosen with probability lambda_0, and the rules are equally likely; each interval has prior (1 - lambda_0) / 100^2. Examples are sampled from the concept (the size principle), 10% of them uniformly from 1-100, and "yes" is the posterior probability that the concept contains the target.

### bigelow_piantadosi_independent — pruned (experiment1 end of experiment)

**Outcome:** 95.2 nats behind gamma_width_parity_range_concepts (7.8× dse)

**Hypothesis:** Bayesian concept learning over 45 rules (odd, even, primes, squares, cubes, powers of 2-10, powers of 2 without 32 or with 37, multiples of 3-9, 11 and 12, ending in 0-9, containing the digit 0-9) and every interval of 1-100, with a prior weight inferred for each rule separately and a share lambda_0 for the rules as a group; each interval has prior (1 - lambda_0) / 100^2. Examples are sampled from the concept (the size principle), 10% of them uniformly from 1-100, and "yes" is the posterior probability that the concept contains the target.

### prior_leaking_membership_judgment — pruned (experiment2 end of experiment)

**Outcome:** 5.1 nats behind mixed_strong_weak_sampling_concepts; retired to keep the live set at 8 models: ELPD-LOO rank 13 of 15

**Hypothesis:** People infer the hidden concept as a normative Bayesian concept learner does (rules with a family-organised prior and arbitrary-exception rules as their own kind, magnitude ranges with a preferred width, the size principle), but their yes/no judgment is distorted by how plausible the target was before they saw any examples: the log-odds of saying yes is the posterior log-odds that the concept contains the target, shifted by an inferred share of the target's prior log-odds of membership. So a priori "typical" numbers (mid-range, even, multiples) get more "yes" than the examples warrant (prior leakage), or, if the share is negative, less (contrast against what was expected anyway).

### without_replacement_sampling_concepts — pruned (experiment2 end of experiment)

**Outcome:** 15.1 nats behind mixed_strong_weak_sampling_concepts (3.0× dse)

**Hypothesis:** People do the same Bayesian concept learning as the best model (rules with a family-organised prior, arbitrary-exception rules as their own kind, magnitude ranges of a characteristic width), but they assume the program never repeats a number, so they treat the examples as drawn from the concept without replacement: each further example is one of the concept's members not yet shown. As a result, examples that nearly exhaust a small concept (e.g. 8, 64 or 81, 9) point to that concept, such as cubes or powers of 3, much more strongly than under sampling with replacement, and generalisation narrows faster onto small, nearly exhausted rules.

### first_example_trust_concepts — pruned (experiment2 end of experiment)

**Outcome:** 8.1 nats behind mixed_strong_weak_sampling_concepts (2.2× dse)

**Hypothesis:** People treat the first example they see differently from the later ones when judging how far each example is a genuine output of the hidden concept: the first example sets the concepts they entertain and is trusted to belong to the concept at its own inferred rate. Each later example is trusted at the usual fixed rate, and one that does not fit can be dismissed as a stray output. Apart from this, they generalise by the same Bayesian concept learning as the best model (size principle, rule prior organised by kind with exception rules as their own kind, gamma prior over range width). So if the first example is trusted more, an odd-one-out example is discounted when it comes late in the set (8, 24, 48, 63 still suggests multiples of 8) but forces a broader concept when it comes first.

### inferred_stray_rate_concepts — pruned (experiment2 end of experiment)

**Outcome:** 7.2 nats behind mixed_strong_weak_sampling_concepts (2.1× dse)

**Hypothesis:** Refinement of `gamma_width_exception_family_concepts`: people do the same Bayesian concept learning over rules (family-organised prior, exception rules as their own kind) and magnitude ranges with a gamma width preference and the size principle, and they also allow that any example may be a stray number the program produced at random rather than a member of the concept. The one change is that how often they expect such strays is no longer fixed at one example in ten but is a belief inferred from their judgments: the more strays they expect, the more readily they keep concepts that miss an example (and so sometimes reject an example they were shown), and the weaker the size principle becomes.

### gamma_width_doubling_chain_concepts — pruned (experiment2 end of experiment)

**Outcome:** 19.3 nats behind mixed_strong_weak_sampling_concepts (2.8× dse)

**Hypothesis:** Refinement of `doubling_chain_concepts`: people do Bayesian concept learning over arithmetic rules (organised by kind, with exception rules as their own kind), magnitude ranges and odd/even-within-a-range concepts, using the size principle, and they also entertain multiplicative chains — each number double (or triple) the one before, starting from a number other than 1 (e.g. 5, 10, 20, 40, 80) — as a kind of rule with its own plausibility. The one change, taken from the best model `gamma_width_exception_family_concepts`, is their expectation about how wide a range is: a gamma prior whose sharpness is inferred along with its typical width, in place of the fixed, loose Erlang form. With ranges judged as the best model judges them, chains are put to a fair test: after a chain-like set such as 5, 10, 20 this model still predicts "yes" to 40 and 80 and "no" to other multiples of 5 such as 15, 25 and 30, where the best model spreads "yes" over all multiples of 5.

### gaussian_neighbourhood_magnitude_concepts — pruned (experiment2 end of experiment)

**Outcome:** 211.5 nats behind mixed_strong_weak_sampling_concepts (7.2× dse)

**Hypothesis:** People generalise by rules exactly as the best Bayesian concept learner does, but their magnitude-based concepts are not hard-edged ranges: such a concept is a neighbourhood around a central number, membership falls off smoothly with numerical distance from that centre along a Gaussian similarity gradient whose breadth they infer from the examples (narrow neighbourhoods favoured by the size principle), and examples are produced in proportion to that similarity. A target's graded similarity to the inferred neighbourhood is the probability they say "yes", so "yes" fades gradually with distance from the examples' centre instead of being flat inside the examples' range and stopping at an edge.

### off_by_one_power_concepts — pruned (experiment2 end of experiment)

**Outcome:** 110.6 nats behind mixed_strong_weak_sampling_concepts (5.1× dse)

**Hypothesis:** People do the same Bayesian concept learning as the best model (rules with a family-organised prior and arbitrary-exception rules as their own kind, magnitude ranges with a beta prior over the share of the 1–100 line they cover, and the size principle), but their space of rules also contains near-misses of the sparse multiplicative rules: "one more than" or "one less than" a power or a square (e.g. 3, 5, 9, 17, 33, 65 or 1, 3, 7, 15, 31, 63, or 2, 5, 10, 17, 26, 37), forming their own kind of rule with its own inferred, presumably low, plausibility. So after examples such as 31, 3, 1, 15 or 33, 17, 5, 9 people say "yes" to the next shifted power (63, 65) and "no" to other odd numbers or nearby numbers, rather than spreading "yes" over all odd numbers or a range.

### beta_share_width_prior_leaking_judgment — pruned (experiment2 end of experiment)

**Outcome:** 2.7 nats behind mixed_strong_weak_sampling_concepts; retired to keep the live set at 8 models: ELPD-LOO rank 9 of 15

**Hypothesis:** Refinement of `beta_share_width_exception_concepts` with one component taken from `prior_leaking_membership_judgment`: people infer the hidden concept exactly as the incumbent's Bayesian learner does (family-organised rules with exception rules as their own kind, magnitude ranges judged as a typical share of the 1–100 line, the size principle), but their yes/no judgment is pulled by how plausible the target was before any examples were shown. The log-odds of saying yes is the posterior log-odds that the concept contains the target, shifted by an inferred share of the target's prior log-odds of membership (under the same share-of-the-line range prior), so a priori typical numbers — mid-range numbers that most moderately wide ranges contain, and numbers belonging to many rules — get more "yes" than the examples alone warrant (or less, if the share is negative).

### prior_leaking_beta_share_concepts — rejected (experiment2 round 1 candidate 4 refine incumbent beta_share_width_exception_concepts)

**Outcome:** predicts like existing model 'beta_share_width_prior_leaking_judgment' (p_yes RMSE 0.00034 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of beta_share_width_prior_leaking_judgment, not a new hypothesis.

**Hypothesis:** Refinement of `beta_share_width_exception_concepts`: people infer the hidden concept by the same Bayesian concept learning (family-organised rule prior with exception rules as their own kind, magnitude ranges with a beta prior over the share of the 1–100 line a range covers, the size principle). The one change, taken from `prior_leaking_membership_judgment`, is in how they turn that inference into a yes/no answer: the log-odds of saying yes is the posterior log-odds that the concept contains the target, shifted by an inferred share of how plausible the target was as a member before any example was seen. So a priori typical numbers (mid-range, even, multiples) get more "yes" than the examples alone warrant (or less, if the share is negative), now measured against the bounded-share range prior rather than the unbounded gamma one.

### beta_share_prior_leaking_judgment — rejected (experiment2 round 1 candidate 5 refine chosen)

**Outcome:** predicts like existing model 'beta_share_width_prior_leaking_judgment' (p_yes RMSE 0.00034 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of beta_share_width_prior_leaking_judgment, not a new hypothesis.

**Hypothesis:** Refinement of `prior_leaking_membership_judgment`: people infer the hidden concept by Bayesian concept learning (rules with a family-organised prior, exception rules as their own kind, magnitude ranges, the size principle), and their yes/no judgment is shifted by an inferred share of how plausible the target was before any examples were seen. The one change, taken from `beta_share_width_exception_concepts`, is their expectation about how wide a range is: a beta prior over the share of the bounded 1–100 line a range covers, in place of an unbounded gamma prior over raw width. Near-total ranges are then implausible both when people infer the concept and in the prior plausibility that leaks into the judgment, so the leakage favours mid-range targets differently from the gamma version and widely spread examples point to a rule.

### beta_share_parity_range_concepts — pruned (experiment2 end of experiment)

**Outcome:** 3.0 nats behind mixed_strong_weak_sampling_concepts; retired to keep the live set at 8 models: ELPD-LOO rank 10 of 15

**Hypothesis:** Refinement of `beta_share_width_exception_concepts` with one component taken from `parity_within_range_concepts`: people do the same Bayesian concept learning over rules (family-organised prior, exception rules as their own kind) and magnitude ranges judged as a typical share of the 1–100 line, with the size principle, but they can also entertain the conjunctive concept "odd (or even) numbers within a range" (e.g. "even numbers above 60") as its own kind of concept, with its own inferred plausibility and the same share-of-the-line width prior as plain ranges. So after 4, 66, 78, 96 people say "yes" to large even numbers (70, 74) but "no" to small ones and to odd numbers between the examples, and after 23, 35, 95, 83 they reject even numbers between the examples, rather than spreading "yes" over every number in the range or over all even (odd) numbers.

### beta_share_parity_range_concepts_2 — rejected (experiment2 round 1 candidate 5 refine chosen repair 1)

**Outcome:** predicts like existing model 'beta_share_parity_range_concepts' (p_yes RMSE 0.00000 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of beta_share_parity_range_concepts, not a new hypothesis.

**Hypothesis:** Refinement of `gamma_width_parity_range_concepts`: people do Bayesian concept learning over rules (family-organised prior, exception rules as their own kind), magnitude ranges and "odd (or even) numbers within a range", with the size principle. The one change, taken from `beta_share_width_exception_concepts`, is how they expect a range (plain or parity-restricted) to be wide: as a typical share of the bounded 1–100 line (a beta prior over that share, mean and sharpness inferred) rather than an unbounded gamma expectation over raw width, so near-total ranges are implausible and widely spread same-parity examples (e.g. 4, 66, 78, 96) point to "even numbers in the upper part of the line" rather than to all even numbers or almost any number.

### exemplar_concept_similarity_pooling — pruned (experiment2 end of experiment)

**Outcome:** 810.7 nats behind mixed_strong_weak_sampling_concepts (19.7× dse)

**Hypothesis:** People do not infer one concept that explains all the examples together; they store each example as a separate exemplar and judge the target by its average similarity to the stored exemplars, where the similarity of two numbers is how readily a familiar number concept (a rule such as "multiples of 9", or a magnitude range of typical width) suggested by the exemplar would also contain the target, with more specific shared concepts counting for more. So a target strongly similar to just one example (e.g. a number close to only one of several widely spread examples, or an odd number after 3, 9, 27, 81) still gets a fair share of "yes" answers, and adding consistent examples narrows generalisation much less than a learner who requires one concept to cover them all.

### pedagogical_teacher_sampling_concepts — pruned (experiment2 end of experiment)

**Outcome:** 19.9 nats behind mixed_strong_weak_sampling_concepts (3.2× dse)

**Hypothesis:** People assume the examples were chosen by a helpful teacher rather than drawn at random from the concept: they reason that a program meaning a given concept would pick examples that point distinctively to that concept (numbers that few other plausible concepts would also produce), and they infer the concept, over the same rules and magnitude ranges as the best model, by asking which concept such a teacher would most likely have chosen these examples for. So when every example also fits a narrower rule (e.g. all multiples of 9), people conclude the teacher meant the narrower rule — a broader one (multiples of 3) would have been shown with examples that are not multiples of 9 — and they reject the broader rule's other members more firmly than random sampling alone implies.

### size_weighted_within_kind_rules — rejected (experiment2 round 2 candidate 3 refine incumbent beta_share_width_exception_concepts)

**Outcome:** predicts like existing model 'specificity_biased_rule_prior' (p_yes RMSE 0.00015 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of specificity_biased_rule_prior, not a new hypothesis.

**Hypothesis:** Refinement of `beta_share_width_exception_concepts`: people do the same Bayesian concept learning over rules (family-organised prior, exception rules as their own kind) and magnitude ranges judged as a typical share of the 1–100 line, with the size principle. The one change is how prior belief is shared among the rules of one kind: not equally, but in proportion to a power of how many numbers each rule contains, with the exponent inferred — so if specific rules ("multiples of 9", "powers of 4") are a priori more natural to people than broad ones of the same kind ("multiples of 3", "powers of 2"), then after 36, 45, 54, 81 they say "yes" to other multiples of 9 and "no" to multiples of 3 that are not multiples of 9 more firmly than the size principle alone predicts (or, with the opposite sign, generalise to the broader rule more).

### specificity_tilted_rule_prior — rejected (experiment2 round 2 candidate 4 refine incumbent beta_share_width_exception_concepts)

**Outcome:** predicts like existing model 'specificity_biased_rule_prior' (p_yes RMSE 0.00000 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of specificity_biased_rule_prior, not a new hypothesis.

**Hypothesis:** Refinement of `beta_share_width_exception_concepts`: people do the same Bayesian concept learning over rules (family-organised prior, exception rules as their own kind) and magnitude ranges judged as a typical share of the 1–100 line, with the size principle. The one change is how prior belief is shared among the rules of one kind: instead of treating every rule of a kind as equally likely a priori, people find the more specific rules of a kind (those picking out fewer numbers, e.g. "multiples of 9" over "multiples of 3", "powers of 4" over "powers of 2") more (or, if the inferred tilt is negative, less) plausible, with the strength of this preference inferred. So after 36, 45, 54, 81 people say "yes" to other multiples of 9 and "no" to multiples of 3 that are not multiples of 9 more sharply than the size principle alone predicts.

### multiple_base_specificity_concepts — rejected (experiment2 round 2 candidate 5 refine chosen)

**Outcome:** predicts like existing model 'beta_share_parity_range_concepts' (p_yes RMSE 0.00059 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of beta_share_parity_range_concepts, not a new hypothesis.

**Hypothesis:** Refinement of `beta_share_parity_range_concepts`: people do the same Bayesian concept learning over rules (family-organised prior, exception rules as their own kind), magnitude ranges and odd/even-within-a-range concepts judged as a typical share of the 1–100 line, with the size principle. The one change is that, within the "multiples of m" kind, the rules are no longer equally plausible a priori: people's prior belief in a multiples rule grows (or shrinks) as a power of its base m, with that exponent inferred, so a rule picking out a more specific, larger-base set (multiples of 9, of 12) is favoured over the coarser rule nested in it (multiples of 3, of 4). As a result, after examples such as 36, 45, 54, 81 people say "yes" to other multiples of 9 much more readily than to multiples of 3 that are not multiples of 9 — a stronger preference for the most specific multiplicative rule than the size principle alone gives.

### times_table_familiarity_concepts — pruned (experiment2 end of experiment)

**Outcome:** 12.3 nats behind mixed_strong_weak_sampling_concepts (2.6× dse)

**Hypothesis:** Refinement of `beta_share_width_exception_concepts`: people do the same Bayesian concept learning over rules (family-organised prior, exception rules as their own kind) and magnitude ranges judged as a typical share of the 1–100 line, with the size principle. The one change is how prior belief is shared among the "multiples of m" rules: not equally, but by how familiar each times table is to them, with a separate inferred plausibility for each table that is not tied to how many numbers it contains. So familiar tables (e.g. multiples of 5 or 9) are a readier explanation of examples they fit than unfamiliar (7, 11, 12) or broader (3) tables, and after examples that are all multiples of 9 people confine "yes" to the 9-table more firmly than the size principle alone implies (or, if the 3-table is the more familiar, spread it further).

### dominated_rule_discount_concepts — pruned (experiment2 end of experiment)

**Outcome:** 5.1 nats behind mixed_strong_weak_sampling_concepts (2.8× dse)

**Hypothesis:** Refinement of `beta_share_width_exception_concepts`: people do the same Bayesian concept learning over rules (family-organised prior, exception rules as their own kind) and magnitude ranges judged as a typical share of the 1–100 line, with the size principle. The one change is which rules they seriously entertain given the examples: when every example also fits a strictly narrower rule (all multiples of 9 fit "multiples of 3"; 4, 16, 64 fit both "powers of 4" and "powers of 2"), they settle on the narrower rule and discount the broader, dominated one by an inferred factor beyond what the size principle already does (arbitrary-exception rules never count as the narrower rule). So after 36, 45, 54, 81 people say "yes" to other multiples of 9 and "no" to multiples of 3 that are not multiples of 9 more sharply than the incumbent predicts — unlike a prior favouring specific rules, the discount applies only when a narrower rule actually fits the examples.

### beta_share_without_replacement_concepts — pruned (experiment2 end of experiment)

**Outcome:** 6.4 nats behind mixed_strong_weak_sampling_concepts (2.3× dse)

**Hypothesis:** Refinement of `without_replacement_sampling_concepts` with one component taken from `beta_share_width_exception_concepts`: people assume the program never repeats a number, so they treat the examples as drawn from the hidden concept without replacement (each further example is one of the concept's members not yet shown), and they do Bayesian concept learning over rules (family-organised prior, exception rules as their own kind) and magnitude ranges on that basis. The one change is their prior about how wide a range is: they judge a range as a typical share of the bounded 1–100 line, so near-total "almost any number" ranges are implausible. As a result, when the examples use up a good part of a small rule (e.g. 36, 45, 54, 81 out of the multiples of 9), people favour that most specific rule over the broader one around it (multiples of 3) more strongly than random sampling with replacement allows, saying "yes" to other multiples of 9 and "no" to multiples of 3 that are not multiples of 9.

### hundred_chart_block_concepts — pruned (experiment2 end of experiment)

**Outcome:** 7.7 nats behind mixed_strong_weak_sampling_concepts; retired to keep the live set at 8 models: ELPD-LOO rank 15 of 15

**Hypothesis:** People picture the numbers 1–100 laid out on the familiar hundred chart (tens digit as the row, units digit as the column), and besides arithmetic rules and ranges on the number line they entertain rectangular blocks of that chart as concepts, such as "numbers in the 20s to 80s that end in 2 to 6", with this kind's plausibility inferred and its size judged as a share of the chart. With the size principle, examples that cluster in a few columns across several rows (e.g. 22, 62, 26, 82) lead people to say "yes" to other numbers in that block (42, 66) and "no" to numbers outside its columns (28, 63), instead of spreading "yes" over all even numbers or one wide range.

### range_placement_prior_concepts — pruned (experiment2 end of experiment)

**Outcome:** 3.7 nats behind mixed_strong_weak_sampling_concepts; retired to keep the live set at 8 models: ELPD-LOO rank 12 of 15

**Hypothesis:** Refinement of `beta_share_width_exception_concepts`: people do the same Bayesian concept learning over rules (family-organised prior, exception rules as their own kind) and magnitude ranges, with the size principle and the same expectation that a range covers a typical share of the 1–100 line. The one change is that people also expect ranges to sit in a typical place on the line: instead of every placement of a range of a given width being equally plausible, they hold a prior (inferred mean position and sharpness) over where a range's centre lies, so ranges among, say, small numbers can be a priori more plausible than equally wide ranges among large numbers. As a result, examples falling in a disfavoured region of the line are explained by a rule more readily, and "yes" to nearby numbers there is lower, than for the same pattern of examples in a favoured region.

### generative_next_output_likelihood — pruned (experiment2 end of experiment)

**Outcome:** 279.1 nats behind mixed_strong_weak_sampling_concepts (10.5× dse)

**Hypothesis:** People read "would the program produce this number?" as a question about how likely the program is to output that particular number next, not about whether the number belongs to the hidden concept. They infer the concept as a Bayesian concept learner does (family-organised rules with exception rules as their own kind, magnitude ranges covering a typical share of the 1–100 line, the size principle), but they say "yes" according to how much more likely the target is as the next output than a number picked at random. So targets that only a broad concept contains (any even number, anywhere in a wide range) get fewer "yes" answers than their membership probability implies, while members of a small, well-fitting concept (powers of 2, a tight range) are accepted readily.

### directional_range_momentum — rejected (experiment2 round 4 candidate 1 lens 6)

**Outcome:** predicts like existing model 'mixed_strong_weak_sampling_concepts' (p_yes RMSE 0.00147 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of mixed_strong_weak_sampling_concepts, not a new hypothesis.

**Hypothesis:** People read the examples in the order shown as a path along the number line, and when they think the program produces a range of magnitudes they expect that range to carry on in the direction the examples were moving (from the first example toward the last), extending further ahead of the path than behind it, to an inferred degree. Otherwise they do the same Bayesian concept learning as the best model (rules with a family-organised prior and exception rules as their own kind, ranges judged as a typical share of the 1–100 line, mixed strong/weak sampling). So the same examples shown in ascending order lead to more "yes" for numbers above them, and shown in descending order to more "yes" for numbers below them.

### tversky_prototype_feature_similarity — pruned (experiment2 end of experiment)

**Outcome:** 520.0 nats behind mixed_strong_weak_sampling_concepts (15.5× dse)

**Hypothesis:** People judge the target by its similarity to the prototype of the examples — the set of familiar number properties (parity, being a multiple of some number, being a square, cube or power, being prime, final digit, decade) that all the examples share, plus their place on the number line — using Tversky's ratio-of-features similarity: the salience-weighted properties the target has in common with that prototype, relative to those plus the prototype's properties the target lacks and the target's own properties the prototype lacks, with numerical closeness to the nearest example falling off exponentially and counting as one more weighted property. The more similar the target is to the prototype by this metric, the more likely they say "yes", so a target is rejected both for missing what the examples share and for having salient properties of its own that the examples do not.

### inferred_sampling_regime_concepts — pruned (experiment2 end of experiment)

**Outcome:** 3.2 nats behind mixed_strong_weak_sampling_concepts (2.6× dse)

**Hypothesis:** Refinement of `mixed_strong_weak_sampling_concepts`: people do the same Bayesian concept learning over rules (family-organised prior, exception rules as their own kind) and magnitude ranges judged as a typical share of the 1–100 line, with occasional stray examples, and they allow both random sampling from the concept (the size principle) and weak sampling (examples merely members of it). The one change is where that uncertainty sits: rather than each example independently being a random draw or not, people are unsure whether the program as a whole draws its outputs at random from its concept, and they infer this from the whole set together with the concept. So a suspicious coincidence (every example fits a small rule, e.g. 2, 8, 16, 64) is itself taken as evidence of random sampling and sharpens generalisation onto the small rule, while sets that no small concept explains well tilt belief towards weak sampling and keep broad rules and wide ranges in play.

### set_level_strong_weak_sampling — rejected (experiment2 round 4 candidate 4 refine incumbent mixed_strong_weak_sampling_concepts)

**Outcome:** predicts like existing model 'beta_share_width_exception_concepts' (p_yes RMSE 0.00190 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of beta_share_width_exception_concepts, not a new hypothesis.

**Hypothesis:** Refinement of `mixed_strong_weak_sampling_concepts`: people do the same Bayesian concept learning over rules (family-organised prior, exception rules as their own kind) and magnitude ranges judged as a typical share of the 1–100 line, with occasional stray examples, and they are unsure whether the examples were drawn at random from the concept (strong sampling, the size principle) or merely picked as numbers it contains (weak sampling). The one change is that they hold this belief about the example set as a whole — the program generated all its examples one way or the other — instead of deciding it separately for each example. So the weak-sampling possibility does not fade with every added example: after three or four examples that fit both a narrow and a broad concept (multiples of 9 vs multiples of 3, a tight range vs a wide one) people still keep the broad concept in play and say "yes" to its other members more than the per-example mixture predicts, while after a single example they generalise almost as pure strong sampling does.

### decade_kind_beta_share_concepts — pruned (experiment2 end of experiment)

**Outcome:** 3.7 nats behind mixed_strong_weak_sampling_concepts; retired to keep the live set at 8 models: ELPD-LOO rank 11 of 15

**Hypothesis:** Refinement of `beta_share_width_exception_concepts`, with one component taken from the pruned `decade_concept_family_prior`. People do the same Bayesian concept learning over rules (family-organised prior, exception rules as their own kind) and magnitude ranges judged as a typical share of the 1–100 line, with the size principle. They also hold the decade as a ready-made concept ("single-digit numbers", "numbers in the teens", "numbers in the 60s"), which forms its own kind of rule with its own inferred plausibility. So when the examples share a tens digit (e.g. 63, 67 or 2, 8), people confine "yes" sharply to that decade, accepting 60 and 69 but rejecting 70 and 59. A free-floating range of the same width would give a smooth fall-off around the examples instead.

### arbitrary_opening_example — pruned (experiment2 end of experiment)

**Outcome:** 227.0 nats behind mixed_strong_weak_sampling_concepts (10.1× dse)

**Hypothesis:** People take the first example shown as an arbitrary starting point — just some number the hidden concept contains, picked without regard to how many numbers the concept holds — and treat only the examples that follow as drawn at random from the concept, so only those later examples make them favour small concepts (the size principle). Otherwise they do the same Bayesian concept learning as the best model (rules with a family-organised prior and exception rules as their own kind, ranges judged as a typical share of the 1–100 line, mixed strong/weak sampling for the later examples). So a single example leads to broad generalisation, each further example narrows it sharply, and the same numbers in a different order give different answers depending on which one came first.

### uncertainty_damped_decisiveness — rejected (experiment3 round 0 candidate 4 refine incumbent uncertainty_shifted_criterion)

**Outcome:** predicts like existing model 'beta_share_width_exception_concepts' (p_yes RMSE 0.00113 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of beta_share_width_exception_concepts, not a new hypothesis.

**Hypothesis:** Refinement of `uncertainty_shifted_criterion`: people infer the hidden concept exactly as its Bayesian concept learner does, and their uncertainty about which concept the program uses (how spread their belief is over rules and range widths) again shapes the yes/no answer — but the one change is that uncertainty does not shift the criterion toward "no" or "yes"; it changes how decisively they act on the inferred membership probability. When the examples leave many concepts open, people answer closer to a coin flip in both directions (saying "no" more to likely members and "yes" more to unlikely ones), and when the examples point clearly to one concept they follow it sharply, to an inferred degree.

### mixed_sampling_uncertainty_criterion_2 — rejected (experiment3 round 0 candidate 5 refine chosen)

**Outcome:** predicts like existing model 'mixed_sampling_uncertainty_criterion' (p_yes RMSE 0.00000 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of mixed_sampling_uncertainty_criterion, not a new hypothesis.

**Hypothesis:** Refinement of `mixed_strong_weak_sampling_concepts`, with one component taken from `uncertainty_shifted_criterion`. People infer the hidden concept as the mixed-sampling learner does: rules organised by kind, magnitude ranges judged as a typical share of the 1–100 line, occasional stray examples, and only an inferred share of examples treated as random draws that drive the size principle, so broad rules such as "multiples of 3" stay in play. The one change is the yes/no criterion: the more their belief is spread over competing concepts, the further they shift away from the inferred membership probability, toward "no" (uncertainty aversion) or, if the inferred shift is negative, toward "yes". So two targets with the same membership probability get different answers depending on whether the examples point clearly to one concept or leave many open.

### previous_set_concept_intrusion — rejected (experiment3 round 1 candidate 2 lens 2)

**Outcome:** model binds ['trial_index'] — response-row bookkeeping a bare stimulus never carries — so it cannot be evaluated on a stimulus pool (the novelty gate now, the held-out evaluation later); compute everything from number_set and target.

**Hypothesis:** People infer the hidden concept as the best model's Bayesian concept learner does (and shift their criterion with how unsure they are what kind of program they face), but the example sets of successive trials interfere in memory: on an inferred share of judgments, the concept suggested by the previous trial's examples intrudes and the target is judged by whether that previous concept would contain it rather than the current one (proactive interference). So the same (set, target) pair gets more "yes" when the target also fits the set seen on the trial just before, and more "no" when it does not, which a learner that treats every trial in isolation cannot produce.

### rule_vs_range_uncertainty_criterion — rejected (experiment3 round 1 candidate 4 refine incumbent kind_uncertainty_shifted_criterion)

**Outcome:** predicts like existing model 'rule_vs_range_doubt_criterion' (p_yes RMSE 0.00000 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of rule_vs_range_doubt_criterion, not a new hypothesis.

**Hypothesis:** Refinement of `kind_uncertainty_shifted_criterion`: people infer the hidden concept exactly as its Bayesian concept learner does and shift their yes/no criterion with how unsure they are about the kind of program, but the one change is how coarse that uncertainty is. What moves their criterion is only whether the program follows an arithmetic rule at all or picks out a stretch of the number line — not which kind of rule (parity, primes, powers, multiples, digits) it is. So examples that leave rule-versus-range open (e.g. 12, 16, 20) shift answers toward "no" (or "yes"), while examples that clearly call for some rule but leave several kinds of rule open (e.g. 4, 16, 64: powers, squares, even numbers) do not.

### person_specific_range_breadth — rejected (experiment3 round 1 candidate 5 refine chosen)

**Outcome:** predicts like existing model 'beta_share_width_exception_concepts' (p_yes RMSE 0.00048 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of beta_share_width_exception_concepts, not a new hypothesis.

**Hypothesis:** Refinement of `individual_rule_vs_range_tendency`: people do the same Bayesian concept learning over rules (family-organised prior, exception rules as their own kind) and magnitude ranges judged as a typical share of the 1–100 line, with the size principle, and they differ stably across their session — but the one change is in *what* they differ on: not how readily they think of a rule rather than a range, but how wide they expect a range to be (each person has their own expected share of the line, drawn from a population distribution whose mean and spread are inferred; the sharpness of that expectation is shared). So for the same examples a "broad" person says "yes" to numbers well outside the examples' span while a "narrow" person confines "yes" to the immediate neighbourhood, and this difference shows up only for targets reached by magnitude, not for rule members. (The model also counts participants correctly; the earlier model silently fell back to the population average when there were more than 12.)

### target_primed_rule_retrieval — rejected (experiment3 round 2 candidate 0 lens 3)

**Outcome:** predicts like existing model 'beta_share_width_exception_concepts' (p_yes RMSE 0.00118 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of beta_share_width_exception_concepts, not a new hypothesis.

**Hypothesis:** When people are asked about a number, that number itself brings to mind the arithmetic properties it has (being even, a multiple of 7, a square, ending in 5), so while they judge it, the rules that contain the target get extra consideration beyond what their prior and the examples warrant, to an inferred degree (or less consideration, if the degree is negative). Magnitude ranges are not primed this way, because a bare number does not suggest a stretch of the line. Otherwise they infer the hidden concept as a Bayesian concept learner does, so the priming raises "yes" most for targets that share a still-plausible rule with the examples, and leaves targets reached only by a range unchanged.

### rule_size_misperception_concepts — rejected (experiment3 round 2 candidate 1 lens 4)

**Outcome:** predicts like existing model 'beta_share_width_exception_concepts' (p_yes RMSE 0.00165 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of beta_share_width_exception_concepts, not a new hypothesis.

**Hypothesis:** People do Bayesian concept learning over arithmetic rules (organised by kind, with exception rules as their own kind) and magnitude ranges judged as a typical share of the 1–100 line, using the size principle, but they misjudge how many numbers an arithmetic rule picks out: a rule's felt size grows only as an inferred power of its true number of members (e.g. "even numbers" does not feel fifty times bigger than "powers of 2"), while the size of a stretch of the number line is judged correctly. Because the distortion acts on the evidence each example gives, it grows with the number of examples: with a compressed sense of rule size, examples that fit both a sparse and a broad rule (2, 8, 16; 9, 27, 81) narrow generalisation onto the sparse rule less than a normative learner would, and broad rules compete with ranges more strongly than their true size warrants.

### kind_conflict_shifted_criterion — rejected (experiment3 round 2 candidate 5 refine chosen)

**Outcome:** predicts like existing model 'beta_share_width_exception_concepts' (p_yes RMSE 0.00031 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of beta_share_width_exception_concepts, not a new hypothesis.

**Hypothesis:** Refinement of `uncertainty_shifted_criterion`: people infer the hidden concept exactly as its Bayesian concept learner does (family-organised rules with exception rules as their own kind, ranges judged as a typical share of the 1–100 line, the size principle) and shift their yes/no criterion with uncertainty, but the one change is what that uncertainty is about: not how spread their belief is over concepts for the example set as a whole, but how much the plausible kinds of program disagree about this particular target. When one live kind of explanation (say a multiples rule) says the target belongs and another live kind (say a range) says it does not, people move their answer away from the averaged membership probability toward "no" (conflict aversion) or, if the inferred shift is negative, toward "yes"; targets on which all plausible kinds agree are answered at their membership probability, even for sets that leave the kind of program open.

### base_rate_neglect_concepts — rejected (experiment3 round 2 candidate 1 lens 4 repair 1)

**Outcome:** predicts like existing model 'beta_share_width_exception_concepts' (p_yes RMSE 0.00050 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of beta_share_width_exception_concepts, not a new hypothesis.

**Hypothesis:** People do Bayesian concept learning over arithmetic rules (organised by kind, with exception rules as their own kind) and magnitude ranges judged as a typical share of the 1–100 line, using the size principle, but they neglect base rates: when weighing the concepts against each other they give the prior plausibility of each concept less weight than the examples' evidence (the prior counts only to an inferred fraction of its full weight, as in partial base-rate neglect). So concepts that are a priori unlikely — one rule among many of its kind, an unusually narrow or wide range, a rare kind of rule — are discounted less than an ideal learner would discount them, and once examples fit such a concept, people generalise from it more readily than the normative learner does.

### mixed_sampling_two_cluster_doubt — rejected (experiment3 round 2 candidate 5 refine chosen repair 1)

**Outcome:** predicts like existing model 'two_cluster_kind_doubt_criterion' (p_yes RMSE 0.00183 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of two_cluster_kind_doubt_criterion, not a new hypothesis.

**Hypothesis:** Refinement of `two_cluster_kind_doubt_criterion`, with one component taken from `mixed_strong_weak_sampling_concepts`. People do Bayesian concept learning over arithmetic rules (organised by kind, with exception rules as their own kind), single magnitude ranges and "two separate groups" concepts, all judged as a typical share of the 1–100 line, and they shift their yes/no criterion the more the examples leave open what kind of program it is (a kind of rule, one stretch of the line, or two separate groups). The one change is their sampling assumption: they take only an inferred share of the examples as random draws from the concept (the size principle) and the rest as mere members of it, so broad rules, wide ranges and wide two-group concepts stay in play after examples that also fit a narrower concept. This broadens "yes" to those broad concepts' other members and raises exactly the kind-level doubt that moves the criterion for those sets.

### nearest_example_proximity_pull — rejected (experiment3 round 3 candidate 1 lens 7)

**Outcome:** predicts like existing model 'beta_share_width_exception_concepts' (p_yes RMSE 0.00123 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of beta_share_width_exception_concepts, not a new hypothesis.

**Hypothesis:** People infer the hidden concept as a Bayesian concept learner does (rules organised by kind with exception rules as their own kind, magnitude ranges judged as a typical share of the 1–100 line, the size principle), but their yes/no answer is also pulled by raw numerical similarity: the closer the target lies on the number line to the nearest example, the more readily they say "yes", with this pull falling off exponentially with distance at an inferred rate and having an inferred strength. So even after a clear rule (e.g. 16, 8, 64), numbers right next to an example (15, 17, 63) get more "yes" than the inferred concept warrants, and rule members far from every example get fewer.

### exception_as_variant_doubt_criterion — rejected (experiment3 round 3 candidate 3 refine incumbent rule_family_doubt_criterion)

**Outcome:** predicts like existing model 'rule_family_doubt_criterion' (p_yes RMSE 0.00039 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of rule_family_doubt_criterion, not a new hypothesis.

**Hypothesis:** Refinement of the incumbent `rule_family_doubt_criterion`: people infer the hidden concept exactly as its Bayesian concept learner does (exception rules such as "powers of 2 except 32" keep their own low prior plausibility), and doubt about which kind of arithmetic rule the program follows, weighted by belief that it is a rule, shifts their yes/no criterion toward "no" (or "yes"). The one change is what counts as a different kind of rule when judging that doubt: an exception rule is felt as a variant of the rule it amends, not as another kind, so wavering between "powers of 2" and "powers of 2 except 32" (or "plus 37") does not count as doubt about the kind of program. Powers-of-2 sets such as 16, 32, 2, 8 therefore get almost no criterion shift, where the incumbent treats them as leaving the kind of rule open.

### specificity_tilted_family_doubt — rejected (experiment3 round 3 candidate 4 refine incumbent rule_family_doubt_criterion)

**Outcome:** predicts like existing model 'rule_family_doubt_criterion' (p_yes RMSE 0.00105 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of rule_family_doubt_criterion, not a new hypothesis.

**Hypothesis:** Refinement of `rule_family_doubt_criterion` with one component taken from `specificity_biased_rule_prior`. People infer the hidden concept as the incumbent's Bayesian learner does and shift their yes/no criterion with how unsure they are which kind of arithmetic rule the program follows (weighted by how much they believe it is a rule at all). The one change is that, within each kind of rule, they find specific rules a priori more plausible than broad ones: a rule's prior grows (or shrinks) with how few numbers it contains, to an inferred degree. So after examples that fit a sparse rule nested in a broader rule of the same kind (81, 9; 8, 32), people confine "yes" to the sparse rule more firmly than the size principle alone would, while the family-doubt criterion shift stays as in the incumbent.

### satisficing_search_rule_kind_doubt — rejected (experiment3 round 3 candidate 5 refine chosen)

**Outcome:** predicts like existing model 'rule_family_doubt_criterion' (p_yes RMSE 0.00120 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of rule_family_doubt_criterion, not a new hypothesis.

**Hypothesis:** Refinement of `range_first_satisficing_search`, with one component taken from the incumbent `rule_family_doubt_criterion`. People look first for a stretch of the number line that accounts for the examples and search for an arithmetic rule less thoroughly the better that stretch already explains them, otherwise inferring the concept as a Bayesian concept learner does. The one change is their yes/no criterion: the more doubt they are left with about which kind of arithmetic rule the program follows (weighted by how much they believe it is a rule at all), the further they shift their answer from the inferred membership probability toward "no" (or "yes"), so the satisficing search, by deciding how far rules stay in play, also decides how strongly rule-kind doubt moves the answer.
