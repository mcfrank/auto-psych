# Tried before

37 hypotheses proposed earlier in this project are no longer in the model set. Read them before you propose:

- A *pruned* entry entered the set and later lost to the best model by the stated margin, on the data available then. Its mechanism may be partly right: a model that changes it substantively is welcome, but do not re-propose it unchanged or merely re-parameterised, under any name. Pruned models stay readable under `models/pruned/`.
- A *rejected* entry never entered the set. If it was a near-duplicate of a model still in the set, that region is already covered: do not re-propose it. If it failed on its code or its fit (see its outcome), the idea itself was never tested and a correct implementation may be worth trying.

### recency_weighted_exemplar_similarity — pruned (experiment1 end of experiment)

**Outcome:** 53.0 nats behind coherence_sharpened_decision_exemplar (4.4× dse)

**Hypothesis:** People do not infer a hidden rule; they judge a target by its similarity to the remembered examples, as in an exemplar model. Similarity to each example decays exponentially with the psychological distance between the two numbers, a weighted sum of mismatches on how far apart they are, whether they share a final digit, whether they share parity, and whether one divides the other. Memory for the examples fades with how long ago each was shown, so the last-shown example counts most, and the probability of "yes" rises with the target's recency-weighted summed similarity to the examples.

### recency_forgetting_concept_learning — pruned (experiment1 end of experiment)

**Outcome:** 130.3 nats behind coherence_sharpened_decision_exemplar (5.8× dse)

**Hypothesis:** People are Bayesian concept learners (rules and intervals, size principle) who reason only from the examples they still hold in memory: the last-shown example is always remembered, and each earlier example is forgotten with a probability that grows the further back it was shown, so the concept is inferred from a recency-biased subset of the set. This disagrees most with the order-invariant best model on multi-example sets whose early examples point to a different concept than the last one: for targets close to the last example but outside the concept implied by the whole set it predicts many more "yes" answers, and for targets near only the first example, fewer.

### recency_decayed_concept_learning — pruned (experiment1 end of experiment)

**Outcome:** 178.3 nats behind coherence_sharpened_decision_exemplar (7.0× dse)

**Hypothesis:** People infer the hidden program by Bayesian concept learning over mathematical rules and numeric intervals, but their memory of the examples decays with recency: the most recently shown example counts as full evidence while each earlier example is discounted by a constant factor per position back. Concepts consistent with the later examples therefore dominate, so targets near or sharing structure with the last-shown example get more "yes" answers than those tied to the first-shown one, and evidence from long sets is weaker (broader generalization).

### recency_weighted_concept_learning — pruned (experiment1 end of experiment)

**Outcome:** 284.1 nats behind coherence_sharpened_decision_exemplar (7.7× dse)

**Hypothesis:** Refinement of `bigelow_piantadosi_interval`: people do Bayesian concept learning over the same 42 rules and 1-100 intervals (with a midpoint-dependent interval prior and the size principle), but they weigh the examples by recency — each example's evidence is discounted geometrically by how many examples were shown after it (one inferred decay rate, total evidence held fixed), so concepts consistent with the most recently shown examples gain posterior weight and targets near the last example are endorsed more. The single change is this order-dependent evidence weighting, motivated by the critique's recency discrepancy; a decay of 1 recovers the original model.

### mixed_sampling_concept_learning — pruned (experiment1 end of experiment)

**Outcome:** 138.6 nats behind coherence_sharpened_decision_exemplar (5.7× dse)

**Hypothesis:** Refinement of bigelow_piantadosi_independent: the same Bayesian concept learner over 42 rules and all intervals of 1-100 (rule weights inferred separately, intervals equally likely, 10% noise examples), but people only partly believe the examples were sampled from the concept: each example is taken as drawn from the concept with an inferred probability (strong sampling, the size principle) and otherwise as an arbitrary number that merely happens to belong to it (weak sampling). The one change is this strong/weak sampling mixture, which weakens the size principle's penalty on large concepts, so people generalise more broadly than the incumbent predicts, most of all from a single example.

### tempered_size_principle_concept_learning — pruned (experiment1 end of experiment)

**Outcome:** 117.0 nats behind coherence_sharpened_decision_exemplar (5.5× dse)

**Hypothesis:** A refinement of bigelow_piantadosi_independent: people do Bayesian concept learning over the same 42 rules and all intervals of 1-100, but they apply the size principle only partially — they treat the examples as weaker evidence than a fully strong-sampling learner would, so the evidence from the examples is tempered by one inferred strength (1 recovers the incumbent; below 1 means broader generalization). The single change is this evidence-strength exponent, applied equally to every example regardless of order; it predicts more "yes" answers to targets outside the tightest consistent concept, most of all after a single example, where the incumbent generalises too narrowly.

### coherence_tuned_attention_exemplar — dropped (experiment2)

**Outcome:** MCMC fit failed (RuntimeError: warm-start NUTS: the Hessian at the posterior mode is not positive definite (some parameter direction is pinned down by neither the data nor the prior))

**Hypothesis:** People judge a target by its recency-weighted similarity to the remembered examples, but how much attention they give each kind of similarity (closeness in size, shared final digit, shared parity, shared factor) is learned from the examples themselves: while reading the set they notice the respects in which the examples resemble one another and stretch attention toward those dimensions. So when all examples share a factor, final digit or parity, targets matching every example on that property are endorsed far more than pairwise similarity with fixed attention predicts, while a single example gives no such cue and leaves attention at its baseline.

### log_odds_distorted_concept_learning — pruned (experiment1 end of experiment)

**Outcome:** 93.5 nats behind coherence_sharpened_decision_exemplar (5.3× dse)

**Hypothesis:** People are Bayesian concept learners over mathematical rules (powers, multiples, final and contained digits, primes, parity, squares, cubes) and numeric intervals with the size principle, but they misperceive their own posterior belief that the target belongs to the concept the way people distort probabilities generally: the belief is compressed or expanded linearly in log-odds around an inferred crossover point before it is turned into a yes/no answer. Weak beliefs are therefore over-endorsed and near-certain ones under-endorsed (or the reverse), while the ordering of targets is still set entirely by the normative posterior.

### shared_divisor_rule_abstraction — pruned (experiment1 end of experiment)

**Outcome:** 192.3 nats behind coherence_sharpened_decision_exemplar (8.6× dse)

**Hypothesis:** People abstract a single explicit rule from the examples: "multiples of g", where g is the greatest common divisor the examples share. They endorse targets that are multiples of g and reject those that are not, and they trust this rule more the less likely it is that the examples would share g by coincidence (more examples and a larger g); when the examples share no factor above 1, people fall back on nothing but how close the target is to the examples.

### common_factor_exemplar_similarity — pruned (experiment1 end of experiment)

**Outcome:** 54.8 nats behind coherence_sharpened_decision_exemplar (4.3× dse)

**Hypothesis:** Refinement of recency_weighted_exemplar_similarity: people still judge a target by its recency-weighted exponential similarity to the remembered examples (mismatches on magnitude, final digit, parity and pairwise divisibility), but when several examples share a common factor of 3 or more they also encode that shared factor as a property of the whole set, so a target that is not a multiple of it is psychologically farther from every example. The single change is this one set-level "shared common factor" mismatch dimension in the distance, which predicts more "yes" answers to multiples of the examples' greatest common divisor than pairwise exemplar similarity alone.

### common_factor_exemplar_similarity_2 — rejected (experiment1 round 1 candidate 4 refine incumbent recency_weighted_exemplar_similarity)

**Outcome:** predicts like existing model 'common_factor_exemplar_similarity' (p_yes RMSE 0.00000 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of common_factor_exemplar_similarity, not a new hypothesis.

**Hypothesis:** Refinement of recency_weighted_exemplar_similarity: people judge a target by its recency-weighted similarity to the remembered examples, but when several examples share a common factor of 3 or more they also abstract that shared factor as a property of the whole set, and a target that is not a multiple of it is judged psychologically farther from every example. The single change is this set-level "shared common factor" mismatch added to the exemplar distance (an inferred weight; 0 recovers the incumbent), which predicts more "yes" answers to multiples of the examples' greatest common divisor than pairwise divisibility, digit and parity similarity alone.

### tempered_concept_learning_with_guessing — pruned (experiment1 end of experiment)

**Outcome:** 85.7 nats behind coherence_sharpened_decision_exemplar (4.9× dse)

**Hypothesis:** Refinement of tempered_size_principle_concept_learning: people do the same Bayesian concept learning over 42 rules and all intervals of 1-100 with a partially applied (tempered) size principle, but on some fraction of trials they do not consult the inferred concept at all and simply guess, answering "yes" with their own inferred guessing bias. The single change is this inferred lapse-to-guessing process, which predicts that even targets the concept clearly excludes get a substantial floor of "yes" answers (and clearly included targets a ceiling of "no" answers) that is the same for every set and target, rather than the near-certain answers the pure concept learner gives.

### graded_shared_factor_exemplar_similarity — pruned (experiment1 end of experiment)

**Outcome:** 50.9 nats behind coherence_sharpened_decision_exemplar (4.2× dse)

**Hypothesis:** Refinement of recency_weighted_exemplar_similarity: people still judge a target by its recency-weighted exponential similarity to the remembered examples, but the number-theoretic part of that similarity is graded by how large a factor the target and each example share, not by whether one divides the other. The single change replaces the all-or-none divisibility mismatch with a graded shared-factor mismatch (zero when one number divides the other, one when they share no factor, in between in proportion to the log size of their greatest common divisor relative to the smaller number), so targets that share a big factor with the examples (e.g. 42 after 12, 18, 30) are endorsed more than divisibility alone predicts, which is what the common-factor rule gap in the critique shows.

### running_prototype_delta_rule — pruned (experiment1 end of experiment)

**Outcome:** 58.7 nats behind coherence_sharpened_decision_exemplar (4.4× dse)

**Hypothesis:** People do not store the examples separately; they keep a single running summary of what the program's numbers are like (their size, parity, final digit and which small factors they have) and revise it as each example arrives, moving the summary only part of the way toward each new example. A target is endorsed in proportion to how well it fits that final summary, so example order matters through one learning rate: a slow learner stays anchored on the first example (primacy), a fast one is dominated by the last (recency), and properties that every example shares survive the updating at full strength while properties of only one example are diluted.

### tversky_contrast_feature_similarity — pruned (experiment1 end of experiment)

**Outcome:** 93.7 nats behind coherence_sharpened_decision_exemplar (5.4× dse)

**Hypothesis:** People judge a target by Tversky-style featural similarity to the examples: each number is mentally represented as a set of salient arithmetic features (odd/even, being a multiple of 3 through 10, square, cube, prime, power of two, its final digit and its decade), and a target's similarity to an example rises with the features they have in common and falls with the features each has that the other lacks, weighted asymmetrically. The probability of "yes" rises with the target's average featural similarity to the examples, so targets sharing a property with every example (e.g. a common factor) are endorsed sharply more, while closeness in size matters only through sharing a decade.

### luce_ratio_exemplar_decision — pruned (experiment1 end of experiment)

**Outcome:** 40.6 nats behind coherence_sharpened_decision_exemplar (3.5× dse)

**Hypothesis:** People judge a target by its recency-weighted similarity to the remembered examples (closeness in size, shared final digit, shared parity, divisibility), exactly as in the simple exemplar model, but they turn that familiarity into a yes/no answer by a Luce ratio decision rule: they compare the target's familiarity against a fixed internal criterion familiarity and say "yes" with probability equal to the target's share of (familiarity + criterion), each raised to a response-determinism exponent. Because the rule operates on familiarity ratios rather than differences, targets with very low familiarity are rejected almost always and sets whose examples resemble the targets weakly become near-all-"no", producing more extreme set-level yes rates than a linear-in-familiarity criterion.

### set_factor_tuned_attention_exemplar — pruned (experiment1 end of experiment)

**Outcome:** 10.2 nats behind coherence_sharpened_decision_exemplar; retired to keep the live set at 8 models: ELPD-LOO rank 9 of 9

**Hypothesis:** Refinement of coherence_tuned_attention_exemplar: people still judge a target by its recency-weighted similarity to the examples, with attention stretched toward the dimensions on which the examples resemble one another, but when all the examples share a common factor of 3 or more they notice that specific factor and the factor dimension of similarity becomes "is the target a multiple of the examples' shared factor", rather than whether the target shares any factor with each example separately. The single change is this set-specific redefinition of factor similarity, which predicts a sharper "multiples of g" generalisation (e.g. after 12, 18, 30 the target 42 is endorsed but 15 and 21 are not), closing the common-factor rule gap in the critique.

### set_factor_tuned_attention_exemplar_2 — rejected (experiment1 round 2 candidate 4 refine incumbent coherence_tuned_attention_exemplar)

**Outcome:** predicts like existing model 'set_factor_tuned_attention_exemplar' (p_yes RMSE 0.00000 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of set_factor_tuned_attention_exemplar, not a new hypothesis.

**Hypothesis:** Refinement of coherence_tuned_attention_exemplar: people still judge a target by its recency-weighted similarity to the remembered examples, with attention to each dimension (size, final digit, parity, shared factor) stretched toward the respects in which the examples resemble one another. The single change is what the shared-factor dimension compares: when the examples all share a common divisor of 3 or more, people judge each target-example pair on whether the target is a multiple of that set-wide common factor, rather than on whether the two numbers happen to share any factor pairwise (which they still do for single examples or sets with no common factor). So a target that shares some other factor with an example (e.g. 9 after 12, 24, 36) no longer counts as similar on that dimension, predicting a sharper "multiples of g" gap than the incumbent, as the critique's common-factor rule gap shows.

### graded_factor_coherence_tuned_exemplar — dropped (experiment2)

**Outcome:** MCMC fit failed on CV fold 1 (RuntimeError: warm-start NUTS: the Hessian at the posterior mode is not positive definite (some parameter direction is pinned down by neither the data nor the prior))

**Hypothesis:** Refinement of graded_shared_factor_exemplar_similarity: people still judge a target by its recency-weighted exponential similarity to the remembered examples, with the number-theoretic part of similarity graded by how large a factor the target and each example share, but the attention they give each kind of similarity (size, final digit, parity, shared factor) is stretched toward the respects in which the examples resemble one another. The single change grafts this coherence-tuned attention (from coherence_tuned_attention_exemplar) onto the graded model, with the examples' factor coherence itself measured by the same graded shared-factor match, so sets whose examples share a large common factor make people generalize sharply to targets sharing that factor, closing the common-factor rule gap, while a single example leaves attention at baseline.

### capacity_limited_attention_exemplar — pruned (experiment1 end of experiment)

**Outcome:** 25.9 nats behind coherence_sharpened_decision_exemplar (3.0× dse)

**Hypothesis:** Refinement of coherence_tuned_attention_exemplar: people still judge a target by its recency-weighted similarity to the remembered examples, with attention stretched toward the dimensions (size, final digit, parity, shared factor) on which the examples resemble one another, but attention is a fixed-capacity resource, as in selective-attention exemplar theory. The single change is that coherence reallocates a fixed total amount of attention rather than adding attention: stretching attention toward the property the examples share withdraws it from the other dimensions, chiefly closeness in size. So after a coherent set (e.g. 12, 18, 30) far-away targets that share the property are endorsed more and nearby targets that lack it less, sharpening the "multiples of g" gap and making set-level yes rates more extreme than the incumbent predicts, while a single example leaves attention at its baseline.

### residue_progression_concept_learning — pruned (experiment1 end of experiment)

**Outcome:** 62.2 nats behind coherence_sharpened_decision_exemplar (4.4× dse)

**Hypothesis:** People are Bayesian concept learners (with partially applied size principle and occasional guessing), but the rules they entertain are generated from one simple grammar rather than a hand-picked list: every arithmetic progression "numbers that leave remainder r when divided by g" (which includes even/odd, multiples of any g, and "ends in d" as step 10), alongside powers, squares, cubes, primes, contained digits and numeric intervals. A progression's prior plausibility falls off as a power of its step size, with extra plausibility for pure multiples (remainder 0) and for decimal steps of 10, so sets sharing an unusual common factor (e.g. 14, 28, 42) or a final digit evoke a sharp "multiples of g" or "ends in d" rule that the fixed rule list lacks.

### feature_distribution_prototype — pruned (experiment1 end of experiment)

**Outcome:** 34.6 nats behind coherence_sharpened_decision_exemplar (2.9× dse)

**Hypothesis:** People do not store the examples one by one or infer a rule; they summarise the set as a single statistical category prototype — its central size and spread, and how often each property (final digit, parity, divisibility by 3 to 10) occurs among the examples, smoothed toward how common that property is among numbers in general — and say "yes" in proportion to how typical the target is of that prototype relative to an arbitrary number. Because a property shared by every example becomes more typical the more examples share it, targets with the shared final digit or common factor are endorsed sharply and increasingly so with set size, while the size component broadens with the examples' spread.

### step_lattice_progression_inference — pruned (experiment1 end of experiment)

**Outcome:** 160.8 nats behind coherence_sharpened_decision_exemplar (7.4× dse)

**Hypothesis:** People imagine the hidden program as counting in a fixed step: they infer the largest step size that would carry them from each example to every other (the greatest common divisor of the examples' differences) and reject targets that cannot be reached from the examples in whole steps, so after 13, 23, 43 only numbers ending in 3 qualify and after 12, 18, 30 only multiples of 6. How firmly they hold this stepping rule grows with how unlikely it is that the examples would line up on such a step by chance (more examples beyond the two that define the step, and a larger step); with one or two examples there is no confirmed step and they answer only by how close the target is to the examples.

### dimension_specific_coherence_exemplar — dropped (experiment2)

**Outcome:** MCMC fit failed on CV fold 1 (RuntimeError: warm-start NUTS: the Hessian at the posterior mode is not positive definite (some parameter direction is pinned down by neither the data nor the prior))

**Hypothesis:** Refinement of graded_factor_coherence_tuned_exemplar: people still judge a target by its recency-weighted exponential similarity to the remembered examples (size, final digit, parity, graded shared factor), with attention stretched toward the respects in which the examples resemble one another, but how strongly noticing a shared property stretches attention differs by kind of property rather than being one common sensitivity. The single change gives each dimension its own coherence sensitivity, so a conspicuous surface regularity such as every example ending in the same digit, or every example sharing a large factor, can make people generalize an "ends in d" or "multiples of g" rule far more sharply than shared size or parity does, closing the final-digit and common-factor rule gaps in the critique.

### iter3_candidate4 — rejected (experiment1 round 3 candidate 4 refine incumbent graded_factor_coherence_tuned_exemplar)

**Outcome:** predicts like existing model 'dimension_specific_coherence_exemplar' (p_yes RMSE 0.00000 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of dimension_specific_coherence_exemplar, not a new hypothesis.

**Hypothesis:** Refinement of graded_factor_coherence_tuned_exemplar: people still judge a target by its recency-weighted exponential similarity to the remembered examples (size, final digit, parity, graded shared factor), with attention stretched toward the respects in which the examples resemble one another, but how readily a shared property captures attention differs by kind of property. The single change is that each dimension gets its own coherence sensitivity instead of one shared stretch factor, so a set whose examples all end in the same digit (or share a large factor) can make people generalize an "ends in d" (or "multiples of g") rule far more sharply than coherence in size or parity does, closing the shared-final-digit and common-factor rule gaps in the critique.

### dimension_specific_coherence_attention — dropped (experiment1 round 3)

**Outcome:** MCMC fit failed on CV fold 3 (RuntimeError: warm-start NUTS: the Hessian at the posterior mode is not positive definite (some parameter direction is pinned down by neither the data nor the prior))

**Hypothesis:** Refinement of coherence_tuned_attention_exemplar: people still judge a target by its recency-weighted similarity to the remembered examples, with attention stretched toward the dimensions (size, final digit, parity, shared factor) on which the examples resemble one another, but how readily a shared property captures attention differs by kind of property: a run of examples all ending in the same digit is far more conspicuous than shared parity or closeness in size. The single change is that each dimension has its own coherence sensitivity instead of one common stretch rate, so sets whose examples share a final digit (or a factor) make people generalize sharply to targets matching that property, closing the "ends in d" and "multiples of g" rule gaps, while a single example still leaves attention at its baseline.

### conjunctive_worst_fit_exemplar — pruned (experiment1 end of experiment)

**Outcome:** 57.6 nats behind coherence_sharpened_decision_exemplar (4.5× dse)

**Hypothesis:** People judge a target by how well it fits the examples as a whole, conjunctively: rather than summing its similarity to each remembered example (so that closeness to any one example suffices), they ask whether the target fits every example, so its familiarity is set by its worst-fitting examples (a soft maximum of its psychological distances to the examples on size, final digit, parity and graded shared factor, weighted by recency). This disagrees most sharply with the best (additive, coherence-tuned) exemplar model on multi-example sets for targets that closely resemble one or two examples but not the others — e.g. a near neighbour of one example that lacks the factor or final digit all the examples share — which it predicts are rejected, so endorsements concentrate on targets sharing the set-wide property (the "multiples of g" gap) without any separate attention tuning.

### compressed_number_line_exemplar — dropped (experiment1 round 4)

**Outcome:** MCMC fit failed on CV fold 2 (RuntimeError: warm-start NUTS: the Hessian at the posterior mode is not positive definite (some parameter direction is pinned down by neither the data nor the prior))

**Hypothesis:** Refinement of coherence_sharpened_decision_exemplar: people still judge a target by its recency-weighted similarity to the remembered examples (size, final digit, parity, graded shared factor), with attention stretched toward the respects in which the examples resemble one another and set coherence sharpening the decision, but they judge closeness in size on a compressed, Weber-like mental number line rather than a linear one. The single change is that size differences among small numbers loom larger than equal differences among large numbers (e.g. 4 and 8 feel farther apart than 84 and 88), with the degree of compression inferred, so generalization from small examples is narrow and from large examples broad.

### shared_factor_rule_sharpened_exemplar — dropped (experiment2)

**Outcome:** MCMC fit failed on CV fold 2 (RuntimeError: warm-start NUTS: the Hessian at the posterior mode is not positive definite (some parameter direction is pinned down by neither the data nor the prior))

**Hypothesis:** Refinement of coherence_sharpened_decision_exemplar: people still judge a target by its recency-weighted exponential similarity to the remembered examples (size, final digit, parity, graded shared factor), with attention stretched toward the respects in which the examples resemble one another and set coherence sharpening how decisively that similarity becomes an answer, but when all the examples share a common factor of 3 or more they also notice that factor as an explicit "multiples of g" rule, as in shared_divisor_rule_abstraction, and lean toward "yes" for targets that are multiples of g and toward "no" for those that are not. The single change is this grafted rule-based push on the answer, whose strength grows with how unlikely such a shared factor would be by coincidence (a larger g and more examples), so it acts even on targets far in size from every example — closing the common-factor rule gap in the critique, where people generalize "multiples of g" more sharply than similarity alone predicts.

### tenenbaum_rules_similarity — pruned (experiment1 end of experiment)

**Outcome:** 512.8 nats behind coherence_sharpened_decision_exemplar (7.1× dse)

**Hypothesis:** Bayesian concept learning over 33 mathematical rules (odd, even, squares, cubes, primes, multiples of 3-12, powers of 2-10, ending in 1-9) and every interval of 1-100. The rules share a prior fraction lambda equally; the intervals share the rest by an Erlang density on their size (scale sigma). Examples are sampled from the concept (the size principle), 10% of them uniformly from 1-100, and "yes" is the posterior probability that the concept contains the target.

### bigelow_piantadosi_independent — pruned (experiment1 end of experiment)

**Outcome:** 250.2 nats behind coherence_sharpened_decision_exemplar (8.4× dse)

**Hypothesis:** Bayesian concept learning over 42 rules (powers of 2-10, multiples of 3-12, final digit 1-9, contained digit 1-9, primes, even, odd, squares, cubes) and every interval of 1-100, with a prior weight inferred for each rule separately and a share p_mix for the rules as a group; intervals are equally likely. Examples are sampled from the concept (the size principle), 10% of them uniformly from 1-100, and "yes" is the posterior probability that the concept contains the target.

### bigelow_piantadosi_interval — pruned (experiment1 end of experiment)

**Outcome:** 260.2 nats behind coherence_sharpened_decision_exemplar (7.7× dse)

**Hypothesis:** Bayesian concept learning over 42 rules (powers of 2-10, multiples of 3-12, final digit 1-9, contained digit 1-9, primes, even, odd, squares, cubes) and every interval of 1-100, with a prior weight inferred for each rule separately and a share p_mix for the rules as a group. An interval's prior follows a Beta density over its midpoint, with both shapes inferred, so concepts in some parts of 1-100 are favoured. Examples are sampled from the concept (the size principle), 10% of them uniformly from 1-100, and "yes" is the posterior probability that the concept contains the target.

### dual_route_factor_and_digit_rule_exemplar_2 — rejected (experiment2 round 0 candidate 4 refine incumbent dual_route_shared_factor_rule_exemplar)

**Outcome:** predicts like existing model 'dual_route_factor_and_digit_rule_exemplar' (p_yes RMSE 0.00000 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of dual_route_factor_and_digit_rule_exemplar, not a new hypothesis.

**Hypothesis:** Refinement of dual_route_shared_factor_rule_exemplar: people still judge a target by its recency-weighted exponential similarity to the remembered examples (size, final digit, parity, graded shared factor) with coherence-stretched attention, and still consult an abstracted "multiples of g" rule at the decision when the examples share a factor of 3 or more. The single change is that the rule route also abstracts the explicit rule "ends in d" when two or more examples all end in the same digit, and consults it the same way — raising "yes" for targets ending in d and lowering it for targets that do not, regardless of size, more firmly the more examples share the digit — so the "ends in d" contrast becomes sharper than final-digit similarity alone allows (the shared_last_digit_contrast gap in the critique).

### session_broadening_similarity_gradient — rejected (experiment2 round 1 candidate 0 lens 7)

**Outcome:** model binds ['trial_index'] — response-row bookkeeping a bare stimulus never carries — so it cannot be evaluated on a stimulus pool (the novelty gate now, the held-out evaluation later); compute everything from number_set and target.

**Hypothesis:** People judge a target by its recency-weighted exponential similarity to the remembered examples, where psychological distance is a weighted sum of mismatches in size, final digit, parity and shared factors, but their generalization gradient broadens as the session goes on: with repeated exposure to the task, every mismatch counts for proportionally less, so the same distance yields more similarity late in the session than early. This predicts a drift toward "yes" over the session that is concentrated on targets dissimilar to the examples, while targets that closely match the examples are endorsed about equally early and late.

### drifting_criterion_exemplar — rejected (experiment2 round 1 candidate 1 lens 8)

**Outcome:** model binds ['trial_index'] — response-row bookkeeping a bare stimulus never carries — so it cannot be evaluated on a stimulus pool (the novelty gate now, the held-out evaluation later); compute everything from number_set and target.

**Hypothesis:** People judge a target by its plain recency-weighted similarity to the remembered examples (closeness in size, shared final digit, shared parity, graded shared factor), but the criterion they compare that familiarity against is not fixed: as the session wears on their response criterion relaxes steadily, so the same familiarity earns "yes" more often late in the session than early. The single claim is a decision-level, gradually drifting yes-bias over trials (a criterion shift), with the inference itself left as simple exemplar similarity.

### region_distance_generalization — rejected (experiment2 round 1 candidate 0 lens 7 repair 1)

**Outcome:** MCMC did not converge (max R-hat 1.054 > 1.05; min bulk ESS 93 < 100), even at target_accept 0.99 (the fitter already refits a near miss with smaller NUTS steps, so raising target_accept will not help). Change the model's geometry instead: non-centred parameterisations for hierarchical or scale parameters, tighter (weakly informative) priors on parameters the data barely constrain, fewer weakly identified parameters (drop or merge parameters that trade off against each other), and no hard thresholds or discontinuities in the likelihood.

**Hypothesis:** People do not compare the target with each example separately; they judge its similarity to the region of number space the examples jointly occupy. Distance to that region is the sum of how far the target lies outside the examples' size range (measured relative to how widely the examples are spread, so tightly clustered examples give a steep gradient and widely spread ones a shallow one) plus a mismatch on each property for which the target takes a value none of the examples has (final digit, parity) or lacks the factor all the examples share, and the probability of "yes" falls off exponentially with that distance.

### lapse_guessing_exemplar — rejected (experiment2 round 1 candidate 1 lens 8 repair 1)

**Outcome:** MCMC did not converge (max R-hat 1.050 > 1.05; min bulk ESS 90 < 100), even at target_accept 0.99 (the fitter already refits a near miss with smaller NUTS steps, so raising target_accept will not help). Change the model's geometry instead: non-centred parameterisations for hierarchical or scale parameters, tighter (weakly informative) priors on parameters the data barely constrain, fewer weakly identified parameters (drop or merge parameters that trade off against each other), and no hard thresholds or discontinuities in the likelihood.

**Hypothesis:** People judge a target by its plain recency-weighted similarity to the remembered examples (closeness in size, shared final digit, shared parity, graded shared factor) and answer from that familiarity, but on some fraction of trials they do not consult the examples at all and simply guess, answering "yes" at their own guessing rate. The single claim is a decision-level lapse-to-guessing process on top of simple exemplar similarity, so even targets totally unlike the examples get a fixed floor of "yes" answers and even near-identical targets a fixed ceiling of "no" answers, the same for every set and target.
