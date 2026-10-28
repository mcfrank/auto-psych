# Tried before

69 hypotheses proposed earlier in this project are no longer in the model set. Read them before you propose:

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

### coherence_sharpened_decision_exemplar — pruned (experiment2 end of experiment)

**Outcome:** 67.3 nats behind zigzag_breadth_sequence_exemplar (4.4× dse)

**Hypothesis:** Refinement of graded_factor_coherence_tuned_exemplar: people still judge a target by its recency-weighted exponential similarity to the remembered examples (size, final digit, parity, graded shared factor), with attention stretched toward the respects in which the examples resemble one another, but how much the examples resemble one another also sets how decisively people turn that similarity into an answer. The single change is that set coherence sharpens the decision rule: after a coherent set (examples alike in size, digit, parity or factors) people answer near-deterministically from similarity, endorsing similar targets and rejecting dissimilar ones almost always, whereas after an incoherent set or a single example their answers stay noisy, which makes set-level yes rates more extreme than the incumbent predicts (the between-set yes-rate spread in the critique) and sharpens the "ends in d" and "multiples of g" gaps for coherent sets.

### conjunctive_worst_fit_exemplar — pruned (experiment1 end of experiment)

**Outcome:** 57.6 nats behind coherence_sharpened_decision_exemplar (4.5× dse)

**Hypothesis:** People judge a target by how well it fits the examples as a whole, conjunctively: rather than summing its similarity to each remembered example (so that closeness to any one example suffices), they ask whether the target fits every example, so its familiarity is set by its worst-fitting examples (a soft maximum of its psychological distances to the examples on size, final digit, parity and graded shared factor, weighted by recency). This disagrees most sharply with the best (additive, coherence-tuned) exemplar model on multi-example sets for targets that closely resemble one or two examples but not the others — e.g. a near neighbour of one example that lacks the factor or final digit all the examples share — which it predicts are rejected, so endorsements concentrate on targets sharing the set-wide property (the "multiples of g" gap) without any separate attention tuning.

### compressed_number_line_exemplar — dropped (experiment1 round 4)

**Outcome:** MCMC fit failed on CV fold 2 (RuntimeError: warm-start NUTS: the Hessian at the posterior mode is not positive definite (some parameter direction is pinned down by neither the data nor the prior))

**Hypothesis:** Refinement of coherence_sharpened_decision_exemplar: people still judge a target by its recency-weighted similarity to the remembered examples (size, final digit, parity, graded shared factor), with attention stretched toward the respects in which the examples resemble one another and set coherence sharpening the decision, but they judge closeness in size on a compressed, Weber-like mental number line rather than a linear one. The single change is that size differences among small numbers loom larger than equal differences among large numbers (e.g. 4 and 8 feel farther apart than 84 and 88), with the degree of compression inferred, so generalization from small examples is narrow and from large examples broad.

### independent_retrieval_exemplar_search — pruned (experiment2 end of experiment)

**Outcome:** 76.1 nats behind zigzag_breadth_sequence_exemplar (4.3× dse)

**Hypothesis:** Refinement of graded_factor_coherence_tuned_exemplar: people still compare the target with each remembered example by exponential similarity (size, final digit, parity, graded shared factor), with attention stretched toward the respects in which the examples resemble one another. The single change is the decision process. People do not add up familiarity and threshold it. They search memory example by example: each example is recalled (less often the longer ago it was shown) and independently accepts the target with probability equal to its similarity, and they answer "yes" as soon as any recalled example accepts it (otherwise they guess "yes" at a low base rate). So one highly similar example is enough for "yes", and every additional example is one more chance to accept, so yes rates rise with set size.

### coherence_sharpened_set_factor_exemplar — pruned (experiment2 end of experiment)

**Outcome:** 69.9 nats behind zigzag_breadth_sequence_exemplar (4.4× dse)

**Hypothesis:** Refinement of coherence_sharpened_decision_exemplar: people still judge a target by its recency-weighted exponential similarity to the remembered examples (size, final digit, parity, graded shared factor), with attention stretched toward the respects in which the examples resemble one another and set coherence sharpening how decisively similarity is turned into an answer. The single change (the set-level factor from set_factor_tuned_attention_exemplar) is that when several examples share a common factor of 3 or more, people notice that factor and judge the factor part of similarity by whether the target is a multiple of the examples' shared factor (graded by how much of that factor the target shares), rather than by the factors it shares with each example separately — so after 12, 18, 30 the target 42 is close and 15 or 20 far, sharpening the "multiples of g" gap the critique reports as still too small.

### shared_factor_rule_sharpened_exemplar — dropped (experiment2)

**Outcome:** MCMC fit failed on CV fold 2 (RuntimeError: warm-start NUTS: the Hessian at the posterior mode is not positive definite (some parameter direction is pinned down by neither the data nor the prior))

**Hypothesis:** Refinement of coherence_sharpened_decision_exemplar: people still judge a target by its recency-weighted exponential similarity to the remembered examples (size, final digit, parity, graded shared factor), with attention stretched toward the respects in which the examples resemble one another and set coherence sharpening how decisively that similarity becomes an answer, but when all the examples share a common factor of 3 or more they also notice that factor as an explicit "multiples of g" rule, as in shared_divisor_rule_abstraction, and lean toward "yes" for targets that are multiples of g and toward "no" for those that are not. The single change is this grafted rule-based push on the answer, whose strength grows with how unlikely such a shared factor would be by coincidence (a larger g and more examples), so it acts even on targets far in size from every example — closing the common-factor rule gap in the critique, where people generalize "multiples of g" more sharply than similarity alone predicts.

### dual_route_shared_factor_rule_exemplar — pruned (experiment2 end of experiment)

**Outcome:** 19.7 nats behind zigzag_breadth_sequence_exemplar (2.1× dse)

**Hypothesis:** Refinement of dimension_specific_coherence_exemplar: people still judge a target by its recency-weighted exponential similarity to the remembered examples (size, final digit, parity, graded shared factor), with each dimension's attention stretched by its own sensitivity to how much the examples resemble one another, but when the examples all share a common factor of 3 or more they also abstract the explicit rule "multiples of that factor" and consult it alongside similarity when answering. The single change is this rule as a separate source of evidence at the decision, not a dimension of similarity: it raises "yes" for multiples and lowers it for non-multiples regardless of how far they are in size, in proportion to how unlikely the shared factor would be by coincidence (larger factors and more examples), so far-away multiples of g are endorsed and nearby non-multiples rejected, closing the common-factor rule gap that similarity alone leaves.

### tenenbaum_rules_similarity — pruned (experiment1 end of experiment)

**Outcome:** 512.8 nats behind coherence_sharpened_decision_exemplar (7.1× dse)

**Hypothesis:** Bayesian concept learning over 33 mathematical rules (odd, even, squares, cubes, primes, multiples of 3-12, powers of 2-10, ending in 1-9) and every interval of 1-100. The rules share a prior fraction lambda equally; the intervals share the rest by an Erlang density on their size (scale sigma). Examples are sampled from the concept (the size principle), 10% of them uniformly from 1-100, and "yes" is the posterior probability that the concept contains the target.

### bigelow_piantadosi_independent — pruned (experiment1 end of experiment)

**Outcome:** 250.2 nats behind coherence_sharpened_decision_exemplar (8.4× dse)

**Hypothesis:** Bayesian concept learning over 42 rules (powers of 2-10, multiples of 3-12, final digit 1-9, contained digit 1-9, primes, even, odd, squares, cubes) and every interval of 1-100, with a prior weight inferred for each rule separately and a share p_mix for the rules as a group; intervals are equally likely. Examples are sampled from the concept (the size principle), 10% of them uniformly from 1-100, and "yes" is the posterior probability that the concept contains the target.

### bigelow_piantadosi_interval — pruned (experiment1 end of experiment)

**Outcome:** 260.2 nats behind coherence_sharpened_decision_exemplar (7.7× dse)

**Hypothesis:** Bayesian concept learning over 42 rules (powers of 2-10, multiples of 3-12, final digit 1-9, contained digit 1-9, primes, even, odd, squares, cubes) and every interval of 1-100, with a prior weight inferred for each rule separately and a share p_mix for the rules as a group. An interval's prior follows a Beta density over its midpoint, with both shapes inferred, so concepts in some parts of 1-100 are favoured. Examples are sampled from the concept (the size principle), 10% of them uniformly from 1-100, and "yes" is the posterior probability that the concept contains the target.

### factorized_rule_and_range_inference — pruned (experiment2 end of experiment)

**Outcome:** 330.7 nats behind zigzag_breadth_sequence_exemplar (10.4× dse)

**Hypothesis:** People are Bayesian concept learners who think the program's numbers obey both an arithmetic rule (a step pattern such as "even", "multiples of 3" or "ends in 7", or no pattern at all) and a numeric range, each inferred from the examples with the size principle, but — as a resource-saving distortion of joint inference — they work out the rule and the range as two separate, independent questions and say "yes" only to the extent that the target passes both. So a target is endorsed when it fits the inferred step pattern AND lies within the inferred range: far-away numbers sharing the examples' final digit or factor are rejected, as are nearby numbers that break the pattern, and the yes-probability is the product of the two separate judgements.

### shared_property_coincidence_rule — pruned (experiment2 end of experiment)

**Outcome:** 208.0 nats behind zigzag_breadth_sequence_exemplar (8.2× dse)

**Hypothesis:** People read the examples for the arithmetic properties they all have in common (same parity, same final digit, same decade, being multiples of the same number) and treat each shared property as evidence for a rule in proportion to how surprising it would be for that many examples to share it by coincidence. They then say "yes" to a target according to the surprise-weighted balance of shared properties it has versus lacks, discounted only by how far the target falls outside the range the examples span — no similarity to individual examples is computed.

### memoryless_win_stay_lose_shift_rules — pruned (experiment2 end of experiment)

**Outcome:** 255.0 nats behind zigzag_breadth_sequence_exemplar (9.0× dse)

**Hypothesis:** People test one candidate rule at a time while reading the examples in order, in a win-stay, lose-shift fashion: the first example suggests a single rule (a final digit, a multiple of some number, odd or even, or a range of sizes, smaller rules being favoured), they keep that rule as long as each new example fits it, and when an example contradicts it they discard it and pick a fresh rule suggested by that contradicting example alone, without re-checking the earlier ones. They then answer "yes" if the target obeys the rule they ended up holding, so the final judgement is driven by the last examples that forced a switch and by whichever rule survived from the start, and the same examples shown in a different order lead to different generalisations.

### dual_route_factor_and_digit_rule_exemplar — pruned (experiment2 end of experiment)

**Outcome:** 15.6 nats behind zigzag_breadth_sequence_exemplar (2.2× dse)

**Hypothesis:** Refinement of dual_route_shared_factor_rule_exemplar: people still judge a target by its recency-weighted exponential similarity to the remembered examples (size, final digit, parity, graded shared factor, each dimension's attention stretched by how much the examples resemble one another), and still consult an abstracted explicit rule alongside similarity at the decision, but the rule route is not limited to shared factors. The single change is that when all the examples end in the same digit, people also abstract the explicit rule "numbers ending in that digit" and consult it as separate evidence, raising "yes" for targets ending in that digit and lowering it for the rest regardless of size distance, in proportion to how unlikely that shared ending would be by coincidence (more examples), closing the "ends in d" gap the critique reports (people apply the final-digit rule more sharply than final-digit similarity predicts).

### dual_route_factor_and_digit_rule_exemplar_2 — rejected (experiment2 round 0 candidate 4 refine incumbent dual_route_shared_factor_rule_exemplar)

**Outcome:** predicts like existing model 'dual_route_factor_and_digit_rule_exemplar' (p_yes RMSE 0.00000 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of dual_route_factor_and_digit_rule_exemplar, not a new hypothesis.

**Hypothesis:** Refinement of dual_route_shared_factor_rule_exemplar: people still judge a target by its recency-weighted exponential similarity to the remembered examples (size, final digit, parity, graded shared factor) with coherence-stretched attention, and still consult an abstracted "multiples of g" rule at the decision when the examples share a factor of 3 or more. The single change is that the rule route also abstracts the explicit rule "ends in d" when two or more examples all end in the same digit, and consults it the same way — raising "yes" for targets ending in d and lowering it for targets that do not, regardless of size, more firmly the more examples share the digit — so the "ends in d" contrast becomes sharper than final-digit similarity alone allows (the shared_last_digit_contrast gap in the critique).

### set_rule_route_sharpened_exemplar — pruned (experiment2 end of experiment)

**Outcome:** 27.3 nats behind zigzag_breadth_sequence_exemplar (3.1× dse)

**Hypothesis:** Refinement of coherence_sharpened_decision_exemplar: people still judge a target by its recency-weighted exponential similarity to the remembered examples (size, final digit, parity, graded shared factor), with attention stretched toward the respects in which the examples resemble one another and set coherence sharpening how decisively similarity is turned into an answer. The single change is an abstracted explicit-rule route at the decision: when all the examples share a final digit or a common factor of 3 or more, people extract the narrowest such rule ("ends in d" or "multiples of g") and consult it alongside similarity, raising "yes" for targets that obey it and lowering it for those that do not regardless of size, in proportion to how unlikely the examples would be to obey that rule by coincidence (more examples, and a rule that covers fewer of the numbers 1-100). Unlike the incumbent's factor-only rule, this treats a shared final digit as a rule too, closing the "ends in d" gap the critique reports (people apply it more sharply than similarity predicts).

### coherence_decisive_dual_route_exemplar — pruned (experiment2 end of experiment)

**Outcome:** 18.9 nats behind zigzag_breadth_sequence_exemplar (2.3× dse)

**Hypothesis:** Refinement of dual_route_shared_factor_rule_exemplar: people still judge a target by its recency-weighted exponential similarity to the remembered examples (size, final digit, parity, graded shared factor, each dimension's attention stretched by how much the examples resemble one another) and still consult the abstracted "multiples of g" rule as separate evidence at the decision when the examples share a factor of 3 or more. The single change, taken from coherence_sharpened_decision_exemplar, is that how much the examples resemble one another also sets how decisively people answer: after a coherent set the combined evidence (similarity plus rule, relative to their criterion) is turned into a near-deterministic yes or no, whereas after an incoherent set or a single example answers stay noisy — so responses to the same item are more consistent across people than the incumbent predicts (the cell_response_dispersion gap) and the "ends in d" contrast in coherent same-digit sets is sharper.

### session_broadening_similarity_gradient — rejected (experiment2 round 1 candidate 0 lens 7)

**Outcome:** model binds ['trial_index'] — response-row bookkeeping a bare stimulus never carries — so it cannot be evaluated on a stimulus pool (the novelty gate now, the held-out evaluation later); compute everything from number_set and target.

**Hypothesis:** People judge a target by its recency-weighted exponential similarity to the remembered examples, where psychological distance is a weighted sum of mismatches in size, final digit, parity and shared factors, but their generalization gradient broadens as the session goes on: with repeated exposure to the task, every mismatch counts for proportionally less, so the same distance yields more similarity late in the session than early. This predicts a drift toward "yes" over the session that is concentrated on targets dissimilar to the examples, while targets that closely match the examples are endorsed about equally early and late.

### drifting_criterion_exemplar — rejected (experiment2 round 1 candidate 1 lens 8)

**Outcome:** model binds ['trial_index'] — response-row bookkeeping a bare stimulus never carries — so it cannot be evaluated on a stimulus pool (the novelty gate now, the held-out evaluation later); compute everything from number_set and target.

**Hypothesis:** People judge a target by its plain recency-weighted similarity to the remembered examples (closeness in size, shared final digit, shared parity, graded shared factor), but the criterion they compare that familiarity against is not fixed: as the session wears on their response criterion relaxes steadily, so the same familiarity earns "yes" more often late in the session than early. The single claim is a decision-level, gradually drifting yes-bias over trials (a criterion shift), with the inference itself left as simple exemplar similarity.

### step_progression_rule_exemplar — pruned (experiment2 end of experiment)

**Outcome:** 55.9 nats behind zigzag_breadth_sequence_exemplar (4.4× dse)

**Hypothesis:** Refinement of dual_route_factor_and_digit_rule_exemplar: people still judge a target by its recency-weighted exponential similarity to the remembered examples (size, final digit, parity, graded shared factor, attention stretched by how much the examples resemble one another) and still consult an abstracted explicit rule alongside similarity at the decision, but the rules they can abstract are not just "multiples of g" and "ends in d". The single change is the rule space: people entertain every counting-by-steps rule the examples fit ("start from an example and step by s", i.e. numbers leaving the same remainder as the examples when divided by s, for any step s that carries every example to every other), each favoured a priori the smaller its step, with steps of 10 extra natural because of base-10 notation, and weigh them by how unlikely it is that the examples would line up on that step by chance. So after 5, 9 people endorse 13 and 17 and after 83, 77 they endorse 71 and 89, regardless of size, which neither the factor nor the final-digit rule predicts, while "ends in d" (step 10) and "multiples of g" (remainder 0) remain special cases.

### drifting_criterion_dual_route_exemplar — pruned (experiment2 end of experiment)

**Outcome:** 11.7 nats behind zigzag_breadth_sequence_exemplar; retired to keep the live set at 8 models: ELPD-LOO rank 13 of 14

**Hypothesis:** Refinement of dual_route_factor_and_digit_rule_exemplar: people still judge a target by its recency-weighted exponential similarity to the remembered examples (size, final digit, parity, graded shared factor, each dimension's attention stretched by how much the examples resemble one another), and they still weigh the abstracted "multiples of g" and "ends in d" rules as separate evidence when they decide. The single change is that the criterion for saying "yes" relaxes steadily over the session: as people answer more and more questions without feedback, they need less evidence before they endorse a target. The same evidence therefore earns more "yes" answers late in the session than early on. This closes the late-minus-early yes-rate drift the critique reports, which a stationary model cannot produce.

### iter1_candidate4 — pruned (experiment2 end of experiment)

**Outcome:** 15.7 nats behind zigzag_breadth_sequence_exemplar; retired to keep the live set at 8 models: ELPD-LOO rank 14 of 14

**Hypothesis:** Refinement of dual_route_factor_and_digit_rule_exemplar: people still judge a target by its recency-weighted exponential similarity to the remembered examples (size, final digit, parity, graded shared factor, attention stretched by how much the examples resemble one another) and still consult the abstracted "multiples of g" and "ends in d" rules as separate evidence at the decision. The single change is progressive disengagement over the session: on a growing share of trials as the session wears on, people stop consulting the examples at all and simply guess, with their own guessing bias toward "yes", so answers late in the session are drawn toward that guessing rate for every kind of target alike — closing the critique's session-drift gap (yes rates rise from the first to the last third of the session, by similar amounts for near and far targets) and putting a floor and ceiling on how extreme any item's yes rate can be.

### coherence_decisive_factor_digit_rules — pruned (experiment2 end of experiment)

**Outcome:** 17.0 nats behind zigzag_breadth_sequence_exemplar (2.3× dse)

**Hypothesis:** Refinement of coherence_decisive_dual_route_exemplar: people still judge a target by its recency-weighted exponential similarity to the remembered examples (size, final digit, parity, graded shared factor, each dimension's attention stretched by how much the examples resemble one another), still consult the abstracted "multiples of g" rule as separate evidence at the decision, and still let set coherence set how decisively the combined evidence is turned into a yes or no. The single change, taken from dual_route_factor_and_digit_rule_exemplar, is that when all the examples end in the same digit people also abstract the explicit rule "numbers ending in that digit" and consult it as separate evidence (in proportion to how unlikely the shared ending would be by coincidence) — and because such sets are coherent, the digit rule is applied near-deterministically, predicting a sharper "ends in d" contrast than either parent model alone.

### region_distance_generalization — rejected (experiment2 round 1 candidate 0 lens 7 repair 1)

**Outcome:** MCMC did not converge (max R-hat 1.054 > 1.05; min bulk ESS 93 < 100), even at target_accept 0.99 (the fitter already refits a near miss with smaller NUTS steps, so raising target_accept will not help). Change the model's geometry instead: non-centred parameterisations for hierarchical or scale parameters, tighter (weakly informative) priors on parameters the data barely constrain, fewer weakly identified parameters (drop or merge parameters that trade off against each other), and no hard thresholds or discontinuities in the likelihood.

**Hypothesis:** People do not compare the target with each example separately; they judge its similarity to the region of number space the examples jointly occupy. Distance to that region is the sum of how far the target lies outside the examples' size range (measured relative to how widely the examples are spread, so tightly clustered examples give a steep gradient and widely spread ones a shallow one) plus a mismatch on each property for which the target takes a value none of the examples has (final digit, parity) or lacks the factor all the examples share, and the probability of "yes" falls off exponentially with that distance.

### lapse_guessing_exemplar — rejected (experiment2 round 1 candidate 1 lens 8 repair 1)

**Outcome:** MCMC did not converge (max R-hat 1.050 > 1.05; min bulk ESS 90 < 100), even at target_accept 0.99 (the fitter already refits a near miss with smaller NUTS steps, so raising target_accept will not help). Change the model's geometry instead: non-centred parameterisations for hierarchical or scale parameters, tighter (weakly informative) priors on parameters the data barely constrain, fewer weakly identified parameters (drop or merge parameters that trade off against each other), and no hard thresholds or discontinuities in the likelihood.

**Hypothesis:** People judge a target by its plain recency-weighted similarity to the remembered examples (closeness in size, shared final digit, shared parity, graded shared factor) and answer from that familiarity, but on some fraction of trials they do not consult the examples at all and simply guess, answering "yes" at their own guessing rate. The single claim is a decision-level lapse-to-guessing process on top of simple exemplar similarity, so even targets totally unlike the examples get a fixed floor of "yes" answers and even near-identical targets a fixed ceiling of "no" answers, the same for every set and target.

### upward_skewed_exemplar_gradient — pruned (experiment2 end of experiment)

**Outcome:** 20.7 nats behind zigzag_breadth_sequence_exemplar (2.7× dse)

**Hypothesis:** Refinement of drifting_criterion_dual_route_exemplar: people still judge a target by its recency-weighted exponential similarity to the remembered examples (size, final digit, parity, graded shared factor, attention stretched by how much the examples resemble one another), still weigh the abstracted "multiples of g" and "ends in d" rules as separate evidence, and still relax their yes-criterion over the session. The single change is that the size gradient around each stored example is directional: a number larger than a remembered example feels closer to it than a number smaller by the same amount (as if the program is expected to count upward), so people generalize further upward than downward from the examples, closing the upward-vs-downward generalization gap the critique reports.

### log_scale_magnitude_range_inference — pruned (experiment2 end of experiment)

**Outcome:** 374.7 nats behind zigzag_breadth_sequence_exemplar (10.7× dse)

**Hypothesis:** People do not look for arithmetic rules or compare the target with each example; they infer the range of magnitudes the program produces, reasoning about size on a compressed, logarithmic mental number line. They favour the narrowest range (in ratio terms) that covers all the examples, more strongly the more examples there are, and say "yes" in proportion to their belief that the target falls inside that range — so generalization extends further above the largest example than below the smallest (equal ratios span more numbers upward), and tightly clustered sets are generalized narrowly while widely spread sets are generalized broadly.

### directional_upward_size_exemplar — rejected (experiment2 round 2 candidate 2 lens 1)

**Outcome:** predicts like existing model 'upward_skewed_exemplar_gradient' (p_yes RMSE 0.00000 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of upward_skewed_exemplar_gradient, not a new hypothesis.

**Hypothesis:** Refinement of drifting_criterion_dual_route_exemplar: people still judge a target by its recency-weighted exponential similarity to the remembered examples (size, final digit, parity, graded shared factor, coherence-stretched attention), still consult the "multiples of g" and "ends in d" rules as separate evidence, and still relax their criterion over the session. The single change is that size similarity is directional: people imagine the program as producing numbers that continue upward from what it has shown, so a target a given distance above an example feels closer to it than one the same distance below. The disagreement with the best model lives just outside the examples' range: it predicts more "yes" for targets 1-20 above the largest example than for targets 1-20 below the smallest (the upward-vs-downward generalization gap the critique reports), where the best model's symmetric size distance predicts equal rates.

### directional_size_drifting_dual_route — rejected (experiment2 round 2 candidate 3 refine incumbent drifting_criterion_dual_route_exemplar)

**Outcome:** predicts like existing model 'upward_skewed_exemplar_gradient' (p_yes RMSE 0.00000 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of upward_skewed_exemplar_gradient, not a new hypothesis.

**Hypothesis:** Refinement of drifting_criterion_dual_route_exemplar: people still judge a target by its recency-weighted exponential similarity to the remembered examples (size, final digit, parity, graded shared factor, attention stretched by how much the examples resemble one another), still weigh the abstracted "multiples of g" and "ends in d" rules as separate evidence, and still relax their yes-criterion over the session. The single change is that size similarity is directionally asymmetric: people treat the program as more likely to go on to larger numbers than back to smaller ones, so a target a given distance above an example is judged closer to it than one the same distance below. This predicts more "yes" answers to targets just above the largest example than just below the smallest, closing the critique's upward-vs-downward generalization gap that symmetric size distance cannot produce.

### upward_biased_size_dual_route_exemplar — rejected (experiment2 round 2 candidate 4 refine incumbent drifting_criterion_dual_route_exemplar)

**Outcome:** predicts like existing model 'upward_skewed_exemplar_gradient' (p_yes RMSE 0.00000 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of upward_skewed_exemplar_gradient, not a new hypothesis.

**Hypothesis:** Refinement of drifting_criterion_dual_route_exemplar: people still judge a target by its recency-weighted exponential similarity to the remembered examples (size, final digit, parity, graded shared factor, attention stretched by how much the examples resemble one another), still weigh the abstracted "multiples of g" and "ends in d" rules as separate evidence, and still relax their yes-criterion over the session. The single change is that psychological distance in size is asymmetric: people treat the program's numbers as more likely to continue upward, so a target a given amount larger than an example feels closer to it than a target the same amount smaller. They therefore generalize further to numbers above the examples than below them, which closes the upward-vs-downward generalization gap the critique reports and which a symmetric size distance cannot produce.

### upward_biased_size_disengaging_exemplar — pruned (experiment2 end of experiment)

**Outcome:** 27.1 nats behind zigzag_breadth_sequence_exemplar (2.5× dse)

**Hypothesis:** Refinement of iter1_candidate4 (dual-route exemplar with progressive disengagement): people still judge a target by its recency-weighted exponential similarity to the remembered examples (size, final digit, parity, graded shared factor, attention stretched by how much the examples resemble one another), still consult the abstracted "multiples of g" and "ends in d" rules as separate evidence, and still disengage and guess on a growing share of trials over the session. The single change is that size similarity is directional: people treat a number larger than a remembered example as psychologically closer to it than an equally distant smaller number (as if the program is expected to keep counting upward), so they generalise more to targets above the examples than below them — closing the critique's upward-versus-downward generalisation gap, which symmetric size distance cannot produce.

### set_size_scaled_magnitude_gradient — pruned (experiment2 end of experiment)

**Outcome:** 5.7 nats behind zigzag_breadth_sequence_exemplar; retired to keep the live set at 8 models: ELPD-LOO rank 9 of 14

**Hypothesis:** Refinement of drifting_criterion_dual_route_exemplar: people still judge a target by its recency-weighted exponential similarity to the remembered examples (size, final digit, parity, graded shared factor, attention stretched by how much the examples resemble one another), still weigh the abstracted "multiples of g" and "ends in d" rules as separate evidence, and still relax their yes-criterion over the session. The single change is that how broadly people generalize in size around each example depends on how many examples the program has shown: they take a larger sample as evidence that the program ranges more widely, so the size gradient around each example widens with the number of examples (a single example is generalized narrowly in size, three or four examples broadly), the reverse of a size-principle narrowing. This changes how generously whole sets are generalized relative to one another, addressing the critique's between-set yes-rate spread.

### log_compressed_size_dual_route_exemplar — pruned (experiment2 end of experiment)

**Outcome:** 24.4 nats behind zigzag_breadth_sequence_exemplar (2.3× dse)

**Hypothesis:** Refinement of drifting_criterion_dual_route_exemplar: people still judge a target by its recency-weighted exponential similarity to the remembered examples (size, final digit, parity, graded shared factor, attention stretched by how much the examples resemble one another), still weigh the abstracted "multiples of g" and "ends in d" rules as separate evidence, and still relax their yes-criterion over the session. The single change is that closeness in size is judged on a compressed, logarithmic mental number line (by ratio, not by difference), both when comparing the target with each example and when judging how alike the examples are in size. So people generalize narrowly around small examples and broadly around large ones, and further above an example than below it, which predicts the upward-vs-downward generalization gap and the larger set-to-set differences in breadth that the critique reports.

### recency_anchored_span_exemplar — dropped (experiment3)

**Outcome:** MCMC fit failed on CV fold 1 (RuntimeError: warm-start NUTS: the Hessian at the posterior mode is not positive definite (some parameter direction is pinned down by neither the data nor the prior))

**Hypothesis:** Refinement of span_filling_size_exemplar (all other parts unchanged: recency-weighted exemplar similarity on final digit, parity and graded shared factor with coherence-stretched attention, the "multiples of g" and "ends in d" rules, and a criterion that relaxes over the session). The single change is a memory process in how the filled-in size span is held: people anchor the span on the most recently shown example, and the positions of earlier examples fade in memory, so the stretch each earlier example adds to the span shrinks back toward the last example the further back it was shown. Targets lying on the last example's side of the set are therefore treated as inside the span (and endorsed) more often than equally placed targets on the first example's side, the recency bias in size generalization that the critique reports.

### memory_limited_span_exemplar — pruned (experiment2 end of experiment)

**Outcome:** 5.8 nats behind zigzag_breadth_sequence_exemplar; retired to keep the live set at 8 models: ELPD-LOO rank 10 of 14

**Hypothesis:** Refinement of span_filling_size_exemplar: people still infer that the program fills in the stretch of the number line the examples mark out (size mismatch = how far the target falls outside that stretch), still judge final digit, parity and shared factor by recency-weighted similarity to the examples, consult the "multiples of g" and "ends in d" rules, and relax their criterion over the session. The single change is a memory limit on that span inference: the stretch is marked out only by the examples people still remember, and the same fading memory that down-weights older examples makes them more likely to be forgotten altogether, so the inferred stretch often shrinks toward the most recently shown example. Targets near the last example are therefore endorsed more than equally placed targets near the first, which the full-span model cannot produce (the critique's last-versus-first recency gap).

### partially_filled_span_exemplar — pruned (experiment2 end of experiment)

**Outcome:** 6.2 nats behind zigzag_breadth_sequence_exemplar; retired to keep the live set at 8 models: ELPD-LOO rank 11 of 14

**Hypothesis:** Refinement of span_filling_size_exemplar: people still judge a target by its recency-weighted exponential similarity to the remembered examples (final digit, parity, graded shared factor, coherence-stretched attention), still consult the "multiples of g" and "ends in d" rules as separate evidence, still relax their criterion over the session, and still treat the span between the smallest and largest example as filled in. The single change is that the span is only partly filled in: a target's size mismatch to each remembered example blends how far it falls outside the span with its own distance from that example, so inside a wide span targets near an example still feel closer to it — and because the most recently shown example is remembered best, size generalization favours targets near the last example over targets near the first (the critique's recency_last_vs_first_example gap, which a purely span-based mismatch cannot produce).

### partial_span_filling_exemplar — pruned (experiment2 end of experiment)

**Outcome:** 6.7 nats behind zigzag_breadth_sequence_exemplar; retired to keep the live set at 8 models: ELPD-LOO rank 12 of 14

**Hypothesis:** Refinement of span_filling_size_exemplar: people still judge a target by its recency-weighted exponential similarity to the remembered examples (final digit, parity, graded shared factor, coherence-stretched attention), still consult the "multiples of g" and "ends in d" rules as separate evidence, and still relax their criterion over the session. The single change is that the span between the examples is only partly filled in: a target's size mismatch with each example is a blend of how far it falls outside the span of the examples and its own distance from that example, with the filling-in share inferred. Because the per-example part of size closeness is weighted by memory, targets inside the span that lie nearer the last-shown example are endorsed more than those nearer the first — the recency-in-size-generalization the critique reports, which a pure span mismatch (identical for every example) cannot produce.

### spread_scaled_size_exemplar — pruned (experiment2 end of experiment)

**Outcome:** 154.5 nats behind zigzag_breadth_sequence_exemplar (6.4× dse)

**Hypothesis:** People estimate how variable the program's output is from how spread out the examples are, and generalize in size accordingly: they judge a target by its recency-weighted similarity to the remembered examples (size, final digit, parity, shared factor), but the size gradient around every example is stretched in proportion to the spread of the examples. A tightly clustered set (66, 78) is generalized narrowly in size and a widely spread set (10, 50, 100) very broadly, both inside and beyond its range, so whole sets differ sharply in how generously they are generalized.

### stride_scaled_sequence_exemplar — rejected (experiment2 round 4 candidate 1 lens 6)

**Outcome:** predicts like existing model 'sequence_momentum_span_exemplar' (p_yes RMSE 0.00107 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of sequence_momentum_span_exemplar, not a new hypothesis.

**Hypothesis:** Refinement of sequence_momentum_span_exemplar (all else unchanged: recency-weighted similarity on size, final digit, parity and graded shared factor with coherence-stretched attention, the "multiples of g" and "ends in d" rules, a relaxing criterion, the filled-in span and the momentum of the sequence's direction of travel). The single change is that people read the examples in the order shown as a sequence of jumps and take the typical jump between successive examples as the program's stride: the bigger the steps it has been taking from one example to the next, the further beyond the examples they expect its next number could land, so their size gradient is broad after big successive jumps and narrow after small ones. Because the stride depends on order (10, 20, 90 has smaller steps than 10, 90, 20), sets are generalized more differently from one another than a span-only model predicts, addressing the critique's too-compressed between-set yes-rate spread.

### spread_scaled_momentum_span_exemplar — rejected (experiment2 round 4 candidate 3 refine incumbent sequence_momentum_span_exemplar)

**Outcome:** predicts like existing model 'spread_calibrated_size_gradient' (p_yes RMSE 0.00113 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of spread_calibrated_size_gradient, not a new hypothesis.

**Hypothesis:** Refinement of sequence_momentum_span_exemplar (all its parts kept: recency-weighted exemplar similarity on size, final digit, parity and graded shared factor with coherence-stretched attention, the filled-in span with its order-directed momentum, the "multiples of g" and "ends in d" rules, and a criterion that relaxes over the session). The single change is that people read how widely the examples are spread as evidence of how widely the program ranges: the size gradient beyond the filled-in span broadens in proportion to the span's width, so after 10, 90 a target 15 units outside the span is still endorsed readily, whereas after 40, 42 (or a single example) the same distance is rejected. Widely spread sets are therefore generalized much more generously than tight ones, the between-set yes-rate spread the critique reports as too compressed.

### set_specific_criterion_momentum_exemplar — rejected (experiment2 round 4 candidate 4 refine incumbent sequence_momentum_span_exemplar)

**Outcome:** p_yes could not be predicted on the 512-stimulus novelty pool (ValueError: more distinct example sets than MAX_SETS). The model must define p_yes (a pm.Deterministic named 'p_yes') with one value per stimulus row, and its compute_features or prepare_observed must work for every stimulus the design can show (e.g. {1} -> 79, {1} -> 83, {2} -> 16), including stimuli unlike the training ones.

**Hypothesis:** Refinement of sequence_momentum_span_exemplar (all of it kept: recency-weighted exemplar similarity on size, final digit, parity and graded shared factor with coherence-stretched attention, an order-directed momentum in size generalization beyond the filled-in span, the "multiples of g" and "ends in d" rules, and a criterion that relaxes over the session). The single change is that people do not bring one fixed yes-criterion to every set: each set of examples leaves its own holistic impression of how inclusive the program is, so the criterion shifts set by set (more generous after some sets, stricter after others) by amounts the similarity and rule evidence do not capture. This predicts that whole sets differ in how generously they are generalized more than the incumbent allows — the too-compressed between-set yes-rate spread the critique reports — while the ordering of targets within a set is still set by similarity, momentum and rules.

### coherence_decisive_momentum_span — rejected (experiment2 round 4 candidate 3 refine incumbent sequence_momentum_span_exemplar repair 1)

**Outcome:** predicts like existing model 'sequence_momentum_span_exemplar' (p_yes RMSE 0.00164 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of sequence_momentum_span_exemplar, not a new hypothesis.

**Hypothesis:** Refinement of sequence_momentum_span_exemplar (all its parts kept: recency-weighted exemplar similarity on size, final digit, parity and graded shared factor with coherence-stretched attention, the filled-in span with its order-directed momentum, the "multiples of g" and "ends in d" rules, and a criterion that relaxes over the session). The single change, grafted from coherence_decisive_dual_route_exemplar, is that how much the examples resemble one another sets how decisively people answer: after a coherent set (examples alike in size, final digit, parity or factors) the weighed evidence against their criterion is turned into a near-deterministic yes or no, whereas after an incoherent set or a single example answers stay noisy. Coherent sets thus get more extreme yes rates and incoherent ones rates nearer the middle, spreading set-level generosity apart — the too-compressed between-set yes-rate spread the critique reports.

### set_homogeneity_criterion_momentum — dropped (experiment3)

**Outcome:** MCMC fit failed on CV fold 1 (RuntimeError: warm-start NUTS: the Hessian at the posterior mode is not positive definite (some parameter direction is pinned down by neither the data nor the prior))

**Hypothesis:** Refinement of sequence_momentum_span_exemplar (all of it kept: recency-weighted exemplar similarity on size, final digit, parity and graded shared factor with coherence-stretched attention, an order-directed momentum in size generalization beyond the filled-in span, the "multiples of g" and "ends in d" rules, and a criterion that relaxes over the session). The single change is that each set's overall look sets how inclusive people take the program to be: the more the examples resemble one another (in size, final digit, parity and shared factors), the narrower a program people think produced them, so they raise their yes-criterion for every target after a homogeneous set and lower it after a heterogeneous one. Whole sets therefore differ in how generously they are generalized more than similarity alone predicts — the too-compressed between-set yes-rate spread the critique reports — while the ordering of targets within a set is still set by similarity, momentum and rules.
