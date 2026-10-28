# Tried before

4 hypotheses proposed earlier in this project are no longer in the model set. Read them before you propose:

- A *pruned* entry entered the set and later lost to the best model by the stated margin, on the data available then. Its mechanism may be partly right: a model that changes it substantively is welcome, but do not re-propose it unchanged or merely re-parameterised, under any name. Pruned models stay readable under `models/pruned/`.
- A *rejected* entry never entered the set. If it was a near-duplicate of a model still in the set, that region is already covered: do not re-propose it. If it failed on its code or its fit (see its outcome), the idea itself was never tested and a correct implementation may be worth trying.

### common_factor_exemplar_similarity_2 — rejected (experiment1 round 1 candidate 4 refine incumbent recency_weighted_exemplar_similarity)

**Outcome:** predicts like existing model 'common_factor_exemplar_similarity' (p_yes RMSE 0.00000 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of common_factor_exemplar_similarity, not a new hypothesis.

**Hypothesis:** Refinement of recency_weighted_exemplar_similarity: people judge a target by its recency-weighted similarity to the remembered examples, but when several examples share a common factor of 3 or more they also abstract that shared factor as a property of the whole set, and a target that is not a multiple of it is judged psychologically farther from every example. The single change is this set-level "shared common factor" mismatch added to the exemplar distance (an inferred weight; 0 recovers the incumbent), which predicts more "yes" answers to multiples of the examples' greatest common divisor than pairwise divisibility, digit and parity similarity alone.

### set_factor_tuned_attention_exemplar_2 — rejected (experiment1 round 2 candidate 4 refine incumbent coherence_tuned_attention_exemplar)

**Outcome:** predicts like existing model 'set_factor_tuned_attention_exemplar' (p_yes RMSE 0.00000 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of set_factor_tuned_attention_exemplar, not a new hypothesis.

**Hypothesis:** Refinement of coherence_tuned_attention_exemplar: people still judge a target by its recency-weighted similarity to the remembered examples, with attention to each dimension (size, final digit, parity, shared factor) stretched toward the respects in which the examples resemble one another. The single change is what the shared-factor dimension compares: when the examples all share a common divisor of 3 or more, people judge each target-example pair on whether the target is a multiple of that set-wide common factor, rather than on whether the two numbers happen to share any factor pairwise (which they still do for single examples or sets with no common factor). So a target that shares some other factor with an example (e.g. 9 after 12, 24, 36) no longer counts as similar on that dimension, predicting a sharper "multiples of g" gap than the incumbent, as the critique's common-factor rule gap shows.

### iter3_candidate4 — rejected (experiment1 round 3 candidate 4 refine incumbent graded_factor_coherence_tuned_exemplar)

**Outcome:** predicts like existing model 'dimension_specific_coherence_exemplar' (p_yes RMSE 0.00000 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of dimension_specific_coherence_exemplar, not a new hypothesis.

**Hypothesis:** Refinement of graded_factor_coherence_tuned_exemplar: people still judge a target by its recency-weighted exponential similarity to the remembered examples (size, final digit, parity, graded shared factor), with attention stretched toward the respects in which the examples resemble one another, but how readily a shared property captures attention differs by kind of property. The single change is that each dimension gets its own coherence sensitivity instead of one shared stretch factor, so a set whose examples all end in the same digit (or share a large factor) can make people generalize an "ends in d" (or "multiples of g") rule far more sharply than coherence in size or parity does, closing the shared-final-digit and common-factor rule gaps in the critique.

### dimension_specific_coherence_attention — dropped (experiment1 round 3)

**Outcome:** MCMC fit failed on CV fold 3 (RuntimeError: warm-start NUTS: the Hessian at the posterior mode is not positive definite (some parameter direction is pinned down by neither the data nor the prior))

**Hypothesis:** Refinement of coherence_tuned_attention_exemplar: people still judge a target by its recency-weighted similarity to the remembered examples, with attention stretched toward the dimensions (size, final digit, parity, shared factor) on which the examples resemble one another, but how readily a shared property captures attention differs by kind of property: a run of examples all ending in the same digit is far more conspicuous than shared parity or closeness in size. The single change is that each dimension has its own coherence sensitivity instead of one common stretch rate, so sets whose examples share a final digit (or a factor) make people generalize sharply to targets matching that property, closing the "ends in d" and "multiples of g" rule gaps, while a single example still leaves attention at its baseline.
