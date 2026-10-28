# Tried before

1 hypothesis proposed earlier in this project is no longer in the model set. Read them before you propose:

- A *pruned* entry entered the set and later lost to the best model by the stated margin, on the data available then. Its mechanism may be partly right: a model that changes it substantively is welcome, but do not re-propose it unchanged or merely re-parameterised, under any name. Pruned models stay readable under `models/pruned/`.
- A *rejected* entry never entered the set. If it was a near-duplicate of a model still in the set, that region is already covered: do not re-propose it. If it failed on its code or its fit (see its outcome), the idea itself was never tested and a correct implementation may be worth trying.

### common_factor_exemplar_similarity_2 — rejected (experiment1 round 1 candidate 4 refine incumbent recency_weighted_exemplar_similarity)

**Outcome:** predicts like existing model 'common_factor_exemplar_similarity' (p_yes RMSE 0.00000 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of common_factor_exemplar_similarity, not a new hypothesis.

**Hypothesis:** Refinement of recency_weighted_exemplar_similarity: people judge a target by its recency-weighted similarity to the remembered examples, but when several examples share a common factor of 3 or more they also abstract that shared factor as a property of the whole set, and a target that is not a multiple of it is judged psychologically farther from every example. The single change is this set-level "shared common factor" mismatch added to the exemplar distance (an inferred weight; 0 recovers the incumbent), which predicts more "yes" answers to multiples of the examples' greatest common divisor than pairwise divisibility, digit and parity similarity alone.
