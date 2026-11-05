# Tried before

1 hypothesis proposed earlier in this project is no longer in the model set. Read them before you propose:

- A *pruned* entry entered the set and later lost to the best model by the stated margin, on the data available then. Its mechanism may be partly right: a model that changes it substantively is welcome, but do not re-propose it unchanged or merely re-parameterised, under any name. Pruned models stay readable under `models/pruned/`.
- A *rejected* entry never entered the set. If it was a near-duplicate of a model still in the set, that region is already covered: do not re-propose it. If it failed on its code or its fit (see its outcome), the idea itself was never tested and a correct implementation may be worth trying.

### base_ten_multiple_salience_overlap — rejected (experiment1 round 2 candidate 4 refine incumbent family_salience_feature_overlap)

**Outcome:** predicts like existing model 'decimal_multiple_salience_overlap' (p_yes RMSE 0.00000 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of decimal_multiple_salience_overlap, not a new hypothesis.

**Hypothesis:** Refinement of `family_salience_feature_overlap`: people still note every salient mathematical property the examples all share and say "yes" in proportion to the salience-weighted fraction of that shared profile the target also has, with no size principle. The one change is that multiples of five and ten (the round numbers of base-ten counting) form their own family of property with their own salience, separate from multiples of other numbers (3, 4, 6, 7, 8, 9, 11, 12), so when examples share both (e.g. 90 and 45 are multiples of 5 and of 9) people generalize along the round-number property far more than along the other multiple.
