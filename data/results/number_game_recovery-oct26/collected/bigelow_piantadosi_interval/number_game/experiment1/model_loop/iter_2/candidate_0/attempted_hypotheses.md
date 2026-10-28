# Tried before

2 hypotheses proposed earlier in this project are no longer in the model set. Read them before you propose:

- A *pruned* entry entered the set and later lost to the best model by the stated margin, on the data available then. Its mechanism may be partly right: a model that changes it substantively is welcome, but do not re-propose it unchanged or merely re-parameterised, under any name. Pruned models stay readable under `models/pruned/`.
- A *rejected* entry never entered the set. If it was a near-duplicate of a model still in the set, that region is already covered: do not re-propose it. If it failed on its code or its fit (see its outcome), the idea itself was never tested and a correct implementation may be worth trying.

### inferred_noise_concept_learning — rejected (experiment1 round 1 candidate 4 refine incumbent concept_learning_with_guessing)

**Outcome:** predicts like existing model 'example_noise_concept_learning' (p_yes RMSE 0.00131 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of example_noise_concept_learning, not a new hypothesis.

**Hypothesis:** Refinement of concept_learning_with_guessing: people do the same Bayesian concept learning over 42 weighted rules and all intervals of 1-100 (size principle), with the same guessing on a fraction of trials, but the share of examples they believe are random noise rather than drawn from the concept is not fixed at 10% — it is a property of their reasoning that we infer. The single change is this inferred noise belief: the more examples people treat as possibly random, the more readily a rule survives an example that violates it and the weaker the size principle's pull toward the tightest concept, changing generalization on sets with an odd-one-out example.

### inferred_outlier_rate_concept_learning — rejected (experiment1 round 1 candidate 5 refine chosen)

**Outcome:** predicts like existing model 'inferred_example_noise_belief' (p_yes RMSE 0.00166 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of inferred_example_noise_belief, not a new hypothesis.

**Hypothesis:** Refinement of bigelow_piantadosi_independent: people do the same Bayesian concept learning over the 42 weighted rules and all intervals of 1-100 with the size principle, but how often they believe an example is a stray, random number rather than a true member of the concept is not fixed at 10% — it is a property of the learner, inferred from the data. The one change is this inferred outlier rate: a high rate lets a concept that fits most (but not all) of the examples keep posterior weight and weakens how sharply the examples narrow generalization, whereas a low rate makes a single misfit example rule a concept out.
