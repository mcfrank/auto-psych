# Tried before

1 hypothesis proposed earlier in this project is no longer in the model set. Read them before you propose:

- A *pruned* entry entered the set and later lost to the best model by the stated margin, on the data available then. Its mechanism may be partly right: a model that changes it substantively is welcome, but do not re-propose it unchanged or merely re-parameterised, under any name. Pruned models stay readable under `models/pruned/`.
- A *rejected* entry never entered the set. If it was a near-duplicate of a model still in the set, that region is already covered: do not re-propose it. If it failed on its code or its fit (see its outcome), the idea itself was never tested and a correct implementation may be worth trying.

### specificity_biased_rule_prior_lapse — rejected (experiment1 round 2 candidate 5 refine chosen)

**Outcome:** predicts like existing model 'rule_breadth_prior_concepts' (p_yes RMSE 0.00031 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of rule_breadth_prior_concepts, not a new hypothesis.

**Hypothesis:** Refinement of `concept_learning_with_guess_lapse`: people do Bayesian concept learning over the same 45 rules (including the digit concepts) and every interval of 1-100. They apply the size principle, treat 10% of examples as random, and on a fixed fraction of trials ignore the concept and guess "yes" at their own baseline rate. The one change is that their prior over rules is neither 45 free weights nor flat. It follows a specificity bias: a rule's prior plausibility falls (or rises) as a power of how many numbers it contains, so narrow rules like "powers of 3" are favoured over broad ones like "odd numbers" before any example is seen. This predicts weaker generalization to broad rules (e.g. all odd numbers after 85, 19, 91) than an equal-prior learner, with the strength of the bias set by a single parameter.
