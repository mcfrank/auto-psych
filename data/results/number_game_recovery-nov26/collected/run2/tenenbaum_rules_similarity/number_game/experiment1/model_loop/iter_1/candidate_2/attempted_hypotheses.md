# Tried before

1 hypothesis proposed earlier in this project is no longer in the model set. Read them before you propose:

- A *pruned* entry entered the set and later lost to the best model by the stated margin, on the data available then. Its mechanism may be partly right: a model that changes it substantively is welcome, but do not re-propose it unchanged or merely re-parameterised, under any name. Pruned models stay readable under `models/pruned/`.
- A *rejected* entry never entered the set. If it was a near-duplicate of a model still in the set, that region is already covered: do not re-propose it. If it failed on its code or its fit (see its outcome), the idea itself was never tested and a correct implementation may be worth trying.

### bayesian_concepts_with_guess_lapse — rejected (experiment1 round 0 candidate 4 refine incumbent bigelow_piantadosi_independent repair 1)

**Outcome:** predicts like existing model 'bayesian_concepts_with_guessing_lapse' (p_yes RMSE 0.00154 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of bayesian_concepts_with_guessing_lapse, not a new hypothesis.

**Hypothesis:** Refinement of bigelow_piantadosi_independent: people do the same Bayesian concept learning over the 45 rules and all intervals (rule weights inferred separately, size principle, 10% noisy examples), but on a fraction of trials they do not consult that posterior at all and instead guess, answering "yes" with a fixed bias. The one change is this response-lapse process: it predicts that even targets among the examples, or clearly inside the learned concept, get "no" at a non-trivial rate (and clearly excluded targets get occasional "yes"), which the incumbent, whose answers follow the posterior exactly, cannot produce.
