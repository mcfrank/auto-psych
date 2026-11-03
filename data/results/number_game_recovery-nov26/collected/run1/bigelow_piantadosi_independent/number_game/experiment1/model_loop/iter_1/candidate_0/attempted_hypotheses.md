# Tried before

1 hypothesis proposed earlier in this project is no longer in the model set. Read them before you propose:

- A *pruned* entry entered the set and later lost to the best model by the stated margin, on the data available then. Its mechanism may be partly right: a model that changes it substantively is welcome, but do not re-propose it unchanged or merely re-parameterised, under any name. Pruned models stay readable under `models/pruned/`.
- A *rejected* entry never entered the set. If it was a near-duplicate of a model still in the set, that region is already covered: do not re-propose it. If it failed on its code or its fit (see its outcome), the idea itself was never tested and a correct implementation may be worth trying.

### tempered_size_principle_mixture_2 — rejected (experiment1 round 0 candidate 4 refine incumbent bigelow_piantadosi_mixture)

**Outcome:** predicts like existing model 'tempered_size_principle_mixture' (p_yes RMSE 0.00000 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of tempered_size_principle_mixture, not a new hypothesis.

**Hypothesis:** Refinement of `bigelow_piantadosi_mixture`: people do Bayesian concept learning over the same 45 arithmetic rules and all intervals of 1-100 (same prior), but they do not weigh the examples' evidence as an ideal learner would. The one change is that the size principle has a free strength: the evidence the examples give for each candidate concept is tempered (or sharpened) by a single exponent, so people may favour a broad rule such as "even numbers" over a tight interval around the examples more (or less) than strong sampling predicts — which governs how sharply they reject nearby numbers of the wrong parity and how far they extend a rule to distant numbers.
