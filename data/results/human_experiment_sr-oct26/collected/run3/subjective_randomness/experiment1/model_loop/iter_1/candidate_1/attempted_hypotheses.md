# Tried before

1 hypothesis proposed earlier in this project is no longer in the model set. Read them before you propose:

- A *pruned* entry entered the set and later lost to the best model by the stated margin, on the data available then. Its mechanism may be partly right: a model that changes it substantively is welcome, but do not re-propose it unchanged or merely re-parameterised, under any name. Pruned models stay readable under `models/pruned/`.
- A *rejected* entry never entered the set. If it was a near-duplicate of a model still in the set, that region is already covered: do not re-propose it. If it failed on its code or its fit (see its outcome), the idea itself was never tested and a correct implementation may be worth trying.

### negative_recency_expectation_fit — rejected (experiment1 round 0 candidate 0 lens 0)

**Outcome:** MCMC did not converge (401 divergent transitions of 12000), too far from converging for smaller NUTS steps to help (the fitter refits only a near miss — at most 2% divergent transitions, R-hat <= 1.2, bulk ESS >= 20 — at target_accept 0.95), so raising target_accept will not help. Change the model's geometry instead: non-centred parameterisations for hierarchical or scale parameters, tighter (weakly informative) priors on parameters the data barely constrain, fewer weakly identified parameters (drop or merge parameters that trade off against each other), and no hard thresholds or discontinuities in the likelihood.

**Hypothesis:** People judge randomness by reading each sequence flip by flip and checking how well every flip matches a gambler's-fallacy expectation: after a streak, they expect the coin to switch, increasingly so the longer the streak has run. A sequence looks random to the extent its flips conform to this negative-recency expectation, with the most recently read flips (near the end of the sequence) weighing more heavily, and people differ in how strongly this conformity drives their choice.
