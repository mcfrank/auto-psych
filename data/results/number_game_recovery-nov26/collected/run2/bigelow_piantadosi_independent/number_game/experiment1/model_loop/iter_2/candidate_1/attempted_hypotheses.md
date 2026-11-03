# Tried before

2 hypotheses proposed earlier in this project are no longer in the model set. Read them before you propose:

- A *pruned* entry entered the set and later lost to the best model by the stated margin, on the data available then. Its mechanism may be partly right: a model that changes it substantively is welcome, but do not re-propose it unchanged or merely re-parameterised, under any name. Pruned models stay readable under `models/pruned/`.
- A *rejected* entry never entered the set. If it was a near-duplicate of a model still in the set, that region is already covered: do not re-propose it. If it failed on its code or its fit (see its outcome), the idea itself was never tested and a correct implementation may be worth trying.

### hierarchical_rule_prior_mixture — rejected (experiment1 round 0 candidate 4 refine incumbent bigelow_piantadosi_mixture)

**Outcome:** MCMC did not converge (3 divergent transitions of 2000), even at target_accept 0.99 (the fitter already refits a near miss with smaller NUTS steps, so raising target_accept will not help). Change the model's geometry instead: non-centred parameterisations for hierarchical or scale parameters, tighter (weakly informative) priors on parameters the data barely constrain, fewer weakly identified parameters (drop or merge parameters that trade off against each other), and no hard thresholds or discontinuities in the likelihood.

**Hypothesis:** Refinement of bigelow_piantadosi_mixture: people learn concepts by the same Bayesian inference over 45 mathematical rules and all intervals of 1-100 with the size principle, but they do not hold the rules as equally likely a priori — each rule (e.g. odd, even, ends in a digit) carries its own prior plausibility, inferred from the data under a shared hierarchical spread. The single change is replacing the uniform prior over rules with a learned, non-uniform one (the core idea of Bigelow & Piantadosi's prior inference), so favoured rules such as parity and shared final digits can dominate interval-based similarity when the examples support them.

### tempered_learned_rule_prior_2 — rejected (experiment1 round 1 candidate 4 refine incumbent learned_rule_prior_mixture)

**Outcome:** predicts like existing model 'tempered_learned_rule_prior' (p_yes RMSE 0.00000 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of tempered_learned_rule_prior, not a new hypothesis.

**Hypothesis:** Refinement of `learned_rule_prior_mixture`: people do the same Bayesian concept learning over the 45 mathematical rules (each with its own learned prior plausibility) and all intervals of 1-100, but they apply the size principle only partially — the evidence that the examples were sampled from a small concept is tempered by a free "sampling strength" exponent (1 = full strong sampling, 0 = weak sampling). The single change is grafting this tempered likelihood from `tempered_size_principle_mixture` onto the learned rule prior, predicting that broad rules and wide intervals keep more posterior weight after a few examples than an ideal learner allows, so generalization is less sharply confined to the tightest consistent concept.
