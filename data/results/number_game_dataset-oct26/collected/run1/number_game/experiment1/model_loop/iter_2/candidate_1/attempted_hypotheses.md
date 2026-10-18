# Tried before

1 hypothesis proposed earlier in this project is no longer in the model set. Read them before you propose:

- A *pruned* entry entered the set and later lost to the best model by the stated margin, on the data available then. Its mechanism may be partly right: a model that changes it substantively is welcome, but do not re-propose it unchanged or merely re-parameterised, under any name. Pruned models stay readable under `models/pruned/`.
- A *rejected* entry never entered the set. If it was a near-duplicate of a model still in the set, that region is already covered: do not re-propose it. If it failed on its code or its fit (see its outcome), the idea itself was never tested and a correct implementation may be worth trying.

### nearest_exemplar_feature_similarity — rejected (experiment1 round 1 candidate 3 refine incumbent exemplar_feature_similarity_gcm)

**Outcome:** MCMC did not converge (max R-hat 1.375 > 1.05; min bulk ESS 9 < 100), too far from converging for smaller NUTS steps to help (the fitter refits only a near miss — at most 2% divergent transitions, R-hat <= 1.2, bulk ESS >= 20 — at target_accept 0.99), so raising target_accept will not help. Change the model's geometry instead: non-centred parameterisations for hierarchical or scale parameters, tighter (weakly informative) priors on parameters the data barely constrain, fewer weakly identified parameters (drop or merge parameters that trade off against each other), and no hard thresholds or discontinuities in the likelihood.

**Hypothesis:** A refinement of `exemplar_feature_similarity_gcm`. It keeps the same exemplar representation: magnitude plus arithmetic features, with similarity falling off exponentially with attention-weighted distance. The one change is that people judge the target by its similarity to the single most similar example shown (nearest-neighbour retrieval), not by its similarity summed over all the examples. So adding examples does not inflate "yes" for every target. A lone example generalizes as broadly as a rich set, and a target that closely matches any one example on its arithmetic features is accepted however far it is from the others. This addresses the critique's finding that the summed model under-generalizes from single examples and from numerically distant targets.
