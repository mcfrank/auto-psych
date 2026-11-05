# Tried before

1 hypothesis proposed earlier in this project is no longer in the model set. Read them before you propose:

- A *pruned* entry entered the set and later lost to the best model by the stated margin, on the data available then. Its mechanism may be partly right: a model that changes it substantively is welcome, but do not re-propose it unchanged or merely re-parameterised, under any name. Pruned models stay readable under `models/pruned/`.
- A *rejected* entry never entered the set. If it was a near-duplicate of a model still in the set, that region is already covered: do not re-propose it. If it failed on its code or its fit (see its outcome), the idea itself was never tested and a correct implementation may be worth trying.

### tempered_evidence_concept_learning — rejected (experiment1 round 0 candidate 4 refine incumbent bigelow_piantadosi_mixture)

**Outcome:** predicts like existing model 'conservative_evidence_concept_learning' (p_yes RMSE 0.00008 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of conservative_evidence_concept_learning, not a new hypothesis.

**Hypothesis:** Refinement of bigelow_piantadosi_mixture: the same Bayesian concept learner over 45 rules and all intervals of 1-100 (rules chosen with probability lambda_0 and equally likely, each interval (1 - lambda_0)/100^2, examples sampled from the concept with 10% noise), except that people under-weight the evidence of the examples — the likelihood of the examples is raised to a single inferred exponent (tempered sampling evidence) before being combined with the prior. With an exponent below 1 the size principle is weakened, so people are less confident in the smallest consistent rule and spread belief onto broader rules and intervals, endorsing minimal-rule members less and nearby non-members more.
