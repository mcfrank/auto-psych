# Tried before

3 hypotheses proposed earlier in this project are no longer in the model set. Read them before you propose:

- A *pruned* entry entered the set and later lost to the best model by the stated margin, on the data available then. Its mechanism may be partly right: a model that changes it substantively is welcome, but do not re-propose it unchanged or merely re-parameterised, under any name. Pruned models stay readable under `models/pruned/`.
- A *rejected* entry never entered the set. If it was a near-duplicate of a model still in the set, that region is already covered: do not re-propose it. If it failed on its code or its fit (see its outcome), the idea itself was never tested and a correct implementation may be worth trying.

### inferred_sampling_noise_concept_learning — rejected (experiment1 round 0 candidate 3 refine incumbent bigelow_piantadosi_independent)

**Outcome:** predicts like existing model 'bigelow_piantadosi_independent' (p_yes RMSE 0.00165 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of bigelow_piantadosi_independent, not a new hypothesis.

**Hypothesis:** A refinement of bigelow_piantadosi_independent: people do Bayesian concept learning over the same 45 rules and all intervals of 1-100, with a prior weight for each rule and a share lambda_0 for rules, and answer "yes" with the posterior probability that the concept contains the target. The one change is that how far people trust the examples to have come from the concept (the strength of the size principle, versus examples being random numbers from 1-100) is inferred from their judgments rather than fixed at 90%, so the data decide how strongly small concepts are favoured and how readily a concept is kept despite an example it does not contain.

### inferred_outlier_rate_concept_learning — rejected (experiment1 round 0 candidate 4 refine incumbent bigelow_piantadosi_independent)

**Outcome:** predicts like existing model 'bigelow_piantadosi_independent' (p_yes RMSE 0.00130 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of bigelow_piantadosi_independent, not a new hypothesis.

**Hypothesis:** A refinement of `bigelow_piantadosi_independent`: people do Bayesian concept learning over the same 45 rules and all intervals of 1-100 with independently weighted rules and the size principle, but how far they trust each example to come from the concept is not fixed at 90%. Instead the rate at which examples are believed to be random noise drawn uniformly from 1-100 is a free parameter inferred from the data, so that sets containing an "odd one out" (e.g. 8, 24, 48, 63) are explained by a rule plus an outlier to the degree people actually tolerate outliers.

### interval_size_erlang_prior — rejected (experiment1 round 0 candidate 4 refine incumbent bigelow_piantadosi_independent repair 1)

**Outcome:** predicts like existing model 'erlang_interval_size_concept_learning' (p_yes RMSE 0.00000 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of erlang_interval_size_concept_learning, not a new hypothesis.

**Hypothesis:** A refinement of `bigelow_piantadosi_independent`: people do Bayesian concept learning over the same 45 independently weighted rules and all intervals of 1-100 with the size principle (10% of examples treated as noise), but their prior over magnitude ranges is not flat over endpoint pairs. Instead people expect a range of a characteristic size: the prior on an interval's width follows an Erlang distribution (as in Tenenbaum's number game) whose scale is inferred from the data, so ranges much narrower or much wider than that typical size are judged implausible a priori, changing how far "yes" extends around examples that suggest a magnitude range.
