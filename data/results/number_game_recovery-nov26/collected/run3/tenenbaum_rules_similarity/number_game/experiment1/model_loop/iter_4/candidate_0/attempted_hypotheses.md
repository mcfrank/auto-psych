# Tried before

8 hypotheses proposed earlier in this project are no longer in the model set. Read them before you propose:

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

### posterior_sharpened_concept_selection — rejected (experiment1 round 1 candidate 2 lens 5)

**Outcome:** predicts like existing model 'evidence_overweighting_concept_learning' (p_yes RMSE 0.00121 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of evidence_overweighting_concept_learning, not a new hypothesis.

**Hypothesis:** People do not average their predictions over every number concept consistent with the examples; they commit to (approximately) the single most plausible concept and generalize only from it. Concretely, given the same rules and magnitude ranges, family-organised rule prior and size-principle likelihood as the current best model, the concept a person generalizes from is chosen with probability proportional to its posterior raised to an inferred decisiveness exponent (above 1 = near-MAP hypothesis selection, so generalization collapses onto the smallest well-fitting concept more sharply as examples accumulate), and "yes" means the chosen concept contains the target.

### iter1_candidate3 — rejected (experiment1 round 1 candidate 3 refine incumbent rule_family_prior_concepts)

**Outcome:** predicts like existing model 'rule_family_prior_concepts' (p_yes RMSE 0.00180 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of rule_family_prior_concepts, not a new hypothesis.

**Hypothesis:** Refinement of `rule_family_prior_concepts`: people do the same Bayesian concept learning over the 45 rules (with the inferred family-level rule prior) and all intervals, with "yes" the posterior probability that the concept contains the target. The one change is how strongly people believe the examples were sampled from the concept: instead of a fixed 90% (10% of examples random noise), the share of examples they treat as genuine samples from the concept is a free parameter inferred from the data, so the strength of the size principle — how sharply generalization narrows as consistent examples accumulate, and how readily an inconsistent example is dismissed as noise — is estimated rather than assumed.

### inferred_sampling_reliability_concepts — rejected (experiment1 round 1 candidate 4 refine incumbent rule_family_prior_concepts)

**Outcome:** predicts like existing model 'evidence_overweighting_concept_learning' (p_yes RMSE 0.00000 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of evidence_overweighting_concept_learning, not a new hypothesis.

**Hypothesis:** Refinement of `rule_family_prior_concepts`: people do the same Bayesian concept learning over the 45 rules (with the family-organised rule prior) and all intervals of 1-100, with size-principle sampling, but how strongly they treat the examples as drawn from the concept is not fixed at 90% — the share of examples they believe are genuine samples of the concept (versus random numbers from 1-100) is a free quantity inferred from their judgments. The one change is replacing the fixed 10% noise rate with this inferred sampling reliability, which sets how sharply each extra example narrows generalization to the smallest concept that fits (the critique shows people narrow with more examples differently from what the fixed rate predicts) and how readily a concept that misses one example is still entertained.

### family_prior_erlang_interval_concepts — rejected (experiment1 round 1 candidate 3 refine incumbent rule_family_prior_concepts repair 1)

**Outcome:** predicts like existing model 'erlang_interval_family_prior_concepts' (p_yes RMSE 0.00176 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of erlang_interval_family_prior_concepts, not a new hypothesis.

**Hypothesis:** Refinement of `rule_family_prior_concepts`: people do the same Bayesian concept learning over the 45 rules, with the inferred family-level rule prior, and over intervals of 1-100, using the size principle (10% of examples random), with "yes" the posterior probability that the concept contains the target. The one change, taken from `erlang_interval_size_concept_learning`, is the prior over intervals: instead of every interval being equally likely, which makes wide ranges far more numerous a priori, people expect a range of a characteristic width (an Erlang prior over interval size with an inferred scale). Narrow, magnitude-based generalisation is then favoured from the first few examples, so adding examples narrows the range less than the flat-interval model predicts.

### generative_next_output_judgment — dropped (experiment1 round 3)

**Outcome:** MCMC fit failed on CV fold 1 (RuntimeError: warm-start NUTS: the Hessian at the posterior mode is not positive definite (some parameter direction is pinned down by neither the data nor the prior))

**Hypothesis:** People read the question "would the program produce this number?" generatively: instead of asking whether the target belongs to the hidden concept, they judge how likely the program is to produce that particular number as its next output (averaging over the concepts that explain the examples, each of which spreads its output thinly when it contains many numbers), and say "yes" in proportion to how that likelihood compares with the most likely next output, with an inferred sensitivity. So targets that only a large concept contains (any even number, anywhere in a wide range) get far fewer "yes" answers than their membership probability implies, while targets of a small, well-fitting concept are accepted.
