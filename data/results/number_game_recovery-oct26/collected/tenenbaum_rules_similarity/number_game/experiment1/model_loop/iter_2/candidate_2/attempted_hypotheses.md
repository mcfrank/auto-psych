# Tried before

2 hypotheses proposed earlier in this project are no longer in the model set. Read them before you propose:

- A *pruned* entry entered the set and later lost to the best model by the stated margin, on the data available then. Its mechanism may be partly right: a model that changes it substantively is welcome, but do not re-propose it unchanged or merely re-parameterised, under any name. Pruned models stay readable under `models/pruned/`.
- A *rejected* entry never entered the set. If it was a near-duplicate of a model still in the set, that region is already covered: do not re-propose it. If it failed on its code or its fit (see its outcome), the idea itself was never tested and a correct implementation may be worth trying.

### few_sample_majority_concept — rejected (experiment1 round 1 candidate 1 lens 4)

**Outcome:** predicts like existing model 'few_sample_majority_vote' (p_yes RMSE 0.00165 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of few_sample_majority_vote, not a new hypothesis.

**Hypothesis:** People do Bayesian concept learning over arithmetic rules and intervals of 1-100 (size principle, a width-based Erlang prior on intervals), but instead of reporting the posterior probability that the target belongs to the concept, they decide from a small number of hypotheses sampled from that posterior and go with the majority. This resource limit distorts the normative judgement by a single response-gain: posterior beliefs are pushed toward certain "yes"/"no" (few samples, majority rule) or compressed toward 50/50, with the degree of distortion inferred from the data.

### erlang_interval_with_response_lapse — rejected (experiment1 round 1 candidate 4 refine incumbent erlang_interval_size_prior)

**Outcome:** predicts like existing model 'erlang_interval_response_lapse' (p_yes RMSE 0.00062 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of erlang_interval_response_lapse, not a new hypothesis.

**Hypothesis:** Refinement of erlang_interval_size_prior: people do the same Bayesian concept learning over 42 rules and all intervals of 1-100 (size principle, Erlang prior over interval width), but on some fraction of trials they do not consult their posterior at all and instead answer by a fixed yes/no guessing bias. The one change is this response lapse, which explains why people say "no" even to targets that are among the examples (about a third of the time here) and "yes" to targets no plausible concept contains, rather than forcing the concept posterior itself to absorb those answers.
