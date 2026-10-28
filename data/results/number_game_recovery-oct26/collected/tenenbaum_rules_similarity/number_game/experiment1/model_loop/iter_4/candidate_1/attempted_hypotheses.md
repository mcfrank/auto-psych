# Tried before

5 hypotheses proposed earlier in this project are no longer in the model set. Read them before you propose:

- A *pruned* entry entered the set and later lost to the best model by the stated margin, on the data available then. Its mechanism may be partly right: a model that changes it substantively is welcome, but do not re-propose it unchanged or merely re-parameterised, under any name. Pruned models stay readable under `models/pruned/`.
- A *rejected* entry never entered the set. If it was a near-duplicate of a model still in the set, that region is already covered: do not re-propose it. If it failed on its code or its fit (see its outcome), the idea itself was never tested and a correct implementation may be worth trying.

### few_sample_majority_concept — rejected (experiment1 round 1 candidate 1 lens 4)

**Outcome:** predicts like existing model 'few_sample_majority_vote' (p_yes RMSE 0.00165 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of few_sample_majority_vote, not a new hypothesis.

**Hypothesis:** People do Bayesian concept learning over arithmetic rules and intervals of 1-100 (size principle, a width-based Erlang prior on intervals), but instead of reporting the posterior probability that the target belongs to the concept, they decide from a small number of hypotheses sampled from that posterior and go with the majority. This resource limit distorts the normative judgement by a single response-gain: posterior beliefs are pushed toward certain "yes"/"no" (few samples, majority rule) or compressed toward 50/50, with the degree of distortion inferred from the data.

### erlang_interval_with_response_lapse — rejected (experiment1 round 1 candidate 4 refine incumbent erlang_interval_size_prior)

**Outcome:** predicts like existing model 'erlang_interval_response_lapse' (p_yes RMSE 0.00062 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of erlang_interval_response_lapse, not a new hypothesis.

**Hypothesis:** Refinement of erlang_interval_size_prior: people do the same Bayesian concept learning over 42 rules and all intervals of 1-100 (size principle, Erlang prior over interval width), but on some fraction of trials they do not consult their posterior at all and instead answer by a fixed yes/no guessing bias. The one change is this response lapse, which explains why people say "no" even to targets that are among the examples (about a third of the time here) and "yes" to targets no plausible concept contains, rather than forcing the concept posterior itself to absorb those answers.

### gamma_shape_interval_width_prior_2 — rejected (experiment1 round 2 candidate 4 refine incumbent erlang_interval_size_prior)

**Outcome:** predicts like existing model 'gamma_shape_interval_width_prior' (p_yes RMSE 0.00137 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of gamma_shape_interval_width_prior, not a new hypothesis.

**Hypothesis:** Refinement of erlang_interval_size_prior: people do the same Bayesian concept learning over 42 arithmetic rules and every interval of 1-100 (inferred rule weights and rule share, size principle with 10% random examples, "yes" = posterior probability the concept contains the target), but their prior over how wide a range is takes a Gamma shape whose shape is inferred, instead of the Erlang's fixed shape of two. The one change is this free shape: if people's width prior falls off from the narrowest ranges (shape near or below one) rather than peaking at a moderate width, they believe the range ends close to the outermost examples, so they say "yes" to numbers inside the examples' span but sharply less to numbers just beyond it.

### tempered_evidence_erlang_width — rejected (experiment1 round 2 candidate 5 refine chosen)

**Outcome:** predicts like existing model 'erlang_interval_size_prior' (p_yes RMSE 0.00082 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of erlang_interval_size_prior, not a new hypothesis.

**Hypothesis:** Refinement of `tempered_evidence_interval`: people do Bayesian concept learning over the same 42 arithmetic rules and every interval of 1-100 (inferred rule weights and rule share, size principle with 10% random examples, "yes" = posterior probability the concept contains the target), and they weigh the examples' evidence at an inferred strength, raising the size-principle likelihood to a power rather than using it at exactly full strength. The one change is the interval prior: instead of a Beta density over where an interval lies, intervals get an Erlang prior over their width (from `erlang_interval_size_prior`), so evidence strength and an expected range width are separated. The claim is that people over-weight the examples relative to the ideal learner (a power above one), which makes them commit to ranges that just cover the examples, giving a sharper drop in "yes" at the edge of the examples' span than the ideal learner predicts, as the critique of the best model reports.

### per_end_decade_boundary_prior — rejected (experiment1 round 3 candidate 4 refine incumbent decade_bounded_interval_prior)

**Outcome:** predicts like existing model 'erlang_interval_size_prior' (p_yes RMSE 0.00085 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of erlang_interval_size_prior, not a new hypothesis.

**Hypothesis:** Refinement of `decade_bounded_interval_prior`: people do the same Bayesian concept learning over 42 arithmetic rules and every interval of 1-100 (inferred rule weights and rule share, size principle with 10% random examples, Erlang prior over interval width, "yes" = posterior probability the concept contains the target) with a prior preference for ranges bounded by round decade numbers, but each end of a range is judged round or not on its own: a range gains the inferred round-boundary factor once for a round lower end (1, 11, 21, ...) and once more for a round upper end (10, 20, ..., 100), instead of only when both ends are round. The one change is this per-end preference, so people can stop generalizing at the decade edge on one side of the examples (e.g. "up to 50") while extending past it on the other, rather than favouring only whole-decade blocks.
