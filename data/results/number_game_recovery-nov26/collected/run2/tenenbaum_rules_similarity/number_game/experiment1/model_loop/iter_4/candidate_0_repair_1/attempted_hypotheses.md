# Tried before

4 hypotheses proposed earlier in this project are no longer in the model set. Read them before you propose:

- A *pruned* entry entered the set and later lost to the best model by the stated margin, on the data available then. Its mechanism may be partly right: a model that changes it substantively is welcome, but do not re-propose it unchanged or merely re-parameterised, under any name. Pruned models stay readable under `models/pruned/`.
- A *rejected* entry never entered the set. If it was a near-duplicate of a model still in the set, that region is already covered: do not re-propose it. If it failed on its code or its fit (see its outcome), the idea itself was never tested and a correct implementation may be worth trying.

### bayesian_concepts_with_guess_lapse — rejected (experiment1 round 0 candidate 4 refine incumbent bigelow_piantadosi_independent repair 1)

**Outcome:** predicts like existing model 'bayesian_concepts_with_guessing_lapse' (p_yes RMSE 0.00154 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of bayesian_concepts_with_guessing_lapse, not a new hypothesis.

**Hypothesis:** Refinement of bigelow_piantadosi_independent: people do the same Bayesian concept learning over the 45 rules and all intervals (rule weights inferred separately, size principle, 10% noisy examples), but on a fraction of trials they do not consult that posterior at all and instead guess, answering "yes" with a fixed bias. The one change is this response-lapse process: it predicts that even targets among the examples, or clearly inside the learned concept, get "no" at a non-trivial rate (and clearly excluded targets get occasional "yes"), which the incumbent, whose answers follow the posterior exactly, cannot produce.

### pedagogical_example_selection — dropped (experiment1 round 3)

**Outcome:** fold 2 too slow to fit: the fit of 'pedagogical_example_selection' (target_accept 0.99) was still sampling after the 30-minute limit and was stopped. Every sampling run of a candidate's admission fit has a 30-minute limit. Make the model cheaper to evaluate and easier to sample: vectorise the likelihood over trials (no Python loops, pytensor scan or per-trial subgraphs), compute features once per unique sequence (in compute_features or prepare_observed, not in the graph), and drop or merge parameters the data barely constrain — a weakly identified posterior makes NUTS take maximal-length trajectories.

**Hypothesis:** People assume the examples were chosen by a helpful teacher rather than drawn at random from the concept (pedagogical sampling): a teacher picks examples that point to the true concept, so a number counts more as evidence for a concept the fewer other plausible concepts it also belongs to. People do Bayesian concept learning over the same rules and ranges of 1-100, but they interpret each example by how diagnostic it is. Concepts made up mostly of numbers a teacher would happily show (distinctive, rarely shared ones) are penalised when such numbers were not shown, so generalisation narrows to the concepts that the chosen examples single out.

### new_set_first_query_caution — rejected (experiment1 round 3 candidate 3 refine incumbent noisy_examples_erlang_intervals)

**Outcome:** model binds ['trial_index'] — response-row bookkeeping a bare stimulus never carries — so it cannot be evaluated on a stimulus pool (the novelty gate now, the held-out evaluation later); compute everything from number_set and target.

**Hypothesis:** A refinement of noisy_examples_erlang_intervals: people do the same Bayesian concept learning over the 45 independently weighted rules and Erlang-size-weighted intervals of 1-100, with the size principle and an inferred example-noise rate, but on the first question asked right after a new set of examples appears they are cautious and, some inferred fraction of the time, answer "no" regardless of their posterior. The one change is this new-set caution, which predicts a lower yes-rate on the first query about a set than on later queries repeating the same set.

### noisy_examples_log_width_intervals — dropped (experiment1 round 3)

**Outcome:** fold 0 too slow to fit: the fit of 'noisy_examples_log_width_intervals' (target_accept 0.99) was still sampling after the 30-minute limit and was stopped. Every sampling run of a candidate's admission fit has a 30-minute limit. Make the model cheaper to evaluate and easier to sample: vectorise the likelihood over trials (no Python loops, pytensor scan or per-trial subgraphs), compute features once per unique sequence (in compute_features or prepare_observed, not in the graph), and drop or merge parameters the data barely constrain — a weakly identified posterior makes NUTS take maximal-length trajectories.

**Hypothesis:** A refinement of log_scale_interval_width_prior: people do Bayesian concept learning over the 45 independently weighted rules and all intervals of 1-100 with the size principle, and they expect a range to have a typical width judged on a compressed, logarithmic mental number line (narrow among small numbers, wide among large ones), but they do not assume a fixed 10% of examples are random. The one change is grafting the inferred example-noise rate from inferred_example_noise_concepts: how often people believe the program emits an example from outside its concept is inferred from the data, governing how sharply the size principle accumulates over examples and how readily a rule or range survives an example it does not cover.
