# Tried before

6 hypotheses proposed earlier in this project are no longer in the model set. Read them before you propose:

- A *pruned* entry entered the set and later lost to the best model by the stated margin, on the data available then. Its mechanism may be partly right: a model that changes it substantively is welcome, but do not re-propose it unchanged or merely re-parameterised, under any name. Pruned models stay readable under `models/pruned/`.
- A *rejected* entry never entered the set. If it was a near-duplicate of a model still in the set, that region is already covered: do not re-propose it. If it failed on its code or its fit (see its outcome), the idea itself was never tested and a correct implementation may be worth trying.

### serial_position_weighted_concepts — rejected (experiment1 round 0 candidate 2 lens 2)

**Outcome:** predicts like existing model 'bigelow_piantadosi_independent' (p_yes RMSE 0.00118 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of bigelow_piantadosi_independent, not a new hypothesis.

**Hypothesis:** People learn the hidden program's concept by Bayesian inference over arithmetic rules and numeric intervals (size principle), but they do not weigh the examples equally: how strongly each example counts as evidence depends on its serial position in the order shown, with later (recency) or earlier (primacy) examples dominating the inferred concept. The total evidence is fixed by the number of examples and only its distribution across positions changes, so a single example gives ordinary concept learning, while reordering the same examples changes which targets get a "yes".

### fatigue_lapse_interval_concepts — rejected (experiment1 round 0 candidate 3 refine incumbent bigelow_piantadosi_interval)

**Outcome:** model binds ['trial_index'] — response-row bookkeeping a bare stimulus never carries — so it cannot be evaluated on a stimulus pool (the novelty gate now, the held-out evaluation later); compute everything from number_set and target.

**Hypothesis:** Refinement of `bigelow_piantadosi_interval`: people answer by the same Bayesian concept learning over 42 arithmetic rules and all intervals of 1-100 (rule weights inferred, interval prior a Beta density over the midpoint, size principle), but on some trials they stop consulting the concept and answer yes or no at random, and these random lapses become more frequent as the session wears on (fatigue). The single change is this time-dependent lapse, which both softens the model's near-certain predictions and produces the observed drift toward more yeses late in the session.

### bayesian_concepts_drifting_guess — rejected (experiment1 round 0 candidate 4 refine incumbent bigelow_piantadosi_interval)

**Outcome:** model binds ['trial_index'] — response-row bookkeeping a bare stimulus never carries — so it cannot be evaluated on a stimulus pool (the novelty gate now, the held-out evaluation later); compute everything from number_set and target.

**Hypothesis:** Refinement of bigelow_piantadosi_interval: people judge by the same Bayesian concept learning over rules and midpoint-weighted intervals, but on some trials they do not consult the concept posterior and instead guess, and the yes-rate of that guessing drifts over the session as time on task grows. The one change is this session-drifting guess process mixed into the response, which addresses the critique that yes responses become more frequent later in the session.

### periodic_spacing_concepts — rejected (experiment1 round 0 candidate 2 lens 2 repair 1)

**Outcome:** MCMC did not converge (max R-hat 1.143 > 1.05; min bulk ESS 19 < 100), too far from converging for smaller NUTS steps to help (the fitter refits only a near miss — at most 2% divergent transitions, R-hat <= 1.2, bulk ESS >= 20 — at target_accept 0.99), so raising target_accept will not help. Change the model's geometry instead: non-centred parameterisations for hierarchical or scale parameters, tighter (weakly informative) priors on parameters the data barely constrain, fewer weakly identified parameters (drop or merge parameters that trade off against each other), and no hard thresholds or discontinuities in the likelihood.

**Hypothesis:** People learn the hidden program's concept by Bayesian inference with the size principle, but besides named arithmetic rules and numeric ranges they also consider any regular spacing the examples suggest: "every d-th number starting from r" for any step d (e.g. 3, 7 suggests 3, 7, 11, 15, ...; 20, 32, 92 suggests every 12th number from 8), with shorter steps more plausible a priori. Targets that continue the examples' common spacing therefore get "yes" even when no named rule (multiple of k, ending digit) covers them.

### overestimated_randomness_concepts — dropped (experiment1 round 1)

**Outcome:** fold 0 too slow to fit: the fit of 'overestimated_randomness_concepts' (target_accept 0.99) was still sampling after the 30-minute limit and was stopped. Every sampling run of a candidate's admission fit has a 30-minute limit. Make the model cheaper to evaluate and easier to sample: vectorise the likelihood over trials (no Python loops, pytensor scan or per-trial subgraphs), compute features once per unique sequence (in compute_features or prepare_observed, not in the graph), and drop or merge parameters the data barely constrain — a weakly identified posterior makes NUTS take maximal-length trajectories.

**Hypothesis:** People learn the program's concept by ideal Bayesian inference over arithmetic rules and numeric intervals with the size principle, but they believe the program is far less reliable than it is: they think a sizeable, inferred share of its outputs are arbitrary numbers unrelated to its rule. Because any single example might be such a random output, it rules out the concepts that do not contain it only weakly, so people generalize broadly — saying "yes" to many unrelated targets — after one example, and tighten only as consistent examples accumulate.

### sampling_assumption_inference_intervals — dropped (experiment1 round 1)

**Outcome:** fold 0 too slow to fit: the fit of 'sampling_assumption_inference_intervals' (target_accept 0.99) was still sampling after the 30-minute limit and was stopped. Every sampling run of a candidate's admission fit has a 30-minute limit. Make the model cheaper to evaluate and easier to sample: vectorise the likelihood over trials (no Python loops, pytensor scan or per-trial subgraphs), compute features once per unique sequence (in compute_features or prepare_observed, not in the graph), and drop or merge parameters the data barely constrain — a weakly identified posterior makes NUTS take maximal-length trajectories.

**Hypothesis:** Refinement of `size_principle_strength_intervals`: people do the same Bayesian concept learning over the 42 arithmetic rules and the midpoint-weighted intervals of 1-100, but instead of applying a fixed partial size principle to every example, they are uncertain whether the examples were deliberately sampled from the concept (strong sampling) or merely happen to be consistent with it (weak sampling), and they infer this from the examples themselves. The one change replaces the per-example size-principle strength with this set-level sampling-assumption inference (as in Navarro et al., 2012): a single example is weak evidence that sampling is strong, so people generalize broadly from one example, while each further example that tightly fits a small concept makes strong sampling — and hence narrow, size-principle generalization — more credible.
