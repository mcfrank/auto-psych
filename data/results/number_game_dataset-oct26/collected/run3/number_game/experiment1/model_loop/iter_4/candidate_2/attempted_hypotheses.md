# Tried before

10 hypotheses proposed earlier in this project are no longer in the model set. Read them before you propose:

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

### recency_forgetting_concepts — rejected (experiment1 round 2 candidate 0 lens 6)

**Outcome:** too slow to fit: the fit of 'recency_forgetting_concepts' (target_accept 0.99) was still sampling after the 30-minute limit and was stopped. Every sampling run of a candidate's admission fit has a 30-minute limit. Make the model cheaper to evaluate and easier to sample: vectorise the likelihood over trials (no Python loops, pytensor scan or per-trial subgraphs), compute features once per unique sequence (in compute_features or prepare_observed, not in the graph), and drop or merge parameters the data barely constrain — a weakly identified posterior makes NUTS take maximal-length trajectories.

**Hypothesis:** People learn the program's concept by conservative Bayesian concept learning over arithmetic rules and numeric intervals (size principle), but they reason only from the examples they still hold in memory when the target is asked: the last example shown is always remembered, while earlier examples drop out of memory with a probability that grows the further back in the sequence they were shown. A forgotten example neither rules out concepts nor sharpens them, so generalization is driven mostly by the most recent examples, and reordering the same examples changes which targets get a "yes" (targets that fit only the first example are endorsed less often than ones fitting only the last).

### session_drifting_evidence_concepts — rejected (experiment1 round 2 candidate 3 refine incumbent sample_size_growing_evidence_concepts)

**Outcome:** too slow to fit: the fit of 'session_drifting_evidence_concepts' (target_accept 0.99) was still sampling after the 30-minute limit and was stopped. Every sampling run of a candidate's admission fit has a 30-minute limit. Make the model cheaper to evaluate and easier to sample: vectorise the likelihood over trials (no Python loops, pytensor scan or per-trial subgraphs), compute features once per unique sequence (in compute_features or prepare_observed, not in the graph), and drop or merge parameters the data barely constrain — a weakly identified posterior makes NUTS take maximal-length trajectories.

**Hypothesis:** Refinement of `sample_size_growing_evidence_concepts`: people do the same conservative Bayesian concept learning over the 42 arithmetic rules and midpoint-weighted intervals of 1-100, with evidence weight growing with the number of examples, but how much weight they give the examples' evidence drifts over the session — as the task wears on, people scrutinise the examples less and so hold a flatter, broader posterior over concepts. The one change is that the evidence strength is scaled by an exponential drift in the trial's position in the session (rate inferred), which predicts more liberal, broader generalization (more "yes") late in the session, the discrepancy the incumbent's critique flags.

### session_drift_liberal_concepts — rejected (experiment1 round 2 candidate 4 refine incumbent sample_size_growing_evidence_concepts)

**Outcome:** model binds ['trial_index'] — response-row bookkeeping a bare stimulus never carries — so it cannot be evaluated on a stimulus pool (the novelty gate now, the held-out evaluation later); compute everything from number_set and target.

**Hypothesis:** Refinement of `sample_size_growing_evidence_concepts`: people do the same conservative Bayesian concept learning over the 42 arithmetic rules and midpoint-weighted intervals of 1-100, with evidence weight growing with set size, but as the session wears on they grow more liberal, increasingly accepting targets their concept does not contain. The one change is a "yes"-drift: with a probability that changes smoothly (logistically, slope inferred) with how far into the session the trial is, people answer "yes" regardless of the concept, which predicts the rising yes-rate across the session that the incumbent's critique flags (and reduces to a constant yes-bias if the slope is zero).

### session_drifting_yes_bias_concepts — rejected (experiment1 round 2 candidate 3 refine incumbent sample_size_growing_evidence_concepts repair 1)

**Outcome:** model binds ['trial_index'] — response-row bookkeeping a bare stimulus never carries — so it cannot be evaluated on a stimulus pool (the novelty gate now, the held-out evaluation later); compute everything from number_set and target.

**Hypothesis:** Refinement of `sample_size_growing_evidence_concepts`: people do the same conservative Bayesian concept learning over the 42 arithmetic rules and midpoint-weighted intervals of 1-100, with evidence weight growing with the number of examples, but their willingness to say "yes" given that belief drifts over the session — as the task wears on, people become more liberal and endorse targets they are only partly confident about. The one change is a response bias on the log-odds of the posterior belief that shifts linearly with the trial's position in the session (rate inferred, zero at mid-session), which predicts a higher yes-rate late in the session than early, the discrepancy the incumbent's critique flags.
