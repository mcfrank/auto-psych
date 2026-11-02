# Tried before

5 hypotheses proposed earlier in this project are no longer in the model set. Read them before you propose:

- A *pruned* entry entered the set and later lost to the best model by the stated margin, on the data available then. Its mechanism may be partly right: a model that changes it substantively is welcome, but do not re-propose it unchanged or merely re-parameterised, under any name. Pruned models stay readable under `models/pruned/`.
- A *rejected* entry never entered the set. If it was a near-duplicate of a model still in the set, that region is already covered: do not re-propose it. If it failed on its code or its fit (see its outcome), the idea itself was never tested and a correct implementation may be worth trying.

### serial_position_weighted_concepts — rejected (experiment1 round 0 candidate 2 lens 2)

**Outcome:** predicts like existing model 'bigelow_piantadosi_independent' (p_yes RMSE 0.00147 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of bigelow_piantadosi_independent, not a new hypothesis.

**Hypothesis:** People infer the hidden concept by Bayesian generalization over rules and intervals, but they do not weigh the examples equally: how much an example counts as evidence depends on where it came in the order shown. One serial-position gradient, inferred from the data, can favour the first examples (primacy) or the most recent ones (recency). So a concept that fits the heavily weighted examples is endorsed even if it fits the others poorly, and the same set shown in a different order would give a different pattern of generalization.

### partial_size_principle_concepts — rejected (experiment1 round 0 candidate 3 refine incumbent bigelow_piantadosi_independent)

**Outcome:** too slow to fit: the fit of 'partial_size_principle_concepts' (target_accept 0.99) was still sampling after the 30-minute limit and was stopped. Every sampling run of a candidate's admission fit has a 30-minute limit. Make the model cheaper to evaluate and easier to sample: vectorise the likelihood over trials (no Python loops, pytensor scan or per-trial subgraphs), compute features once per unique sequence (in compute_features or prepare_observed, not in the graph), and drop or merge parameters the data barely constrain — a weakly identified posterior makes NUTS take maximal-length trajectories.

**Hypothesis:** Refinement of `bigelow_piantadosi_independent`: the same Bayesian concept learner over its 45 rules and all intervals of 1-100, with a prior weight inferred for each rule, but people apply the size principle only partially — each example's likelihood under a concept scales with the concept's size raised to an inferred exponent (1 = the incumbent's strong sampling, 0 = weak sampling that ignores concept size). The single change is this free size-principle strength, which governs how quickly a few examples make people favour small, specific concepts over broad ones.

### conservative_evidence_weighting — rejected (experiment1 round 1 candidate 1 lens 4)

**Outcome:** predicts like existing model 'bigelow_piantadosi_independent' (p_yes RMSE 0.00116 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of bigelow_piantadosi_independent, not a new hypothesis.

**Hypothesis:** Refinement of `bigelow_piantadosi_independent`: people do the same Bayesian concept learning over the 45 rules and all intervals of 1-100 (size principle, a prior weight inferred per rule), but they are conservative updaters who under-weight (or over-weight) the evidence of the examples — the examples' likelihood enters the posterior raised to a single inferred evidence-weight exponent (1 = normative; below 1 = conservatism, so a few examples move belief less than they should and generalization stays broad; above 1 = over-reaction). The one change is this mis-weighting of the evidence relative to the prior, which predicts that adding examples sharpens generalization more slowly (or faster) than Bayes prescribes.

### size_principle_tempered_concepts — rejected (experiment1 round 1 candidate 4 refine incumbent bigelow_piantadosi_independent)

**Outcome:** predicts like existing model 'bigelow_piantadosi_independent' (p_yes RMSE 0.00153 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of bigelow_piantadosi_independent, not a new hypothesis.

**Hypothesis:** Refinement of `bigelow_piantadosi_independent`: people do the same Bayesian concept learning over its 45 rules (each with its own inferred prior weight) and all intervals of 1-100, but they weigh the examples' evidence with an inferred strength rather than as an ideal sampler — the likelihood of the examples under each concept (the size principle) is raised to a free exponent, so values below 1 mean people generalise more broadly than the examples warrant (weaker sampling assumptions) and values above 1 mean they commit to the smallest consistent concept faster. The single change is this evidence-weighting exponent, which governs how sharply additional examples narrow generalisation, e.g. from "even numbers" to "powers of 2" after 16, 8, 64.

### decade_aligned_interval_prior — rejected (experiment1 round 1 candidate 4 refine incumbent bigelow_piantadosi_independent repair 1)

**Outcome:** predicts like existing model 'bigelow_piantadosi_independent' (p_yes RMSE 0.00177 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of bigelow_piantadosi_independent, not a new hypothesis.

**Hypothesis:** Refinement of `bigelow_piantadosi_independent`: people do the same Bayesian concept learning over its 45 rules (each with its own inferred prior weight) and all intervals of 1-100 with the size principle, but they think about magnitude in decades, so ranges that line up with decade boundaries ("the 60s", "1 to 20", "the 30s and 40s") are a priori more plausible than arbitrary ranges like 57-73; the boost for decade-aligned intervals is inferred, with the intervals' total prior kept fixed. The single change is this decade-aligned interval prior, which predicts sharp drops in generalisation at decade boundaries (after 61, 64, 67, a target of 69 is endorsed but 71 much less so) rather than the smooth magnitude falloff of the incumbent.
