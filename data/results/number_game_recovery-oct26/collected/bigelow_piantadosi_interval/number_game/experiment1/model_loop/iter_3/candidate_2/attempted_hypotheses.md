# Tried before

4 hypotheses proposed earlier in this project are no longer in the model set. Read them before you propose:

- A *pruned* entry entered the set and later lost to the best model by the stated margin, on the data available then. Its mechanism may be partly right: a model that changes it substantively is welcome, but do not re-propose it unchanged or merely re-parameterised, under any name. Pruned models stay readable under `models/pruned/`.
- A *rejected* entry never entered the set. If it was a near-duplicate of a model still in the set, that region is already covered: do not re-propose it. If it failed on its code or its fit (see its outcome), the idea itself was never tested and a correct implementation may be worth trying.

### inferred_noise_concept_learning — rejected (experiment1 round 1 candidate 4 refine incumbent concept_learning_with_guessing)

**Outcome:** predicts like existing model 'example_noise_concept_learning' (p_yes RMSE 0.00131 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of example_noise_concept_learning, not a new hypothesis.

**Hypothesis:** Refinement of concept_learning_with_guessing: people do the same Bayesian concept learning over 42 weighted rules and all intervals of 1-100 (size principle), with the same guessing on a fraction of trials, but the share of examples they believe are random noise rather than drawn from the concept is not fixed at 10% — it is a property of their reasoning that we infer. The single change is this inferred noise belief: the more examples people treat as possibly random, the more readily a rule survives an example that violates it and the weaker the size principle's pull toward the tightest concept, changing generalization on sets with an odd-one-out example.

### inferred_outlier_rate_concept_learning — rejected (experiment1 round 1 candidate 5 refine chosen)

**Outcome:** predicts like existing model 'inferred_example_noise_belief' (p_yes RMSE 0.00166 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of inferred_example_noise_belief, not a new hypothesis.

**Hypothesis:** Refinement of bigelow_piantadosi_independent: people do the same Bayesian concept learning over the 42 weighted rules and all intervals of 1-100 with the size principle, but how often they believe an example is a stray, random number rather than a true member of the concept is not fixed at 10% — it is a property of the learner, inferred from the data. The one change is this inferred outlier rate: a high rate lets a concept that fits most (but not all) of the examples keep posterior weight and weakens how sharply the examples narrow generalization, whereas a low rate makes a single misfit example rule a concept out.

### sublinear_evidence_accumulation — rejected (experiment1 round 2 candidate 4 refine incumbent tempered_evidence_concept_guessing)

**Outcome:** predicts like existing model 'redundant_examples_tempered_concepts' (p_yes RMSE 0.00130 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of redundant_examples_tempered_concepts, not a new hypothesis.

**Hypothesis:** Refinement of tempered_evidence_concept_guessing: people do the same Bayesian concept learning over 42 weighted rules and all intervals of 1-100 with tempered evidence and occasional guessing, but the weight each example's evidence carries depends on how many examples they were shown — evidence accumulates with diminishing (or accelerating) returns rather than at a fixed rate per example. The one change is this inferred set-size dependence of the evidence weight: with diminishing returns, a single example already sharpens the concept strongly while a third or fourth example adds little, so people generalize more narrowly from one example and more broadly from four than a fixed per-example weight predicts; no dependence recovers the incumbent.

### tempered_evidence_inferred_noise — rejected (experiment1 round 2 candidate 4 refine incumbent tempered_evidence_concept_guessing repair 1)

**Outcome:** too slow to fit: the fit of 'tempered_evidence_inferred_noise' (target_accept 0.99) was still sampling after the 30-minute limit and was stopped. Every sampling run of a candidate's admission fit has a 30-minute limit. Make the model cheaper to evaluate and easier to sample: vectorise the likelihood over trials (no Python loops, pytensor scan or per-trial subgraphs), compute features once per unique sequence (in compute_features or prepare_observed, not in the graph), and drop or merge parameters the data barely constrain — a weakly identified posterior makes NUTS take maximal-length trajectories.

**Hypothesis:** Refinement of tempered_evidence_concept_guessing: people do the same Bayesian concept learning over 42 weighted rules and all intervals of 1-100, with tempered evidence (each example counts for an inferred fraction or multiple of a full draw) and occasional guessing, but how often they believe an example is a stray random number rather than a member of the concept is inferred from the data instead of fixed at 10%. The one change, taken from example_noise_concept_learning, is this inferred example-noise rate: tempering sets how sharply consistent concepts are favoured by their size, while the noise rate separately sets how readily a concept survives an example that breaks it (e.g. "ends in 2" after 92, 82, 27, 42), so sets with an outlier generalize differently from what either change alone predicts.
