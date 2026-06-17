# Theory Report — Experiment 2

## length_sensitive_alternation

**Hypothesis:** People evaluate how close a sequence is to their alternation prototype by comparing the *count* of transitions (not the proportion) to the ideal count — so longer sequences provide a proportionally stronger signal when they deviate from the prototype.

**Motivation:** Experiment 1's dominant model (inner_loop_model / iter1_candidate0) operated on the alternation *proportion* p_alts = alts / (n-1). But this commits to a specific view: deviations are comparable across sequence lengths on the proportion scale. The experiment 1 stimuli included pairs with different lengths (4 vs 6, 6 vs 8, etc.), and the proportion-scale model won only modestly over the fixed-at-0.5 model (iter1_candidate2, ELPD diff = 2.08, within 2·dse). If length matters for how strongly deviations register, the count-scale model should outperform the proportion-scale one on stimuli that pair sequences of different lengths.

**Mechanism:** The model computes an ideal alternation count as `theta_alt * (n - 1)` for each sequence (same learned prototype, applied to the sequence's own length), then penalizes the squared deviation of the observed count from that ideal. This is mathematically distinct from the proportion-scale model: for two sequences with the same p_alts but different lengths, the proportion model gives them equal scores while this model gives the longer sequence a larger deviation penalty. The only free cognitive parameters are the prototype location `theta_alt`, the temperature `beta`, and a side bias.

---

## bayesian_markov_fairness

**Hypothesis:** People approximate Bayesian observers who compare each sequence's observed transitions against a fair Markov chain (p_transition = 0.5) versus a biased alternative, choosing whichever sequence is more consistent with fair-coin generation.

**Motivation:** All 8 experiment 1 models are heuristic: they score sequences on hand-crafted summary statistics (alternation proportion, max run, imbalance). None implement the Griffiths (2001) Bayesian account in which subjects reason about how likely a sequence is under a generative model. The model that accumulated most of the posterior mass (iter1_candidate0) is a prototype-similarity heuristic, not a likelihood-based account. Adding a Bayesian competitor tests whether the heuristic outperforms principled inference, or whether the two are empirically indistinguishable.

**Mechanism:** For each sequence, the model computes the log-Bayes-factor comparing the observed alternation count under a fair Markov model (p_transition = 0.5) against a biased alternative (p_transition = theta_biased, estimated from data). A positive LBF means the sequence is more consistent with the fair model. The decision uses a softmax over LBF_A − LBF_B. This is a distinct hypothesis from all existing models: it uses the *full count information* (alts and n-1 jointly as log-likelihoods, not their ratio), and its scoring function is non-linear in a way that emerges from Bayes' theorem rather than from a chosen similarity metric.
