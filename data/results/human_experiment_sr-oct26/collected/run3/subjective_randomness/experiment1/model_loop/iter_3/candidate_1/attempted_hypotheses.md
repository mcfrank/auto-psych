# Tried before

2 hypotheses proposed earlier in this project are no longer in the model set. Read them before you propose:

- A *pruned* entry entered the set and later lost to the best model by the stated margin, on the data available then. Its mechanism may be partly right: a model that changes it substantively is welcome, but do not re-propose it unchanged or merely re-parameterised, under any name. Pruned models stay readable under `models/pruned/`.
- A *rejected* entry never entered the set. If it was a near-duplicate of a model still in the set, that region is already covered: do not re-propose it. If it failed on its code or its fit (see its outcome), the idea itself was never tested and a correct implementation may be worth trying.

### negative_recency_expectation_fit — rejected (experiment1 round 0 candidate 0 lens 0)

**Outcome:** MCMC did not converge (401 divergent transitions of 12000), too far from converging for smaller NUTS steps to help (the fitter refits only a near miss — at most 2% divergent transitions, R-hat <= 1.2, bulk ESS >= 20 — at target_accept 0.95), so raising target_accept will not help. Change the model's geometry instead: non-centred parameterisations for hierarchical or scale parameters, tighter (weakly informative) priors on parameters the data barely constrain, fewer weakly identified parameters (drop or merge parameters that trade off against each other), and no hard thresholds or discontinuities in the likelihood.

**Hypothesis:** People judge randomness by reading each sequence flip by flip and checking how well every flip matches a gambler's-fallacy expectation: after a streak, they expect the coin to switch, increasingly so the longer the streak has run. A sequence looks random to the extent its flips conform to this negative-recency expectation, with the most recently read flips (near the end of the sequence) weighing more heavily, and people differ in how strongly this conformity drives their choice.

### goldilocks_gamblers_surprise — rejected (experiment1 round 1 candidate 5 refine chosen)

**Outcome:** too slow to fit: the fit of 'goldilocks_gamblers_surprise' (target_accept 0.8) was still sampling after the 30-minute limit and was stopped. Every sampling run of a candidate's admission fit has a 30-minute limit. Make the model cheaper to evaluate and easier to sample: vectorise the likelihood over trials (no Python loops, pytensor scan or per-trial subgraphs), compute features once per unique sequence (in compute_features or prepare_observed, not in the graph), and drop or merge parameters the data barely constrain — a weakly identified posterior makes NUTS take maximal-length trajectories.

**Hypothesis:** Refinement of `recency_weighted_gamblers_surprise`: people still read each sequence flip by flip with a gambler's-fallacy expectation (the longer the current run, the more they expect it to break), with recent flips weighing more, but a sequence looks random when its felt surprise is close to the moderate level they expect from a real coin, not when it is minimal. The single change is that randomness is the closeness of the recency-weighted surprise to a fitted "just-right" level rather than its plain absence, so sequences that are too predictable under the gambler's expectation — above all perfect alternation, which confirms every predicted reversal — look contrived, addressing the critique that the incumbent picks perfectly alternating sequences more often than people do.
