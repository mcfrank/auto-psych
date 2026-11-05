---
name: mixture-variant-mle-screen
description: Quick offline MLE screen (round 0) of one-change variants of bigelow_piantadosi_mixture — which changes help most
metadata:
  type: project
---
Round 0, cell_1 (2026-11-01): in-sample NLL of mixture variants on 2520 responses: base 1840, free noise alpha 1796, tempered size principle (likelihood^beta, beta≈0.52) 1686, 50/50 lapse (~41%) 1705, Erlang interval-size prior 1840 (no help), temper+lapse 1676. Incumbent independent ELPD -1778. Submitted tempering as candidate_5 `tempered_size_principle_mixture`.

**Why:** saves re-screening; tempering/lapse are the big levers. **How to apply:** later refinements could try lapse on top of tempering, or tempering on the independent model.

Round 1 screen (same 2520 responses, NLL): tempered base 1686.0; tempered+decision gain/bias 1675.2; +target-confusion smoothing 1686.0 (none); +lapse 1674.8; fuzzy interval edges 1685.5 (none); **memory-limited (each example kept w.p. r≈0.71, average over remembered subsets), no tempering: 1701.6**; memory+temper 1684.8. Exemplar incumbent ~ELPD -1624. Bayesian variants stay ~60+ nats behind exemplar; graded-similarity-like structure seems missing from the hypothesis space. Submitted memory model as iter_1 candidate_1 `memory_limited_concept_learning`.
