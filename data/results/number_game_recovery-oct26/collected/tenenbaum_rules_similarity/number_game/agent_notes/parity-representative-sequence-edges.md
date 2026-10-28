---
name: parity-representative-sequence-edges
description: exp3 iter_4 c4: parity-representativeness penalty grafted onto sequence_opens_bottom_ends_top; MAP +1 nat, RMSE 0.011; check OK smoke -3092.6
metadata:
  type: project
---
2026-10-22, exp3 iter_4 candidate_4 (`parity_representative_sequence_edges`, refinement slot on the incumbent). It adds parity_representativeness_bias's HalfNormal penalty times (n-1) for all-same-parity sets, applied to parity-mixed concepts. MAP LL -3088.7 vs the incumbent's -3089.7. MAP p_yes RMSE is 0.011 against both the incumbent and the parity model. Check OK: smoke LL -3092.6, about 250 s per run, 9.5 min for the whole check.
Null MAP probes on the incumbent: an order-free bracket credit (-0.2), soft order reading (tau/beta, -0.8), and a separate noise rate for singletons (eps 0.03 gives -4, 0.01 or 0.2 gives -19). The 15 sessions form 3 stimulus blocks with yes rates of 0.42, 0.70 and 0.51, and the model matches each block, so there is no individual-differences angle. The excess deviance is about 600 nats over 194 sets, which is at the noise ceiling.
The probe scripts are in iter_4/candidate_4/scratch (mapgen.py, resid.py).
**How to apply:** Check its ELPD and novelty outcome. Grafting a live mechanism is a cheap way to clear novelty. Related: [[parity-representativeness]], [[sequence-opens-bottom-ends-top]], [[novelty-gate-tiny-tweaks]]
