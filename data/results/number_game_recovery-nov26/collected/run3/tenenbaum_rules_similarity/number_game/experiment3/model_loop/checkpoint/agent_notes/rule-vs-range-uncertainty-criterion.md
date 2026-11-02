---
name: rule-vs-range-uncertainty-criterion
description: exp3 iter_1 c4: incumbent's criterion shift driven by binary rule-vs-range entropy instead of 9-kind entropy; smoke ll -3653.5
metadata:
  type: project
---
exp3 iter_1/candidate_4 (2026-10-31), refining kind_uncertainty_shifted_criterion: the only change is entropy = binary entropy (bits) of P(range) from the posterior, same shift·(H−0.5) readout. Smoke OK in ~15 s per run, ll −3653.5 (incumbent smoke −3653.3). Critique had 0/8 significant, so no PPC target; novelty vs the incumbent was not checked (risk: shift small, so it sits close to beta_share).
**How to apply:** if it gets rejected as a duplicate, coarser entropy measures have run out; try something other than the entropy readout. See [[kind-uncertainty-shifted-criterion]].
