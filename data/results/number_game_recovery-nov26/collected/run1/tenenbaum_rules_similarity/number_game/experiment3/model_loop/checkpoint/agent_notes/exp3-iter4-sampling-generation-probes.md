---
name: exp3-iter4-sampling-generation-probes
description: exp3 iter_4 cand_2 MAP probes on an equal-prior uniform-range base (MAP -3514.61) — prior-sample generation, enumeration sampling and a hard first-example anchor are dead; set-size decisiveness gain submitted
metadata:
  type: project
---
Probes from 2026-11-02, on the equal-prior base with uniform ranges and no round edges (MAP -3514.61; scale 75, dsum_decay -0.25):
- Importance sampling from the prior, where a program comes to mind with prob 1-exp(-K*prior): K≈88 but only +0.01 nats. K trades off against the kind weights, so it is unidentified.
- Rules emit members by enumeration order (weight rank^-b): b = -0.013, sharply pinned. LL is -18 nats at b=±0.15 and -75 at ±0.3, so it would be a near-duplicate. People expect rules to sample their members uniformly.
- Hard first-example anchor (a concept must contain the first example; later examples may be noise): -1257 nats. Dead.
- Submitted evidence_quantity_decisiveness. It multiplies the logit gain by exp(g*(n-1)), with g=0.022. MAP is -3513.40 (+1.2), and pool RMSE against the base is about 0.005 at MAP. exp3 has no 1-example sets; sizes are 2, 3 and 4. The check passed, at 32 s for 100+100 draws.
- Scripts: /tmp/map_c2.py (MAP plus LL) and /tmp/prof_c2.py (profile one parameter, then pool RMSE). Neither persists after the session.
**How to apply:** don't propose prior-distortion or within-rule sampling-shape mechanisms. See [[exp3-iter1-process-probes]] and [[exp3-iter2-novelty-gate-map-vs-posterior]].
