---
name: exp3-prototype-probe
description: Exp3 iter_0 exemplar-brief probe: attention-weighted set prototype MAP LL -4466 vs constant -4617 (best concept model ELPD -3524)
metadata:
  type: project
---
Experiment 3 iter_0 candidate_2 (2026-11-01, exemplar/prototype brief): prototype = per-set share of examples having each of the 45 rule properties (attention = share^gamma * log(100/|f|)) plus mean magnitude; p = sigmoid(a - b_f*attn-weighted share of prototype properties the target lacks - b_m*|t-mean|/(1+k*SD)). MAP LL -4466 (constant rate -4617 on 6750 rows; best concept model ELPD -3524), gamma ~7.3 (attention collapses onto properties all examples share), k ~0.2. Gaussian magnitude falloff is worse (-4511). The first parameterisation (base_tol + spread) had a weight/tolerance ridge, fixed by using 1 + k*spread. Smoke check OK in 23 s. Submitted as attention_weighted_set_prototype.
**How to apply:** prototype/similarity models stay ~900 nats behind concept learning on exp3 too; don't spend more effort tuning them.
