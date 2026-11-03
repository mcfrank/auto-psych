---
name: exp3-iter3-place-value-similarity
description: exp3 iter_3 cand_1 similarity slot — GCM on (tens,units) grid, mean similarity, logistic; smoke loglik -4552 vs best ELPD -3520
metadata:
  type: project
---
Exp3 iter_3 candidate_1 (similarity brief): `place_value_grid_similarity` — exponential decay of attention-weighted city-block distance on tens/units grid, averaged over examples, logistic(bias+gain*sim). 4 params, 3 s smoke fit, loglik -4552 (~1000 nats behind concept learning, like other similarity models [[exp3-prototype-probe]]).
**Why:** similarity briefs always lose ~800-1000 nats; aim is a distinct, fast, valid model.
**How to apply:** for future similarity slots, don't expect fit gains; pick an untried metric and keep it cheap.
