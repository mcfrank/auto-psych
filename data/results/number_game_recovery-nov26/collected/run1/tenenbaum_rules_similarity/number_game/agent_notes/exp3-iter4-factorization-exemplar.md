---
name: exp3-iter4-factorization-exemplar
description: exp3 iter_4 cand_1 exemplar slot — GCM with magnitude + log(lcm/gcd) factorisation distance, smoke loglik -4580
metadata:
  type: project
---
Exp3 iter_4 candidate_1 (exemplar/prototype brief, 2026-11-02): `factorization_exemplar_similarity` — mean exp(-c*((1-w)|t-x|/10 + w*log(lcm/gcd)/log100)) over examples, logistic(bias+gain*sim). 4 params, 3 s smoke fit, loglik -4580 (similar to other similarity models, ~1000 nats behind). Submitted.
**How to apply:** untried similarity metrics left: none needed; similarity slots just need a distinct cheap valid model. See [[exp3-iter3-place-value-similarity]].
