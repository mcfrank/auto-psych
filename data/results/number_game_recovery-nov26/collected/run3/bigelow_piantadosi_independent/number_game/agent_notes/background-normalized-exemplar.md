---
name: background-normalized-exemplar
description: exp3 iter4 GCM exemplar sim / mean sim to 1-100 baseline, smoke loglik -4382 (~900 behind)
metadata:
  type: project
---
Exp3 iter4 cand1 (exemplar brief): `background_normalized_exemplar_familiarity` — GCM similarity over magnitude + 15 binary props + final/tens digit, p=sigmoid(a+b*log(S_ex/S_bg)). 4 params, 5 s smoke fit, smoke loglik -4382.4, roughly as bad as [[minerva-echo-exemplar]].
**Why:** the exemplar family stays ~900 nats behind the concept learners in exp3.
**How to apply:** don't expect exemplar slots to compete; aim for a cheap, valid model.
