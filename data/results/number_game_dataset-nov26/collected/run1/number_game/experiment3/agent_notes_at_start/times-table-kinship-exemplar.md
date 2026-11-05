---
name: times-table-kinship-exemplar
description: exp2 iter2 c2: multiples route replaced by shared-factor>=3 kinship decaying with smallest shared factor; smoke -2706.3
metadata:
  type: project
---
Exp2 iter_2 candidate_2 (2026-11-02): `times_table_kinship_exemplar`. It is [[multiple-route-exemplar]] with the additive route changed to "shares a factor >=3 with the example" (t != e). Its weight is multiplied by exp(-table_decay*log(d/3)), where d is the smallest factor >=3 of gcd(t,e). 11 params. Smoke log-lik -2706.3 (100/100, 1 chain).
**Why:** aimed at the two critiques that survive FDR in [[exp2-iter2-critique-stats]]: factor-kin non-multiples after lone examples, and far multiples of two-digit examples being endorsed less.
**How to apply:** if it is admitted, check whether table_decay > 0 and whether it beats the incumbent by LOO.
