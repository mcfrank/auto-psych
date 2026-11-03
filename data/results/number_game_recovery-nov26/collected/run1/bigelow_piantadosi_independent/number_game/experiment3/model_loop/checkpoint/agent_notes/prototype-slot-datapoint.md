---
name: prototype-slot-datapoint
description: Exemplar/prototype-brief data point (exp3 iter_4/cand_1) — naive-Bayes single-cluster prototype fits fast but ~750 nats short of the rule/range learner
metadata:
  type: project
---
2026-10-31, experiment3 iter_4/candidate_1 (exemplar/prototype brief): independent_feature_prototype = one-cluster rational-model category density (Student-t on log magnitude normalised over 1..100, Dirichlet final digit and parity), sigmoid(a + s*log typicality ratio). 5 params, prepare_observed per set; check OK, 5 s sampling, smoke log-lik -3349.7 (incumbent ELPD about -2598; isolation exemplar -3417).
**Why:** similarity/prototype accounts without a set-level size principle over rules lose by hundreds of nats; see also [[interval-prefix-sum-speedup]] (nearest-radius concepts -2967).
**How to apply:** for later exemplar/prototype briefs, expect a large deficit; it is cheap to build, so don't spend time on speedups.
