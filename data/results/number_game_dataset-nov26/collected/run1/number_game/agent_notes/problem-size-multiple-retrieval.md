---
name: problem-size-multiple-retrieval
description: exp2 iter3 c1: multiples route decays with factor sum (problem-size effect) on factor_kin_multiple_route, +5 nats MAP
metadata:
  type: project
---
exp2 iter3 candidate_1 `problem_size_multiple_retrieval`: incumbent factor_kin_multiple_route with the multiples route scaled by exp(-size_decay*(e + t/e - 4)/10). Targets the critique stat large_example_multiple_contrast. MAP -2677.4 vs incumbent -2682.5 (+5 nats, 1 param); smoke fit -2683.4, check OK.
Related: [[times-table-multiple-route]], [[factor-kin-multiple-route]], [[exp2-iter3-critique-stats]].
