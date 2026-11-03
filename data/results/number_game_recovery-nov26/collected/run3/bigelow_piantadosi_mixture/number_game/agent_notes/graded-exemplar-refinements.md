---
name: graded-exemplar-refinements
description: exp2 iter_3 c4: refining graded_exemplar_concept_retrieval — r_e collapses to r; familiar strays +1.3 MAP (taken by sibling c3); submitted first-example anchor (+0.06 MAP)
metadata:
  type: project
---
exp2 iter_3 candidate_4 (2026-11-01), refining graded_exemplar_concept_retrieval (MAP ll -2030.47).
- Separate endorsement gradient r_e (range cells tabulated by target distance d, exact in r_e): r_e≈r≈0.016, MAP ll -2030.47. No gain and a near-duplicate, so it was dropped.
- Built graded_exemplar_familiar_strays, but sibling c3 did the same graft with an inferred mix, so I dropped it. My version had The stray density q(x)=(1-m)/100+m*mean_rules 1[x∈rule]/|rule|, with m FIXED at 0.26. It is exact with tabulation: use w_i=s_i/q_i in place of 100 s_i, folded into ex_rule_dcount and the range tables. Zero new params. MAP -2029.15 (+1.3); m=0.15: -2029.28, m=0.4: -2029.72. p_yes RMSE vs the incumbent is .012 (max .09). Smoke OK, 11 s/200 iters.
- Gotcha: to get p_yes at MAP, use compile_fn(model.replace_rvs_by_values([p_yes])[0], inputs=model.value_vars, point_fn=True, on_unused_input="ignore"). A hand-built pytensor.function gives inputs in the wrong order.
- Submitted graded_exemplar_first_example_anchor. The anchor is exact: (a1 w1 + 1 - a1) times e_j over the later examples, and the range tables are split into two parts that are linear in a1. MAP -2030.41 (+0.06), a1≈.90≈alpha, RMSE .016. Smoke OK. Expect an ELPD tie at best.
**How to apply:** the incumbent is saturated on the reliability, endorsement and anchor axes. for an inferred mix you would need a 2-D (r, m) grid. See [[graded-exemplar-uncertain-reliability]], [[familiar-strays-family-level]].
