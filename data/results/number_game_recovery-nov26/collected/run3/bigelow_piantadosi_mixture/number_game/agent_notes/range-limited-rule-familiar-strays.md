---
name: range-limited-rule-familiar-strays
description: exp3 iter_0 c4: rule∩range concepts on rule_count_familiar_strays; rho≈0.095 (free, not collapsing), +1.0 MAP, RMSE .009; equal rule prior makes it (set,subset,size) tables; p_yes eval gotcha
metadata:
  type: project
---
exp3 iter_0/candidate_4 (2026-11-01), range_limited_rule_familiar_strays. Motivated by the critique clean_rule_member_inside_minus_outside_hull (obs 0.18 vs null 0.10). Restricted rules = proper non-empty member blocks of each rule, weighted by # ranges cutting them (gap products), prior lambda_0*rho/(45*5050). rho_logit ~ N(-2,1). MAP ll -3320.79 vs incumbent -3321.79; rho 0.095 (unlike exp2, where it collapsed to the prior, see [[range-restricted-rules]]). Brute force check exact.
Speed: because rules are equally weighted, sum over rules into count tables, then fold the size axis in prepare_observed (count @ size^-j → [set|cell,16,5]) and multiply by e_sub*c. With dense (cells,16,100)*sub_size_lik the smoke fit took 236 s; folded, 21 s.
Gotcha: to evaluate p_yes at a MAP point, use pytensor.function(value_vars, model.replace_rvs_by_values([p_yes])[0]). A raw pytensor.function on p_yes samples the RVs, and compile_fn with a dict mis-maps. Both give garbage.
