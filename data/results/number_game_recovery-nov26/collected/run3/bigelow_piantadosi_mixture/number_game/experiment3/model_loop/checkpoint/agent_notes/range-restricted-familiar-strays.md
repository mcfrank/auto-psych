---
name: range-restricted-familiar-strays
description: exp3 iter_0 c3: runs of consecutive rule members as concepts on rule_count_familiar_strays; rho≈0.06, +0.66 MAP, RMSE 0.16 (distinct)
metadata:
  type: project
---
exp3 iter_0/candidate_3 (2026-11-01). This adds restricted rules to the incumbent rule_count_familiar_strays, motivated by the critique clean_rule_member_inside_minus_outside_hull (q=.29). A restricted rule is a contiguous run of a rule's members, excluding the full rule. Its prior is rho*lambda_0, split equally over rules and then equally over each rule's runs (6890 runs in total), with rho ~ Beta(2,18).
- MAP ll: -3321.13 vs -3321.79 for the incumbent. rho = 0.063.
- p_yes RMSE vs the incumbent is 0.11-0.16 (find_MAP varies between runs). NUTS posterior: rho .08±.04. That is large because odd-number sets (5,77,89) switch to "odd from 5 to 89", and the even targets drop from 0.65 to 0.19 when the data say 0.6.
- Exp3 data give restricted rules more support than exp2 did: there the share collapsed to 0.014 ([[range-restricted-rules]]).
Speed: the first version, with dense (sets, 16, 100 sizes) tables, took 150 s per smoke run. Pre-summing the tables over size with @ _SIZE_POW to (sets/cells, 16, j) and multiplying by e_sub*c_set gives 5.9 ms/grad, 6 NUTS steps/iter and a 22 s smoke run. Earlier note on the layout: with an equal rule prior, the restricted-rule prior mass can be tabulated as dense (sets, 16 subsets, 100 sizes) and (cells, 16, 100) tables, reusing the incumbent's sub_size_lik.
find_MAP lands at different points between runs (saddles), so compare with NUTS.
