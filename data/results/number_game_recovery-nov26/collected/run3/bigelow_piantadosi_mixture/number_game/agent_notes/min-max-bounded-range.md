---
name: min-max-bounded-range
description: exp3 iter_2 c2: extra prior on the exact min–max range (n>=2) on range_limited incumbent: boost≈7, +1.7 MAP, RMSE .009–.012 vs base/cluster-cued; narrow-span gap barely moves
metadata:
  type: project
---
exp3 iter_2/candidate_2 (2026-11-02), min_max_bounded_range. Base is range_limited_rule_familiar_strays. One extra concept is added: the range from the smallest to the largest example (n>=2), with prior boost·(1-λ)/100². Its evidence is sub_size_lik[set, full subset, span].
- MAP ll: -3319.10, vs -3320.79 for the base and -3318.97 for cluster-cued. boost = 7.3.
- Training p_yes RMSE: .009 vs the base, .012 vs cluster-cued. So it is distinct.
- Critique stat (narrow-span in-hull rule non-members): observed .578. The base predicts .407, this model .430, cluster-cued .436. Range-prior levers keep moving it by only about .02.
- Check OK; smoke fit 31 s.
- Data hint: fill-in looks local, at non-members next to an example (17, 19 after 16; 78–88 after 77, 89; 7, 11, 13, 15 after 4, 16, 12). The set 31,3,1,15 → 2, 4, 16 may be a powers-of-2 association.
See [[cluster-cued-range-salience]], [[variable-rule-range-expectation]].
