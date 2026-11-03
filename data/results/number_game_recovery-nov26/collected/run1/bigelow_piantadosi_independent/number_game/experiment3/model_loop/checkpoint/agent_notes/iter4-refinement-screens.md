---
name: iter4-refinement-screens
description: MAP screens on salient_contained_digit_rules (exp3 iter_4/cand_5) — recognition and residue progressions null; per-person criterion +9 nats
metadata:
  type: project
---
2026-10-31, experiment3 iter_4/candidate_5 (free refinement slot). MAP log-lik on the incumbent salient_contained_digit_rules (base -2536.2): example recognition (p = r*is_ex + (1-r)*belief) -2534.7 (r 0.2, +1.5); residue-class progressions (r mod m, own kappa_p) -2536.4 (null, kappa_p -3.5); per-participant log-odds criterion (non-centred, sd 0.76) -2527.1 (+9). Also at MAP alpha (rule-plus-exception share) is about 0.001, so the exception machinery barely matters for this learner.
Same round: candidate_3 did incumbent + range-plus-exception; candidate_4 did odds compression, special-number salience and decade rules.
Submitted individual_response_criteria: participants from trial_index resets ([[participant-segmentation]]); rows with no trial_index (novelty pool) get a Gauss-Hermite marginal over criteria, so the bare-row prediction is not identical to the incumbent's.
**Why:** the novelty pool uses bare stimulus rows; a per-person effect with a 0 default would equal the incumbent and be rejected.
**How to apply:** screen with iter_4/candidate_5/scratch/map/{mk.py,mkH.py,map.py} (pass the responses CSV as an absolute path). Related: [[decision-rule-map-screen]], [[interval-prefix-sum-speedup]].
Repair (iter_4/candidate_5_repair_1): the first submission was rejected because the Hessian at the posterior mode was not positive definite. Diagnosis (with /tmp/diag5/diag.py, which runs the same L-BFGS mode search plus a Hessian eigendecomposition): L-BFGS fell into the centred-hierarchy funnel (log crit_sd -15, every criterion 0, gradient norm 4e3). The fix was independent criteria Normal(0,1) with MAX_PEOPLE=15 and no hyper-sd; bare rows use the mean of sigmoid over the 15 fitted criteria. After the fix the smallest eigenvalue is 0.99.
**Lesson:** for warm_start_nuts, avoid centred hierarchies with a free sd: the mode search collapses into the funnel.
