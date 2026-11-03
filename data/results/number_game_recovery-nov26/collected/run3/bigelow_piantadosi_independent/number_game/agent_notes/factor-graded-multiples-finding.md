---
name: factor-graded-multiples-finding
description: exp2 incumbent + multiples-of-m cost growing with log(m/3) gains +34 loglik MAP (one param); fitted stray rate / guess rate / rule-vs-range temper ~null
metadata:
  type: project
---
2026-10-30, cell_1/experiment2 iter_0/candidate_4_repair_1 (replaced the trial_index drift model that was rejected for binding trial_index):
- Incumbent costly_squares_final_digit_concepts MAP loglik -2207.6. Adding mult_cost*log(m/3) on the multiples-of-m rules: -2173.5, mult_cost ~2.3. A linear (m-3)/3 form gave -2178.9. Submitted as `factor_graded_multiples_concepts`; smoke loglik -2178.0.
- Null: a fitted lapse guess rate (lapse fits to 0), a fitted stray rate (0.09, vs the fixed 0.1), and a separate rule-vs-range temper kappa (+5.7 loglik, but only +1 logpost).
- Remaining misfits: 83,38,58,68 (people answer "contains 8": 85/87/89 get yes); 8,64 (the multiples of 8 are still too high); 20,32,92; 100,1,16.
- GOTCHA: `model.compile_fn(model["p_yes"], inputs=value_vars)` forward-samples the RVs, which gives wrong p. Wrap it in `model.replace_rvs_by_values([...])[0]` first. find_MAP lapse logodds varies by start, so compare loglik, not logpost.

**Why:** a large, cheap gain on exp2, and a scripting trap that wasted time.
**How to apply:** scripts are in that candidate's scratch/ (screen2.py, res.py). See [[squares-cost-split-finding]] and [[session-drift-null-finding]].
