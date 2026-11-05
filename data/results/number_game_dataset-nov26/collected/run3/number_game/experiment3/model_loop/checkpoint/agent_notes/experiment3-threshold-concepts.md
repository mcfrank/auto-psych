---
name: experiment3-threshold-concepts
description: "iter_3 cand_4 magnitude_threshold_concepts — one-sided \"above x\"/\"below x\" range boost gives +29 nats MAP on exp3 incumbent; dead ends tried that round"
metadata:
  node_type: memory
  type: project
  originSessionId: af58213b-dd6c-4e55-a5d7-99d080eaa250
  modified: 2026-11-04T00:23:50.487Z
---

iter_3 cand_4 (2026-11-03), refining iter2_candidate1 (incumbent MAP loglik -3987.5):
- Extra prior weight log1p(e^a·reaches_top + e^b·reaches_bottom) on intervals (normalised per family; groups keyed also by up/down flag). MAP -3958.2 (a 3.77, b 2.95, size_power 1.98→-0.27). Up only -3971.8; boosting only [1,100] -3966.5 (part of it is a yes-floor, like a lapse); one shared weight -3959.7. p_yes RMSE vs incumbent .032.
- Dead ends: adding primes (or every special-number rule) to the lone-example discount was worse (-3988.8/-3989.2). A dense log-compressed mental number line for interval likelihoods takes ~400 ms per gradient, far too slow.
- That round's cand_3 was cumulative_explaining_away_with_lapse (lapse on the incumbent).
Scratch: experiment3/model_loop/iter_3/candidate_4/scratch (map.py writes *_pred.csv).
Related: [[experiment3-data-patterns]], [[incumbent-speedup-sparse-gather]]
- iter_4 cand_5 (graded_explaining_away_with_thresholds, 2026-11-03): graded_explaining_away_with_guessing + the threshold boost (one family, so groups keyed by size,n_in,up,down). MAP -3949.3 vs incumbent MAP -3954.3, GEG -3971.8 (a 3.74, b 2.74, lapse .08, guess .29, size_power -.21); p_yes RMSE vs incumbent .020. Check OK, 120 s per 100/100 chain. Dead ends on GEG at MAP: an inferred stray rate for ranges only (→.075, +0); ranges must cover every example (-3961). None of these fixes mixed-parity 2-ex out-of-range (pred .36 vs obs .21). At the incumbent's MAP, ruleless 2-ex sets put ~all their posterior on intervals, much of it on intervals that miss one example or that reach an end of 1-100. Scratch: iter_4/candidate_5/scratch (map.py, dec2.py decomposition, /tmp/c5stats.py critique stats).

iter_4 cand_3 (rule_exceptionless_stray_belief, 2026-11-03), refining magnitude_threshold_concepts (MAP -3958.2): rule-specific stray rate (logit ~N(-2.2,3); ranges keep 0.1) → MAP -3950.8, rule stray .002, log_beta drops -.58→-1.12; p_yes RMSE vs incumbent .016. Hard "rules must contain every example" is -50 WORSE. Barely moves iter_4 critique stats (2-ex mixed-parity in .37/out .36 vs obs .47/.21). MAP null/dead ends: majority-unexplained penalty (→0), log-size curvature in range prior (+1.4), separate evidence for one-sided ranges (0), span [min,max] boost (0 from any start), range-only stray rate (+4, goes UP to .26). Check OK, 238 s per 100/100 chain. Scratch iter_4/candidate_3/scratch (map2.py dumps rule/interval/bounded contributions; an*.py slices).
