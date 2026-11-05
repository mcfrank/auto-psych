---
name: experiment3-exception-weighted-rules
description: iter_4 cand_4 — rule-specific exception (stray) cost on the threshold incumbent gives +7.3 nats MAP; span-concept and threshold-decay dead ends
metadata:
  type: project
---
iter_4 cand_4 (2026-11-03, exception_weighted_rule_rejection), refining magnitude_threshold_concepts (MAP loglik -3958.2, re-run of models/ source; iter_3 v_both_pred.csv is a slightly different variant at -3954.3):
- Rule log-lik rebuilt from per-set rule hit counts: hits*log(0.9/size+.001) + misses*(log .001 - exception_shift), exception_shift~N(0,3). MAP -3950.9, shift 3.9, log_beta -.58→-1.12 (size principle on rules gentler, per-exception cost ~unchanged). Inferred rule-only stray rate (Beta(9,1)) gives -3950.3 but alpha → .9996 (boundary), so the shift form is better for sampling.
- Barely moves iter_4 critique stats (mixed-parity in-out z 3.6→3.1, others unchanged).
- Dead ends: exact-span [min,max] interval bonus (+0, bonus -0.1); threshold weight decaying with log n (+3.7, decay .9).
- Machine had 1 CPU this round; MAP ~1 min each. Scratch: iter_4/candidate_4/scratch (map.py, stats.py runs critique test_stats on pred csvs).
Related: [[experiment3-threshold-concepts]], [[experiment3-data-patterns]], [[incumbent-speedup-sparse-gather]]
