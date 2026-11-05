---
name: digit-run-rule-attention
description: iter_4 c3 — digit-run boosts final-digit attention on shared-factor incumbent, -15.4 nats MLE; smoke -1543.6
metadata:
  type: project
---
iter_4 candidate_3 (2026-11-02), `digit_run_rule_attention_exemplar`: incumbent `shared_factor_parity_exemplar` plus a log-multiplier on final-digit attention when >=3 examples share one final digit. MLE NLL 1539.1 vs base 1554.5 (boost ≈ e^0.75). A criterion shift, the same shift on non-matching targets only, and the attention boost all fit identically, because the data have no digit-matching targets in digit-run sets (252 trials, yes-rate 0.135). Multiples-of-5 dimensions: about 0 to -3 nats. check OK.
**Why:** it targets the strongest iter_4 critique (digit_run_nonmatch_rate, q=.016). **How to apply:** to tell a stricter criterion apart from rule attention, the design needs digit-matching targets in run sets. See [[digit-run-strict-criterion]], [[iter4-critique-stats]].
