---
name: exp3-iter2-equal-rule-prior-probes
description: Exp3 iter_2 cand_3 MAP probes on graded_prior_digit_sum_concepts — noise rate 0.9 is a sharp optimum (data likely simulated at 0.9), endpoint range bias dead, equal per-rule prior costs only 1.3 nats with 5 fewer params (submitted)
metadata:
  type: project
---
2026-11-01, exp3 iter_2 candidate_3 (incumbent refinement slot). Base MAP LL -3513.54.
- Example-noise ALPHA (fixed 0.9 in all models) is a sharp optimum: 0.86 -31, 0.88 -11.5, 0.92 -2.2, 0.95 -54, 0.8 -115. A separate rule-only alpha also peaks at 0.9 (0.85 -33, 0.95 -37). Learned alpha is dead. The data look simulated with alpha 0.9.
- Ranges touching 1 or 100 (prior bias): bias -0.02, +0.01 nats. Dead, despite the endpoint critique stat.
- At MAP, the fitted per-rule prior is nearly equal (~0.013 per rule) across parity, specials, multiples and ends-in families; contains-digit and exception rules get weight 0. A single equal-share familiar-rule mass (Dirichlet(3): familiar rules / digit sums / ranges), with contains and exception rules removed, gives -3514.87 (-1.3) with 5 fewer params and RMSE 0.0087 vs base. Submitted as equal_prior_familiar_rules_digit_sums. Putting contains and exception rules in the equal share gives -4106.
- Gradient cost is ~60 ms/eval for both base and candidate. Probe scripts are in /tmp/c3i2.
**How to apply:** check the ELPD of the submitted model before trying more parsimony; see [[exp3-iter1-graded-incumbent-probes]], [[exp3-iter1-speed-repair]].
