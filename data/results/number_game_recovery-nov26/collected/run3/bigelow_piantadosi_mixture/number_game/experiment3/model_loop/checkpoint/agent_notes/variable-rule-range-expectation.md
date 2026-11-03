---
name: variable-rule-range-expectation
description: exp3 iter_1 c4: lambda varies per judgement (logit-normal, 9-node GH) on range_limited incumbent: +1.55 MAP, sd .73 not collapsing; width/Erlang/parity/kind-reliability all collapse
metadata:
  type: project
---
exp3 iter_1/candidate_4 (2026-11-01) is variable_rule_range_expectation. The incumbent range_limited_rule_familiar_strays has a fixed lambda; here logit(lambda) ~ N(mu, sd), and the membership posterior is averaged over 9 Gauss-Hermite nodes on the (sets, concepts, K) log-joint.
- MAP ll: -3319.24 vs -3320.79 for the incumbent. lambda_sd = .73.
- Train p_yes RMSE vs the incumbent: .007. Smoke fit 57 s. Check OK.
- It does NOT close the critique narrow_rule_inhull_nonmember_span_contrast: narrow in-span non-member yes is .52 observed vs .41 predicted. Nothing tried moves that number.
Tried on the incumbent, all at MAP, none useful:
- interval width^-gamma prior: gamma .03, ll +0.06.
- Erlang interval prior: ll -4.1.
- separate parity-rule weight: beta -.09, ll +0.25.
- separate range reliability: +0.42.
Odd residual: 16,26,96 → 67 gets 5/5 yes while 61, 64, 65 and 69 get 0/5 (probably the "6 7" meme, which the models cannot express).
**How to apply:** prior *variability* (integrated per judgement) is a lever that does not collapse, unlike prior *shape* changes. See [[normative-distortions-collapse]] and [[range-limited-rule-familiar-strays]].
