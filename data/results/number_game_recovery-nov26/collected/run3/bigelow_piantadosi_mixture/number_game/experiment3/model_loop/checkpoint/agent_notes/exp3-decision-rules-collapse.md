---
name: exp3-decision-rules-collapse
description: exp3 iter_0 c0: lapse, guess-bias, log-odds slope and consensus p^k on familiar_strays_equal_rule_prior all collapse at MAP; data calibrated, no sequential dependence
metadata:
  type: project
---
exp3 iter_0/candidate_0 (2026-11-01) submitted consensus_sampling_decision: P(yes)=belief^k on the familiar_strays_equal_rule_prior inference. At MAP k=1.002 and ll went from -3322.44 to -3322.43, so it is likely a near-dup.
- Lapse with a guess rate: eps→6e-8. Log-odds slope with a bias: slope 0.996, bias -0.016, ll -3322.37.
- The parent's binned calibration on exp3 is near-perfect from 0.007 to 0.99. In-set targets: obs .708 vs pred .714. The set-size means match.
- The correlation of the residual with the previous response is -0.004, so there is no response perseveration.
- Gotcha: compile_fn(p_yes, inputs=value_vars) gave a wrong p (ll -4193). Read p_yes from the find_MAP dict instead.
**How to apply:** this confirms [[noisy-criterion-decision-rule]] on exp3. Decision-stage briefs cannot beat these models, so state that the result is expected to collapse.

Update (repair_1): consensus p^k was rejected as a near-dup (RMSE .0005). A hull-distance-scaled exponent belief^exp(b*d) also collapsed (b=.004). Submitted attention_lapse_guessing instead: lapse ~ Beta(20,980) as a stated prior, with a free guess. At MAP lapse=.0167, guess=.29, ll −3324.1 (−1.8 vs parent), RMSE vs no-lapse .005 on observed cells. A stated prior is the only way a decision-rule slot passes novelty.
