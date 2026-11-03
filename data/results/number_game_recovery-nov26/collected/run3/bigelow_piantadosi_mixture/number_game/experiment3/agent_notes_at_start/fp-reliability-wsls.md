---
name: fp-reliability-wsls
description: exp2 iter_0/cand_5: family-pooled WSLS + Beta-marginalised reliability (built by swapping inner_loop_model's Dirichlet for the family prior)
metadata:
  type: project
---
2026-11-01, experiment2 iter_0/candidate_5: family_pooled_reliability_win_stay_lose_sample. Rationale from the standings: family pooling gives about 30 nats; WSLS and uncertain reliability each give about 4 over family_pooled_rule_prior. So the triple (incumbent + stickiness) is the obvious bet. Built with a script that took inner_loop_model.py and replaced the Dirichlet/zero_split block with the softmax(family_level + tau*rule_dev) prior.
**How to apply:** see [[wsls-uncertain-reliability-graft]] and [[family-pooled-reliability-graft]]. Speed risk: the tau funnel plus the count tables.
