---
name: check-sibling-candidates-first
description: In a refinement slot, read the sibling candidates' hypothesis.md in the same iter before building — the obvious graft is often already taken
metadata:
  type: feedback
---
iter_3/candidate_5 (2026-10-30): I built family_pooled_rule_prior + Beta-marginalised reliability. Its check passed but took 291 s per 200 draws (6 ms/grad, deep trees). Then I found that iter_3/candidate_3 had already submitted the same graft under the same name. I switched to family_pooled_win_stay_lose_sample (the WSLS model with the family-pooled hierarchical rule prior in place of the Dirichlet). That took 2.4 ms/grad and 115 s per 200 draws, with smoke loglik -835.0.
**Why:** sibling slots work in parallel and reach for the top-2 combination first.
**How to apply:** start by running `cat iter_N/candidate_*/hypothesis.md`. See [[wsls-exact-recursion]], [[number-game-refinement-findings]].
