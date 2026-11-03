---
name: iter3-concept-family-screens
description: Round 3 cand_0 screens of extra concept families on rule_breadth_prior_concepts (NLL 903.26); all at ceiling; submitted counting_by_twos_run_concepts
metadata:
  type: project
---
Round 3, candidate_0 (2026-10-30). Baseline is rule_breadth_prior_concepts, MAP NLL 903.26. None of these extra families gained anything (best family weight → -inf): residue classes mod 3-12 with nonzero offsets; bounded arithmetic progressions with steps 2-10 or 2-20; rule ∩ contiguous range (6003 concepts). Two families came out flat, neither gaining nor losing. Step-2 runs ("counting by twos") gave 903.18 with family mass e^-3 of the rules, and runs of multiples of 10 gave 903.22. The all-steps AP family *hurts* at share ≥ e^-3, by 2.6 nats or more.

I submitted `counting_by_twos_run_concepts`, which is the incumbent plus step-2 runs, with log_run_share ~ N(-3, 1.5). Smoke loglik was -905.0 and the check took about 3 s per 200 iterations. Novelty RMSE against the incumbent is 0.006 at share -4 and 0.014 at -3, so it clears the 0.002 gate.

**Why:** this confirms [[number-game-ceiling]] and [[iter2-screens]]. Concept-space additions can't move the fit.
**How to apply:** don't re-screen these families. Note the screener bug: noise term is 0.1/100 = 0.001, not 0.01.
