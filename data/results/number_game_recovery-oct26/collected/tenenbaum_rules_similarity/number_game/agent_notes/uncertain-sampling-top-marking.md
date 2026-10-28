---
name: uncertain-sampling-top-marking
description: exp2 iter_4 c5 — uncertain_sampling_assumption with incumbent's generative top-marking credit per sampling mode; smoke LL -2054.4, 52s/run
metadata:
  type: project
---
2026-10-19, exp2 iter_4 candidate_5 (refinement slot): `uncertain_sampling_top_marking` = uncertain_sampling_assumption with the constant top_bonus replaced by log1p(top_odds*top_share/draw_prob), where draw_prob is per-width under strong sampling and 0.01 under weak. Built by string-replace on the source; check OK, smoke LL -2054.4 (best smoke seen so far), 52 s per sampling run.
Also: the iter_4 critique was all null (8 stats), including session_yes_rate_sd p=0.60, so the session yes-rate differences come from the stimuli and not from people (see [[session-heterogeneity]]).
**How to apply:** check the outcome in the next round's existing_hypotheses before reusing it. Related: [[uncertain-sampling-candidate]], [[teacher-top-edge-form]].
