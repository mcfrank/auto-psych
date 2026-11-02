---
name: decade-block-ranges
description: exp3 iter2 cand2 — decade-aligned (30-39) range bonus instead of hundred-chart rows (31-40), +2.6 MAP on set_size_evidence_privileged_fives
metadata:
  type: project
---
Swapping IV_ROW to decade blocks (start at 1 or multiple of 10, end in 9 or at 100) on set_size_evidence_privileged_fives: MAP log-post -3462.5 vs -3465.1 (+2.6), no new param, other params unchanged.
Motivated by critique round_target_bonus_no_round_examples (model over-predicts yes to multiples of 10 when no example is round). Raw data: decade-base round target (30 after 34) yes 0.36 vs chart-row top (40 after 34) 0.30.
**Why:** small but free gain; tests block alignment. **How to apply:** row/decade alignment is a cheap lever; see [[prime-plus-chart-rows]].
