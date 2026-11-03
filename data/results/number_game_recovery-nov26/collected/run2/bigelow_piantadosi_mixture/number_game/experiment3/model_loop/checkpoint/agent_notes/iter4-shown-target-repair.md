---
name: iter4-shown-target-repair
description: Iter-4 cand_4 repair — free-stray resubmission rejected as duplicate of cand_3 (RMSE 0.0005); replaced by shown-target "no repeat" discount, smoke -904.2
metadata:
  type: project
---
Iter 4, candidate_4 (2026-10-30). The first submission (free stray rate on the WOR incumbent) was rejected as a near-duplicate of candidate_3's `without_replacement_free_outlier_rate`, with p_yes RMSE 0.00047. Parallel candidates in the same round can collide like this.

The repair is `shown_examples_not_repeated`: the incumbent, with p_concept scaled by (1 - repeat_no) when the target is one of the shown examples. The prior is repeat_no ~ Beta(1,6). It is framed as "the program doesn't repeat", which follows from the WOR belief. The novelty pool has 16 in-set targets out of 512. The MAP screen gained +0.3 nats. Smoke loglik -904.2, at about 4 s per 200 iterations.

**How to apply:** check its ELPD and whether it passed the novelty gate next round. Before submitting, check what the other candidates in the round are doing. See [[iter4-wor-free-stray]].
