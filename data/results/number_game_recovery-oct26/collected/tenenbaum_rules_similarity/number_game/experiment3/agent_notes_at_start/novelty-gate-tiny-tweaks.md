---
name: novelty-gate-tiny-tweaks
description: The novelty gate rejects candidates whose p_yes RMSE vs a live model is <0.002; the conj top-edge split scored 0.00056 and was rejected
metadata:
  type: feedback
---
Exp2 iter_2 cand_4 (`top_edge_marking_all_ranges`) was rejected at admission as a near-duplicate of upper_edge_marking_ranges: p_yes RMSE was 0.00056, under the 0.002 threshold on the 512-stimulus pool. The repair (2026-10-19) grafted range_discounted_size_principle's sharpness onto the incumbent's upper-edge split as `upper_edge_discounted_range_size`, pooling intervals by (at_top, group, k). The check passed with smoke LL -2055.5 (incumbent smoke about -2057); sampling took 275-385 s per run and the whole check about 13 min.
**Why:** Tweaks that only touch the conjunction columns barely move predictions.
**How to apply:** In refinement slots, pick a change known to have moved predictions already, e.g. a mechanism that passed novelty as a model of its own. See [[conj-top-edge-runtime]] and [[compose-near-ties]].
