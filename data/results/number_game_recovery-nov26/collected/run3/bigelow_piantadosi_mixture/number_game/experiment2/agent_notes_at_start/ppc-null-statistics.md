---
name: ppc-null-statistics
description: Number-game PPC test-stat ideas already tried (iter_0-2) that all came back non-significant
metadata:
  type: project
---
In iters 0-2 (2026-10-30), 24 PPC statistics found 0 significant discrepancies: early-vs-late drift, lag-1 perseveration, target-in-set yes rate, far-target yes rate, proximity gradients, inside-vs-outside range, upward vs downward extension, parity and shared-last-digit rule match, set-size slope, single-example yes rate, isolated-outlier neighbourhood, recency vs primacy, round-number bias, decade similarity, within-set spread.
iter_3 tried new ones: item unanimity, Weber-magnitude breadth, outlier-vs-clean last-digit rule, gcd multiples contrast, 1/100 edge extension, within-run position slope, participant-by-set breadth variance, single-example last digit.
iter_4 proposed (results pending): order-permuted set gap (only 16,8 vs 8,16), digit-reversal targets, sum/difference-of-examples targets, off-rule residue class mod gcd, between-set yes variance, shared non-last digit, prime vs odd-composite in prime sets, in-hull non-rule targets for small-rule sets.
**Why:** avoids re-proposing duplicates. Data facts: each (set, target) item was answered by all 5 participants; trial_index→set is shared across participants.
**How to apply:** check iter_3/critique/ppc_results.json for which new ones hit; see [[number-game-refinement-findings]].
