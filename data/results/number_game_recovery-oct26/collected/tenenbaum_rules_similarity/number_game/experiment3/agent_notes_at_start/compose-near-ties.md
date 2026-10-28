---
name: compose-near-ties
description: exp2 iter_2 cand_5 refinement slot: grafted upper-edge bonus onto odd_composite_concept; smoke fit OK, ~150s/100+100 draws
metadata:
  type: project
---
In a refinement slot (exp2 iter_2 candidate_5, 2026-10-19), I composed two near-tied live mechanisms: odd_composite_concept + upper-edge top_bonus (from upper_edge_marking_ranges). It is a two-line edit to the incumbent file (append odd-composite rule as kind 5, N_KINDS=6). The check passed, with a smoke log-lik of -2056.2; the smoke fit took 152s.
**Why:** the incumbent family is saturated ([[incumbent-family-saturated]]), and stacking small tied gains is the cheap option left.
**How to apply:** check the final admission/LOO outcome before reusing this pattern; if the gain is <1 nat, the stacking route is exhausted too.
