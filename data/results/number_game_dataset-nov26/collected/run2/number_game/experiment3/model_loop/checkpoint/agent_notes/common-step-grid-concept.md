---
name: common-step-grid-concept
description: exp3 iter_4 c0 — "common grid" concept (gcd of gaps >=3, both directions) on parity_doubt_three_example_memory: smoke -4658.9, a tie
metadata:
  type: project
---
exp3 iter_4 candidate_0 (2026-11-02): added a grid concept to the incumbent parity_doubt_three_example_memory. Let g = gcd of the gaps between examples; when g >= 3, a target counts as a hit if it is congruent to the examples mod g. Its weight is grid_boost (LogNormal(0, .75)) × the progression salience, overshadowed by a shared final digit. The check passed. Smoke fit -4658.9; the incumbent's smoke is about -4658.5, so this is a tie.
Raw rates on multi-example sets: grid hit with nonzero residue .42 vs miss .40, so the signal is weak.
Only significant critique this round: early_session_yes_gap. It needs trial_index, which models may not use.
**Why:** the concept space for exp3 looks saturated, like [[concept-space-additions]] and [[power-chain-generation]].
**How to apply:** don't retry lattice or residue concepts. The check takes about 11 minutes, so run it in the background.
