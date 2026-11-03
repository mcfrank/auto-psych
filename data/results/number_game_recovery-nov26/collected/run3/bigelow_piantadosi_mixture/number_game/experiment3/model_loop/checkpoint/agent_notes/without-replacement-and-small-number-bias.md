---
name: without-replacement-and-small-number-bias
description: iter_4 number-game outcomes — y^-gamma small-number frequency sampling collapses (gamma→0, slow); without-replacement size principle is parameter-free, distinct, OK
metadata:
  type: project
---
iter_4/candidate_0 (2026-10-30):
- Small-number frequency sampling (in-concept draw ∝ y^-gamma, strays too, so prod w(x) cancels): MAP gamma = -0.006, ll -817.3 (the same as inferred_example_noise). A dense (sets×5050) graph cost 141 ms/grad. Dead end; scratch is in iter_4/candidate_0/scratch/small_number_freq.py.
- Without-replacement sampling (swap s^-j for (s-j)!/s! in the incumbent's (n,k,s,j) table): no new parameter. MAP ll -816.9 vs incumbent -815.9; pool RMSE 0.0063, so distinct. Check OK, 44 s smoke fit, loglik -831.2. Submitted as without_replacement_example_sampling.
Harness: iter_4/candidate_0/scratch/cmp.py <candidate> prints MAP ll for the incumbent and the candidate, plus novelty-pool RMSE and the largest disagreements.
**Why:** the data keep pinning any magnitude-asymmetry parameter at 0, but structural, parameter-free likelihood changes still move predictions past the 0.002 novelty threshold.
**How to apply:** prefer parameter-free swaps inside the (n,k,s) table; see [[interval-tempering-speed-trick]], [[number-game-refinement-findings]].
