---
name: pedagogical-teacher-sampling
description: Pedagogical (helpful-teacher) sampling model for the number game: formulation, speed fix, and fit outcome
metadata:
  type: project
---
iter_3/candidate_2 (2026-10-30), pedagogical_teacher_sampling: the teacher picks x in h with probability ∝ P0(h|x)^tau. The prior cancels, leaving w(x)=Z0(x)^-tau over W(h)=Σ_{x∈h} w. Exact in-graph intervals (sets×4×5050) took 1.9 s/grad, far too slow. Fix: give the naive learner a fixed prior Z0 (rules vs intervals 50/50, equal within each) and fix alpha=0.9. The intervals then depend only on tau, so tabulate them on an 81-point tau grid using the Gaussian interpolation from [[grid-tabulated-range-evidence]]. Rules stay exact in the graph. Result: 2 ms/grad, MAP tau≈0.42, lambda_0≈0.22, loglik -818.1 (BP base -818.0, incumbent -815.9). Check OK, with a 10 s smoke fit (loglik -831.8). Novelty result still pending.
**Why:** this is a different process family (social/communicative inference), and tau did not collapse to 0.
**How to apply:** reuse the grid trick for any teacher or recursive-sampling variant. See [[interval-tempering-speed-trick]].
