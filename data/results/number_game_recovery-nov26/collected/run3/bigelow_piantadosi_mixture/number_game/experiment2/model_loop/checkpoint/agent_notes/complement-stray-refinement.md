---
name: complement-stray-refinement
description: Strays drawn from the concept's complement (not uniform 1-100) in the number-game incumbent — small MAP gain, distinct predictions; width prior is a near-dup
metadata:
  type: project
---
iter_3/candidate_4 (2026-10-30), refining uncertain_reliability_noise_concepts (MAP loglik -815.9):
- Interval prior ∝ width^gamma: gamma ≈ 0.07, -815.6, novelty-pool RMSE 0.0056 — a near-duplicate, so dropped.
- Strays drawn from the complement of the concept: p(X|h) = s^-k (100-s)^-(n-k) B(u+k,v+n-k)/B(u,v), with no j-sum. MAP -815.3 with the same parameters, pool RMSE 0.031, data RMSE 0.023. Submitted as complement_stray_reliability_concepts; check OK, 40 s smoke fit, smoke loglik -830.2.
- Candidate_3 in the same round did incumbent + family-pooled prior, so check sibling candidates' agent.jsonl to avoid duplicates.
- The residuals look like per-cell binomial noise (item unanimity PPC matched).
Scratch harness: iter_3/candidate_4/scratch/var.py + scr.py (VAR env switch; prints MAP ll and RMSE vs base).
**Why:** the incumbent is near ceiling; changing the noise model's shape is one of the few changes that still moves predictions.
**How to apply:** see [[interval-tempering-speed-trick]], [[number-game-refinement-findings]].
