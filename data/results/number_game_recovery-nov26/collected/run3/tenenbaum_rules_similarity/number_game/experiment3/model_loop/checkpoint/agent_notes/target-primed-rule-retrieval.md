---
name: target-primed-rule-retrieval
description: exp3 iter_2 c0: target primes its own rules (weight exp(prime)); MAP 0.04, cond. sd≈0.05; plus 4 process ideas that collapsed or lost
metadata:
  type: project
---
exp3 iter_2 candidate_0 (2026-10-31), submitted as target_primed_rule_retrieval on the beta_share base: rules containing the target get posterior weight ×exp(prime). numpy MAP: prime 0.04, nll −3647.03 vs −3647.27; Δnll 5.7 at ±0.15 → p RMSE ~0.02. Smoke OK (ll −3653.4, 14 s). Novelty is borderline because prime≈0.
Numpy MAP tests on the same base that collapsed or lost:
- sequential anchoring-and-adjustment (belief moves a share toward each prefix posterior): a→0.993, null
- Gaussian blur of example magnitudes in range likelihood: worse as σ grows
- property-checking error ε on rules (targets only, or examples+targets): ε→0; ε=0.002 costs 15 nats
- fixed 3-example capacity: last3 +433, first3 +658, random 3-of-4 +219 nats
Scripts in /tmp/c0i2 (seq.py base loader, prime.py, verr2.py, cap.py).
**How to apply:** exp3 people use every example exactly and check properties accurately; process limits on memory or arithmetic don't help.
