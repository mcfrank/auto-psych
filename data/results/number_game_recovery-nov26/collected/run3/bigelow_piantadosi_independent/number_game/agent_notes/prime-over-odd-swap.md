---
name: prime-over-odd-swap
description: exp2 one-param prior shift odd→primes on hundred_chart incumbent, MAP +24 nats; prime-membership tweaks null
metadata:
  type: project
---
2026-10-30, exp2 iter_2/candidate_4_repair_1 (`prime_over_odd_reading`). The first attempt, a plain prime bonus, was rejected as a near-duplicate of prime_salient_hundred_chart (RMSE 0.0008).
MAP negative log-likelihoods: incumbent 2151.1; prime bonus 2136.4; odd cost alone 2137.9; separate prime bonus and odd cost 2123.7. A single delta, added to primes and subtracted from odd, gave 2126.9 (delta≈1.2). Smoke loglik was -2132.5 and the check printed OK.
Null results: graded prime recognition by magnitude (k→0); fuzzy sieve membership (q→0); primes at one primitive with no parameter (+4.5 only).
Scratch scripts are screen*.py in that candidate's scratch/. They use a fast numpy+Powell MAP.

**Why:** prime-related variants are crowded and the novelty gate rejects them.
**How to apply:** the remaining large residuals are 3,33,99→23; 14→98; 14,98,91→7/77. Related: [[marked-odd-cost-finding]], [[prime-salience-finding]].
