---
name: marked-odd-cost-finding
description: exp2 odd-only prior cost on hundred_chart_row_ranges +13 loglik MAP; guessing, rule/range gating, heterogeneity, predictive-density decision all null/worse
metadata:
  type: project
---
2026-10-30, exp2 iter_2/candidate_1 (`marked_odd_parity_concepts`). MAP screens on hundred_chart_row_ranges (base loglik -2151.3, logpost -2178.8):
- Extra prior cost on "odd" only (rule index 0): loglik -2138.1, logpost -2172.0, odd_cost≈0.97. Smoke loglik -2143.0. Motivated by all-odd sets: odd non-prime same-parity targets got 0.51 yes vs 0.65 predicted (in range), 0.30 vs 0.46 (out of range); all-even sets were well calibrated.
- Null or worse: uncertainty-gated guessing (+2, absorbed by beta<1), set-cue-gated lambda (span, n, density ~0), two-lambda population heterogeneity (~0), yes = predictive-density Bayes factor vs random (-56).
- In iter_2, four other candidates took the prime bonus/sieve fix for the prime critique.

**Why:** the incumbent is well calibrated by n and distance. Its remaining structured misfits are parity asymmetry and primes.
**How to apply:** an odd cost may stack with a prime bonus, since both lower yes to odd composites after prime sets. Related: [[prime-salience-finding]], [[uncertainty-triggered-guessing]].
