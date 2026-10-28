---
name: exp3-r4-digit-pattern-scan
description: Exp3 r4 MAP screen of exemplar/prototype additions to leading_digit_string_similarity — mirror digit pattern +8, prototype item 0
metadata:
  type: project
---
Exp3 round 4 (2026-10-18), incumbent leading_digit_string_similarity MAP ll −4630.2. Single additions tested:
- Mirror digit pattern dimension (reversal 13/31, or both palindromes 11..99): −4622.0 (+8.2). Reversal alone +5, palindromes alone +5. Submitted as mirror_digit_pattern_similarity (iter_4/candidate_1). Its p_yes RMSE vs the incumbent is 0.021, so it passes novelty.
- Set-prototype memory item (mean size plus feature frequencies, Beta weight): −4631 (no gain).
- Shared category (square, cube, power of 2, prime) dimension: about 0. Any shared digit +3.5. Cross-position digit match 0.
Residuals left over: targets 1 and 2 are still over-predicted by 0.07 (n=486), and set SD is 0.147 observed vs 0.118 predicted. 31,19,49,100 is under-predicted by 0.25 (target 1 gets 0.89 yes).
**Why:** saves re-screening.
**How to apply:** the scripts are in iter_4/candidate_1/scratch (mk.py builds the variants; grp.py groups residuals). See [[leading-digit-similarity-gain]].

Repair (iter_4/candidate_1_repair_1): the first submission failed convergence (R-hat 1.13) because the mirror dimension's coherence tuning was almost never active (few sets contain mirror pairs). Fixing mirror attention with no coherence tuning and a HalfNormal(1) weight converged cleanly: 4 chains, R-hat 1.00, min ESS 1753, 0 divergences. Lesson: never give coherence tuning to a dimension whose set coherence is almost always 0.
