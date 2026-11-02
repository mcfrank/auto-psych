---
name: exp3-iter4-offset-progressions
description: Exp3 iter-4 cand_0 concept-space screens on position-decaying incumbent; bounded stepped runs strongly rejected; residue classes +0.28 submitted
metadata:
  type: project
---
Exp3 iter 4, candidate_0 (2026-10-31), a concept-space slot. All numbers are likelihood-only MAP on the position_decaying incumbent, which sits at -3189.70.
- Bounded APs with step ≥2 (or ≥3, or ≥5) and at least 3 members, added to the interval side with their own share: the share collapses to -19 or lower, which makes it a duplicate. Fixing the share at -3 costs about 30 nats; at 0 it costs about 330-380. People do not use stepped runs.
- These families are null (gain ≤0.07, share collapses): multiples of 13-50, unions of two "ends in" digits, repdigit/digit-order/digit-parity families, odd- or even-digit-only numbers, rules cut at a decade threshold.
- Residue classes (m 3-12, except 10, nonzero r): +0.28 at a share of about -8.7. Submitted as `offset_progression_concepts` with prior N(-6,2). Pool RMSE vs the incumbent is 0.005, and the smoke fit gave -3192.5.
- The fourth_example_prefix critique (31,3,1→31,3,1,15) is just noisy cells near 0.6. No concept family explains it.
The screener is iter_4/candidate_0/scratch/fam.py (FAM env var; it splits the digit-sum share) and pool.py does the RMSE.
See [[exp3-iter0-concept-family-screens]], [[number-game-ceiling]].
