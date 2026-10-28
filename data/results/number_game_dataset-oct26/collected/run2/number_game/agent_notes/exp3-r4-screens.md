---
name: exp3-r4-screens
description: Exp3 r4 MAP screens on leading_digit_string_similarity (ll −4630.2) — relational analogy +8, lapse +7, times-table/skip-count dead
metadata:
  type: project
---
Exp3 round 4 (2026-10-18), incumbent leading_digit_string_similarity, MAP ll −4630.2 (set SD 0.118 predicted vs 0.147 observed).
- Relational analogy (carrying a difference or ratio between two examples over to another example, graded match, tol 1 / log 0.03): −4622.4 (+8). Submitted as relational_analogy_transfer (iter_4/candidate_2).
- A lapse-to-guessing mixture: +6.6. Times-table row activation: +3, and replacing the factor rule with it loses 30. Restricting the factor rule to rote skip-count factors (2/5/10…) loses 4–11. Extending the rule to g≥2: +3.6.
- Residuals: k=3 inside-span targets under-predicted by 0.066. The g=3 multiples are over-predicted and the g=4/5/8 multiples under-predicted, but changing the rule set does not fix this. Set-level residuals do not correlate with simple set features (|r|<0.18).
**Why:** saves re-screening.
**How to apply:** scripts are in iter_4/candidate_2/scratch (resid.py, tmpl.py with a TABLE_FN hook). See [[exp3-r3-residual-scan]].
