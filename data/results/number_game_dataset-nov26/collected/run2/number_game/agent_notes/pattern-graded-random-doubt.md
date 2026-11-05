---
name: pattern-graded-random-doubt
description: exp3 n=1 doubt as logistic in lone example's profile salience: smoke -4678.1 (vs -4682.4), converges; slope +1.7 (patterned → more flat-yes)
metadata:
  type: project
---
exp3 iter_1 c4 (2026-11-02): a refinement of [[single-example-random-doubt]]. The global Dirichlet doubt is replaced by doubt = sigmoid(base + slope * centred total profile salience, i.e. the den of the lone example) plus random_yes ~ Beta. Smoke -4678.1 (incumbent -4682.4). A 2x400 fit gave R-hat <= 1.01, 0 divergences.
Posterior: doubt_slope +1.7 (+0.6 to +2.8), random_yes 0.82. So richly patterned lone examples are answered more flatly "yes". This is the opposite of the naive guess. Observed lone ×5/×10 examples (85, 55, 30, 100) are endorsed least (~0.25-0.4).

**Why:** it targeted the n1_set_mean_variance critique (q=0.032).
**How to apply:** if this is admitted, check the PPC for whether n1 variance closed. Related: [[exp3-n1-data-facts]]
