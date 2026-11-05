---
name: undiluted-magnitude-dimension
description: magnitude band moved out of the normalized share into its own direct logit term (exp2 iter_4 c0); smoke -3181, worse than in-share band
metadata:
  type: project
---
exp2 iter_4 candidate_0 (`undiluted_magnitude_range_dimension`): the best model (progression band + primacy) with the two Gaussian magnitude-bound properties taken out of the salience-normalized share and added straight to the logit with gains gain_lo and gain_hi. It targeted the inrange_nonshared_rate critique. Smoke fit -3181.3, against about -3133 for the in-share band ([[progression-band-primacy]]), so it is probably worse. Magnitude seems to work better inside the normalized share.

**Why:** shows whether dilution of the magnitude band explains the in-range critique.
**How to apply:** if it gets pruned, don't move magnitude out of the share again; try a different functional form for in-range credit.
