---
name: graded-magnitude-band
description: Replacing hard one-sided magnitude bounds with a Gaussian magnitude gradient gave +13.8 MAP on the exp2 incumbent seed_overshadowing_last_digit_recency
metadata:
  type: project
---
Tested 2026-11-02 (experiment2 iter_1/candidate_0, graded_magnitude_band_overlap). Incumbent MAP loglik is -3152.7. Using exp(-(gap/w)^2) for the gap below min / above max (shared width) gives -3138.9 (w about 0.86 of the 1-100 scale, so very broad). Results for the other forms: exponential -3141.1; distance to the nearest example -3145.3; width scaled with the set's spread +0. This targets the critique single_example_yes_gap (people generalize more from one example). Predicted non-example yes for single sets goes from 0.539 to 0.558, against 0.588 observed, so only part of the gap closes.
**Why:** the hard bounds penalized every non-example target of a one-example set.
**How to apply:** the single-example gap is still partly open. Try a set-size-dependent mechanism next. See [[overlap-model-tweaks]], [[last-example-digit-recency]].

Update 2026-11-02 (iter_2/candidate_3, progression_graded_magnitude_band): grafting the same Gaussian band onto the gap_step_progression_overlap incumbent passed check_candidate, with a smoke-fit loglik of -3143.5. The full ELPD result was not known when this was written.
