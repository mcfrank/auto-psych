---
name: single-example-magnitude-neighborhood
description: "About this size" Gaussian property for one-example sets only, on gap_step incumbent: +16.6 meanp loglik; closes single-gap and proximity critiques
metadata:
  type: project
---
Tested 2026-11-02 (experiment2 iter_2/candidate_2, single_example_magnitude_neighborhood). Base gap_step_progression_overlap scores -3145.7 on posterior-mean-p loglik (2x300 NUTS). Results:
- Lone example adds a property exp(-(d/w)^2) with its own salience, for n=1 only: -3129.1 (w about 1.07, so broad). Singles predicted 0.574 vs 0.588 observed. Proximity stat predicted 0.067 vs 0.056 observed (base predicts it negative). Submitted. Smoke -3133.2.
- The same property for all sets (nearest-example distance, overshadowed): -3135.9. Not overshadowed: ~0.
- For n=1, replacing the one-sided bounds with the neighborhood (no new salience): -3131.6, with an rhat warning.
**Why:** the n=1 critiques. The one-sided bounds made far-above targets look better than near-below ones.
**How to apply:** this partly overlaps with single-example leniency because the width is broad. See [[single-example-breadth]], [[graded-magnitude-band]].
