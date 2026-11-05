---
name: span-interval-similarity
description: "between the examples" interval property (n>=2) with an exponential falloff, added to progression_band_first_digit_primacy: +5.9 grouped LOO
metadata:
  type: project
---
Experiment 2, iter_4 candidate_2 (2026-11-02), span_interval_similarity. Using the 2x300 NUTS harness in scratch/ev.py, grouped LOO went from −3142.2 for the incumbent to −3136.4. The added property is an interval in [min,max] with exp(-d/w) decay, w≈0.085, active only for n>=2. The Gaussian form gave −3137.2. Turning it on for n=1 as well cut the gain to +1.7. The critique stat inrange_nonshared_rate went from −0.016 to +0.021 against 0.066 observed, so it is only partly closed. Smoke fit: −3128.4.
**Why:** the broad one-sided band (w≈0.6–0.8) cannot also produce sharp in-range vs out-of-range contrast in multi-example sets.
**How to apply:** the in-range gap is still partly open. See [[graded-magnitude-band]], [[band-plus-leniency]].
