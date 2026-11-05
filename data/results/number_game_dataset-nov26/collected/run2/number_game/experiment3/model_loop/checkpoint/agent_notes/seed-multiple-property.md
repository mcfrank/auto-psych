---
name: seed-multiple-property
description: "Smallest example divides all others → multiples-of-it property" gave +10 MAP loglik on the overlap model; gcd/spacing/coincidence/contrast/specificity tweaks gave ~0
metadata:
  type: project
---
Tested 2026-11-02 (iter_4/candidate_1) as MAP log-lik on top of one_sided_magnitude_bound_overlap (incumbent MAP -1454.8):
- Seed-multiple family (min example ≥2 divides all examples; target is a multiple of it), with its own Dirichlet salience: -1444.8 (+10 for 1 param). Submitted as seed_multiple_salience_overlap. It helps 7,63 (even multiples of 7 accepted), 81,9, 8,64.
- ~0: gcd of the examples as seed, residue-mod-m "spacing" family, coincidence weight (1-b^n)^gamma, slope scaled by profile specificity.
- Worse (-1462): Tversky contrast (absolute match minus mismatch) instead of the ratio.
- The incumbent tops out near p=0.71 for full matches; 28,98 targets ending in 8 are seen ~0.9 (the critique's final-digit miss). Nothing tried fixed that.
**How to apply:** relations between the examples (one dividing another) are where new signal is; see [[overlap-model-tweaks]].

Update 2026-11-02 (experiment2 iter_0/candidate_4): grafted the seed family onto final_digit_overshadowing_overlap, with the seed salience also discounted by `retain`. Submitted as seed_multiple_digit_overshadowing_overlap. The smoke check passed and a fit takes about 6 s. Its ELPD was not measured before submitting. The two mechanisms affect different sets, so their gains may add up. See [[final-digit-overshadowing]].
