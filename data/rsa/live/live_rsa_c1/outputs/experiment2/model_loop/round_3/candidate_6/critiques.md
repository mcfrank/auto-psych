# Critique of `contrastive_extremity_noisy_speaker`

8 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **prior_intermediate_features_rate** — Rate of choosing intermediate-complexity objects on mumble trials where the display contains both extreme (empty and full) objects and intermediate objects. (observed 0.186 vs null mean 0.287, z=-4.79, p=0.002, q=0.00266) [survives FDR]
- **prior_singleton_rate_with_duplicates** — Rate of choosing any visual singleton object on mumble trials where the display contains both singleton and duplicated objects. (observed 0.569 vs null mean 0.661, z=-4.60, p=0.002, q=0.00266) [survives FDR]
- **foil_choice_rate_four_object_displays** — Rate of choosing any non-matching foil object in utterance trials with four-object displays. (observed 0.0338 vs null mean 0.017, z=4.58, p=0.002, q=0.00266) [survives FDR]
- **prior_dup_full_rate_with_empty_singleton** — Rate of choosing the full duplicated object on mumble trials where the display contains an empty singleton object and a full object with duplicate copies. (observed 0.525 vs null mean 0.412, z=4.28, p=0.002, q=0.00266) [survives FDR]
- **target_rate_redundant_feature_utterance** — Rate of choosing the minimal-feature target in ambiguous utterance trials where the heard word is a redundant feature sharing its exact object extension with another feature. (observed 0.711 vs null mean 0.597, z=3.79, p=0.002, q=0.00266) [survives FDR]
- **competitor_choice_rate_large_feature_gap** — Rate of choosing the maximal-feature competitor in ambiguous utterance trials where the competitor has at least 3 more features than the minimal target. (observed 0.299 vs null mean 0.351, z=-3.00, p=0.002, q=0.00266) [survives FDR]
- **target_rate_three_matching_objects** — Rate of choosing the minimal-feature target in utterance trials where the heard word matches exactly three objects with differing feature counts. (observed 0.658 vs null mean 0.717, z=-2.79, p=0.012, q=0.012) [survives FDR]
- **target_rate_scaling_4obj_vs_2obj** — Difference in minimal-feature target choice rate between 4-object displays and 2-object displays in utterance trials where the heard word matches two objects. (observed 0.0436 vs null mean -0.0364, z=2.78, p=0.00799, q=0.00913) [survives FDR]
