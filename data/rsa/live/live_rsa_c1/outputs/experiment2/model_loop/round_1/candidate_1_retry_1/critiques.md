# Critique of `contrastive_similarity_speaker`

8 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **prior_intermediate_features_rate** — Rate of choosing intermediate-complexity objects on mumble trials where the display contains both extreme (empty and full) objects and intermediate objects. (observed 0.186 vs null mean 0.31, z=-6.11, p=0.002, q=0.00228) [survives FDR]
- **foil_choice_rate_four_object_displays** — Rate of choosing any non-matching foil object in utterance trials with four-object displays. (observed 0.0338 vs null mean 0.017, z=4.60, p=0.002, q=0.00228) [survives FDR]
- **prior_dup_full_rate_with_empty_singleton** — Rate of choosing the full duplicated object on mumble trials where the display contains an empty singleton object and a full object with duplicate copies. (observed 0.525 vs null mean 0.409, z=4.45, p=0.002, q=0.00228) [survives FDR]
- **target_rate_two_matching_objects** — Rate of choosing the minimal-feature target in utterance trials where the heard word matches exactly two objects with different feature counts. (observed 0.687 vs null mean 0.642, z=3.47, p=0.002, q=0.00228) [survives FDR]
- **target_rate_redundant_feature_utterance** — Rate of choosing the minimal-feature target in ambiguous utterance trials where the heard word is a redundant feature sharing its exact object extension with another feature. (observed 0.711 vs null mean 0.617, z=3.09, p=0.002, q=0.00228) [survives FDR]
- **intermediate_matching_competitor_rate** — Rate of choosing an intermediate-feature competitor in utterance trials where the heard word matches three or more objects with differing feature counts. (observed 0.122 vs null mean 0.186, z=-3.04, p=0.002, q=0.00228) [survives FDR]
- **competitor_choice_rate_large_feature_gap** — Rate of choosing the maximal-feature competitor in ambiguous utterance trials where the competitor has at least 3 more features than the minimal target. (observed 0.299 vs null mean 0.351, z=-2.95, p=0.002, q=0.00228) [survives FDR]
- **prior_max_feature_rate_all_singletons** — Rate of choosing the maximal-feature object on mumble trials where all objects in the display are visual singletons. (observed 0.603 vs null mean 0.546, z=2.53, p=0.016, q=0.016) [survives FDR]
