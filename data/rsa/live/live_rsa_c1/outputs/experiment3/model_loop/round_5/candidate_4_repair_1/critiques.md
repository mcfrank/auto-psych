# Critique of `feature_aspect_qud_listener`

8 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **foil_choice_rate_four_object_displays** — Rate of choosing any non-matching foil object in utterance trials with four-object displays. (observed 0.032 vs null mean 0.017, z=4.96, p=0.002, q=0.00266) [survives FDR]
- **prior_intermediate_features_rate** — Rate of choosing intermediate-complexity objects on mumble trials where the display contains both extreme (empty and full) objects and intermediate objects. (observed 0.183 vs null mean 0.252, z=-4.17, p=0.002, q=0.00266) [survives FDR]
- **target_rate_redundant_feature_utterance** — Rate of choosing the minimal-feature target in ambiguous utterance trials where the heard word is a redundant feature sharing its exact object extension with another feature. (observed 0.726 vs null mean 0.622, z=4.16, p=0.002, q=0.00266) [survives FDR]
- **intermediate_matching_competitor_rate** — Rate of choosing an intermediate-feature competitor in utterance trials where the heard word matches three or more objects with differing feature counts. (observed 0.114 vs null mean 0.173, z=-3.91, p=0.002, q=0.00266) [survives FDR]
- **prior_singleton_rate_with_duplicates** — Rate of choosing any visual singleton object on mumble trials where the display contains both singleton and duplicated objects. (observed 0.601 vs null mean 0.663, z=-3.80, p=0.002, q=0.00266) [survives FDR]
- **target_rate_scaling_4obj_vs_2obj** — Difference in minimal-feature target choice rate between 4-object displays and 2-object displays in utterance trials where the heard word matches two objects. (observed 0.0344 vs null mean -0.0505, z=3.67, p=0.002, q=0.00266) [survives FDR]
- **prior_max_feature_rate_all_singletons** — Rate of choosing the maximal-feature object on mumble trials where all objects in the display are visual singletons. (observed 0.618 vs null mean 0.568, z=2.76, p=0.00999, q=0.0114) [survives FDR]
- **prior_dup_full_rate_with_empty_singleton** — Rate of choosing the full duplicated object on mumble trials where the display contains an empty singleton object and a full object with duplicate copies. (observed 0.531 vs null mean 0.474, z=2.18, p=0.032, q=0.032) [survives FDR]
