# Critique of `bounded_typicality_outlier_listener`

8 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **mumble_full_duplicate_choice_rate** — Rate of choosing a duplicated candidate possessing all features on mumble trials where identical objects are present. (observed 0.587 vs null mean 0.475, z=5.00, p=0.002, q=0.004) [survives FDR]
- **mumble_singleton_choice_rate** — Rate of choosing the unique singleton object on mumble trials where duplicate objects are present in the display. (observed 0.607 vs null mean 0.686, z=-4.83, p=0.002, q=0.004) [survives FDR]
- **four_object_multifeature_target_vs_all_feature_rate** — Rate of choosing the multi-feature pragmatic target possessing at least two features in four-object displays where the utterance matches exactly two candidates and the competitor possesses all features. (observed 0.773 vs null mean 0.635, z=4.46, p=0.002, q=0.004) [survives FDR]
- **all_match_max_feature_choice_rate** — Rate of choosing the object with the maximal feature count when the heard utterance is true of all objects in the display. (observed 0.298 vs null mean 0.261, z=3.39, p=0.002, q=0.004) [survives FDR]
- **four_object_two_feature_target_vs_three_feature_competitor_rate** — Rate of choosing the two-feature pragmatic target in four-object displays where the heard utterance matches exactly two candidates and the competitor possesses three features. (observed 0.685 vs null mean 0.533, z=2.93, p=0.004, q=0.00639) [survives FDR]
- **three_candidate_competitor_choice_rate** — Rate of choosing the maximal-feature competitor among displays where exactly three objects match the heard utterance. (observed 0.273 vs null mean 0.246, z=2.79, p=0.00999, q=0.0133) [survives FDR]
- **three_object_one_feature_target_vs_two_feature_competitor_rate** — Rate of choosing the single-feature pragmatic target in three-object displays where the heard utterance matches exactly two candidates and the competitor possesses two features. (observed 0.562 vs null mean 0.738, z=-2.74, p=0.022, q=0.022) [survives FDR]
- **mumble_max_feature_choice_rate** — Rate of choosing the object with the maximal number of features on mumble trials where objects differ in feature counts. (observed 0.628 vs null mean 0.594, z=2.54, p=0.012, q=0.0137) [survives FDR]
