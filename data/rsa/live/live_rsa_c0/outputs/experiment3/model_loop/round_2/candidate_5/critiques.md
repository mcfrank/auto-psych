# Critique of `avoidable_ambiguity_speaker`

8 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **four_object_multifeature_target_vs_all_feature_rate** — Rate of choosing the multi-feature pragmatic target possessing at least two features in four-object displays where the utterance matches exactly two candidates and the competitor possesses all features. (observed 0.773 vs null mean 0.64, z=4.30, p=0.002, q=0.00533) [survives FDR]
- **mumble_full_duplicate_choice_rate** — Rate of choosing a duplicated candidate possessing all features on mumble trials where identical objects are present. (observed 0.587 vs null mean 0.505, z=3.68, p=0.002, q=0.00533) [survives FDR]
- **mumble_singleton_choice_rate** — Rate of choosing the unique singleton object on mumble trials where duplicate objects are present in the display. (observed 0.607 vs null mean 0.665, z=-3.41, p=0.002, q=0.00533) [survives FDR]
- **unique_alternative_target_choice_rate** — Rate of choosing the pragmatic target in displays where the target possesses a unique alternative feature while the competitor does not. (observed 0.652 vs null mean 0.405, z=3.34, p=0.004, q=0.00639) [survives FDR]
- **four_object_two_feature_target_vs_three_feature_competitor_rate** — Rate of choosing the two-feature pragmatic target in four-object displays where the heard utterance matches exactly two candidates and the competitor possesses three features. (observed 0.685 vs null mean 0.533, z=3.10, p=0.004, q=0.00639) [survives FDR]
- **three_object_one_feature_target_vs_two_feature_competitor_rate** — Rate of choosing the single-feature pragmatic target in three-object displays where the heard utterance matches exactly two candidates and the competitor possesses two features. (observed 0.562 vs null mean 0.741, z=-2.75, p=0.016, q=0.0183) [survives FDR]
- **mumble_max_feature_choice_rate** — Rate of choosing the object with the maximal number of features on mumble trials where objects differ in feature counts. (observed 0.628 vs null mean 0.596, z=2.42, p=0.012, q=0.016) [survives FDR]
- **three_candidate_competitor_choice_rate** — Rate of choosing the maximal-feature competitor among displays where exactly three objects match the heard utterance. (observed 0.273 vs null mean 0.249, z=2.38, p=0.024, q=0.024) [survives FDR]
