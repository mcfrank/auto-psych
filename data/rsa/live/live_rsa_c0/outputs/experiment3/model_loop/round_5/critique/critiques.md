# Critique of `boundary_polarization_listener`

8 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **three_object_target_vs_duplicate_competitor_rate** — Rate of choosing the pragmatic target in three-object displays where the heard utterance matches the target and a duplicated competitor. (observed 0.677 vs null mean 0.812, z=-4.75, p=0.002, q=0.0032) [survives FDR]
- **mumble_full_duplicate_choice_rate** — Rate of choosing a duplicated candidate possessing all features on mumble trials where identical objects are present. (observed 0.587 vs null mean 0.484, z=4.62, p=0.002, q=0.0032) [survives FDR]
- **mumble_singleton_choice_rate** — Rate of choosing the unique singleton object on mumble trials where duplicate objects are present in the display. (observed 0.607 vs null mean 0.682, z=-4.43, p=0.002, q=0.0032) [survives FDR]
- **four_object_multifeature_target_vs_all_feature_rate** — Rate of choosing the multi-feature pragmatic target possessing at least two features in four-object displays where the utterance matches exactly two candidates and the competitor possesses all features. (observed 0.773 vs null mean 0.638, z=4.11, p=0.002, q=0.0032) [survives FDR]
- **two_object_one_feature_target_vs_three_feature_competitor_rate** — Rate of choosing the single-feature pragmatic target in two-object displays where the heard utterance matches both objects and the competitor possesses three features. (observed 0.603 vs null mean 0.744, z=-3.92, p=0.002, q=0.0032) [survives FDR]
- **unique_alternative_target_choice_rate** — Rate of choosing the pragmatic target in displays where the target possesses a unique alternative feature while the competitor does not. (observed 0.652 vs null mean 0.433, z=3.06, p=0.00799, q=0.00913) [survives FDR]
- **four_object_two_feature_target_vs_three_feature_competitor_rate** — Rate of choosing the two-feature pragmatic target in four-object displays where the heard utterance matches exactly two candidates and the competitor possesses three features. (observed 0.685 vs null mean 0.54, z=2.88, p=0.00799, q=0.00913) [survives FDR]
- **all_match_max_feature_choice_rate** — Rate of choosing the object with the maximal number of features in displays with at least three objects where the heard utterance matches all objects. (observed 0.271 vs null mean 0.237, z=2.58, p=0.016, q=0.016) [survives FDR]
