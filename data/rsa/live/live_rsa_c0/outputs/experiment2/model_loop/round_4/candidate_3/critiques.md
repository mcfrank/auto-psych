# Critique of `prototype_typicality_bounded_listener`

8 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **four_object_multifeature_target_vs_all_feature_rate** — Rate of choosing the multi-feature pragmatic target possessing at least two features in four-object displays where the utterance matches exactly two candidates and the competitor possesses all features. (observed 0.773 vs null mean 0.573, z=4.93, p=0.002, q=0.00533) [survives FDR]
- **mumble_singleton_choice_rate** — Rate of choosing the unique singleton object on mumble trials where duplicate objects are present in the display. (observed 0.626 vs null mean 0.708, z=-4.70, p=0.002, q=0.00533) [survives FDR]
- **mumble_full_duplicate_choice_rate** — Rate of choosing a duplicated candidate possessing all features on mumble trials where identical objects are present. (observed 0.598 vs null mean 0.493, z=4.03, p=0.002, q=0.00533) [survives FDR]
- **all_match_max_feature_choice_rate** — Rate of choosing the object with the maximal feature count when the heard utterance is true of all objects in the display. (observed 0.295 vs null mean 0.252, z=3.30, p=0.004, q=0.00799) [survives FDR]
- **four_object_three_candidate_max_feature_rate** — Rate of choosing the maximal-feature candidate in four-object displays where the heard utterance matches exactly three objects. (observed 0.31 vs null mean 0.26, z=2.93, p=0.00799, q=0.0128) [survives FDR]
- **duplicate_matching_candidate_choice_rate** — Rate of choosing a duplicated candidate in displays where the heard utterance matches both a duplicated object and a unique object. (observed 0.378 vs null mean 0.341, z=2.83, p=0.014, q=0.0186) [survives FDR]
- **three_object_distractor_intrusion_rate** — Rate of choosing the non-matching distractor (solitary foil) in three-object displays where two objects match the utterance. (observed 0.0452 vs null mean 0.0256, z=2.34, p=0.038, q=0.038) [survives FDR]
- **two_object_target_vs_full_competitor_rate** — Rate of choosing the minimal-feature pragmatic target in two-object displays where the utterance matches both objects and the competitor possesses all features. (observed 0.639 vs null mean 0.7, z=-2.26, p=0.032, q=0.0365) [survives FDR]
