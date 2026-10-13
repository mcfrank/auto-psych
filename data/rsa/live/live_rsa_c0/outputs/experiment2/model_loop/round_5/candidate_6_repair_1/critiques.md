# Critique of `prototype_typicality_bounded_listener`

7 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **four_object_multifeature_target_vs_all_feature_rate** — Rate of choosing the multi-feature pragmatic target possessing at least two features in four-object displays where the utterance matches exactly two candidates and the competitor possesses all features. (observed 0.773 vs null mean 0.569, z=4.98, p=0.002, q=0.00533) [survives FDR]
- **mumble_singleton_choice_rate** — Rate of choosing the unique singleton object on mumble trials where duplicate objects are present in the display. (observed 0.626 vs null mean 0.71, z=-4.54, p=0.002, q=0.00533) [survives FDR]
- **mumble_full_duplicate_choice_rate** — Rate of choosing a duplicated candidate possessing all features on mumble trials where identical objects are present. (observed 0.598 vs null mean 0.492, z=3.87, p=0.002, q=0.00533) [survives FDR]
- **all_match_max_feature_choice_rate** — Rate of choosing the object with the maximal feature count when the heard utterance is true of all objects in the display. (observed 0.295 vs null mean 0.253, z=3.15, p=0.00799, q=0.0128) [survives FDR]
- **four_object_three_candidate_max_feature_rate** — Rate of choosing the maximal-feature candidate in four-object displays where the heard utterance matches exactly three objects. (observed 0.31 vs null mean 0.261, z=2.91, p=0.00799, q=0.0128) [survives FDR]
- **duplicate_matching_candidate_choice_rate** — Rate of choosing a duplicated candidate in displays where the heard utterance matches both a duplicated object and a unique object. (observed 0.378 vs null mean 0.341, z=2.76, p=0.00999, q=0.0133) [survives FDR]
- **three_object_distractor_intrusion_rate** — Rate of choosing the non-matching distractor (solitary foil) in three-object displays where two objects match the utterance. (observed 0.0452 vs null mean 0.0257, z=2.36, p=0.046, q=0.0525)
