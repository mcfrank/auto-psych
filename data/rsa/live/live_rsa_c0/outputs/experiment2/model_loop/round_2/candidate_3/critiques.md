# Critique of `inhibitory_codability_listener`

8 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **mumble_singleton_choice_rate** — Rate of choosing the unique singleton object on mumble trials where duplicate objects are present in the display. (observed 0.626 vs null mean 0.71, z=-4.30, p=0.002, q=0.00799) [survives FDR]
- **all_match_max_feature_choice_rate** — Rate of choosing the object with the maximal feature count when the heard utterance is true of all objects in the display. (observed 0.295 vs null mean 0.251, z=3.47, p=0.002, q=0.00799) [survives FDR]
- **four_object_three_candidate_max_feature_rate** — Rate of choosing the maximal-feature candidate in four-object displays where the heard utterance matches exactly three objects. (observed 0.31 vs null mean 0.258, z=3.23, p=0.004, q=0.0107) [survives FDR]
- **duplicate_matching_candidate_choice_rate** — Rate of choosing a duplicated candidate in displays where the heard utterance matches both a duplicated object and a unique object. (observed 0.378 vs null mean 0.339, z=2.85, p=0.00999, q=0.016) [survives FDR]
- **pragmatic_target_rate_two_object** — Rate of choosing the minimal-feature matching object (the pragmatic target) in two-object displays where an utterance is referentially ambiguous. (observed 0.629 vs null mean 0.695, z=-2.63, p=0.012, q=0.016) [survives FDR]
- **four_object_target_vs_all_feature_rate** — Rate of choosing the minimal-feature matching candidate (the pragmatic target) in four-object displays where the utterance matches exactly two candidates and the competitor possesses all features. (observed 0.703 vs null mean 0.633, z=2.58, p=0.012, q=0.016) [survives FDR]
- **three_object_distractor_intrusion_rate** — Rate of choosing the non-matching distractor (solitary foil) in three-object displays where two objects match the utterance. (observed 0.0452 vs null mean 0.0257, z=2.31, p=0.05, q=0.05) [survives FDR]
- **alternative_word_competitor_choice_rate** — Rate of choosing the competitor candidate possessing an unmentioned unique feature (an unambiguous alternative word) in displays with two matching candidates. (observed 0.331 vs null mean 0.285, z=2.10, p=0.046, q=0.05) [survives FDR]
