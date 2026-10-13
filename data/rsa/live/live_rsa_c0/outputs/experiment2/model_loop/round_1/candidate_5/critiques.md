# Critique of `inhibitory_load_listener`

8 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **pragmatic_target_rate_two_object** — Rate of choosing the minimal-feature matching object (the pragmatic target) in two-object displays where an utterance is referentially ambiguous. (observed 0.629 vs null mean 0.732, z=-4.37, p=0.002, q=0.0032) [survives FDR]
- **mumble_singleton_choice_rate** — Rate of choosing the unique singleton object on mumble trials where duplicate objects are present in the display. (observed 0.626 vs null mean 0.708, z=-4.32, p=0.002, q=0.0032) [survives FDR]
- **all_match_max_feature_choice_rate** — Rate of choosing the object with the maximal feature count when the heard utterance is true of all objects in the display. (observed 0.295 vs null mean 0.242, z=4.19, p=0.002, q=0.0032) [survives FDR]
- **alternative_word_competitor_choice_rate** — Rate of choosing the competitor candidate possessing an unmentioned unique feature (an unambiguous alternative word) in displays with two matching candidates. (observed 0.331 vs null mean 0.26, z=3.53, p=0.002, q=0.0032) [survives FDR]
- **mumble_max_feature_choice_rate** — Rate of choosing the object with the maximal number of features on mumble trials where objects differ in feature counts. (observed 0.637 vs null mean 0.586, z=3.24, p=0.002, q=0.0032) [survives FDR]
- **pragmatic_target_rate_four_object_two_match** — Rate of choosing the minimal-feature matching candidate (the pragmatic target) in four-object displays where the utterance matches exactly two candidates. (observed 0.708 vs null mean 0.655, z=2.75, p=0.004, q=0.00533) [survives FDR]
- **three_object_distractor_intrusion_rate** — Rate of choosing the non-matching distractor (solitary foil) in three-object displays where two objects match the utterance. (observed 0.0452 vs null mean 0.0263, z=2.29, p=0.03, q=0.03) [survives FDR]
- **three_candidate_competitor_choice_rate** — Rate of choosing the maximal-feature competitor among displays where exactly three objects match the heard utterance. (observed 0.291 vs null mean 0.262, z=2.28, p=0.016, q=0.0183) [survives FDR]
