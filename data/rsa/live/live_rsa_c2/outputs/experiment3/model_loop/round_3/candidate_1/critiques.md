# Critique of `competition_endpoint_binding_lapse_l2`

8 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **complex_false_object_rate** — On word trials, the proportion of choices falling on false objects possessing high visual complexity (at least 2 features, the highest feature count among non-matching objects in the display). (observed 0.0298 vs null mean 0.0166, z=4.85, p=0.002, q=0.00266) [survives FDR]
- **mumble_full_over_empty_rate** — On mumble trials contrasting completely empty (0 features) and completely full (all features) objects, the proportion of choices falling on the full object. (observed 0.704 vs null mean 0.612, z=4.72, p=0.002, q=0.00266) [survives FDR]
- **word_intermediate_match_rate** — On word trials where matching candidates span at least three distinct feature counts, the proportion of choices falling on strictly intermediate-complexity matching objects. (observed 0.106 vs null mean 0.178, z=-4.13, p=0.002, q=0.00266) [survives FDR]
- **non_nested_fewer_features_rate** — On word trials where matching candidates have different feature counts and cross-cutting (non-nested) features, the proportion of choices for the candidate with the fewest total features. (observed 0.62 vs null mean 0.527, z=4.12, p=0.002, q=0.00266) [survives FDR]
- **mumble_intermediate_rate** — Proportion of choices on mumble trials falling on strictly intermediate-complexity objects whose feature count is strictly between the display minimum and maximum. (observed 0.0851 vs null mean 0.114, z=-4.07, p=0.002, q=0.00266) [survives FDR]
- **mumble_singleton_when_duplicates** — On mumble trials in displays containing both unique singleton objects and duplicate objects, the proportion of choices falling on unique singleton objects. (observed 0.575 vs null mean 0.641, z=-3.93, p=0.002, q=0.00266) [survives FDR]
- **two_object_subset_target_rate** — On 2-object word trials where candidate objects have a strict subset-superset feature relationship, the proportion of choices for the minimal subset object (the pragmatic target). (observed 0.664 vs null mean 0.728, z=-2.99, p=0.004, q=0.00457) [survives FDR]
- **ambiguous_word_singleton_rate** — On word trials where the heard utterance is true of all objects in the display and candidates include both a unique singleton and duplicate objects, the proportion of choices for the singleton object. (observed 0.651 vs null mean 0.686, z=-2.25, p=0.032, q=0.032) [survives FDR]
