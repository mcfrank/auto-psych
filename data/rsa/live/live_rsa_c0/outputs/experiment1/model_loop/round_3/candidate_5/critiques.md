# Critique of `limited_attention_listener`

8 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **ubiquitous_utterance_max_feature_rate** — Rate of choosing an object with the maximum number of features on utterance trials where the heard word is true of every object in the display. (observed 0.332 vs null mean 0.199, z=7.09, p=0.002, q=0.004) [survives FDR]
- **redundant_display_scalar_target_rate** — Rate of choosing the sparser matching object (scalar target) on utterance trials in displays with redundant feature dimensions where the heard word is true of multiple objects with unequal feature counts. (observed 0.678 vs null mean 0.75, z=-5.83, p=0.002, q=0.004) [survives FDR]
- **two_object_scalar_implicature_rate** — Rate of choosing the sparser object (pragmatic scalar target) on utterance trials in two-object displays where the heard word is true of both objects with unequal feature counts. (observed 0.63 vs null mean 0.785, z=-5.69, p=0.002, q=0.004) [survives FDR]
- **utterance_singleton_over_duplicate_rate** — Rate of choosing a unique singleton object on utterance trials where the heard word is true of both a singleton object and at least one duplicate object. (observed 0.594 vs null mean 0.681, z=-4.00, p=0.002, q=0.004) [survives FDR]
- **single_foil_choice_rate** — Rate of choosing the solitary non-matching foil object on utterance trials in displays with at least three objects where the heard word is true of all objects except exactly one. (observed 0.0617 vs null mean 0.0336, z=3.87, p=0.00599, q=0.00685) [survives FDR]
- **three_object_scalar_rich_competitor_rate** — Rate of choosing the feature-richer logical competitor on utterance trials in three-object displays where the heard word is true of two objects with unequal feature counts. (observed 0.273 vs null mean 0.2, z=3.19, p=0.00599, q=0.00685) [survives FDR]
- **two_object_mumble_feature_rich_rate** — Rate of choosing the feature-richer object on mumble (prior) trials in two-object displays where objects have unequal feature counts. (observed 0.75 vs null mean 0.625, z=2.87, p=0.004, q=0.00639) [survives FDR]
- **mumble_singleton_choice_rate** — Rate of choosing a unique singleton object on mumble (prior) trials in displays containing both singleton objects and duplicate objects. (observed 0.644 vs null mean 0.71, z=-2.72, p=0.00799, q=0.00799) [survives FDR]
