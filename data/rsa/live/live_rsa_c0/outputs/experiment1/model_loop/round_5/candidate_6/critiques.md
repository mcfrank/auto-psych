# Critique of `candidate_load_attention_listener`

8 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **single_foil_choice_rate** — Rate of choosing the solitary non-matching foil object on utterance trials in displays with at least three objects where the heard word is true of all objects except exactly one. (observed 0.0617 vs null mean 0.0337, z=3.83, p=0.002, q=0.00266) [survives FDR]
- **three_match_intermediate_feature_rate** — Rate of choosing the intermediate feature-count matching object on utterance trials where exactly three objects match the heard word with three strictly distinct feature counts. (observed 0.0923 vs null mean 0.228, z=-3.68, p=0.002, q=0.00266) [survives FDR]
- **two_object_mumble_feature_rich_rate** — Rate of choosing the feature-richer object on mumble (prior) trials in two-object displays where objects have unequal feature counts. (observed 0.75 vs null mean 0.599, z=3.55, p=0.002, q=0.00266) [survives FDR]
- **two_object_scalar_implicature_rate** — Rate of choosing the sparser object (pragmatic scalar target) on utterance trials in two-object displays where the heard word is true of both objects with unequal feature counts. (observed 0.63 vs null mean 0.736, z=-3.54, p=0.002, q=0.00266) [survives FDR]
- **mumble_singleton_choice_rate** — Rate of choosing a unique singleton object on mumble (prior) trials in displays containing both singleton objects and duplicate objects. (observed 0.644 vs null mean 0.731, z=-3.49, p=0.002, q=0.00266) [survives FDR]
- **utterance_singleton_over_duplicate_rate** — Rate of choosing a unique singleton object on utterance trials where the heard word is true of both a singleton object and at least one duplicate object. (observed 0.594 vs null mean 0.673, z=-3.48, p=0.004, q=0.00457) [survives FDR]
- **ubiquitous_utterance_max_feature_rate** — Rate of choosing an object with the maximum number of features on utterance trials where the heard word is true of every object in the display. (observed 0.332 vs null mean 0.259, z=3.34, p=0.002, q=0.00266) [survives FDR]
- **popout_literal_accuracy** — Rate of choosing the unique matching object on popout utterance trials where the heard word is true of exactly one object in the display. (observed 1 vs null mean 0.987, z=2.17, p=0.00799, q=0.00799) [survives FDR]
