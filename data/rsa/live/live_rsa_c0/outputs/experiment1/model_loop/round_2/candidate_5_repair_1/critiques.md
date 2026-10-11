# Critique of `rsa_l2_confusion_singleton_l0`

8 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **four_object_foil_error_rate** — Rate of choosing a non-matching foil object (an object lacking the heard feature) on utterance trials in four-object displays. (observed 0.0369 vs null mean 0.0169, z=5.30, p=0.002, q=0.00266) [survives FDR]
- **utterance_singleton_over_duplicate_rate** — Rate of choosing a singleton matching object on utterance trials where the heard word is true of both a singleton object and at least one duplicate object. (observed 0.594 vs null mean 0.705, z=-4.83, p=0.002, q=0.00266) [survives FDR]
- **ubiquitous_utterance_max_feature_rate** — Rate of choosing the object with the maximum number of features on utterance trials where the heard word is true of every object in the display. (observed 0.332 vs null mean 0.236, z=4.56, p=0.002, q=0.00266) [survives FDR]
- **mumble_singleton_choice_rate** — Rate of choosing a unique singleton object on mumble (prior) trials where displays contain both singletons and duplicate objects. (observed 0.644 vs null mean 0.737, z=-4.10, p=0.002, q=0.00266) [survives FDR]
- **zero_feature_foil_selection_rate** — Rate of choosing a zero-feature object on utterance trials where the display contains an object with no features. (observed 0.0372 vs null mean 0.0134, z=3.92, p=0.002, q=0.00266) [survives FDR]
- **two_object_scalar_implicature_rate** — Rate of choosing the sparser object (pragmatic scalar target) in two-object displays where the heard word is true of both objects with unequal feature counts. (observed 0.63 vs null mean 0.732, z=-3.27, p=0.004, q=0.00457) [survives FDR]
- **catch_trial_literal_accuracy** — Accuracy of choosing the matching target object on forced-choice catch trials where the heard utterance is true of exactly one object. (observed 1 vs null mean 0.971, z=3.22, p=0.002, q=0.00266) [survives FDR]
- **two_object_mumble_feature_rich_rate** — Rate of choosing the feature-richer object on mumble (prior) trials in two-object displays where objects have unequal feature counts. (observed 0.75 vs null mean 0.649, z=2.36, p=0.02, q=0.02) [survives FDR]
