# Critique of `rsa_l2_singleton_feat_color_valence_l0`

8 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **ambiguous_word_foil_choice_rate** — Rate of choosing non-matching foil objects on utterance trials where the heard word is true of two or more objects. (observed 0.0274 vs null mean 0.0138, z=14.94, p=0.002, q=0.002) [survives FDR]
- **unambiguous_word_accuracy** — Rate of choosing the uniquely matching referent on utterance trials where the heard word is true of exactly one object. (observed 0.989 vs null mean 0.973, z=11.83, p=0.002, q=0.002) [survives FDR]
- **scalar_intermediate_competitor_rate** — Rate of choosing intermediate competitors on utterance trials where three or more objects match the heard word with a strict hierarchy of feature counts. (observed 0.107 vs null mean 0.271, z=-8.54, p=0.002, q=0.002) [survives FDR]
- **standard_scalar_logical_choice_rate** — Rate of choosing the logical competitor on three-object scalar implicature displays with one foil, one target, and one competitor matching the heard word. (observed 0.185 vs null mean 0.24, z=-5.75, p=0.002, q=0.002) [survives FDR]
- **twin_uniform_singleton_choice_rate** — Rate of choosing the unique singleton object on utterance trials in the E9 twins experiment where the heard word is true of all objects. (observed 0.383 vs null mean 0.694, z=-5.33, p=0.002, q=0.002) [survives FDR]
- **size_prior_max_feature_rate** — Rate of choosing the object with the maximal feature count on mumble trials in the size experiment. (observed 0.58 vs null mean 0.441, z=5.15, p=0.002, q=0.002) [survives FDR]
- **four_object_maximal_competitor_rate** — Rate of choosing the maximal four-feature competitor on utterance trials in four-object displays where the heard word matches both a one-feature target and a four-feature competitor. (observed 0.251 vs null mean 0.145, z=4.93, p=0.002, q=0.002) [survives FDR]
- **two_object_pragmatic_target_rate** — Rate of choosing the pragmatic target with fewer features on utterance trials with exactly two objects that both match the heard word. (observed 0.677 vs null mean 0.77, z=-3.39, p=0.002, q=0.002) [survives FDR]
