# Critique of `rsa_l1_singleton_feat_color_valence_l0`

8 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **universal_word_singleton_rating** — Mean Likert rating of a probed visual singleton when the heard word is true of all objects in the scene and distractors are duplicated; tests under-prediction from M-implicature penalty. (observed 5.99 vs null mean 5.36, z=9.83, p=0.002, q=0.002) [survives FDR]
- **asymmetric_5object_contrast_gap** — Difference in mean Likert rating between minimal target and duplicate competitor in 5-object asymmetric displays; tests under-production of pragmatic contrast in large scenes. (observed 4.82 vs null mean 3.12, z=8.15, p=0.002, q=0.002) [survives FDR]
- **scalar_competitor_unique_feature_rating** — Mean Likert rating of a probed scalar competitor possessing an unmentioned unique feature when distractors sharing the heard word are duplicated; tests spurious singleton prior preservation. (observed 2.59 vs null mean 3.27, z=-7.17, p=0.002, q=0.002) [survives FDR]
- **redundant_multifeature_competitor_rating** — Mean Likert rating of a probed scalar competitor with redundant features when a minimal referent exists; tests lack of feature parsimony on informative trials. (observed 3.04 vs null mean 3.52, z=-5.79, p=0.002, q=0.002) [survives FDR]
- **scalar_implicature_minimal_vs_maximal_gap** — Difference in mean Likert rating between minimal referent and maximal competitor in 4-object scalar implicature scenes; tests over-polarization of scalar implicatures. (observed 2.81 vs null mean 3.21, z=-4.68, p=0.002, q=0.002) [survives FDR]
- **maximal_referent_duplicate_distractors_rating** — Mean Likert rating of a probed maximal referent when distractors sharing the heard word are duplicated minimal objects; tests under-prediction of maximal referent in duplicate scenes. (observed 3.35 vs null mean 2.9, z=4.63, p=0.002, q=0.002) [survives FDR]
- **prior_singleton_rating** — Mean Likert rating of a probed visual singleton on uninformative prior trials with no word heard; tests over-confidence of singleton prior in continuous likelihood ratings. (observed 4.36 vs null mean 4.64, z=-4.31, p=0.002, q=0.002) [survives FDR]
- **high_ambiguity_informative_rating** — Mean Likert rating across informative trials where the heard word is true of 4 or more objects; tests under-prediction of candidate likelihood under broad referential ambiguity. (observed 3.79 vs null mean 3.58, z=3.97, p=0.002, q=0.002) [survives FDR]
