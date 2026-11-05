# Inner Model Loop Report

Each model below is ONE distinct cognitive hypothesis. The posterior mass shows which single hypothesis best explains the data — it is **not** a recipe to combine the top models into a blend.

- Best model: **coincidence_sharpened_exemplar** (posterior=0.551, elpd_loo=-1571.17)
- Trials: 2520
- Models compared: 8

## Posterior over models (ELPD-LOO)

| model | posterior | elpd_loo |
| --- | --- | --- |
| coincidence_sharpened_exemplar | 0.5510 | -1571.17 |
| digit_run_rule_attention_exemplar | 0.1074 | -1572.81 |
| digit_run_criterion_shared_factor | 0.1005 | -1572.87 |
| common_step_lattice_exemplar | 0.0979 | -1572.90 |
| digit_run_raised_criterion_exemplar | 0.0830 | -1573.07 |
| capacity_limited_attention_exemplar | 0.0573 | -1573.44 |
| shared_factor_parity_exemplar | 0.0025 | -1576.57 |
| divisibility_kinship_exemplar | 0.0005 | -1578.19 |

## Hypotheses

- **coincidence_sharpened_exemplar**: People generalize from the examples by summed similarity (magnitude, final digit, parity, shared factors), but how sharply similarity falls off is set by how suspicious a coincidence the examples are: the less likely it is that the examples would share their most specific common property (all the same final digit, all multiples of 5, all the same parity) by chance, the more specific they take the program to be. So they tighten their generalization on every dimension at once. This disagrees most with the best model on coherent sets, e.g. "78, 48, 88" or "15, 85, 45" (near non-matching targets rejected far more), "90, 45" (multiples of 5 favoured more sharply), and on how much sets differ in overall endorsement, while single examples and incoherent sets are generalized as before.
- **digit_run_rule_attention_exemplar**: Refinement of the incumbent `shared_factor_parity_exemplar`. People still judge a target by its summed similarity to the remembered examples in a space of magnitude, final digit, parity and shared factor, with attention to magnitude and final digit tuned by how variable the examples are. The one change, adapted from `digit_run_strict_criterion_exemplar` but placed inside the similarity space rather than the response criterion: when three or more examples all end in the same digit, people read the final digit as the program's rule, and their attention to it jumps. A target with a different final digit then loses much more similarity to every example, so after sets like "46, 76, 66" or "15, 85, 45" magnitude-near targets with the wrong digit are rejected far more often than graded digit attention allows. Targets that match the digit are not penalized.
- **digit_run_criterion_shared_factor**: Refinement of `digit_run_strict_criterion_exemplar`: people still judge a target by its summed similarity to the remembered examples (magnitude and final-digit attention tuned to how variable the examples are, a fixed parity penalty), and still raise their response criterion when three or more examples all end in the same digit, reading such a set as a narrow rule. The one change, grafted from `shared_factor_parity_exemplar`, is a fixed shared-factor dimension: an example contributes less similarity to a target that shares no common divisor with it (they are coprime). This predicts strict rejection of wrong-digit targets after same-digit runs (e.g. "62, 32, 22") together with endorsement of factor-sharing targets far from the examples (36 and 39 after "3, 63") and rejection of coprime ones (37 after "8, 32").
- **common_step_lattice_exemplar**: Refinement of `shared_factor_parity_exemplar`: people still judge a target by its summed similarity to the remembered examples (magnitude, final digit, parity and shared factors, with attention tuned to how variable the examples are), but they also read the spacing of the examples. When three or more examples are separated by gaps that are all multiples of one common step of at most 20 (e.g. step 10 in "78, 48, 88", step 6 in "63, 27, 99, 33", step 2 in "12, 14, 16"), people expect the program to move in that step, and a target that cannot be reached from the examples in whole steps feels dissimilar to every example by a fixed factor. So off-step targets near the examples are rejected more than similarity alone predicts, while sets with no such small common step are unaffected.
- **digit_run_raised_criterion_exemplar**: Refinement of the incumbent `shared_factor_parity_exemplar`. People still judge a target by its summed similarity to the remembered examples (magnitude, final digit, parity and shared factor, with attention tuned to how variable the examples are). The one change, grafted from `digit_run_strict_criterion_exemplar`: when three or more examples all end in the same digit, people read the set as coming from a narrow, rule-bound program and demand more similarity before saying yes to *any* target. So such sets are endorsed less across the board. That includes targets that do match the shared digit, which a sharper digit attention (`digit_run_rule_attention_exemplar`) would leave untouched. Wrong-digit targets near the examples are rejected far more often than the incumbent predicts.
- **capacity_limited_attention_exemplar**: People judge a target by its similarity to the remembered examples on magnitude, final digit, parity and shared factors, but their attention is a fixed, limited budget split across these dimensions: whatever the examples are consistent on (a shared final digit, a shared parity, a tight magnitude range) pulls attention onto that dimension and necessarily away from the others. So after a set that shares a final digit, people reject wrong-digit targets and also become more tolerant of magnitude distance for right-digit targets, because the attention spent on the digit is taken from magnitude, and the reverse holds after scattered-digit sets.
- **shared_factor_parity_exemplar**: Refinement of `variability_tuned_parity_attention`: people still judge a target by its summed similarity to the remembered examples in a space of magnitude, final digit and parity, with attention to each set by how variable the examples are on it. The one change is a further, fixed "shared factor" dimension: a target that has no common divisor with an example (they are coprime) loses similarity to that example by a constant factor. So after "3, 63" people endorse 36 and 39 (they share the factor 3) although these match neither example's final digit nor magnitude, and after "8, 32" they reject 37, unlike targets that share a factor with the examples.
- **divisibility_kinship_exemplar**: Refinement of `graded_parity_variability_exemplar`: people still judge a target by its summed similarity to the remembered examples. Attention to magnitude and final digit is still set by how variable the examples are, and parity is still a fixed dimension. The one change is that the parity dimension is widened into divisibility kinship: an example also contributes less similarity when it and the target have no common factor at all (they are coprime), so two numbers that share a factor such as 3, 5 or 7 feel related even when their parity differs. This predicts that after sets like "3, 63", "6, 66" or "55, 15, 75, 45", targets that share a factor with the examples (30, 36, 39, 60, 50) are endorsed well above what magnitude, final digit and parity alone allow, while targets coprime to the examples (e.g. 37 after "8, 32") are rejected more often.

## Distinguishability (arviz.compare, PSIS-LOO)

`elpd_diff` and `dse` are relative to the best model. A model is only clearly worse than the best when `elpd_diff > 2 * dse`; models within ~2·dse of the top are statistically indistinguishable. `LOO reliable` is False when PSIS-LOO flagged this model's estimate as untrustworthy (many high Pareto-k points) — its row should be read with caution.

| model | elpd_diff | dse | distinguishable from best | weight | LOO reliable |
| --- | --- | --- | --- | --- | --- |
| coincidence_sharpened_exemplar ←selected | 0.00 | 0.00 | — (best) | 0.551 | yes |
| digit_run_rule_attention_exemplar | 1.63 | 5.06 | no (within ~2·dse) | 0.107 | yes |
| digit_run_criterion_shared_factor | 1.70 | 5.23 | no (within ~2·dse) | 0.100 | yes |
| common_step_lattice_exemplar | 1.73 | 6.82 | no (within ~2·dse) | 0.098 | yes |
| digit_run_raised_criterion_exemplar | 1.89 | 5.10 | no (within ~2·dse) | 0.083 | yes |
| capacity_limited_attention_exemplar | 2.26 | 6.86 | no (within ~2·dse) | 0.057 | yes |
| shared_factor_parity_exemplar | 5.40 | 5.70 | no (within ~2·dse) | 0.002 | yes |
| divisibility_kinship_exemplar | 7.01 | 5.90 | no (within ~2·dse) | 0.000 | yes |
