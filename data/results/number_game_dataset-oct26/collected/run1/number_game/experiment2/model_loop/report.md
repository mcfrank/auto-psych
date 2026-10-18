# Inner Model Loop Report

Each model below is ONE distinct cognitive hypothesis. The posterior mass shows which single hypothesis best explains the data — it is **not** a recipe to combine the top models into a blend.

- Best model: **mutual_similarity_contrast_standard** (posterior=0.741, elpd_loo=-3152.98)
- Trials: 5040
- Models compared: 7

## Posterior over models (ELPD-LOO)

| model | posterior | elpd_loo |
| --- | --- | --- |
| mutual_similarity_contrast_standard | 0.7410 | -3152.98 |
| coincidence_graded_rule_preemption | 0.2276 | -3154.16 |
| rule_gated_mixed_parity_criterion | 0.0270 | -3156.29 |
| rule_gated_decade_familiarity | 0.0033 | -3158.38 |
| position_weighted_rule_gated_digits | 0.0010 | -3159.59 |
| ruleless_set_anything_goes_floor | 0.0000 | -3163.30 |
| mixed_parity_loosens_criterion | 0.0000 | -3168.44 |

## Hypotheses

- **mutual_similarity_contrast_standard**: People judge a target's resemblance to the examples relative to how much the examples resemble one another on the same final-digit metric (same digit most alike, same parity partly, opposite parity least): the examples' mutual similarity sets the standard a target must meet. When the examples closely resemble each other (shared digits or parity), a target that matches none of their final digits falls well short of that standard and is rejected. When the examples share little (mixed digits and parities), even a dissimilar target is about as alike to them as they are to each other, so people accept broadly. Apart from this, judgments follow the current best model: rule-pre-empted parity credit, a common-factor boost, and rule-gated decade-band familiarity.
- **coincidence_graded_rule_preemption**: A refinement of `rule_gated_decade_familiarity` (final-digit similarity with an "ends in that digit" rule pre-empting parity credit, a common-factor boost, and decade-band familiarity that a detected rule pre-empts, all kept). The one change: rule detection is not all-or-none but graded by how unlikely the examples' shared property would be by coincidence — a shared final digit (1 in 10 per extra example), a common factor g (1 in g per extra example) or merely a shared parity (1 in 2 per extra example), with evidence piling up across examples — and the more suspicious the coincidence, the more firmly people believe in a rule and the more completely it pre-empts "numbers near these" magnitude familiarity. So four examples sharing a digit or a large factor shut off neighbourhood generalization almost entirely, two examples sharing only parity or a factor of 3 barely do, and sets with no shared property at all keep magnitude familiarity in full, spreading generalization breadth across sets more than a binary rule detector does.
- **rule_gated_mixed_parity_criterion**: A refinement of `mixed_parity_loosens_criterion` (final-digit similarity with parity credit suppressed by a shared final digit, a common-factor boost, decade-band familiarity, and a looser "yes" criterion when the examples mix odd and even numbers, all kept). The one change, a component taken from `rule_gated_decade_familiarity`, is that magnitude familiarity is only a fallback: when two or more examples obey an explicit rule (all end in the same digit, or all are multiples of a common factor of 3 or more), the rule pre-empts "numbers near these" and the decade-band boost is cut by an inferred share, whereas rule-less sets and lone examples get it in full. Together with the mixed-parity criterion shift, this predicts that rule-less, mixed-parity sets are accepted broadly — even far from the examples' decades — while rule-bound sets are rejected outside their rule, widening the differences between sets that the critique reports.
- **rule_gated_decade_familiarity**: A refinement of `shared_digit_rule_preempts_parity` (final-digit similarity averaged over the examples, an abstracted "ends in that digit" rule pre-empting parity credit, a common-factor boost and decade-band familiarity all kept). The one change is that magnitude familiarity is a fallback: when people detect an explicit rule linking the examples — they all end in the same digit, or are all multiples of a common factor of 3 or more — the rule pre-empts "numbers near these", and the decade-band familiarity boost is cut by an inferred share; when the set has no detectable rule (a lone example, or examples with no shared digit or factor), people lean on magnitude neighbourhood at full strength. This addresses the critiques that people accept same-decade, different-digit targets after a single example more than predicted, and that generalization breadth differs between sets more than the model produces.
- **position_weighted_rule_gated_digits**: People judge the target as the current best model says (final-digit similarity to the examples with parity credit pre-empted by a shared final digit, a boost for multiples of the examples' common factor, and decade-band familiarity that an explicit rule pre-empts), but they do not hold every example equally in memory when comparing final digits: each example's weight in the digit comparison changes steadily with its serial position, from the first shown to the last (recency if later examples count more, primacy if the first sets the impression), with the direction and strength inferred. So a target sharing the final digit of the most heavily weighted example is accepted more than one matching an equally similar example in a weaker position, and the same numbers in a different order shift which digit matches drive "yes".
- **ruleless_set_anything_goes_floor**: A refinement of `rule_gated_decade_familiarity` (final-digit similarity with rule-pre-empted parity credit, common-factor boost, and decade-band familiarity cut back when an explicit rule is found, all kept). The one change: when two or more examples show no detectable rule (mixed final digits and no common factor of 3 or more), people entertain, with an inferred probability, the hypothesis that the program simply produces any number at all; under that "anything goes" belief they say "yes" regardless of the target, so rule-less multi-example sets get a floor of acceptance that lifts most the targets that resemble the examples least (far, unmatched targets), while lone examples and rule-bearing sets keep the incumbent's judgments. This addresses the critique that people accept far, digit-unmatched targets of rule-less sets more than predicted and that generalization breadth differs between sets more than the model produces.
- **mixed_parity_loosens_criterion**: A refinement of `final_digit_rule_suppresses_parity` (final-digit similarity with parity credit suppressed by a shared final digit, common-factor boost and decade-band familiarity all kept). The one change: whether the examples agree in parity sets how broad people think the program is — when the examples mix odd and even numbers, people conclude the program is not bound by any parity or digit rule and lower their criterion, saying "yes" more readily to every target, whereas examples that all share a parity signal a constrained program and keep the criterion strict. This addresses the critique that generalization breadth differs between sets more than the incumbent produces: mixed-parity sets (e.g. 92, 14, 20, 5) are accepted broadly while same-parity sets (e.g. 68, 58) are rejected almost everywhere outside their rule.

## Distinguishability (arviz.compare, PSIS-LOO)

`elpd_diff` and `dse` are relative to the best model. A model is only clearly worse than the best when `elpd_diff > 2 * dse`; models within ~2·dse of the top are statistically indistinguishable. `LOO reliable` is False when PSIS-LOO flagged this model's estimate as untrustworthy (many high Pareto-k points) — its row should be read with caution.

| model | elpd_diff | dse | distinguishable from best | weight | LOO reliable |
| --- | --- | --- | --- | --- | --- |
| mutual_similarity_contrast_standard ←selected | 0.00 | 0.00 | — (best) | 0.741 | yes |
| coincidence_graded_rule_preemption | 1.18 | 4.00 | no (within ~2·dse) | 0.228 | yes |
| rule_gated_mixed_parity_criterion | 3.31 | 4.24 | no (within ~2·dse) | 0.027 | yes |
| rule_gated_decade_familiarity | 5.40 | 3.61 | no (within ~2·dse) | 0.003 | yes |
| position_weighted_rule_gated_digits | 6.61 | 3.95 | no (within ~2·dse) | 0.001 | yes |
| ruleless_set_anything_goes_floor | 10.33 | 4.35 | yes | 0.000 | yes |
| mixed_parity_loosens_criterion | 15.46 | 7.01 | yes | 0.000 | yes |
