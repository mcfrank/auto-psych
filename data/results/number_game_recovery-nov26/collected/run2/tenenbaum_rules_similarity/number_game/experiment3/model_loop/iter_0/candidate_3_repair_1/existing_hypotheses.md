# Existing hypotheses

Each model below is ONE cognitive hypothesis, with how it stands on the current data by ELPD-LOO (best first). `elpd_diff ± dse` is a model's deficit against the best and the standard error of that difference (with the responses to each stimulus pair counted together, as they are correlated): within about 2·dse the two are statistically tied on this data; beyond it the model has lost. "PSIS-LOO unreliable" means the estimate itself is untrustworthy (too many high-Pareto-k trials), not that the model is bad. Propose a hypothesis that is genuinely different from these, or a refinement of a single one of them — never a combination of several.

## no_digit_containment_span_concepts  — rank 0, the best model on this data, ELPD-LOO -2008.4

People do the same Bayesian concept learning as the current best model (arithmetic and final-digit rules plus multiples of 13-50, rule runs confined to a stretch of the number line, ranges of a typical width, an inferred share of random examples, examples drawn without replacement), except that they never entertain "contains the digit d" concepts: a digit that merely appears somewhere in the written numbers (such as a shared tens digit) is not noticed as a pattern. This simpler concept space predicts that near targets get no extra "yes" for sharing an example's decade and that sets like 8, 80, 48 do not generalise to other numbers containing an 8.

## above_max_stricter_criterion  — no comparison row

People do the same Bayesian concept learning as the current best model, but when they turn their belief into a yes/no answer they demand more certainty for a target larger than every example shown: the largest example acts as a soft ceiling, so going beyond it needs stronger evidence than going below the smallest. This one decision-stage criterion shift (its size inferred, zero recovering the best model) predicts fewer "yes" answers just above the largest example than just below the smallest one, at equal belief that the concept contains the target.

## off_by_one_sequence_concepts  — no comparison row

People do the same Bayesian concept learning as the current best model (arithmetic and final-digit rules plus multiples of 13-50, rule runs confined to a stretch of the number line, ranges of a typical width, an inferred share of random examples, examples drawn without replacement), but they also entertain "near-miss" sequences: one more or one less than the members of a familiar sparse sequence (squares, cubes, powers of 2 to 10), such as 2, 5, 10, 17, 26 (squares plus one) or 1, 3, 7, 15, 31, 63 (powers of two minus one). Each such shifted sequence is equally likely a priori, the family taking an inferred share of the rule prior, so sets like 31, 3, 1, 15 or 33, 17, 5, 9 generalise sharply to the other members of the shifted sequence (7, 63 or 65) and not to broad rules such as odd numbers.

## feature_prototype_typicality  — no comparison row

People do not infer a rule; they abstract a prototype of the example set — a summary of how often each familiar number property (odd/even, prime, square, cube, power, multiple of 3-12) and each final digit occurs among the examples, plus the examples' average magnitude and spread — with each property tracked independently and the estimates hedged toward how common the property is among all numbers when there are few examples. They say "yes" to the degree that the target looks more typical of this prototype than of numbers in general (the summed property-by-property and magnitude likelihood ratio), so a target lacking a property every example shares is rejected more firmly the more examples showed it, and a target near the examples' centre is favoured whatever rule it fits.
