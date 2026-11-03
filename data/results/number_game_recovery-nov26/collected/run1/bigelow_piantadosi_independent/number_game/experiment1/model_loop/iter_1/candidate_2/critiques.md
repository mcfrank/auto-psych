# Critique of `tempered_size_principle_mixture`

4 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **partial_last_digit_match** — For sets whose examples do NOT all share a last digit, yes rate among non-example targets sharing a last digit with at least one example minus those sharing none; the model only rewards last-digit matches via full 'ends in d' rules, so observed above null means people use feature-level digit similarity to individual examples more than the model predicts. (observed 0.351 vs null mean 0.136, z=8.13, p=0.002, q=0.00799) [survives FDR]
- **far_rule_consistent_generalisation** — Yes rate among non-example targets more than 10 from every example that are divisible by the examples' common divisor (gcd >= 2) — i.e. distant targets licensed only by an arithmetic rule; observed above null means people generalise broad arithmetic rules to distant numbers more than the model predicts, below null means less. (observed 0.503 vs null mean 0.361, z=4.16, p=0.002, q=0.00799) [survives FDR]
- **proximity_gradient** — Yes rate among non-example targets within 3 of the nearest example minus among those 20+ away; observed above null means human generalisation falls off with numerical distance more steeply than the model's interval hypotheses predict, below null means less steeply. (observed 0.0135 vs null mean 0.077, z=-2.91, p=0.00599, q=0.012) [survives FDR]
- **cell_unanimity** — Fraction of (set, target) cells with >=2 responses where all responses agree; observed above null means people are more consistent with each other (more deterministic/extreme yes-probabilities) than the model's graded posterior predicts, below null means more disagreement. (observed 0.62 vs null mean 0.553, z=2.72, p=0.004, q=0.0107) [survives FDR]
