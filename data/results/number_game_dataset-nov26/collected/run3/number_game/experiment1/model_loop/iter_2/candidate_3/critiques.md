# Critique of `tempered_size_principle_independent`

1 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **unsupported_far_target_floor** — Yes rate for non-example targets that lie outside the [min,max] example range, more than 5 away from every example, and belong to no small rule (<=20 members, of the model's 45) containing all examples; observed above null means people have a guessing/lapse floor the model lacks, below null means they reject such targets even more firmly. (observed 0.384 vs null mean 0.465, z=-2.26, p=0.032, q=0.256)
