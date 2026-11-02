# Critique of `set_size_evidence_privileged_fives`

1 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **round_target_bonus_no_round_examples** — Among sets containing no multiple of 10, yes rate for targets that are multiples of 10 minus yes rate for other targets; observed above null means people say yes to round numbers more than the model predicts when nothing in the examples supports them (a round-number salience), below null means less. (observed -0.0567 vs null mean 0.00408, z=-2.84, p=0.00799, q=0.0639)
