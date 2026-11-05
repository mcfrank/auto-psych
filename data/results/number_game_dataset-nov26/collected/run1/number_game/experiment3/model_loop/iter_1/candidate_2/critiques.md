# Critique of `whole_set_round_category`

3 of 9 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **cell_polarization_fraction** — Fraction of (set, target) cells (9 answers each) whose yes-rate is <=1/9 or >=8/9; observed > null means people's answers are more consensual/all-or-none per item (deterministic rule use) than the model's graded probabilities produce, observed < null means more item-level ambivalence. (observed 0.373 vs null mean 0.324, z=3.12, p=0.002, q=0.018) [survives FDR]
- **near_target_set_size_contrast** — Yes-rate for near targets (1-9 from the nearest example, no shared final digit) in single-example sets minus in 3-4-example sets; observed > null means extra examples dilute near-target support more than the model's n^-kappa averaging predicts, observed < null means the model's averaging over-dilutes near support in larger sets. (observed 0.178 vs null mean 0.0586, z=2.64, p=0.012, q=0.0539)
- **odd_vs_even_set_parity_violation** — In parity-homogeneous sets of >=3 examples (excluding sets whose examples all end in 0 or 5), yes-rate for opposite-parity targets (>=10 away, no shared final digit) in all-odd sets minus that in all-even sets; observed > null means people treat 'odd' as a much weaker/less salient rule than 'even' (the model's parity attention is symmetric), observed < null the reverse. (observed -0.00869 vs null mean 0.102, z=-2.35, p=0.03, q=0.0899)
