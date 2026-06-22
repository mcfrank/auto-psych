# Critique of `inner_loop_model`

5 of 8 test statistics show a significant discrepancy (p ≤ 0.05), over 200 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **length_preference_for_balanced** — The choice rate for the longer sequence when both sequences are well-balanced (imbalance <= 0.25). (observed 0.538 vs null mean 0.457, z=3.70, p=0.00995, q=0.0265) [survives FDR]
- **exact_balance_preference** — The choice rate for exactly balanced sequences (imbalance=0) over slightly imbalanced ones (0 < imbalance <= 0.3). (observed 0.0844 vs null mean 0.102, z=-3.20, p=0.00995, q=0.0265) [survives FDR]
- **extreme_imbalance_penalty** — The choice rate for moderately imbalanced sequences (0.1 <= imbalance <= 0.3) over extremely imbalanced sequences (imbalance >= 0.5). (observed 0.106 vs null mean 0.0914, z=3.00, p=0.00995, q=0.0265) [survives FDR]
- **length_preference** — The average choice rate for the longer sequence across all pairs where the two sequences have different lengths. (observed 0.56 vs null mean 0.524, z=2.96, p=0.0199, q=0.0398) [survives FDR]
- **periodicity_penalty** — The choice rate for the less periodic sequence, conditioned on the sequences having similar alternation rates and imbalances. (observed 0.4 vs null mean 0.488, z=-2.27, p=0.0398, q=0.0637)
