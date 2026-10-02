# Critique of `terminal_run_streak_aversion_lapse`

2 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **palindrome_choice_rate** — Among trials where exactly one sequence is a palindrome (mirror-symmetric, e.g. HTTTTTTH, HHHTTHHH), the proportion choosing the palindrome; observed below the null means people penalise visible symmetry beyond what the model's alternation/balance/streak terms predict, above means they favour it. (observed 0.49 vs null mean 0.536, z=-5.86, p=0.002, q=0.016) [survives FDR]
- **longer_initial_run_choice_rate** — Among trials whose sequences begin with runs of different length, the proportion choosing the sequence with the LONGER initial run; observed below the null means people penalise an opening streak (primacy) more than the model, which weighs the final run but not the first, predicts. (observed 0.368 vs null mean 0.347, z=2.96, p=0.004, q=0.016) [survives FDR]
