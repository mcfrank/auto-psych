# Critique of `iter0_candidate3`

3 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **perfect_alternation_choice_rate** — Proportion of choices for a perfectly alternating sequence (HTHT.../THTH..., length>=4) when it is paired with a non-alternating one; observed above null_mean means the model over-penalises perfect alternation (people find it more random than predicted), below means it under-penalises it. (observed 0.768 vs null mean 0.858, z=-5.43, p=0.002, q=0.016) [survives FDR]
- **participant_left_rate_sd** — Standard deviation across participants of each person's proportion of Left choices; observed above null_mean means people differ in side bias more than the model's single shared side_bias allows. (observed 0.0768 vs null mean 0.0621, z=2.46, p=0.024, q=0.0693)
- **longest_run_avoidance** — Among pairs whose longest runs differ by >=2 flips, the proportion of choices for the sequence with the SHORTER longest run; observed above null_mean means the model under-penalises long streaks (it has no run-length term beyond windows of 4), below means it over-penalises them. (observed 0.84 vs null mean 0.861, z=-2.28, p=0.026, q=0.0693)
