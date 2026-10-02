# Critique of `personal_complement_symmetry_aversion`

3 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **participant_majority_agreement_variance** — Variance across participants of each participant's rate of agreeing with the pair's overall majority choice (pair identified regardless of side); observed above null_mean means people still differ in how consistently they follow the consensus more than the model's personal terms produce. (observed 0.0262 vs null mean 0.0213, z=2.88, p=0.004, q=0.032) [survives FDR]
- **participant_side_bias_variance** — Variance across participants of each participant's proportion of Left choices; observed above null_mean means real people have personal left/right button biases the model (which has no side term) does not produce. (observed 0.00556 vs null mean 0.00435, z=2.45, p=0.03, q=0.0799)
- **split_half_agreement_correlation** — Across participants, Pearson correlation between majority-agreement rate on odd-numbered trials and on even-numbered trials; observed above null_mean means agreement with the consensus is a more stable personal trait than the model's person-level parameters imply, below means less stable. (observed 0.847 vs null mean 0.777, z=2.19, p=0.016, q=0.0639)
