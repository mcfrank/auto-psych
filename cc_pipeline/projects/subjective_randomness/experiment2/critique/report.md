# Critique Report — Subjective Randomness Experiment 2

## Summary

Task: participants judged which of two binary sequences (H/T) looked more random. Experiment 2 targeted pairs that discriminate between five models: alternation_bias, balance_heuristic, griffiths_representativeness, weighted_balance_alternation, and run_aversion (the latter two new to experiment 2). The posterior was computed pooling both experiments (200 trials total: 100 per experiment, 18–32 unique stimulus pairs, 5 simulated participants per experiment). The Bonferroni-corrected significance threshold for 4 test statistics is p < 0.0125.

---

## Model fit

| Model | Log-likelihood | Posterior |
|-------|---------------|-----------|
| run_aversion | -134.20 | 0.9973 |
| weighted_balance_alternation | -130.22 | 0.0027 |
| balance_heuristic | -161.82 | 0.0000 |
| alternation_bias | -195.02 | 0.0000 |
| griffiths_representativeness | -280.55 | 0.0000 |

run_aversion and weighted_balance_alternation are the only competitive models. Their raw log-likelihood difference is only 3.98 (favouring weighted_balance_alternation), but the complexity penalty (845 vs 746 characters) reverses this, giving run_aversion the dominant posterior. The log-likelihood gap between these two and balance_heuristic (next best) is 28–32 points — decisive evidence against the simpler heuristics. griffiths_representativeness performs worst, likely because it penalises over-alternation while participants consistently prefer alternating sequences.

### Log-likelihood by experiment

| Model | Experiment 1 | Experiment 2 | Total |
|-------|-------------|-------------|-------|
| run_aversion | -67.92 | -66.28 | -134.20 |
| weighted_balance_alternation | -67.78 | -62.44 | -130.22 |
| balance_heuristic | -80.17 | -81.64 | -161.82 |
| alternation_bias | -100.75 | -94.27 | -195.02 |
| griffiths_representativeness | -134.79 | -145.76 | -280.55 |

Model rankings are consistent across both experiments. weighted_balance_alternation slightly outperforms run_aversion in both experiments in raw log-likelihood (difference ~0.14 in experiment 1, ~3.84 in experiment 2), but the posterior reversal is entirely driven by the complexity prior. run_aversion's performance improves from experiment 1 to experiment 2 relative to balance_heuristic, consistent with experiment 2's stimuli being better designed to discriminate run-based from balance-based strategies.

---

## Model critiques

### run_aversion

Four test statistics were applied (Bonferroni threshold: p < 0.0125 for k=4):

**1. mean_abs_error** (MAE between run_aversion predictions and observed)
- Observed T: 0.156, Mean under model: 0.130, p = 0.068
- Not significant after Bonferroni correction.

**2. sd_chose_left** (SD of response rates across stimuli)
- Observed T: 0.253, Mean under model: 0.325, p = 0.998
- Not significant. The model generates more extreme choices (higher SD) than observed, indicating overconfidence in its predictions. The beta=5.0 parameter may be too high, causing predicted probabilities to cluster near 0 and 1 while actual responses are more moderate.

**3. direction_error_rate** (fraction of stimuli where model is confidently wrong)
- Observed T: 0.156, Mean under model: 0.041, p = 0.014
- Not significant after Bonferroni correction (p = 0.014 > 0.0125), though near the threshold. The model generates direction errors at a rate of ~4% under its own distribution, but 15.6% of stimuli showed confident reversals in the data. Notable cases: HTHTHTHT vs HHHTTTTH (pred=0.89 for HTHTHTHT, obs=0.20); HHTHTTTH vs HTHTHTH (pred=0.19, obs=0.70); HHHTTTTH vs HTHTHTH (pred=0.11, obs=0.60). In all three cases, participants preferred a sequence with a longer maximum run over a more (or fully) alternating one.

**4. mean_chose_alternating** (preference for the more-alternating sequence)
- Observed T: 0.690, Mean under model: 0.783, p = 0.998
- Not significant. Participants show a reliable alternation preference (69%), but the model predicts even stronger alternation preference (78%), confirming the overconfidence pattern from the SD statistic. Participants do prefer alternating sequences, but not as decisively as run_aversion predicts.

**No test statistics survive Bonferroni correction.** The closest failure is direction_error_rate (p = 0.014), which narrowly misses the threshold. The pattern across all four statistics is consistent: run_aversion correctly identifies the direction of preference in most cases but is systematically overconfident, and it makes a small number of confident direction errors on specific pairs.

---

## Conclusions

run_aversion is the best-supported model pooled across both experiments (posterior = 0.997), and no PPC critique survives Bonferroni correction. However, the pattern of near-failures points to two distinct issues:

**1. Overconfidence.** The model uses a fixed beta=5.0, which produces very extreme softmax probabilities. The observed SD of response rates (0.25) is substantially lower than what run_aversion would generate (null mean 0.32), and the observed alternation preference (69%) is lower than the model's predicted preference (78%). The model is qualitatively correct — participants prefer shorter maximum runs — but the steepness of this preference is overstated.

**2. Residual direction errors (~16% of stimuli).** Three stimulus pairs showed confident reversals, all involving sequences where the more-alternating option also had a cross-length comparison or an unusual length structure (e.g., HTHTHTHT length 8 vs HHHTTTTH length 8; HHTHTTTH length 8 vs HTHTHTH length 7). These failures suggest run_aversion's single-feature score (max run) does not fully capture preference when run differences interact with sequence length.

weighted_balance_alternation achieves a slightly better raw log-likelihood (difference of ~4 points), which becomes substantial in experiment 2 alone (~3.8 points). Its posterior is suppressed only by the complexity penalty. This model combines balance and alternation rate, which may handle cross-length comparisons more gracefully.

---

## Recommendations for next experiment

1. **Add a free beta parameter.** Run_aversion's fixed beta=5.0 is too steep. An experiment fitting beta from data (or testing beta variants) would clarify the true steepness of the run-aversion preference.

2. **Cross-length pairs targeting run vs. alternation divergence.** The direction errors concentrate on pairs where run_aversion and alternation_bias make different predictions AND the sequences have different lengths. More cross-length pairs with careful control of run structure would sharpen this test.

3. **Test weighted_balance_alternation directly.** Its raw log-likelihood advantage over run_aversion grows in experiment 2 (−62.4 vs −66.3). Adding stimuli that discriminate between these two models — specifically pairs where balance and alternation make opposing predictions — would resolve which cognitive feature dominates.

4. **Calibrate the beta parameter.** A model variant with beta fitted per participant, or a lower fixed beta (~2–3), would reduce overconfidence and likely improve both log-likelihood and PPC performance. This could be a new model: `run_aversion_calibrated`.
