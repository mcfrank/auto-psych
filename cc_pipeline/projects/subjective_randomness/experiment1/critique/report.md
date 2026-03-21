# Critique Report — Subjective Randomness Experiment 1

## Summary

**Task:** Two-alternative forced choice. Participants chose which of two coin-flip sequences (each 4–8 flips) looks "more random." The left option was always a short, perfectly alternating sequence (HTHT or THTH); the right option was one of 20 longer sequences with 6H/2T or 2H/6T balance.

**Stimuli:** 30 unique pairs (20 HTHT-based, 10 THTH-based), 5 responses each, 150 total responses, 5 simulated participants.

**Models tested:** griffiths_representativeness, alternation_bias, balance_heuristic.

**Critical design issue discovered:** All three models output *identical, constant* predictions across all 30 stimulus pairs. This occurred because HTHT and THTH are statistically equivalent (both have alternation rate = 1.0, H/T balance = 0.5), and all comparison sequences share the same alternation rate (3/7 ≈ 0.43) and the same |balance − 0.5| = 0.25. Consequently, Pearson r between any model's predictions and observed data is undefined (zero variance in predictions → r = 0.0 for all models). The correlations are not informative of model quality.

**Bonferroni correction:** 5 test statistics → significance threshold p < 0.05/5 = **0.01**.

---

## Overall model fit

| Model | Pearson r | Predicted P(left) | Observed mean P(left) | Mean |error| | Significant failures (Bonferroni) |
|-------|-----------|-------------------|-----------------------|--------------|-----------------------------------|
| griffiths_representativeness | 0.0* | 0.189 (constant) | 0.38 | 0.191 | 2/5 |
| alternation_bias | 0.0* | 0.946 (constant) | 0.38 | 0.566 | 3/5 |
| balance_heuristic | 0.0* | 0.881 (constant) | 0.38 | 0.501 | 3/5 |

*Pearson r is 0.0 for all models because predictions are constant across stimuli (zero variance); this reflects a design confound, not a computation error.

**Observed data summary:** Mean P(chose left) = 0.38; participants chose the right (comparison) sequence slightly more often overall. Choice proportions varied considerably across stimuli (SD = 0.227), ranging from 0.0 to 0.8.

---

## Model critiques

### griffiths_representativeness

This model (Griffiths & Tenenbaum 2001) scores sequences by how closely their alternation rate and H/T balance match the statistics of a fair coin (both ~0.5). HTHT/THTH, being perfectly alternating (alt_rate = 1.0) but balanced (H/T = 0.5), receive a low randomness score. The model therefore predicts participants will prefer the right (comparison) sequence as "more random" about 81% of the time (P(left) = 0.189).

**2 tests run, 2 significant failures** (Bonferroni threshold p < 0.01):

- **Test: mean_chose_left** — mean proportion of left choices across stimuli
  - Observed T: 0.380, Mean under model: 0.189, p = 0.002
  - Interpretation: Griffiths predicts participants rarely choose HTHT/THTH as "more random" (only ~19% of the time). In fact, participants chose the left alternating sequence 38% of the time — twice as often as predicted. This is a systematic upward bias: the model drastically underestimates how often people perceive perfectly alternating sequences as random, likely because human participants also weight alternation as a cue to randomness.

- **Test: fraction_moderate_choices** — fraction of stimuli where 30%–70% chose left
  - Observed T: 0.500, Mean under model: 0.236, p = 0.004
  - Interpretation: Half of all stimuli elicited moderate, near-even preferences in the data (with participants split roughly 40-60 either way). The model, with its confident prediction of P(left) = 0.189, generates very few moderate splits in simulated data (only ~24% of stimuli expected near-50-50). This reveals that griffiths_representativeness is overconfident: it predicts strong, consistent right-preference that the data does not support.

### alternation_bias

This model predicts that participants prefer whichever sequence has the higher alternation rate. Since HTHT/THTH have alt_rate = 1.0 versus the comparison sequences' alt_rate ≈ 0.43, the model strongly predicts left preference (P(left) = 0.946) for every pair.

**3 tests run, 3 significant failures** (Bonferroni threshold p < 0.01):

- **Test: mean_abs_error** — SD of P(chose left) across stimuli (underdispersion)
  - Observed T: 0.227, Mean under model: 0.096, p = 0.002
  - Interpretation: The model's constant prediction of P(left) = 0.946 can produce stimulus-level variability only through binomial sampling noise (expected SD ≈ 0.096). The observed variability (SD = 0.227) is more than twice as large, indicating systematic stimulus-level differences in preference that the model completely ignores. The model cannot account for why some pairs elicit strong left-preference and others elicit strong right-preference.

- **Test: mean_chose_right** — mean proportion of right choices
  - Observed T: 0.620, Mean under model: 0.052, p = 0.002
  - Interpretation: Participants chose the right (comparison) sequence 62% of the time on average. The alternation bias model predicts only 5.2% right choices. This is the most striking single failure: the direction of the effect is correct (alternating sequences are preferred), but the magnitude is wildly off. Participants are far less biased toward alternation than the model assumes.

- **Test: fraction_moderate_choices** — fraction of stimuli with 30–70% left choices
  - Observed T: 0.500, Mean under model: 0.027, p = 0.002
  - Interpretation: Half the stimuli produced near-even choice splits. The alternation model's extreme prediction (P(left) = 0.946) makes near-even outcomes almost impossible in simulated data (only 2.7% expected). This confirms that the alternation bias model is severely miscalibrated: the logistic scaling factor (× 5 on alternation rate difference) produces probabilities that are far too extreme.

### balance_heuristic

This model predicts preference for whichever sequence has an H/T ratio closer to 0.5. Since HTHT/THTH are perfectly balanced (H/T = 0.5) and all comparison sequences have extreme imbalance (6H/2T or 2H/6T), the model strongly predicts left preference (P(left) = 0.881).

**3 tests run, 3 significant failures** (Bonferroni threshold p < 0.01):

- **Test: mean_abs_error** — SD of P(chose left) across stimuli (underdispersion)
  - Observed T: 0.227, Mean under model: 0.141, p = 0.002
  - Interpretation: Like the alternation model, balance_heuristic predicts constant P(left) = 0.881. Simulated data has expected SD ≈ 0.141 (slightly larger than alternation due to less extreme p), while observed SD = 0.227. The model cannot explain why preferences vary across pairs that are, from the balance perspective, all identical.

- **Test: mean_chose_right** — mean proportion of right choices
  - Observed T: 0.620, Mean under model: 0.118, p = 0.002
  - Interpretation: Participants chose right 62% of the time, while the balance model predicts only 11.8% right choices. Despite balance being the only feature that should differentiate pairs under this model (and all pairs have maximally unbalanced comparison sequences), participants did not consistently prefer the balanced left sequence. The logistic scale factor (× 8 on balance difference) is too strong.

- **Test: fraction_moderate_choices** — fraction of stimuli with 30–70% left choices
  - Observed T: 0.500, Mean under model: 0.110, p = 0.002
  - Interpretation: The balance heuristic predicts strongly peaked distributions (overwhelmingly left), leaving only ~11% of simulated stimuli near 50-50. The observed 50% moderate-choice rate shows that participants are, in aggregate, much less swayed by H/T balance than the model predicts.

---

## Conclusions

**No model provides an adequate quantitative fit.** All three suffer from a fundamental confound in the experiment design: because HTHT and THTH have identical statistics (alt_rate = 1.0, balance = 0.5), and all comparison sequences share identical statistics (alt_rate = 3/7, |balance − 0.5| = 0.25), every model predicts the same P(left) for every stimulus pair. Pearson r is therefore undefined, and the models cannot explain any stimulus-level variability in the data.

**Qualitative ranking** (based on level calibration and number of significant failures):

1. **griffiths_representativeness** (best of three): Predicted P(left) = 0.189 vs observed 0.380 — off by 0.191 on average. It correctly identifies that the alternating sequences are not maximally "random" by rational-statistical standards, but underestimates how much participants are drawn to them. 2 significant failures.

2. **balance_heuristic** (second): Predicted P(left) = 0.881 vs observed 0.380 — off by 0.501. Strongly overestimates left preference. 3 significant failures.

3. **alternation_bias** (worst): Predicted P(left) = 0.946 vs observed 0.380 — off by 0.566. Most extreme overestimation. 3 significant failures.

**What the failures reveal about cognition:** The data (mean P(left) = 0.38) shows that participants actually preferred the *comparison* sequences slightly more overall — opposite to the alternation bias and balance heuristic predictions. This suggests that in this task, the 8-flip imbalanced sequences are not strongly penalized by participants, or that the 4-flip alternating sequences (HTHT) feel "too patterned" rather than random. The griffiths model captures this intuition directionally (alternating sequences look non-random to a rational observer) but underestimates the residual appeal of alternation.

**The key missing mechanism:** No model accounts for sequence *length* as a factor. The left sequences (HTHT/THTH) are shorter (4 flips) than the right sequences (8 flips). Participants may be responding partly to length, or to the conspicuousness of a perfect alternation pattern that becomes more recognizable at length 4.

---

## Recommendations for next experiment

1. **Fix the design confound.** Ensure that sequence_b options are not all statistically identical. Vary both the alternation rate AND the balance of comparison sequences so that models can make discriminating predictions across stimuli. Without this, no model can be tested against data.

2. **Deconfound alternation rate from sequence length.** Use comparison sequences of the same length as the target (HTHT), or explicitly model the effect of length.

3. **Add a mixed-cue model.** The current results suggest neither pure alternation nor pure balance explains the data. Try a model that combines both cues with a free mixing weight, or a model that explicitly penalizes *recognizable patterns* (like HTHT) as "too regular" — which is what griffiths does but it undershoots the magnitude.

4. **Recalibrate logistic scale parameters.** All three models use fixed scale parameters (×5 or ×8) in the logistic that produce extremely confident predictions. A temperature parameter (softmax β or logistic scale) should be fit to data or varied across experiments.

5. **Consider a length-normalized model.** Add a model that normalizes sequence randomness scores by length, or that explicitly penalizes the conspicuousness of short repeating patterns.

6. **Increase n per stimulus.** With n=5 per stimulus, the binomial noise is large and PPC null distributions are wide. Aim for n≥20 to increase statistical power for detecting stimulus-level effects.
