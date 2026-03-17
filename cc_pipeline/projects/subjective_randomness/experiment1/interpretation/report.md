# Interpretation Report — Subjective Randomness Experiment 1

## Summary

**Task**: Participants judged which of two coin-flip sequences (H/T strings) looked more random.
**Stimuli**: 30 stimulus pairs drawn from sequences of length 4, 6, and 8.
**Participants**: 5 simulated participants (mode: `simulated_participants`).
**Total responses**: 150.
**Models tested**: `representativeness`, `alternation`, `griffiths_representativeness`.

---

## Results

### Overall response pattern

Participants chose the left sequence 60% of the time on average (`mean_chose_left = 0.60`), reflecting mild left-side preference consistent with the stimulus design (sequences that the designer believed look more random were placed on the left more often).

### Key stimulus-level patterns

Several pronounced patterns emerge from the choice proportions:

| Pair (A vs B) | chose_left | Interpretation |
|---|---|---|
| TTTHTT vs THTHTH | 0.00 | Strongly alternating THTHTH dominates completely |
| TTTHTT vs TTTHHH | 0.20 | Both "blocky"; slight edge to TTTHHH |
| TTTHHH vs TTHTHT | 0.20 | Alternating TTHTHT preferred over blocky TTTHHH |
| HTTTHH vs TTHTHT | 0.80 | Balanced HTTTHH preferred over maximally-alternating TTHTHT |
| THTHTH vs TTHHTT | 0.80 | Both somewhat balanced; THTHTH (more alternating) preferred |
| THTHTH vs TTTHHT | 0.80 | THTHTH preferred |
| TTTHHH vs TTHTHT | 0.20 | TTHTHT (alternating) preferred over blocky |
| TTTHHH vs HTTHTT | 0.60 | Moderate preference for TTTHHH against HTTHTT |
| HTTTHH vs HTHHTH | 1.00 | Unanimous preference for HTTTHH |

**Central finding**: Participants strongly disfavor sequences that deviate from 50/50 balance (`TTTHHH`, `TTTHTT`, `THTTTT`) and favor sequences with near-equal H/T counts. Among balanced sequences, those with moderate alternation (e.g. `THTHTH`, `HTTTHH`) tend to be preferred. However, sequences at the extreme of alternation (e.g. `TTHTHT`) are not consistently preferred over moderately-balanced alternatives, suggesting alternation alone is not the driving factor.

---

## Model Comparison

Pearson correlations between model predictions and observed `chose_left_pct` across all 30 stimuli:

| Model | r | Interpretation |
|---|---|---|
| `representativeness` | **+0.631** | Strong positive fit — best model |
| `alternation` | −0.056 | Near-zero, slightly negative — no predictive power |
| `griffiths_representativeness` | −0.037 | Near-zero, slightly negative — no predictive power |

### Representativeness (balance heuristic)

This model predicts that participants prefer the sequence whose H/T proportion is closer to 50/50. The correlation of r = 0.63 is strong for a single-parameter cognitive model and indicates the balance heuristic captures the dominant pattern in the data. Stimuli where the balance heuristic makes a clear prediction (e.g. blocky `TTTHHH` vs nearly-balanced alternatives) consistently show data patterns matching the model.

### Alternation heuristic

The alternation model predicts that participants prefer sequences with more H→T and T→H transitions. Despite the well-documented alternation bias in the literature, this model shows essentially no correlation (r = −0.056) with observed choices. This may reflect the stimulus design, which pitted alternation against balance in ways that expose the balance heuristic as the stronger driver. Alternatively, the simulated participants may not have been calibrated to the alternation bias.

### Griffiths representativeness (Markov chain, p_alt = 0.7)

This model implements the rational basis of representativeness (Griffiths & Tenenbaum): participants judge sequences by their likelihood under a subjective Markov generator that over-weights alternation. The near-zero correlation (r = −0.037) is surprising, given that this model is theoretically grounded. It may indicate that the subjective alternation parameter (p_alt = 0.7) does not match this participant sample, or that the balance heuristic captures participants' judgments more directly than the Markov likelihood model.

---

## Conclusions

1. **Balance/representativeness is the dominant cognitive principle** in this experiment. Participants reliably judge sequences with near-equal H and T counts as "more random," consistent with the classic representativeness heuristic.

2. **Alternation bias was not detected** in this data. Neither the simple alternation heuristic nor the Griffiths Markov model predicted choices better than chance. This may be a property of the simulated participants or may reflect that the experiment design did not sufficiently isolate alternation from balance.

3. **The experiment successfully discriminated between models**, with one model (representativeness, r = 0.63) substantially outperforming the others (|r| < 0.06). This is a useful outcome for an initial experiment.

---

## Recommendations

1. **Retain and refine the representativeness model.** Consider parameterizing the balance weight (how strongly participants penalize deviations from 50/50) rather than using a hard nearest-50/50 rule.

2. **Re-examine the alternation and Griffiths models.** Design follow-up stimuli that hold balance constant while varying alternation rate, to isolate whether alternation bias is present at all.

3. **Explore the Griffiths model with different p_alt values** (e.g. 0.6, 0.8, 0.9). The negative-near-zero result may reflect parameter mismatch rather than a fundamental failure of the model.

4. **Add a length-adjusted balance model**: participants may perceive balance relative to expected variance, which depends on sequence length. A length-normalized balance heuristic could improve fit.

5. **Increase participant N** (currently 5) to reduce noise in choice proportions. With binary trials and N=5, choice proportions are limited to {0, 0.2, 0.4, 0.6, 0.8, 1.0}, which may obscure graded model differences.
