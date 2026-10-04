# Inner Model Loop Report

Each model below is ONE distinct cognitive hypothesis. The posterior mass shows which single hypothesis best explains the data — it is **not** a recipe to combine the top models into a blend.

- Best model: **graded_periodicity_personal_ideal** (posterior=0.642, elpd_loo=-1046.16)
- Trials: 2560
- Models compared: 8

## Posterior over models (ELPD-LOO)

| model | posterior | elpd_loo |
| --- | --- | --- |
| graded_periodicity_personal_ideal | 0.6417 | -1046.16 |
| person_sensitivity_length_scaled_ideal | 0.2586 | -1047.07 |
| heads_default_alternation_ideal | 0.0997 | -1048.02 |
| periodic_penalized_alternation_ideal | 0.0000 | -1071.98 |
| personal_ideal_with_personal_lapse | 0.0000 | -1071.22 |
| length_scaled_alternation_ideal | 0.0000 | -1069.90 |
| personal_lapse_ideal_streak_excess | 0.0000 | -1067.55 |
| heads_favoring_lapse_ideal_streak | 0.0000 | -1068.76 |

## Hypotheses

- **graded_periodicity_personal_ideal**: Refinement of the incumbent `person_sensitivity_length_scaled_ideal`: each person still judges a sequence as random by how close its proportion of H/T switches is to their own ideal switching rate (with a person-specific, length-scaled sensitivity), and everyone still counts a visibly repeating pattern against randomness — but the pattern detector is graded rather than all-or-nothing: people also notice a sequence that *almost* repeats a short unit (period 1–4) with one or two flips out of place, and the penalty fades smoothly with the number of flips that break the pattern. The one change is replacing the strict periodic indicator with this graded near-periodicity penalty (shared strength, shared fall-off per mismatch), addressing the critique that people avoid almost-repeating patterns that the incumbent's strict check misses.
- **person_sensitivity_length_scaled_ideal**: Refinement of the incumbent `length_scaled_alternation_ideal`: each person still judges a sequence as random by how close its proportion of H/T switches is to their own ideal switching rate (with a length-scaled sensitivity and a shared penalty for visibly periodic sequences), but people also differ in how decisively they act on that impression — some pick the sequence nearer their ideal almost every time, others only weakly lean toward it. The one change is a person-specific sensitivity to the alternation-distance, drawn from a population distribution, so that how sharply a person discriminates is an individual trait just as their ideal switching rate is.
- **heads_default_alternation_ideal**: People do not treat the two faces of the coin symmetrically: heads is the default, expected outcome of a coin toss, so a sequence dominated by tails reads as a coin that is "off" (biased toward the unusual face) and looks less random, while a heads-leaning sequence looks like ordinary coin flipping. This label asymmetry operates on top of each person's judgement of how close a sequence's switching rate is to their own ideal (with person-specific, length-scaled sensitivity and a shared penalty for visibly periodic sequences): of two sequences that switch equally often, people pick the one with more heads.
- **periodic_penalized_alternation_ideal**: Refinement of the incumbent `personal_alternation_ideal`: each person still judges a sequence as random to the extent that its proportion of H/T switches is close to their own personal ideal switching rate, but in addition everyone notices when a sequence is built by repeating a short unit (e.g. HTHTHTHT, HHTTHHTT, HTTHTTHT) and counts that visible periodic pattern against its randomness. The one change is this shared periodic-pattern penalty, addressing the critique that people reject regular repeating patterns (and perfect alternation) far more than the alternation-distance mechanism alone predicts.
- **personal_ideal_with_personal_lapse**: People judge which sequence looks more random by how close its proportion of H/T switches is to their own personal ideal switching rate, but the decision rule is not always engaged: each person has their own lapse rate, the share of trials on which they do not evaluate the sequences at all and pick a side at random. People therefore differ not only in what they think randomness looks like but in how consistently they act on it, so even a pair with a clear winner is split by some participants, more for some people than for others.
- **length_scaled_alternation_ideal**: Refinement of the incumbent `periodic_penalized_alternation_ideal`: each person still judges a sequence as random by how close its proportion of H/T switches is to their own ideal switching rate, with a shared penalty for visibly periodic sequences, but the weight people give to a deviation from their ideal grows (or shrinks) with how many flips they have seen — a switch rate read off a long sequence is treated as stronger evidence than the same rate in a short one. The one change is a fitted power-law scaling of the alternation-distance sensitivity with the number of transitions in the sequence, addressing the critique that the incumbent mispredicts how the preference for the more-switching sequence changes with sequence length.
- **personal_lapse_ideal_streak_excess**: Refinement of `personal_ideal_with_personal_lapse`: each person still judges a sequence as random by how close its proportion of H/T switches is to their own personal ideal switching rate, and still has their own lapse rate (trials on which they pick a side at random), but when they do evaluate the sequences they also notice a streak of identical outcomes as soon as it is long in absolute terms — every flip beyond two in the longest run (HHH, HHHH, ...) counts against randomness. The one change is this shared absolute-streak penalty inside the engaged decision, addressing the critique that among pairs with equal switch counts people prefer the sequence with the shorter longest run more than the incumbent predicts.
- **heads_favoring_lapse_ideal_streak**: Refinement of `personal_lapse_ideal_streak_excess`: each person still judges a sequence as random by how close its proportion of H/T switches is to their own personal ideal switching rate, still penalises every flip of the longest streak beyond two, and still lapses to a random pick at their own rate — but the two coin faces are not psychologically symmetric: heads is the canonical, expected outcome of a coin toss, so a sequence with a larger share of heads reads as more like "real" coin flipping and looks more random. The one change is a shared heads-share bias inside the engaged decision, addressing the critique that among pairs with equal switch counts people choose the heads-heavier sequence more often than an H/T-symmetric model allows.

## Distinguishability (arviz.compare, PSIS-LOO)

`elpd_diff` and `dse` are relative to the best model. A model is only clearly worse than the best when `elpd_diff > 2 * dse`; models within ~2·dse of the top are statistically indistinguishable. `LOO reliable` is False when PSIS-LOO flagged this model's estimate as untrustworthy (many high Pareto-k points) — its row should be read with caution.

| model | elpd_diff | dse | distinguishable from best | weight | LOO reliable |
| --- | --- | --- | --- | --- | --- |
| graded_periodicity_personal_ideal ←selected | 0.00 | 0.00 | — (best) | 0.833 | yes |
| person_sensitivity_length_scaled_ideal | 0.91 | 1.49 | no (within ~2·dse) | 0.000 | yes |
| heads_default_alternation_ideal | 1.86 | 1.64 | no (within ~2·dse) | 0.000 | yes |
| personal_lapse_ideal_streak_excess | 21.39 | 8.11 | yes | 0.167 | yes |
| heads_favoring_lapse_ideal_streak | 22.59 | 8.12 | yes | 0.000 | yes |
| length_scaled_alternation_ideal | 23.74 | 7.42 | yes | 0.000 | yes |
| personal_ideal_with_personal_lapse | 25.06 | 8.37 | yes | 0.000 | yes |
| periodic_penalized_alternation_ideal | 25.82 | 7.87 | yes | 0.000 | yes |
