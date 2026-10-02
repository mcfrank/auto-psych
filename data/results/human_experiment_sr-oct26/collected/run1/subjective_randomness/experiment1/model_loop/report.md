# Inner Model Loop Report

Each model below is ONE distinct cognitive hypothesis. The posterior mass shows which single hypothesis best explains the data — it is **not** a recipe to combine the top models into a blend.

- Best model: **individual_streak_aversion_lapse** (posterior=1.000, elpd_loo=-931.39)
- Trials: 2560
- Models compared: 6

## Posterior over models (ELPD-LOO)

| model | posterior | elpd_loo |
| --- | --- | --- |
| individual_streak_aversion_lapse | 1.0000 | -931.39 |
| lapse_individual_streak_aversion | 0.0000 | -943.11 |
| personal_lapse_ideal_alternation | 0.0000 | -973.33 |
| lapse_individual_balance_alternation | 0.0000 | -952.39 |
| edge_weighted_alternation_memory | 0.0000 | -954.03 |
| edge_streak_balance_alternation_lapse | 0.0000 | -951.87 |

## Hypotheses

- **individual_streak_aversion_lapse**: Refinement of the incumbent `lapse_individual_balance_alternation`: each person still judges a sequence as more random the closer its proportion of alternations lies to their own ideal switching rate, still penalises lopsided heads/tails counts by a weight of their own, and still guesses on some trials at a personal lapse rate. The one change is that people also notice the longest streak of identical flips and treat it as a sign of non-randomness by a weight that is each person's own (drawn from a population distribution, so some people are strongly streak-averse and others barely care), addressing the critique that individuals differ in preferring the sequence with the shorter longest run more than the incumbent produces, and that streaks are penalised beyond its alternation and balance terms.
- **lapse_individual_streak_aversion**: Refinement of `personal_lapse_ideal_alternation`: each person still judges a sequence as more random the closer its proportion of alternations lies to their own personal ideal switching rate, and still guesses on some trials at a personal, trait-like lapse rate, but people also treat a long streak of identical flips as a sign of non-randomness, and how strongly a streak puts them off is each person's own trait (a personal weight on the longest run relative to sequence length, drawn from a population). The one change is this individual streak-aversion weight, added because the critique shows participants differ in their preference for the sequence with the shorter longest run more than the model's alternation, balance and lapse heterogeneity produces, and that streaks are penalised beyond alternation among alternation-matched pairs.
- **personal_lapse_ideal_alternation**: People judge a sequence as more random the closer its proportion of alternations lies to their own personal ideal switching rate, but the decision rule is not always engaged: on some trials a person does not compare the sequences at all and picks a side by guessing. How often this happens is a stable trait that differs between people, so some participants follow their alternation preference almost every trial while others are close to indifferent, which spreads individual choice proportions beyond what a single shared decisiveness produces.
- **lapse_individual_balance_alternation**: Refinement of `individual_balance_ideal_alternation` (statistically tied with the incumbent): each person still judges a sequence as more random the closer its proportion of alternations lies to their own personal ideal switching rate, and lopsided heads/tails counts make a sequence look less random by a weight that is each person's own. The one change is grafting in the incumbent's decision rule: on some trials a person does not compare the sequences at all and guesses, at a personal trait-like lapse rate, so indifferent participants are explained by guessing rather than by weak preferences, and the balance and alternation preferences of engaged people can be as sharp as the data show — addressing the critique that among alternation-matched pairs people pick the sequence without a long streak (usually the more balanced one) more often than the lapse-only incumbent predicts.
- **edge_weighted_alternation_memory**: People encode a coin-flip sequence with serial-position effects in memory: the flips at the start and end of the sequence (primacy and recency) are encoded more strongly than those in the middle, so the switching pattern they perceive is dominated by what happens at the sequence's edges. Each person compares this edge-weighted impression of how often the sequence switches between heads and tails with their own personal ideal switching rate, and on some trials guesses at a personal lapse rate; a streak or rigid alternation sitting at an edge therefore makes a sequence look much less random than the same pattern buried in the middle.
- **edge_streak_balance_alternation_lapse**: Refinement of the incumbent `lapse_individual_balance_alternation`: each person still judges a sequence as more random the closer its proportion of alternations lies to their own ideal switching rate, still penalises lopsided heads/tails counts by a weight of their own, and still guesses on some trials at a personal lapse rate. The one change is that a streak sitting at the very start or end of a sequence is especially salient (primacy/recency of what is read first and last): the number of flips by which a run of identical flips at either edge exceeds two makes the sequence look less random by a shared weight, while the same streak buried in the middle carries no extra penalty — addressing the critique that among alternation-matched pairs people avoid the sequence with an edge streak more than the incumbent predicts.

## Distinguishability (arviz.compare, PSIS-LOO)

`elpd_diff` and `dse` are relative to the best model. A model is only clearly worse than the best when `elpd_diff > 2 * dse`; models within ~2·dse of the top are statistically indistinguishable. `LOO reliable` is False when PSIS-LOO flagged this model's estimate as untrustworthy (many high Pareto-k points) — its row should be read with caution.

| model | elpd_diff | dse | distinguishable from best | weight | LOO reliable |
| --- | --- | --- | --- | --- | --- |
| individual_streak_aversion_lapse ←selected | 0.00 | 0.00 | — (best) | 0.774 | yes |
| lapse_individual_streak_aversion | 11.73 | 4.47 | yes | 0.000 | yes |
| edge_streak_balance_alternation_lapse | 20.49 | 5.63 | yes | 0.000 | yes |
| lapse_individual_balance_alternation | 21.01 | 5.25 | yes | 0.000 | yes |
| edge_weighted_alternation_memory | 22.65 | 9.42 | yes | 0.226 | yes |
| personal_lapse_ideal_alternation | 41.94 | 8.60 | yes | 0.000 | yes |
