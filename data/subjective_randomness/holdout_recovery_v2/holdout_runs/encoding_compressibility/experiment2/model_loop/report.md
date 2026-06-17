# Inner Model Loop Report

Each model below is ONE distinct cognitive hypothesis. The posterior mass shows which single hypothesis best explains the data — it is **not** a recipe to combine the top models into a blend.

- Best model: **iter0_candidate0** (posterior=0.369, elpd_loo=-621.37)
- Trials: 1200
- Models compared: 11

## Posterior over models (ELPD-LOO)

| model | posterior | elpd_loo |
| --- | --- | --- |
| iter0_candidate0 | 0.3685 | -621.37 |
| prototype_similarity | 0.2453 | -622.08 |
| inner_loop_model | 0.2453 | -622.08 |
| iter0_candidate2 | 0.0955 | -623.42 |
| bayesian_diagnosticity | 0.0426 | -622.08 |
| iter1_candidate0 | 0.0029 | -626.33 |
| iter1_candidate2 | 0.0000 | -631.22 |
| max_run_length | 0.0000 | -662.49 |
| rle_description_length | 0.0000 | -725.77 |
| iter0_candidate1 | 0.0000 | -832.76 |
| iter1_candidate1 | 0.0000 | -3631.60 |

## Hypotheses

- **iter0_candidate0**: People judge a sequence as more random when it is close to a prototype with balanced H/T counts and an ideal alternation rate, where closeness is measured by squared deviation rather than absolute deviation. This means the randomness gradient is steepest exactly at the prototype — even modest departures are penalized disproportionately — rather than declining at a constant rate regardless of how far the sequence already is from ideal.
- **prototype_similarity**: Random-looking sequences are close to a prototype with balanced H/T counts
and an ideal alternation rate.
- **inner_loop_model**: Best PyMC model found by the inner model-improvement loop.
- **iter0_candidate2**: People judge a sequence as more random when its proportion of heads is closer to 0.5 — that is, when the sequence is more balanced. They ignore run structure, alternation patterns, and periodicity entirely; only the overall H/T ratio guides the judgment. The sequence with the less imbalanced coin-flip count looks more random.
- **bayesian_diagnosticity**: Random-looking sequences are diagnostic of a fair coin over salient
non-random alternatives: alternating, biased, and streaky generators.
- **iter1_candidate0**: People judge sequence randomness by comparing each sequence to a mental prototype with balanced heads-tails counts and an ideal alternation rate, penalizing deviations quadratically (as in the leading existing model). However, the penalty is asymmetric around the ideal alternation rate: sequences that are too streaky — fewer alternations than ideal — are treated as more non-random than sequences that are equally distant in the over-alternating direction. This asymmetry reflects the well-documented human tendency to find runs of the same symbol particularly diagnostic of non-randomness, stronger than the reverse surprise of excessive alternation.
- **iter1_candidate2**: People judge a sequence as more random when its alternation rate is closer to their internal ideal for what randomness looks like. When comparing two sequences, they pick whichever one's alternation rate deviates less from this ideal. The ideal alternation rate is a free parameter inferred from the data — not fixed at 0.5 — because people's intuitions about random sequences are known to be biased toward over-alternation.
- **max_run_length**: People judge a sequence as more random when its longest consecutive run is
shorter, because a long run is the most compact single-symbol run-length
encoding unit and is the strongest single cue that the sequence is structured.
- **rle_description_length**: People judge a sequence as more random when its run-length encoding requires
more blocks (alternations + 1 normalized by length), predicting a monotonic
relationship between alternation rate and perceived randomness with no ideal rate.
- **iter0_candidate1**: People judge a sequence as more random when it contains less periodic structure. Sequences that repeat a regular cycle (e.g., HTHT… or HHTTHHTT…) are perceived as non-random because the mind detects the underlying period. When comparing two sequences, people choose the one with lower periodicity as the more random-looking.
- **iter1_candidate1**: People judge a sequence as more random when its longest consecutive run is close to the length expected from a fair coin of that length — approximately log₂(n). A max run that is too long signals a streaky, non-random process, but a max run that is too short signals an artificially regular, over-alternating process; both extremes feel non-random. Perceived randomness therefore peaks at the characteristic max-run length of a fair coin and declines quadratically as the actual max run departs in either direction.

## Distinguishability (arviz.compare, PSIS-LOO)

`elpd_diff` and `dse` are relative to the best model. A model is only clearly worse than the best when `elpd_diff > 2 * dse`; models within ~2·dse of the top are statistically indistinguishable.

| model | elpd_diff | dse | distinguishable from best | weight |
| --- | --- | --- | --- | --- |
| iter0_candidate0 | 0.00 | 0.00 | — (best) | 0.631 |
| prototype_similarity | 0.71 | 1.36 | no (within ~2·dse) | 0.000 |
| inner_loop_model | 0.71 | 1.36 | no (within ~2·dse) | 0.000 |
| bayesian_diagnosticity | 0.71 | 1.36 | no (within ~2·dse) | 0.136 |
| iter0_candidate2 | 2.05 | 2.78 | no (within ~2·dse) | 0.233 |
| iter1_candidate0 | 4.96 | 3.15 | no (within ~2·dse) | 0.000 |
| iter1_candidate2 | 9.85 | 4.20 | yes | 0.000 |
| max_run_length | 41.11 | 10.35 | yes | 0.000 |
| rle_description_length | 104.40 | 14.50 | yes | 0.000 |
| iter0_candidate1 | 211.38 | 17.63 | yes | 0.000 |
| iter1_candidate1 | 3010.23 | 255.45 | yes | 0.000 |
