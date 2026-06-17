# Inner Model Loop Report

Each model below is ONE distinct cognitive hypothesis. The posterior mass shows which single hypothesis best explains the data — it is **not** a recipe to combine the top models into a blend.

- Best model: **inner_loop_model** (posterior=0.689, elpd_loo=-1224.93)
- Trials: 1800
- Models compared: 13

## Posterior over models (ELPD-LOO)

| model | posterior | elpd_loo |
| --- | --- | --- |
| inner_loop_model | 0.6891 | -1224.93 |
| length_sensitive_2d_prototype | 0.2088 | -1226.02 |
| iter1_candidate0 | 0.0908 | -1226.56 |
| prototype_similarity | 0.0100 | -1229.02 |
| iter0_candidate0 | 0.0013 | -1231.01 |
| iter1_candidate2 | 0.0000 | -1237.73 |
| run_length_prototype | 0.0000 | -1239.19 |
| encoding_compressibility | 0.0000 | -1245.87 |
| length_sensitive_alternation | 0.0000 | -21626.27 |
| bayesian_markov_fairness | 0.0000 | -1249.26 |
| iter0_candidate1 | 0.0000 | -1248.43 |
| iter0_candidate2 | 0.0000 | -1245.87 |
| iter1_candidate1 | 0.0000 | -1243.80 |

## Hypotheses

- **inner_loop_model**: People judge a sequence as more random-looking when it is close to an internal
2D prototype specifying both an ideal alternation rate and balanced heads and
tails, with Gaussian (quadratic) decay in each dimension independently.
- **length_sensitive_2d_prototype**: People evaluate sequences using the two-dimensional prototype (alternation
rate and balance), but their sensitivity to deviations scales linearly with
sequence length because longer sequences provide stronger statistical evidence
of non-randomness for the same proportional deviation.
- **iter1_candidate0**: People evaluate sequences against an internal two-dimensional prototype specifying both an ideal alternation rate and balanced outcomes, as in the leading model. However, their sensitivity to departures in the alternation dimension is asymmetric: sequences that are too streaky (alternation rate below the ideal) are penalized more heavily than sequences that are too alternating (alternation rate above the ideal), because streaks feel more distinctively non-random than excess back-and-forth switching.
- **prototype_similarity**: Random-looking sequences are close to a prototype with balanced H/T counts
and an ideal alternation rate.
- **iter0_candidate0**: People judge a coin sequence as more random-looking when it falls close to an internal two-dimensional prototype specifying both an ideal alternation rate and balanced heads-to-tails proportions. Their sensitivity to departures from this prototype is linear — each additional unit of distance from the prototype ideal reduces perceived randomness by a fixed amount, rather than accelerating the further from the ideal the sequence strays.
- **iter1_candidate2**: People judge a sequence as more random-looking when its alternation rate is closer to 0.5 — the rate a fair coin produces. Their internal prototype is fixed at exactly 0.5 rather than being a learned parameter, so there is only one free parameter: how sensitively they respond to deviations from that fixed standard.
- **run_length_prototype**: People judge sequences by how close the maximum run length is to an internal
prototype for a random sequence — both too-long streaks and too-short maximum
runs look non-random, so sequences whose longest run matches the ideal are
judged most random.
- **encoding_compressibility**: Random-looking sequences are those with low simple-description penalties:
long runs, periodic templates, and imbalance.
- **length_sensitive_alternation**: People evaluate alternation deviation on the count scale rather than the
proportion scale — a sequence with two extra transitions relative to the
ideal count looks equally deviant regardless of whether it is 4 or 8 flips
long, so the randomness signal scales with sequence length.
- **bayesian_markov_fairness**: People implicitly compute the log-Bayes-factor comparing each sequence's
observed transitions against a fair Markov chain (p_transition = 0.5) versus
a biased one, and choose the sequence whose transitions are more consistent
with the fair-coin hypothesis.
- **iter0_candidate1**: People judge a sequence as more random-looking when it contains less detectable periodic structure. A truly random coin sequence should not repeat a regular pattern, so sequences with stronger periodicity feel non-random. When comparing two sequences, people choose the one with lower periodicity as the more random-looking one.
- **iter0_candidate2**: People judge a sequence as more random-looking when its outcomes are more evenly balanced between heads and tails. The single cognitive mechanism is outcome balance: a sequence where heads and tails are equal looks maximally random, and any departure from that balance — in either direction — reduces how random the sequence feels.
- **iter1_candidate1**: People judge a coin sequence as more random-looking based solely on the length of its longest streak: longer maximum runs always feel less random, with no ideal run length to aim for. When comparing two sequences, people simply choose whichever has the shorter maximum run as the more random-looking one.

## Distinguishability (arviz.compare, PSIS-LOO)

`elpd_diff` and `dse` are relative to the best model. A model is only clearly worse than the best when `elpd_diff > 2 * dse`; models within ~2·dse of the top are statistically indistinguishable.

| model | elpd_diff | dse | distinguishable from best | weight |
| --- | --- | --- | --- | --- |
| inner_loop_model | 0.00 | 0.00 | — (best) | 0.880 |
| length_sensitive_2d_prototype | 1.09 | 1.44 | no (within ~2·dse) | 0.000 |
| iter1_candidate0 | 1.63 | 0.63 | yes | 0.000 |
| prototype_similarity | 4.09 | 2.36 | no (within ~2·dse) | 0.000 |
| iter0_candidate0 | 6.08 | 3.33 | no (within ~2·dse) | 0.000 |
| iter1_candidate2 | 12.80 | 5.02 | yes | 0.000 |
| run_length_prototype | 14.26 | 6.09 | yes | 0.120 |
| iter1_candidate1 | 18.87 | 5.97 | yes | 0.000 |
| iter0_candidate2 | 20.94 | 6.27 | yes | 0.000 |
| encoding_compressibility | 20.94 | 6.13 | yes | 0.000 |
| iter0_candidate1 | 23.50 | 6.73 | yes | 0.000 |
| bayesian_markov_fairness | 24.34 | 6.63 | yes | 0.000 |
| length_sensitive_alternation | 20401.34 | 732.68 | yes | 0.000 |
