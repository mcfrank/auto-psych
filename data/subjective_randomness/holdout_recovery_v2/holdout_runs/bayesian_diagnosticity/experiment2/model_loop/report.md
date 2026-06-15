# Inner Model Loop Report

Each model below is ONE distinct cognitive hypothesis. The posterior mass shows which single hypothesis best explains the data — it is **not** a recipe to combine the top models into a blend.

- Best model: **iter1_candidate0** (posterior=0.919, elpd_loo=-821.99)
- Trials: 1200
- Models compared: 11

## Posterior over models (ELPD-LOO)

| model | posterior | elpd_loo |
| --- | --- | --- |
| iter1_candidate0 | 0.9189 | -821.99 |
| prototype_similarity | 0.0401 | -824.97 |
| inner_loop_model | 0.0218 | -825.63 |
| iter0_candidate2 | 0.0095 | -826.86 |
| iter0_candidate1 | 0.0037 | -827.81 |
| iter0_candidate0 | 0.0036 | -827.47 |
| iter1_candidate2 | 0.0012 | -828.79 |
| encoding_compressibility | 0.0011 | -828.10 |
| iter1_candidate1 | 0.0000 | -832.49 |
| bayesian_markov_fairness | 0.0000 | -831.98 |
| length_sensitive_alternation | 0.0000 | -12323.69 |

## Hypotheses

- **iter1_candidate0**: People judge a sequence as more random-looking when it is close to an internal 2D prototype that specifies both an ideal alternation rate and balanced heads and tails (50/50 split). Closeness to the prototype follows Gaussian decay in both dimensions simultaneously — small deviations in alternation or balance are disproportionately forgiven, and the two dimensions contribute independently by adding their squared distances from the ideal.
- **prototype_similarity**: Random-looking sequences are close to a prototype with balanced H/T counts
and an ideal alternation rate.
- **inner_loop_model**: People judge a sequence as more random-looking when its alternation rate is
closer to an internal prototype ideal, computed quadratically (Gaussian decay),
so small departures are disproportionately forgiven relative to large ones.
- **iter0_candidate2**: People judge a sequence as more random-looking when its proportion of heads is closer to 0.5, because a fair coin produces equal heads and tails and balance is the primary signal of randomness. The more imbalanced a sequence (too many heads or too many tails), the less random it looks — and this single balance cue, not alternation or run structure, drives the choice.
- **iter0_candidate1**: People judge a sequence as more random-looking when it contains a shorter maximum consecutive run of the same outcome. A long streak of identical flips is the most salient cue that a sequence is not random, so when comparing two sequences people choose the one whose longest run is shorter, and the strength of that preference scales with how different the two maximum runs are.
- **iter0_candidate0**: People judge a sequence as more random-looking when its alternation rate is close to an internal ideal, but they penalize under-alternation (too few changes, making the sequence look predictable and patterned) more harshly than over-alternation (too many changes, which still looks somewhat erratic). This asymmetric sensitivity — a steeper slope on the under-alternation side than the over-alternation side — is the single mechanism driving randomness judgments.
- **iter1_candidate2**: People judge a sequence as more random-looking when its alternation rate is closer to 0.5, the rate a fair coin naturally produces. They carry a fixed internal prototype at exactly 0.5 — not a learned or context-dependent ideal — and their preference is driven by linear distance from that standard: whichever sequence's alternation rate deviates less from 0.5 looks more random, scaled by a single sensitivity parameter.
- **encoding_compressibility**: Random-looking sequences are those with low simple-description penalties:
long runs, periodic templates, and imbalance.
- **iter1_candidate1**: People judge a sequence as more random-looking when it contains less periodic structure — fewer regularly-repeating patterns. A truly random sequence should have no detectable rhythm or cycle, so a sequence with high periodicity looks engineered rather than chance-produced. When comparing two sequences, people choose the one with lower periodicity as the more random-looking one.
- **bayesian_markov_fairness**: People implicitly compute the log-Bayes-factor comparing each sequence's
observed transitions against a fair Markov chain (p_transition = 0.5) versus
a biased one, and choose the sequence whose transitions are more consistent
with the fair-coin hypothesis.
- **length_sensitive_alternation**: People evaluate alternation deviation on the count scale rather than the
proportion scale — a sequence with two extra transitions relative to the
ideal count looks equally deviant regardless of whether it is 4 or 8 flips
long, so the randomness signal scales with sequence length.

## Distinguishability (arviz.compare, PSIS-LOO)

`elpd_diff` and `dse` are relative to the best model. A model is only clearly worse than the best when `elpd_diff > 2 * dse`; models within ~2·dse of the top are statistically indistinguishable.

| model | elpd_diff | dse | distinguishable from best | weight |
| --- | --- | --- | --- | --- |
| iter1_candidate0 | 0.00 | 0.00 | — (best) | 0.924 |
| prototype_similarity | 2.98 | 1.31 | yes | 0.000 |
| inner_loop_model | 3.64 | 2.74 | no (within ~2·dse) | 0.000 |
| iter0_candidate2 | 4.87 | 3.14 | no (within ~2·dse) | 0.000 |
| iter0_candidate0 | 5.48 | 3.04 | no (within ~2·dse) | 0.000 |
| iter0_candidate1 | 5.82 | 3.69 | no (within ~2·dse) | 0.076 |
| encoding_compressibility | 6.11 | 3.20 | no (within ~2·dse) | 0.000 |
| iter1_candidate2 | 6.80 | 3.85 | no (within ~2·dse) | 0.000 |
| bayesian_markov_fairness | 9.99 | 4.33 | yes | 0.000 |
| iter1_candidate1 | 10.50 | 4.64 | yes | 0.000 |
| length_sensitive_alternation | 11501.71 | 521.69 | yes | 0.000 |
