# Inner Model Loop Report

Each model below is ONE distinct cognitive hypothesis. The posterior mass shows which single hypothesis best explains the data — it is **not** a recipe to combine the top models into a blend.

- Best model: **bayesian_diagnosticity** (posterior=0.780, elpd_loo=-71.85)
- Trials: 320
- Models compared: 10

## Posterior over models (ELPD-LOO)

| model | posterior | elpd_loo |
| --- | --- | --- |
| bayesian_diagnosticity | 0.7802 | -71.85 |
| prototype_similarity | 0.2198 | -74.87 |
| encoding_compressibility | 0.0000 | -98.86 |
| statistical_inference | 0.0000 | -115.94 |
| iter0_candidate0 | 0.0000 | -167.84 |
| iter0_candidate1 | 0.0000 | -116.49 |
| iter0_candidate2 | 0.0000 | -171.91 |
| iter1_candidate0 | 0.0000 | -222.74 |
| iter1_candidate1 | 0.0000 | -222.78 |
| iter1_candidate2 | 0.0000 | -196.69 |

## Hypotheses

- **bayesian_diagnosticity**: Random-looking sequences are diagnostic of a fair coin over salient
non-random alternatives: alternating, biased, and streaky generators.
- **prototype_similarity**: Random-looking sequences are close to a prototype with balanced H/T counts
and an ideal alternation rate.
- **encoding_compressibility**: Random-looking sequences are those with low simple-description penalties:
long runs, periodic templates, and imbalance.
- **statistical_inference**: Randomness is the log-likelihood ratio of a fair coin versus a
complexity-penalized motif process (Griffiths et al. 2018): sequences
with no short motif description are evidence for a random generator.
- **iter0_candidate0**: People judge which sequence looks more random by computing Bayesian diagnosticity against a single implicit alternative: a repetitive (streaky) generator that rarely alternates. The sequence whose log-likelihood ratio — fair coin versus repetitive generator — is higher is chosen as more random. The repetitive generator's alternation probability is a free parameter inferred from behavior, expected to sit well below 0.5.
- **iter0_candidate1**: People judge which sequence looks more random by focusing on the longest unbroken run of identical outcomes (e.g., five heads in a row). The sequence with the shorter maximum run length appears more random, because long streaks are the most perceptually salient violation of randomness expectations. All other structural features of the sequence are ignored.
- **iter0_candidate2**: People judge a sequence as more random when its proportion of heads is closer to 0.5 — that is, when its head-count is more balanced. The sequence with lower imbalance (smaller deviation from equal heads and tails) is chosen as more random. No other feature of the sequence — alternation patterns, run lengths, or motif structure — matters; only proximity to a 50/50 split drives the choice.
- **iter1_candidate0**: People judge sequences as random by comparing them to a mental prototype defined by two dimensions: alternation rate and head-balance. Crucially, they apply a conjunctive criterion — a sequence must be near the ideal on *both* dimensions to look random, so the dimension with the larger deviation from the prototype governs the judgment (Chebyshev / worst-case distance). A sequence that is perfectly balanced but wildly alternating looks just as non-random as one that alternates ideally but is heavily biased; only meeting both criteria simultaneously yields a high randomness rating.
- **iter1_candidate1**: People judge which sequence is more random by computing the Falk-Konold Difficulty Predictor (DP = repetition motifs + 2 × alternation motifs), which measures how many motifs are needed to describe the sequence under minimal run-based encoding. The sequence requiring more motifs — one that resists compact description because it has no simple repeating structure — is chosen as more random. Purely alternating and purely streaky sequences both have low DP and are rejected as non-random; only sequences that mix streaks and alternations in an unpredictable way have high DP and appear random.
- **iter1_candidate2**: People judge which sequence looks more random by detecting periodic structure: regular, cyclic patterns signal a non-random (patterned or deterministic) generator, so the sequence with lower periodicity is chosen as more random. Only this one perceptual cue — how strongly a sequence repeats itself at regular intervals — drives the choice; balance and run structure are ignored.

## Distinguishability (arviz.compare, PSIS-LOO)

`elpd_diff` and `dse` are relative to the best model. A model is only clearly worse than the best when `elpd_diff > 2 * dse`; models within ~2·dse of the top are statistically indistinguishable.

| model | elpd_diff | dse | distinguishable from best | weight |
| --- | --- | --- | --- | --- |
| bayesian_diagnosticity | 0.00 | 0.00 | — (best) | 1.000 |
| prototype_similarity | 3.02 | 1.21 | yes | 0.000 |
| encoding_compressibility | 27.00 | 6.58 | yes | 0.000 |
| statistical_inference | 44.09 | 6.88 | yes | 0.000 |
| iter0_candidate1 | 44.64 | 7.07 | yes | 0.000 |
| iter0_candidate0 | 95.98 | 8.86 | yes | 0.000 |
| iter0_candidate2 | 100.06 | 9.11 | yes | 0.000 |
| iter1_candidate2 | 124.84 | 8.27 | yes | 0.000 |
| iter1_candidate0 | 150.88 | 7.81 | yes | 0.000 |
| iter1_candidate1 | 150.93 | 7.81 | yes | 0.000 |
