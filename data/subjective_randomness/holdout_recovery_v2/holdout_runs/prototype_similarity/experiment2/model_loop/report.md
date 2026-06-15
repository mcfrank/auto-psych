# Inner Model Loop Report

Each model below is ONE distinct cognitive hypothesis. The posterior mass shows which single hypothesis best explains the data — it is **not** a recipe to combine the top models into a blend.

- Best model: **iter0_candidate0** (posterior=0.252, elpd_loo=-507.80)
- Trials: 1200
- Models compared: 10

## Posterior over models (ELPD-LOO)

| model | posterior | elpd_loo |
| --- | --- | --- |
| iter0_candidate0 | 0.2521 | -507.80 |
| inner_loop_model | 0.2343 | -507.82 |
| iter1_candidate0 | 0.2286 | -507.85 |
| encoding_compressibility | 0.1410 | -509.93 |
| bayesian_diagnosticity | 0.1403 | -508.64 |
| iter0_candidate1 | 0.0037 | -514.58 |
| alternation_prototype | 0.0000 | -544.10 |
| iter0_candidate2 | 0.0000 | -564.14 |
| iter1_candidate1 | 0.0000 | -796.40 |
| iter1_candidate2 | 0.0000 | -804.98 |

## Hypotheses

- **iter0_candidate0**: People judge sequences by how diagnostic they are of a fair-coin process against three salient non-random alternatives — alternating, biased, and streaky. While the alternating generator has a canonical switch probability (0.95), people's mental prototype for streakiness is flexible: the characteristic persistence of the streaky generator (its switch probability) is a learned cognitive parameter rather than a fixed constant at 0.15.
- **inner_loop_model**: Best PyMC model found by the inner model-improvement loop.
- **iter1_candidate0**: People judge sequences by how diagnostic they are of a fair-coin process against three salient non-random alternatives — alternating, biased, and streaky. While the alternating and streaky generators are defined by canonical transition probabilities, people's mental model of what a biased sequence looks like is flexible: the characteristic head probability of the biased generator is a learned cognitive parameter rather than a fixed constant.
- **encoding_compressibility**: Random-looking sequences are those with low simple-description penalties:
long runs, periodic templates, and imbalance.
- **bayesian_diagnosticity**: Random-looking sequences are diagnostic of a fair coin over salient
non-random alternatives: alternating, biased, and streaky generators.
- **iter0_candidate1**: People judge a sequence as more random when its proportion of heads is closer to 0.5. When comparing two sequences, they choose the one whose head count deviates less from an equal split as more random — imbalance is the sole cue driving the choice.
- **alternation_prototype**: People judge a sequence as more random when its alternation rate is closer
(L1 distance) to an internalized prototype value that is biased above 0.5,
reflecting the well-documented human tendency to overestimate alternation in
random sequences.
- **iter0_candidate2**: People judge a sequence as more random when its longest unbroken run of identical outcomes is shorter. The maximum run length is the sole cue: a long streak signals a non-random, streaky process, so people pick whichever sequence has the shorter maximum run as looking more random.
- **iter1_candidate1**: People judge a sequence as more random when it contains less detectable periodic structure. When comparing two sequences, they choose the one with lower periodicity as the more random-looking one — periodic regularity signals a non-random, patterned generator.
- **iter1_candidate2**: People judge a sequence as more random when its transitions between outcomes are closer to maximum uncertainty — that is, when each outcome is equally likely to be followed by the same or a different outcome. When comparing two sequences, people choose the one whose transition process has higher Shannon entropy (closer to 50% alternation rate) as the more random-looking sequence.

## Distinguishability (arviz.compare, PSIS-LOO)

`elpd_diff` and `dse` are relative to the best model. A model is only clearly worse than the best when `elpd_diff > 2 * dse`; models within ~2·dse of the top are statistically indistinguishable.

| model | elpd_diff | dse | distinguishable from best | weight |
| --- | --- | --- | --- | --- |
| iter0_candidate0 | 0.00 | 0.00 | — (best) | 0.640 |
| inner_loop_model | 0.02 | 0.38 | no (within ~2·dse) | 0.000 |
| iter1_candidate0 | 0.05 | 0.16 | no (within ~2·dse) | 0.000 |
| bayesian_diagnosticity | 0.84 | 0.18 | yes | 0.000 |
| encoding_compressibility | 2.13 | 3.61 | no (within ~2·dse) | 0.030 |
| iter0_candidate1 | 6.78 | 5.48 | no (within ~2·dse) | 0.282 |
| alternation_prototype | 36.31 | 8.72 | yes | 0.049 |
| iter0_candidate2 | 56.34 | 10.20 | yes | 0.000 |
| iter1_candidate1 | 288.60 | 19.62 | yes | 0.000 |
| iter1_candidate2 | 297.18 | 19.97 | yes | 0.000 |
