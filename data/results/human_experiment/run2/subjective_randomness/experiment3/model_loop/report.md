# Inner Model Loop Report

Each model below is ONE distinct cognitive hypothesis. The posterior mass shows which single hypothesis best explains the data — it is **not** a recipe to combine the top models into a blend.

- Best model: **inner_loop_model** (posterior=0.492, elpd_loo=-2418.99)
- Trials: 3840
- Models compared: 15

## Posterior over models (ELPD-LOO)

| model | posterior | elpd_loo |
| --- | --- | --- |
| inner_loop_model | 0.4920 | -2418.99 |
| evidence_accumulation_per_run | 0.4600 | -2419.06 |
| iter0_candidate2 | 0.0318 | -2422.48 |
| iter1_candidate1 | 0.0162 | -2422.95 |
| prototype_similarity | 0.0000 | -2453.26 |
| encoding_compressibility | 0.0000 | -2573.96 |
| bayesian_diagnosticity | 0.0000 | -2449.28 |
| window_typicality | 0.0000 | -2561.23 |
| falk_konold_difficulty | 0.0000 | -2610.91 |
| manhattan_messy_prototype | 0.0000 | -2445.50 |
| evidence_accumulation_periodicity | 0.0000 | -2616.78 |
| iter0_candidate0 | 0.0000 | -2474.98 |
| iter0_candidate1 | 0.0000 | -2519.08 |
| iter1_candidate0 | 0.0000 | -2469.83 |
| iter1_candidate2 | 0.0000 | -2557.52 |

## Hypotheses

- **inner_loop_model**: Random-looking sequences are judged by an evidence accumulation process where each item in the sequence provides a baseline "weight of evidence" for randomness, which is then discounted by the sequence's quadratic deviation from a messy prototype (an ideal positive imbalance and ideal alternation rate).
- **evidence_accumulation_per_run**: Random-looking sequences are judged by an evidence accumulation process where each distinct run (streak of identical outcomes) provides a baseline weight of evidence for randomness, which is then discounted by the sequence's quadratic deviation from a messy prototype.
- **iter0_candidate2**: Random-looking sequences are judged by an evidence accumulation process where each item provides a baseline weight of evidence for randomness. This accumulated evidence is then discounted by the sequence's deviation from a messy prototype—an ideal positive imbalance and an ideal alternation rate—using an asymmetric quadratic penalty that punishes under-alternation (streaks) with a different, flexibly stronger weight than over-alternation.
- **iter1_candidate1**: Random-looking sequences are judged by an evidence accumulation process where each item provides a baseline weight of evidence, which is discounted by the sequence's quadratic deviation from a three-dimensional messy prototype. This prototype consists of an ideal positive imbalance, an ideal alternation rate, and an ideal normalized maximum run length, directly penalizing sequences that contain disproportionately long local streaks even when their global alternation rate is typical.
- **prototype_similarity**: Random-looking sequences are close to a prototype with balanced H/T counts and an ideal alternation rate.
- **encoding_compressibility**: Random-looking sequences are those with low simple-description penalties like long runs, periodic templates, and imbalance.
- **bayesian_diagnosticity**: Randomness is the log-likelihood ratio of a fair coin versus a regular process (a mixture of a complexity-penalized motif process and a biased coin).
- **window_typicality**: Random-looking sequences have a longest run typical of a fair coin seen through a limited memory window; over-long streaks look non-random.
- **falk_konold_difficulty**: Random-looking sequences are those that are cognitively harder to encode into chunks, as quantified by Falk and Konold's Difficulty Predictor (the number of repetition motifs plus twice the number of alternation motifs) normalized by sequence length.
- **manhattan_messy_prototype**: Random-looking sequences are judged by their Manhattan (absolute) distance to a messy prototype with a strictly positive ideal imbalance and an ideal alternation rate, with absolute-value penalties that are asymmetric for alternation rate.
- **evidence_accumulation_periodicity**: Random-looking sequences are judged by an evidence accumulation process where baseline evidence is discounted by the sequence's deviation from a messy prototype consisting of an ideal imbalance and an ideal (low) periodicity.
- **iter0_candidate0**: Random-looking sequences are judged by an evidence accumulation process where each item provides a baseline weight of evidence for randomness, which is then discounted by the sequence's quadratic deviation from a messy prototype defined by an ideal repeating-motif rate and an ideal alternating-motif rate.
- **iter0_candidate1**: Random-looking sequences are judged by an evidence accumulation process where each item in the sequence provides a baseline weight of evidence for randomness. This accumulated evidence is then discounted by the sequence's deviation from a messy prototype, which is characterized by an ideal positive imbalance and an aversion to long streaks of identical outcomes.
- **iter1_candidate0**: Random-looking sequences are judged by an evidence accumulation process where each item provides a baseline weight of evidence, which is discounted by the sequence's deviation from a messy prototype. Rather than evaluating the global alternation rate, this prototype specifies an ideal run length, and each run incurs a penalty proportional to its squared deviation from this ideal length (alongside a penalty for overall imbalance), simultaneously punishing extreme alternation rates and disproportionately long local streaks.
- **iter1_candidate2**: Random-looking sequences are evaluated by their constituent streaks (runs), where each run incurs a cognitive penalty proportional to its squared deviation from an ideal 'messy' run length. Because this single penalty operates locally on every run, it naturally accounts for ideal alternation rates, strongly punishes disproportionately long clusters, and implicitly penalizes global imbalance without requiring separate tracking of these global features.

## Distinguishability (arviz.compare, PSIS-LOO)

`elpd_diff` and `dse` are relative to the best model. A model is only clearly worse than the best when `elpd_diff > 2 * dse`; models within ~2·dse of the top are statistically indistinguishable. `LOO reliable` is False when PSIS-LOO flagged this model's estimate as untrustworthy (many high Pareto-k points) — its row should be read with caution.

| model | elpd_diff | dse | distinguishable from best | weight | LOO reliable |
| --- | --- | --- | --- | --- | --- |
| inner_loop_model ←selected | 0.00 | 0.00 | — (best) | 0.000 | yes |
| evidence_accumulation_per_run | 0.07 | 2.64 | no (within ~2·dse) | 0.084 | yes |
| iter0_candidate2 | 3.49 | 7.69 | no (within ~2·dse) | 0.281 | yes |
| iter1_candidate1 | 3.96 | 9.40 | no (within ~2·dse) | 0.271 | yes |
| manhattan_messy_prototype | 26.51 | 9.27 | yes | 0.000 | yes |
| bayesian_diagnosticity | 30.29 | 10.16 | yes | 0.120 | yes |
| prototype_similarity | 34.27 | 10.68 | yes | 0.000 | yes |
| iter1_candidate0 | 50.84 | 13.08 | yes | 0.215 | yes |
| iter0_candidate0 | 55.99 | 12.99 | yes | 0.029 | yes |
| iter0_candidate1 | 100.09 | 16.92 | yes | 0.000 | yes |
| iter1_candidate2 | 138.53 | 17.73 | yes | 0.000 | yes |
| window_typicality | 142.24 | 18.49 | yes | 0.000 | yes |
| encoding_compressibility | 154.97 | 19.04 | yes | 0.000 | yes |
| falk_konold_difficulty | 191.92 | 19.27 | yes | 0.000 | yes |
| evidence_accumulation_periodicity | 197.79 | 18.77 | yes | 0.000 | yes |
