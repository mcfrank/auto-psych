# Inner Model Loop Report

Each model below is ONE distinct cognitive hypothesis. The posterior mass shows which single hypothesis best explains the data — it is **not** a recipe to combine the top models into a blend.

- Best model: **iter0_candidate1** (posterior=0.620, elpd_loo=-1598.07)
- Trials: 2560
- Models compared: 13

## Posterior over models (ELPD-LOO)

| model | posterior | elpd_loo |
| --- | --- | --- |
| iter0_candidate1 | 0.6198 | -1598.07 |
| iter1_candidate0 | 0.3664 | -1598.50 |
| iter0_candidate2 | 0.0139 | -1600.77 |
| prototype_similarity | 0.0000 | -1632.53 |
| encoding_compressibility | 0.0000 | -1722.85 |
| bayesian_diagnosticity | 0.0000 | -1631.89 |
| window_typicality | 0.0000 | -1711.20 |
| inner_loop_model | 0.0000 | -1618.54 |
| falk_konold_difficulty | 0.0000 | -1756.79 |
| manhattan_messy_prototype | 0.0000 | -1619.58 |
| iter0_candidate0 | 0.0000 | -1663.65 |
| iter1_candidate1 | 0.0000 | -1699.39 |
| iter1_candidate2 | 0.0000 | -1618.71 |

## Hypotheses

- **iter0_candidate1**: Hypothesis: Random-looking sequences are judged by an evidence accumulation process where each item in the sequence provides a baseline "weight of evidence" for randomness, which is then discounted by the sequence's quadratic deviation from a messy prototype (an ideal positive imbalance and ideal alternation rate). This mechanism naturally creates a preference for longer sequences when they are near-ideal, but more heavily penalizes long sequences that clearly deviate from the prototype.
- **iter1_candidate0**: Random-looking sequences are judged by an evidence accumulation process where each distinct run (streak of identical outcomes) provides a baseline weight of evidence for randomness, which is then discounted by the sequence's deviation from a messy prototype (ideal imbalance and alternation rate). By accumulating evidence per run rather than per item, this mechanism naturally penalizes periodic patterns with artificially few runs (like HHTTHHTT) without needing a separate periodicity cue, while still using the prototype distance to heavily penalize extreme over-alternation.
- **iter0_candidate2**: People evaluate randomness by assessing the structural diversity of a sequence's run-lengths. They compute a positive cognitive score based on the combinatorial diversity of the sequence (the log number of ways to arrange the observed counts of short, medium, and long runs) combined with a weighted preference for the sheer number of those runs. Because this mechanism accumulates positive structural evidence, longer balanced sequences naturally achieve higher scores than shorter ones, while highly regular sequences (zero diversity) and extremely imbalanced sequences (very few runs) are inherently penalized.
- **prototype_similarity**: Random-looking sequences are close to a prototype with balanced H/T counts and an ideal alternation rate.
- **encoding_compressibility**: Random-looking sequences are those with low simple-description penalties like long runs, periodic templates, and imbalance.
- **bayesian_diagnosticity**: Randomness is the log-likelihood ratio of a fair coin versus a regular process (a mixture of a complexity-penalized motif process and a biased coin).
- **window_typicality**: Random-looking sequences have a longest run typical of a fair coin seen through a limited memory window; over-long streaks look non-random.
- **inner_loop_model**: Random-looking sequences are judged by their Euclidean distance to a messy prototype with a strictly positive ideal imbalance and an ideal alternation rate, with quadratic asymmetric penalties.
- **falk_konold_difficulty**: Random-looking sequences are those that are cognitively harder to encode into chunks, as quantified by Falk and Konold's Difficulty Predictor (the number of repetition motifs plus twice the number of alternation motifs) normalized by sequence length.
- **manhattan_messy_prototype**: Random-looking sequences are judged by their Manhattan (absolute) distance to a messy prototype with a strictly positive ideal imbalance and an ideal alternation rate, with absolute-value penalties that are asymmetric for alternation rate.
- **iter0_candidate0**: Random-looking sequences are judged by their deviation from a messy prototype (an ideal positive imbalance and ideal alternation rate), but the cognitive penalty grows as a quartic (power of 4) function of the deviation rather than a quadratic one. This different functional form creates a much wider, flatter tolerance for near-ideal sequences but imposes significantly harsher penalties on extreme deviations like exact balance or severe imbalance.
- **iter1_candidate1**: Random-looking sequences are judged by an evidence accumulation process where baseline evidence is discounted by the sequence's deviation from a messy prototype, but rather than tracking local alternation rates, people explicitly penalize global repeating templates (periodicity). The prototype thus consists of an ideal imbalance and an ideal (low) periodicity, naturally explaining the harsh rejection of structured, repeating sequences like HHTTHHTT that a simple alternation-rate penalty misses.
- **iter1_candidate2**: Random-looking sequences are judged by an evidence accumulation process where each item provides a baseline weight of evidence, which is discounted by the sequence's quadratic deviation from a messy prototype. Crucially, this prototype evaluates alternation not as an absolute rate, but relative to the maximum possible alternations given the sequence's specific tally of heads and tails (alongside a penalty for non-ideal imbalance). By penalizing sequences that deviate from an ideal *relative* alternation rate, this mechanism naturally captures why people reject highly structured or periodic sequences (like HHTTHHTT) that have typical absolute alternation rates but are severely under-alternating for their composition.

## Distinguishability (arviz.compare, PSIS-LOO)

`elpd_diff` and `dse` are relative to the best model. A model is only clearly worse than the best when `elpd_diff > 2 * dse`; models within ~2·dse of the top are statistically indistinguishable. `LOO reliable` is False when PSIS-LOO flagged this model's estimate as untrustworthy (many high Pareto-k points) — its row should be read with caution.

| model | elpd_diff | dse | distinguishable from best | weight | LOO reliable |
| --- | --- | --- | --- | --- | --- |
| iter0_candidate1 ←selected | 0.00 | 0.00 | — (best) | 0.226 | yes |
| iter1_candidate0 | 0.43 | 2.16 | no (within ~2·dse) | 0.000 | yes |
| iter0_candidate2 | 2.70 | 7.66 | no (within ~2·dse) | 0.501 | yes |
| inner_loop_model | 20.47 | 8.53 | yes | 0.001 | yes |
| iter1_candidate2 | 20.63 | 6.00 | yes | 0.000 | yes |
| manhattan_messy_prototype | 21.51 | 8.80 | yes | 0.271 | yes |
| bayesian_diagnosticity | 33.82 | 8.99 | yes | 0.000 | yes |
| prototype_similarity | 34.45 | 10.21 | yes | 0.000 | yes |
| iter0_candidate0 | 65.58 | 11.57 | yes | 0.000 | yes |
| iter1_candidate1 | 101.32 | 15.13 | yes | 0.000 | yes |
| window_typicality | 113.13 | 16.35 | yes | 0.001 | yes |
| encoding_compressibility | 124.78 | 16.30 | yes | 0.000 | yes |
| falk_konold_difficulty | 158.72 | 17.18 | yes | 0.000 | yes |
