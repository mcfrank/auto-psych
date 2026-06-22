# Inner Model Loop Report

Each model below is ONE distinct cognitive hypothesis. The posterior mass shows which single hypothesis best explains the data — it is **not** a recipe to combine the top models into a blend.

- Best model: **iter1_candidate0** (posterior=0.994, elpd_loo=-2479.65)
- Trials: 3840
- Models compared: 14

## Posterior over models (ELPD-LOO)

| model | posterior | elpd_loo |
| --- | --- | --- |
| iter1_candidate0 | 0.9945 | -2479.65 |
| bayesian_diagnosticity | 0.0055 | -2484.89 |
| prototype_similarity | 0.0000 | -2543.22 |
| encoding_compressibility | 0.0000 | -2624.22 |
| window_typicality | 0.0000 | -2625.70 |
| smoothed_prototype_distance | 0.0000 | -2529.96 |
| falk_konold_complexity | 0.0000 | -2664.29 |
| inner_loop_model | 0.0000 | -2529.96 |
| gaussian_smoothed_prototype | 0.0000 | -2519.78 |
| binomial_feature_surprise | 0.0000 | -2635.67 |
| iter0_candidate0 | 0.0000 | -2570.47 |
| iter0_candidate1 | 0.0000 | -2660.27 |
| iter0_candidate2 | 0.0000 | -2634.68 |
| iter1_candidate2 | 0.0000 | -2622.10 |

## Hypotheses

- **iter1_candidate0**: People judge the randomness of a sequence by computing its Bayesian diagnosticity—the log-likelihood ratio of a fair coin versus a subjective "regular" generative process. We refine this model by adding an "artificial balance" component to the regular hypothesis, representing a generator that intentionally targets an exact equal count of heads and tails. This explains why people strongly penalize perfectly symmetric sequences, recognizing them as suspiciously contrived rather than naturally random.
- **bayesian_diagnosticity**: Randomness is the log-likelihood ratio of a fair coin versus a regular process: a mixture of a complexity-penalized motif process (Griffiths et al. 2018) and a biased coin. Merges the former diagnosticity and statistical-inference accounts.
- **prototype_similarity**: Random-looking sequences are close to a prototype with balanced H/T counts and an ideal alternation rate.
- **encoding_compressibility**: Random-looking sequences are those with low simple-description penalties: long runs, periodic templates, and imbalance.
- **window_typicality**: Random-looking sequences have a longest run typical of a fair coin seen through a limited memory window; over-long streaks look non-random (Hahn & Warren 2009).
- **smoothed_prototype_distance**: People judge the randomness of a sequence by comparing its feature proportions to a subjective ideal, but they estimate these proportions using Bayesian smoothing with subjective pseudo-counts, naturally tolerating extreme imbalances in short sequences because their prior pulls the estimates toward the ideal.
- **falk_konold_complexity**: People judge the randomness of a sequence by its structural complexity when parsed into continuous alternating and repeating sub-sequences (Falk & Konold's difficulty of encoding), perceiving sequences with a lower rate of sub-sequences as simpler and therefore less random.
- **inner_loop_model**: Best PyMC model found by the inner model-improvement loop.
- **gaussian_smoothed_prototype**: People judge randomness by estimating a sequence's feature proportions using Bayesian smoothing and evaluating its similarity to a subjective ideal, where this similarity follows a Gaussian generalization gradient that disproportionately penalizes glaring squared deviations.
- **binomial_feature_surprise**: People judge the randomness of a sequence by intuitively calculating the exact Binomial probability of its observed number of heads and alternations under a subjective ideal, naturally penalizing deviations more strictly in longer sequences without needing ad-hoc smoothing.
- **iter0_candidate0**: People judge the randomness of a sequence by comparing its smoothed feature proportions to a subjective ideal, but instead of expecting exact perfection, they expect a 'typical' amount of sampling variation. They penalize sequences based on the absolute difference between their smoothed deviations and these expected typical deviations, naturally penalizing exactly balanced sequences as suspiciously artificial.
- **iter0_candidate1**: People judge the randomness of a sequence by computing its likelihood under a Gambler's Fallacy mental model, expecting that the probability of a coin switching states increases the longer it produces the same outcome. They penalize sequences containing long runs because each successive identical flip violates a progressively stronger subjective expectation of an alternation.
- **iter0_candidate2**: People evaluate the randomness of a sequence by intuitively parsing it into runs of identical outcomes and estimating the likelihood of these run lengths. Instead of a standard geometric expectation, they assume the length of any streak follows a subjective Poisson distribution, which inherently applies a severe factorial penalty to excessively long runs and seamlessly accounts for both streak aversion and the preference for alternations without mixing separate heuristics.
- **iter1_candidate2**: People judge the randomness of a sequence by computing its Bayesian diagnosticity—the log-likelihood ratio of a fair coin versus a subjective "regular" generative process. We model this regular hypothesis as a two-state Markov chain with symmetric Beta priors on its transition probabilities, an alternative that effortlessly assigns high marginal probability to sequences with glaring anomalies like over-long streaks, global imbalance, or artificial perfect alternation, thereby heavily penalizing them as non-random without blending multiple ad-hoc heuristics.

## Distinguishability (arviz.compare, PSIS-LOO)

`elpd_diff` and `dse` are relative to the best model. A model is only clearly worse than the best when `elpd_diff > 2 * dse`; models within ~2·dse of the top are statistically indistinguishable. `LOO reliable` is False when PSIS-LOO flagged this model's estimate as untrustworthy (many high Pareto-k points) — its row should be read with caution.

| model | elpd_diff | dse | distinguishable from best | weight | LOO reliable |
| --- | --- | --- | --- | --- | --- |
| iter1_candidate0 ←selected | 0.00 | 0.00 | — (best) | 0.646 | yes |
| bayesian_diagnosticity | 5.24 | 3.57 | no (within ~2·dse) | 0.107 | yes |
| gaussian_smoothed_prototype | 40.13 | 12.61 | yes | 0.215 | yes |
| inner_loop_model | 50.32 | 13.10 | yes | 0.000 | yes |
| smoothed_prototype_distance | 50.32 | 13.10 | yes | 0.000 | yes |
| prototype_similarity | 63.57 | 12.70 | yes | 0.000 | yes |
| iter0_candidate0 | 90.82 | 14.09 | yes | 0.000 | yes |
| iter1_candidate2 | 142.45 | 17.70 | yes | 0.000 | yes |
| encoding_compressibility | 144.57 | 16.51 | yes | 0.000 | yes |
| window_typicality | 146.06 | 19.28 | yes | 0.032 | yes |
| iter0_candidate2 | 155.03 | 18.71 | yes | 0.000 | yes |
| binomial_feature_surprise | 156.03 | 18.94 | yes | 0.000 | yes |
| iter0_candidate1 | 180.63 | 18.95 | yes | 0.000 | yes |
| falk_konold_complexity | 184.64 | 18.67 | yes | 0.000 | yes |
