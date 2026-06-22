# Existing hypotheses

Each model below is ONE cognitive hypothesis, with how well it currently explains the data. Propose a hypothesis that is genuinely different from these, or a refinement of a single one of them — never a combination of several.

## prototype_similarity  — posterior 0.000, ELPD-LOO -2543.22

Random-looking sequences are close to a prototype with balanced H/T counts and an ideal alternation rate.

## encoding_compressibility  — posterior 0.000, ELPD-LOO -2624.22

Random-looking sequences are those with low simple-description penalties: long runs, periodic templates, and imbalance.

## bayesian_diagnosticity  — posterior 1.000, ELPD-LOO -2484.89

Randomness is the log-likelihood ratio of a fair coin versus a regular process: a mixture of a complexity-penalized motif process (Griffiths et al. 2018) and a biased coin. Merges the former diagnosticity and statistical-inference accounts.

## window_typicality  — posterior 0.000, ELPD-LOO -2625.70

Random-looking sequences have a longest run typical of a fair coin seen through a limited memory window; over-long streaks look non-random (Hahn & Warren 2009).

## smoothed_prototype_distance  — posterior 0.000, ELPD-LOO -2529.96

People judge the randomness of a sequence by comparing its feature proportions to a subjective ideal, but they estimate these proportions using Bayesian smoothing with subjective pseudo-counts, naturally tolerating extreme imbalances in short sequences because their prior pulls the estimates toward the ideal.

## falk_konold_complexity  — posterior 0.000, ELPD-LOO -2664.29

People judge the randomness of a sequence by its structural complexity when parsed into continuous alternating and repeating sub-sequences (Falk & Konold's difficulty of encoding), perceiving sequences with a lower rate of sub-sequences as simpler and therefore less random.

## inner_loop_model  — posterior 0.000, ELPD-LOO -2529.96

Best PyMC model found by the inner model-improvement loop.

## gaussian_smoothed_prototype  — posterior 0.000, ELPD-LOO -2519.78

People judge randomness by estimating a sequence's feature proportions using Bayesian smoothing and evaluating its similarity to a subjective ideal, where this similarity follows a Gaussian generalization gradient that disproportionately penalizes glaring squared deviations.

## binomial_feature_surprise  — posterior 0.000, ELPD-LOO -2635.67

People judge the randomness of a sequence by intuitively calculating the exact Binomial probability of its observed number of heads and alternations under a subjective ideal, naturally penalizing deviations more strictly in longer sequences without needing ad-hoc smoothing.

## iter0_candidate0

People judge the randomness of a sequence by comparing its smoothed feature proportions to a subjective ideal, but instead of expecting exact perfection, they expect a 'typical' amount of sampling variation. They penalize sequences based on the absolute difference between their smoothed deviations and these expected typical deviations, naturally penalizing exactly balanced sequences as suspiciously artificial.

## iter0_candidate1

People judge the randomness of a sequence by computing its likelihood under a Gambler's Fallacy mental model, expecting that the probability of a coin switching states increases the longer it produces the same outcome. They penalize sequences containing long runs because each successive identical flip violates a progressively stronger subjective expectation of an alternation.
