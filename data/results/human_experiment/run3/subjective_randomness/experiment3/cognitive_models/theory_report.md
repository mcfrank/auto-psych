# Theory Report — Experiment 3

## gaussian_smoothed_prototype
**Hypothesis:** People judge randomness by estimating a sequence's feature proportions using Bayesian smoothing and evaluating its similarity to a subjective ideal, where this similarity follows a Gaussian generalization gradient that disproportionately penalizes glaring squared deviations.
**Motivation:** The previous inner loop identified `iter1_candidate0` as an excellent candidate (posterior mass 0.3256, statistically indistinguishable from the top model), which proposed a Gaussian-like generalization gradient over Bayesian smoothed features. This translates that insight into a clean squared-distance hypothesis.
**Mechanism:** It implements the `smoothed_prototype_distance` mechanism (Bayesian smoothing of proportions) but replaces the absolute (linear) penalty with a squared penalty, reflecting an exponentially dropping Gaussian similarity gradient.

## binomial_feature_surprise
**Hypothesis:** People judge the randomness of a sequence by intuitively calculating the exact Binomial probability of its observed number of heads and alternations under a subjective ideal, naturally penalizing deviations more strictly in longer sequences without needing ad-hoc smoothing.
**Motivation:** Inspired by `iter1_candidate2` from the previous inner loop's exploration, which noted that probabilistic surprise of macroscopic features naturally handles length scaling (a major challenge in earlier experiments).
**Mechanism:** Evaluates sequences using the `pm.Binomial` log-probability of the exact counts of heads and alternations, avoiding continuous proportion approximations entirely. A higher exact log-probability under the subjective parameters maps to a higher randomness score.
