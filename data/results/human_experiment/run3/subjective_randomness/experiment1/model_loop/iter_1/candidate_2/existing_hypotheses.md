# Existing hypotheses

Each model below is ONE cognitive hypothesis, with how well it currently explains the data. Propose a hypothesis that is genuinely different from these, or a refinement of a single one of them — never a combination of several.

## prototype_similarity  — posterior 0.997, ELPD-LOO -864.21

Random-looking sequences are close to a prototype with balanced H/T counts
and an ideal alternation rate.

## encoding_compressibility  — posterior 0.000, ELPD-LOO -889.63

Random-looking sequences are those with low simple-description penalties:
long runs, periodic templates, and imbalance.

## bayesian_diagnosticity  — posterior 0.002, ELPD-LOO -867.87

Randomness is the log-likelihood ratio of a fair coin versus a regular
process: a mixture of a complexity-penalized motif process (Griffiths et
al. 2018) and a biased coin. Merges the former diagnosticity and
statistical-inference accounts.

## window_typicality  — posterior 0.000, ELPD-LOO -887.51

Random-looking sequences have a longest run typical of a fair coin seen
through a limited memory window; over-long streaks look non-random (Hahn &
Warren 2009).

## iter0_candidate0  — posterior 0.001, ELPD-LOO -871.43

People judge the randomness of a sequence based on its Gaussian similarity to an ideal subjective prototype, meaning the sequence's perceived randomness decays exponentially with its squared distance in feature space (proportion of heads and alternations) from the expected prototype. This formulation penalizes extreme deviations more severely than a linear distance metric.

## iter0_candidate1  — posterior 0.000, ELPD-LOO -877.19

People judge the randomness of a sequence by evaluating the statistical typicality of its macroscopic features. They compute the joint Binomial probability of observing the sequence's specific number of heads and alternations under their subjective expectations for a random process, perceiving sequences with highly improbable feature counts as less random.

## iter0_candidate2  — posterior 0.000, ELPD-LOO -886.39

People judge the randomness of a sequence by the diversity of its run lengths, perceiving sequences as more random when they contain an unpredictable mix of short and long streaks, which is evaluated as the Shannon entropy of the sequence's run-length distribution.

## iter1_candidate0

People judge the randomness of a sequence by evaluating its distance from an ideal subjective prototype, but their sensitivity to deviations follows a psychophysical power law with a compressive exponent. This sub-linear scaling means that while initial departures from the prototype are penalized, further deviations toward extremes (like perfect imbalance) result in diminishing additional penalties.

## iter1_candidate1

People judge the randomness of a sequence by comparing it to an ideal prototype, but they evaluate its departure from this ideal using the absolute difference in the raw counts of heads and alternations, rather than their proportions. By using unnormalized counts, people naturally exhibit length-dependent tolerance, applying much smaller penalties to perfectly imbalanced short sequences than to perfectly imbalanced long sequences.
