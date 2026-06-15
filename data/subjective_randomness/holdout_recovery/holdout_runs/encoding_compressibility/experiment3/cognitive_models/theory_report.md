# Theory Report — Experiment 3

## student_extended_compressibility
**Motivation:** The previous model loop showed that the inner loop candidate using `StudentT` priors over features (`iter0_candidate2`, which was the best model at -680.91 ELPD-LOO) performed slightly better than models relying on strictly normal or beta priors, likely because heavy-tailed priors can capture both small background effects and occasional strong heuristics without over-regularization. We apply this lesson to `extended_compressibility` (which performed decently at -682.02 ELPD-LOO) by using `StudentT` priors for the weights of all compressibility features including alternation rate.
**Mechanism:** The model combines normalized run length, periodicity, imbalance, and alternation proportion. It differs from the base `extended_compressibility` only by using the more robust, heavy-tailed priors discovered by the inner loop.

## length_sensitive_compressibility
**Motivation:** While the inner loop models capture structure within the sequences well, none of the top-performing models in the previous experiment account for sequence length differences when evaluating sequences. The stimuli involve sequences of different lengths (e.g., 4 vs 6), and human judgment might penalize shorter sequences as inherently less random.
**Mechanism:** This model adds an explicit main effect of sequence length (`n`) alongside the core features discovered by the inner loop (normalized max run length, periodicity, and imbalance), testing whether length independently contributes to the subjective randomness score.
