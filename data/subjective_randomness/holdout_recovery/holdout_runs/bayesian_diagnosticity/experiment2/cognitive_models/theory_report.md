# Theory Report — Experiment 2

## inferred_mixture_model
**Motivation:** The previous inner loop identified a Bayesian Mixture model (`inner_loop_model`) as the most successful, outperforming simple feature heuristics like prototype similarity and encoding compressibility. However, it hardcoded its alternative structural hypotheses (`ALT_SWITCH_PROB = 0.95`, `STREAK_SWITCH_PROB = 0.15`, `BIAS_HEAD_PROB = 0.85`). This model variant relaxes that assumption and allows the optimal transition and bias probabilities to be fit to human data directly, testing whether people have different priors for structured alternatives than the hardcoded values.
**Mechanism:** Computes the length-normalized log Bayes factor between a fair coin and a mixture of three alternative hypotheses (alternating Markov, streaky Markov, biased coin). Crucially, the transition and bias parameters for these alternatives are estimated as free cognitive parameters rather than fixed constants.

## unnormalized_mixture_model
**Motivation:** The successful `inner_loop_model` length-normalized its log likelihoods. It is theoretically debated in the literature whether subjective Bayes factors scale linearly with the sequence length $N$ or whether judges extract a per-observation summary statistic (normalization). This model tests the necessity of the length normalization component.
**Mechanism:** Identical to the `inner_loop_model` mixture of a fair coin versus three alternatives (with fixed hardcoded structural constants), but computes raw log probabilities instead of dividing by the sequence length $N$.
