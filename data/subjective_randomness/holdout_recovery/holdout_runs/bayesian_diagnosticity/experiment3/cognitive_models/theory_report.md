# Theory Report — Experiment 3

## feature_heuristic_model
**Motivation:** In experiment 2, `encoding_compressibility` and `prototype_similarity` performed poorly compared to the complex mixture models (`inner_loop_model`). However, it remains possible that participants use a simple feature-based heuristic rather than a full Bayesian likelihood comparison. This model tests whether directly weighing alternation proportion, imbalance, and max run length provides a parsimonious alternative to the Bayesian mixture models.
**Mechanism:** Participants linearly combine standardized feature values (`p_alts`, `imbalance`, `max_run_norm`) to compute a "randomness score" for each sequence, bypassing explicit likelihood calculations entirely.

## fair_vs_alternating
**Motivation:** `inner_loop_model` and `inferred_mixture_model` were highly successful in Experiment 2, but they assume participants contrast a fair coin with a mixture of *three* distinct alternatives (alternating, streaky, biased). It is worth testing if all three are necessary, or if the primary driver of subjective randomness is merely the contrast between a fair coin and an over-alternating process.
**Mechanism:** This model simplifies the Bayesian setup by evaluating only the log Bayes factor between a fair coin and an inferred alternating Markov process, omitting streaky and biased alternatives. This isolates the core cognitive principle of alternation bias.