# Theory Report — Experiment 3

## power_law_accumulated_typicality
**Hypothesis:** People evaluate the randomness of a sequence by accumulating a subjective sense of typicality over its length, but this evidence integration exhibits diminishing returns scaling according to a power-law function of sequence length.
**Motivation:** Introduced because the `iter0_candidate0` model in the previous inner loop showed strong relative performance (statistically indistinguishable from the top model). This formally tests whether accumulated evidence for randomness saturates for very long sequences, rather than growing linearly.
**Mechanism:** Implements a power-law transformation on the sequence length multiplier (`pt.pow(n_a, length_power)`) before scaling typicality, governed by a `length_power` free parameter. This differs from linear accumulation by explicitly modeling diminishing returns to additional sequence elements.

## leaky_accumulated_typicality
**Hypothesis:** People evaluate the randomness of a sequence by accumulating a subjective sense of typicality over its events, but working memory for past events is leaky with exponential decay.
**Motivation:** Inspired by `iter0_candidate1` from the prior inner loop, which scored very close to the best model (indistinguishable from best). It tests whether saturation effects for longer sequences can be explained by recency bias/leaky working memory rather than a direct power law on total length.
**Mechanism:** Replaces the direct multiplication by sequence length `n` with an effective length under exponential decay: `(1 - gamma^n) / (1 - gamma)`, where `gamma` is a retention rate parameter. This distinctly captures the mechanism of event-by-event forgetting.

## asymmetric_alternation_typicality
**Hypothesis:** People evaluate the randomness of a sequence by accumulating a subjective sense of typicality over its length, but their penalty for deviating from the ideal alternation rate is asymmetric: clustering (under-alternating) is heavily penalized while over-alternating is penalized less severely.
**Motivation:** Derived from `iter1_candidate1` from the prior inner loop, a high-scoring candidate. The literature suggests humans expect alternations more than clustering in "random" sequences, making an asymmetric penalty a strong, distinct hypothesis for typicality models.
**Mechanism:** Introduces two separate half-normal weights (`w_alt_under` and `w_alt_over`) for alternation deviation, applied conditionally via `pt.switch` based on whether the sequence under- or over-alternates relative to the ideal rate. This differs from a symmetric quadratic penalty by allowing the model to adaptively penalize clustering more than over-switching.