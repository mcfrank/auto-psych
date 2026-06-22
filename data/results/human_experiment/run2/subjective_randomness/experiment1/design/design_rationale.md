# Design Rationale

## Overview
This design includes 32 stimulus pairs selected to maximize Expected Information Gain (EIG) among a suite of subjective randomness cognitive models, including `bayesian_diagnosticity`, `encoding_compressibility`, `prototype_similarity`, and `window_typicality`.

## Pool Generation
A candidate pool of 300 sequence pairs was generated uniformly from the space of all possible coin flip sequences of lengths 2 to 8. The generation deliberately sampled both pairs of the same length and pairs of different lengths to ensure sufficient variance in the candidate features (length, alternations, balance, run length, motifs).

## EIG Selection and Range
The EIG script computed the prior predictive distributions of the cognitive models and selected the 32 pairs with the highest EIG. 
- **Stimulus count:** 32 pairs
- **EIG range:** 0.182 to 0.354

## Model Discrimination
The final design distinguishes between models by contrasting specific feature combinations:
- Models such as `prototype_similarity` often focus on local properties (e.g., specific string length and expected number of alternations vs. streaks). 
- Models like `encoding_compressibility` depend on the motif structure and repetition/alternation compression length, penalizing lengthy runs. 
- `bayesian_diagnosticity` compares relative probabilities under a fair coin vs an alternative hypothesis, which makes contrasting sequence lengths and the extent of deviations from uniformity highly diagnostic.

The chosen pairs (e.g., comparing "THTTTTHH" to "TTTHHTTT" or "HHTH" to "TTTTHHTH") vary run lengths, alternations, and sequence lengths to tease apart precisely these differences in the theoretical accounts.
