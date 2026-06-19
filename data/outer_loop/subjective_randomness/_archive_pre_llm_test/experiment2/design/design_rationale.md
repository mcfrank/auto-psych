# Design Rationale

## Overview
This design selects 30 stimulus pairs of coin flips (H/T) to maximize Expected Information Gain (EIG) over a set of competing cognitive models for subjective randomness.

## Candidate Generation
We generated ~300 candidate sequence pairs by randomly sampling all possible coin flip sequences of lengths 3 to 8. This guarantees variability in lengths, head-proportions, and alternation rates.

## EIG and Model Discrimination
The selected 30 stimuli have an EIG range of 0.0840 to 0.1514. This design aims to discriminate among 6 theoretical models:
1. `equally_likely`: Distance of H-proportion from 50%.
2. `alternation_rate`: Deviation of alternation proportion from a subjective ideal.
3. `bayesian_fair_coin`: Log Bayes factor for a fair-coin null vs. biased-coin alternative.
4. `inner_loop_model`: The champion model emerging from the inner model-improvement loop.
5. `subjective_markov_evidence`: Log Bayes factor of transitions under a subjective Markov process vs. a fair coin.
6. `absolute_alternation_deviation`: Distance of alternation absolute count from expected under subjective ideal rate.

The sequence pairs selected frequently pit extreme sequences (e.g., highly alternating or long constant-character runs) against more balanced or shorter sequences, effectively pulling apart predictions made by models relying on global proportion versus local transition patterns.
