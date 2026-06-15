# Experiment 3 Design Rationale

**Number of stimuli:** 30
**EIG Range:** 0.1276 – 0.3026

## Methodology

We generated a candidate set of 100 stimulus pairs, heavily randomizing sequences of length 4 through 8, while explicitly injecting theoretically relevant corner cases (e.g. perfect alternations vs. pure runs: `HHHHHHHH` vs. `THTHTHTH`). Expected Information Gain (EIG) was then computed over the prior predictive distribution of the available PyMC models. The pipeline processed these candidates and retained the top 30 pairs with the highest EIG.

## Model Discrimination

The design primarily discriminates between cognitive models that focus on local sequence transitions (like the alternation-bias and feature-heuristic models) and those that evaluate global structural properties (like encoding compressibility and periodicity). 

This is evident in the selected highly-diagnostic pairs, which often pitch extreme sequence types against each other:
1. **Pure Runs vs. Irregularity**: E.g., `HHHHH` vs `THTTHTHH`. Models relying on runs or extreme imbalance will diverge significantly from models evaluating generic entropy.
2. **Pure Runs vs. Perfect Alternation**: E.g., `HHHHHHHH` vs `THTHTHTH`. This isolates predictions from models that recognize periodic sub-patterns (which see both as highly compressible/patterned) against simpler heuristic models that strongly penalize constant runs but accept alternations as random.

By ensuring a spread across different sequence lengths (N=4 to 8) and balancing sequence lengths within pairs, the experiment is heavily optimized to parse exactly which structural biases dictate human randomness perception.