# Experiment 3 Design Rationale

## Candidate Generation
To ensure a tractable design phase, 300 unique pairs of sequences of coin flips (H and T) were randomly sampled from the full space of sequences with lengths between 2 and 8 (inclusive). This sample included pairs with both matching and mixed lengths to cover a broad space of possible stimuli.

## EIG Scoring and Selection
The 300 candidate pairs were evaluated by computing the Expected Information Gain (EIG) over the theorist's PyMC models. The models evaluated represent a range of cognitive theories including bayesian diagnosticity, binomial feature surprise, encoding compressibility, Falk & Konold complexity, Gaussian smoothed prototypes, window typicality, and other related variations. 

The top 32 stimuli were greedily selected to maximize joint informativeness. 

## Design Summary
- **Number of Stimuli**: 32 pairs
- **EIG Range**: 0.140 to 0.240
- **Model Discrimination**: The selected set discriminates between models by presenting contrastive pairs that evoke different judgments from complexity-based theories (e.g., encoding compressibility, Falk & Konold complexity), similarity/prototype theories (e.g., prototype similarity, smoothed prototype distance), and statistical/diagnosticity theories (e.g., bayesian diagnosticity, binomial feature surprise). The presence of both long and short sequence comparisons, along with sequences of varying alternations and string balances, targets the divergent predictions of these theoretical accounts.