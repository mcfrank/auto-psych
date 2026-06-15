# Theory Report — Experiment 2

## extended_compressibility
**Motivation:** The inner model loop in Experiment 1 found that a linear combination model using `max_run_norm`, `periodicity`, and `imbalance` achieved the best predictive performance (ELPD-LOO = -407.78). However, previous cognitive literature indicates alternation proportion (`p_alts`) is a highly salient feature for subjective randomness, which was absent from the winning model. We include it here to see if combining compressibility metrics with alternation metrics improves predictive accuracy.
**Mechanism:** Implements a linear decision rule over sequence features, extending the previously best-performing compressibility features (run length, periodicity, imbalance) by adding the proportion of alternations. 

## alternation_and_run
**Motivation:** The full inner loop model is effective but potentially overparameterized. We extract the most commonly cited heuristics—longest run and alternation rate—into a simplified linear model to test if these two features alone capture the bulk of the subjective randomness variance seen in Experiment 1.
**Mechanism:** Uses a simple linear combination of normalized maximum run length and alternation proportion to produce a subjective randomness score.
