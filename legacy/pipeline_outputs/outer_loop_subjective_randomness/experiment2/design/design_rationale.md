# Design Rationale — Experiment 2

## Overview

Experiment 2 extends the three-model setup from experiment 1 (alternation_bias, balance_heuristic, griffiths_representativeness) with two new models: **run_aversion** and **weighted_balance_alternation**. The design targets pairs that maximally discriminate between all five models.

## Candidate Generation

110 candidate pairs were generated covering the following contrast types:

### 1. run_aversion vs alternation_bias divergent pairs
The key theoretical prediction unique to experiment 2: `run_aversion` scores sequences by the inverse of their maximum run length, while `alternation_bias` scores by total alternation rate. These can diverge when a sequence has many small runs (low max_run, moderate alt) vs. fewer but clustered alternations (higher alt, longer max run). Example: HHTTHHTT (max_run=2, alt≈0.43) vs HHHTHTHT (max_run=3, alt≈0.71).

### 2. griffiths_representativeness vs alternation_bias pairs
GR uniquely penalizes *over-alternation*: sequences with alt_rate near 1.0 are judged *less* random than those near 0.5. Pairs like HTHTH vs HHTTH (both 3H2T, but HHTTH has alt=0.5) produce large GR-vs-AB disagreements.

### 3. balance_heuristic vs weighted_balance_alternation
WBA combines balance and alternation equally; BH ignores alternation. Pairs where one sequence is imbalanced-but-alternating vs. balanced-but-streaky expose this difference.

### 4. Cross-length pairs
Pairs of different lengths (e.g., 5 vs 6, 4 vs 8) to test whether model predictions generalize across sequence lengths.

### 5. Anchors
Pairs with near-universal model agreement (e.g., HTHTHTHT vs HHHHHHHH) to verify participants understand the task.

## EIG Scoring and Top-20 Selection

All 110 candidates were scored by Expected Information Gain under a uniform prior over the 5 models. The EIG quantifies how much each stimulus reduces posterior uncertainty across models.

- **EIG range selected**: 0.2803 – 0.3799
- **Top 20 stimuli** were retained for the experiment

The highest-EIG pairs are cross-length GR-vs-AB contrasts (e.g., HTHTH vs HHHTTT, EIG=0.380; HTHTH vs HTTTH, EIG=0.357), indicating that moderate-alternation sequences near the 0.5 ideal are the most discriminative. Key run_aversion contrasts appear in the selected set (e.g., HHTHTTTH vs HTHTHTH, EIG=0.310), and WBA vs BH pairs appear near the bottom of the selection (e.g., HTHTHTHH vs TTTTHHHH and HTHHTHTH vs HHHHTTTT, EIG≈0.280).

## Model-Coverage Summary

| Primary contrast | Example pair | EIG |
|---|---|---|
| GR vs AB (over-alternation) | HTHTH vs HHHTTT | 0.380 |
| GR vs AB (alt=0.5 ideal) | HTHTH vs HTTTH | 0.357 |
| GR/RA vs AB (cross-length) | HTHTH vs HHTTHHTT | 0.343 |
| GR/RA vs AB | HTHTHTH vs HHHHTTTT | 0.338 |
| RA vs AB (max-run divergent) | HHTHTTTH vs HTHTHTH | 0.310 |
| GR vs AB (long length) | HTHTHTHT vs HHHTTTTH | 0.307 |
| GR vs AB/WBA (short) | HTHTH vs HHTTH | 0.303 |
| GR vs AB (length 4) | HTHT vs HHTT | 0.301 |
| GR vs AB (length 7) | HTHTHTH vs HHTTHHT | 0.289 |
| GR vs AB/RA (8-len) | HTHTHTHT vs HHTTHHTT | 0.285 |
| WBA vs BH | HTHTHTHH vs TTTTHHHH | 0.280 |
| WBA vs BH | HTHHTHTH vs HHHHTTTT | 0.280 |

The design provides good coverage across all five model contrasts, with particular strength on the GR-vs-AB and RA-vs-AB distinctions that are new to experiment 2.
