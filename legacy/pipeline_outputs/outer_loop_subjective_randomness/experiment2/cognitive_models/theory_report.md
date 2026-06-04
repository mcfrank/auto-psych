# Theory Report — Experiment 2

## weighted_balance_alternation

**Motivation:** `balance_heuristic` was the best-fitting model in experiment 1 (posterior = 0.999999) but failed two PPCs. The `mean_chose_alternating` test showed participants chose the more-alternating sequence 67% of the time vs. the model's predicted 46% (p = 0.002), and `mean_abs_error` was 0.26 vs. the expected 0.17 (p = 0.004). These failures share a common cause: the balance heuristic scores HTHTHTHT and HHTTHHTT identically (both perfectly balanced), but participants strongly prefer the alternating sequence. This model adds an explicit alternation component to fix that blind spot.

**Mechanism:** Scores each sequence as an equal-weighted average of its balance score (1 − 2|p_H − 0.5|) and its alternation rate (fraction of consecutive pairs that differ). The softmax over the two scores (β = 5.0) produces a probability distribution over left/right choices. Unlike `balance_heuristic`, this model predicts that HTHTHTHT > HHTTHHTT, and unlike `alternation_bias`, it still rewards H/T balance. The w = 0.5 split is a neutral starting point; future experiments can vary stimuli to estimate w empirically.

## run_aversion

**Motivation:** The critique's `mean_chose_alternating` failure establishes that people are sensitive to sequential structure beyond balance, but does not distinguish two possible mechanisms: (a) a preference for high alternation rate generally, or (b) an aversion specifically to long streaks. A sequence like HTHTTTTH has moderate alternation rate (4/7 ≈ 0.57) but contains a 4-element run of T. `alternation_bias` and `weighted_balance_alternation` would rate this moderately, but if participants are driven by run aversion they should rate it much lower. This model tests that hypothesis.

**Mechanism:** Scores each sequence by 1 − (max_run_length − 1) / (len − 1), normalised to [0, 1]. A fully alternating sequence (max run = 1) scores 1.0; a sequence with one long run spanning the whole string scores 0.0. This is genuinely distinct from alternation rate: two sequences can have the same alternation rate but different maximum run lengths (e.g. HHTHTH vs HTHTHH both have alternation rate 4/5 = 0.8, but HHTHTH has max run 2 and HTHTHH has max run 2 as well — longer examples make the divergence clearer). The softmax with β = 5.0 converts scores to choice probabilities.
