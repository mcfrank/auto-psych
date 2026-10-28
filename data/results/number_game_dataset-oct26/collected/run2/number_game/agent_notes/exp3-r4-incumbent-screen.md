---
name: exp3-r4-incumbent-screen
description: Exp3 r4 MAP screen of single changes to leading_digit_string_similarity — even rule +6.5, log compression −14, partial span for 2-example sets +3
metadata:
  type: project
---
Exp3 round 4 (2026-10-18). The incumbent leading_digit_string_similarity has MAP ll −4630.2 and set SD 0.118 predicted vs 0.147 observed.
MAP gains from single additions:
- Even rule for all-even sets: +6.5. The (k−1)log2, k·log2 and log k scalings are all about the same; unscaled ±1 gives +5. It was submitted as even_rule_leading_digit_similarity (iter_4/candidate_4).
- Even-target-only boost: +5.5.
- Set-parity criterion shift (all-even +, all-odd −): +2.5–3.
- Symmetric even and odd rules: +0.4.
- Log-compressed out-of-span distance: −14.
- Partial span filling only for 2-example sets: +3.
Residual pattern: all-odd and wide 2-example sets are over-predicted (19,73; 3,81; 2,47; 8,92; 83,11). Target 1 is over-predicted, except after 31,19,49,100, where it gets 0.89, likely a squares reading.
**Why:** avoids re-screening these.
**How to apply:** scripts are in iter_4/candidate_4/scratch (mk.py, mk2.py, resid.py). See [[exp3-r3-residual-scan]] and [[leading-digit-similarity-gain]].
