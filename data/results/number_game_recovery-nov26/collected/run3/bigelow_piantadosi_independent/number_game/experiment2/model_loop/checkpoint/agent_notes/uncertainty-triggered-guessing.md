---
name: uncertainty-triggered-guessing
description: exp2 guessing that rises with belief uncertainty 4p(1-p) gains +23 loglik on costly_squares base; flat/set-size/concentration lapse null
metadata:
  type: project
---
2026-10-30, exp2 iter_1/candidate_1 (`uncertainty_triggered_guessing`), MAP screens on costly_squares_final_digit_concepts (base loglik -2207.6, logpost -2231.6):
- Guess prob = sigmoid(a + b*4p(1-p)) with a fitted guess rate: loglik -2184.3, logpost -2201.1 (a≈-4.9, b≈6.3, g≈0.53). Smoke loglik -2188.5.
- Linear-in-log-odds weighting: -2204.1 (gamma 0.84). Fitted default lapse, set-size-gated lapse, max-posterior-gated lapse: all within 1 nat of base.
- find_MAP from a jittered start can get stuck at b=0. Use several starts.

**Why:** people flatten mid-range beliefs toward 0.5 more than the inference predicts. This is a decision-stage gain that may stack on [[factor-graded-multiples-finding]].
**How to apply:** scripts are in that candidate's scratch/ (mk.py, screen2.py).
