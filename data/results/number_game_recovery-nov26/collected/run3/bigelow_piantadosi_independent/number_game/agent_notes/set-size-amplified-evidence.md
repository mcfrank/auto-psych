---
name: set-size-amplified-evidence
description: exp2 per-example evidence weight n**g on prime_over_odd incumbent, MAP +5 nats (g≈0.29); prior tempering null, uncertainty guessing +2.8
metadata:
  type: project
---
2026-10-30, exp2 iter_3/candidate_2 (`set_size_amplified_evidence`). Fast MAP NLL on prime_over_odd_reading: base 2126.9. Changes tried:
- loglik × n**g: 2121.8 (g≈0.29).
- Uncertainty-triggered guessing: 2124.1.
- Prior tempering (prior × exp(a)): 2126.8.
In the incumbent the lapse goes to about 0 at MAP. Smoke loglik was -2128.6, about 130 s per 200-draw chain, and the check printed OK.
Residuals are scattered across single cells (3,33,99→23; 14→98; 64,94,84→59) with no shared pattern.
The screen.py scratch script must not filter its own traceback through grep -v arviz. Parallel runs can also crash on arviz's daily-warning file race.

**Why:** gains available on top of the incumbent are now small, about 5 nats or less.
**How to apply:** for bigger wins, look for new concept families, not response-stage tweaks. Related: [[uncertainty-triggered-guessing]], [[prime-over-odd-swap]].
