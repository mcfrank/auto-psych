---
name: prime-odd-decoupled-exp3
description: exp3 MAP screens on repdigit incumbent; separate prime bonus + odd cost +11.8 loglik; noise/shape/n-beta/contains/parity/ugs null
metadata:
  type: project
---
2026-10-31, exp3 iter_0/candidate_4 (`prime_bonus_odd_cost_decoupled`). MAP on repdigit_pattern_chart_concepts base: loglik -3465.9, logpost -3499.9.
- Decouple prime_over_odd into separate prime bonus (~1.6) and odd cost (~0.6): loglik -3454.1, logpost -3491.6.
- Null (≤2 nats): fitted stray noise, Erlang shape, commitment growing with set size, contains-d bonus, parity bonus, uncertainty-triggered guessing.
- The largest residual sets: 3,33,99 (93 at 5/5 under-predicted), 8,64 (multiples of 8 over-predicted, ~0.3 observed vs 0.7), 63,27,99,33 (odd non-examples yes ~0.9 vs 0.6), and 83,38,58,68 (85/87/89 yes vs 0.46).
- The singleton near/far critique (q=0.29) was not fixed by any of these.

**Why:** shows which tweaks are already exhausted on exp3 data.
**How to apply:** scripts are in that candidate's scratch/ (var.py with a V env switch, screen.py, resid.py). For p_yes, use pytensor.function(value_vars, replace_rvs_by_values). Related: [[prime-over-odd-swap]], [[uncertainty-triggered-guessing]].
