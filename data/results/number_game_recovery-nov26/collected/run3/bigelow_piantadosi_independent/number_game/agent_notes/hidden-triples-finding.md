---
name: hidden-triples-finding
description: exp2 iter3 cand3 — extra cost on "multiples of 3" over prime_over_odd_reading gives +6.6 MAP loglik; many other tweaks null on that base
metadata:
  type: project
---
2026-10-30, exp2 iter_3/candidate_3 (`hidden_triples_concepts`). MAP NLL on prime_over_odd_reading: base 2126.6, mult3 cost 2120.0 (cost≈0.74, pred RMSE vs base 0.025). The check printed OK with smoke loglik -2126.3.
Null on this base (≤2 nats): uncertainty-gated guessing (lapse→0 here), separate weight for the 1-100 interval, row bonus graded by width, chain-length powers, contains-d cost, fitted stray rate, costs on all multiples except 5. Set-size-scaled commitment gave +4.3, but row_bonus ran off to 142.
Index note: rules 16-24 are mult 3..12, ends-in-d is 25-34, contains-d is 35-44. LOG_FACTOR still penalises ends-in-0/1.
Scripts are screen*.py, resid.py and cmp.py in that candidate's scratch/.

**Why:** the incumbent is close to saturated on prior tweaks. The data also refuse "multiples of 3" readings, as [[odd-mult3-set-pattern]] predicted.
**How to apply:** start future screens from scratch/screen5.py. Related: [[prime-over-odd-swap]], [[chain-length-powers-finding]].
