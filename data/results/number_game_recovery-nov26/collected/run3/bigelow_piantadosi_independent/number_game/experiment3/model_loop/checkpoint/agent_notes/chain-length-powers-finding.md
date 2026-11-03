---
name: chain-length-powers-finding
description: incumbent factor_graded_multiples has log_m index offset bug (actually grades multiples 5-12 + penalises ends-in-0/1); chain-length power cost +7 loglik
metadata:
  type: project
---
2026-10-30, cell_1/experiment2 iter_1/candidate_4 (scratch/ has screen2.py, resid.py, cmp.py, v1-v9).
- BUG in factor_graded_multiples_concepts: `log_m[18:27]` should be `[16:25]`. As written, multiples of 3,4 get no cost, 5..12 are shifted, and ends-in-0 / ends-in-1 get extra cost. Fixing the index gives WORSE MAP loglik (-2195.8 vs -2173.5). Fix plus no cost for m<=5 gives -2178; the ends-0/1 penalty is worth ~4.
- Added on the incumbent (bug kept): powers-of-b cost chain_cost*log(len/3), loglik -2166.3, chain_cost ~3. A log-base form fit about the same (-2165.2). Nearly all of the gain is on set 8,64. Contains-d bonus +3.7 only.
- Submitted as `chain_length_graded_powers`.
**Why:** the incumbent's stated mechanism does not match its code. Future refinements should know this.
**How to apply:** before claiming a mechanism, check the incumbent's index masks. See [[factor-graded-multiples-finding]].
