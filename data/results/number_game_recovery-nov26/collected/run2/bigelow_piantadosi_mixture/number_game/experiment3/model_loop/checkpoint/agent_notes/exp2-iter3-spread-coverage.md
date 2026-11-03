---
name: exp2-iter3-spread-coverage
description: Exp2 iter-3 cand_0 screens on digit_sum_open_ended incumbent — new concept families/digit reversal/decade ranges null; spread-coverage likelihood +0.45 MAP, pool RMSE 0.0055, submitted
metadata:
  type: project
---
Exp2 iter 3, candidate_0 (2026-10-30). The incumbent's MAP NLL is 1923.1. These screens were null (Δ ≤ 0.1): multiples of 13-50, numbers built from digit pair {a,b}, all multiples 2-50, tens-digit parity, digit-sum parity, digit-reversal misreading (rho→0), decade-bounded ranges. A round-10 both-ends range boost gave 0.8, but it came only from singleton round numbers.
Submitted `spread_coverage_expectation`. It tilts each concept's likelihood by r^kappa, where r is the rank span of the member examples divided by (size-1). It is normalised exactly with the without-replacement span distribution P(d)=(S-d)C(d-1,n-2)/C(S,n), giving a table of shape (S, n, d) built in the model. MAP kappa≈0.10, NLL 1922.67. RMSE against kappa=0 on the novelty pool is 0.0055. kappa=±1 loses 50-90 nats.
Implementation: per set, intervals go into columns keyed by (width-group, n members, span); there are at most 538 columns. A smoke run of 200 iterations takes 27 s, so 7000 iterations take about 16 min.
Pitfall: `cmd | grep` in a background job buffers its output, so the job looks hung. Screener scripts are in iter_3/candidate_0/scratch/ (screen.py does family grafts, cover.py is the dense hypothesis-level screen). See [[exp2-iter2-open-range-graft]], [[fit-speed-budget]].
