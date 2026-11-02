---
name: incumbent-residual-diagnostics
description: Round-3 tests on arbitrary_exception_rule_prior (quick 300-draw NUTS lppd, incumbent -1068.1); which single changes help
metadata:
  type: project
---
Round 3 (2026-10-30), iter_3/candidate_3. Incumbent is well calibrated (rescaling rule/range shares gains <1 nat); worst sets are 4,66,78,96, 31,3,1,15, 23,35,95,83 (parity+magnitude patterns).
- Helped: "odd/even within a range" concept kind (third Dirichlet kind weight ≈0.07, same Erlang width prior): lppd -1065.1 (+3). Submitted as parity_within_range_concepts.
- No help: fixed ALPHA 0.7/0.8/0.97 (all much worse; 0.9 is best), set-level max-gap penalty on ranges (-0.25), hierarchical within-family rule weights (+0.7, tau→0.1), decade-round range endpoints (-0.6).
- Too slow: per-interval (5050-column) likelihood with an exponent on range evidence — >10 min for 600 draws; keep per-size aggregation in prepare_observed.
**Why:** avoids re-running these next round.
**How to apply:** build on parity-within-range if it was admitted; see [[powers-of-2-37-misfit]].

Round 4 (2026-10-30), iter_4/candidate_5 (gamma_width_parity_range_concepts = incumbent parity_within_range_concepts with the gamma width prior instead of Erlang): quick 2x300 LOO -1071.4 vs incumbent -1072.3 (+0.9 only; shape≈4.2±2.3, mean width≈60). Gamma width and parity-range gains do not stack much.
