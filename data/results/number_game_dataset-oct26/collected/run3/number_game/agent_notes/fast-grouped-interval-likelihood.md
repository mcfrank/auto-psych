---
name: fast-grouped-interval-likelihood
description: Pooling intervals by (size, n-contained) keys makes concept-learning candidates fast enough for the 30-min limit, even with midpoint/decade priors
metadata:
  type: project
---
Many concept-learning candidates were dropped as "too slow to fit". An interval's likelihood depends only on its length and on how many examples it contains, so the 5,050 intervals can be grouped into 500 keys per set.
- Equal-prior intervals: pool the counts in prepare_observed (`models/any_of_sampled_concepts_decision.py`). A 5-parameter smoke fit took about 42 s (2026-10-22).
- Midpoint/decade interval priors: constant sparse CSR (set,key)->interval indicators, applied to the interval prior (`models/late_session_acquiescence_concepts.py`). This is exact and costs about 37 ms per gradient, versus a dense (sets x 5092) posterior matrix that took 802 s for a 100+100 smoke fit. Reusing it, single_exception_rule_concepts (experiment3 iter_3 candidate_3_repair_1) dropped to 166 s. Any likelihood that is a function of (j contained, |h|, n) fits this scheme.
**Why:** a dense per-interval graph with a 42-rule Dirichlet times out at target_accept 0.99.
**How to apply:** start from the late_session file for any outlier-tolerant-family refinement and change only the per-key log-likelihood.
- Incumbent posterior (experiment3/design/_fit_cache/outlier_tolerant_rule_concepts.*.nc): size_growth about -0.55 and decade_bonus about 6.5. Weak evidence at n=4 keeps the evens and multiples-of-5 rules alive for all-round sets, which is the cause of the tens_sets_nonround critique. Weighting only the gain over the stray baseline matters: scaling the raw log-likelihood by family-specific weights does the opposite. family_specific_evidence_growth (exp3 iter_4 candidate_4) took about 150 s per smoke fit.

Extra concept families with equal priors within a group also fit the scheme: bounded arithmetic progressions (steps 2-10, about 9.6k concepts) keyed by (size, count) per step. Use dense per-set counts plus a sparse nonzero gather for per-cell counts (`pt.zeros(n)[idx].inc(vals)`). bounded_progression_concepts (experiment3 iter_4 candidate_0) took 39 ms per gradient, tree depth about 4, and 227 s per 100+100 smoke run (2026-10-22). Never `pkill -f <name>` from a command line that contains that name: it kills its own shell.
