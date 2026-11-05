---
name: log-number-line-similarity
description: exp3 iter_3 c1 — log-ratio gaps for band + span similarity on parity_bounded_ambiguity_doubt: smoke -4664.8, a tie with the incumbent
metadata:
  type: project
---
exp3 iter_3 candidate_1 (2026-11-02): `log_number_line_similarity`. It is parity_bounded_ambiguity_doubt with the linear /100 gaps in the Gaussian band and the span exp similarity replaced by log(min/t) and log(t/max). Widths are LogNormal(log .8) and LogNormal(log .2). Smoke log-likelihood -4664.8, the same as the incumbent's -4664.8.

**Why:** in exp3 the choice between a linear and a log magnitude metric doesn't matter; the magnitude terms carry little weight next to the doubt mechanism.
**How to apply:** don't spend more slots on magnitude-metric variants in exp3. Related: [[parity-bounded-ambiguity-doubt]] [[span-interval-similarity]]
