---
name: write-candidate-first
description: Refinement slots can be lost by over-exploring; write candidate.py before probing (exp2 iter_1 candidate_3 lost its first attempt)
metadata:
  type: feedback
---
On 2026-10-19, exp2 iter_1 candidate_3's first attempt ended without a candidate.py. The retry copied the incumbent with a sed edit (named sets each their own prior kind, `named_sets_distinct_familiarity`), and the check passed in about 4 minutes (smoke LL -2057.8; the incumbent's conjunction model samples in about 200 s).

**Why:** The family is saturated (see [[incumbent-near-ceiling-exp2]] and [[concept-space-saturated]]), so long probing rarely finds anything to gain.
**How to apply:** Write all three files first, then run the check, and only then explore if time is left.

Confirmed again in exp2 iter_2 candidate_3 (2026-10-19): editing a copy of upper_edge_marking_ranges with a Python string-replace script (a graded top-edge bonus over gaps 0-3, `graded_upper_edge_marking_ranges`) passed the check in about 7 minutes (smoke LL -2057.5, about 190 s per sampling run).
