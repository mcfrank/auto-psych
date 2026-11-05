---
name: first-impression-span-interior
description: "between min and max example" property on first_impression_anchoring: +5.6 MAP; smoke -3128.3 (exp2 iter_4 c5)
metadata:
  type: project
---
Exp2 iter_4 candidate_5 (2026-11-02), first_impression_span_interior. This adds a property for multi-example sets, "the target lies between the smallest and the largest example", with its own Dirichlet salience slot (fitted at about 0.04). A visible shared final digit overshadows it. It targets the inrange_nonshared_rate critique (q=0.11). Best-of-3 find_MAP: parent -3126.3, candidate -3120.6. Band width grew from 0.44 to 0.62. Smoke check OK at -3128.3, in about 4.5 minutes.
**Why:** a two-sided in-span property still helps even with the graded one-sided band.
**How to apply:** this could also be grafted onto the incumbent; see [[first-impression-anchoring]] and [[graded-magnitude-band]].
