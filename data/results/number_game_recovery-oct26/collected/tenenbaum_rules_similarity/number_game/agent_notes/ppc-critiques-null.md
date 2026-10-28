---
name: ppc-critiques-null
description: exp2 PPC critiques iter_0-2 found 0/24 significant stats; iter_3 tried AP/repdigit/round/digit-reversal/set-size stats
metadata:
  type: project
---
Exp2 critique rounds iter_0..iter_2 (24 test stats: proximity, parity, decade, up/down asymmetry, lag-1, trial drift, cell extremity, primes, mult-5) all had p>0.1 against the incumbent. iter_3 (2026-10-19) submitted: AP-congruence (gcd 3-9 offset), repdigit, round-number, digit-reversal, between-set SD, set-size slope, small targets after high sets, inside-vs-outside span (>=3 examples). Check iter_3/critique/ppc_results.json before reusing these ideas.
**Why:** avoid re-proposing stats already shown null. **How to apply:** read prior ppc_results.json first; target concepts absent from the rule space. Related: [[concept-space-saturated]]
