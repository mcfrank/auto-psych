---
name: big-step-multiples-finding
description: exp2 multiples of m=13..50 (flat extra cost) on prime_over_odd incumbent +9 MAP loglik; repdigits +4.8; effective-n exponent null
metadata:
  type: project
---
2026-10-30, exp2 iter_3/candidate_0 (`multiples_of_any_step_size`). Incumbent prime_over_odd_reading MAP nll 2126.9 (lapse→0 at MAP).
- Adding "multiples of m" for m=13..50, each costing 2 primitives plus one flat fitted cost (≈0.3): nll 2118.0 (+9). With log(m/3)-graded cost plus an offset: 2116.4 (+10.6). Grading alone with no offset: about 0.
- The gain is mostly 3,33,99 (+5; read as multiples of 33, so 66 yes and 23 no), then 14,98,91, 96,48, 63,27,81 and 13,91. The biggest loss is 84,56 (-2.3).
- Repdigits 11..99 with a fitted bonus: +4.8. Scaling set loglik by n^-k: k≈0, null.
- Check passed OK. Smoke loglik -2124.2 (incumbent smoke -2132.5). The 100+100 fit took 112s.
- Scratch is scratch/w/screen.py (numpy+Powell, extra-rule hook).

**Why:** shows which concept-family extensions still pay on exp2.
**How to apply:** don't re-screen these. Related: [[prime-over-odd-swap]], [[prime-salience-finding]], [[factor-graded-multiples-finding]].
