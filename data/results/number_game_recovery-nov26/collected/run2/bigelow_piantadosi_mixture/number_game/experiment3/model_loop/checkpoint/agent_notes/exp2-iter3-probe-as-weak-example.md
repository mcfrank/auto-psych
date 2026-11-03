---
name: exp2-iter3-probe-as-weak-example
description: Exp2 iter-3 cand_2 — target treated as tempered extra example (power w) on digit_sum_open_ended incumbent; smoke -1926.7, submitted
metadata:
  type: project
---
Exp2 iter 3, candidate_2 (2026-10-30): `probe_as_weak_example`. Each member of h gets a factor (1+90/|h|)**w, and P(yes) = N/(N+1-P_in). This is exact and cheap: for rules and sums the per-cell arrays are post×member (cells×45 and cells×18), and intervals reuse the width groups because the factor depends only on width. w=0 gives back the incumbent. The smoke fit's loglik is -1926.7 (the incumbent is about -1928), taking 15 s. I did not look at the posterior of w.
**How to apply:** if it enters, the probe-pragmatics region is covered. See [[exp2-iter3-critique]].
