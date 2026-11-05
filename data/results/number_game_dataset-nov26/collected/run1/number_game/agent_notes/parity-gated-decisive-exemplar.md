---
name: parity-gated-decisive-exemplar
description: exp3 iter2 c4 — parity_preserving gate grafted onto coincidence_decisive incumbent, +10 MAP; slope-by-set-size and multiplier-decay no help
metadata:
  type: project
---
Exp3 iter_2 candidate_4 (2026-11-02), `parity_gated_decisive_exemplar`: the incumbent `coincidence_decisive_exemplar` with its multiples route × exp(-c_par_route·par_diff). MAP log-p: incumbent -4077.6, graft -4067.8 (+9.9), gating factor-kin as well -4069.7. Smoke log-lik -4075.0 (16 s).
Aimed at incoherent_set_cell_polarization (q=0.032). The most under-predicted polarized cells were lone-example far multiples predicted near 0.5 (7→14, 33→99, 16→64 got yes; 17→34, 43→86, 49→98, 13→78, 23→92, 3→54/60 got no). The graft barely moves the polarization stat (expected 0.138→0.141, observed 0.216).
No help: a single-example response-slope term (+3.8; it wants a shallower slope, not a steeper one), a slope × n^b term (+0.7), multiples route decaying with log multiplier (0).
**Why:** polarization in incoherent sets is a target-level miss, not a decision-slope one.
**How to apply:** the leftover misses are 7→14 yes vs 49→98 no (both doubles of odd numbers). Look for a target-level cue there, not a slope change. Related: [[coincidence-decisive-exemplar]], [[parity-preserving-multiple-route]], [[exp3-iter2-critique-stats]].
