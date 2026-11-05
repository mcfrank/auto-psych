---
name: problem-size-decisive-exemplar
description: exp3 iter2 c4 repair: problem-size multiples decay on coincidence_decisive, +6.3 MAP; smoke -4078.8
metadata:
  type: project
---
Exp3 iter_2 candidate_4_repair_1 (2026-11-02): the first attempt (parity gate on the multiples route) was rejected because it predicted exactly like parity_preserving_decisive_exemplar (RMSE 0). The repair is `problem_size_decisive_exemplar`: the incumbent's multiples route × exp(-c_size·(e + t/e - 4)/10).
MAP log-lik: incumbent -4077.6, factor sum -4071.3 (c_size≈0.22), log factor sum -4071.5, log product -4074.3, example size -4075.5, multiplier -4077.0. The parity gate scores -4067.8 by comparison. On training trials the repair's p_yes differs from the parity graft by RMSE 0.032. Smoke log-lik -4078.8.
Screening script: /tmp/scr4/screen.py (find_MAP plus response log-lik; may be gone).
**Why:** [[parity-gated-decisive-exemplar]] said the leftover misses were target-level (7→14 yes vs 49→98 no).
**How to apply:** if one parity-switch idea is already live, a second parity-gate variant is a duplicate. Related: [[problem-size-multiple-retrieval]].
