---
name: round-landmark-veto-exp3
description: exp3 iter2 round-target critique — veto of unsupported round targets +7.9 loglik MAP; decade blocks, even-as-2-table, ending mismatch null
metadata:
  type: project
---
2026-10-31, cell_1/experiment3 iter_2/candidate_0, on incumbent set_size_evidence_privileged_fives (MAP loglik -3423.2):
- Over-predicted round targets come mostly from the "even numbers" rule after power/multiples-of-4/8 sets (8,64 → 10,20,70 get 0-1/5 yes; 40,80 get more).
- p *= (1 - v·[round target, no round example]) gave -3415.3 (v≈0.18). Submitted as `round_landmark_veto`; check OK, smoke loglik -3423.1 (no better than the base smoke fit). A logit-shift form gave -3418.2.
- Null: decade blocks (x0-x9) instead of hundred-chart rows (x1-(x+1)0); a free even cost; a penalty when no example shares the target's ending. Pricing "even" as the 2 times table: -205.

**Why:** the critique's round-number gap is real but small; the cheap screens are listed above.
**How to apply:** scripts are in that candidate's scratch/ (map.py, res.py, dec2.py for rule/interval decomposition). See [[exp3-iter2-critique-stats]].
