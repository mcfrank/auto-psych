---
name: atypical-round-spacing-exp3
description: exp3 iter3 cand5 — spacing shift on round_number_atypical_members MAP -3411.1; "examples-only list" concept null; cands 3+4 both did veto+spacing
metadata:
  type: project
---
2026-10-31, exp3 iter_3/candidate_5 (`atypical_round_spacing_sparsity`), a refinement slot.
- My first build (veto + spacing on the spacing model) duplicated iter3 candidates 3 and 4. Both had done veto+spacing, at MAP -3410.1. I dropped it.
- Adding an "the program outputs exactly the examples" list concept with a free log prior fits to -19.7, null. Without the veto the optimizer falls into a lapse≈0.5 trap.
- Spacing shift on round_number_atypical_members: MAP -3411.1 (typicality 0.84, shift -0.57). Submitted. Check OK, smoke -3419.1, ~6 min.
- Residuals of veto+spacing: calibration is good. Singleton near targets are over-predicted (0.48 vs 0.375); endpoints 1/100 0.307 vs 0.263.
**Why:** single-mechanism refinements in exp3 now land at ≤5 nats. Parallel slots in the same round converge on the same graft.
**How to apply:** check the sibling iter_N/candidate_*/model_name.txt files before building. Scripts are in iter_3/candidate_5/scratch (var.py with L/V/SP env flags, map.py, res.py). Related: [[spacing-on-set-size-exp3]], [[round-landmark-veto-exp3]].
