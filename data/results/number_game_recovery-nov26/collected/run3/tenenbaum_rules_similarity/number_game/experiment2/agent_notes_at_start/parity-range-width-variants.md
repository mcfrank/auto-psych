---
name: parity-range-width-variants
description: Round-4 MAP tests on parity_within_range_concepts (MAP ll -1064.0): gamma width +1.7, separate parity-range width scale 0
metadata:
  type: project
---
Round 4 (2026-10-30), iter_4/candidate_4. Incumbent parity_within_range_concepts MAP ll -1064.0 (sigma≈30, kind weights .47/.47/.06).
- Gamma width prior (shape, mean inferred) grafted on: -1062.2 (shape≈4.1, mean≈56). Submitted as gamma_width_parity_range_concepts.
- Separate Erlang scale for parity ranges vs plain ranges: -1064.2 (no gain).
**Why:** saves re-running these.
**How to apply:** width-prior tweaks are close to exhausted; see [[incumbent-residual-diagnostics]], [[concept-space-additions-map]].

Round 4, iter_4/candidate_5 was rejected as an exact duplicate of gamma_width_parity_range_concepts (candidate_3 had already submitted that graft). Repair (candidate_5_repair_1): decade_and_parity_range_concepts = parity_within_range_concepts + decade family. Smoke ll -1068.5; 1x300 NUTS gives decade family weight ≈0.066, and p_yes RMSE vs the incumbent is 0.075 on decade sets (61,64,67 etc.), so it passes novelty by a wide margin.
**How to apply:** before grafting, check which grafts sibling slots in the same round already submitted.
