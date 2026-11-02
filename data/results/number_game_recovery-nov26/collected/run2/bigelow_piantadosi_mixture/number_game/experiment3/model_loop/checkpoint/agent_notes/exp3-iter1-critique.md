---
name: exp3-iter1-critique
description: Exp3 iter-1 critique of heading_extrapolation_weber_width — 8 new raw-slice stats (reversal, doubling, AP lattice, decade boundary, etc.); iter-0 got 0/8
metadata:
  type: project
---
On 2026-10-31 I submitted 8 statistics for the exp3 iter-1 critique, written by a bash `mk` helper that shares one header:
- digit reversal (obs 0.46)
- double/half of an example (0.49)
- AP lattice on-minus-off (+0.10)
- SD of per-set yes rates (0.25)
- targets 91-100 on high sets (0.31)
- targets 1-10 on sets ≥30 (0.29)
- decade boundary in near extrapolation (+0.07)
- 4-example interior near-minus-far (-0.14; near examples get fewer yes)

Each takes about 0.02 s per call. The iter-0 critique found 0 of 8 significant. See [[exp3-iter0-critique]], [[critique-stat-no-imports]].
