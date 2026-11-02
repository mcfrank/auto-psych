---
name: exp3-iter0-centre-weighted-probes
description: Exp3 iter_0 MAP probes on centre_weighted_range_sampling (LL -3516.53, kappa 0.02, scale 73) — alpha, gamma width shape, progressions dead; digit sums +1.5, primes +1.5, round edges +0.95
metadata:
  type: project
---
iter_0 candidate_4 (2026-11-01), base centre_weighted_range_sampling MAP LL -3516.53 on exp3 data (6750 rows, 154 sets; MAP takes ~30 s):
- Fixed example reliability ALPHA: 0.9 is optimal (0.88 -3526.8, 0.92 -3520.3, 0.8/0.95 lose 60-110). Dead.
- Gamma width prior with learned shape: shape 2.07, so it recovers the Erlang. Dead.
- Offset arithmetic progressions family: weight 0, dead.
- Digit-sum family: -3515.03 (+1.5), RMSE vs base 0.009 → submitted as centre_weighted_ranges_digit_sums.
- Primes own family: -3515.05 (+1.5). Round-number edges: bonus -0.54, -3515.59 (+0.95). Sibling candidate_3/5 took these.
Probe scripts are in /tmp/c4e3 (map.py, variant*.py) and may not persist.
**How to apply:** a new concept family is the only thing that moves this base; see [[exp2-map-probe-results]], [[exp2-iter4-round-number-refinements]].
