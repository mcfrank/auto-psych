---
name: exp3-iter2-verification-and-sharpening-probes
description: Exp3 iter_2 cand_1 MAP probes on fast uniform-range base (-3515.04) — rule verification errors, recognition misses, rules-only size-principle strength, kind-level sharpening all pin at normative; submitted kind_level_overcommitment
metadata:
  type: project
---
2026-11-01, exp3 iter_2 candidate_1 (exploratory). Base = location_biased_range_prior with slope 0 (no centre weighting): MAP LL -3515.04, Erlang scale ~116.
- Symmetric rule-membership verification error on the target: e=0. Misses for arithmetic families (special, multiples, exceptions, dsum): 0. Global miss: 0.
- Rules-only size-principle exponent: 1.01 (+0.19). Kind-level posterior sharpening exponent 1.013 (+0.13).
- Submitted kind_level_overcommitment (two-stage: kind posterior^g, Bayes within kind); smoke fit 9 s per 200 iterations (~5 min/chain).
Probe script /tmp/c1i2/probe.py (variants via argv) may not persist.
**How to apply:** decision/verification-side and evidence-strength tweaks are dead on exp3 too; see [[exp3-iter1-process-probes]], [[exp3-iter1-saturation-probes]].
